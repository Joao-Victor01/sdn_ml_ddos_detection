"""
Hop Count Filtering (HCF) — Etapa de classificação de origem dos ataques.

O HCF é um mecanismo de análise passiva do campo TTL (Time To Live) do
cabeçalho IP para inferir a distância topológica do pacote ao controlador SDN.
Ataques externos tipicamente têm TTL degradado (cruzaram muitos roteadores);
zumbis internos têm TTL quase intacto (1-2 hops dentro da LAN).

## Como funciona

1. O controlador recebe estatísticas de fluxo via ODL (etapa [3/6]).
2. Para fluxos suspeitos (alta taxa de pacotes), o HCF é consultado com
   o TTL estimado ou real do pacote.
3. O módulo retorna a classificação: BENIGN / EXTERNAL / INTERNAL.
4. A classificação alimenta o módulo de traceback e isolamento.

## Integração com o loop de controle

O HCFAnalyzer é chamado explicitamente via endpoint REST (/detect/classify)
ou internamente pelo monitor de tráfego quando um fluxo ultrapassa o limiar
de suspeição (SUSPICIOUS_PPS_THRESHOLD). Ele não modifica o state.graph
nem instala flows — apenas classifica. A ação é responsabilidade do traceback.

SRP: este módulo classifica. Não instala flows, não modifica estado.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from orchestrator.domain.state import state
from orchestrator.config import MAX_LINK_CAPACITY

# ── Constantes de HCF ─────────────────────────────────────────────────────────
TTL_INITIAL_LINUX = 64      # TTL padrão de sistemas Linux/Unix
TTL_INITIAL_WIN   = 128     # TTL padrão de sistemas Windows
HOP_EXTERNAL_MIN  = 10      # mínimo de saltos para considerar "veio da internet"
SUSPICIOUS_PPS    = 10_000  # pacotes/segundo acima deste valor → suspeito


class TrafficClass(IntEnum):
    """Classificação triclasse do tráfego."""
    BENIGN   = 0
    EXTERNAL = 1   # Ataque externo (IP spoofing, DDoS da internet)
    INTERNAL = 2   # Zumbi interno (botnet local, host comprometido)


@dataclass
class HCFResult:
    """Resultado da análise HCF para um fluxo."""
    src_ip:          str
    traffic_class:   TrafficClass
    ttl_observed:    int
    hop_count:       int
    is_known_host:   bool   # IP registrado no SDN (hosts_by_mac)
    flow_pkts_s:     float
    confidence:      float  # 0.0–1.0: certeza da classificação
    reason:          str    # explicação legível

    @property
    def label(self) -> str:
        return ["Benigno", "Ataque Externo", "Zumbi Interno"][int(self.traffic_class)]

    def to_dict(self) -> dict:
        return {
            "src_ip":        self.src_ip,
            "class":         int(self.traffic_class),
            "label":         self.label,
            "ttl_observed":  self.ttl_observed,
            "hop_count":     self.hop_count,
            "is_known_host": self.is_known_host,
            "flow_pkts_s":   round(self.flow_pkts_s, 2),
            "confidence":    round(self.confidence, 3),
            "reason":        self.reason,
        }


class HCFAnalyzer:
    """
    Classifica fluxos de rede em Benigno / Externo / Interno usando HCF.

    Algoritmo:
      1. Verificar se o IP de origem é conhecido no SDN (hosts_by_mac).
         - Desconhecido → nunca passou pelo controlador → suspeito externo.
      2. Calcular hop_count = TTL_inicial - TTL_observado.
         - hop_count >= HOP_EXTERNAL_MIN → veio da internet → Externo.
         - hop_count <  HOP_EXTERNAL_MIN → está na LAN → Interno.
      3. Se flow_pkts_s < SUSPICIOUS_PPS → provavelmente Benigno.

    Em produção, ttl_observed vem do campo TTL do Packet-In do ODL.
    Em simulação, pode ser estimado a partir dos contadores de porta.
    """

    def classify(
        self,
        src_ip:       str,
        ttl_observed: int,
        flow_pkts_s:  float,
        ttl_initial:  int = TTL_INITIAL_LINUX,
    ) -> HCFResult:
        """
        Classifica um fluxo de rede.

        Parameters
        ----------
        src_ip       : IP de origem do fluxo
        ttl_observed : TTL observado no pacote ao chegar ao switch de borda
        flow_pkts_s  : taxa de pacotes por segundo do fluxo
        ttl_initial  : TTL inicial esperado (64=Linux, 128=Windows)

        Returns
        -------
        HCFResult com classificação e metadados de explicabilidade.
        """
        hop_count      = max(0, ttl_initial - ttl_observed)
        is_known       = self._is_known_host(src_ip)
        is_high_rate   = flow_pkts_s >= SUSPICIOUS_PPS

        # ── Regra 1: tráfego de baixa taxa → provavelmente benigno ───────────
        if not is_high_rate:
            return HCFResult(
                src_ip=src_ip, traffic_class=TrafficClass.BENIGN,
                ttl_observed=ttl_observed, hop_count=hop_count,
                is_known_host=is_known, flow_pkts_s=flow_pkts_s,
                confidence=0.85,
                reason=f"Taxa baixa ({flow_pkts_s:.0f} pps < {SUSPICIOUS_PPS} pps) — benigno.",
            )

        # ── Regra 2: muitos saltos → veio da internet → Externo ──────────────
        if hop_count >= HOP_EXTERNAL_MIN:
            conf = min(0.95, 0.70 + (hop_count - HOP_EXTERNAL_MIN) * 0.02)
            return HCFResult(
                src_ip=src_ip, traffic_class=TrafficClass.EXTERNAL,
                ttl_observed=ttl_observed, hop_count=hop_count,
                is_known_host=is_known, flow_pkts_s=flow_pkts_s,
                confidence=conf,
                reason=(f"hop_count={hop_count} ≥ {HOP_EXTERNAL_MIN} → "
                        f"TTL degradado indica origem externa (internet). "
                        f"IP {'desconhecido' if not is_known else 'conhecido (spoofado?)'}.")
            )

        # ── Regra 3: poucos saltos + alta taxa → Zumbi interno ───────────────
        # Host de dentro da LAN gerando tráfego volumétrico
        reason_parts = [
            f"hop_count={hop_count} < {HOP_EXTERNAL_MIN} → host na LAN.",
            f"Taxa alta ({flow_pkts_s:.0f} pps).",
        ]
        if is_known:
            reason_parts.append("IP registrado no SDN — host interno comprometido (zumbi).")
            conf = 0.92
        else:
            reason_parts.append("IP não registrado no SDN — possível MAC spoofing interno.")
            conf = 0.78

        return HCFResult(
            src_ip=src_ip, traffic_class=TrafficClass.INTERNAL,
            ttl_observed=ttl_observed, hop_count=hop_count,
            is_known_host=is_known, flow_pkts_s=flow_pkts_s,
            confidence=conf,
            reason=" ".join(reason_parts),
        )

    def classify_batch(
        self,
        flows: list[dict],
    ) -> list[HCFResult]:
        """
        Classifica múltiplos fluxos em lote.

        Parameters
        ----------
        flows : lista de dicts com chaves: src_ip, ttl_observed, flow_pkts_s
                (e opcionalmente ttl_initial)
        """
        return [
            self.classify(
                src_ip=f["src_ip"],
                ttl_observed=int(f.get("ttl_observed", 60)),
                flow_pkts_s=float(f.get("flow_pkts_s", 0)),
                ttl_initial=int(f.get("ttl_initial", TTL_INITIAL_LINUX)),
            )
            for f in flows
        ]

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_known_host(ip: str) -> bool:
        """Verifica se o IP está registrado no estado do SDN."""
        with state.lock:
            return ip in state.ip_to_mac
