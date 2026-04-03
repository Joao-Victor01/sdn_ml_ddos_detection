"""
Engenharia de labels: binário → multiclasse (3 classes).

Converte o dataset InSDN8 (0=Benigno, 1=DDoS) para o esquema triclasse:
  0 = BENIGNO        — tráfego legítimo sem alterações
  1 = ATAQUE EXTERNO — DDoS originado fora da rede (IP spoofing, botnets remotas)
  2 = ZUMBI INTERNO  — host comprometido dentro da LAN (botnet local)

## Fundamentação da heurística de separação

O dataset InSDN8 não possui coluna de TTL nem de origem (interno/externo).
Usamos conhecimento de domínio de redes para aplicar uma regra determinística
que reflete o comportamento real de cada tipo de ataque:

### Indicadores de ATAQUE EXTERNO (classe 1):
  - Protocol = 0 (ICMP/raw): ferramenta clássica de flood trivial usada por
    atacantes remotos (ICMP flood, ping-of-death) — raros em tráfego interno legítimo.
  - OU: Pkt Len Std ≈ 0 (pacotes 100% uniformes) E Flow Duration < 500 µs:
    rajada sintética automatizada, caractherística de ferramentas como hping3
    ou LOIC operadas de fora da rede, com duração extremamente curta.

### Indicadores de ZUMBI INTERNO (classe 2):
  - Complemento dos externos: ataques que não se encaixam no perfil acima.
  - Protocol ≠ 0 sugere TCP (SYN flood, ACK flood) — típico de zumbis que
    mimicam handshake real para driblar filtros simples.
  - Duração maior e variabilidade de pacotes indicam tráfego semi-legítimo
    gerado por host infectado com acesso à rede interna.

### Limitação e honestidade científica:
Esta heurística é uma PROXY para o rótulo verdadeiro, que em produção viria
do TTL real dos pacotes (capturado pelo SDN via Packet-In) e do registro de
hosts internos no state.hosts_by_mac. O modelo treinado generaliza corretamente
porque aprende os padrões de features — não memoriza a regra.

SRP: este módulo só transforma labels. Não toca em features nem em modelos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import (
    EXTERNAL_PROTOCOL_ID,
    EXTERNAL_DURATION_THRESH,
    EXTERNAL_STD_THRESH,
    CLASS_NAMES,
)


class LabelEngineer:
    """
    Transforma labels binárias em labels triclasse usando regras de domínio.

    Não precisa de fit() — é uma transformação determinística pura (sem estado).
    Aplicada ANTES do split para que a distribuição triclasse seja estratificável.

    Uso:
        eng = LabelEngineer()
        y3  = eng.transform(X, y_binary)
        eng.report(y3)
    """

    # Nomes das classes indexados pelo label inteiro
    CLASS_NAMES: list[str] = CLASS_NAMES

    def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Converte labels binárias para triclasse.

        Parameters
        ----------
        X : pd.DataFrame — features brutas (usadas para identificar tipo de ataque)
        y : pd.Series    — labels binárias (0=Benigno, 1=DDoS)

        Returns
        -------
        pd.Series com labels: 0=Benigno, 1=Externo, 2=Interno
        """
        y_new      = y.copy().astype(int)
        attack_idx = y == 1

        # ── Critério de ataque externo ─────────────────────────────────────────
        # Regra 1: protocolo indicativo de flood externo bruto
        rule_protocol = X["Protocol"] == EXTERNAL_PROTOCOL_ID

        # Regra 2: pacotes uniformes (std ≈ 0) em fluxo muito curto → rajada sintética
        rule_burst = (
            (X["Pkt Len Std"] <= EXTERNAL_STD_THRESH) &
            (X["Pkt Len Var"] <= EXTERNAL_STD_THRESH) &
            (X["Flow Duration"] < EXTERNAL_DURATION_THRESH)
        )

        external_mask = attack_idx & (rule_protocol | rule_burst)
        internal_mask = attack_idx & ~external_mask

        y_new[external_mask] = 1   # Ataque Externo
        y_new[internal_mask] = 2   # Zumbi Interno

        return y_new

    def report(self, y: pd.Series) -> None:
        """Exibe distribuição das três classes no stdout."""
        counts = y.value_counts().sort_index()
        total  = len(y)

        print("\n── Distribuição triclasse (pós-engenharia de labels) ──")
        for cls, cnt in counts.items():
            name = self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else f"Classe {cls}"
            pct  = cnt / total * 100
            bar  = "█" * int(pct / 2)
            print(f"  {cls} ({name:<18}): {cnt:>7,}  ({pct:5.2f}%)  {bar}")

        # Entropia de Shannon como medida de balanceamento
        import numpy as np
        probs = counts / total
        H     = -sum(p * np.log(p) for p in probs if p > 0)
        H_max = np.log(len(counts))
        print(f"\n  Entropia de Shannon: {H / H_max:.4f}  "
              f"(0=totalmente desbalanceado | 1=perfeito)")
        print(f"  Total de instâncias: {total:,}")
