"""
IP Traceback e Isolamento Cirúrgico — módulo de mitigação multiclasse.

Implementa os dois mecanismos de resposta ao ataque:

## 1. IP Traceback (teoria dos grafos + visão global do SDN)

Usa o grafo NetworkX do estado para rastrear o caminho do tráfego malicioso
de volta à sua origem. O controlador SDN tem visão global da topologia —
sabe exatamente em qual switch e porta cada host está conectado.

Algoritmo:
  1. Encontrar o host de origem em state.hosts_by_mac (via ip_to_mac).
  2. Identificar o switch de borda onde o host está conectado.
  3. Calcular o caminho Dijkstra entre o switch de origem e o switch alvo.
  4. Retornar a lista ordenada de switches no caminho de ataque.

## 2. Isolamento Cirúrgico (OpenFlow via docker exec ovs-ofctl)

Para zumbis internos, instala um flow DROP de alta prioridade (65400)
APENAS na porta de borda onde o host comprometido está conectado.

Isso é cirúrgico: só bloqueia o host infectado específico, sem afetar
nenhum outro host na mesma VLAN ou sub-rede. O tráfego legítimo dos
demais usuários continua fluindo normalmente.

Diferença de /manage/ip (bloqueia em todos os switches):
  /manage/ip instala flow DROP em TODOS os switches da topologia (proteção ampla).
  isolate()  instala flow DROP APENAS na porta de borda do host infectado
             (isolamento preciso — outros hosts no mesmo switch não são afetados).

SRP: este módulo rastreia e isola. Não classifica, não modifica topologia.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import networkx as nx

from orchestrator.domain.state import state
from orchestrator.infrastructure.docker_adapter import container_for
from orchestrator.infrastructure.flow_specs import flow_ip_drop

# Prioridade dos flows de isolamento — entre o DROP de IP (65500) e o IPv4 (60000)
ISOLATION_PRIORITY = 65400

# Registro em memória dos hosts isolados: {ip → IsolationRecord}
_isolated_hosts: dict[str, "IsolationRecord"] = {}


@dataclass
class TracebackResult:
    """Resultado do rastreamento de um IP de origem."""
    src_ip:        str
    src_mac:       str | None
    src_switch:    str | None          # openflow:X do switch de borda
    src_port:      str | None          # porta onde o host está conectado
    dst_switch:    str | None          # switch de destino (ou controlador)
    attack_path:   list[str]           # switches no caminho de ataque
    found:         bool
    reason:        str

    def to_dict(self) -> dict:
        return {
            "src_ip":      self.src_ip,
            "src_mac":     self.src_mac,
            "src_switch":  self.src_switch,
            "src_port":    self.src_port,
            "attack_path": self.attack_path,
            "found":       self.found,
            "reason":      self.reason,
        }


@dataclass
class IsolationRecord:
    """Registro de um host isolado."""
    ip:         str
    mac:        str | None
    switch_id:  str
    port:       str
    flow_id:    str
    active:     bool = True


class IPTraceback:
    """
    Rastreia e isola hosts maliciosos usando a topologia SDN.

    Uso:
        tb = IPTraceback()

        # Rastrear origem
        result = tb.traceback("10.0.0.5")
        print(result.attack_path)   # ['openflow:3', 'openflow:1']

        # Isolar zumbi interno
        ok = tb.isolate("10.0.0.5")

        # Liberar host após análise/limpeza
        tb.release("10.0.0.5")

        # Status atual
        tb.list_isolated()
    """

    # ── Traceback ──────────────────────────────────────────────────────────────

    def traceback(
        self,
        src_ip:     str,
        dst_switch: str | None = None,
    ) -> TracebackResult:
        """
        Rastreia o caminho do tráfego entre o host de origem e o destino.

        Parameters
        ----------
        src_ip     : IP do host suspeito
        dst_switch : switch de destino (padrão: primeiro switch da topologia)

        Returns
        -------
        TracebackResult com o caminho de ataque na topologia.
        """
        with state.lock:
            ip_to_mac   = dict(state.ip_to_mac)
            hosts_by_mac = dict(state.hosts_by_mac)
            graph        = state.graph.copy()

        # 1. Localizar o host de origem
        mac = ip_to_mac.get(src_ip)
        if mac is None:
            return TracebackResult(
                src_ip=src_ip, src_mac=None, src_switch=None, src_port=None,
                dst_switch=dst_switch, attack_path=[], found=False,
                reason=f"IP {src_ip} não registrado no SDN — possível IP externo/spoofado.",
            )

        host_info  = hosts_by_mac.get(mac, {})
        src_switch = host_info.get("switch")
        src_port   = host_info.get("port")

        if src_switch is None:
            return TracebackResult(
                src_ip=src_ip, src_mac=mac, src_switch=None, src_port=None,
                dst_switch=dst_switch, attack_path=[], found=False,
                reason=f"MAC {mac} encontrado mas switch de borda não identificado.",
            )

        # 2. Calcular caminho Dijkstra até o destino
        if dst_switch is None:
            nodes = list(graph.nodes)
            dst_switch = nodes[0] if nodes else src_switch

        try:
            path = nx.shortest_path(graph, source=src_switch, target=dst_switch)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [src_switch]

        return TracebackResult(
            src_ip=src_ip, src_mac=mac, src_switch=src_switch, src_port=src_port,
            dst_switch=dst_switch, attack_path=path, found=True,
            reason=(f"Host {src_ip} ({mac}) conectado em {src_switch}:{src_port}. "
                    f"Caminho: {' → '.join(path)}."),
        )

    # ── Isolamento cirúrgico ───────────────────────────────────────────────────

    def isolate(self, src_ip: str) -> dict:
        """
        Instala flow DROP cirúrgico na porta de borda do host infectado.

        Diferente de /manage/ip (DROP global em todos os switches), este método
        instala o flow APENAS no switch de borda onde o host está diretamente
        conectado — sem afetar outros hosts no mesmo switch.

        Returns
        -------
        dict com status da operação.
        """
        if src_ip in _isolated_hosts and _isolated_hosts[src_ip].active:
            return {"status": "already_isolated", "ip": src_ip}

        tb = self.traceback(src_ip)
        if not tb.found:
            return {
                "status": "failed",
                "ip":     src_ip,
                "reason": tb.reason,
            }

        sw        = tb.src_switch
        port      = tb.src_port
        mac       = tb.src_mac
        container = container_for(sw)

        if not container:
            return {
                "status": "failed",
                "ip":     src_ip,
                "reason": f"Switch {sw} não mapeado para container Docker.",
            }

        # Flow cirúrgico: DROP apenas tráfego saindo da porta onde o zumbi está
        # in_port=porta → garante que só bloqueia o host infectado, não o switch inteiro
        flow_id  = f"ISOLATE_{src_ip.replace('.', '_')}"
        flow_str = (
            f"priority={ISOLATION_PRIORITY},"
            f"in_port={port},"
            f"actions=drop"
        )

        cmd = [
            "docker", "exec", container,
            "ovs-ofctl", "add-flow", "br0", flow_str, "-O", "OpenFlow13",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if r.returncode != 0:
                return {
                    "status": "failed",
                    "ip":     src_ip,
                    "reason": f"ovs-ofctl error: {r.stderr.strip()[:200]}",
                }

            # Registrar isolamento no estado global
            record = IsolationRecord(
                ip=src_ip, mac=mac, switch_id=sw,
                port=port, flow_id=flow_id, active=True,
            )
            _isolated_hosts[src_ip] = record

            with state.lock:
                state.active_flows[(sw, flow_id)] = flow_str

            print(f"[IPTraceback] ✓ Zumbi ISOLADO: {src_ip} ({mac}) "
                  f"em {sw}:{port}")

            return {
                "status":    "isolated",
                "ip":        src_ip,
                "mac":       mac,
                "switch":    sw,
                "port":      port,
                "flow":      flow_str,
                "attack_path": tb.attack_path,
            }

        except Exception as e:
            return {"status": "error", "ip": src_ip, "reason": str(e)}

    def release(self, src_ip: str) -> dict:
        """
        Remove o flow de isolamento — libera o host após análise/limpeza.

        Usado quando o host foi desinfectado ou identificado como falso positivo.
        """
        record = _isolated_hosts.get(src_ip)
        if record is None or not record.active:
            return {"status": "not_isolated", "ip": src_ip}

        container = container_for(record.switch_id)
        if not container:
            return {"status": "failed", "reason": "Container não encontrado."}

        # Remover o flow de isolamento do OVS
        del_str = (
            f"priority={ISOLATION_PRIORITY},"
            f"in_port={record.port}"
        )
        cmd = [
            "docker", "exec", container,
            "ovs-ofctl", "del-flows", "br0", del_str, "-O", "OpenFlow13",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except Exception as e:
            return {"status": "error", "reason": str(e)}

        record.active = False
        with state.lock:
            state.active_flows.pop((record.switch_id, record.flow_id), None)

        print(f"[IPTraceback] ✓ Host LIBERADO: {src_ip} ({record.mac})")
        return {"status": "released", "ip": src_ip, "mac": record.mac}

    def list_isolated(self) -> list[dict]:
        """Retorna lista de todos os hosts atualmente isolados."""
        return [
            {
                "ip":        r.ip,
                "mac":       r.mac,
                "switch":    r.switch_id,
                "port":      r.port,
                "active":    r.active,
            }
            for r in _isolated_hosts.values()
        ]
