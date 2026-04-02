"""
Caso de uso: Descoberta de Hosts e ARP Probing — Etapas [2/6] e [2b].

Responsabilidades:
  - Descobrir hosts via l2switch host-tracker do ODL (fetch_hosts)
  - Gerenciar TTL de hosts (remover hosts ausentes do ODL)
  - Enviar ARP probes para detectar hosts fisicamente desconectados (probe_hosts)
  - Manter state.hosts_by_mac, state.ip_to_mac e state.edge_ports atualizados

Nota v13: removido o filtro inter_switch_ports que excluía hosts cujo
attachment point tinha o mesmo número de porta que um enlace inter-switch.
O ODL l2switch host-tracker já garante que entradas host:MAC só são criadas
para dispositivos de borda — o LLDP flow em priority=1000 impede que pacotes
de outro switch sejam tratados como host.
"""

import requests

import orchestrator.domain.state as state_module
from orchestrator.config import (
    AUTH, HEADERS_JSON, URL_TOPO_OP, ODL_IP,
    HOST_TTL_CYCLES, HOST_PROBE_CYCLES, HOST_PROBE_MISS,
)
from orchestrator.domain.state import state
from orchestrator.infrastructure.ovs_adapter import delete_flows_parallel


# ── ARP Probe ───────────────────────────────────────────────────────────────

def _send_arp_probe(sw_id: str, port: str) -> bool:
    """
    Envia packet-out com ARP request broadcast pelo switch/porta indicados.

    Isso força o host a responder com ARP reply, que o arphandler do ODL usa
    para atualizar o timestamp de last-seen — sem resposta = host offline.
    """
    url = (f"http://{ODL_IP}:8181/rests/operations/"
           f"opendaylight-packet-processing:transmit-packet")
    # ARP request genérico: ff:ff:ff:ff:ff:ff broadcast, who-has 0.0.0.0
    # Mínimo válido: Ethernet(14) + ARP(28) = 42 bytes, padding para 64
    arp_hex = (
        "ffffffffffff"   # dst MAC: broadcast
        "000000000001"   # src MAC: fictício
        "0806"           # EtherType: ARP
        "0001"           # hardware type: Ethernet
        "0800"           # protocol type: IPv4
        "06" "04"        # hw size, proto size
        "0001"           # opcode: request
        "000000000001"   # sender MAC
        "00000000"       # sender IP: 0.0.0.0
        "ffffffffffff"   # target MAC: broadcast
        "00000000"       # target IP: 0.0.0.0
        "0000000000000000000000000000"  # padding
    )
    payload = {
        "input": {
            "node": f"openflow:{sw_id.split(':')[1]}",
            "egress-port": f"{sw_id}:{port}",
            "payload": arp_hex
        }
    }
    try:
        r = requests.post(url, json=payload, auth=AUTH,
                          headers=HEADERS_JSON, timeout=3)
        return r.status_code in (200, 204)
    except Exception:
        return False


def probe_hosts() -> None:
    """
    Etapa [2b]: envia ARP probes a cada HOST_PROBE_CYCLES ciclos.

    - Envia ARP probe para cada host conhecido
    - Incrementa contador de miss de probe
    - Se miss >= HOST_PROBE_MISS: força remoção local (host offline)

    Isso é necessário porque o ODL host-tracker nunca remove hosts
    automaticamente — sem probing, hosts desconectados ficam no estado
    para sempre.
    """
    if state_module.CYCLE_COUNT % HOST_PROBE_CYCLES != 0:
        return

    with state.lock:
        hosts_snap = dict(state.hosts_by_mac)

    for mac, info in hosts_snap.items():
        sw   = info.get("switch", "")
        port = info.get("port", "")
        if sw and port:
            ok = _send_arp_probe(sw, port)
            with state.lock:
                if ok:
                    # Probe enviado — incrementa contador de miss
                    # (reset a 0 quando host responder via fetch_hosts)
                    prev = state._host_probe_sent.get(mac, 0)
                    state._host_probe_sent[mac] = prev + 1
                    if prev + 1 >= HOST_PROBE_MISS:
                        # Forçar remoção local: marca como ausente
                        state.host_missing_cycles[mac] = HOST_TTL_CYCLES
                        print(f"  📡 Probe sem resposta ({prev+1}x): "
                              f"{info.get('ips')} | {mac} — marcando como ausente")

    # Reset probe miss para hosts que responderam ao último fetch_hosts
    # (host_missing_cycles == 0 significa que foi visto neste ciclo)
    with state.lock:
        for mac in list(state._host_probe_sent.keys()):
            if state.host_missing_cycles.get(mac, 0) == 0:
                state._host_probe_sent[mac] = 0


# ── Caso de uso principal ───────────────────────────────────────────────────

def fetch_hosts() -> None:
    """
    Etapa [2/6]: descobre hosts via l2switch host-tracker do ODL.

    - Busca nós host:MAC na topologia operacional do ODL
    - Atualiza state.hosts_by_mac e state.ip_to_mac
    - Aplica TTL de hosts (remove hosts ausentes e seus flows IPv4)
    """
    print("--- [2/6] Hosts (via l2switch) ---")
    try:
        resp = requests.get(URL_TOPO_OP, auth=AUTH,
                            headers=HEADERS_JSON, timeout=6)
        if resp.status_code != 200:
            print(f"  ⚠️  HTTP {resp.status_code}")
            return

        data     = resp.json()
        topo_raw = (data.get("network-topology:topology") or
                    data.get("topology", [{}]))
        topo     = topo_raw[0] if isinstance(topo_raw, list) else topo_raw

        # Conjunto de termination-points que são sabidamente inter-switch
        # (usados apenas para avisar no log, não para filtrar hosts)
        with state.lock:
            g_snap = state.graph.copy()
        inter_switch_tps: set[str] = set()
        for u, v, attrs in g_snap.edges(data=True):
            inter_switch_tps.add(attrs.get("src_port", ""))
            inter_switch_tps.add(attrs.get("dst_port", ""))

        new_hosts  = {}
        new_ip2mac = {}

        for node in topo.get("node", []):
            nid = node.get("node-id", "")
            if not nid.startswith("host:"):
                continue
            mac = nid.replace("host:", "")
            if len(mac.split(":")) != 6:
                continue

            # Tenta todos os campos possíveis onde o ODL pode guardar IPs
            addr_raw = (node.get("address-tracker:addresses") or
                        node.get("host-tracker-service:addresses") or
                        node.get("addresses") or [])
            if not isinstance(addr_raw, list):
                addr_raw = []

            attachments = (node.get("host-tracker-service:attachment-points") or
                           node.get("attachment-points", []))

            ips = [a.get("ip") for a in addr_raw
                   if isinstance(a, dict) and a.get("ip")
                   and not a.get("ip", "").startswith("fe80")]

            if not ips:
                # Host registrado no ODL sem IP — provavelmente ARP chegou
                # mas sem sender-IP (VPC sem `ip` configurado) ou campo diferente
                if attachments:
                    tp = attachments[0].get("tp-id", "?")
                    print(f"  ⚠️  MAC-only (sem IP): {mac} → {tp} "
                          f"— VPC não tem `ip` configurado?")
                continue

            for att in attachments:
                tp_id = att.get("tp-id", "")
                if not tp_id or "LOCAL" in tp_id:
                    continue
                parts = tp_id.split(":")
                if len(parts) < 3:
                    continue
                switch_id = ":".join(parts[:2])
                port_num  = parts[-1]

                # Avisa se ODL está reportando um host em porta inter-switch
                # (suspeito, mas não filtramos — confiamos no host-tracker)
                if tp_id in inter_switch_tps:
                    print(f"  ⚠️  Host {ips} em porta inter-switch {tp_id} "
                          f"— possível falso positivo do ODL")

                for ip in ips:
                    new_hosts[mac] = {
                        "mac": mac, "ips": ips,
                        "switch": switch_id, "port": port_num,
                    }
                    new_ip2mac[ip] = mac

        with state.lock:
            # ── Atualiza hosts encontrados ──────────────────────────────
            for mac, info in new_hosts.items():
                is_new = mac not in state.hosts_by_mac
                state.hosts_by_mac[mac] = info
                state.host_missing_cycles[mac] = 0  # visto neste ciclo
                for ip in info.get("ips", []):
                    state.ip_to_mac[ip] = mac
                sw = info["switch"]
                if sw not in state.edge_ports:
                    state.edge_ports[sw] = set()
                state.edge_ports[sw].add(info["port"])
                if is_new:
                    print(f"  🔍 Novo host: {info['ips']} | {mac}"
                          f" → {info['switch']} porta {info['port']}")

            # ── TTL: incrementa ausentes, remove os que passaram do limite ──
            expired = []
            for mac in list(state.hosts_by_mac.keys()):
                if mac not in new_hosts:
                    state.host_missing_cycles[mac] = (
                        state.host_missing_cycles.get(mac, 0) + 1
                    )
                    if state.host_missing_cycles[mac] >= HOST_TTL_CYCLES:
                        expired.append(mac)

            expired_ips: dict[str, list] = {}  # mac → [ips] para usar fora do lock
            for mac in expired:
                info = state.hosts_by_mac.pop(mac, {})
                state.host_missing_cycles.pop(mac, None)
                ips = info.get("ips", [])
                expired_ips[mac] = ips  # salva ANTES de limpar ip_to_mac
                for ip in ips:
                    state.ip_to_mac.pop(ip, None)
                    fid   = f"IPv4_{ip.replace('.', '_')}"
                    stale = [(sw, fid) for sw in state.graph.nodes
                             if (sw, fid) in state.active_flows]
                    for sw, f in stale:
                        state.active_flows.pop((sw, f), None)
                print(f"  🗑️  Host removido (ausente {HOST_TTL_CYCLES} ciclos): "
                      f"{ips} | {mac}")

            total = len(state.hosts_by_mac)

        # Remove flows IPv4 dos hosts expirados fora do lock (docker exec)
        if expired:
            del_tasks = []
            with state.lock:
                g_snap = state.graph.copy()
            for mac, ips in expired_ips.items():
                for ip in ips:
                    fid = f"IPv4_{ip.replace('.', '_')}"
                    for sw in g_snap.nodes:
                        del_tasks.append((sw, fid))
            if del_tasks:
                n = len(set(ip for ips in expired_ips.values() for ip in ips))
                print(f"  🗑️  Removendo flows IPv4 de {n} host(s) expirado(s)")
                delete_flows_parallel(del_tasks)

        if not new_hosts:
            print("  Aguardando tráfego ARP das VPCs...")
        else:
            print(f"  OK: {total} host(s) conhecido(s)")

    except Exception as e:
        print(f"  ❌ fetch_hosts: {e}")
