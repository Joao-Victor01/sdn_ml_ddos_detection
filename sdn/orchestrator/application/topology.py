"""
Caso de uso: Topologia de Switches — Etapa [1/6].

Responsabilidades:
  - Buscar a topologia de switches e enlaces via REST do ODL
  - Detectar mudanças de topologia (enlaces adicionados/removidos)
  - Gerenciar TTL de switches (remover switches que saíram da rede)
  - Instalar flows base (TABLE-MISS, LLDP, BDDP) em switches novos
  - Remover flows rogue do arphandler do ODL (watchdog)
  - Manter o conjunto de switches válidos (com tabelas de flow confirmadas)

Funções utilitárias link_key() e out_port() também residem aqui pois
são exclusivamente relacionadas à lógica de topologia e usadas pelos
casos de uso de roteamento.
"""

import subprocess
import requests
import networkx as nx
from collections import defaultdict

import orchestrator.domain.state as state_module
from orchestrator.config import (
    AUTH, HEADERS_JSON, URL_TOPO, URL_NODES, SW_TTL_CYCLES,
)
from orchestrator.domain.state import state
from orchestrator.infrastructure.flow_specs import (
    flow_table_miss, flow_lldp, flow_bddp,
)
from orchestrator.infrastructure.ovs_adapter import (
    install_flows_parallel, delete_flows_parallel, verify_table_miss,
)


# ── Utilitários de topologia ────────────────────────────────────────────────

def link_key(u: str, v: str) -> tuple:
    """Chave canônica de enlace (sempre na mesma ordem) para uso em dicionários."""
    return tuple(sorted([u, v]))


def out_port(edge_attrs: dict, current_switch: str) -> str:
    """Retorna o número da porta de saída do current_switch para este enlace."""
    src_tp = edge_attrs.get("src_port", "")
    dst_tp = edge_attrs.get("dst_port", "")
    if src_tp.startswith(current_switch + ":"):
        return src_tp.split(":")[-1]
    return dst_tp.split(":")[-1]


# ── Watchdog: switches válidos ──────────────────────────────────────────────

def refresh_valid_switches() -> None:
    """
    Atualiza o conjunto de switches com tabelas de flow confirmadas pelo ODL.
    Executado a cada 10 ciclos no loop de controle.
    """
    try:
        r = requests.get(URL_NODES, auth=AUTH, headers=HEADERS_JSON, timeout=5)
        if r.status_code != 200:
            return
        nodes = (r.json()
                  .get("opendaylight-inventory:nodes", {})
                  .get("node", []))
        valid = set()
        for n in nodes:
            nid    = n.get("id", "")
            tables = n.get("flow-node-inventory:table", [])
            if "openflow" in nid and len(tables) >= 2:
                valid.add(nid)
        with state.lock:
            state._valid_switches = valid
    except Exception:
        pass


# ── Watchdog: flows rogue do arphandler ────────────────────────────────────

def remove_rogue_arp_flows() -> None:
    """
    Remove flows rogue priority=65000,arp instalados pelo arphandler do ODL.

    O arphandler instala 'priority=65000,arp,actions=CONTROLLER:65535,output:ALL'
    que causa flood massivo e CPU 99%. Este watchdog remove esses flows a cada
    3 ciclos para garantir que não reapareçam após reinício do arphandler.
    """
    with state.lock:
        switches   = list(state.graph.nodes)
        containers = dict(state.sw_to_container)
    if not switches:
        return
    for sw in switches:
        container = containers.get(sw)
        if not container:
            continue
        try:
            # Verifica se o flow rogue existe
            r = subprocess.run(
                ["docker", "exec", container,
                 "ovs-ofctl", "dump-flows", "br0", "-O", "OpenFlow13"],
                capture_output=True, text=True, timeout=8
            )
            if "priority=65000" in r.stdout and "arp" in r.stdout:
                # Remove o flow rogue
                subprocess.run(
                    ["docker", "exec", container,
                     "ovs-ofctl", "--strict", "del-flows", "br0",
                     "priority=65000,arp", "-O", "OpenFlow13"],
                    capture_output=True, text=True, timeout=8
                )
                print(f"  🧹 Flow rogue priority=65000,arp removido de {sw}")
        except Exception:
            pass


# ── Flows base ──────────────────────────────────────────────────────────────

def install_base_flows(nodes: list) -> None:
    """
    Instala TABLE_MISS + LLDP + BDDP em cada switch novo — paralelo via executor.
    Idempotente: switches já em _guard_done são pulados.

    v14.1: adiciona verificação real via dump-flows para detectar switches
    onde o docker exec "teve sucesso" mas o flow não foi efetivamente instalado
    (ex: race condition, OVS reiniciou, fail-mode mudou).
    """
    with state.lock:
        guard = set(state._guard_done)

    # Dois grupos: novos (nunca guard_done) + suspeitos (guard_done mas sem TABLE-MISS real)
    new_nodes     = [sw for sw in nodes if sw not in guard]
    # A cada 15 ciclos verifica switches já marcados como done — detecta TABLE-MISS que sumiu
    suspect_nodes = []
    if state_module.CYCLE_COUNT % 15 == 1:
        for sw in guard:
            if sw in nodes and not verify_table_miss(sw):
                with state.lock:
                    state._guard_done.discard(sw)
                    # Limpa active_flows para forçar reinstalação
                    for key in ["TABLE_MISS_CONTROLLER", "LLDP_to_Controller", "BDDP_to_Controller"]:
                        state.active_flows.pop((sw, key), None)
                suspect_nodes.append(sw)
                print(f"  ⚠️  {sw}: TABLE-MISS ausente — reinstalando flows base")

    pending = new_nodes + suspect_nodes
    if not pending:
        return

    tasks = []
    for sw in pending:
        tasks.append((sw, "TABLE_MISS_CONTROLLER", flow_table_miss(), False))
        tasks.append((sw, "LLDP_to_Controller",    flow_lldp(),       False))
        tasks.append((sw, "BDDP_to_Controller",    flow_bddp(),       False))

    install_flows_parallel(tasks)

    # Verifica quais switches tiveram os 3 flows instalados com sucesso
    with state.lock:
        for sw in pending:
            keys = [(sw, "TABLE_MISS_CONTROLLER"),
                    (sw, "LLDP_to_Controller"),
                    (sw, "BDDP_to_Controller")]
            if all(k in state.active_flows for k in keys):
                state._guard_done.add(sw)
                print(f"  ✅ Flows base instalados em {sw}")
            else:
                missing = [k[1] for k in keys if k not in state.active_flows]
                print(f"  ❌ {sw}: falha ao instalar flows base: {missing}")


# ── Caso de uso principal ───────────────────────────────────────────────────

def fetch_topology() -> None:
    """
    Etapa [1/6]: descobre switches e enlaces via REST do ODL.

    - Constrói o grafo NetworkX com os switches ativos
    - Detecta mudança de topologia por comparação de conjunto de arestas
    - Aplica TTL de switches (remove switches desaparecidos e seus hosts órfãos)
    - Instala flows base nos switches novos
    """
    print("--- [1/6] Topologia de Switches ---")
    try:
        resp = requests.get(URL_TOPO, auth=AUTH, headers=HEADERS_JSON, timeout=6)
        if resp.status_code != 200:
            return
        data     = resp.json()
        topo_raw = (data.get("network-topology:topology") or
                    data.get("topology", [{}]))
        topo     = topo_raw[0] if isinstance(topo_raw, list) else topo_raw
        if not topo:
            return

        with state.lock:
            valid_sw   = set(state._valid_switches)
            blocked_sw = list(state.blocked_switches)
            link_costs = dict(state.link_costs)

        new_graph  = nx.Graph()
        edge_ports = defaultdict(set)

        for node in topo.get("node", []):
            nid = node.get("node-id", "")
            if nid.startswith("host:") or "::" in nid:
                continue
            if nid in blocked_sw:
                continue
            if valid_sw and nid not in valid_sw:
                continue
            new_graph.add_node(nid)
            for tp in node.get("termination-point", []):
                tpid = tp.get("tp-id", "")
                if "LOCAL" not in tpid:
                    edge_ports[nid].add(tpid.split(":")[-1])

        lk_count = 0
        for link in topo.get("link", []):
            src    = link["source"]["source-node"]
            dst    = link["destination"]["dest-node"]
            src_tp = link["source"]["source-tp"]
            dst_tp = link["destination"]["dest-tp"]
            if src.startswith("host:") or dst.startswith("host:"):
                continue
            if src in blocked_sw or dst in blocked_sw:
                continue
            if src not in new_graph or dst not in new_graph:
                continue
            edge_ports[src].discard(src_tp.split(":")[-1])
            edge_ports[dst].discard(dst_tp.split(":")[-1])
            cost = link_costs.get(link_key(src, dst), 1)
            if src_tp.startswith(src + ":"):
                new_graph.add_edge(src, dst, weight=cost,
                                   src_port=src_tp, dst_port=dst_tp)
            else:
                new_graph.add_edge(src, dst, weight=cost,
                                   src_port=dst_tp, dst_port=src_tp)
            lk_count += 1

        with state.lock:
            old_nodes    = set(state.graph.nodes)   # salva ANTES de atualizar
            state.graph      = new_graph
            state.edge_ports = edge_ports
            # Detecta mudança de topologia por comparação de conjunto de arestas
            curr_edges = frozenset(
                link_key(u, v) for u, v in new_graph.edges()
            )
            if curr_edges != state._prev_edges and state._prev_edges != frozenset():
                state.topo_changed = True
                delta = lk_count - len(state._prev_edges)
                sign  = "+" if delta > 0 else ""
                print(f"  🔄 TOPOLOGIA MUDOU: {sign}{delta} enlaces "
                      f"({len(state._prev_edges)} → {lk_count})"
                      f" — recalculando MST...")
            state._prev_edges = curr_edges

        # ── TTL de switches: detecta remoção e limpa flows órfãos ──────────
        with state.lock:
            prev_nodes     = old_nodes - set(new_graph.nodes)
            sw_miss_before = dict(state._sw_missing_cycles)

        removed_switches = []
        for sw in prev_nodes:
            cycles = sw_miss_before.get(sw, 0) + 1
            with state.lock:
                state._sw_missing_cycles[sw] = cycles
            if cycles >= SW_TTL_CYCLES:
                removed_switches.append(sw)

        # Limpa ciclos de switches que voltaram
        for sw in new_graph.nodes:
            with state.lock:
                state._sw_missing_cycles.pop(sw, None)

        if removed_switches:
            print(f"  🗑️  {len(removed_switches)} switch(es) removido(s): "
                  f"{[s.replace('openflow:','sw') for s in removed_switches]}")
            orphan_ips: list[str] = []
            with state.lock:
                surviving = set(new_graph.nodes)
                for sw in removed_switches:
                    # Remove todos os flows deste switch do active_flows
                    stale = [k for k in state.active_flows if k[0] == sw]
                    for k in stale:
                        state.active_flows.pop(k, None)
                    state._guard_done.discard(sw)
                    state._sw_missing_cycles.pop(sw, None)
                    # Remove hosts cujo attachment point era este switch
                    # e coleta IPs para deletar flows IPv4 nos switches restantes
                    orphan_macs = [mac for mac, info in state.hosts_by_mac.items()
                                   if info.get("switch") == sw]
                    for mac in orphan_macs:
                        info = state.hosts_by_mac.pop(mac, {})
                        ips  = info.get("ips", [])
                        orphan_ips.extend(ips)
                        for ip in ips:
                            state.ip_to_mac.pop(ip, None)
                            # Remove entries de active_flows para estes IPs
                            fid    = f"IPv4_{ip.replace('.', '_')}"
                            stale2 = [k for k in state.active_flows if k[1] == fid]
                            for k in stale2:
                                state.active_flows.pop(k, None)
                        print(f"    🗑️  Host órfão removido: {ips} | {mac}")
            # Deleta flows IPv4 dos hosts órfãos nos switches sobreviventes
            if orphan_ips and surviving:
                del_tasks2 = []
                for ip in orphan_ips:
                    fid = f"IPv4_{ip.replace('.', '_')}"
                    for sw in surviving:
                        del_tasks2.append((sw, fid))
                if del_tasks2:
                    print(f"    🗑️  Removendo rotas IPv4 de {len(orphan_ips)} host(s) órfão(s)")
                    delete_flows_parallel(del_tasks2)

        sw_count = len(new_graph.nodes)
        if lk_count == 0:
            print(f"  {sw_count} switches | 0 enlaces — aguardando LLDP...")
        else:
            print(f"  OK: {sw_count} switches | {lk_count} enlaces")

        install_base_flows(list(new_graph.nodes))

    except Exception as e:
        print(f"  ❌ fetch_topology: {e}")
