"""
Casos de uso: Roteamento IPv4, ARP Spanning-Tree e Reroute — Etapas [4-6/6].

Três casos de uso relacionados ao plano de dados:

  install_ipv4_routes()       [4/6]
    Calcula rotas host-by-host via Dijkstra e instala flows IPv4 proativos
    (priority=60000, permanentes) em todos os switches do caminho.

  install_arp_spanning_tree() [5/6]
    Calcula o MST da topologia e instala:
      - Flows ARP loop-free (priority=5000) por porta de entrada
      - Flows DROP em portas não-MST (priority=3) para bloquear flooding

  check_and_reroute()         [6/6]
    Se algum enlace está congestionado (>REROUTE_THRESH), instala flows
    de desvio temporários (priority=62000, idle=15s) evitando esses enlaces.
"""

import networkx as nx
from collections import defaultdict

from orchestrator.config import MAX_LINK_CAPACITY, REROUTE_THRESH, ENABLE_REROUTING
from orchestrator.domain.state import state
from orchestrator.infrastructure.flow_specs import (
    flow_ipv4_forward, flow_arp_mst, flow_ip_drop, flow_flood_block,
)
from orchestrator.infrastructure.ovs_adapter import (
    install_flows_parallel, delete_flows_parallel, port_to_iface,
)
from orchestrator.application.topology import link_key, out_port


def install_ipv4_routes() -> None:
    """
    Etapa [4/6]: rotas IPv4 host-by-host via Dijkstra — paralelo.

    Acumula TODOS os flows de todos os switches em uma lista de tasks
    e submete ao ThreadPoolExecutor de uma vez. Tempo: max(1 exec) ≈ 0.5-1s
    independente de quantos switches/hosts existam.
    """
    print("--- [4/6] Rotas IPv4 ---")
    with state.lock:
        g           = state.graph.copy()
        hosts       = dict(state.hosts_by_mac)
        ip_to_mac   = dict(state.ip_to_mac)
        # pending_unblocks: IPs sendo desbloquados agora — não reinstalar
        blocked_ips = [ip for ip in state.blocked_ips
                       if ip not in state.pending_unblocks]

    if not list(g.nodes) or len(g.edges) == 0:
        print("  Aguardando topologia...")
        return

    routable = [
        (ip, mac, hosts[mac])
        for ip, mac in ip_to_mac.items()
        if hosts.get(mac, {}).get("switch") in g.nodes
    ]
    if not routable:
        print("  Aguardando hosts...")
        return

    # Acumula tasks: (sw_id, flow_id, ovs_flow_str, silent)
    tasks: list[tuple] = []

    for dst_ip, dst_mac, host_info in routable:
        dst_switch = host_info["switch"]
        dst_port   = host_info["port"]
        dst_iface  = port_to_iface(dst_port)
        flow_id    = f"IPv4_{dst_ip.replace('.', '_')}"

        try:
            _, paths = nx.single_source_dijkstra(g, dst_switch, weight="weight")
        except Exception:
            continue

        for src_switch, path_rev in paths.items():
            if src_switch == dst_switch:
                fstr = flow_ipv4_forward(dst_ip, dst_iface, 60000)
                tasks.append((dst_switch, flow_id, fstr, True))
                continue

            path = list(reversed(path_rev))
            for i in range(len(path) - 1):
                curr, nxt = path[i], path[i + 1]
                edge = g.get_edge_data(curr, nxt)
                if not edge:
                    continue
                out_iface = port_to_iface(out_port(edge, curr))
                fstr = flow_ipv4_forward(dst_ip, out_iface, 60000)
                tasks.append((curr, flow_id, fstr, True))

    # Instala todos em paralelo
    ok, fail = install_flows_parallel(tasks)

    # Flows de bloqueio de IP (também paralelos)
    if blocked_ips:
        block_tasks = [
            (sw, f"Block_{ip}", flow_ip_drop(ip), True)
            for sw in g.nodes
            for ip in blocked_ips
        ]
        install_flows_parallel(block_tasks)

    print(f"  OK: {ok} entradas | {len(routable)} hosts roteáveis"
          + (f" | {fail} falhas" if fail else ""))

    # Mostra resumo de caminho por destino
    for dst_ip, dst_mac, host_info in routable:
        dst_switch = host_info["switch"]
        try:
            _, paths = nx.single_source_dijkstra(g, dst_switch, weight="weight")
        except Exception:
            continue
        hops_seen: set = set()
        for src_sw, path_rev in sorted(paths.items()):
            if src_sw == dst_switch:
                continue
            path = list(reversed(path_rev))
            path_str = " → ".join(sw.replace("openflow:", "sw") for sw in path)
            if path_str not in hops_seen:
                hops_seen.add(path_str)
        if hops_seen:
            print(f"    {dst_ip}: " + " | ".join(sorted(hops_seen)))


def install_arp_spanning_tree() -> None:
    """
    Etapa [5/6]: controle de flooding via Spanning Tree Mínimo + bloqueio de portas não-MST.

    Dois tipos de flow são instalados:

    1. ARP MST forward (priority=5000, por porta de entrada):
       Pacotes ARP entram por in_port → saem apenas pelas portas MST + hosts.
       Substitui o flood irrestrito do l2switch para tráfego ARP.

    2. Flood block DROP (priority=3, em portas não-MST inter-switch):
       Qualquer pacote vindo de uma porta não-MST que não tenha match de
       prioridade maior (LLDP=1000, ARP-MST=5000, IPv4=60000) é descartado.
       Isso corta os loops causados pelos floods reativos do l2switch (p=2).
       Links redundantes continuam disponíveis para unicast proativo (p=60000).

    Na mudança de topologia (cabo adicionado/removido):
       - Flows DROP antigos são removidos dos switches
       - MST é recalculado com a nova topologia
       - Novos flows DROP são instalados nos novos links não-MST

    Nota: o MST usa pesos fixos unitários (não os custos de tráfego) para
    evitar que congestionamento temporário cause mudanças no MST, o que
    geraria floods durante a reconvergência.
    """
    print("--- [5/6] ARP Spanning-Tree + Anti-Storm ---")
    with state.lock:
        g              = state.graph.copy()
        edge_ports_map = {k: set(v) for k, v in state.edge_ports.items()}
        topo_changed   = state.topo_changed
        old_blocks     = set(state._flood_blocks)

    if len(g.nodes) == 0 or len(g.edges) == 0:
        print("  Aguardando topologia com enlaces...")
        return

    # ── Calcula MST por componente conexo ─────────────────────────────────
    # USA PESOS FIXOS (atributo "topo_weight") para evitar oscilação do MST
    # quando link_costs mudam com tráfego. O MST só deve mudar se a topologia
    # física mudar (enlaces adicionados/removidos), não por congestionamento.
    mst_edges: set = set()
    for component in nx.connected_components(g):
        subg   = g.subgraph(component).copy()
        topo_g = nx.Graph()
        for u, v, attrs in subg.edges(data=True):
            topo_g.add_edge(u, v, weight=1)
        for u, v in nx.minimum_spanning_tree(topo_g, weight="weight").edges():
            mst_edges.add(link_key(u, v))

    # ── Identifica portas não-MST inter-switch por switch ─────────────────
    non_mst_iface: dict[str, set[str]] = defaultdict(set)
    mst_iface:     dict[str, set[str]] = defaultdict(set)
    for sw in g.nodes:
        for nb in g.neighbors(sw):
            port  = out_port(g.get_edge_data(sw, nb), sw)
            iface = port_to_iface(port)
            if link_key(sw, nb) in mst_edges:
                mst_iface[sw].add(iface)
            else:
                non_mst_iface[sw].add(iface)

    # ── Topologia mudou: remove blocos antigos em paralelo ────────────────
    if topo_changed:
        if old_blocks:
            delete_flows_parallel(list(old_blocks))
        with state.lock:
            state._flood_blocks.clear()
            state.topo_changed = False
        if old_blocks:
            print(f"  🧹 {len(old_blocks)} blocos antigos removidos (topologia mudou)")

    # ── Acumula tasks de instalação: DROP não-MST + ARP MST ──────────────
    install_tasks: list[tuple] = []
    new_blocks: set = set()

    # Flows DROP nas portas não-MST
    for sw, ifaces in non_mst_iface.items():
        for iface in ifaces:
            fid  = f"FLOOD_BLOCK_{iface}"
            fstr = flow_flood_block(iface)
            install_tasks.append((sw, fid, fstr, True))
            new_blocks.add((sw, fid))

    # Flows ARP MST (encaminhamento loop-free)
    for sw in g.nodes:
        host_ports  = set(edge_ports_map.get(sw, set()))
        safe_ifaces = mst_iface[sw] | {port_to_iface(p) for p in host_ports}
        all_ifaces  = safe_ifaces | non_mst_iface[sw]
        for in_iface in all_ifaces:
            out_ifaces = sorted(safe_ifaces - {in_iface})
            fid  = f"ARP_MST_{in_iface}"
            fstr = flow_arp_mst(in_iface, out_ifaces)
            install_tasks.append((sw, fid, fstr, True))

        # Flow ARP catch-all (priority=4999): captura ARP em qualquer porta
        # ainda não coberta por ARP_MST específico (ex: porta de VPC nova,
        # antes de aparecer em edge_ports). Manda ao CONTROLLER para que o
        # ODL registre o host. Prioridade abaixo do ARP_MST (5000) e acima
        # do flood reativo do l2switch (2) e do DROP de porta não-MST (3).
        # Sem este flow, uma VPC em porta desconhecida só seria vista pelo
        # TABLE-MISS (p=0), que em alguns cenários é suprimido pelo l2switch.
        install_tasks.append((
            sw, "ARP_CATCHALL",
            f"priority=4999,dl_type=0x0806,actions=controller:65535",
            True
        ))

    # Instala tudo em paralelo de uma vez
    arp_ok, arp_fail = install_flows_parallel(install_tasks)
    blocks_inst = len(new_blocks)

    with state.lock:
        state._flood_blocks.update(new_blocks)

    # ── Resumo detalhado ─────────────────────────────────────────────────
    mst_link_count = len(mst_edges)
    non_mst_count  = sum(len(v) for v in non_mst_iface.values())
    blocked_label  = (f" | 🛡️  {blocks_inst} portas redundantes bloqueadas"
                      if blocks_inst else "")
    print(f"  MST: {mst_link_count} enlaces ativos | "
          f"{non_mst_count} portas redundantes{blocked_label}")
    for sw in sorted(g.nodes):
        mst_s  = sorted(mst_iface[sw])
        nmst_s = sorted(non_mst_iface[sw])
        if mst_s or nmst_s:
            mst_str  = ",".join(mst_s)  if mst_s  else "—"
            nmst_str = ",".join(nmst_s) if nmst_s else "—"
            sw_s = sw.replace("openflow:", "sw")
            print(f"    {sw_s}  MST:{mst_str}  bloqueadas:{nmst_str}")
    arp_count   = sum(1 for _, fid, _, _ in install_tasks if fid.startswith("ARP_MST"))
    block_count = sum(1 for _, fid, _, _ in install_tasks if fid.startswith("FLOOD_BLOCK"))
    if arp_fail or blocks_inst:
        print(f"  ARP flows: {arp_count} instalados | "
              f"Blocos DROP: {block_count} | "
              f"Falhas: {arp_fail}")


def check_and_reroute() -> None:
    """Etapa [6/6]: desvio de links congestionados em priority=62000 — paralelo."""
    if not ENABLE_REROUTING:
        return

    with state.lock:
        g         = state.graph.copy()
        link_load = dict(state.link_load)
        ip_to_mac = dict(state.ip_to_mac)
        hosts     = dict(state.hosts_by_mac)

    congested = {lk for lk, bps in link_load.items()
                 if bps / MAX_LINK_CAPACITY > REROUTE_THRESH}
    if not congested:
        return

    print(f"  [REROUTE] {len(congested)} link(s) congestionado(s)")

    tasks: list[tuple] = []
    for dst_ip, dst_mac in ip_to_mac.items():
        info       = hosts.get(dst_mac, {})
        dst_switch = info.get("switch")
        dst_port   = info.get("port")
        if dst_switch not in g.nodes or not dst_port:
            continue

        g_temp = g.copy()
        for u, v in congested:
            if g_temp.has_edge(u, v):
                g_temp[u][v]["weight"] = 9999

        flow_id = f"LB_{dst_ip.replace('.', '_')}"
        try:
            _, paths = nx.single_source_dijkstra(g_temp, dst_switch, weight="weight")
        except Exception:
            continue

        for src_switch, path_rev in paths.items():
            dst_iface = port_to_iface(dst_port)
            if src_switch == dst_switch:
                fstr = flow_ipv4_forward(dst_ip, dst_iface, 62000, idle=15)
                tasks.append((dst_switch, flow_id, fstr, True))
                continue
            path = list(reversed(path_rev))
            for i in range(len(path) - 1):
                curr, nxt = path[i], path[i + 1]
                edge = g_temp.get_edge_data(curr, nxt)
                if not edge:
                    continue
                out_iface = port_to_iface(out_port(edge, curr))
                fstr = flow_ipv4_forward(dst_ip, out_iface, 62000, idle=15)
                tasks.append((curr, flow_id, fstr, True))

    if tasks:
        install_flows_parallel(tasks)
