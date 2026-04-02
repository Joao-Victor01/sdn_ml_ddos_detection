"""
Entidade central de estado da rede (NetworkState).

NetworkState encapsula todo o estado mutável compartilhado entre as
camadas de aplicação e apresentação. O acesso concorrente é protegido
por um único threading.Lock.

O singleton `state` e o contador `CYCLE_COUNT` são definidos aqui para
que qualquer módulo possa importá-los de um único ponto (DIP).
"""

import threading
import networkx as nx


class NetworkState:
    """
    Entidade de domínio que centraliza o estado observável da rede SDN.

    Atributos principais:
      graph            — grafo de switches e enlaces (NetworkX)
      hosts_by_mac     — hosts conhecidos: mac → {mac, ips, switch, port}
      ip_to_mac        — índice reverso: ip → mac
      edge_ports       — portas de borda por switch: sw_id → set(port_nums)
      sw_to_container  — mapeamento openflow:X → container Docker
      active_flows     — flows instalados: (sw_id, flow_id) → ovs_flow_str
      blocked_switches — lista de switches isolados pela API
      blocked_ips      — lista de IPs bloqueados pela API
      pending_unblocks — IPs em processo de desbloqueio (evita race condition)
      port_stats       — últimas contagens de bytes por porta ODL
      link_load        — carga atual por enlace: (u,v) → bps
      link_costs       — custo Dijkstra por enlace: (u,v) → custo
      _valid_switches  — switches com tabelas de flow confirmadas pelo ODL
      _guard_done      — switches com flows base (TABLE-MISS/LLDP/BDDP) instalados
      _prev_edges      — conjunto de arestas do ciclo anterior (detecção de mudança)
      topo_changed     — flag: topologia mudou desde o último ciclo ARP-MST
      _flood_blocks    — flows DROP anti-storm ativos: set de (sw_id, flow_id)
      host_missing_cycles — ciclos consecutivos sem ver o host no ODL
      _sw_missing_cycles  — ciclos consecutivos sem ver o switch no ODL
      _host_probe_sent    — ciclo em que o último ARP probe foi enviado por MAC
    """

    def __init__(self):
        self.lock             = threading.Lock()
        self.graph            = nx.Graph()
        self.blocked_switches = []
        self.blocked_ips      = []
        self.pending_unblocks : set = set()
        self.port_stats       = {}
        self.link_load        = {}
        self.link_costs       = {}
        self._valid_switches  = set()
        self._guard_done      = set()
        self.hosts_by_mac     = {}
        self.ip_to_mac        = {}
        self.edge_ports       = {}
        self.sw_to_container  = {}
        self.active_flows     = {}
        self._prev_edges      = frozenset()
        self.topo_changed     = False
        self._flood_blocks    = set()
        self.host_missing_cycles: dict[str, int] = {}
        self._sw_missing_cycles: dict[str, int]  = {}
        self._host_probe_sent: dict[str, int]    = {}


# ── Singletons globais ──────────────────────────────────────────────────────
state: NetworkState = NetworkState()

# Contador de ciclos do loop de controle.
# Centralizado aqui para que topology.py, hosts.py e main.py
# possam ler/incrementar a partir de um único módulo.
CYCLE_COUNT: int = 0
