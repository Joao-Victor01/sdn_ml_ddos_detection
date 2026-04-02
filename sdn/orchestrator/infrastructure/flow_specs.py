"""
Construtores de strings de flow no formato ovs-ofctl.

Funções puras (sem efeitos colaterais) que encapsulam o conhecimento das
regras de flow OVS. Seguem o SRP: existem apenas para gerar strings de
flow corretamente formatadas, sem nenhuma I/O ou estado compartilhado.

Tabela de prioridades:
  65500  DROP src_ip (bloqueio manual de IP)
  62000  IPv4 reroute anti-congestionamento (idle=15s)
  60000  IPv4 proativo Dijkstra (PERMANENTE)
   5000  ARP loop-free via MST
   4999  ARP catch-all → controller (hosts em portas desconhecidas)
   1000  LLDP/BDDP → controller
      3  DROP flooding em portas não-MST
      2  floods reativos l2switch (nunca alcançados em portas não-MST)
      0  table-miss → controller
"""


def flow_table_miss() -> str:
    return "priority=0,actions=controller:65535"


def flow_lldp() -> str:
    return "priority=1000,dl_type=0x88cc,actions=controller:65535"


def flow_bddp() -> str:
    return "priority=1000,dl_type=0x8942,actions=controller:65535"


def flow_ipv4_forward(dst_ip: str, out_iface: str,
                      priority: int = 60000, idle: int = 0) -> str:
    """
    Flow de encaminhamento IPv4 unicast.

    idle=0  → flow permanente (padrão para rotas pri=60000).
              Não expira — o orquestrador remove/substitui explicitamente
              quando a topologia muda. Sem idle_timeout, o OVS não remove
              o flow por inatividade, garantindo que QUALQUER pacote IP
              destinado ao host sempre encontre a rota SDN instalada.
    idle>0  → somente para reroute (pri=62000, idle=15): desvio temporário
              que pode expirar sozinho quando o congestionamento passa.
    """
    idle_part = f"idle_timeout={idle}," if idle > 0 else ""
    return (f"priority={priority},{idle_part}"
            f"ip,nw_dst={dst_ip},"
            f"actions=output:{out_iface}")


def flow_arp_mst(in_iface: str, out_ifaces: list[str],
                 priority: int = 5000) -> str:
    """Flow ARP loop-free: entra por in_iface, sai pelas out_ifaces + CONTROLLER."""
    out_actions = ",".join(f"output:{i}" for i in sorted(out_ifaces))
    all_actions = f"{out_actions},controller:65535" if out_actions else "controller:65535"
    return (f"priority={priority},"
            f"dl_type=0x0806,in_port={in_iface},"
            f"actions={all_actions}")


def flow_ip_drop(src_ip: str) -> str:
    """Flow de bloqueio por IP de origem (prioridade máxima do orquestrador)."""
    return f"priority=65500,ip,nw_src={src_ip},actions=drop"


def flow_flood_block(in_iface: str) -> str:
    """
    Bloqueia flooding vindo de uma porta não-MST em priority=3.
    Fica acima dos floods reativos do l2switch (priority=2) mas abaixo
    de tudo instalado pelo orquestrador, LLDP e IPv4 proativo.
    Pacotes que chegam desta porta e NÃO têm match de prioridade maior
    (unknown unicast, broadcasts não-ARP) são descartados → sem loops.
    Unicast explícito (60000) e ARP-MST (5000) continuam funcionando.
    """
    return f"priority=3,in_port={in_iface},actions=drop"
