"""
Configurações globais do SDN Orchestrator.

Centraliza todas as constantes de ambiente, thresholds e URLs de forma que
qualquer mudança de configuração afete apenas este módulo (SRP).
"""

from requests.auth import HTTPBasicAuth

ENABLE_REROUTING = True

# ── OpenDaylight ────────────────────────────────────────────────────────────
ODL_IP      = "172.16.1.1"
ODL_PORT    = "8181"
ODL_USER    = "admin"
ODL_PASS    = "admin"
TOPOLOGY_ID = "flow:1"


# ── Thresholds de capacidade e congestionamento ─────────────────────────────
# sdn_orchestrator/config.py
#MAX_LINK_CAPACITY = 100_000_000 # 100 Mbps
POLLING_INTERVAL  = 5
WARN_THRESH       = 0.50
MAX_LINK_CAPACITY = 20000000  # 20 Mbps em bps
REROUTE_THRESH    = 0.65         # 65% = 13 Mbps em links de 20 Mbps — garante rerouting com tráfego de fundo de 7 Mbps
CONGESTED_THRESH  = 0.90         # Autoajustado pelo runner
# ── TTL e probing de hosts/switches ────────────────────────────────────────
HOST_TTL_CYCLES   = 3   # ciclos sem aparecer no ODL antes de remover host
SW_TTL_CYCLES     = 3   # ciclos sem aparecer no ODL antes de remover switch
HOST_PROBE_CYCLES = 6   # a cada N ciclos, envia ARP probe para detectar host offline
HOST_PROBE_MISS   = 3   # ciclos de probe sem resposta antes de remover host

# ── Switch do servidor FL (tap1) ────────────────────────────────────────────
# ODL node-id do switch core ao qual o tap1 (plano de dados FL) está conectado.
# Usado em /metrics/hosts para calcular o bottleneck no caminho cliente→servidor.
# Formato: "openflow:X" onde X é o DPID do switch no OpenDaylight.
# Verifique com: curl -s http://localhost:8000/state | python3 -c "import sys,json; print(json.load(sys.stdin)['switches'])"
FL_SERVER_SWITCH = "openflow:1"

# ── Docker ─────────────────────────────────────────────────────────────────
# Prefixo dos containers GNS3 — ajuste se o nome for diferente no seu projeto
DOCKER_NAME_PREFIX = "GNS3.OVS-"

# ── Autenticação e headers HTTP ────────────────────────────────────────────
AUTH         = HTTPBasicAuth(ODL_USER, ODL_PASS)
HEADERS_JSON = {"Content-Type": "application/json", "Accept": "application/json"}

# ── URLs da API REST do ODL ────────────────────────────────────────────────
BASE         = f"http://{ODL_IP}:{ODL_PORT}"
URL_TOPO     = f"{BASE}/rests/data/network-topology:network-topology/topology={TOPOLOGY_ID}"
URL_NODES    = f"{BASE}/rests/data/opendaylight-inventory:nodes"
URL_NODES_OP = f"{BASE}/rests/data/opendaylight-inventory:nodes?content=nonconfig"
URL_TOPO_OP  = f"{BASE}/rests/data/network-topology:network-topology/topology={TOPOLOGY_ID}?content=nonconfig"
