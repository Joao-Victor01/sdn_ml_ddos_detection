"""
Loop de controle do SDN Orchestrator e ponto de entrada do pacote.

O loop executa as 6 etapas do ciclo de controle a cada POLLING_INTERVAL
segundos, orquestrando a leitura do ODL e a escrita nos switches OVS.

Etapas do ciclo:
  [1/6] fetch_topology()            — switches e enlaces (ODL leitura)
  [2/6] fetch_hosts()               — hosts via l2switch (ODL leitura)
  [2b]  probe_hosts()               — ARP probe para detectar hosts offline
  [3/6] monitor_traffic_load()      — utilização (ODL leitura)
  [4/6] install_ipv4_routes()       — Dijkstra → docker exec
  [5/6] install_arp_spanning_tree() — ARP loop-free → docker exec
  [6/6] check_and_reroute()         — desvio de congestionamento

Watchdogs periódicos (executados em ciclos específicos):
  A cada  3 ciclos — remove flows rogue do arphandler (priority=65000,arp)
  A cada  5 ciclos — re-descobre containers Docker
  A cada 10 ciclos — atualiza conjunto de switches válidos no ODL
"""

import threading
import time

import requests
import uvicorn

from orchestrator.utils import metrics_collector as _metrics_mod
from orchestrator.utils.metrics_collector import MetricsCollector

import orchestrator.domain.state as state_module
from orchestrator.config import (
    POLLING_INTERVAL, AUTH, HEADERS_JSON, URL_TOPO,
)
from orchestrator.infrastructure.docker_adapter import discover_containers
from orchestrator.application.topology import (
    fetch_topology, refresh_valid_switches, remove_rogue_arp_flows,
)
from orchestrator.application.hosts import fetch_hosts, probe_hosts
from orchestrator.application.traffic import monitor_traffic_load
from orchestrator.application.routing import (
    install_ipv4_routes, install_arp_spanning_tree, check_and_reroute,
)
from orchestrator.presentation.api import app


def control_loop() -> None:
    """Loop de controle principal — executa em thread daemon."""
    print("\n🔌 Testando conexão com ODL...")
    try:
        r = requests.get(URL_TOPO, auth=AUTH, headers=HEADERS_JSON, timeout=5)
        print(f"  {'✅' if r.status_code == 200 else '❌'} ODL HTTP {r.status_code}")
    except Exception as e:
        print(f"  ❌ Sem resposta: {e}")

    print("\n🐳 Descobrindo containers Docker...")
    discover_containers()
    refresh_valid_switches()

    _metrics = MetricsCollector()
    _metrics_mod._instance = _metrics

    while True:
        state_module.CYCLE_COUNT += 1
        print("\n" + "=" * 60)
        print(f"  CICLO #{state_module.CYCLE_COUNT}  —  {time.strftime('%H:%M:%S')}")
        print("=" * 60)

        # Watchdog anti-rogue: remove flows priority=65000,arp do arphandler
        if state_module.CYCLE_COUNT % 3 == 0:
            remove_rogue_arp_flows()

        if state_module.CYCLE_COUNT % 10 == 0:
            refresh_valid_switches()
        if state_module.CYCLE_COUNT % 5 == 0:
            # Re-descobre containers para o caso de reiniciar um switch
            discover_containers()

        _cycle_start = time.monotonic()

        fetch_topology()            # [1/6] switches e enlaces (ODL leitura)
        fetch_hosts()               # [2/6] hosts via l2switch (ODL leitura)
        probe_hosts()               # [2b]  ARP probe para detectar hosts offline
        monitor_traffic_load()      # [3/6] utilização (ODL leitura)
        install_ipv4_routes()       # [4/6] Dijkstra → docker exec
        install_arp_spanning_tree() # [5/6] ARP loop-free → docker exec
        check_and_reroute()         # [6/6] desvio de congestionamento

        _metrics.collect(state_module.CYCLE_COUNT, time.monotonic() - _cycle_start)

        time.sleep(POLLING_INTERVAL)


def main() -> None:
    """Inicia o loop de controle em thread daemon e sobe o servidor FastAPI."""
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
