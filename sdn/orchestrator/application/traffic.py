"""
Caso de uso: Monitoramento de Tráfego — Etapa [3/6].

Responsabilidade única: ler estatísticas de portas do ODL, calcular
a utilização de cada enlace em bps e atualizar os custos Dijkstra no
grafo de estado para influenciar o roteamento proativo.

FIX v14: usa tempo REAL decorrido entre medições (não POLLING_INTERVAL fixo).

O bug anterior: bw_bps = delta / POLLING_INTERVAL (5s)
Com ciclos de 2-3 minutos, delta acumula 160s de tráfego mas é dividido
por 5 → leituras 32x maiores (500% de utilização = 15% real).

Agora: bw_bps = delta / elapsed_real_seconds
Se elapsed < 1s (primeira medição), ignora o delta (sem baseline ainda).
"""

import time
import requests

from orchestrator.config import (
    AUTH, HEADERS_JSON, URL_NODES_OP, URL_NODES,
    MAX_LINK_CAPACITY, CONGESTED_THRESH, WARN_THRESH,
)
from orchestrator.domain.state import state
from orchestrator.application.topology import link_key

# Rastreia o timestamp da última medição para calcular o elapsed real
_last_traffic_ts: float = 0.0


def monitor_traffic_load() -> None:
    """
    Etapa [3/6]: monitora utilização dos enlaces e ajusta custos Dijkstra.

    - Lê contadores de bytes das portas via ODL REST
    - Calcula bandwidth em bps usando o tempo real decorrido
    - Atualiza state.link_load e state.link_costs
    - Atualiza os pesos do grafo para influenciar roteamento no próximo ciclo
    """
    global _last_traffic_ts
    print("--- [3/6] Tráfego ---")
    now     = time.monotonic()
    elapsed = now - _last_traffic_ts if _last_traffic_ts > 0 else 0
    _last_traffic_ts = now

    try:
        resp = requests.get(URL_NODES_OP, auth=AUTH,
                            headers=HEADERS_JSON, timeout=4)
        if resp.status_code != 200:
            resp = requests.get(URL_NODES, auth=AUTH,
                                headers=HEADERS_JSON, timeout=4)
        if resp.status_code != 200:
            return

        nodes = (resp.json()
                 .get("opendaylight-inventory:nodes", {})
                 .get("node", []))

        with state.lock:
            graph_snap = state.graph.copy()

        monitored = congested = 0
        new_costs  = {}

        for node in nodes:
            if "openflow" not in node.get("id", ""):
                continue
            for nc in node.get("node-connector", []):
                stats = nc.get(
                    "opendaylight-port-statistics:"
                    "flow-capable-node-connector-statistics", {})
                if not stats:
                    continue
                pid = nc.get("id", "")
                if "LOCAL" in pid:
                    continue
                try:
                    total = (int(stats.get("bytes", {}).get("transmitted", 0)) +
                             int(stats.get("bytes", {}).get("received", 0)))
                except Exception:
                    continue

                # Usa tempo REAL decorrido — evita leituras infladas em ciclos lentos
                if elapsed < 1.0 or pid not in state.port_stats:
                    # Primeira medição ou elapsed muito curto: apenas guarda baseline
                    state.port_stats[pid] = total
                    continue

                delta  = max(0, total - state.port_stats[pid])
                bw_bps = (delta * 8) / elapsed   # ← elapsed real, não POLLING_INTERVAL
                state.port_stats[pid] = total
                monitored += 1

                for u, v, attrs in graph_snap.edges(data=True):
                    if (attrs.get("src_port") == pid or
                            attrs.get("dst_port") == pid):
                        key   = link_key(u, v)
                        ratio = bw_bps / MAX_LINK_CAPACITY
                        cost  = (100 if ratio > CONGESTED_THRESH else
                                 10  if ratio > WARN_THRESH else
                                 3   if ratio > 0.20 else 1)
                        new_costs[key] = cost
                        with state.lock:
                            state.link_load[key] = bw_bps
                        if cost >= 10:
                            congested += 1
                            print(f"  ⚠️  {u}↔{v}: {ratio:.0%} → custo={cost}")
                        break

        with state.lock:
            state.link_costs.update(new_costs)
            for (u, v), cost in new_costs.items():
                if state.graph.has_edge(u, v):
                    state.graph[u][v]["weight"] = cost

        # Mostra barra de utilização por enlace
        with state.lock:
            link_load_snap = dict(state.link_load)
            links_snap     = list(state.graph.edges(data=True))
        shown: set = set()
        for u, v, attrs in links_snap:
            key = link_key(u, v)
            if key in shown:
                continue
            shown.add(key)
            bps = link_load_snap.get(key, 0)
            if bps == 0:
                continue
            pct  = bps / MAX_LINK_CAPACITY * 100
            bar  = "█" * int(min(pct, 100) / 5) + "░" * (20 - int(min(pct, 100) / 5))
            warn = " ⚠️" if pct > 75 else ""
            sa   = u.replace("openflow:", "sw")
            sb   = v.replace("openflow:", "sw")
            print(f"    {sa}↔{sb}  [{bar}] {bps/1000:6.1f} kbps  {pct:5.1f}%{warn}")
        print(f"  OK: {monitored} portas monitoradas | {congested} congestionadas")
    except Exception:
        pass
