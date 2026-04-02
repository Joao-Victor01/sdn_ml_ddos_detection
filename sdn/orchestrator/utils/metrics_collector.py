"""
Coleta métricas do SDN Orchestrator a cada ciclo e persiste em CSV.

Interface principal:
    collector = MetricsCollector()
    collector.collect(cycle_num, cycle_duration_sec)

Não modifica nenhum comportamento existente — apenas observa o estado
compartilhado (state) de forma thread-safe e escreve no CSV.

Arquivo gerado: sdn_metrics.csv (diretório de trabalho atual)
Colunas:
    timestamp           — Unix timestamp do fim do ciclo
    cycle               — Número do ciclo
    elapsed_sec         — Segundos desde o início do orquestrador
    cycle_duration_sec  — Duração deste ciclo (s)
    n_switches          — Switches ativos na topologia
    n_hosts             — Hosts conhecidos
    n_flows             — Flows instalados (active_flows)
    n_reroute_flows     — Flows de reroute ativos (LB_*, priority=62000)
    max_link_load_bps   — Maior carga de enlace observada (bps)
    avg_link_load_bps   — Carga média por enlace (bps)
    congested_links     — Número de enlaces acima de REROUTE_THRESH (0.75)
    warn_links          — Número de enlaces acima de WARN_THRESH (0.50)
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional

from orchestrator.config import MAX_LINK_CAPACITY, REROUTE_THRESH, WARN_THRESH
from orchestrator.domain.state import state

# Instância ativa — definida em main.py e acessada pela API REST
_instance: Optional["MetricsCollector"] = None

_FIELDNAMES = [
    "timestamp",
    "cycle",
    "elapsed_sec",
    "cycle_duration_sec",
    "n_switches",
    "n_hosts",
    "n_flows",
    "n_reroute_flows",
    "max_link_load_bps",
    "avg_link_load_bps",
    "congested_links",
    "warn_links",
]


class MetricsCollector:
    """
    Coleta e persiste métricas do orquestrador SDN a cada ciclo de controle.

    Thread-safe: lê sob state.lock; escrita no CSV é single-threaded
    pois é chamado exclusivamente pelo loop de controle (thread daemon).
    """

    def __init__(self, output_path: str = "sdn_metrics.csv") -> None:
        stem, ext = os.path.splitext(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path        = f"{stem}_{timestamp}{ext}"
        self._start_time: Optional[float] = None
        self._file        = None
        self._writer      = None
        self._initialized = False

        # Estado da sessão FL ativa
        self._fl_file:   Optional[object] = None
        self._fl_writer: Optional[csv.DictWriter] = None
        self._fl_round:  Optional[int]   = None
        self._fl_start:  Optional[float] = None

    def _init(self) -> None:
        """Abre um CSV novo e escreve o cabeçalho (lazy — apenas no primeiro ciclo)."""
        self._file   = open(self._path, "w", newline="", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDNAMES)
        self._writer.writeheader()
        self._start_time  = time.monotonic()
        self._initialized = True
        print(f"[metrics] Coletando em {os.path.abspath(self._path)}")

    def start_fl_session(self, round_num: int) -> str:
        """
        Abre um CSV dedicado para o round de FL informado.
        Chamado via POST /fl/training/start pelo servidor FL.
        Se já houver uma sessão aberta, fecha antes de abrir a nova.
        """
        if self._fl_file:
            self._fl_file.close()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"fl_metrics_round{round_num}_{timestamp}.csv"
        self._fl_file   = open(path, "w", newline="", buffering=1)
        self._fl_writer = csv.DictWriter(self._fl_file, fieldnames=_FIELDNAMES)
        self._fl_writer.writeheader()
        self._fl_round = round_num
        self._fl_start = time.monotonic()
        print(f"[metrics] FL round {round_num} iniciado → {os.path.abspath(path)}")
        return os.path.abspath(path)

    def stop_fl_session(self) -> dict:
        """
        Fecha o CSV do round FL atual e devolve um resumo.
        Chamado via POST /fl/training/stop pelo servidor FL.
        """
        if not self._fl_file:
            return {"status": "no_active_session"}
        self._fl_file.close()
        self._fl_file   = None
        self._fl_writer = None
        elapsed = round(time.monotonic() - self._fl_start, 2) if self._fl_start else 0.0
        rnd = self._fl_round
        self._fl_round = None
        self._fl_start = None
        print(f"[metrics] FL round {rnd} finalizado ({elapsed}s)")
        return {"status": "ok", "round": rnd, "duration_sec": elapsed}

    def collect(self, cycle: int, cycle_duration_sec: float) -> None:
        """
        Registra uma linha de métricas para o ciclo atual.

        Args:
            cycle               — Número do ciclo (state_module.CYCLE_COUNT)
            cycle_duration_sec  — Duração do ciclo em segundos
        """
        if not self._initialized:
            self._init()

        now     = time.time()
        elapsed = time.monotonic() - self._start_time

        with state.lock:
            n_switches = state.graph.number_of_nodes()
            n_hosts    = len(state.hosts_by_mac)
            n_flows    = len(state.active_flows)
            n_reroute  = sum(
                1 for (_, fid) in state.active_flows
                if fid.startswith("LB_")
            )
            loads = list(state.link_load.values())

        if loads:
            max_load  = max(loads)
            avg_load  = sum(loads) / len(loads)
            congested = sum(
                1 for bps in loads if bps / MAX_LINK_CAPACITY > REROUTE_THRESH
            )
            warn = sum(
                1 for bps in loads if bps / MAX_LINK_CAPACITY > WARN_THRESH
            )
        else:
            max_load = avg_load = 0.0
            congested = warn = 0

        row = {
            "timestamp":          round(now, 3),
            "cycle":              cycle,
            "elapsed_sec":        round(elapsed, 2),
            "cycle_duration_sec": round(cycle_duration_sec, 3),
            "n_switches":         n_switches,
            "n_hosts":            n_hosts,
            "n_flows":            n_flows,
            "n_reroute_flows":    n_reroute,
            "max_link_load_bps":  round(max_load, 0),
            "avg_link_load_bps":  round(avg_load, 0),
            "congested_links":    congested,
            "warn_links":         warn,
        }
        self._writer.writerow(row)
        # Se há uma sessão FL ativa, espelha a linha no CSV do round
        if self._fl_writer:
            self._fl_writer.writerow(row)

    def close(self) -> None:
        """Fecha o arquivo CSV (shutdown gracioso, opcional)."""
        if self._file:
            self._file.close()
            self._file = None
