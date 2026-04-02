"""
Testes unitários para _get_path_bottleneck e get_host_metrics.

Valida que:
  - _get_path_bottleneck retorna o maior load no caminho src→dst
  - Identifica corretamente o gargalo nos enlaces agregador↔core
  - Fallback funciona quando não há caminho
  - get_host_metrics calcula bandwidth_mbps e latency_ms corretamente
"""

import threading
import types
import unittest
from unittest.mock import patch, MagicMock

import networkx as nx


# ---------------------------------------------------------------------------
# Importação de _get_path_bottleneck sem efeitos colaterais do FastAPI
# ---------------------------------------------------------------------------
# A função é pura (recebe graph e link_load como args, sem estado global).
# Importamos o módulo com o app FastAPI desabilitado para evitar subir o server.

import sys

# Stubs mínimos antes de qualquer import do pacote orchestrator/presentation
_stub_state_obj = types.SimpleNamespace(
    lock=threading.Lock(),
    graph=nx.Graph(),
    link_load={},
    link_costs={},
    hosts_by_mac={},
    ip_to_mac={},
    sw_to_container={},
    blocked_switches=[],
    blocked_ips=[],
    pending_unblocks=set(),
    active_flows={},
)

import orchestrator.domain.state as _real_state  # noqa: E402
_real_state.state = _stub_state_obj
_real_state.CYCLE_COUNT = 0

# Stub de docker_adapter e ovs_adapter para evitar chamadas reais
import orchestrator.infrastructure.docker_adapter as _docker  # noqa: E402
_docker.container_for = lambda sw_id: None

import orchestrator.infrastructure.ovs_adapter as _ovs  # noqa: E402
if not hasattr(_ovs, "FLOW_EXECUTOR"):
    from concurrent.futures import ThreadPoolExecutor
    _ovs.FLOW_EXECUTOR = ThreadPoolExecutor(max_workers=1)
if not hasattr(_ovs, "delete_ip_block_direct"):
    _ovs.delete_ip_block_direct = lambda *a, **kw: None

import orchestrator.utils.metrics_collector as _mc  # noqa: E402
if not hasattr(_mc, "_instance"):
    _mc._instance = None

# Agora é seguro importar a função de api.py
from orchestrator.presentation.api import _get_path_bottleneck  # noqa: E402
from orchestrator.presentation.api import get_host_metrics       # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph(*edges):
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def _link_load(*items):
    """Chaves sempre ordenadas, como o orquestrador popula state.link_load."""
    return {tuple(sorted([u, v])): bps for u, v, bps in items}


# ---------------------------------------------------------------------------
# Testes de _get_path_bottleneck
# ---------------------------------------------------------------------------

class TestGetPathBottleneck(unittest.TestCase):

    def test_linear_path_returns_max_load(self):
        """
        sw6 ─── sw3 ─── sw1
        Cargas: sw6↔sw3=5Mbps, sw3↔sw1=18Mbps → bottleneck=18Mbps
        """
        g = _build_graph(("sw6", "sw3"), ("sw3", "sw1"))
        load = _link_load(("sw6", "sw3", 5_000_000), ("sw3", "sw1", 18_000_000))

        self.assertEqual(_get_path_bottleneck("sw6", "sw1", load, g), 18_000_000)

    def test_bottleneck_is_first_hop(self):
        """
        sw6 ─── sw3 ─── sw1
        Cargas: sw6↔sw3=20Mbps, sw3↔sw1=5Mbps → bottleneck=20Mbps
        """
        g = _build_graph(("sw6", "sw3"), ("sw3", "sw1"))
        load = _link_load(("sw6", "sw3", 20_000_000), ("sw3", "sw1", 5_000_000))

        self.assertEqual(_get_path_bottleneck("sw6", "sw1", load, g), 20_000_000)

    def test_hierarchical_topology_bottleneck_at_agg_core(self):
        """
        Topologia hierárquica (3 camadas, similar ao lab):
          Borda sw6/sw7/sw8 → Agregador sw3 → Core sw1

        O enlace Ag↔Core (sw3↔sw1) está congestionado (19.5Mbps),
        os enlaces de borda têm apenas 2Mbps cada.

        A implementação CORRETA (com _get_path_bottleneck) deve retornar 19.5Mbps.
        A implementação ANTIGA (max load de qualquer enlace do switch de borda)
        retornaria apenas 2Mbps — este teste garante que a regressão não volta.
        """
        g = _build_graph(
            ("sw6", "sw3"), ("sw7", "sw3"), ("sw8", "sw3"),
            ("sw3", "sw1"), ("sw5", "sw1"),
        )
        load = _link_load(
            ("sw6", "sw3",  2_000_000),
            ("sw7", "sw3",  2_000_000),
            ("sw8", "sw3",  2_000_000),
            ("sw3", "sw1", 19_500_000),   # ← gargalo real
            ("sw5", "sw1",  1_000_000),
        )

        result = _get_path_bottleneck("sw6", "sw1", load, g)

        self.assertEqual(result, 19_500_000,
            "Bottleneck deve ser sw3↔sw1 (19.5Mbps), não o enlace de borda (2Mbps)")

    def test_zero_load_links_in_path(self):
        """Links sem tráfego contribuem com 0, não quebram o cálculo."""
        g = _build_graph(("sw9", "sw4"), ("sw4", "sw2"), ("sw2", "sw1"))
        load = _link_load(
            ("sw9", "sw4", 0),
            ("sw4", "sw2", 15_000_000),
            ("sw2", "sw1", 0),
        )
        self.assertEqual(_get_path_bottleneck("sw9", "sw1", load, g), 15_000_000)

    def test_missing_link_in_load_dict_treated_as_zero(self):
        """Enlace no grafo mas ausente em link_load → 0bps (sem exceção)."""
        g = _build_graph(("sw6", "sw3"), ("sw3", "sw1"))
        load = _link_load(("sw6", "sw3", 10_000_000))   # sw3↔sw1 ausente

        self.assertEqual(_get_path_bottleneck("sw6", "sw1", load, g), 10_000_000)

    def test_src_equals_dst_returns_zero(self):
        """Origem == destino: caminho tem 1 nó, sem enlaces → 0."""
        g = _build_graph(("sw1", "sw3"))
        load = _link_load(("sw1", "sw3", 10_000_000))

        self.assertEqual(_get_path_bottleneck("sw1", "sw1", load, g), 0.0)

    def test_no_path_returns_global_max(self):
        """Sem caminho src→dst: fallback retorna máximo global de link_load."""
        g = _build_graph(("sw6", "sw3"))
        g.add_node("sw1")   # sw1 desconectado
        load = _link_load(("sw6", "sw3", 8_000_000))

        self.assertEqual(_get_path_bottleneck("sw6", "sw1", load, g), 8_000_000)

    def test_empty_graph_returns_zero(self):
        """Grafo vazio e link_load vazio: fallback retorna 0."""
        self.assertEqual(_get_path_bottleneck("sw6", "sw1", {}, nx.Graph()), 0.0)

    def test_node_not_in_graph_returns_global_max(self):
        """Nó de origem não existe no grafo: exceção capturada, max global."""
        g = _build_graph(("sw3", "sw1"))
        load = _link_load(("sw3", "sw1", 12_000_000))

        self.assertEqual(_get_path_bottleneck("sw6", "sw1", load, g), 12_000_000)

    def test_key_ordering_lookup(self):
        """
        link_load usa chaves (min, max) ordenadas alfabeticamente.
        Garante que sorted() na função encontra a chave independente
        da direção do caminho.
        """
        g = _build_graph(("sw3", "sw1"))
        load = {("sw1", "sw3"): 17_000_000}   # "sw1" < "sw3" ✓

        self.assertEqual(_get_path_bottleneck("sw3", "sw1", load, g), 17_000_000)


# ---------------------------------------------------------------------------
# Testes de get_host_metrics (estado injetado via mock)
# ---------------------------------------------------------------------------

class TestGetHostMetrics(unittest.TestCase):

    def _run(self, ip_to_mac, hosts_by_mac, link_load, graph):
        """Injeta estado mockado e chama get_host_metrics diretamente.

        Patcha 'state' no namespace de api.py, porque get_host_metrics usa
        'from orchestrator.domain.state import state' — um binding local que
        patch.object no módulo domain.state não alcança.
        """
        mock_state = types.SimpleNamespace(
            lock=threading.Lock(),
            ip_to_mac=ip_to_mac,
            hosts_by_mac=hosts_by_mac,
            link_load=link_load,
            graph=graph,
        )
        with patch("orchestrator.presentation.api.state", mock_state):
            return get_host_metrics()

    def test_congested_path_returns_low_bandwidth(self):
        """
        Link ag↔core a 19Mbps sobre capacidade de 20Mbps (95% uso):
        bandwidth disponível deve ser ~1Mbps e latência > 10ms.
        """
        g = _build_graph(("sw6", "sw3"), ("sw3", "sw1"))
        load = _link_load(("sw6", "sw3", 1_000_000), ("sw3", "sw1", 19_000_000))

        result = self._run(
            ip_to_mac    = {"172.16.1.10": "aa:bb:cc:00:00:01"},
            hosts_by_mac = {"aa:bb:cc:00:00:01": {"switch": "openflow:6", "port": "1"}},
            link_load    = load,
            graph        = g,
        )

        host = result["hosts"]["172.16.1.10"]
        self.assertAlmostEqual(host["bandwidth_mbps"], 1.0, places=1)
        self.assertGreater(host["latency_ms"], 10.0)

    def test_idle_path_returns_full_bandwidth(self):
        """Links ociosos → bandwidth ≈ 20Mbps, latência próxima de 2ms."""
        g = _build_graph(("sw6", "sw3"), ("sw3", "sw1"))
        load = _link_load(("sw6", "sw3", 100_000), ("sw3", "sw1", 50_000))

        result = self._run(
            ip_to_mac    = {"172.16.1.10": "aa:bb:cc:00:00:01"},
            hosts_by_mac = {"aa:bb:cc:00:00:01": {"switch": "openflow:6", "port": "1"}},
            link_load    = load,
            graph        = g,
        )

        host = result["hosts"]["172.16.1.10"]
        self.assertGreater(host["bandwidth_mbps"], 19.0)
        self.assertLess(host["latency_ms"], 3.0)

    def test_non_fl_ip_excluded(self):
        """IPs fora de 172.16.1.x não aparecem no resultado."""
        result = self._run(
            ip_to_mac    = {"10.0.0.1": "aa:bb:cc:00:00:99"},
            hosts_by_mac = {"aa:bb:cc:00:00:99": {"switch": "openflow:6", "port": "1"}},
            link_load    = {},
            graph        = nx.Graph(),
        )
        self.assertNotIn("10.0.0.1", result["hosts"])

    def test_host_without_switch_info_no_exception(self):
        """Host sem switch mapeado não deve gerar exceção."""
        g = _build_graph(("sw3", "sw1"))
        load = _link_load(("sw3", "sw1", 5_000_000))

        result = self._run(
            ip_to_mac    = {"172.16.1.11": "aa:bb:cc:00:00:02"},
            hosts_by_mac = {},   # MAC não mapeado
            link_load    = load,
            graph        = g,
        )
        self.assertIn("172.16.1.11", result["hosts"])

    def test_result_has_required_fields(self):
        """Cada entrada deve ter os 6 campos obrigatórios."""
        g = _build_graph(("sw6", "sw3"), ("sw3", "sw1"))
        load = _link_load(("sw6", "sw3", 1_000_000), ("sw3", "sw1", 5_000_000))

        result = self._run(
            ip_to_mac    = {"172.16.1.10": "aa:bb:cc:00:00:01"},
            hosts_by_mac = {"aa:bb:cc:00:00:01": {"switch": "openflow:6", "port": "2"}},
            link_load    = load,
            graph        = g,
        )

        host = result["hosts"]["172.16.1.10"]
        for field in ("bandwidth_mbps", "latency_ms", "packet_loss", "jitter_ms", "switch", "port"):
            self.assertIn(field, host, f"Campo '{field}' ausente na resposta")

    def test_multiple_hosts_different_paths(self):
        """
        Dois hosts em caminhos distintos:
          sw6─sw3─sw1: sw3↔sw1 congestionado (19Mbps)
          sw9─sw4─sw1: sw4↔sw1 ocioso (500kbps)

        Host no caminho congestionado deve ter banda muito menor.
        """
        g = _build_graph(
            ("sw6", "sw3"), ("sw3", "sw1"),
            ("sw9", "sw4"), ("sw4", "sw1"),
        )
        load = _link_load(
            ("sw6", "sw3",  1_000_000),
            ("sw3", "sw1", 19_000_000),
            ("sw9", "sw4",   500_000),
            ("sw4", "sw1",   500_000),
        )

        result = self._run(
            ip_to_mac = {
                "172.16.1.10": "aa:00:00:00:00:01",
                "172.16.1.13": "aa:00:00:00:00:02",
            },
            hosts_by_mac = {
                "aa:00:00:00:00:01": {"switch": "openflow:6", "port": "1"},
                "aa:00:00:00:00:02": {"switch": "openflow:9", "port": "1"},
            },
            link_load = load,
            graph     = g,
        )

        bw_congested = result["hosts"]["172.16.1.10"]["bandwidth_mbps"]
        bw_idle      = result["hosts"]["172.16.1.13"]["bandwidth_mbps"]

        self.assertLess(bw_congested, 5.0)
        self.assertGreater(bw_idle, 15.0)
        self.assertGreater(bw_idle, bw_congested)


if __name__ == "__main__":
    unittest.main(verbosity=2)
