"""
Testes unitários para HCFAnalyzer (Hop Count Filtering).

Verifica as três regras de classificação:
  1. Taxa baixa → BENIGN independente do TTL
  2. hop_count >= HOP_EXTERNAL_MIN → EXTERNAL
  3. hop_count <  HOP_EXTERNAL_MIN + alta taxa → INTERNAL

Sem dependência do estado SDN (state.ip_to_mac é mockado).
"""

import pytest
from unittest.mock import patch

# Importação condicional: módulo SDN usa imports próprios
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdn"))

from orchestrator.application.hcf import (
    HCFAnalyzer,
    TrafficClass,
    TTL_INITIAL_LINUX,
    HOP_EXTERNAL_MIN,
    SUSPICIOUS_PPS,
)


@pytest.fixture()
def analyzer():
    return HCFAnalyzer()


def _classify(analyzer, src_ip, ttl_observed, flow_pkts_s,
               ttl_initial=TTL_INITIAL_LINUX, known=False):
    """Helper: classifica com estado SDN mockado."""
    with patch.object(HCFAnalyzer, "_is_known_host", return_value=known):
        return analyzer.classify(
            src_ip=src_ip,
            ttl_observed=ttl_observed,
            flow_pkts_s=flow_pkts_s,
            ttl_initial=ttl_initial,
        )


# ── Regra 1: taxa baixa → BENIGN ─────────────────────────────────────────────

def test_taxa_baixa_e_benigno(analyzer):
    """Qualquer fluxo abaixo de SUSPICIOUS_PPS deve ser BENIGN."""
    r = _classify(analyzer, "10.0.0.1", ttl_observed=48, flow_pkts_s=SUSPICIOUS_PPS - 1)
    assert r.traffic_class == TrafficClass.BENIGN
    assert r.confidence >= 0.80


def test_taxa_zero_e_benigno(analyzer):
    r = _classify(analyzer, "10.0.0.1", ttl_observed=10, flow_pkts_s=0)
    assert r.traffic_class == TrafficClass.BENIGN


# ── Regra 2: hop_count >= 10 → EXTERNAL ──────────────────────────────────────

def test_hop_count_exato_no_limiar_e_externo(analyzer):
    """hop_count = HOP_EXTERNAL_MIN exatamente → EXTERNAL."""
    ttl_obs = TTL_INITIAL_LINUX - HOP_EXTERNAL_MIN  # 64 - 10 = 54
    r = _classify(analyzer, "1.2.3.4", ttl_observed=ttl_obs, flow_pkts_s=SUSPICIOUS_PPS * 2)
    assert r.traffic_class == TrafficClass.EXTERNAL
    assert r.hop_count == HOP_EXTERNAL_MIN


def test_hop_count_alto_e_externo(analyzer):
    """TTL muito degradado (cruzou 20 roteadores) → EXTERNAL."""
    ttl_obs = TTL_INITIAL_LINUX - 20  # hop_count = 20
    r = _classify(analyzer, "8.8.8.8", ttl_observed=ttl_obs, flow_pkts_s=SUSPICIOUS_PPS * 3)
    assert r.traffic_class == TrafficClass.EXTERNAL
    assert r.hop_count == 20
    assert r.confidence >= 0.70


def test_confianca_cresce_com_hop_count(analyzer):
    """Confiança EXTERNAL deve crescer com o hop_count."""
    ttl_10 = TTL_INITIAL_LINUX - HOP_EXTERNAL_MIN
    ttl_20 = TTL_INITIAL_LINUX - 20
    r10 = _classify(analyzer, "1.1.1.1", ttl_observed=ttl_10, flow_pkts_s=50_000)
    r20 = _classify(analyzer, "1.1.1.1", ttl_observed=ttl_20, flow_pkts_s=50_000)
    assert r20.confidence >= r10.confidence


# ── Regra 3: hop_count < 10 + alta taxa → INTERNAL ───────────────────────────

def test_interno_host_conhecido(analyzer):
    """hop_count=1, taxa alta, IP conhecido no SDN → INTERNAL com alta confiança."""
    r = _classify(analyzer, "10.0.0.5", ttl_observed=63, flow_pkts_s=50_000, known=True)
    assert r.traffic_class == TrafficClass.INTERNAL
    assert r.confidence >= 0.90
    assert r.is_known_host is True


def test_interno_host_desconhecido(analyzer):
    """hop_count=1, taxa alta, IP desconhecido → INTERNAL com confiança menor."""
    r_known   = _classify(analyzer, "10.0.0.5", ttl_observed=63, flow_pkts_s=50_000, known=True)
    r_unknown = _classify(analyzer, "10.0.0.5", ttl_observed=63, flow_pkts_s=50_000, known=False)
    assert r_unknown.traffic_class == TrafficClass.INTERNAL
    assert r_unknown.confidence < r_known.confidence


def test_zero_hops_e_interno(analyzer):
    """TTL intacto (0 hops) + alta taxa → certamente INTERNAL."""
    r = _classify(analyzer, "192.168.0.1", ttl_observed=TTL_INITIAL_LINUX,
                  flow_pkts_s=SUSPICIOUS_PPS * 5, known=True)
    assert r.traffic_class == TrafficClass.INTERNAL
    assert r.hop_count == 0


# ── Metadados do resultado ────────────────────────────────────────────────────

def test_resultado_contem_src_ip(analyzer):
    r = _classify(analyzer, "10.0.0.99", ttl_observed=60, flow_pkts_s=0)
    assert r.src_ip == "10.0.0.99"


def test_hop_count_calculado_corretamente(analyzer):
    """hop_count = ttl_initial - ttl_observed."""
    ttl_obs = 50
    r = _classify(analyzer, "x", ttl_observed=ttl_obs, flow_pkts_s=0)
    assert r.hop_count == TTL_INITIAL_LINUX - ttl_obs


def test_hop_count_negativo_clampado_a_zero(analyzer):
    """TTL observado > TTL inicial → hop_count = 0 (não negativo)."""
    r = _classify(analyzer, "x", ttl_observed=TTL_INITIAL_LINUX + 10, flow_pkts_s=0)
    assert r.hop_count == 0


def test_to_dict_tem_chaves_obrigatorias(analyzer):
    r = _classify(analyzer, "10.0.0.1", ttl_observed=60, flow_pkts_s=50_000, known=True)
    d = r.to_dict()
    for key in ["src_ip", "class", "label", "ttl_observed", "hop_count",
                "is_known_host", "flow_pkts_s", "confidence", "reason"]:
        assert key in d, f"Chave '{key}' ausente no to_dict()"


# ── Batch ─────────────────────────────────────────────────────────────────────

def test_classify_batch_retorna_lista_correta(analyzer):
    """classify_batch deve retornar uma lista com o mesmo tamanho dos fluxos."""
    flows = [
        {"src_ip": "10.0.0.1", "ttl_observed": 63, "flow_pkts_s": 0},
        {"src_ip": "8.8.8.8",  "ttl_observed": 44, "flow_pkts_s": 90_000},
    ]
    with patch.object(HCFAnalyzer, "_is_known_host", return_value=False):
        results = analyzer.classify_batch(flows)
    assert len(results) == 2
