"""
Testes unitários para TopologyFeatureEngineer.

Verifica:
  - 4 novas features são adicionadas corretamente
  - fit_transform usa labels (sem leakage implícito)
  - transform não usa labels (sem leakage explícito)
  - hop_count está no intervalo [0, 60]
  - is_internal é binário (0 ou 1)
  - transform() falha antes de fit_transform() ser chamado
  - TTL real (produção) substitui estimativa sintética
"""

import numpy as np
import pandas as pd
import pytest

from ml.features.topology_features import TopologyFeatureEngineer

_FEATURE_COLS = [
    "Protocol", "Flow Duration", "Flow IAT Max", "Bwd Pkts/s",
    "Pkt Len Std", "Pkt Len Var", "Bwd IAT Tot", "Flow Pkts/s",
]


def _make_X(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "Protocol":      rng.randint(0, 17, n),
        "Flow Duration": rng.uniform(1, 1e8, n),
        "Flow IAT Max":  rng.uniform(1, 1e8, n),
        "Bwd Pkts/s":    rng.uniform(0, 2e6, n),
        "Pkt Len Std":   rng.uniform(0, 1000, n),
        "Pkt Len Var":   rng.uniform(0, 1e6, n),
        "Bwd IAT Tot":   rng.uniform(0, 1e8, n),
        "Flow Pkts/s":   rng.uniform(0, 3e6, n),
    })


def _make_y(n: int = 50, seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randint(0, 3, n))


# ── Estrutura das features ────────────────────────────────────────────────────

def test_novas_features_adicionadas():
    """fit_transform deve adicionar exatamente 4 novas features."""
    eng = TopologyFeatureEngineer()
    X = _make_X()
    y = _make_y()
    X_out = eng.fit_transform(X, y)
    for col in ["ttl_estimated", "hop_count", "is_internal", "ttl_anomaly"]:
        assert col in X_out.columns, f"Feature '{col}' ausente"


def test_shape_correto():
    """Shape deve ter 4 colunas a mais que a entrada."""
    eng = TopologyFeatureEngineer()
    X = _make_X(n=100)
    y = _make_y(n=100)
    X_out = eng.fit_transform(X, y)
    assert X_out.shape == (100, len(_FEATURE_COLS) + 4)


# ── Restrições de valores ─────────────────────────────────────────────────────

def test_hop_count_intervalo_valido():
    """hop_count deve estar em [0, 60] após clipping."""
    eng = TopologyFeatureEngineer()
    X_out = eng.fit_transform(_make_X(200), _make_y(200))
    assert X_out["hop_count"].between(0, 60).all(), \
        "hop_count fora do intervalo [0, 60]"


def test_is_internal_binario():
    """is_internal deve ser 0 ou 1 — sem valores intermediários."""
    eng = TopologyFeatureEngineer()
    X_out = eng.fit_transform(_make_X(200), _make_y(200))
    assert set(X_out["is_internal"].unique()).issubset({0, 1}), \
        "is_internal contém valores não-binários"


def test_ttl_estimated_intervalo():
    """ttl_estimated deve estar em [1, 255]."""
    eng = TopologyFeatureEngineer()
    X_out = eng.fit_transform(_make_X(200), _make_y(200))
    assert X_out["ttl_estimated"].between(1, 255).all(), \
        "ttl_estimated fora do intervalo [1, 255]"


def test_ttl_anomaly_nao_negativo():
    """ttl_anomaly = |hop_count - mediana| deve ser ≥ 0."""
    eng = TopologyFeatureEngineer()
    X_out = eng.fit_transform(_make_X(100), _make_y(100))
    assert (X_out["ttl_anomaly"] >= 0).all()


# ── Sem leakage ───────────────────────────────────────────────────────────────

def test_transform_sem_labels_nao_falha():
    """transform() no teste (sem labels) deve funcionar sem erros."""
    eng = TopologyFeatureEngineer()
    eng.fit_transform(_make_X(50), _make_y(50))   # fase treino
    X_test = _make_X(20, seed=99)
    X_out = eng.transform(X_test)                  # fase teste — sem labels
    for col in ["ttl_estimated", "hop_count", "is_internal", "ttl_anomaly"]:
        assert col in X_out.columns


def test_transform_antes_de_fit_levanta_erro():
    """transform() antes de fit_transform() deve levantar RuntimeError."""
    eng = TopologyFeatureEngineer()
    with pytest.raises(RuntimeError):
        eng.transform(_make_X(10))


# ── TTL real (produção) ───────────────────────────────────────────────────────

def test_ttl_real_substitui_estimativa():
    """Quando ttl_real é fornecido, ttl_estimated deve refletir esses valores."""
    eng = TopologyFeatureEngineer()
    eng.fit_transform(_make_X(50), _make_y(50))
    X_test = _make_X(5, seed=7)
    ttl_real = np.array([60.0, 50.0, 40.0, 62.0, 48.0])
    X_out = eng.transform(X_test, ttl_real=ttl_real)
    np.testing.assert_array_almost_equal(
        X_out["ttl_estimated"].values,
        np.round(ttl_real, 1),
        err_msg="ttl_estimated não reflete ttl_real fornecido",
    )


# ── Sem side-effect ───────────────────────────────────────────────────────────

def test_sem_side_effect_no_X_original():
    """fit_transform() não deve modificar o DataFrame original."""
    eng = TopologyFeatureEngineer()
    X = _make_X(30)
    X_orig = X.copy()
    eng.fit_transform(X, _make_y(30))
    pd.testing.assert_frame_equal(X, X_orig)
