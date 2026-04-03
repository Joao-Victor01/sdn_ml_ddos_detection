"""
Testes unitários para LabelEngineer.

Verifica:
  - Conversão correta de labels binárias → triclasse
  - Benigno (0) sempre permanece benigno
  - Protocol=0 + critérios de burst → Externo (1)
  - DDoS sem critérios externos → Interno (2)
  - Nenhuma modificação no DataFrame original (sem side-effects)
"""

import numpy as np
import pandas as pd
import pytest

from ml.data.label_engineer import LabelEngineer
from ml.config import EXTERNAL_PROTOCOL_ID, EXTERNAL_DURATION_THRESH, EXTERNAL_STD_THRESH


def _make_row(
    label: int,
    protocol: int = 6,
    pkt_len_std: float = 100.0,
    pkt_len_var: float = 100.0,
    flow_duration: float = 100_000.0,
) -> dict:
    return {
        "Protocol":      protocol,
        "Pkt Len Std":   pkt_len_std,
        "Pkt Len Var":   pkt_len_var,
        "Flow Duration": flow_duration,
        "Flow IAT Max":  1000.0,
        "Bwd Pkts/s":    500.0,
        "Bwd IAT Tot":   2000.0,
        "Flow Pkts/s":   1000.0,
        "Label":         label,
    }


@pytest.fixture()
def engineer():
    return LabelEngineer()


# ── Benigno ───────────────────────────────────────────────────────────────────

def test_benigno_permanece_zero(engineer):
    """Amostras benignas (y=0) jamais devem mudar de classe."""
    rows = [_make_row(0, protocol=EXTERNAL_PROTOCOL_ID) for _ in range(10)]
    X = pd.DataFrame(rows).drop(columns=["Label"])
    y = pd.Series([0] * 10)
    y3 = engineer.transform(X, y)
    assert (y3 == 0).all(), "Benigno com Protocol=0 não deve virar Externo"


# ── Ataque Externo (1) ────────────────────────────────────────────────────────

def test_protocol_zero_vira_externo(engineer):
    """DDoS com Protocol=0 → classe 1 (Externo)."""
    row = _make_row(1, protocol=EXTERNAL_PROTOCOL_ID)
    X = pd.DataFrame([row]).drop(columns=["Label"])
    y = pd.Series([1])
    y3 = engineer.transform(X, y)
    assert y3.iloc[0] == 1, f"Esperado 1 (Externo), obtido {y3.iloc[0]}"


def test_burst_uniforme_curto_vira_externo(engineer):
    """DDoS com burst uniforme e curto → classe 1 (Externo)."""
    row = _make_row(
        1,
        protocol=6,
        pkt_len_std=0.0,                               # ≤ STD_THRESH
        pkt_len_var=0.0,                               # ≤ STD_THRESH
        flow_duration=EXTERNAL_DURATION_THRESH - 1,    # < limiar de duração
    )
    X = pd.DataFrame([row]).drop(columns=["Label"])
    y = pd.Series([1])
    y3 = engineer.transform(X, y)
    assert y3.iloc[0] == 1, f"Burst uniforme curto deveria ser Externo, obtido {y3.iloc[0]}"


# ── Zumbi Interno (2) ─────────────────────────────────────────────────────────

def test_ddos_sem_criterio_externo_vira_interno(engineer):
    """DDoS sem indicadores externos → classe 2 (Interno)."""
    row = _make_row(
        1,
        protocol=6,
        pkt_len_std=200.0,
        pkt_len_var=200.0,
        flow_duration=1_000_000.0,
    )
    X = pd.DataFrame([row]).drop(columns=["Label"])
    y = pd.Series([1])
    y3 = engineer.transform(X, y)
    assert y3.iloc[0] == 2, f"DDoS interno deveria ser classe 2, obtido {y3.iloc[0]}"


# ── Propriedades gerais ───────────────────────────────────────────────────────

def test_apenas_classes_0_1_2(engineer):
    """Todas as saídas devem ser 0, 1 ou 2."""
    rows = [_make_row(i % 2) for i in range(50)]
    X = pd.DataFrame(rows).drop(columns=["Label"])
    y = pd.Series([i % 2 for i in range(50)])
    y3 = engineer.transform(X, y)
    assert set(y3.unique()).issubset({0, 1, 2})


def test_sem_side_effect_no_dataframe(engineer):
    """transform() não deve modificar o DataFrame original."""
    row = _make_row(1, protocol=EXTERNAL_PROTOCOL_ID)
    X = pd.DataFrame([row]).drop(columns=["Label"])
    X_original = X.copy()
    y = pd.Series([1])
    engineer.transform(X, y)
    pd.testing.assert_frame_equal(X, X_original)


def test_tamanho_saida_igual_entrada(engineer):
    """A série de saída deve ter o mesmo número de linhas que a entrada."""
    rows = [_make_row(i % 2) for i in range(30)]
    X = pd.DataFrame(rows).drop(columns=["Label"])
    y = pd.Series([i % 2 for i in range(30)])
    y3 = engineer.transform(X, y)
    assert len(y3) == len(y)
