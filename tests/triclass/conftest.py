"""
Fixtures compartilhadas pelos testes triclasse.

Gera dados sintéticos que reproduzem os padrões reais do InSDN:
  - Normal:  tráfego bidirecional, Flow Duration variado, Pkt Len Std alto
  - DDoS:    Pkt Len Std ≈ 0, Flow Duration < 10 µs (burst)
  - BOTNET:  Flow Duration ~31.000 µs (muito > 500), Pkt Len Std ≈ 0
  - DoS:     Flow Duration > 1.000 µs, Pkt Len Std alto (sem burst)
  - Probe:   descartado — não deve aparecer nas 3 classes

As fixtures NÃO carregam os CSVs reais para os testes serem rápidos e
executáveis sem os dados disponíveis.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def insdn_columns() -> list[str]:
    """Colunas do InSDN após RENAME_MAP."""
    return [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Duration",
        "Packet Length Std",
        "Fwd Act Data Pkts",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Label",
    ]


@pytest.fixture
def raw_insdn_df(insdn_columns) -> pd.DataFrame:
    """
    DataFrame sintético com os 4 tipos de tráfego relevantes + Probe.

    Reproduz os padrões reais identificados na análise do InSDN:
      Normal  : bidirec., Std alto, Duration variado
      DDoS    : Std ≈ 0, Duration 1-19 µs → burst=True
      BOTNET  : Std ≈ 0, Duration ~31.000 µs → burst=False
      DoS     : Std alto, Duration > 500 µs → burst=False
      Probe   : fora do escopo → será descartado (label=-1)
    """
    rng = np.random.default_rng(42)
    n   = 100  # amostras por classe

    def make_rows(label, fwd, bwd, fwdb, bwdb, dur, std, act, n_rows=n):
        return pd.DataFrame({
            "Total Fwd Packets":            fwd(n_rows),
            "Total Backward Packets":       bwd(n_rows),
            "Total Length of Fwd Packets":  fwdb(n_rows),
            "Total Length of Bwd Packets":  bwdb(n_rows),
            "Flow Duration":                dur(n_rows),
            "Packet Length Std":            std(n_rows),
            "Fwd Act Data Pkts":            act(n_rows),
            "Fwd Packets/s":                rng.uniform(10, 500, n_rows),
            "Bwd Packets/s":                rng.uniform(5, 300, n_rows),
            "Label":                        label,
        })

    # ── Normal: bidirecional, duração variada, Std alto ───────────────────────
    normal = make_rows(
        "Normal",
        fwd =lambda n: rng.integers(5, 50, n).astype(float),
        bwd =lambda n: rng.integers(5, 50, n).astype(float),
        fwdb=lambda n: rng.uniform(500, 5000, n),
        bwdb=lambda n: rng.uniform(500, 5000, n),
        dur =lambda n: rng.uniform(10_000, 1_000_000, n),
        std =lambda n: rng.uniform(20, 200, n),
        act =lambda n: rng.integers(3, 30, n).astype(float),
    )

    # ── DDoS: burst = True (Std ≤ 1, Duration < 500) ─────────────────────────
    ddos = make_rows(
        "DDoS",
        fwd =lambda n: rng.integers(1, 4, n).astype(float),
        bwd =lambda n: np.zeros(n),
        fwdb=lambda n: rng.uniform(0, 60, n),
        bwdb=lambda n: np.zeros(n),
        dur =lambda n: rng.uniform(1, 19, n),     # µs — muito curto
        std =lambda n: rng.uniform(0, 0.8, n),    # ≤ 1.0
        act =lambda n: np.zeros(n),
    )

    # ── BOTNET: Std ≈ 0 mas Duration ~31.000 µs → burst=False ─────────────────
    botnet = make_rows(
        "BOTNET",
        fwd =lambda n: np.full(n, 4.0),
        bwd =lambda n: np.full(n, 4.0),
        fwdb=lambda n: np.full(n, 203.0),
        bwdb=lambda n: np.full(n, 128.0),
        dur =lambda n: rng.uniform(7_000, 60_000, n),  # µs >> 500
        std =lambda n: rng.uniform(0, 0.8, n),          # ≤ 1.0 mas duration alta
        act =lambda n: rng.integers(0, 2, n).astype(float),
        n_rows=20,  # realista: poucos
    )

    # ── DoS sem burst: Std alto, Duration > 500 ────────────────────────────────
    dos = make_rows(
        "DoS",
        fwd =lambda n: rng.integers(2, 20, n).astype(float),
        bwd =lambda n: rng.integers(1, 10, n).astype(float),
        fwdb=lambda n: rng.uniform(100, 2000, n),
        bwdb=lambda n: rng.uniform(0, 500, n),
        dur =lambda n: rng.uniform(1_000, 100_000, n),  # µs > 500
        std =lambda n: rng.uniform(50, 300, n),           # > 1.0 → burst=False
        act =lambda n: rng.integers(0, 3, n).astype(float),
    )

    # ── Probe: deve ser descartado ─────────────────────────────────────────────
    probe = make_rows(
        "Probe",
        fwd =lambda n: rng.integers(1, 10, n).astype(float),
        bwd =lambda n: rng.integers(1, 10, n).astype(float),
        fwdb=lambda n: rng.uniform(50, 500, n),
        bwdb=lambda n: rng.uniform(50, 500, n),
        dur =lambda n: rng.uniform(1_000, 10_000, n),
        std =lambda n: rng.uniform(5, 50, n),
        act =lambda n: rng.integers(1, 5, n).astype(float),
        n_rows=30,
    )

    return pd.concat([normal, ddos, botnet, dos, probe], ignore_index=True)


@pytest.fixture
def labeled_df(raw_insdn_df) -> pd.DataFrame:
    """DataFrame com label_3class aplicado (linhas -1 já removidas)."""
    from ml.triclass.preprocessing.labeler import TriclassLabeler
    labeler = TriclassLabeler()
    return labeler.fit_transform(raw_insdn_df)


@pytest.fixture
def X_y_train_test(labeled_df):
    """Split 70/30 do DataFrame rotulado, pronto para treino/teste."""
    from sklearn.model_selection import train_test_split
    X = labeled_df.drop(columns=["Label", "label_3class"], errors="ignore")
    X = X.select_dtypes(include=np.number)
    y = labeled_df["label_3class"]
    return train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)


@pytest.fixture
def X_train_engineered(X_y_train_test):
    """X_train com features comportamentais já computadas."""
    from ml.triclass.preprocessing.feature_engineer import BehavioralFeatureEngineer
    X_train, _, _, _ = X_y_train_test
    eng = BehavioralFeatureEngineer()
    return eng.fit_transform(X_train)
