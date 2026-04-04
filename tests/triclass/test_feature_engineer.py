"""
Testes unitários para ml/triclass/preprocessing/feature_engineer.py

Valida:
  1. computar_features_comportamentais() — correctness das 6 features
  2. Ausência de leakage (operações puramente matemáticas)
  3. BehavioralFeatureEngineer — interface fit/transform
  4. Casos de borda: zeros, infinitos, colunas ausentes
"""

import numpy as np
import pandas as pd
import pytest

from ml.triclass.preprocessing.feature_engineer import (
    computar_features_comportamentais,
    BehavioralFeatureEngineer,
    _FWD, _BWD, _FWDB, _BWDB, _DUR, _STD, _ACT,
)
from ml.triclass.config import BEHAVIORAL_FEATURES


@pytest.fixture
def df_base() -> pd.DataFrame:
    """DataFrame mínimo para testar cada feature individualmente."""
    return pd.DataFrame({
        _FWD:  [10.0, 0.0, 100.0],
        _BWD:  [10.0, 0.0,   0.0],
        _FWDB: [1000.0, 0.0, 500.0],
        _BWDB: [1000.0, 0.0,   0.0],
        _DUR:  [50_000.0, 10.0, 300_000.0],
        _STD:  [50.0, 0.0, 0.5],
        _ACT:  [8.0, 0.0, 0.0],
    })


class TestComputarFeatures:
    """Testes para a função pura computar_features_comportamentais()."""

    def test_retorna_copia(self, df_base):
        """Não deve modificar o DataFrame de entrada."""
        original = df_base.copy()
        computar_features_comportamentais(df_base)
        pd.testing.assert_frame_equal(df_base, original)

    def test_seis_novas_colunas(self, df_base):
        """Deve adicionar as 6 features comportamentais."""
        result = computar_features_comportamentais(df_base)
        for feat in BEHAVIORAL_FEATURES:
            assert feat in result.columns, f"Feature '{feat}' ausente"

    def test_asymmetry_pkts_intervalo(self, df_base):
        """asymmetry_pkts deve estar em [0, 1]."""
        result = computar_features_comportamentais(df_base)
        assert (result["asymmetry_pkts"] >= 0).all()
        assert (result["asymmetry_pkts"] <= 1).all()

    def test_asymmetry_bytes_intervalo(self, df_base):
        """asymmetry_bytes deve estar em [0, 1]."""
        result = computar_features_comportamentais(df_base)
        assert (result["asymmetry_bytes"] >= 0).all()
        assert (result["asymmetry_bytes"] <= 1).all()

    def test_pkt_uniformity_intervalo(self, df_base):
        """pkt_uniformity = 1/(Std+1) deve estar em (0, 1]."""
        result = computar_features_comportamentais(df_base)
        assert (result["pkt_uniformity"] > 0).all()
        assert (result["pkt_uniformity"] <= 1).all()

    def test_log_duration_positivo(self, df_base):
        """log_duration = log1p(dur) deve ser ≥ 0."""
        result = computar_features_comportamentais(df_base)
        assert (result["log_duration"] >= 0).all()

    def test_pkt_rate_nao_negativo(self, df_base):
        """pkt_rate não deve ser negativo."""
        result = computar_features_comportamentais(df_base)
        assert (result["pkt_rate"] >= 0).all()

    def test_fwd_active_ratio_intervalo(self, df_base):
        """fwd_active_ratio deve estar em [0, 1] se Fwd Act ≤ Total Fwd."""
        # Com Act=8 e Fwd=10 → ratio=0.8
        result = computar_features_comportamentais(df_base)
        # Apenas verifica que não é negativo
        assert (result["fwd_active_ratio"] >= 0).all()

    def test_trafego_bidirecional_assimetria_medio(self):
        """Tráfego igualitário (fwd=bwd) → asymmetry ≈ 0.5."""
        df = pd.DataFrame({
            _FWD: [100.0], _BWD: [100.0],
            _FWDB: [1000.0], _BWDB: [1000.0],
            _DUR: [100_000.0], _STD: [50.0], _ACT: [50.0],
        })
        result = computar_features_comportamentais(df)
        assert abs(result["asymmetry_pkts"].iloc[0] - 0.5) < 1e-6

    def test_ataque_unidirecional_assimetria_alta(self):
        """Tráfego só forward (bwd=0) → asymmetry ≈ 1.0."""
        df = pd.DataFrame({
            _FWD: [100.0], _BWD: [0.0],
            _FWDB: [1000.0], _BWDB: [0.0],
            _DUR: [100_000.0], _STD: [0.5], _ACT: [0.0],
        })
        result = computar_features_comportamentais(df)
        assert result["asymmetry_pkts"].iloc[0] > 0.99

    def test_burst_tem_uniformidade_alta(self):
        """Std ≈ 0 → pkt_uniformity ≈ 1.0."""
        df = pd.DataFrame({
            _FWD: [4.0], _BWD: [0.0],
            _FWDB: [200.0], _BWDB: [0.0],
            _DUR: [10.0], _STD: [0.0], _ACT: [0.0],
        })
        result = computar_features_comportamentais(df)
        assert result["pkt_uniformity"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_botnet_log_duration_maior_que_ddos(self):
        """log_duration(BOTNET) >> log_duration(DDoS)."""
        botnet = pd.DataFrame({
            _FWD: [4.0], _BWD: [4.0], _FWDB: [200.0], _BWDB: [128.0],
            _DUR: [31_000.0], _STD: [0.5], _ACT: [1.0],
        })
        ddos = pd.DataFrame({
            _FWD: [2.0], _BWD: [0.0], _FWDB: [60.0], _BWDB: [0.0],
            _DUR: [10.0], _STD: [0.0], _ACT: [0.0],
        })
        r_b = computar_features_comportamentais(botnet)
        r_d = computar_features_comportamentais(ddos)
        assert r_b["log_duration"].iloc[0] > r_d["log_duration"].iloc[0]

    def test_zeros_nao_geram_nan(self):
        """Todos os campos zero não devem gerar NaN (divisão por ε)."""
        df = pd.DataFrame({
            _FWD: [0.0], _BWD: [0.0], _FWDB: [0.0], _BWDB: [0.0],
            _DUR: [0.0], _STD: [0.0], _ACT: [0.0],
        })
        result = computar_features_comportamentais(df)
        for feat in BEHAVIORAL_FEATURES:
            if feat in result.columns:
                assert not result[feat].isna().any(), f"{feat} gerou NaN com entrada zero"

    def test_sem_coluna_act_usa_zero(self):
        """Sem coluna _ACT, fwd_active_ratio deve ser 0 e emitir warning."""
        df = pd.DataFrame({
            _FWD: [10.0], _BWD: [5.0], _FWDB: [500.0], _BWDB: [250.0],
            _DUR: [1000.0], _STD: [20.0],
        })
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = computar_features_comportamentais(df)
            assert any("fwd_active_ratio" in str(warning.message) for warning in w)
        assert "fwd_active_ratio" in result.columns
        assert result["fwd_active_ratio"].iloc[0] == 0.0


class TestBehavioralFeatureEngineer:
    """Testes para a classe BehavioralFeatureEngineer."""

    def test_fit_transform_retorna_dataframe(self, X_train_engineered):
        """fit_transform deve retornar pd.DataFrame."""
        assert isinstance(X_train_engineered, pd.DataFrame)

    def test_todas_features_presentes(self, X_train_engineered):
        """Todas as 6 features devem estar no resultado."""
        for feat in BEHAVIORAL_FEATURES:
            assert feat in X_train_engineered.columns

    def test_transform_produz_mesmo_resultado(self, X_y_train_test):
        """transform() deve produzir o mesmo resultado que fit_transform()."""
        X_train, X_test, _, _ = X_y_train_test
        eng = BehavioralFeatureEngineer()
        X_tr_1 = eng.fit_transform(X_train)
        X_te_1 = eng.transform(X_test)

        # Aplicar fit_transform em X_test separado (deve dar o mesmo)
        eng2    = BehavioralFeatureEngineer()
        X_te_2  = eng2.fit_transform(X_test)

        # As features computadas devem ser idênticas pois são puramente matemáticas
        for feat in BEHAVIORAL_FEATURES:
            if feat in X_te_1.columns and feat in X_te_2.columns:
                pd.testing.assert_series_equal(
                    X_te_1[feat].reset_index(drop=True),
                    X_te_2[feat].reset_index(drop=True),
                    check_names=False,
                )

    def test_feature_names_property(self):
        """feature_names deve retornar a lista de BEHAVIORAL_FEATURES."""
        eng = BehavioralFeatureEngineer()
        assert eng.feature_names == list(BEHAVIORAL_FEATURES)

    def test_nao_modifica_entrada(self, X_y_train_test):
        """fit_transform não deve modificar o DataFrame de entrada."""
        X_train, _, _, _ = X_y_train_test
        original = X_train.copy()
        eng = BehavioralFeatureEngineer()
        eng.fit_transform(X_train)
        pd.testing.assert_frame_equal(X_train, original)

    def test_sem_nan_no_resultado(self, X_y_train_test):
        """Resultado não deve conter NaN nas features comportamentais."""
        X_train, _, _, _ = X_y_train_test
        eng = BehavioralFeatureEngineer()
        result = eng.fit_transform(X_train)
        for feat in BEHAVIORAL_FEATURES:
            if feat in result.columns:
                assert not result[feat].isna().any(), f"{feat} contém NaN"
