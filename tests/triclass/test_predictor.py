"""
Testes unitários para ml/triclass/inference/predictor.py

Valida o DDoSPredictorV2 com artefatos mockados — não exige modelos reais.
"""

import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_rf():
    """Mock do RandomForest com predict e predict_proba."""
    rf = MagicMock()
    rf.predict.return_value = np.array([1])
    rf.predict_proba.return_value = np.array([[0.05, 0.90, 0.05]])
    return rf


@pytest.fixture
def mock_imputer():
    imp = MagicMock()
    imp.transform.side_effect = lambda X: np.nan_to_num(X, nan=0.0)
    return imp


@pytest.fixture
def mock_vt():
    vt = MagicMock()
    vt.transform.side_effect = lambda X: X
    return vt


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Features brutas de um fluxo de rede (pós RENAME_MAP, típico de DDoS externo)."""
    return pd.DataFrame({
        "Total Fwd Packets":           [4.0],
        "Total Backward Packets":      [0.0],
        "Total Length of Fwd Packets": [200.0],
        "Total Length of Bwd Packets": [0.0],
        "Flow Duration":               [10.0],
        "Packet Length Std":           [0.0],
        "Fwd Act Data Pkts":           [0.0],
        "Fwd Packets/s":               [400_000.0],
        "Bwd Packets/s":               [0.0],
    })


class TestDDoSPredictorV2Logic:
    """
    Testes da lógica de predição sem carregar artefatos do disco.
    Usa mocks para substituir joblib.load.
    """

    def _build_predictor(self, mock_rf, mock_imputer, mock_vt):
        """Constrói DDoSPredictorV2 com artefatos mockados."""
        from ml.triclass.inference.predictor import DDoSPredictorV2

        with patch("ml.triclass.inference.predictor.joblib") as mock_joblib:
            mock_joblib.load.side_effect = lambda path: (
                mock_rf       if "rf_triclass" in str(path)   else
                mock_imputer  if "imputer" in str(path)        else
                mock_vt       if "variance_filter" in str(path) else
                ["Total Fwd Packets", "Total Backward Packets",
                 "Flow Duration", "Packet Length Std",
                 "asymmetry_pkts", "pkt_uniformity", "log_duration"]
                if "selected_features" in str(path) else None
            )
            predictor = DDoSPredictorV2(
                models_triclass_dir="/fake/models"
            )
        return predictor

    def test_predict_retorna_dict(self, mock_rf, mock_imputer, mock_vt, sample_features):
        """predict() deve retornar dict com campos obrigatórios."""
        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features)

        assert isinstance(result, dict)
        assert "class" in result
        assert "label" in result
        assert "action" in result
        assert "confidence" in result

    def test_classe_1_retorna_block_global(self, mock_rf, mock_imputer, mock_vt, sample_features):
        """Classe 1 (Externo) → action = 'block_global'."""
        mock_rf.predict.return_value = np.array([1])
        mock_rf.predict_proba.return_value = np.array([[0.05, 0.90, 0.05]])

        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features)

        assert result["class"] == 1
        assert result["action"] == "block_global"

    def test_classe_2_retorna_isolate_surgical(self, mock_rf, mock_imputer, mock_vt, sample_features):
        """Classe 2 (Interno) → action = 'isolate_surgical'."""
        mock_rf.predict.return_value = np.array([2])
        mock_rf.predict_proba.return_value = np.array([[0.05, 0.05, 0.90]])

        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features)

        assert result["class"] == 2
        assert result["action"] == "isolate_surgical"

    def test_classe_0_retorna_none(self, mock_rf, mock_imputer, mock_vt, sample_features):
        """Classe 0 (Benigno) → action = 'none'."""
        mock_rf.predict.return_value = np.array([0])
        mock_rf.predict_proba.return_value = np.array([[0.95, 0.03, 0.02]])

        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features)

        assert result["class"] == 0
        assert result["action"] == "none"

    def test_fallback_is_known_host_reclassifica_para_2(
        self, mock_rf, mock_imputer, mock_vt, sample_features
    ):
        """
        Fallback determinístico (Opção C):
        Se is_known_host=True e RF prediz Classe 1 → forçar Classe 2.
        """
        mock_rf.predict.return_value = np.array([1])  # RF diz Externo
        mock_rf.predict_proba.return_value = np.array([[0.05, 0.90, 0.05]])

        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features, is_known_host=True)

        assert result["class"] == 2
        assert result["action"] == "isolate_surgical"
        assert "reason" in result

    def test_is_known_host_false_nao_reclassifica(
        self, mock_rf, mock_imputer, mock_vt, sample_features
    ):
        """is_known_host=False → não aplica fallback determinístico."""
        mock_rf.predict.return_value = np.array([1])
        mock_rf.predict_proba.return_value = np.array([[0.05, 0.90, 0.05]])

        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features, is_known_host=False)

        assert result["class"] == 1

    def test_confidence_entre_0_e_1(self, mock_rf, mock_imputer, mock_vt, sample_features):
        """Confiança deve estar em [0, 1]."""
        mock_rf.predict.return_value = np.array([1])
        mock_rf.predict_proba.return_value = np.array([[0.05, 0.90, 0.05]])

        predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
        result = predictor.predict(sample_features)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_label_consistente_com_classe(self, mock_rf, mock_imputer, mock_vt, sample_features):
        """O label retornado deve corresponder ao CLASS_NAMES da classe predita."""
        from ml.triclass.config import CLASS_NAMES

        for classe, expected_label in CLASS_NAMES.items():
            proba = [0.05, 0.05, 0.05]
            proba[classe] = 0.90
            mock_rf.predict.return_value = np.array([classe])
            mock_rf.predict_proba.return_value = np.array([proba])

            predictor = self._build_predictor(mock_rf, mock_imputer, mock_vt)
            result = predictor.predict(sample_features)

            assert result["label"] == expected_label, (
                f"Classe {classe}: esperado '{expected_label}', "
                f"obtido '{result['label']}'"
            )
