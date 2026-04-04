"""
Testes unitários para ml/triclass/evaluation/evaluator.py
e ml/triclass/evaluation/semantic_validator.py

Valida:
  1. TriclassEvaluationResult — campos calculados corretamente
  2. TriclassEvaluator — métricas corretas em predição perfeita e aleatória
  3. BotnetSemanticValidator — recall semântico do BOTNET
  4. Casos de borda: sem BOTNET no teste, todos errados
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from unittest.mock import MagicMock

from ml.triclass.evaluation.evaluator import (
    TriclassEvaluationResult,
    TriclassEvaluator,
)
from ml.triclass.evaluation.semantic_validator import (
    BotnetSemanticValidator,
    BotnetValidationResult,
)


@pytest.fixture
def perfect_arrays():
    """Predição perfeita — y_pred == y_test."""
    y_true = np.array([0]*50 + [1]*50 + [2]*10)
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def random_arrays():
    """Predição aleatória — baixas métricas esperadas."""
    rng = np.random.default_rng(42)
    y_true = np.array([0]*50 + [1]*50 + [2]*10)
    y_pred = rng.choice([0, 1, 2], size=len(y_true))
    return y_true, y_pred


class TestTriclassEvaluationResult:
    """Testes para o dataclass de resultado."""

    def test_recall_class2_com_3_classes(self):
        r = TriclassEvaluationResult(
            label="test",
            f1_macro=0.9,
            mcc=0.85,
            gm=0.88,
            f1_per_class=[0.95, 0.90, 0.80],
            precision_per_class=[0.95, 0.90, 0.80],
            recall_per_class=[0.95, 0.90, 0.75],
            support_per_class=[50, 50, 10],
        )
        assert r.recall_class2 == pytest.approx(0.75)

    def test_f1_class2(self):
        r = TriclassEvaluationResult(
            label="test",
            f1_macro=0.9, mcc=0.85, gm=0.88,
            f1_per_class=[0.95, 0.90, 0.72],
            precision_per_class=[0.95, 0.90, 0.72],
            recall_per_class=[0.95, 0.90, 0.72],
            support_per_class=[50, 50, 10],
        )
        assert r.f1_class2 == pytest.approx(0.72)

    def test_recall_class2_sem_classe_2_retorna_zero(self):
        r = TriclassEvaluationResult(
            label="test",
            f1_macro=0.9, mcc=0.85, gm=0.88,
            f1_per_class=[0.95, 0.90],
            precision_per_class=[0.95, 0.90],
            recall_per_class=[0.95, 0.90],
            support_per_class=[50, 50],
        )
        assert r.recall_class2 == 0.0


class TestTriclassEvaluator:
    """Testes para a classe TriclassEvaluator."""

    def test_predicao_perfeita_f1_macro_1(self, perfect_arrays, tmp_path):
        y_true, y_pred = perfect_arrays
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(y_true.reshape(-1, 1), y_true)
        # Overrride predict para retornar perfeito
        model.predict = lambda X: y_pred

        evaluator = TriclassEvaluator(save_plots=False)
        result = evaluator.evaluate(model, y_true.reshape(-1, 1), y_true, "Perfeito")

        assert result.f1_macro == pytest.approx(1.0, abs=1e-6)

    def test_predicao_perfeita_mcc_1(self, perfect_arrays, tmp_path):
        y_true, y_pred = perfect_arrays
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(y_true.reshape(-1, 1), y_true)
        model.predict = lambda X: y_pred

        evaluator = TriclassEvaluator(save_plots=False)
        result = evaluator.evaluate(model, y_true.reshape(-1, 1), y_true, "Perfeito")
        assert result.mcc == pytest.approx(1.0, abs=1e-6)

    def test_resultado_tem_campos_corretos(self, perfect_arrays):
        y_true, y_pred = perfect_arrays
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(y_true.reshape(-1, 1), y_true)
        model.predict = lambda X: y_pred

        evaluator = TriclassEvaluator(save_plots=False)
        result = evaluator.evaluate(model, y_true.reshape(-1, 1), y_true, "Test")

        assert isinstance(result.f1_macro, float)
        assert isinstance(result.mcc, float)
        assert isinstance(result.gm, float)
        assert isinstance(result.f1_per_class, list)
        assert isinstance(result.recall_per_class, list)
        assert isinstance(result.report, str)

    def test_suporte_por_classe_correto(self, perfect_arrays):
        y_true, _ = perfect_arrays
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(y_true.reshape(-1, 1), y_true)
        model.predict = lambda X: y_true.copy()

        evaluator = TriclassEvaluator(save_plots=False)
        result = evaluator.evaluate(model, y_true.reshape(-1, 1), y_true, "Test")

        assert result.support_per_class[0] == 50
        assert result.support_per_class[1] == 50
        assert result.support_per_class[2] == 10

    def test_f1_macro_range(self, random_arrays):
        """F1 macro deve estar em [0, 1]."""
        y_true, y_pred = random_arrays
        model = DummyClassifier(strategy="constant", constant=0)
        model.fit(y_true.reshape(-1, 1), y_true)
        model.predict = lambda X: y_pred

        evaluator = TriclassEvaluator(save_plots=False)
        result = evaluator.evaluate(model, y_true.reshape(-1, 1), y_true, "Random")
        assert 0 <= result.f1_macro <= 1

    def test_compare_aceita_multiplos_resultados(self):
        """compare() deve funcionar com 2 ou 3 modelos."""
        def make_result(label):
            return TriclassEvaluationResult(
                label=label, f1_macro=0.9, mcc=0.85, gm=0.88,
                f1_per_class=[0.9, 0.9, 0.8],
                precision_per_class=[0.9, 0.9, 0.8],
                recall_per_class=[0.9, 0.9, 0.8],
                support_per_class=[50, 50, 10],
            )
        evaluator = TriclassEvaluator(save_plots=False)
        # Não deve lançar exceção
        evaluator.compare(make_result("A"), make_result("B"), make_result("C"))


class TestBotnetSemanticValidator:
    """Testes para a validação semântica BOTNET."""

    def _make_data_with_botnet(self, n_botnet=20, n_other=80):
        """Cria DataFrame sintético com BOTNET e outros labels."""
        rng = np.random.default_rng(42)
        labels = (
            ["BOTNET"] * n_botnet +
            ["DDoS"]   * (n_other // 2) +
            ["Normal"] * (n_other // 2)
        )
        df = pd.DataFrame({
            "Label": labels,
            "Flow Duration": rng.uniform(1, 1000, len(labels)),
        })
        return df

    def test_recall_perfeito_quando_tudo_classificado_como_2(self):
        """Recall = 1.0 quando todos os BOTNET preditos como Classe 2."""
        n_botnet = 20
        data = self._make_data_with_botnet(n_botnet=n_botnet, n_other=80)
        test_ix = np.arange(len(data))

        # Mock com predict dinâmico: retorna array do tamanho do input
        model = MagicMock()
        model.predict.side_effect = lambda X: np.full(len(X), 2)
        X_dummy = np.zeros((len(data), 1))

        validator = BotnetSemanticValidator(data, test_ix)
        result = validator.validate(model, X_dummy)

        assert result.n_botnet_test == n_botnet
        assert result.recall == pytest.approx(1.0)
        assert result.passed

    def test_recall_zero_quando_tudo_classificado_como_0(self):
        """Recall = 0.0 quando todos preditos como Classe 0."""
        data = self._make_data_with_botnet(n_botnet=20, n_other=80)
        test_ix = np.arange(len(data))
        model = DummyClassifier(strategy="constant", constant=0)
        X_dummy = np.zeros((len(data), 1))
        model.fit(X_dummy, np.zeros(len(data)))

        validator = BotnetSemanticValidator(data, test_ix)
        result = validator.validate(model, X_dummy, min_recall=0.80)

        assert result.recall == pytest.approx(0.0)
        assert not result.passed

    def test_sem_botnet_no_teste(self):
        """Quando não há BOTNET no teste, resultado deve indicar n=0."""
        data = pd.DataFrame({
            "Label": ["Normal"] * 50 + ["DDoS"] * 50,
            "Flow Duration": np.random.uniform(1, 1000, 100),
        })
        test_ix = np.arange(len(data))
        model = MagicMock()
        model.predict.side_effect = lambda X: np.full(len(X), 2)
        X_dummy = np.zeros((len(data), 1))

        validator = BotnetSemanticValidator(data, test_ix)
        result = validator.validate(model, X_dummy)

        assert result.n_botnet_test == 0
        assert result.recall == 0.0
        assert not result.passed

    def test_contagens_somam_ao_total(self):
        """cls0 + cls1 + cls2 deve igualar n_botnet_test."""
        data = self._make_data_with_botnet(n_botnet=15, n_other=60)
        test_ix = np.arange(len(data))
        rng = np.random.default_rng(0)
        preds = rng.choice([0, 1, 2], size=len(data))
        model = DummyClassifier(strategy="constant", constant=0)
        X_dummy = np.zeros((len(data), 1))
        model.fit(X_dummy, np.zeros(len(data)))
        model.predict = lambda X: preds[:len(X)]

        validator = BotnetSemanticValidator(data, test_ix)
        result = validator.validate(model, X_dummy)

        total = (result.n_classified_as_class0 +
                 result.n_classified_as_class1 +
                 result.n_classified_as_class2)
        assert total == result.n_botnet_test
