"""
Testes de boas práticas de ML — verificam que o pipeline não comete leakage.

Estas são as regras do curso (Thaís Gaudencio / UFPB):
  1. Split ANTES de qualquer transformação
  2. fit() e fit_transform() SOMENTE no treino
  3. SMOTE somente no treino
  4. Test set usado exatamente uma vez (avaliação final)
  5. Scaler ajustado no treino, apenas transform() no teste

Os testes usam dados sintéticos mínimos (sem carregar o dataset real).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier

from ml.preprocessing.cleaner  import DataCleaner
from ml.preprocessing.scaler   import FeatureScaler
from ml.preprocessing.balancer import ClassBalancer
from ml.features.topology_features import TopologyFeatureEngineer


def _make_df(n=200, n_feat=8, seed=42):
    rng = np.random.RandomState(seed)
    cols = ["Protocol", "Flow Duration", "Flow IAT Max", "Bwd Pkts/s",
            "Pkt Len Std", "Pkt Len Var", "Bwd IAT Tot", "Flow Pkts/s"]
    data = rng.uniform(0, 1000, (n, n_feat))
    return pd.DataFrame(data, columns=cols[:n_feat])


def _make_y(n=200, n_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randint(0, n_classes, n))


# ── FeatureScaler ─────────────────────────────────────────────────────────────

class TestFeatureScaler:
    def test_fit_no_treino_transform_no_teste(self):
        """Scaler ajustado no treino → média ≈ 0 e std ≈ 1 no treino."""
        scaler = FeatureScaler()
        X_train = _make_df(100).values.astype(float)
        X_test  = _make_df(50, seed=99).values.astype(float)
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        # Treino: média ≈ 0
        assert abs(X_tr.mean()) < 0.1, "Média do treino escalado não está próxima de 0"
        # Teste: apenas transform, não deve falhar
        assert X_te.shape == X_test.shape

    def test_transform_antes_de_fit_levanta_erro(self):
        """transform() antes de fit_transform() deve falhar."""
        scaler = FeatureScaler()
        with pytest.raises(Exception):
            scaler.transform(np.random.rand(10, 4))

    def test_parametros_do_treino_nao_contaminam_teste(self):
        """O scaler usa os parâmetros do treino no teste — não re-ajusta."""
        scaler = FeatureScaler()
        # Treino com valores em [0, 1]
        X_train = np.random.rand(100, 4)
        # Teste com valores em [1000, 2000] — média muito diferente
        X_test  = np.random.rand(50, 4) * 1000 + 1000
        scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)
        # Se tivesse re-ajustado, a média do teste escalado seria ≈ 0
        # Como usa parâmetros do treino (média ≈ 0.5, std ≈ 0.29), o teste escalado
        # terá valores muito > 0 (≈ 3000 - 1.7 / 0.29 ≈ 3447)
        assert abs(X_te_scaled.mean()) > 100, \
            "Scaler parece ter re-ajustado no teste (leakage)"


# ── DataCleaner ───────────────────────────────────────────────────────────────

class TestDataCleaner:
    def test_duplicatas_removidas_apenas_do_treino(self):
        """DataCleaner remove duplicatas do treino; teste permanece intacto."""
        X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 0, 1, 2])
        cleaner = DataCleaner()
        X_tr, y_tr = cleaner.fit_transform(X.copy(), y.copy())
        assert len(X_tr) < len(X), "Duplicatas não foram removidas do treino"

    def test_imputer_ajustado_no_treino(self):
        """Imputer aprende mediana do treino e aplica no teste sem recalcular."""
        X_train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        X_test  = pd.DataFrame({"a": [np.nan, 10.0]})
        y_train = pd.Series([0, 1, 0, 1, 0])
        cleaner = DataCleaner()
        _, _ = cleaner.fit_transform(X_train.copy(), y_train)
        X_te_clean = cleaner.transform(X_test.copy())
        # Mediana do treino (sem NaN) = mediana([1, 2, 4, 5]) = 3.0
        assert X_te_clean.iloc[0, 0] == pytest.approx(3.0, abs=0.1), \
            "Imputer não usou a mediana do treino para imputar o teste"

    def test_inf_substituido_por_nan(self):
        """Valores Inf devem ser convertidos para NaN antes da imputação."""
        X_train = pd.DataFrame({"a": [1.0, np.inf, 3.0, 4.0]})
        y_train = pd.Series([0, 1, 0, 1])
        cleaner = DataCleaner()
        X_out, _ = cleaner.fit_transform(X_train.copy(), y_train)
        assert not np.isinf(X_out.values).any(), "Inf não foi removido"
        assert not X_out.isnull().any().any(), "NaN restante após imputação"


# ── ClassBalancer (SMOTE) ─────────────────────────────────────────────────────

class TestClassBalancer:
    def test_smote_apenas_no_treino(self):
        """SMOTE deve aumentar o treino; teste não deve passar pelo balancer."""
        # SMOTE exige k_neighbors+1 = 6 amostras mínimas por classe
        X_train = _make_df(120).values.astype(float)
        y_train = np.array([0] * 80 + [1] * 30 + [2] * 10)  # desbalanceado, ≥6/classe
        balancer = ClassBalancer()
        X_bal, y_bal = balancer.fit_resample(X_train, y_train)
        # Após SMOTE: cada classe deve ter aproximadamente o mesmo tamanho
        for cls in [0, 1, 2]:
            count = (y_bal == cls).sum()
            assert count >= 80, f"Classe {cls} com apenas {count} amostras após SMOTE"

    def test_balancer_nao_tem_transform(self):
        """ClassBalancer não deve ter método transform() — só fit_resample()."""
        balancer = ClassBalancer()
        assert not hasattr(balancer, "transform"), \
            "ClassBalancer não deveria ter transform() — SMOTE não é aplicado no teste"

    def test_distribuicao_pos_smote_mais_equilibrada(self):
        """Coeficiente de variação das contagens de classe deve cair após SMOTE."""
        # SMOTE exige k_neighbors+1 = 6 amostras mínimas por classe
        X = _make_df(140).values.astype(float)
        y = np.array([0] * 100 + [1] * 30 + [2] * 10)
        balancer = ClassBalancer()
        _, y_bal = balancer.fit_resample(X, y)
        counts_before = np.array([100, 15, 5])
        counts_after  = np.array([(y_bal == c).sum() for c in [0, 1, 2]])
        cv_before = counts_before.std() / counts_before.mean()
        cv_after  = counts_after.std()  / counts_after.mean()
        assert cv_after < cv_before, "SMOTE não equilibrou as classes"


# ── TopologyFeatureEngineer — sem leakage ─────────────────────────────────────

class TestTopologyFeaturesLeakage:
    def test_fit_transform_usa_labels(self):
        """fit_transform usa y para gerar TTL por classe (documentado)."""
        eng = TopologyFeatureEngineer(random_state=0)
        X = _make_df(90)
        y = pd.Series([0] * 30 + [1] * 30 + [2] * 30)
        X_out = eng.fit_transform(X, y)
        assert "ttl_estimated" in X_out.columns

    def test_transform_nao_usa_labels(self):
        """transform() do teste não aceita labels — API forçada."""
        eng = TopologyFeatureEngineer(random_state=0)
        eng.fit_transform(_make_df(50), _make_y(50))
        # transform() não tem parâmetro y → leakage impossível por interface
        import inspect
        sig = inspect.signature(eng.transform)
        assert "y" not in sig.parameters, \
            "transform() tem parâmetro 'y' — risco de leakage"

    def test_ttl_treino_separado_por_classe(self):
        """Classes diferentes devem ter distribuições de TTL diferentes no treino."""
        eng = TopologyFeatureEngineer(random_state=42)
        X = _make_df(600)
        y = pd.Series([0] * 200 + [1] * 200 + [2] * 200)
        X_out = eng.fit_transform(X, y)
        ttl_benigno  = X_out.loc[y == 0, "ttl_estimated"].mean()
        ttl_externo  = X_out.loc[y == 1, "ttl_estimated"].mean()
        ttl_interno  = X_out.loc[y == 2, "ttl_estimated"].mean()
        # Externo deve ter TTL médio menor que Benigno e Interno
        assert ttl_externo < ttl_benigno, \
            "TTL médio de Externo deve ser menor que Benigno"
        assert ttl_externo < ttl_interno, \
            "TTL médio de Externo deve ser menor que Interno"
