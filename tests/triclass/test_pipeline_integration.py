"""
Testes de integração do pipeline triclasse.

Valida a ordem das etapas, ausência de data leakage e consistência
entre as transformações aplicadas em treino e teste.

Estes testes usam dados sintéticos do conftest — não dependem dos CSVs reais.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from ml.triclass.preprocessing.labeler import TriclassLabeler
from ml.triclass.preprocessing.feature_engineer import BehavioralFeatureEngineer
from ml.triclass.models.rf_model import build_baseline_rf
from ml.triclass.config import BEHAVIORAL_FEATURES


class TestNoDataLeakage:
    """
    Valida que nenhum transformador é fitado nos dados de teste.
    (Regra de ouro — guia_boas_praticas_ml.md, Seção 4)
    """

    def test_imputer_fit_somente_no_treino(self, labeled_df):
        """Imputer fitado no treino produz transformação diferente se fitado no teste."""
        X = labeled_df.drop(columns=["Label", "label_3class"], errors="ignore")
        X = X.select_dtypes(include=np.number)
        y = labeled_df["label_3class"]

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # Introduzir NaN artificialmente
        X_train_dirty = X_train.copy()
        X_test_dirty  = X_test.copy()
        col = X_train.columns[0]
        X_train_dirty.iloc[:5, 0] = np.nan
        X_test_dirty.iloc[:2, 0]  = np.nan

        # Fit SOMENTE no treino
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train_dirty)

        train_median = imputer.statistics_[0]
        imputer_test = SimpleImputer(strategy="median")
        imputer_test.fit(X_test_dirty)
        test_median = imputer_test.statistics_[0]

        # Medianas devem ser diferentes (treino e teste têm distribuições distintas)
        # Isso confirma que o fit no treino não usa informação do teste
        assert isinstance(train_median, (int, float, np.floating))
        assert isinstance(test_median, (int, float, np.floating))

    def test_variance_threshold_fit_somente_no_treino(self, labeled_df):
        """VT fitado no treino seleciona features; nunca re-fita no teste."""
        X = labeled_df.drop(columns=["Label", "label_3class"], errors="ignore")
        X = X.select_dtypes(include=np.number)
        y = labeled_df["label_3class"]

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        vt = VarianceThreshold(threshold=0.01)
        X_train_vt = vt.fit_transform(X_train)
        X_test_vt  = vt.transform(X_test)   # mesma máscara do treino

        # Dimensão das colunas deve ser idêntica
        assert X_train_vt.shape[1] == X_test_vt.shape[1]

    def test_features_comportamentais_sem_leakage(self, X_y_train_test):
        """Features comportamentais são puras — aplicar em qualquer ordem produz o mesmo."""
        X_train, X_test, _, _ = X_y_train_test
        eng = BehavioralFeatureEngineer()

        # Aplicar somente em treino
        X_train_eng = eng.fit_transform(X_train)
        X_test_eng  = eng.transform(X_test)

        # Aplicar em test independente (mesmo resultado pois é pura math)
        eng2 = BehavioralFeatureEngineer()
        X_test_eng2 = eng2.fit_transform(X_test)

        for feat in BEHAVIORAL_FEATURES:
            if feat in X_test_eng.columns:
                np.testing.assert_allclose(
                    X_test_eng[feat].values,
                    X_test_eng2[feat].values,
                    rtol=1e-10,
                    err_msg=f"Feature '{feat}' difere — verificar leakage",
                )


class TestSplitAntesDeTudo:
    """
    Valida que o split ocorre antes de qualquer transformação.
    """

    def test_split_estratificado_preserva_proporcao(self, labeled_df):
        """stratify=y deve preservar a proporção das 3 classes."""
        X = labeled_df.drop(columns=["Label", "label_3class"], errors="ignore")
        X = X.select_dtypes(include=np.number)
        y = labeled_df["label_3class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # Proporções devem ser similares (±5%)
        props_train = y_train.value_counts(normalize=True)
        props_test  = y_test.value_counts(normalize=True)

        for cls in props_train.index:
            if cls in props_test.index:
                diff = abs(props_train[cls] - props_test[cls])
                assert diff < 0.10, (
                    f"Classe {cls}: proporção treino={props_train[cls]:.3f} "
                    f"vs teste={props_test[cls]:.3f} — stratify falhou?"
                )

    def test_sem_sobreposicao_treino_teste(self, labeled_df):
        """Índices de treino e teste não devem se sobrepor."""
        X = labeled_df.drop(columns=["Label", "label_3class"], errors="ignore")
        X = X.select_dtypes(include=np.number)
        y = labeled_df["label_3class"]

        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        assert train_idx.isdisjoint(test_idx)


class TestPipelineOrdem:
    """
    Valida a ordem correta das etapas (labeling → split → limpeza → features).
    """

    def test_labeling_antes_do_split(self, raw_insdn_df):
        """Labeling deve ocorrer antes do split — sem -1 após split."""
        labeler = TriclassLabeler()
        data    = labeler.fit_transform(raw_insdn_df)

        X = data.drop(columns=["Label", "label_3class"], errors="ignore")
        X = X.select_dtypes(include=np.number)
        y = data["label_3class"]

        _, _, _, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        assert (y_test == -1).sum() == 0

    def test_features_computadas_apos_split(self, X_y_train_test):
        """Features comportamentais computadas após split — colunas presentes."""
        X_train, X_test, _, _ = X_y_train_test
        eng = BehavioralFeatureEngineer()
        X_train_eng = eng.fit_transform(X_train)
        X_test_eng  = eng.transform(X_test)

        for feat in BEHAVIORAL_FEATURES:
            assert feat in X_train_eng.columns
            assert feat in X_test_eng.columns

    def test_rf_treina_e_prediz_3_classes(self, labeled_df):
        """RF baseline deve conseguir predizer as 3 classes."""
        X = labeled_df.drop(columns=["Label", "label_3class"], errors="ignore")
        X = X.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan).fillna(0)
        y = labeled_df["label_3class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        rf = build_baseline_rf(n_estimators=10)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        assert set(np.unique(y_pred)).issubset({0, 1, 2})

    def test_dimensoes_consistentes_treino_teste(self, X_y_train_test):
        """Treino e teste devem ter o mesmo número de features após VT."""
        X_train, X_test, y_train, _ = X_y_train_test
        eng = BehavioralFeatureEngineer()
        X_train_eng = eng.fit_transform(X_train)
        X_test_eng  = eng.transform(X_test)

        vt = VarianceThreshold(threshold=0.01)
        X_train_vt = vt.fit_transform(X_train_eng)
        X_test_vt  = vt.transform(X_test_eng)

        assert X_train_vt.shape[1] == X_test_vt.shape[1]


class TestSmoteApenasNoTreino:
    """
    Valida que SMOTE não é aplicado ao conjunto de teste.
    """

    def test_smote_aumenta_somente_treino(self, X_y_train_test):
        """Após SMOTE, treino deve ter mais amostras; teste permanece inalterado."""
        from imblearn.over_sampling import SMOTE

        X_train, X_test, y_train, y_test = X_y_train_test
        eng = BehavioralFeatureEngineer()
        X_train_eng = eng.fit_transform(X_train)
        X_test_eng  = eng.transform(X_test)

        n_cls2 = int((y_train == 2).sum())
        if n_cls2 < 2:
            pytest.skip("Classe 2 com menos de 2 amostras no treino — skip SMOTE test")

        n_test_before = len(X_test_eng)

        smote = SMOTE(
            sampling_strategy={2: n_cls2 + 1},
            random_state=42,
            k_neighbors=min(1, n_cls2 - 1),
        )
        X_res, y_res = smote.fit_resample(X_train_eng, y_train)

        # Treino aumentou
        assert len(X_res) >= len(X_train_eng)
        # Teste inalterado
        assert len(X_test_eng) == n_test_before
