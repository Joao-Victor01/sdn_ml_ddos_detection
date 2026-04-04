"""
Hyperparameter tuning do Random Forest triclasse.

SRP: encapsula exclusivamente o RandomizedSearchCV no conjunto de treino.

Regra absoluta (Aula 11 — boas práticas):
  O test_set NUNCA entra no tuning.
  O tuner usa apenas X_train_bal + y_train_bal via CV interno.

Referência: plano_triclasse_insdn_v4.md, Seção 8.9
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from ml.triclass.config import (
    RANDOM_STATE,
    RF_TUNING_PARAM_DIST,
    RF_TUNING_N_ITER,
    RF_CLASS_WEIGHT,
)
from ml.triclass.models.rf_model import build_baseline_rf


class TriclassTuner:
    """
    RandomizedSearchCV para o Random Forest triclasse.

    Uso:
        tuner    = TriclassTuner()
        rf_best  = tuner.fit(X_train_bal, y_train_bal)
        print(tuner.best_params_)
        print(tuner.best_score_)
    """

    def __init__(
        self,
        param_dist: dict | None = None,
        n_iter: int = RF_TUNING_N_ITER,
        cv_splits: int = 5,
        scoring: str = "f1_macro",
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._param_dist   = param_dist or RF_TUNING_PARAM_DIST
        self._n_iter       = n_iter
        self._cv_splits    = cv_splits
        self._scoring      = scoring
        self._random_state = random_state

        self.best_params_: dict = {}
        self.best_score_: float = 0.0
        self._search: RandomizedSearchCV | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
    ) -> RandomForestClassifier:
        """
        Executa busca aleatória de hiperparâmetros no conjunto de TREINO.

        Parameters
        ----------
        X_train : features balanceadas (pós-SMOTE)
        y_train : labels balanceados (pós-SMOTE)

        Returns
        -------
        RandomForestClassifier com os melhores parâmetros encontrados,
        retreinado sobre todo X_train.
        """
        cv = StratifiedKFold(
            n_splits=self._cv_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        base_rf = RandomForestClassifier(
            class_weight=RF_CLASS_WEIGHT,
            random_state=self._random_state,
            n_jobs=-1,
        )

        self._search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=self._param_dist,
            n_iter=self._n_iter,
            cv=cv,
            scoring=self._scoring,
            random_state=self._random_state,
            n_jobs=-1,
            verbose=1,
        )

        print(f"\n[TriclassTuner] RandomizedSearchCV "
              f"({self._n_iter} iter × {self._cv_splits}-fold)...")
        self._search.fit(X_train, y_train)

        self.best_params_ = self._search.best_params_
        self.best_score_  = float(self._search.best_score_)

        print(f"[TriclassTuner] Melhores parâmetros : {self.best_params_}")
        print(f"[TriclassTuner] Melhor F1 Macro CV  : {self.best_score_:.4f}")

        return self._search.best_estimator_

    @property
    def cv_results(self) -> dict | None:
        """Resultados completos do CV para análise posterior."""
        return self._search.cv_results_ if self._search else None
