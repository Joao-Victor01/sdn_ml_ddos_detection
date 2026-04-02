"""
Hyperparameter Tuning com RandomizedSearchCV.

SRP: responsável exclusivamente pela busca de hiperparâmetros.

Regra do curso:
  - Tuning é feito com StratifiedKFold NO CONJUNTO DE TREINO.
  - O test_set NÃO é tocado durante o tuning.
  - RandomizedSearchCV é mais eficiente que GridSearch para espaços grandes.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from ml.config import (
    RANDOM_STATE,
    CV_N_SPLITS,
    CV_SCORING,
    TUNING_N_ITER,
    TUNING_PARAM_DISTRIBUTIONS,
)
from ml.models.mlp_model import build_mlp_from_params


class HyperparameterTuner:
    """
    Busca os melhores hiperparâmetros do MLP via RandomizedSearchCV.

    Uso:
        tuner = HyperparameterTuner()
        best_model = tuner.fit(X_train_bal, y_train_bal)
        print(tuner.best_params_)
        print(tuner.best_cv_score_)
    """

    def __init__(
        self,
        n_iter: int = TUNING_N_ITER,
        param_distributions: dict | None = None,
        scoring: str = CV_SCORING,
        cv_n_splits: int = CV_N_SPLITS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._n_iter             = n_iter
        self._param_distributions = param_distributions or TUNING_PARAM_DISTRIBUTIONS
        self._scoring            = scoring
        self._cv_n_splits        = cv_n_splits
        self._random_state       = random_state
        self._search: RandomizedSearchCV | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> MLPClassifier:
        """
        Executa a busca de hiperparâmetros no conjunto de TREINO.

        Parameters
        ----------
        X_train : np.ndarray — treino balanceado (pós-SMOTE)
        y_train : np.ndarray — alvo balanceado

        Returns
        -------
        best_estimator_: MLPClassifier treinado com os melhores parâmetros,
                         já fitado em todo X_train.
        """
        cv = StratifiedKFold(
            n_splits=self._cv_n_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        base_mlp = MLPClassifier(
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self._random_state,
            verbose=False,
        )

        self._search = RandomizedSearchCV(
            estimator=base_mlp,
            param_distributions=self._param_distributions,
            n_iter=self._n_iter,
            cv=cv,
            scoring=self._scoring,
            n_jobs=-1,
            random_state=self._random_state,
            verbose=1,
            return_train_score=True,
        )

        print(f"\n[HyperparameterTuner] Iniciando busca ({self._n_iter} combinações, "
              f"{self._cv_n_splits}-fold CV)...")
        print(f"  Scoring: {self._scoring}")
        print(f"  Espaço de busca: {list(self._param_distributions.keys())}")

        t0 = time.monotonic()
        self._search.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        print(f"\n[HyperparameterTuner] Busca concluída em {elapsed:.1f}s")
        print(f"  Melhores parâmetros:")
        for k, v in self._search.best_params_.items():
            print(f"    {k}: {v}")
        print(f"  Melhor {self._scoring} (CV): {self._search.best_score_:.4f}")

        return self._search.best_estimator_

    @property
    def best_params_(self) -> dict:
        """Melhores hiperparâmetros encontrados."""
        if self._search is None:
            raise RuntimeError("HyperparameterTuner: chame fit() primeiro.")
        return self._search.best_params_

    @property
    def best_cv_score_(self) -> float:
        """Melhor score CV encontrado."""
        if self._search is None:
            raise RuntimeError("HyperparameterTuner: chame fit() primeiro.")
        return float(self._search.best_score_)

    @property
    def cv_results_(self) -> dict:
        """Resultados completos do RandomizedSearchCV."""
        if self._search is None:
            raise RuntimeError("HyperparameterTuner: chame fit() primeiro.")
        return self._search.cv_results_
