"""
Hyperparameter tuning genérico via RandomizedSearchCV.
"""

from __future__ import annotations

import time

from sklearn.base import ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from ml.config import CV_N_SPLITS, CV_SCORING, RANDOM_STATE, TUNING_N_ITER


class HyperparameterTuner:
    """Busca hiperparâmetros para qualquer classificador compatível com sklearn."""

    def __init__(
        self,
        n_iter: int = TUNING_N_ITER,
        scoring: str = CV_SCORING,
        cv_n_splits: int = CV_N_SPLITS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._n_iter = n_iter
        self._scoring = scoring
        self._cv_n_splits = cv_n_splits
        self._random_state = random_state
        self._search: RandomizedSearchCV | None = None

    def fit(
        self,
        X_train,
        y_train,
        *,
        estimator: ClassifierMixin,
        param_distributions: dict,
        model_name: str,
    ) -> ClassifierMixin:
        cv = StratifiedKFold(
            n_splits=self._cv_n_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        self._search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=self._n_iter,
            cv=cv,
            scoring=self._scoring,
            n_jobs=-1,
            random_state=self._random_state,
            verbose=1,
            return_train_score=True,
        )

        print(
            f"\n[HyperparameterTuner] Iniciando busca para {model_name} "
            f"({self._n_iter} combinacoes, {self._cv_n_splits}-fold CV)..."
        )
        print(f"  Scoring: {self._scoring}")
        print(f"  Espaco de busca: {list(param_distributions.keys())}")

        t0 = time.monotonic()
        self._search.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        print(f"\n[HyperparameterTuner] Busca concluida em {elapsed:.1f}s")
        print("  Melhores parametros:")
        for key, value in self._search.best_params_.items():
            print(f"    {key}: {value}")
        print(f"  Melhor {self._scoring} (CV): {self._search.best_score_:.4f}")

        return self._search.best_estimator_

    @property
    def best_params_(self) -> dict:
        if self._search is None:
            raise RuntimeError("HyperparameterTuner: chame fit() primeiro.")
        return self._search.best_params_

    @property
    def best_cv_score_(self) -> float:
        if self._search is None:
            raise RuntimeError("HyperparameterTuner: chame fit() primeiro.")
        return float(self._search.best_score_)

    @property
    def cv_results_(self) -> dict:
        if self._search is None:
            raise RuntimeError("HyperparameterTuner: chame fit() primeiro.")
        return self._search.cv_results_
