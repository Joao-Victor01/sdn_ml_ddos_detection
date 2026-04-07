"""
Hyperparameter tuning com RandomizedSearchCV para classificacao multiclasse.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from ml.config import (
    CV_N_SPLITS,
    CV_SCORING,
    MLP_VAL_FRACTION,
    RANDOM_STATE,
    TUNING_N_ITER,
    TUNING_PARAM_DISTRIBUTIONS,
)


class HyperparameterTuner:
    """Busca os melhores hiperparametros do MLP via RandomizedSearchCV."""

    def __init__(
        self,
        n_iter: int = TUNING_N_ITER,
        param_distributions: dict | None = None,
        scoring: str = CV_SCORING,
        cv_n_splits: int = CV_N_SPLITS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._n_iter = n_iter
        self._param_distributions = param_distributions or TUNING_PARAM_DISTRIBUTIONS
        self._scoring = scoring
        self._cv_n_splits = cv_n_splits
        self._random_state = random_state
        self._search: RandomizedSearchCV | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> MLPClassifier:
        # O tuning entra depois do baseline: a ideia é procurar melhorias sem trocar todo o pipeline.
        # StratifiedKFold dentro do tuning garante que cada combinação seja avaliada
        # nas mesmas proporções de classes — sem isso, uma dobra desequilibrada
        # poderia favorecer combinações de hiperparâmetros que não generalizam bem
        cv = StratifiedKFold(
            n_splits=self._cv_n_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        # Modelo base com parâmetros fixos — só os parâmetros do espaço de busca variam
        base_mlp = MLPClassifier(
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=MLP_VAL_FRACTION,
            random_state=self._random_state,
            verbose=False,
        )

        # RandomizedSearchCV testa N combinações aleatórias do espaço definido em config.py
        # Mais rápido que GridSearchCV (que testa todas as combinações) e geralmente eficaz
        self._search = RandomizedSearchCV(
            estimator=base_mlp,
            param_distributions=self._param_distributions,
            n_iter=self._n_iter,      # quantas combinações aleatórias testar
            cv=cv,
            scoring=self._scoring,
            n_jobs=-1,                # usa todos os núcleos disponíveis
            random_state=self._random_state,
            verbose=1,
            return_train_score=True,  # útil para diagnóstico de overfitting no tuning
        )

        print(
            f"\n[HyperparameterTuner] Iniciando busca ({self._n_iter} combinacoes, "
            f"{self._cv_n_splits}-fold CV)..."
        )
        print(f"  Scoring: {self._scoring}")
        print(f"  Espaco de busca: {list(self._param_distributions.keys())}")

        t0 = time.monotonic()
        self._search.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        print(f"\n[HyperparameterTuner] Busca concluida em {elapsed:.1f}s")
        print("  Melhores parametros:")
        for key, value in self._search.best_params_.items():
            print(f"    {key}: {value}")
        print(f"  Melhor {self._scoring} (CV): {self._search.best_score_:.4f}")

        # O sklearn já devolve o melhor estimador reajustado com todo o conjunto de treino.
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
