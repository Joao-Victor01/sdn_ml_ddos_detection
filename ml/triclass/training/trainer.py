"""
Treinamento e validação cruzada dos modelos RF e MLP triclasse.

SRP: gerencia o ciclo treino + CV estratificado.

Regra do curso:
  - Validação cruzada SOMENTE no treino (nunca no test_set).
  - MLP obrigatoriamente dentro de Pipeline(StandardScaler → MLP) para
    evitar leakage do scaler no CV (Aula 5).

Referência: plano_triclasse_insdn_v4.md, Seção 8.8
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.triclass.config import (
    RANDOM_STATE,
    CV_N_SPLITS,
    CV_SCORING,
    MLP_HIDDEN_LAYERS,
    MLP_ACTIVATION,
    MLP_SOLVER,
    MLP_MAX_ITER,
    MLP_EARLY_STOP,
    MLP_N_ITER_NO_CHG,
    OUTPUTS_TRICLASS,
)
from ml.triclass.models.rf_model import build_baseline_rf


def build_mlp_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Constrói Pipeline(StandardScaler → MLP).

    O scaler dentro do Pipeline garante que o fit do scaler ocorra
    somente nos folds de treino durante o CV — zero leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation=MLP_ACTIVATION,
            solver=MLP_SOLVER,
            max_iter=MLP_MAX_ITER,
            early_stopping=MLP_EARLY_STOP,
            n_iter_no_change=MLP_N_ITER_NO_CHG,
            random_state=random_state,
        )),
    ])


class TriclassTrainer:
    """
    Treina RF e MLP triclasse com validação cruzada estratificada.

    Uso:
        trainer = TriclassTrainer()
        rf, mlp = trainer.train(X_train_bal, y_train_bal)
        cv_rf   = trainer.cross_validate_rf(X_train_bal, y_train_bal)
        cv_mlp  = trainer.cross_validate_mlp(X_train_bal, y_train_bal)
    """

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        cv_n_splits: int = CV_N_SPLITS,
        cv_scoring: str = CV_SCORING,
        save_plots: bool = True,
    ) -> None:
        self._random_state = random_state
        self._cv_n_splits  = cv_n_splits
        self._cv_scoring   = cv_scoring
        self._save_plots   = save_plots

    # ── API pública ────────────────────────────────────────────────────────────

    def train_rf(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        rf=None,
    ):
        """Treina Random Forest no treino balanceado."""
        if rf is None:
            rf = build_baseline_rf(random_state=self._random_state)

        print("\n[TriclassTrainer] Treinando Random Forest baseline...")
        print(f"  n_estimators : {rf.n_estimators}")
        print(f"  class_weight : {rf.class_weight}")
        print(f"  Shape treino : {np.array(X_train).shape}")

        t0 = time.monotonic()
        rf.fit(X_train, y_train)
        elapsed = time.monotonic() - t0
        print(f"[TriclassTrainer] RF treinado em {elapsed:.1f}s")
        return rf

    def train_mlp(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        mlp_pipe: Pipeline | None = None,
    ) -> Pipeline:
        """Treina Pipeline(StandardScaler → MLP) no treino balanceado."""
        if mlp_pipe is None:
            mlp_pipe = build_mlp_pipeline(random_state=self._random_state)

        print("\n[TriclassTrainer] Treinando MLP (Pipeline)...")
        mlp = mlp_pipe.named_steps["mlp"]
        print(f"  Arquitetura  : {mlp.hidden_layer_sizes}")
        print(f"  Shape treino : {np.array(X_train).shape}")

        t0 = time.monotonic()
        mlp_pipe.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        mlp_fitted = mlp_pipe.named_steps["mlp"]
        print(f"[TriclassTrainer] MLP treinado em {elapsed:.1f}s")
        print(f"  Épocas : {mlp_fitted.n_iter_}")
        print(f"  Loss   : {mlp_fitted.loss_:.6f}")

        if self._save_plots:
            self._plot_mlp_loss(mlp_fitted)

        return mlp_pipe

    def cross_validate_rf(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
    ) -> dict[str, tuple[float, float]]:
        """
        Validação cruzada do RF no conjunto de TREINO.

        Retorna dict: métrica → (mean, std)
        """
        return self._cross_validate(
            build_baseline_rf(random_state=self._random_state),
            X_train, y_train,
            label="Random Forest",
        )

    def cross_validate_mlp(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
    ) -> dict[str, tuple[float, float]]:
        """
        Validação cruzada do MLP no conjunto de TREINO.

        Pipeline(StandardScaler → MLP) garante que o scaler seja fitado
        somente nos folds de treino — evita leakage no CV.
        """
        return self._cross_validate(
            build_mlp_pipeline(random_state=self._random_state),
            X_train, y_train,
            label="MLP Pipeline",
        )

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _cross_validate(
        self,
        estimator,
        X_train,
        y_train,
        label: str,
    ) -> dict[str, tuple[float, float]]:
        cv = StratifiedKFold(
            n_splits=self._cv_n_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        # F1 macro é a métrica primária para triclasse desbalanceada
        scoring_metrics = ["f1_macro", "accuracy"]
        results: dict[str, tuple[float, float]] = {}

        print(f"\n[TriclassTrainer] CV {self._cv_n_splits}-fold — {label} (treino):")
        print("-" * 55)

        for metric in scoring_metrics:
            scores = cross_val_score(
                estimator, X_train, y_train,
                cv=cv, scoring=metric, n_jobs=-1,
            )
            mean, std = float(scores.mean()), float(scores.std())
            results[metric] = (mean, std)
            print(f"  {metric:<20}: {mean:.4f} ± {std:.4f}")

        print("-" * 55)
        return results

    def _plot_mlp_loss(self, mlp: MLPClassifier) -> None:
        try:
            import matplotlib.pyplot as plt
            OUTPUTS_TRICLASS.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(mlp.loss_curve_, label="Loss de Treino", color="steelblue")
            if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
                ax.plot(
                    mlp.validation_scores_,
                    label="Score Validação Interna",
                    color="orange", linestyle="--",
                )
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Loss / Score")
            ax.set_title("Curva de Convergência — MLP Triclasse (InSDN)")
            ax.legend()
            plt.tight_layout()
            path = OUTPUTS_TRICLASS / "mlp_loss_curve.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[TriclassTrainer] Curva MLP salva → {path.name}")
        except Exception as e:
            print(f"[TriclassTrainer] Não foi possível salvar curva MLP: {e}")
