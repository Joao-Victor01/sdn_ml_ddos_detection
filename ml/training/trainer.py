"""
Treinamento do modelo MLP e validacao cruzada.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from ml.config import CV_N_SPLITS, OUTPUTS_DIR, RANDOM_STATE
from ml.models.mlp_model import build_baseline_mlp


class ModelTrainer:
    """Treina o MLP baseline e executa validacao cruzada no treino."""

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        cv_n_splits: int = CV_N_SPLITS,
        save_plots: bool = True,
    ) -> None:
        self._random_state = random_state
        self._cv_n_splits = cv_n_splits
        self._save_plots = save_plots

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: MLPClassifier | None = None,
        label: str = "baseline",
    ) -> MLPClassifier:
        if model is None:
            model = build_baseline_mlp(self._random_state)

        print("\n[ModelTrainer] Iniciando treinamento do MLP...")
        print(f"  Arquitetura : {model.hidden_layer_sizes}")
        print(f"  Solver      : {model.solver}")
        print(f"  Ativacao    : {model.activation}")
        print(f"  Alpha       : {model.alpha}")
        print(f"  max_iter    : {model.max_iter}")
        print(f"  Shape treino: {X_train.shape}")

        t0 = time.monotonic()
        model.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        print(f"\n[ModelTrainer] Treinamento concluido em {elapsed:.1f}s")
        print(f"  Epocas executadas : {model.n_iter_}")
        print(f"  Loss final        : {model.loss_:.6f}")

        if self._save_plots:
            self._plot_loss_curve(model, label=label)

        return model

    def cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, tuple[float, float]]:
        cv = StratifiedKFold(
            n_splits=self._cv_n_splits,
            shuffle=True,
            random_state=self._random_state,
        )
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
        }

        print(
            f"\n[ModelTrainer] Validacao cruzada ({self._cv_n_splits}-fold) "
            "no conjunto de treino:"
        )
        scores = cross_validate(
            build_baseline_mlp(self._random_state),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results: dict[str, tuple[float, float]] = {}
        for metric in scoring:
            metric_key = f"test_{metric}"
            mean = float(scores[metric_key].mean())
            std = float(scores[metric_key].std())
            results[metric] = (mean, std)
            print(f"  {metric:<18}: {mean:.4f} +/- {std:.4f}")

        return results

    def _plot_loss_curve(self, model: MLPClassifier, label: str) -> None:
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(model.loss_curve_, label="Loss de treino", color="steelblue")

            if hasattr(model, "validation_scores_") and model.validation_scores_:
                ax.plot(
                    model.validation_scores_,
                    label="Score de validacao interna",
                    color="darkorange",
                    linestyle="--",
                )

            ax.set_xlabel("Epocas")
            ax.set_ylabel("Loss / Score")
            ax.set_title(f"Curva de convergencia — MLP {label}")
            ax.legend()
            ax.grid(alpha=0.25)
            plt.tight_layout()
            path = OUTPUTS_DIR / f"loss_curve_{label}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelTrainer] Curva de loss salva em {path}")
        except Exception as exc:
            print(f"[ModelTrainer] Nao foi possivel salvar a curva de loss: {exc}")
