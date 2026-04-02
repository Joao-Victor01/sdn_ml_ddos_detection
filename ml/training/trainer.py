"""
Treinamento do modelo MLP e validação cruzada.

SRP: este módulo gerencia exclusivamente o ciclo de treino e a
validação cruzada no conjunto de treino.

Regra do curso: validação cruzada é feita SOMENTE no conjunto de treino.
O test_set é reservado para avaliação final — nunca é usado aqui.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from ml.config import (
    RANDOM_STATE,
    CV_N_SPLITS,
    CV_SCORING,
    OUTPUTS_DIR,
)
from ml.models.mlp_model import build_baseline_mlp


class ModelTrainer:
    """
    Treina o MLP baseline e avalia com validação cruzada estratificada.

    Uso:
        trainer = ModelTrainer()
        model   = trainer.train(X_train_bal, y_train_bal)
        trainer.cross_validate(X_train_bal, y_train_bal)
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

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: MLPClassifier | None = None,
    ) -> MLPClassifier:
        """
        Treina o MLP no conjunto de treino balanceado.

        Parameters
        ----------
        X_train : np.ndarray — features escalonadas e balanceadas (pós-SMOTE)
        y_train : np.ndarray — alvo balanceado
        model   : MLPClassifier opcional; se None, usa o baseline configurado

        Returns
        -------
        MLPClassifier treinado.
        """
        if model is None:
            model = build_baseline_mlp(self._random_state)

        print("\n[ModelTrainer] Iniciando treinamento MLP baseline...")
        print(f"  Arquitetura : {model.hidden_layer_sizes}")
        print(f"  Solver      : {model.solver}")
        print(f"  Ativação    : {model.activation}")
        print(f"  max_iter    : {model.max_iter}")
        print(f"  Shape treino: {X_train.shape}")

        t0 = time.monotonic()
        model.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        print(f"\n[ModelTrainer] Treinamento concluído em {elapsed:.1f}s")
        print(f"  Épocas executadas : {model.n_iter_}")
        print(f"  Loss final        : {model.loss_:.6f}")

        if self._save_plots:
            self._plot_loss_curve(model)

        return model

    def cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, tuple[float, float]]:
        """
        Avalia o baseline com StratifiedKFold no conjunto de TREINO.

        NUNCA usa X_test — a validação cruzada é exclusivamente no treino.

        Returns
        -------
        dict: métrica → (mean, std)
        """
        cv = StratifiedKFold(
            n_splits=self._cv_n_splits,
            shuffle=True,
            random_state=self._random_state,
        )

        metrics = ["accuracy", "f1", "precision", "recall"]
        results: dict[str, tuple[float, float]] = {}

        print(f"\n[ModelTrainer] Validação Cruzada ({self._cv_n_splits}-fold) — conjunto de treino:")
        print("-" * 50)

        for metric in metrics:
            scores = cross_val_score(
                build_baseline_mlp(self._random_state),
                X_train,
                y_train,
                cv=cv,
                scoring=metric,
                n_jobs=-1,
            )
            mean, std = float(scores.mean()), float(scores.std())
            results[metric] = (mean, std)
            print(f"  {metric:<12}: {mean:.4f} ± {std:.4f}")

        print("-" * 50)
        return results

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _plot_loss_curve(self, model: MLPClassifier) -> None:
        """Salva a curva de convergência do modelo em outputs/."""
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(model.loss_curve_, label="Loss de Treino", color="steelblue")
            if hasattr(model, "validation_scores_") and model.validation_scores_:
                ax.plot(
                    model.validation_scores_,
                    label="Score de Validação Interna",
                    color="orange",
                    linestyle="--",
                )

            ax.set_xlabel("Épocas")
            ax.set_ylabel("Loss / Score")
            ax.set_title("Curva de Convergência — MLP Baseline (InSDN8)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / "loss_curve_baseline.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelTrainer] Curva de loss salva em {OUTPUTS_DIR}/loss_curve_baseline.png")
        except Exception as e:
            print(f"[ModelTrainer] Não foi possível salvar curva de loss: {e}")
