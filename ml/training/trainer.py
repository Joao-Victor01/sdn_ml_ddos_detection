"""
Treinamento do modelo MLP e validacao cruzada.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from ml.config import CV_N_SPLITS, OUTPUTS_DIR, RANDOM_STATE
from ml.features.selector import FeatureSelector
from ml.models.mlp_model import build_baseline_mlp
from ml.preprocessing.balancer import ClassBalancer
from ml.preprocessing.cleaner import DataCleaner
from ml.preprocessing.scaler import FeatureScaler


def fit_fold_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    base_model: MLPClassifier | None = None,
) -> tuple[MLPClassifier, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Ajusta todo o pipeline de preprocessamento dentro de uma dobra.

    Isso evita que limpeza, selecao, escalonamento e SMOTE "vejam" a parte
    de validacao antes da hora.
    """
    cleaner = DataCleaner()
    X_train_clean, y_train_clean = cleaner.fit_transform(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
    )
    X_valid_clean, y_valid_clean = cleaner.transform(
        X_valid.reset_index(drop=True),
        y_valid.reset_index(drop=True),
    )

    selector = FeatureSelector(save_plots=False, compute_importance=False)
    X_train_sel = selector.fit_transform(X_train_clean, y_train_clean)
    X_valid_sel = selector.transform(X_valid_clean)

    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_valid_scaled = scaler.transform(X_valid_sel)

    balancer = ClassBalancer(random_state=random_state)
    X_train_bal, y_train_bal = balancer.fit_resample(X_train_scaled, y_train_clean)

    model = clone(base_model) if base_model is not None else build_baseline_mlp(random_state)
    model.fit(X_train_bal, np.asarray(y_train_bal))

    return (
        model,
        X_train_scaled,
        np.asarray(y_train_clean),
        X_valid_scaled,
        np.asarray(y_valid_clean),
    )


class ModelTrainer:
    """Treina o MLP baseline e executa validacao cruzada no treino."""

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        cv_n_splits: int = CV_N_SPLITS,
        save_plots: bool = True,
        output_dir: Path | str = OUTPUTS_DIR,
    ) -> None:
        self._random_state = random_state
        self._cv_n_splits = cv_n_splits
        self._save_plots = save_plots
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

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
        X_train: pd.DataFrame,
        y_train: pd.Series,
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
            f"\n[ModelTrainer] Validacao cruzada limpa ({self._cv_n_splits}-fold) "
            "no conjunto de treino:"
        )
        history: dict[str, list[float]] = {metric: [] for metric in scoring}

        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
            y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
            X_fold_valid = X_train.iloc[valid_idx].reset_index(drop=True)
            y_fold_valid = y_train.iloc[valid_idx].reset_index(drop=True)

            with redirect_stdout(io.StringIO()):
                model, _, _, X_valid_scaled, y_valid_arr = fit_fold_pipeline(
                    X_fold_train,
                    y_fold_train,
                    X_fold_valid,
                    y_fold_valid,
                    random_state=self._random_state,
                )

            y_pred = model.predict(X_valid_scaled)
            history["accuracy"].append(float(accuracy_score(y_valid_arr, y_pred)))
            history["balanced_accuracy"].append(
                float(balanced_accuracy_score(y_valid_arr, y_pred))
            )
            history["f1_macro"].append(
                float(f1_score(y_valid_arr, y_pred, average="macro", zero_division=0))
            )
            history["precision_macro"].append(
                float(
                    precision_score(
                        y_valid_arr,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            history["recall_macro"].append(
                float(recall_score(y_valid_arr, y_pred, average="macro", zero_division=0))
            )

        results: dict[str, tuple[float, float]] = {}
        for metric in scoring:
            values = np.array(history[metric], dtype=float)
            mean = float(values.mean())
            std = float(values.std())
            results[metric] = (mean, std)
            print(f"  {metric:<18}: {mean:.4f} +/- {std:.4f}")

        return results

    def _plot_loss_curve(self, model: MLPClassifier, label: str) -> None:
        try:
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
            path = self._output_dir / f"loss_curve_{label}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelTrainer] Curva de loss salva em {path}")
        except Exception as exc:
            print(f"[ModelTrainer] Nao foi possivel salvar a curva de loss: {exc}")
