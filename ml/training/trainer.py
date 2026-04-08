"""
Treinamento e validacao cruzada de classificadores supervisionados.

O preprocessamento permanece igual para todos os modelos; o que varia
é apenas o estimador passado para este módulo.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from ml.config import CV_N_SPLITS, OUTPUTS_DIR, RANDOM_STATE
from ml.features.selector import FeatureSelector
from ml.preprocessing.balancer import ClassBalancer
from ml.preprocessing.cleaner import DataCleaner
from ml.preprocessing.scaler import FeatureScaler
from ml.utils.plotting import get_pyplot


def fit_fold_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    base_model: ClassifierMixin,
) -> tuple[ClassifierMixin, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Ajusta todo o pipeline de preprocessamento dentro de uma dobra.

    Isso evita vazamento de dados: imputação, seleção, escalonamento e
    SMOTE são aprendidos apenas com o treino da dobra.
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

    selector = FeatureSelector()
    X_train_sel = selector.fit_transform(X_train_clean, y_train_clean)
    X_valid_sel = selector.transform(X_valid_clean)

    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_valid_scaled = scaler.transform(X_valid_sel)

    balancer = ClassBalancer(random_state=random_state)
    X_train_bal, y_train_bal = balancer.fit_resample(X_train_scaled, y_train_clean)

    model = clone(base_model)
    model.fit(X_train_bal, np.asarray(y_train_bal))

    return (
        model,
        X_train_scaled,
        np.asarray(y_train_clean),
        X_valid_scaled,
        np.asarray(y_valid_clean),
    )


class ModelTrainer:
    """Treina o baseline e executa validacao cruzada limpa."""

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
        model: ClassifierMixin,
        model_name: str,
        label: str,
        supports_loss_curve: bool = False,
    ) -> ClassifierMixin:
        print(f"\n[ModelTrainer] Iniciando treinamento do {model_name}...")
        self._print_model_details(model, X_train)

        t0 = time.monotonic()
        model.fit(X_train, y_train)
        elapsed = time.monotonic() - t0

        print(f"\n[ModelTrainer] Treinamento concluido em {elapsed:.1f}s")
        if hasattr(model, "n_iter_"):
            print(f"  Iteracoes/epocas executadas : {getattr(model, 'n_iter_')}")
        if hasattr(model, "loss_"):
            print(f"  Loss final                  : {getattr(model, 'loss_'):.6f}")

        if self._save_plots and supports_loss_curve:
            self._plot_loss_curve(model, label=label, model_name=model_name)

        return model

    def cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        base_model: ClassifierMixin,
        model_name: str,
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
            f"no conjunto de treino para {model_name}:"
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
                    base_model=base_model,
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

    def _print_model_details(
        self,
        model: ClassifierMixin,
        X_train: np.ndarray,
    ) -> None:
        params = model.get_params(deep=False)
        print(f"  Classe      : {model.__class__.__name__}")
        print(f"  Shape treino: {X_train.shape}")

        tracked_keys = (
            "hidden_layer_sizes",
            "solver",
            "activation",
            "alpha",
            "max_iter",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        )
        for key in tracked_keys:
            if key in params:
                print(f"  {key:<12}: {params[key]}")

    def _plot_loss_curve(
        self,
        model: ClassifierMixin,
        *,
        label: str,
        model_name: str,
    ) -> None:
        if not hasattr(model, "loss_curve_"):
            print(
                f"[ModelTrainer] {model_name} nao expoe loss_curve_; "
                "plot pulado."
            )
            return

        try:
            plt = get_pyplot()
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
            ax.set_title(f"Curva de convergencia — {model_name} {label}")
            ax.legend()
            ax.grid(alpha=0.25)
            plt.tight_layout()
            path = self._output_dir / f"loss_curve_{label}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelTrainer] Curva de loss salva em {path}")
        except Exception as exc:
            print(f"[ModelTrainer] Nao foi possivel salvar a curva de loss: {exc}")
