"""
Ferramentas auxiliares para diagnostico de overfitting.
"""

from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.config import (
    CV_N_SPLITS,
    LEARNING_CURVE_TRAIN_SIZES,
    OUTPUTS_DIR,
    RANDOM_STATE,
)
from ml.evaluation.evaluator import EvaluationResult
from ml.training.trainer import fit_fold_pipeline
from ml.utils.plotting import get_pyplot


class TrainingDiagnostics:
    """Gera graficos de curva de aprendizado e gap de generalizacao."""

    def __init__(self, output_dir: Path | str = OUTPUTS_DIR) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        label: str,
        scoring: str = "f1_macro",
        estimator: ClassifierMixin | None = None,
    ) -> Path:
        if estimator is None:
            raise ValueError(
                "TrainingDiagnostics: informe um estimador baseline para a curva de aprendizado."
            )
        if scoring != "f1_macro":
            raise ValueError(
                "TrainingDiagnostics: atualmente a curva de aprendizado suporta apenas "
                "scoring='f1_macro'."
            )

        # A curva segue a mesma ideia da CV limpa: cada dobra reconstrói o pipeline do zero.
        cv = StratifiedKFold(
            n_splits=CV_N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        train_sizes_abs = self._resolve_train_sizes(len(X))
        train_scores = np.zeros((len(train_sizes_abs), CV_N_SPLITS), dtype=float)
        valid_scores = np.zeros((len(train_sizes_abs), CV_N_SPLITS), dtype=float)

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_fold_train = X.iloc[train_idx].reset_index(drop=True)
            y_fold_train = y.iloc[train_idx].reset_index(drop=True)
            X_fold_valid = X.iloc[valid_idx].reset_index(drop=True)
            y_fold_valid = y.iloc[valid_idx].reset_index(drop=True)

            for size_idx, train_size in enumerate(train_sizes_abs):
                # Aqui simulamos "e se eu tivesse menos dados de treino?" sem perder a proporção das classes.
                X_subset, y_subset = self._sample_stratified_subset(
                    X_fold_train,
                    y_fold_train,
                    train_size,
                )
                # Silenciamos os prints internos para o gráfico não virar uma parede de logs.
                with redirect_stdout(io.StringIO()):
                    (
                        model,
                        X_train_scaled,
                        y_train_arr,
                        X_valid_scaled,
                        y_valid_arr,
                    ) = fit_fold_pipeline(
                        X_subset,
                        y_subset,
                        X_fold_valid,
                        y_fold_valid,
                        random_state=RANDOM_STATE,
                        base_model=estimator,
                    )

                train_pred = model.predict(X_train_scaled)
                valid_pred = model.predict(X_valid_scaled)
                train_scores[size_idx, fold_idx] = f1_score(
                    y_train_arr,
                    train_pred,
                    average="macro",
                    zero_division=0,
                )
                valid_scores[size_idx, fold_idx] = f1_score(
                    y_valid_arr,
                    valid_pred,
                    average="macro",
                    zero_division=0,
                )

        # Depois juntamos as dobras em média + desvio para mostrar tendência, não sorte de um fold só.
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        valid_mean = valid_scores.mean(axis=1)
        valid_std = valid_scores.std(axis=1)

        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue", label="Treino")
        ax.plot(train_sizes_abs, valid_mean, "o-", color="darkorange", label="Validacao CV")
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
        )
        ax.fill_between(
            train_sizes_abs,
            valid_mean - valid_std,
            valid_mean + valid_std,
            alpha=0.15,
        )
        ax.set_xlabel("Numero de amostras de treino")
        ax.set_ylabel(scoring)
        ax.set_title(f"Curva de aprendizado limpa — {label}")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()

        path = self._output_dir / f"learning_curve_{label}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[TrainingDiagnostics] Curva de aprendizado salva em {path}")
        return path

    def plot_generalization_gap(
        self,
        train_result: EvaluationResult,
        test_result: EvaluationResult,
        label: str,
    ) -> Path:
        metrics = {
            "Accuracy": (train_result.accuracy, test_result.accuracy),
            "Bal.Acc": (train_result.balanced_accuracy, test_result.balanced_accuracy),
            "F1 Macro": (train_result.f1_macro, test_result.f1_macro),
            "F1 Weighted": (train_result.f1_weighted, test_result.f1_weighted),
            "MCC": (train_result.mcc, test_result.mcc),
            "ROC-AUC": (train_result.roc_auc_ovr_macro, test_result.roc_auc_ovr_macro),
        }

        names = list(metrics.keys())
        train_vals = np.array([values[0] for values in metrics.values()])
        test_vals = np.array([values[1] for values in metrics.values()])
        x = np.arange(len(names))
        width = 0.35

        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, train_vals, width, label="Treino", color="steelblue")
        ax.bar(x + width / 2, test_vals, width, label="Teste", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Gap de generalizacao — {label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        path = self._output_dir / f"generalization_gap_{label}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[TrainingDiagnostics] Gap de generalizacao salvo em {path}")
        return path

    def save_gap_report(
        self,
        train_result: EvaluationResult,
        test_result: EvaluationResult,
        label: str,
    ) -> Path:
        report = {
            "label": label,
            "train": {
                "accuracy": train_result.accuracy,
                "balanced_accuracy": train_result.balanced_accuracy,
                "f1_macro": train_result.f1_macro,
                "f1_weighted": train_result.f1_weighted,
                "mcc": train_result.mcc,
                "roc_auc_ovr_macro": train_result.roc_auc_ovr_macro,
            },
            "test": {
                "accuracy": test_result.accuracy,
                "balanced_accuracy": test_result.balanced_accuracy,
                "f1_macro": test_result.f1_macro,
                "f1_weighted": test_result.f1_weighted,
                "mcc": test_result.mcc,
                "roc_auc_ovr_macro": test_result.roc_auc_ovr_macro,
            },
            "gap": {
                "accuracy_gap": train_result.accuracy - test_result.accuracy,
                "balanced_accuracy_gap": train_result.balanced_accuracy - test_result.balanced_accuracy,
                "f1_macro_gap": train_result.f1_macro - test_result.f1_macro,
                "f1_weighted_gap": train_result.f1_weighted - test_result.f1_weighted,
                "mcc_gap": train_result.mcc - test_result.mcc,
                "roc_auc_gap": train_result.roc_auc_ovr_macro - test_result.roc_auc_ovr_macro,
            },
        }
        path = self._output_dir / f"generalization_report_{label}.json"
        with open(path, "w") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        print(f"[TrainingDiagnostics] Relatorio de generalizacao salvo em {path}")
        return path

    def _resolve_train_sizes(self, n_samples: int) -> list[int]:
        sizes: list[int] = []
        for size in LEARNING_CURVE_TRAIN_SIZES:
            # Aceita fração (0.1, 0.25...) ou tamanho absoluto, e converte tudo para número de linhas.
            if isinstance(size, float):
                resolved = max(2, int(round(n_samples * size)))
            else:
                resolved = int(size)
            sizes.append(min(resolved, n_samples))
        return sorted(set(sizes))

    def _sample_stratified_subset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if train_size >= len(X):
            return X.reset_index(drop=True), y.reset_index(drop=True)

        # O subconjunto continua estratificado para a learning curve não ficar torta por acaso.
        X_subset, _, y_subset, _ = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=y,
        )
        return X_subset.reset_index(drop=True), y_subset.reset_index(drop=True)
