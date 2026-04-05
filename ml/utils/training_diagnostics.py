"""
Ferramentas auxiliares para diagnostico de overfitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, learning_curve

from ml.config import (
    CV_N_SPLITS,
    LEARNING_CURVE_TRAIN_SIZES,
    OUTPUTS_DIR,
    RANDOM_STATE,
)
from ml.evaluation.evaluator import EvaluationResult


class TrainingDiagnostics:
    """Gera graficos de curva de aprendizado e gap de generalizacao."""

    def __init__(self, output_dir: Path | str = OUTPUTS_DIR) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curve(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        label: str,
        scoring: str = "f1_macro",
    ) -> Path:
        cv = StratifiedKFold(
            n_splits=CV_N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        train_sizes, train_scores, valid_scores = learning_curve(
            estimator=clone(estimator),
            X=X,
            y=y,
            train_sizes=LEARNING_CURVE_TRAIN_SIZES,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        valid_mean = valid_scores.mean(axis=1)
        valid_std = valid_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Treino")
        ax.plot(train_sizes, valid_mean, "o-", color="darkorange", label="Validacao CV")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15)
        ax.set_xlabel("Numero de amostras de treino")
        ax.set_ylabel(scoring)
        ax.set_title(f"Curva de aprendizado — {label}")
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
