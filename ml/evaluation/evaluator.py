"""
Avaliacoes e visualizacoes para classificacao multiclasse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import OUTPUTS_DIR, TARGET_NAMES
from ml.utils.plotting import get_pyplot


@dataclass
class EvaluationResult:
    """Resultado imutavel de uma avaliacao."""

    label: str
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    mcc: float
    gm: float
    roc_auc_ovr_macro: float
    confusion_matrix: list[list[int]]
    class_names: list[str]
    classification_report: str = field(default="", repr=False)

    def print_summary(self) -> None:
        width = 72
        print("\n" + "=" * width)
        print(f"{f'RESULTADOS — {self.label}':^{width}}")
        print("=" * width)
        print(f"  Accuracy            : {self.accuracy:.4f}  ({self.accuracy*100:.2f}%)")
        print(
            f"  Balanced Accuracy   : {self.balanced_accuracy:.4f}  "
            f"({self.balanced_accuracy*100:.2f}%)"
        )
        print(
            f"  Precision Macro     : {self.precision_macro:.4f}  "
            f"({self.precision_macro*100:.2f}%)"
        )
        print(
            f"  Recall Macro        : {self.recall_macro:.4f}  "
            f"({self.recall_macro*100:.2f}%)"
        )
        print(f"  F1 Macro            : {self.f1_macro:.4f}  ({self.f1_macro*100:.2f}%)")
        print(f"  F1 Weighted         : {self.f1_weighted:.4f}  ({self.f1_weighted*100:.2f}%)")
        print(f"  MCC                 : {self.mcc:.4f}")
        print(f"  Geometric Mean      : {self.gm:.4f}")
        print(f"  ROC-AUC OVR Macro   : {self.roc_auc_ovr_macro:.4f}")
        print("=" * width)


class ModelEvaluator:
    """Avalia modelos multiclasse no conjunto de teste ou treino."""

    def __init__(
        self,
        save_plots: bool = True,
        output_dir: Path | str = OUTPUTS_DIR,
    ) -> None:
        self._save_plots = save_plots
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model: ClassifierMixin,
        X: np.ndarray,
        y_true: np.ndarray,
        label: str = "Modelo",
        class_names: list[str] | None = None,
    ) -> EvaluationResult:
        class_names = class_names or TARGET_NAMES

        # Primeiro geramos previsões e probabilidades; o resto das métricas nasce daqui.
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)  # probabilidades para ROC-AUC multiclasse
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

        # Calcula um conjunto amplo de métricas — várias perspectivas do desempenho:
        # - accuracy: % de acertos totais (enganosa quando classes são desbalanceadas)
        # - balanced_accuracy: média do recall por classe (corrige o desbalanceamento)
        # - f1_macro: equilíbrio entre precision e recall, tratando todas as classes igualmente
        # - mcc: correlação de Matthews — considerada a melhor métrica única para multiclasse
        # - gm (Geometric Mean): raiz do produto dos recalls por classe
        # - roc_auc: área sob a curva ROC (OvR = One-vs-Rest para multiclasse)
        result = EvaluationResult(
            label=label,
            accuracy=float(accuracy_score(y_true, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            recall_macro=float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            f1_macro=float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            f1_weighted=float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            mcc=float(matthews_corrcoef(y_true, y_pred)),
            gm=float(self._safe_geometric_mean(y_true, y_pred)),
            roc_auc_ovr_macro=float(
                roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class="ovr",
                    average="macro",
                )
            ),
            confusion_matrix=cm.tolist(),
            class_names=class_names,
            classification_report=classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                digits=4,
                zero_division=0,
            ),
        )

        result.print_summary()
        print("\nRelatorio de classificacao:")
        print(result.classification_report)

        if self._save_plots:
            # O slug vira um nome de arquivo limpo para os plots dessa avaliação.
            slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            self._plot_confusion_matrix(cm, class_names, label, slug)

        return result

    def compare(
        self,
        baseline: EvaluationResult,
        optimized: EvaluationResult,
    ) -> None:
        print("\n" + "=" * 72)
        print(f"{'COMPARACAO DE MODELOS':^72}")
        print("=" * 72)
        print(f"{'Metrica':<24} {'Baseline':>12} {'Otimizado':>12}")
        print("-" * 72)

        metrics_map = {
            "Accuracy (%)": ("accuracy", True),
            "Balanced Acc (%)": ("balanced_accuracy", True),
            "Precision Macro (%)": ("precision_macro", True),
            "Recall Macro (%)": ("recall_macro", True),
            "F1 Macro (%)": ("f1_macro", True),
            "F1 Weighted (%)": ("f1_weighted", True),
            "MCC": ("mcc", False),
            "G-Mean": ("gm", False),
            "ROC-AUC Macro": ("roc_auc_ovr_macro", False),
        }

        for name, (attr, pct) in metrics_map.items():
            base_value = getattr(baseline, attr)
            opt_value = getattr(optimized, attr)
            if pct:
                print(f"{name:<24} {base_value*100:>11.2f} {opt_value*100:>11.2f}")
            else:
                print(f"{name:<24} {base_value:>11.4f} {opt_value:>11.4f}")

        print("=" * 72)
        gain = (optimized.f1_macro - baseline.f1_macro) * 100
        print(f"  Ganho de F1 Macro (tuning): {gain:+.2f} pp")

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list[str],
        label: str,
        slug: str,
    ) -> None:
        try:
            plt = get_pyplot()
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names,
            ).plot(values_format=".0f", ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"Matriz de confusao — {label}")
            plt.tight_layout()
            path = self._output_dir / f"confusion_matrix_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelEvaluator] Matriz de confusao salva: {path.name}")
        except Exception as exc:
            print(f"[ModelEvaluator] Erro ao salvar matriz de confusao: {exc}")

    def _safe_geometric_mean(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        # A API do geometric_mean_score mudou entre versões do imbalanced-learn,
        # então tentamos diferentes assinaturas para manter compatibilidade
        for average in ("macro", "multiclass", None):
            try:
                kwargs = {}
                if average is not None:
                    kwargs["average"] = average
                return float(geometric_mean_score(y_true, y_pred, **kwargs))
            except TypeError:
                continue
        return float(geometric_mean_score(y_true, y_pred))
