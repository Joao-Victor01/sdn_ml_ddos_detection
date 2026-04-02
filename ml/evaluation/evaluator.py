"""
Avaliação completa do modelo com métricas e visualizações.

SRP: responsável exclusivamente pelo cálculo de métricas e geração de plots.

O test_set é usado UMA ÚNICA VEZ aqui — após todo o desenvolvimento.
Nunca é usado para ajustar parâmetros ou tomar decisões de modelo.

Métricas implementadas (conforme boas práticas do curso para dados desbalanceados):
  - Acurácia, Precisão, Recall, F1-Score
  - MCC (Matthews Correlation Coefficient) — melhor para desbalanceamento
  - Geometric Mean — balanceia sensibilidade e especificidade
  - ROC-AUC
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from ml.config import OUTPUTS_DIR, PAPER_METRICS


@dataclass
class EvaluationResult:
    """
    Resultado imutável de uma avaliação.

    Armazena todas as métricas calculadas no test_set para
    fácil comparação entre baseline e modelo otimizado.
    """

    label:     str
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    mcc:       float
    gm:        float
    roc_auc:   float
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Texto do relatório sklearn (opcional)
    classification_report: str = field(default="", repr=False)

    def print_summary(self) -> None:
        """Exibe o resumo formatado das métricas."""
        width = 58
        print("\n" + "=" * width)
        print(f"{f'RESULTADOS — {self.label}':^{width}}")
        print("=" * width)
        print(f"  Acurácia          : {self.accuracy:.4f}  ({self.accuracy*100:.2f}%)")
        print(f"  Precisão          : {self.precision:.4f}  ({self.precision*100:.2f}%)")
        print(f"  Recall/Sensibil.  : {self.recall:.4f}  ({self.recall*100:.2f}%)")
        print(f"  F1-Score          : {self.f1:.4f}  ({self.f1*100:.2f}%)")
        print(f"  MCC               : {self.mcc:.4f}")
        print(f"  Geometric Mean    : {self.gm:.4f}")
        print(f"  ROC-AUC           : {self.roc_auc:.4f}")
        print("-" * width)
        print(f"  TP={self.tp:,}  TN={self.tn:,}  FP={self.fp:,}  FN={self.fn:,}")
        print("=" * width)


class ModelEvaluator:
    """
    Avalia um modelo MLPClassifier no test_set e gera relatórios/plots.

    Uso:
        evaluator = ModelEvaluator()
        result    = evaluator.evaluate(model, X_test_scaled, y_test, label="Baseline")
        evaluator.compare(baseline_result, optimized_result)
    """

    def __init__(self, save_plots: bool = True) -> None:
        self._save_plots = save_plots
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── API pública ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model: MLPClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label: str = "Modelo",
    ) -> EvaluationResult:
        """
        Avalia o modelo no test_set.

        O test_set representa dados "do mundo real" — nunca foi visto
        durante o treinamento ou tuning.

        Parameters
        ----------
        model   : MLPClassifier treinado
        X_test  : np.ndarray — features do teste (escalonadas e selecionadas)
        y_test  : np.ndarray — alvo real do teste
        label   : str        — nome do modelo para exibição

        Returns
        -------
        EvaluationResult com todas as métricas calculadas.
        """
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        cm           = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        result = EvaluationResult(
            label=label,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            mcc=float(matthews_corrcoef(y_test, y_pred)),
            gm=float(geometric_mean_score(y_test, y_pred)),
            roc_auc=float(roc_auc_score(y_test, y_pred_proba)),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            classification_report=classification_report(
                y_test, y_pred,
                target_names=["Benigno (0)", "Ataque DDoS (1)"],
            ),
        )

        result.print_summary()

        print("\nRelatório de Classificação:")
        print(result.classification_report)

        if self._save_plots:
            slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            self._plot_confusion_matrix(cm, label, slug)
            self._plot_roc_curve(y_test, y_pred_proba, result.roc_auc, label, slug)

        return result

    def compare(
        self,
        baseline: EvaluationResult,
        optimized: EvaluationResult,
    ) -> None:
        """
        Exibe tabela comparativa: Baseline vs. Otimizado vs. Artigo.
        """
        print("\n" + "=" * 65)
        print(f"{'COMPARAÇÃO DE MODELOS':^65}")
        print("=" * 65)
        print(f"{'Métrica':<22} {'Baseline':>10} {'Otimizado':>10} {'Artigo':>10}")
        print("-" * 65)

        metrics_map = {
            "Acurácia (%)":   ("accuracy",  "%"),
            "Precisão (%)":   ("precision", "%"),
            "Recall (%)":     ("recall",    "%"),
            "F1-Score (%)":   ("f1",        "%"),
            "MCC":            ("mcc",       "raw"),
            "Geometric Mean": ("gm",        "raw"),
            "ROC-AUC":        ("roc_auc",   "raw"),
        }

        for name, (attr, fmt) in metrics_map.items():
            b_val = getattr(baseline, attr)
            o_val = getattr(optimized, attr)
            paper_key = attr if attr in PAPER_METRICS else None

            if fmt == "%":
                b_str = f"{b_val*100:>10.2f}"
                o_str = f"{o_val*100:>10.2f}"
                paper = f"{PAPER_METRICS[paper_key]:>10.2f}" if paper_key else f"{'—':>10}"
            else:
                b_str = f"{b_val:>10.4f}"
                o_str = f"{o_val:>10.4f}"
                paper = f"{'—':>10}"

            print(f"{name:<22}{b_str}{o_str}{paper}")

        print("=" * 65)

        # Ganho do tuning
        f1_gain = (optimized.f1 - baseline.f1) * 100
        print(f"\n  Ganho de F1 (tuning): {f1_gain:+.2f} pp")

        # Distância ao artigo
        paper_f1 = PAPER_METRICS.get("f1")
        if paper_f1:
            dist = paper_f1 - optimized.f1 * 100
            print(f"  Distância ao artigo (F1): {dist:.2f} pp")

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        label: str,
        slug: str,
    ) -> None:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Benigno", "Ataque DDoS"],
            ).plot(values_format=".0f", ax=ax, cmap="Blues")
            ax.set_title(f"Matriz de Confusão — {label} (InSDN8)")
            plt.tight_layout()
            path = OUTPUTS_DIR / f"confusion_matrix_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelEvaluator] Matriz de confusão salva: {path.name}")
        except Exception as e:
            print(f"[ModelEvaluator] Erro ao salvar matriz de confusão: {e}")

    def _plot_roc_curve(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        auc: float,
        label: str,
        slug: str,
    ) -> None:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_predictions(
                y_test,
                y_pred_proba,
                name=f"{label} (AUC={auc:.4f})",
                ax=ax,
            )
            ax.plot([0, 1], [0, 1], "k--", label="Aleatório")
            ax.set_title(f"Curva ROC — {label} (InSDN8)")
            ax.legend()
            plt.tight_layout()
            path = OUTPUTS_DIR / f"roc_curve_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[ModelEvaluator] Curva ROC salva: {path.name}")
        except Exception as e:
            print(f"[ModelEvaluator] Erro ao salvar curva ROC: {e}")
