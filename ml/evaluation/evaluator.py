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


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação MULTICLASSE — extensão OCP sem modificar ModelEvaluator binário
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dc
from ml.config import CLASS_NAMES, OUTPUTS_DIR as _ODIR


@_dc
class MulticlassResult:
    """
    Resultado de avaliação multiclasse (3 classes).

    Métricas macro equilibram igualmente as três classes —
    adequado porque o erro em qualquer classe é igualmente grave.
    """
    label:          str
    accuracy:       float
    f1_macro:       float
    precision_macro: float
    recall_macro:   float
    mcc:            float
    gm:             float
    per_class_f1:   dict          # {nome_classe: f1}
    per_class_prec: dict          # {nome_classe: precision}
    per_class_rec:  dict          # {nome_classe: recall}
    classification_report: str

    def print_summary(self) -> None:
        w = 62
        print("\n" + "=" * w)
        print(f"{'RESULTADOS MULTICLASSE — ' + self.label:^{w}}")
        print("=" * w)
        print(f"  Acurácia geral   : {self.accuracy*100:.4f}%")
        print(f"  F1 Macro         : {self.f1_macro*100:.4f}%")
        print(f"  Precisão Macro   : {self.precision_macro*100:.4f}%")
        print(f"  Recall Macro     : {self.recall_macro*100:.4f}%")
        print(f"  MCC              : {self.mcc:.4f}")
        print(f"  Geometric Mean   : {self.gm:.4f}")
        print(f"\n  ── Por classe ──")
        for cls in CLASS_NAMES:
            f1   = self.per_class_f1.get(cls,   0.0)
            prec = self.per_class_prec.get(cls, 0.0)
            rec  = self.per_class_rec.get(cls,  0.0)
            print(f"  {cls:<20}: F1={f1*100:.2f}%  "
                  f"P={prec*100:.2f}%  R={rec*100:.2f}%")
        print("=" * w)
        print(f"\n{self.classification_report}")


class MulticlassEvaluator:
    """
    Avalia modelos de classificação triclasse (Benigno / Externo / Interno).

    Extensão de ModelEvaluator para 3 classes — segue OCP: não modifica
    a classe binária existente, apenas adiciona novo comportamento.

    Uso:
        ev  = MulticlassEvaluator()
        res = ev.evaluate(model, X_test, y_test, label="Baseline")
        ev.compare(res_baseline, res_tuned)
    """

    def __init__(self, save_plots: bool = True) -> None:
        self._save   = save_plots
        _ODIR.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        label:   str = "Modelo",
    ) -> MulticlassResult:
        """Calcula todas as métricas multiclasse no test_set."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, matthews_corrcoef, classification_report,
            confusion_matrix,
        )
        from imblearn.metrics import geometric_mean_score

        y_pred = model.predict(X_test)

        acc  = float(accuracy_score(y_test, y_pred))
        f1m  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prm  = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        recm = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        mcc  = float(matthews_corrcoef(y_test, y_pred))
        gm   = float(geometric_mean_score(y_test, y_pred, average="macro"))

        # Métricas por classe
        f1_per   = f1_score(y_test, y_pred, average=None, zero_division=0)
        prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_per  = recall_score(y_test, y_pred, average=None, zero_division=0)
        n_cls    = len(CLASS_NAMES)
        pcf1  = {CLASS_NAMES[i]: float(f1_per[i])   for i in range(min(n_cls, len(f1_per)))}
        pcpre = {CLASS_NAMES[i]: float(prec_per[i]) for i in range(min(n_cls, len(prec_per)))}
        pcrec = {CLASS_NAMES[i]: float(rec_per[i])  for i in range(min(n_cls, len(rec_per)))}

        report = classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        result = MulticlassResult(
            label=label,
            accuracy=acc,
            f1_macro=f1m,
            precision_macro=prm,
            recall_macro=recm,
            mcc=mcc,
            gm=gm,
            per_class_f1=pcf1,
            per_class_prec=pcpre,
            per_class_rec=pcrec,
            classification_report=report,
        )
        result.print_summary()

        if self._save:
            cm   = confusion_matrix(y_test, y_pred)
            slug = label.lower().replace(" ", "_")
            self._plot_cm(cm, label, slug)
            self._plot_per_class_bars(result, slug)

        return result

    def compare(self, a: MulticlassResult, b: MulticlassResult) -> None:
        """Tabela comparativa entre dois modelos."""
        w = 68
        print("\n" + "=" * w)
        print(f"{'COMPARAÇÃO MULTICLASSE':^{w}}")
        print("=" * w)
        print(f"{'Métrica':<24} {a.label:>20} {b.label:>20}")
        print("-" * w)
        pairs = [
            ("Acurácia (%)",      a.accuracy*100,       b.accuracy*100,       True),
            ("F1 Macro (%)",      a.f1_macro*100,       b.f1_macro*100,       True),
            ("Precisão Macro (%)",a.precision_macro*100, b.precision_macro*100, True),
            ("Recall Macro (%)",  a.recall_macro*100,   b.recall_macro*100,   True),
            ("MCC",               a.mcc,                b.mcc,                False),
            ("Geometric Mean",    a.gm,                 b.gm,                 False),
        ]
        for name, va, vb, is_pct in pairs:
            fmt = ".2f" if is_pct else ".4f"
            print(f"  {name:<22} {va:>20{fmt}} {vb:>20{fmt}}")
        print("-" * w)
        print(f"\n  ── Por classe (F1) ──")
        for cls in CLASS_NAMES:
            va = a.per_class_f1.get(cls, 0.0) * 100
            vb = b.per_class_f1.get(cls, 0.0) * 100
            diff = vb - va
            print(f"  {cls:<22} {va:>20.2f}% {vb:>20.2f}%  ({diff:+.2f} pp)")
        print("=" * w)

    # ── plots ──────────────────────────────────────────────────────────────────

    def _plot_cm(self, cm: np.ndarray, label: str, slug: str) -> None:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=CLASS_NAMES,
            ).plot(values_format=".0f", ax=ax, cmap="Blues")
            ax.set_title(f"Matriz de Confusão 3×3 — {label}")
            plt.tight_layout()
            path = _ODIR / f"cm_multi_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[MulticlassEvaluator] CM salva → {path.name}")
        except Exception as e:
            print(f"[MulticlassEvaluator] Erro CM: {e}")

    def _plot_per_class_bars(self, result: MulticlassResult, slug: str) -> None:
        try:
            import matplotlib.pyplot as plt
            classes = CLASS_NAMES
            x = np.arange(len(classes))
            w = 0.25
            f1s   = [result.per_class_f1.get(c,   0)*100 for c in classes]
            precs = [result.per_class_prec.get(c, 0)*100 for c in classes]
            recs  = [result.per_class_rec.get(c,  0)*100 for c in classes]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - w, precs, w, label="Precisão",  color="#4C72B0", alpha=0.85)
            ax.bar(x,     f1s,   w, label="F1-Score",  color="#DD8452", alpha=0.85)
            ax.bar(x + w, recs,  w, label="Recall",    color="#55A868", alpha=0.85)

            for xi, (p, f, r) in zip(x, zip(precs, f1s, recs)):
                for off, val in [(-w, p), (0, f), (w, r)]:
                    ax.text(xi + off, val + 0.3, f"{val:.1f}", ha="center",
                            va="bottom", fontsize=8)

            ax.set_xticks(x); ax.set_xticklabels(classes)
            ax.set_ylabel("Score (%)")
            ax.set_title(f"Métricas por Classe — {result.label}")
            ax.legend(); ax.set_ylim(0, 108); ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = _ODIR / f"per_class_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[MulticlassEvaluator] Barras por classe → {path.name}")
        except Exception as e:
            print(f"[MulticlassEvaluator] Erro barras: {e}")
