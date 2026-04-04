"""
Avaliação triclasse com métricas adequadas para dados desbalanceados.

Métricas escolhidas (seção 6 do plano — Aula 7):
  - F1 Macro      : média não-ponderada do F1 por classe — penaliza classes
                    com baixo desempenho independente do volume.
  - MCC           : Matthews Correlation Coefficient — melhor métrica única
                    para desbalanceamento multiclasse.
  - Geometric Mean: raiz do produto dos recalls — equilibra sensibilidade
                    de todas as classes.
  - Recall Classe 2: o mais crítico em produção — falso negativo em Zumbi
                    Interno = host comprometido não isolado.

NÃO usar: acurácia sozinha, F1-weighted (mascara classes minoritárias).

SRP: calcula métricas e gera plots — nunca toca no modelo ou nos dados de treino.

Referência: plano_triclasse_insdn_v4.md, Seção 8.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)

from ml.triclass.config import CLASS_NAMES, OUTPUTS_TRICLASS


@dataclass
class TriclassEvaluationResult:
    """
    Resultado imutável de uma avaliação triclasse.

    Armazena métricas globais e por classe para comparação entre modelos.
    """

    label: str
    f1_macro: float
    mcc: float
    gm: float

    # Métricas por classe (índice = classe)
    f1_per_class: list[float]
    precision_per_class: list[float]
    recall_per_class: list[float]
    support_per_class: list[int]

    # Texto do classification_report (sklearn)
    report: str = field(default="", repr=False)

    @property
    def recall_class2(self) -> float:
        """Recall da Classe 2 (Zumbi Interno) — métrica mais crítica em produção."""
        return self.recall_per_class[2] if len(self.recall_per_class) > 2 else 0.0

    @property
    def f1_class2(self) -> float:
        return self.f1_per_class[2] if len(self.f1_per_class) > 2 else 0.0

    def print_summary(self) -> None:
        width = 62
        print("\n" + "=" * width)
        print(f"{f'RESULTADOS — {self.label}':^{width}}")
        print("=" * width)
        print(f"  F1 Macro      : {self.f1_macro:.4f}")
        print(f"  MCC           : {self.mcc:.4f}")
        print(f"  Geometric Mean: {self.gm:.4f}")
        print()
        print(f"  {'Classe':<22} {'F1':>7} {'Precision':>10} {'Recall':>8} {'Support':>9}")
        print(f"  {'-'*58}")
        for i, name in CLASS_NAMES.items():
            if i < len(self.f1_per_class):
                print(
                    f"  {i} {name:<20} "
                    f"{self.f1_per_class[i]:>7.4f} "
                    f"{self.precision_per_class[i]:>10.4f} "
                    f"{self.recall_per_class[i]:>8.4f} "
                    f"{self.support_per_class[i]:>9,}"
                )
        print()
        print(f"  ⚠ Recall Classe 2 (Zumbi): {self.recall_class2:.4f}"
              f"  {'OK' if self.recall_class2 >= 0.60 else 'BAIXO — revisar'}")
        print("=" * width)


class TriclassEvaluator:
    """
    Avalia modelos triclasse no test_set e gera relatórios/plots.

    Regra absoluta: test_set usado UMA ÚNICA VEZ, ao final de todo o pipeline.

    Uso:
        evaluator = TriclassEvaluator()
        result_rf  = evaluator.evaluate(rf_best,  X_test_vt, y_test, "RF Otimizado")
        result_mlp = evaluator.evaluate(mlp_pipe, X_test_vt, y_test, "MLP")
        evaluator.compare(result_rf, result_mlp)
    """

    def __init__(self, save_plots: bool = True) -> None:
        self._save_plots = save_plots
        OUTPUTS_TRICLASS.mkdir(parents=True, exist_ok=True)

    # ── API pública ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label: str = "Modelo",
    ) -> TriclassEvaluationResult:
        """
        Avalia o modelo no test_set (usado UMA ÚNICA VEZ).

        Parameters
        ----------
        model  : estimador sklearn treinado (RF ou Pipeline com MLP)
        X_test : features do teste (pós-VT, sem SMOTE)
        y_test : alvo real do teste
        label  : nome do modelo para exibição e nomes de arquivos

        Returns
        -------
        TriclassEvaluationResult com todas as métricas.
        """
        y_pred = model.predict(X_test)
        classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

        cm = confusion_matrix(y_test, y_pred, labels=classes)

        f1_per   = f1_score(y_test, y_pred, average=None, zero_division=0, labels=classes)
        prec_per = __import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(
            y_test, y_pred, average=None, zero_division=0, labels=classes
        )
        rec_per  = recall_score(y_test, y_pred, average=None, zero_division=0, labels=classes)
        sup_per  = [int((np.array(y_test) == c).sum()) for c in classes]

        result = TriclassEvaluationResult(
            label=label,
            f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            mcc=float(matthews_corrcoef(y_test, y_pred)),
            gm=float(geometric_mean_score(y_test, y_pred, average="macro")),
            f1_per_class=f1_per.tolist(),
            precision_per_class=prec_per.tolist(),
            recall_per_class=rec_per.tolist(),
            support_per_class=sup_per,
            report=classification_report(
                y_test, y_pred,
                target_names=[CLASS_NAMES.get(c, str(c)) for c in classes],
                zero_division=0,
            ),
        )

        result.print_summary()
        print("\nRelatório completo:")
        print(result.report)

        if self._save_plots:
            slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            self._plot_confusion_matrix(cm, label, slug)

        return result

    def compare(self, *results: TriclassEvaluationResult) -> None:
        """
        Tabela comparativa entre N modelos avaliados.
        """
        width = 72
        print("\n" + "=" * width)
        print(f"{'COMPARAÇÃO DE MODELOS TRICLASSE':^{width}}")
        print("=" * width)

        header = f"{'Métrica':<28}"
        for r in results:
            header += f" {r.label[:12]:>12}"
        print(header)
        print("-" * width)

        def row(name, values):
            line = f"{name:<28}"
            for v in values:
                line += f" {v:>12.4f}"
            print(line)

        row("F1 Macro",      [r.f1_macro for r in results])
        row("MCC",           [r.mcc      for r in results])
        row("Geometric Mean",[r.gm       for r in results])
        row("Recall Cl.2 (Zumbi)", [r.recall_class2 for r in results])
        row("F1 Cl.2 (Zumbi)",     [r.f1_class2     for r in results])
        print("=" * width)

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        label: str,
        slug: str,
    ) -> None:
        try:
            display_labels = [CLASS_NAMES.get(i, str(i)) for i in range(cm.shape[0])]
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=display_labels,
            ).plot(values_format=".0f", ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"Matriz de Confusão — {label} (InSDN Triclasse)")
            plt.tight_layout()
            path = OUTPUTS_TRICLASS / f"confusion_matrix_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[TriclassEvaluator] Matriz salva → {path.name}")
        except Exception as e:
            print(f"[TriclassEvaluator] Erro ao salvar matriz: {e}")
