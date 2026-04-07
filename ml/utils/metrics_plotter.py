"""
Visualizacao comparativa do historico de runs do pipeline multiclasse.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.config import OUTPUTS_DIR

METRICS_FILE = OUTPUTS_DIR / "metrics_history.json"
DISPLAY_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Bal.Acc"),
    ("f1_macro", "F1 Macro"),
    ("f1_weighted", "F1 Weighted"),
    ("mcc", "MCC"),
    ("gm", "G-Mean"),
    ("roc_auc_ovr_macro", "ROC-AUC"),
]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class MetricsPlotter:
    """Gera graficos comparativos a partir do metrics_history.json."""

    def __init__(self, metrics_file: Path | str = METRICS_FILE) -> None:
        self._path = Path(metrics_file)
        self._history = self._load()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def plot_evolution(self, save: bool = True) -> None:
        df = self._to_dataframe()
        if len(df) < 2:
            print("[MetricsPlotter] plot_evolution requer pelo menos 2 runs.")
            return

        # Aqui a ideia é simples: acompanhar como as métricas evoluíram de run para run.
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (metric, label) in enumerate(DISPLAY_METRICS):
            ax.plot(x, df[metric], marker="o", label=label, color=COLORS[idx % len(COLORS)])

        ax.set_xticks(x)
        ax.set_xticklabels(df["run_id"].tolist(), rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.set_title("Evolucao das metricas por run")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()

        if save:
            path = OUTPUTS_DIR / "metrics_evolution.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Evolucao salva -> {path}")
        plt.show()
        plt.close()

    def plot_comparison(self, run_id_a: str, run_id_b: str, save: bool = True) -> None:
        a = self._get_entry(run_id_a)
        b = self._get_entry(run_id_b)
        if a is None or b is None:
            return

        # Comparação direta barra a barra: bom para ver rapidamente quem ganhou em cada critério.
        metrics_a = [a["metrics"][metric] for metric, _ in DISPLAY_METRICS]
        metrics_b = [b["metrics"][metric] for metric, _ in DISPLAY_METRICS]
        x = np.arange(len(DISPLAY_METRICS))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, metrics_a, width, label=a["label"], color=COLORS[0])
        ax.bar(x + width / 2, metrics_b, width, label=b["label"], color=COLORS[1])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in DISPLAY_METRICS], rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.set_title(f"Comparacao: {run_id_a} vs {run_id_b}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        if save:
            path = OUTPUTS_DIR / f"comparison_{run_id_a}_vs_{run_id_b}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Comparacao salva -> {path}")
        plt.show()
        plt.close()

    def plot_radar(self, run_id: str, save: bool = True) -> None:
        entry = self._get_entry(run_id)
        if entry is None:
            return

        # O radar não é o mais "científico", mas ajuda muito numa leitura visual rápida de equilíbrio.
        values = [entry["metrics"][metric] for metric, _ in DISPLAY_METRICS]
        values += [values[0]]
        labels = [label for _, label in DISPLAY_METRICS]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.plot(angles, values, "o-", linewidth=2, color=COLORS[0])
        ax.fill(angles, values, alpha=0.25, color=COLORS[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Radar de metricas — {run_id}")

        if save:
            path = OUTPUTS_DIR / f"radar_{run_id}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Radar salvo -> {path}")
        plt.show()
        plt.close()

    def plot_confusion_heatmap(self, run_id: str, save: bool = True) -> None:
        entry = self._get_entry(run_id)
        if entry is None:
            return

        # Puxa a matriz salva no histórico para poder revisitar a run sem rerodar o modelo.
        cm = np.array(entry.get("confusion_matrix", []), dtype=float)
        class_names = entry.get("class_names", [f"Classe {idx}" for idx in range(len(cm))])
        if cm.size == 0:
            print("[MetricsPlotter] Matriz de confusao indisponivel para esta run.")
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicao")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusao — {run_id}")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)

        plt.tight_layout()
        if save:
            path = OUTPUTS_DIR / f"cm_history_{run_id}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Heatmap salvo -> {path}")
        plt.show()
        plt.close()

    def plot_dashboard(self, save: bool = True) -> None:
        df = self._to_dataframe()
        if df.empty:
            print("[MetricsPlotter] Nenhuma run registrada.")
            return

        # O dashboard junta uma visão "macro" em uma figura só para consulta rápida.
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        x = np.arange(len(df))

        axes[0].plot(x, df["f1_macro"], marker="o", label="F1 Macro", color=COLORS[0])
        axes[0].plot(x, df["balanced_accuracy"], marker="o", label="Balanced Accuracy", color=COLORS[1])
        axes[0].plot(x, df["roc_auc_ovr_macro"], marker="o", label="ROC-AUC", color=COLORS[2])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df["run_id"].tolist(), rotation=20, ha="right")
        axes[0].set_ylim(0, 1.02)
        axes[0].set_title("Metricas principais por run")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        width = 0.35
        axes[1].bar(x - width / 2, df["accuracy"], width, label="Accuracy", color=COLORS[3])
        axes[1].bar(x + width / 2, df["mcc"], width, label="MCC", color=COLORS[4])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df["run_id"].tolist(), rotation=20, ha="right")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("Accuracy e MCC por run")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()

        plt.tight_layout()
        if save:
            path = OUTPUTS_DIR / "dashboard_metrics.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Dashboard salvo -> {path}")
        plt.show()
        plt.close()

    def list_runs(self) -> None:
        if not self._history:
            print("[MetricsPlotter] Nenhuma run registrada.")
            return
        print(f"\n{'Run ID':<30} {'Label':<25} {'Timestamp':<20} {'F1 Macro':>10}")
        print("-" * 95)
        for entry in self._history:
            metrics = entry.get("metrics", {})
            print(
                f"{entry['run_id']:<30} {entry.get('label', ''):<25} "
                f"{entry['timestamp'][:19]:<20} {metrics.get('f1_macro', 0):>10.4f}"
            )

    def _load(self) -> list[dict]:
        if self._path.exists():
            with open(self._path) as file:
                return json.load(file)
        return []

    def _to_dataframe(self) -> pd.DataFrame:
        rows = []
        for entry in self._history:
            # Achata a estrutura do JSON para ficar fácil de plotar com pandas/matplotlib.
            row = {"run_id": entry["run_id"], "label": entry.get("label", ""), "timestamp": entry["timestamp"]}
            row.update(entry.get("metrics", {}))
            rows.append(row)
        return pd.DataFrame(rows)

    def _get_entry(self, run_id: str) -> dict | None:
        for entry in self._history:
            if entry["run_id"] == run_id:
                return entry
        print(f"[MetricsPlotter] Run '{run_id}' nao encontrada.")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plota metricas do historico de experimentos.")
    parser.add_argument("--list", action="store_true", help="Lista todas as runs")
    parser.add_argument("--evolve", action="store_true", help="Plota evolucao das metricas")
    parser.add_argument("--dashboard", action="store_true", help="Dashboard completo")
    parser.add_argument("--radar", metavar="RUN_ID", help="Radar chart de uma run")
    parser.add_argument("--confusion", metavar="RUN_ID", help="Heatmap da matriz de confusao")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))

    args = parser.parse_args()
    plotter = MetricsPlotter()

    if args.list:
        plotter.list_runs()
    elif args.evolve:
        plotter.plot_evolution()
    elif args.radar:
        plotter.plot_radar(args.radar)
    elif args.confusion:
        plotter.plot_confusion_heatmap(args.confusion)
    elif args.compare:
        plotter.plot_comparison(args.compare[0], args.compare[1])
    else:
        plotter.plot_dashboard()
