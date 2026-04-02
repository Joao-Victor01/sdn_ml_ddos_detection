"""
Visualização comparativa de múltiplas runs de treinamento.

Lê o histórico salvo pelo MetricsLogger e gera gráficos para:
  1. Evolução das métricas ao longo das runs
  2. Comparação lado a lado de dois modelos
  3. Radar chart com todas as métricas de uma run
  4. Dashboard completo (todos os gráficos em um único arquivo)

Uso direto (CLI):
    python -m ml.utils.metrics_plotter               # dashboard completo
    python -m ml.utils.metrics_plotter --compare baseline_v1 tuned_v1

Uso como módulo:
    from ml.utils.metrics_plotter import MetricsPlotter
    plotter = MetricsPlotter()
    plotter.plot_evolution()
    plotter.plot_comparison("baseline_v1", "tuned_v1")
    plotter.plot_radar("baseline_v1")
    plotter.plot_dashboard()
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from ml.config import OUTPUTS_DIR

METRICS_FILE = OUTPUTS_DIR / "metrics_history.json"

# Métricas exibidas nos gráficos (em %)
DISPLAY_METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "gm", "roc_auc"]
DISPLAY_LABELS  = ["Acurácia", "Precisão", "Recall", "F1", "MCC", "G-Mean", "ROC-AUC"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class MetricsPlotter:
    """
    Gera gráficos comparativos a partir do histórico de métricas.
    """

    def __init__(self, metrics_file: Path | str = METRICS_FILE) -> None:
        self._path    = Path(metrics_file)
        self._history = self._load()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── API pública ────────────────────────────────────────────────────────────

    def plot_evolution(self, save: bool = True) -> None:
        """
        Gráfico de linhas: evolução das métricas ao longo das runs.
        Útil para acompanhar melhoras iterativas entre experimentos.
        """
        if len(self._history) < 2:
            print("[MetricsPlotter] plot_evolution requer ≥ 2 runs. "
                  "Execute o pipeline mais vezes com diferentes parâmetros.")
            return

        df = self._to_dataframe()
        run_labels = df["run_id"].tolist()
        x = np.arange(len(run_labels))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot principal: accuracy, f1, recall, precision
        ax = axes[0]
        for metric, label, color in zip(
            ["accuracy", "f1", "recall", "precision"],
            ["Acurácia", "F1-Score", "Recall", "Precisão"],
            COLORS[:4],
        ):
            vals = df[metric].values * 100
            ax.plot(x, vals, marker="o", label=label, color=color)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}%", (xi, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=30, ha="right")
        ax.set_ylabel("Score (%)")
        ax.set_title("Evolução das Métricas Principais por Run")
        ax.legend(loc="lower right")
        ax.set_ylim(max(0, df["accuracy"].min() * 100 - 2), 101)
        ax.grid(alpha=0.3)

        # Plot secundário: MCC, G-Mean, ROC-AUC
        ax2 = axes[1]
        for metric, label, color in zip(
            ["mcc", "gm", "roc_auc"],
            ["MCC", "G-Mean", "ROC-AUC"],
            COLORS[4:7],
        ):
            vals = df[metric].values
            ax2.plot(x, vals, marker="s", label=label, color=color)
            for xi, v in zip(x, vals):
                ax2.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                             xytext=(0, 6), ha="center", fontsize=7)

        ax2.set_xticks(x)
        ax2.set_xticklabels(run_labels, rotation=30, ha="right")
        ax2.set_ylabel("Score (0–1)")
        ax2.set_title("Evolução de MCC, G-Mean e ROC-AUC por Run")
        ax2.legend(loc="lower right")
        ax2.set_ylim(max(0, df["mcc"].min() - 0.05), 1.02)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save:
            path = OUTPUTS_DIR / "metrics_evolution.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Evolução salva → {path}")
        plt.show()
        plt.close()

    def plot_comparison(
        self,
        run_id_a: str,
        run_id_b: str,
        save: bool = True,
    ) -> None:
        """
        Barras agrupadas comparando duas runs lado a lado.

        Ideal para comparar baseline vs. otimizado.
        """
        a = self._get_entry(run_id_a)
        b = self._get_entry(run_id_b)
        if a is None or b is None:
            return

        metrics_a = [a["metrics"][m] * 100 if m != "mcc" else a["metrics"][m]
                     for m in DISPLAY_METRICS]
        metrics_b = [b["metrics"][m] * 100 if m != "mcc" else b["metrics"][m]
                     for m in DISPLAY_METRICS]

        x = np.arange(len(DISPLAY_METRICS))
        w = 0.35

        fig, ax = plt.subplots(figsize=(13, 6))
        bars_a = ax.bar(x - w/2, metrics_a, w, label=a.get("label", run_id_a),
                        color=COLORS[0], alpha=0.85)
        bars_b = ax.bar(x + w/2, metrics_b, w, label=b.get("label", run_id_b),
                        color=COLORS[1], alpha=0.85)

        for bar, val in zip(bars_a, metrics_a):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars_b, metrics_b):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_LABELS)
        ax.set_ylabel("Score (% ou 0–1 para MCC)")
        ax.set_title(f"Comparação: {a.get('label', run_id_a)} vs. {b.get('label', run_id_b)}")
        ax.legend()
        ax.set_ylim(min(min(metrics_a), min(metrics_b)) - 2, 102)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save:
            slug = f"{run_id_a}_vs_{run_id_b}".replace(" ", "_")
            path = OUTPUTS_DIR / f"comparison_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Comparação salva → {path}")
        plt.show()
        plt.close()

    def plot_radar(self, run_id: str, save: bool = True) -> None:
        """
        Radar chart (spider plot) com todas as métricas de uma run.

        Visualização intuitiva da "forma" do desempenho — fácil de ver
        onde o modelo é forte e onde há espaço para melhora.
        """
        entry = self._get_entry(run_id)
        if entry is None:
            return

        values = [entry["metrics"][m] for m in DISPLAY_METRICS]
        # Fechar o polígono
        values_closed   = values + [values[0]]
        labels_for_plot = DISPLAY_LABELS + [DISPLAY_LABELS[0]]
        N = len(DISPLAY_METRICS)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

        ax.plot(angles_closed, values_closed, "o-", linewidth=2, color=COLORS[0])
        ax.fill(angles_closed, values_closed, alpha=0.25, color=COLORS[0])

        ax.set_xticks(angles)
        ax.set_xticklabels(DISPLAY_LABELS, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.5, 0.7, 0.9, 1.0])
        ax.set_yticklabels(["50%", "70%", "90%", "100%"], fontsize=8)
        ax.set_title(f"Radar de Métricas — {entry.get('label', run_id)}\n"
                     f"({entry['timestamp'][:10]})", pad=20)

        # Anotar valores
        for angle, value, label in zip(angles, values, DISPLAY_LABELS):
            ax.annotate(
                f"{value:.4f}",
                xy=(angle, value),
                xytext=(angle, value + 0.05),
                ha="center", va="center", fontsize=8,
            )

        if save:
            slug = run_id.replace(" ", "_")
            path = OUTPUTS_DIR / f"radar_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Radar salvo → {path}")
        plt.show()
        plt.close()

    def plot_confusion_heatmap(self, run_id: str, save: bool = True) -> None:
        """
        Heatmap da matriz de confusão a partir do JSON histórico.
        """
        entry = self._get_entry(run_id)
        if entry is None:
            return

        cm_dict = entry.get("confusion_matrix", {})
        tn = cm_dict.get("tn", 0)
        fp = cm_dict.get("fp", 0)
        fn = cm_dict.get("fn", 0)
        tp = cm_dict.get("tp", 0)

        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)

        labels = ["Benigno", "Ataque DDoS"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Predição")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão — {entry.get('label', run_id)}")

        for i in range(2):
            for j in range(2):
                val   = cm[i, j]
                pct   = val / total * 100
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{val:,}\n({pct:.2f}%)",
                        ha="center", va="center", color=color, fontsize=11)

        plt.tight_layout()
        if save:
            slug = run_id.replace(" ", "_")
            path = OUTPUTS_DIR / f"cm_history_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Heatmap salvo → {path}")
        plt.show()
        plt.close()

    def plot_dashboard(self, save: bool = True) -> None:
        """
        Dashboard completo: evolução + tabela de resumo em um único arquivo PNG.
        """
        df = self._to_dataframe()
        if df.empty:
            print("[MetricsPlotter] Nenhuma run registrada.")
            return

        n_runs = len(df)
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

        # ── Plot 1: barras agrupadas de F1/Recall/Precision por run ──────────
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(n_runs)
        w = 0.25
        metrics_to_bar = [
            ("f1",        "F1-Score",  COLORS[0]),
            ("recall",    "Recall",    COLORS[1]),
            ("precision", "Precisão",  COLORS[2]),
        ]
        for i, (metric, label, color) in enumerate(metrics_to_bar):
            vals = df[metric].values * 100
            bars = ax1.bar(x + (i - 1) * w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f"{v:.2f}%", ha="center", va="bottom", fontsize=7)

        ax1.set_xticks(x)
        ax1.set_xticklabels(df["run_id"].tolist(), rotation=20, ha="right")
        ax1.set_ylabel("Score (%)")
        ax1.set_title("F1 / Recall / Precisão por Experimento")
        ax1.legend()
        ax1.set_ylim(max(0, df["f1"].min() * 100 - 3), 101)
        ax1.grid(axis="y", alpha=0.3)

        # ── Plot 2: MCC ao longo das runs ─────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x, df["mcc"].values, "o-", color=COLORS[3], linewidth=2)
        for xi, v in zip(x, df["mcc"].values):
            ax2.annotate(f"{v:.4f}", (xi, v), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["run_id"].tolist(), rotation=25, ha="right")
        ax2.set_ylabel("MCC (0–1)")
        ax2.set_title("MCC por Experimento")
        ax2.set_ylim(max(0, df["mcc"].min() - 0.05), 1.02)
        ax2.grid(alpha=0.3)

        # ── Plot 3: FP e FN (erros) ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        w2 = 0.35
        fp_vals = df["cm_fp"].values if "cm_fp" in df.columns else np.zeros(n_runs)
        fn_vals = df["cm_fn"].values if "cm_fn" in df.columns else np.zeros(n_runs)
        ax3.bar(x - w2/2, fp_vals, w2, label="Falsos Positivos (alarmes)", color=COLORS[4], alpha=0.85)
        ax3.bar(x + w2/2, fn_vals, w2, label="Falsos Negativos (ataques perdidos)", color="crimson", alpha=0.85)
        for xi, v in zip(x, fp_vals):
            ax3.text(xi - w2/2, v + 0.3, str(int(v)), ha="center", fontsize=9)
        for xi, v in zip(x, fn_vals):
            ax3.text(xi + w2/2, v + 0.3, str(int(v)), ha="center", fontsize=9)
        ax3.set_xticks(x)
        ax3.set_xticklabels(df["run_id"].tolist(), rotation=25, ha="right")
        ax3.set_ylabel("Contagem")
        ax3.set_title("Erros de Classificação (FP vs FN)")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

        fig.suptitle("Dashboard de Métricas — Experimentos DDoS MLP (InSDN8)",
                     fontsize=14, fontweight="bold")

        if save:
            path = OUTPUTS_DIR / "dashboard_metrics.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[MetricsPlotter] Dashboard salvo → {path}")
        plt.show()
        plt.close()

    def list_runs(self) -> None:
        """Lista todas as runs registradas."""
        if not self._history:
            print("[MetricsPlotter] Nenhuma run registrada.")
            return
        print(f"\n{'Run ID':<30} {'Label':<25} {'Timestamp':<20} {'F1':>8} {'Recall':>8}")
        print("-" * 95)
        for e in self._history:
            m = e.get("metrics", {})
            print(f"{e['run_id']:<30} {e.get('label',''):<25} "
                  f"{e['timestamp'][:19]:<20} "
                  f"{m.get('f1',0)*100:>7.4f}% "
                  f"{m.get('recall',0)*100:>7.4f}%")

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return []

    def _to_dataframe(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()
        rows = []
        for e in self._history:
            row = {"run_id": e["run_id"], "label": e.get("label", ""), "timestamp": e["timestamp"]}
            row.update(e.get("metrics", {}))
            row.update({f"cm_{k}": v for k, v in e.get("confusion_matrix", {}).items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def _get_entry(self, run_id: str) -> dict | None:
        for e in self._history:
            if e["run_id"] == run_id:
                return e
        print(f"[MetricsPlotter] Run '{run_id}' não encontrada.")
        print(f"  Runs disponíveis: {[e['run_id'] for e in self._history]}")
        return None


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plota métricas do histórico de experimentos.")
    parser.add_argument("--list",    action="store_true", help="Lista todas as runs")
    parser.add_argument("--evolve",  action="store_true", help="Plota evolução das métricas")
    parser.add_argument("--dashboard", action="store_true", help="Dashboard completo")
    parser.add_argument("--radar",   metavar="RUN_ID", help="Radar chart de uma run")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Comparação lado a lado entre duas runs")

    args = parser.parse_args()
    plotter = MetricsPlotter()

    if args.list:
        plotter.list_runs()
    elif args.evolve:
        plotter.plot_evolution()
    elif args.radar:
        plotter.plot_radar(args.radar)
    elif args.compare:
        plotter.plot_comparison(args.compare[0], args.compare[1])
    else:
        # padrão: dashboard completo
        plotter.plot_dashboard()
