"""
Curvas de aprendizado e diagnóstico de overfit.

Gera três tipos de visualização para investigar overfitting:

  1. plot_learning_curve  — Acurácia/F1 treino vs. validação por tamanho de treino.
     Diagnóstico: se treino >> validação em todos os pontos → overfit.

  2. plot_train_val_gap   — Gap entre score de treino e validação ao longo das épocas
     (para MLP). Indica quando o modelo começa a memorizar.

  3. plot_cv_scores       — Distribuição dos scores dos folds do CV.
     Alta variância entre folds → instabilidade, possível overfit.

  4. plot_overfit_dashboard — Dashboard completo com os três gráficos.

Uso standalone (CLI):
    python -m ml.utils.learning_curve_plotter \\
        --model-path models_triclass/rf_triclass.joblib \\
        --data-dir   dataset/InSDN_DatasetCSV/

Uso como módulo:
    from ml.utils.learning_curve_plotter import LearningCurvePlotter
    plotter = LearningCurvePlotter(model, X_train, y_train)
    plotter.plot_learning_curve()
    plotter.plot_cv_scores(cv_results_dict)
    plotter.plot_overfit_dashboard(mlp_model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve

# ── Diretório padrão para outputs ─────────────────────────────────────────────
try:
    from ml.triclass.config import OUTPUTS_TRICLASS as _OUTPUTS_DIR
except ImportError:
    _OUTPUTS_DIR = Path("outputs_triclass")


class LearningCurvePlotter:
    """
    Gerador de curvas de aprendizado para diagnóstico de overfit.

    Parâmetros
    ----------
    model     : estimador sklearn (RF, MLP Pipeline, etc.)
    X_train   : features de treino (pós-VT, pré-SMOTE para diagnóstico real)
    y_train   : labels de treino
    scoring   : métrica de avaliação ('f1_macro' recomendado para triclasse)
    cv        : número de folds ou objeto CV
    output_dir: diretório para salvar os PNGs
    """

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = "f1_macro",
        cv: int = 5,
        output_dir: Path | str = _OUTPUTS_DIR,
        random_state: int = 42,
    ) -> None:
        self._model       = model
        self._X           = np.array(X_train)
        self._y           = np.array(y_train)
        self._scoring     = scoring
        self._cv          = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        self._out         = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._random_state = random_state

    # ── API pública ────────────────────────────────────────────────────────────

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray | None = None,
        save: bool = True,
        title: str = "Curva de Aprendizado",
    ) -> dict[str, np.ndarray]:
        """
        Plota score de treino vs. validação para diferentes tamanhos de treino.

        Como interpretar:
          - Treino alto, Validação baixa → overfit (modelo memoriza).
          - Treino baixo, Validação baixa → underfit (modelo não aprendeu).
          - Treino ≈ Validação, ambos altos → generalização adequada.
          - Gap decresce com mais dados → mais dados ajudariam.
          - Gap estável com mais dados → mudança de modelo necessária.

        Returns
        -------
        dict com 'train_sizes', 'train_scores_mean', 'val_scores_mean'
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.10, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            self._model,
            self._X, self._y,
            train_sizes=train_sizes,
            cv=self._cv,
            scoring=self._scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=self._random_state,
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        gap        = train_mean - val_mean

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Plot 1: curvas treino e validação ──────────────────────────────────
        ax = axes[0]
        ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue",
                label=f"Treino ({self._scoring})")
        ax.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.15, color="steelblue")
        ax.plot(train_sizes_abs, val_mean, "s--", color="darkorange",
                label=f"Validação CV ({self._scoring})")
        ax.fill_between(train_sizes_abs,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.15, color="darkorange")

        ax.set_xlabel("Amostras de Treino")
        ax.set_ylabel(f"Score ({self._scoring})")
        ax.set_title(f"{title}\nTreino vs. Validação")
        ax.legend(loc="lower right")
        ax.set_ylim(max(0, min(val_mean) - 0.05), 1.02)
        ax.grid(alpha=0.3)

        # Anotar gap no último ponto
        last_gap = gap[-1]
        ax.annotate(
            f"Gap final: {last_gap:.4f}",
            xy=(train_sizes_abs[-1], (train_mean[-1] + val_mean[-1]) / 2),
            xytext=(-80, 0),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="crimson",
        )

        # ── Plot 2: gap ao longo dos tamanhos de treino ─────────────────────
        ax2 = axes[1]
        ax2.plot(train_sizes_abs, gap, "^-", color="crimson", label="Gap (Treino − Validação)")
        ax2.fill_between(train_sizes_abs, 0, gap, alpha=0.15, color="crimson")
        ax2.axhline(0.05, linestyle="--", color="gray", linewidth=0.8,
                    label="Threshold 0.05 (referência)")
        ax2.set_xlabel("Amostras de Treino")
        ax2.set_ylabel("Gap de Score")
        ax2.set_title("Gap Treino − Validação\n(overfit > 0.05 é preocupante)")
        ax2.legend()
        ax2.set_ylim(-0.02, max(gap) + 0.05)
        ax2.grid(alpha=0.3)

        # Diagnóstico automático
        overfit_flag = last_gap > 0.05
        underfit_flag = val_mean[-1] < 0.70
        diag = []
        if overfit_flag:
            diag.append("OVERFIT detectado (gap > 0.05 com treino completo)")
        if underfit_flag:
            diag.append("UNDERFIT suspeito (val < 0.70)")
        if not diag:
            diag.append("Sem sinais claros de overfit/underfit")

        fig.suptitle(
            f"{title}\nDiagnóstico: {' | '.join(diag)}",
            fontsize=11, fontweight="bold",
            color="crimson" if overfit_flag else "darkgreen",
        )

        plt.tight_layout()
        if save:
            slug = title.lower().replace(" ", "_").replace("/", "_")
            path = self._out / f"learning_curve_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[LearningCurvePlotter] Salvo → {path.name}")
        plt.show()
        plt.close()

        return {
            "train_sizes":      train_sizes_abs,
            "train_scores_mean": train_mean,
            "val_scores_mean":   val_mean,
            "gap":               gap,
            "overfit_flag":      overfit_flag,
        }

    def plot_cv_scores(
        self,
        cv_results: dict[str, tuple[float, float]],
        label: str = "Modelo",
        save: bool = True,
    ) -> None:
        """
        Boxplot / barras dos scores por fold do CV.

        Alta variância entre folds → modelo instável.
        Variância baixa + score alto → boa generalização.

        Parameters
        ----------
        cv_results : dict retornado por TriclassTrainer.cross_validate_*
                     formato: {'f1_macro': (mean, std), 'accuracy': (mean, std)}
        """
        metrics  = list(cv_results.keys())
        means    = [cv_results[m][0] for m in metrics]
        stds     = [cv_results[m][1] for m in metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics))
        bars = ax.bar(x, means, yerr=stds, capsize=6, color="steelblue",
                      alpha=0.8, ecolor="crimson")

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.005,
                f"{mean:.4f}\n±{std:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylabel("Score (CV médio)")
        ax.set_ylim(max(0, min(means) - 3 * max(stds) - 0.05), 1.05)
        ax.set_title(f"Scores de Validação Cruzada — {label}\n"
                     f"(barra de erro = ±1 desvio padrão entre folds)")
        ax.grid(axis="y", alpha=0.3)

        # Diagnóstico de instabilidade
        max_std = max(stds) if stds else 0
        if max_std > 0.05:
            ax.text(0.98, 0.02,
                    f"⚠ Instabilidade: std máx = {max_std:.4f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="crimson",
                    bbox=dict(boxstyle="round", facecolor="lightyellow"))

        plt.tight_layout()
        if save:
            slug = label.lower().replace(" ", "_")
            path = self._out / f"cv_scores_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[LearningCurvePlotter] CV scores salvo → {path.name}")
        plt.show()
        plt.close()

    def plot_mlp_loss_gap(
        self,
        mlp_pipeline,
        label: str = "MLP",
        save: bool = True,
    ) -> None:
        """
        Plota loss de treino vs. score de validação interna do MLP por época.

        Requer MLP com early_stopping=True (gera validation_scores_).

        Como interpretar:
          - Loss treino cai mas val_score estagna → overfit começa.
          - Ponto de cruzamento val_score / loss → quando parar (early_stop).
          - Gap crescente após cruzamento → overfitting progressivo.
        """
        mlp = (
            mlp_pipeline.named_steps.get("mlp", None)
            if hasattr(mlp_pipeline, "named_steps")
            else mlp_pipeline
        )

        if mlp is None or not hasattr(mlp, "loss_curve_"):
            print("[LearningCurvePlotter] MLP sem loss_curve_ — treinar antes de plotar.")
            return

        loss_curve = mlp.loss_curve_
        val_scores = getattr(mlp, "validation_scores_", None)

        fig, ax1 = plt.subplots(figsize=(11, 5))
        epochs = np.arange(1, len(loss_curve) + 1)

        ax1.plot(epochs, loss_curve, color="steelblue", label="Loss de Treino", linewidth=1.5)
        ax1.set_xlabel("Épocas")
        ax1.set_ylabel("Loss de Treino", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")

        if val_scores is not None and len(val_scores) > 0:
            ax2 = ax1.twinx()
            # Alinhar val_scores ao mesmo comprimento do loss_curve
            v_epochs = np.linspace(1, len(loss_curve), len(val_scores))
            ax2.plot(v_epochs, val_scores, color="darkorange",
                     linestyle="--", label="Score Validação Interna", linewidth=1.5)
            ax2.set_ylabel("Score Validação", color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange")

            # Detectar ponto de overfitting: val_score começa a cair
            if len(val_scores) > 3:
                best_val_epoch = int(np.argmax(val_scores))
                best_epoch_abs = int(v_epochs[best_val_epoch])
                ax1.axvline(best_epoch_abs, color="green", linestyle=":",
                            linewidth=1.5, label=f"Melhor val (época {best_epoch_abs})")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        ax1.set_title(
            f"Convergência MLP — {label}\n"
            f"Loss treino (azul) | Score validação interna (laranja)"
        )
        ax1.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            slug = label.lower().replace(" ", "_")
            path = self._out / f"mlp_loss_gap_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[LearningCurvePlotter] Loss gap salvo → {path.name}")
        plt.show()
        plt.close()

    def plot_overfit_dashboard(
        self,
        cv_rf: dict[str, tuple[float, float]] | None = None,
        cv_mlp: dict[str, tuple[float, float]] | None = None,
        mlp_pipeline=None,
        train_sizes: np.ndarray | None = None,
        label: str = "Triclasse",
        save: bool = True,
    ) -> None:
        """
        Dashboard completo de diagnóstico de overfit em um único arquivo.

        Inclui: curva de aprendizado + CV RF + CV MLP + loss MLP.

        Parameters
        ----------
        cv_rf         : dict de CV do RF (de TriclassTrainer.cross_validate_rf)
        cv_mlp        : dict de CV do MLP (de TriclassTrainer.cross_validate_mlp)
        mlp_pipeline  : Pipeline(StandardScaler → MLP) treinado
        train_sizes   : pontos da curva de aprendizado (default: 10 pontos)
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.10, 1.0, 8)

        # Calcular curva de aprendizado
        _, train_scores, val_scores = learning_curve(
            self._model, self._X, self._y,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=3, shuffle=True,
                               random_state=self._random_state),
            scoring=self._scoring,
            n_jobs=-1,
        )
        train_sizes_abs = (train_sizes * len(self._X)).astype(int)

        train_mean = train_scores.mean(axis=1)
        val_mean   = val_scores.mean(axis=1)
        gap        = train_mean - val_mean

        # Número de subplots dinâmico
        n_rows = 2
        n_cols = 2
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                hspace=0.45, wspace=0.35)

        # ── Plot 1: curva de aprendizado ───────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(train_sizes_abs, train_mean, "o-", color="steelblue",
                 label=f"Treino ({self._scoring})")
        ax1.plot(train_sizes_abs, val_mean, "s--", color="darkorange",
                 label="Validação CV")
        ax1.set_xlabel("Amostras de Treino")
        ax1.set_ylabel("Score")
        ax1.set_title("Curva de Aprendizado")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # ── Plot 2: gap de overfit ─────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(train_sizes_abs, gap, "^-", color="crimson", label="Gap")
        ax2.fill_between(train_sizes_abs, 0, gap, alpha=0.15, color="crimson")
        ax2.axhline(0.05, linestyle="--", color="gray", linewidth=0.8)
        ax2.set_xlabel("Amostras de Treino")
        ax2.set_ylabel("Gap (Treino − Val)")
        ax2.set_title("Gap de Overfit\n(> 0.05 = preocupante)")
        ax2.grid(alpha=0.3)

        # ── Plot 3: CV RF ──────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        if cv_rf:
            metrics  = list(cv_rf.keys())
            means_rf = [cv_rf[m][0] for m in metrics]
            stds_rf  = [cv_rf[m][1] for m in metrics]
            x3 = np.arange(len(metrics))
            ax3.bar(x3, means_rf, yerr=stds_rf, capsize=5,
                    color="steelblue", alpha=0.8, ecolor="crimson")
            ax3.set_xticks(x3)
            ax3.set_xticklabels(metrics, rotation=15)
            ax3.set_title("CV Scores — RF\n(±std entre folds)")
            ax3.set_ylim(0, 1.05)
            ax3.grid(axis="y", alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "CV RF não disponível",
                     transform=ax3.transAxes, ha="center")

        # ── Plot 4: CV MLP ─────────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        if cv_mlp:
            metrics   = list(cv_mlp.keys())
            means_mlp = [cv_mlp[m][0] for m in metrics]
            stds_mlp  = [cv_mlp[m][1] for m in metrics]
            x4 = np.arange(len(metrics))
            ax4.bar(x4, means_mlp, yerr=stds_mlp, capsize=5,
                    color="darkorange", alpha=0.8, ecolor="crimson")
            ax4.set_xticks(x4)
            ax4.set_xticklabels(metrics, rotation=15)
            ax4.set_title("CV Scores — MLP Pipeline\n(±std entre folds)")
            ax4.set_ylim(0, 1.05)
            ax4.grid(axis="y", alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "CV MLP não disponível",
                     transform=ax4.transAxes, ha="center")

        # Diagnóstico no título geral
        last_gap = gap[-1]
        diag = "OVERFIT detectado" if last_gap > 0.05 else "Sem overfit detectado"
        color = "crimson" if last_gap > 0.05 else "darkgreen"

        fig.suptitle(
            f"Dashboard de Diagnóstico de Overfit — {label}\n{diag}  |  "
            f"Gap final={last_gap:.4f}",
            fontsize=12, fontweight="bold", color=color,
        )

        if save:
            slug = label.lower().replace(" ", "_")
            path = self._out / f"overfit_dashboard_{slug}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[LearningCurvePlotter] Dashboard salvo → {path.name}")
        plt.show()
        plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plota curvas de aprendizado para diagnóstico de overfit."
    )
    parser.add_argument("--model-path", required=True,
                        help="Caminho para o .joblib do modelo treinado")
    parser.add_argument("--scoring", default="f1_macro",
                        help="Métrica de scoring (default: f1_macro)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Número de folds do CV (default: 5)")
    parser.add_argument("--label", default="Modelo",
                        help="Nome do modelo para os gráficos")

    args = parser.parse_args()

    import joblib
    model = joblib.load(args.model_path)
    print(f"Modelo carregado: {args.model_path}")
    print("Para usar o plotter, instancie LearningCurvePlotter(model, X_train, y_train)")
    print("e chame .plot_learning_curve(), .plot_cv_scores() ou .plot_overfit_dashboard()")
