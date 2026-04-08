"""
Permutation Importance para qualquer estimador sklearn.

Mede a contribuição de cada feature embaralhando seus valores uma por vez
e observando a queda no score do modelo. Ao contrário da feature importance
nativa do RandomForest, funciona com qualquer estimador — incluindo o MLP —
e é calculada no conjunto de TESTE, refletindo a importância real de
generalização, não a importância de memorização do treino.

Referência: Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml.config import (
    OUTPUTS_DIR,
    PERMUTATION_IMPORTANCE_N_REPEATS,
    PERMUTATION_IMPORTANCE_SCORING,
    RANDOM_STATE,
)
from ml.utils.plotting import get_pyplot


class PermutationImportanceAnalyzer:
    """
    Calcula e plota permutation importance para qualquer estimador sklearn.

    A análise é realizada no conjunto de teste (X_test, y_test) para capturar
    quais features são realmente úteis para a generalização do modelo,
    e não apenas para o ajuste nos dados de treino.

    Responsabilidade única (SRP): gerar o ranking de importância por permutação
    e seus artefatos visuais. Não treina, não avalia, não persiste modelos.
    """

    def __init__(
        self,
        output_dir: Path | str = OUTPUTS_DIR,
        n_repeats: int = PERMUTATION_IMPORTANCE_N_REPEATS,
        scoring: str = PERMUTATION_IMPORTANCE_SCORING,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._n_repeats = n_repeats
        self._scoring = scoring
        self._random_state = random_state

    def analyze(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        label: str,
    ) -> pd.DataFrame:
        """
        Executa permutation importance no conjunto de teste e salva os artefatos.

        Parâmetros
        ----------
        model:
            Estimador sklearn já treinado.
        X_test:
            Features do conjunto de teste (após todo o pré-processamento).
        y_test:
            Rótulos do conjunto de teste.
        label:
            Identificador usado no nome dos arquivos de saída.

        Retorna
        -------
        pd.DataFrame
            DataFrame com colunas 'feature', 'importance_mean', 'importance_std',
            ordenado por importância decrescente.
        """
        print(
            f"\n[PermutationImportance] Calculando com {self._n_repeats} repeticoes "
            f"por feature | scoring={self._scoring} | n_features={X_test.shape[1]}"
        )

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=self._n_repeats,
            scoring=self._scoring,
            random_state=self._random_state,
            n_jobs=-1,
        )

        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_test.columns.tolist(),
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        print("\n[PermutationImportance] Ranking de importância (média ± std da queda em f1_macro):")
        print(importance_df.to_string(index=False))

        plot_path = self._plot(importance_df, label=label)
        print(f"[PermutationImportance] Plot salvo em {plot_path}")

        report_path = self._save_report(importance_df, label=label)
        print(f"[PermutationImportance] Relatório salvo em {report_path}")

        return importance_df

    def _plot(self, importance_df: pd.DataFrame, *, label: str) -> Path:
        """Gera gráfico de barras horizontais com mean ± std para as top 20 features."""
        top_df = importance_df.head(20).sort_values("importance_mean", ascending=True)

        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            top_df["feature"],
            top_df["importance_mean"],
            xerr=top_df["importance_std"],
            color="steelblue",
            ecolor="black",
            capsize=4,
        )
        ax.set_xlabel(f"Queda média em {self._scoring} ao embaralhar a feature")
        ax.set_title(f"Permutation Importance (MLP) — {label}")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8, label="sem impacto")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = self._output_dir / f"permutation_importance_{label}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _save_report(self, importance_df: pd.DataFrame, *, label: str) -> Path:
        """Salva ranking completo em CSV para análise posterior."""
        path = self._output_dir / f"permutation_importance_{label}.csv"
        importance_df.to_csv(path, index=False)
        return path
