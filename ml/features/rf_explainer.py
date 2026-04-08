"""
Explicabilidade específica do RandomForest.

Gera dois artefatos:
  - feature importance nativa do modelo
  - ranking SHAP (quando disponível)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.config import OUTPUTS_DIR, RANDOM_STATE, SHAP_SAMPLE_SIZE
from ml.utils.plotting import get_pyplot


class RandomForestExplainer:
    """Gera relatórios de explicabilidade apenas para RandomForest."""

    def __init__(
        self,
        output_dir: Path | str = OUTPUTS_DIR,
        shap_sample_size: int = SHAP_SAMPLE_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._shap_sample_size = shap_sample_size
        self._random_state = random_state

    def explain(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
        label: str,
    ) -> dict[str, Path]:
        artifacts: dict[str, Path] = {}
        importance_df = pd.DataFrame(
            {
                "feature": X.columns.tolist(),
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        print("\n[RandomForestExplainer] Ranking de feature importance:")
        print(importance_df.to_string(index=False))
        artifacts["feature_importance"] = self._plot_importance(
            importance_df,
            label=label,
            metric_name="importance",
            title="RandomForest Feature Importance",
            filename=f"rf_feature_importance_{label}.png",
        )

        shap_df = self._compute_shap_importance(model, X)
        if shap_df is not None:
            print("\n[RandomForestExplainer] Ranking de importancia SHAP:")
            print(shap_df.to_string(index=False))
            artifacts["shap_importance"] = self._plot_importance(
                shap_df,
                label=label,
                metric_name="shap_importance",
                title="RandomForest SHAP Importance",
                filename=f"rf_shap_importance_{label}.png",
            )

        return artifacts

    def _compute_shap_importance(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
    ) -> pd.DataFrame | None:
        try:
            import shap
        except ImportError:
            print("[RandomForestExplainer] shap nao instalado; pulando ranking SHAP.")
            return None

        sample_size = min(self._shap_sample_size, len(X))
        if sample_size == 0:
            return None

        rng = np.random.RandomState(self._random_state)
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]

        print(
            f"[RandomForestExplainer] Calculando SHAP com amostra de {sample_size:,} linhas..."
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        importances = self._reduce_multiclass_shap(shap_values, X_sample.shape[1])

        return pd.DataFrame(
            {
                "feature": X_sample.columns.tolist(),
                "shap_importance": importances,
            }
        ).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    def _reduce_multiclass_shap(
        self,
        shap_values: object,
        n_features: int,
    ) -> np.ndarray:
        if isinstance(shap_values, list):
            class_means = [np.abs(values).mean(axis=0) for values in shap_values]
            return np.mean(class_means, axis=0)

        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 2:
            return np.abs(shap_array).mean(axis=0)
        if shap_array.ndim == 3:
            if shap_array.shape[1] == n_features:
                return np.abs(shap_array).mean(axis=(0, 2))
            if shap_array.shape[2] == n_features:
                return np.abs(shap_array).mean(axis=(0, 1))

        raise ValueError(
            f"Formato inesperado de SHAP values: shape={getattr(shap_array, 'shape', None)}"
        )

    def _plot_importance(
        self,
        importance_df: pd.DataFrame,
        *,
        label: str,
        metric_name: str,
        title: str,
        filename: str,
    ) -> Path:
        top_df = importance_df.head(20).sort_values(metric_name, ascending=True)

        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_df["feature"], top_df[metric_name], color="forestgreen")
        ax.set_xlabel("Importancia media")
        ax.set_title(f"{title} — {label}")
        plt.tight_layout()

        path = self._output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RandomForestExplainer] Plot salvo em {path}")
        return path
