"""
Selecao de features em duas etapas:
  1. VarianceThreshold
  2. Importancia SHAP (ou fallback por RandomForest)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from ml.config import (
    N_FEATURES_TO_SELECT,
    OUTPUTS_DIR,
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    VARIANCE_THRESHOLD,
)


class FeatureSelector:
    """Seleciona features com VarianceThreshold e ranking por importancia."""

    def __init__(
        self,
        n_features: int | None = N_FEATURES_TO_SELECT,
        variance_threshold: float = VARIANCE_THRESHOLD,
        shap_sample_size: int = SHAP_SAMPLE_SIZE,
        random_state: int = RANDOM_STATE,
        save_plots: bool = True,
    ) -> None:
        self._n_features = n_features
        self._var_thresh = VarianceThreshold(threshold=variance_threshold)
        self._shap_sample_size = shap_sample_size
        self._random_state = random_state
        self._save_plots = save_plots
        self._selected_features: list[str] = []
        self._shap_importance: pd.DataFrame | None = None
        self._fitted = False

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        cols_before = X.columns.tolist()
        self._var_thresh.fit(X)
        surviving_cols = X.columns[self._var_thresh.get_support()].tolist()

        X_var = pd.DataFrame(
            self._var_thresh.transform(X),
            columns=surviving_cols,
            index=X.index,
        )

        removed_var = sorted(set(cols_before) - set(surviving_cols))
        print(f"[FeatureSelector] VarianceThreshold removeu {len(removed_var)} feature(s).")
        if removed_var:
            for feature in removed_var:
                print(f"  - {feature}")

        importance_df = self._compute_feature_importance(X_var, y)
        self._shap_importance = importance_df

        if self._n_features is None:
            self._selected_features = importance_df["feature"].tolist()
        else:
            n_keep = min(self._n_features, len(importance_df))
            self._selected_features = importance_df.head(n_keep)["feature"].tolist()

        print("\n[FeatureSelector] Ranking de importancia:")
        print(importance_df.to_string(index=False))
        print(f"\n[FeatureSelector] Features selecionadas ({len(self._selected_features)}):")
        for idx, feature in enumerate(self._selected_features, start=1):
            print(f"  {idx:2d}. {feature}")

        if self._save_plots:
            self._save_importance_plot(importance_df)

        self._fitted = True
        return X_var[self._selected_features].reset_index(drop=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "FeatureSelector: chame fit_transform() no treino antes de transform()."
            )

        X_var = pd.DataFrame(
            self._var_thresh.transform(X),
            columns=X.columns[self._var_thresh.get_support()].tolist(),
            index=X.index,
        )
        return X_var[self._selected_features].reset_index(drop=True)

    @property
    def selected_features(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("FeatureSelector ainda nao foi ajustado.")
        return self._selected_features.copy()

    @property
    def variance_filter(self) -> VarianceThreshold:
        return self._var_thresh

    @property
    def shap_importance(self) -> pd.DataFrame | None:
        return self._shap_importance

    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        try:
            import shap
        except ImportError:
            print("[FeatureSelector] shap nao instalado; usando feature_importances_ como fallback.")
            return self._fallback_importance(X, y)

        sample_size = min(self._shap_sample_size, len(X))
        rng = np.random.RandomState(self._random_state)
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = np.array(y)[idx]

        print(
            f"[FeatureSelector] Treinando RandomForest auxiliar para SHAP "
            f"(amostra: {sample_size:,})..."
        )
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=self._random_state,
            n_jobs=-1,
        )
        rf.fit(X_sample, y_sample)

        print("[FeatureSelector] Calculando importancias SHAP...")
        explainer = shap.TreeExplainer(rf)
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

    def _fallback_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=self._random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        return pd.DataFrame(
            {
                "feature": X.columns.tolist(),
                "shap_importance": rf.feature_importances_,
            }
        ).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    def _save_importance_plot(self, importance_df: pd.DataFrame) -> None:
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            top_df = importance_df.head(20).sort_values("shap_importance", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_df["feature"], top_df["shap_importance"], color="steelblue")
            ax.set_xlabel("Importancia media absoluta")
            ax.set_title("Feature Importance (SHAP/RandomForest)")
            plt.tight_layout()
            path = OUTPUTS_DIR / "feature_importance_multiclass.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[FeatureSelector] Plot de importancia salvo em {path}")
        except Exception as exc:
            print(f"[FeatureSelector] Nao foi possivel salvar o plot de importancia: {exc}")
