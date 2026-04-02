"""
Seleção de features em duas etapas:

  1. VarianceThreshold — remove features com variância ≤ threshold (zero info).
  2. SHAP (TreeExplainer) — rankeia features por importância e seleciona top-N.

SRP: este módulo lida apenas com seleção de features.

Nota sobre o dataset insdn8:
  O dataset já contém apenas 8 features pré-selecionadas do InSDN original.
  O VarianceThreshold ainda é executado para detectar constantes acidentais.
  A análise SHAP é mantida para confirmar a importância relativa das features
  e gerar o relatório/plot — mesmo que nenhuma feature seja descartada ao final.
  Se N_FEATURES_TO_SELECT=None (padrão), mantém todas as features sobreviventes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from ml.config import (
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    N_FEATURES_TO_SELECT,
    VARIANCE_THRESHOLD,
    OUTPUTS_DIR,
)


class FeatureSelector:
    """
    Seleciona as features mais importantes usando VarianceThreshold + SHAP.

    Uso correto (evita data leakage — SHAP só vê dados de treino):
        selector = FeatureSelector()
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel  = selector.transform(X_test)
    """

    def __init__(
        self,
        n_features: int | None = N_FEATURES_TO_SELECT,
        variance_threshold: float = VARIANCE_THRESHOLD,
        shap_sample_size: int = SHAP_SAMPLE_SIZE,
        random_state: int = RANDOM_STATE,
        save_plots: bool = True,
    ) -> None:
        self._n_features        = n_features
        self._var_thresh        = VarianceThreshold(threshold=variance_threshold)
        self._shap_sample_size  = shap_sample_size
        self._random_state      = random_state
        self._save_plots        = save_plots
        self._selected_features: list[str] = []
        self._shap_importance: pd.DataFrame | None = None
        self._fitted: bool = False

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """
        Ajusta o seletor ao treino e retorna as features selecionadas.

        Etapas:
          1. VarianceThreshold.fit_transform(X_train)
          2. RandomForest + SHAP nos dados de treino (amostra)
          3. Filtra top-N features por importância SHAP
        """
        # ── Etapa 1: VarianceThreshold ─────────────────────────────────────────
        cols_before = X.columns.tolist()
        self._var_thresh.fit(X)

        X_var = pd.DataFrame(
            self._var_thresh.transform(X),
            columns=X.columns[self._var_thresh.get_support()].tolist(),
            index=X.index,
        )

        removed_var = set(cols_before) - set(X_var.columns)
        print(f"[FeatureSelector] VarianceThreshold removeu {len(removed_var)} feature(s):")
        if removed_var:
            for f in sorted(removed_var):
                print(f"  - {f}")
        else:
            print("  (nenhuma feature removida — todas possuem variância > 0)")
        print(f"[FeatureSelector] Features restantes após VarianceThreshold: {X_var.shape[1]}")

        # ── Etapa 2: SHAP com RandomForest ────────────────────────────────────
        importance_df = self._compute_shap_importance(X_var, y)
        self._shap_importance = importance_df

        print("\n[FeatureSelector] Importância SHAP (todas as features):")
        print(importance_df.to_string(index=False))

        # ── Etapa 3: Seleção top-N ─────────────────────────────────────────────
        if self._n_features is not None:
            n = min(self._n_features, len(importance_df))
            self._selected_features = importance_df.head(n)["feature"].tolist()
        else:
            # Manter todas as features que sobreviveram ao VarianceThreshold
            self._selected_features = importance_df["feature"].tolist()

        print(f"\n[FeatureSelector] Features selecionadas ({len(self._selected_features)}):")
        for i, f in enumerate(self._selected_features, 1):
            print(f"  {i:2d}. {f}")

        if self._save_plots:
            self._save_shap_plot(X_var, y)

        self._fitted = True
        return X_var[self._selected_features].reset_index(drop=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a seleção de features ao conjunto de TESTE.

        Não recalcula SHAP — usa as features selecionadas no fit_transform().
        """
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
        """Lista de features selecionadas (disponível após fit_transform)."""
        if not self._fitted:
            raise RuntimeError("FeatureSelector ainda não foi ajustado.")
        return self._selected_features.copy()

    @property
    def variance_filter(self) -> VarianceThreshold:
        """Acesso ao VarianceThreshold fitado para persistência."""
        return self._var_thresh

    @property
    def shap_importance(self) -> pd.DataFrame | None:
        """DataFrame com importâncias SHAP calculadas no treino."""
        return self._shap_importance

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _compute_shap_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """
        Treina RandomForest rápido e calcula SHAP values.

        Usa amostra aleatória (SHAP_SAMPLE_SIZE) para ser eficiente.
        TreeExplainer é exato para modelos baseados em árvore.
        """
        try:
            import shap
        except ImportError:
            print("[FeatureSelector] shap não instalado — usando feature_importances_ como fallback.")
            return self._fallback_importance(X, y)

        sample_size = min(self._shap_sample_size, len(X))
        idx = np.random.RandomState(self._random_state).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = np.array(y)[idx]

        print(f"\n[FeatureSelector] Treinando RandomForest auxiliar para SHAP "
              f"(amostra: {sample_size:,})...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self._random_state,
            n_jobs=-1,
        )
        rf.fit(X_sample, y_sample)

        print("[FeatureSelector] Calculando SHAP values...")
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        # Compatibilidade com diferentes versões do SHAP:
        #   SHAP < 0.41 : lista de arrays 2D  → shap_values[1] shape (n, f)
        #   SHAP >= 0.41: array 3D             → shap_values shape (n, f, classes)
        if isinstance(shap_values, list):
            # versão antiga: lista [classe_0, classe_1]
            shap_vals = shap_values[1]
        elif shap_values.ndim == 3:
            # versão nova: (n_samples, n_features, n_classes) — pegar classe 1
            shap_vals = shap_values[:, :, 1]
        else:
            # array 2D direto (alguns modelos retornam isso)
            shap_vals = shap_values

        importances = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({
            "feature":          X.columns.tolist(),
            "shap_importance":  importances,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

        return df

    def _fallback_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Fallback: usa feature_importances_ do RandomForest quando SHAP não disponível."""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self._random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        df = pd.DataFrame({
            "feature":          X.columns.tolist(),
            "shap_importance":  rf.feature_importances_,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
        return df

    def _save_shap_plot(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """Gera e salva o beeswarm plot do SHAP em outputs/."""
        try:
            import shap
            import matplotlib.pyplot as plt

            sample_size = min(self._shap_sample_size, len(X))
            idx = np.random.RandomState(self._random_state).choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[idx]
            y_sample = np.array(y)[idx]

            rf = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=self._random_state, n_jobs=-1,
            )
            rf.fit(X_sample, y_sample)
            explainer   = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                vals = shap_values[1]
            elif shap_values.ndim == 3:
                vals = shap_values[:, :, 1]
            else:
                vals = shap_values

            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

            # Beeswarm plot
            shap.summary_plot(vals, X_sample, max_display=20, show=False)
            plt.title("SHAP Summary — InSDN8 (treino)")
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Bar plot
            shap.summary_plot(vals, X_sample, plot_type="bar", max_display=20, show=False)
            plt.title("SHAP Feature Importance — InSDN8 (treino)")
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[FeatureSelector] Plots SHAP salvos em {OUTPUTS_DIR}/")

        except Exception as e:
            print(f"[FeatureSelector] Não foi possível salvar plots SHAP: {e}")
