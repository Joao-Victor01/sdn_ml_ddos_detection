"""
Selecao de features em duas camadas:
  1. filtro de dominio no loader (26 features estatisticas de fluxo)
  2. VarianceThreshold no pipeline (remove features completamente constantes)

A importancia supervisionada das features — quais pesam mais para o modelo —
e calculada separadamente pelo PermutationImportanceAnalyzer apos o treino.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from ml.config import OUTPUTS_DIR, VARIANCE_THRESHOLD


class FeatureSelector:
    """Remove apenas features constantes após o filtro de domínio."""

    def __init__(
        self,
        variance_threshold: float = VARIANCE_THRESHOLD,
        output_dir: Path | str = OUTPUTS_DIR,
    ) -> None:
        self._var_thresh = VarianceThreshold(threshold=variance_threshold)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._selected_features: list[str] = []
        self._fitted = False

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        del y  # a seleção por variância independe do alvo

        cols_before = X.columns.tolist()
        self._var_thresh.fit(X)
        self._selected_features = X.columns[self._var_thresh.get_support()].tolist()

        removed_var = sorted(set(cols_before) - set(self._selected_features))
        print(f"[FeatureSelector] VarianceThreshold removeu {len(removed_var)} feature(s).")
        if removed_var:
            for feature in removed_var:
                print(f"  - {feature}")

        print(
            f"[FeatureSelector] Features mantidas apos selecao: "
            f"{len(self._selected_features)}"
        )
        self._fitted = True
        return self.transform(X)

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
