"""
Escalonamento de features.

SRP: responsável exclusivamente pelo escalonamento via StandardScaler.

Regra absoluta (boas práticas do curso):
  fit() APENAS no treino — transform() em treino e teste.
  MLP é especialmente sensível à escala: sem escalonamento o treino pode divergir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    Wrapper do StandardScaler que força o uso correto (fit no treino apenas).

    Uso:
        scaler = FeatureScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)     # sem re-fit
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler = StandardScaler()
        self._columns: list[str] = []
        self._fitted: bool = False

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Fit no treino e transforma.

        Returns
        -------
        np.ndarray com as features escalonadas (z-score: média≈0, std≈1).
        """
        if isinstance(X, pd.DataFrame):
            self._columns = X.columns.tolist()

        X_scaled = self._scaler.fit_transform(X)
        self._fitted = True

        mean_of_means = X_scaled.mean(axis=0).mean()
        mean_of_stds  = X_scaled.std(axis=0).mean()
        print(f"[FeatureScaler] Escalonamento aplicado ao treino.")
        print(f"  Média das médias (deve ser ≈0): {mean_of_means:.4f}")
        print(f"  Média dos desvios (deve ser ≈1): {mean_of_stds:.4f}")

        return X_scaled

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Aplica escalonamento ao teste com os parâmetros do treino.

        Não faz re-fit — evita data leakage.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler: chame fit_transform() no treino antes de transform()."
            )
        return self._scaler.transform(X)

    @property
    def scaler(self) -> StandardScaler:
        """Acesso ao scaler fitado para persistência."""
        return self._scaler
