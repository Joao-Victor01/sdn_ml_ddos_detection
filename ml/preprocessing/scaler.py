"""
Escalonamento de features.

SRP: responsável exclusivamente pelo escalonamento via StandardScaler.

Regra absoluta (boas práticas do curso):
  fit() APENAS no treino — transform() em treino e teste.
  MLP é especialmente sensível à escala: sem escalonamento o treino pode divergir.
  Colunas binarias 0/1 devem ser preservadas sem padronizacao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.config import BINARY_PASSTHROUGH_FEATURES


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
        self._binary_columns: list[str] = []
        self._scaled_columns: list[str] = []
        self._fitted: bool = False

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Fit no treino e transforma.

        Returns
        -------
        np.ndarray com as features escalonadas (z-score: média≈0, std≈1).
        """
        X_df, return_dataframe = self._to_dataframe(X, use_known_columns=False)
        self._columns = X_df.columns.tolist()
        self._binary_columns = self._detect_binary_columns(X_df)
        self._scaled_columns = [col for col in self._columns if col not in self._binary_columns]

        X_scaled = X_df.copy()
        if self._scaled_columns:
            X_scaled.loc[:, self._scaled_columns] = self._scaler.fit_transform(
                X_df[self._scaled_columns]
            )
        self._fitted = True

        print(f"[FeatureScaler] Escalonamento aplicado ao treino.")
        print(
            f"  Colunas padronizadas : {len(self._scaled_columns)} | "
            f"Colunas preservadas : {len(self._binary_columns)}"
        )
        if self._binary_columns:
            print(f"  Binarias sem escala  : {self._binary_columns}")

        if self._scaled_columns:
            mean_of_means = X_scaled[self._scaled_columns].mean(axis=0).mean()
            mean_of_stds = X_scaled[self._scaled_columns].std(axis=0).mean()
            print(f"  Média das médias (deve ser ≈0): {mean_of_means:.4f}")
            print(f"  Média dos desvios (deve ser ≈1): {mean_of_stds:.4f}")
        else:
            print("  Nenhuma coluna continua exigiu padronizacao.")

        if return_dataframe:
            return X_scaled
        return X_scaled.to_numpy()

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Aplica escalonamento ao teste com os parâmetros do treino.

        Não faz re-fit — evita data leakage.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler: chame fit_transform() no treino antes de transform()."
            )
        X_df, return_dataframe = self._to_dataframe(X, use_known_columns=True)
        X_scaled = X_df.copy()
        if self._scaled_columns:
            X_scaled.loc[:, self._scaled_columns] = self._scaler.transform(
                X_df[self._scaled_columns]
            )
        if return_dataframe:
            return X_scaled
        return X_scaled.to_numpy()

    @property
    def scaler(self) -> StandardScaler:
        """Acesso ao scaler fitado para persistência."""
        return self._scaler

    @property
    def binary_columns(self) -> list[str]:
        return self._binary_columns.copy()

    @property
    def scaled_columns(self) -> list[str]:
        return self._scaled_columns.copy()

    def _to_dataframe(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        use_known_columns: bool,
    ) -> tuple[pd.DataFrame, bool]:
        if isinstance(X, pd.DataFrame):
            return X.copy(), True

        columns = self._columns if use_known_columns else None
        return pd.DataFrame(X, columns=columns), False

    def _detect_binary_columns(self, X: pd.DataFrame) -> list[str]:
        binary_columns: list[str] = []
        known_binary = set(BINARY_PASSTHROUGH_FEATURES)

        for col in X.columns:
            non_null = pd.Series(X[col]).dropna()
            unique_values = set(non_null.unique().tolist())
            if col in known_binary or unique_values.issubset({0, 1}):
                binary_columns.append(col)

        return binary_columns
