"""
Limpeza e preparacao dos dados apos o split.

Responsabilidades:
  - remover duplicatas
  - substituir infinitos por NaN
  - tratar valores negativos impossiveis por dominio
  - imputar ausentes com estatisticas aprendidas apenas no treino
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ml.config import IMPUTER_STRATEGY, NON_NEGATIVE_FEATURES


class DataCleaner:
    """
    Limpa conjuntos de treino e teste sem vazar informacao do test_set.
    """

    def __init__(
        self,
        strategy: str = IMPUTER_STRATEGY,
        non_negative_columns: list[str] | None = None,
    ) -> None:
        self._imputer: SimpleImputer = SimpleImputer(strategy=strategy)
        self._columns: list[str] = []
        self._non_negative_columns = non_negative_columns or NON_NEGATIVE_FEATURES
        self._fitted = False

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Limpa o conjunto de treino e ajusta o imputador.
        Chamado APENAS no treino — o imputador aprende as estatísticas aqui.
        """
        #guarda os nomes da colunas
        self._columns = [col for col in X.columns if col != "__row_hash__"]

        #remover duplicatas
        X_clean, y_clean, dupes_removed = self._drop_duplicates(X, y)

        # Substitui infinitos e negativos inválidos por NaN
        X_clean = self._sanitize_numeric_noise(X_clean)

        # O fit aprende a mediana de cada coluna com base nos dados de TREINO
        # Para o teste, usaremos essa mesma mediana 
        self._imputer.fit(X_clean) #calcula a mediana
        X_imputed = pd.DataFrame(
            self._imputer.transform(X_clean),
            columns=self._columns,
            index=X_clean.index,
        ).reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)

        self._fitted = True
        print(f"[DataCleaner] Duplicatas removidas no treino: {dupes_removed:,}")
        print(f"[DataCleaner] Shape final do treino: {X_imputed.shape}")
        print(f"[DataCleaner] NaN restantes no treino: {int(X_imputed.isnull().sum().sum())}")

        return X_imputed, y_clean

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Aplica a limpeza ao teste usando o imputador ajustado no treino.
        """
        if not self._fitted:
            raise RuntimeError(
                "DataCleaner: chame fit_transform() no treino antes de transform()."
            )

        if y is not None:
            X, y, dupes_removed = self._drop_duplicates(X, y)
            print(f"[DataCleaner] Duplicatas removidas no teste: {dupes_removed:,}")

        X_clean = self._sanitize_numeric_noise(X)
        X_imputed = pd.DataFrame(
            self._imputer.transform(X_clean),#medianas aprendidas no treino 
            columns=self._columns,
            index=X_clean.index,
        ).reset_index(drop=True)
        print(f"[DataCleaner] NaN restantes apos imputacao: {int(X_imputed.isnull().sum().sum())}")

        if y is None:
            return X_imputed
        return X_imputed, y.reset_index(drop=True)

    @property
    def imputer(self) -> SimpleImputer:
        return self._imputer

    def _drop_duplicates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, int]:
        df_tmp = X.copy()
        df_tmp["__target__"] = y.values
        before = len(df_tmp)

        # Usa o hash da linha (calculado no loader) para detectar duplicatas de forma eficiente
        # Considera o target junto: a mesma linha com classes diferentes NÃO é duplicata
        if "__row_hash__" in df_tmp.columns:
            df_tmp = df_tmp.drop_duplicates(subset=["__row_hash__", "__target__"], keep="first")
            df_tmp = df_tmp.drop(columns=["__row_hash__"])
        else:
            df_tmp = df_tmp.drop_duplicates(keep="first")
        after = len(df_tmp)
        X_dedup = df_tmp.drop(columns=["__target__"])
        y_dedup = pd.Series(df_tmp["__target__"].values, name=y.name)
        return X_dedup, y_dedup, before - after

    def _sanitize_numeric_noise(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()

        # Infinitos aparecem quando há divisão por zero na extração de features (ex.: Byts/s)
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

        # Features de rede não podem ser negativas (duração, bytes, pacotes etc.)
        # Valores negativos são ruído da ferramenta de extração — tratamos como ausentes
        invalid_negative_total = 0
        for col in self._non_negative_columns:
            if col in X_clean.columns:
                mask = X_clean[col] < 0
                invalid_negative_total += int(mask.sum())
                X_clean.loc[mask, col] = np.nan

        print(f"[DataCleaner] Valores negativos invalidos -> NaN: {invalid_negative_total:,}")
        print(f"[DataCleaner] NaN antes da imputacao: {int(X_clean.isnull().sum().sum())}")
        return X_clean
