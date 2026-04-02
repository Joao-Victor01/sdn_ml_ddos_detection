"""
Limpeza e preparação dos dados de treino.

SRP: este módulo lida exclusivamente com a remoção de ruído dos dados
(duplicatas, infinitos, valores ausentes).

Regra de ouro: todas as operações de fit são realizadas APENAS no
conjunto de treino. O teste recebe apenas .transform().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ml.config import IMPUTER_STRATEGY


class DataCleaner:
    """
    Remove duplicatas, trata infinitos e imputa valores ausentes.

    Uso correto (evita data leakage):
        cleaner = DataCleaner()
        X_train, y_train = cleaner.fit_transform(X_train, y_train)
        X_test            = cleaner.transform(X_test)
    """

    def __init__(self, strategy: str = IMPUTER_STRATEGY) -> None:
        self._imputer: SimpleImputer = SimpleImputer(strategy=strategy)
        self._columns: list[str] = []
        self._fitted: bool = False

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Aplica limpeza completa ao conjunto de TREINO.

        Etapas (nesta ordem):
          1. Remove duplicatas (inclui y para consistência)
          2. Substitui Inf/-Inf por NaN
          3. Fit + transform do imputador de medianas

        Returns
        -------
        X_clean : pd.DataFrame
        y_clean : pd.Series  (alinhado com X após remoção de duplicatas)
        """
        self._columns = X.columns.tolist()

        # 1. Duplicatas — reconstruir temporariamente com target
        df_tmp = X.copy()
        df_tmp["__target__"] = y.values

        before = len(df_tmp)
        df_tmp = df_tmp.drop_duplicates(keep="first")
        after  = len(df_tmp)
        print(f"[DataCleaner] Duplicatas removidas: {before - after:,} "
              f"({before:,} → {after:,})")

        X_clean = df_tmp.drop(columns=["__target__"])
        y_clean = pd.Series(df_tmp["__target__"].values, name=y.name)

        # 2. Infinitos → NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        inf_total = (before - after)  # já contabilizado
        nan_count = X_clean.isnull().sum().sum()
        print(f"[DataCleaner] NaN após substituição de Inf: {nan_count:,}")

        # 3. Imputação (fit SOMENTE no treino)
        self._imputer.fit(X_clean)
        X_imputed = pd.DataFrame(
            self._imputer.transform(X_clean),
            columns=self._columns,
        )
        self._fitted = True

        print(f"[DataCleaner] NaN após imputação (treino): "
              f"{X_imputed.isnull().sum().sum()}")
        print(f"[DataCleaner] Shape final do treino: {X_imputed.shape}")

        return X_imputed, y_clean

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica apenas a transformação ao conjunto de TESTE.

        Não faz fit — usa os parâmetros aprendidos em fit_transform().
        """
        if not self._fitted:
            raise RuntimeError(
                "DataCleaner: chame fit_transform() no treino antes de transform()."
            )

        X_clean = X.replace([np.inf, -np.inf], np.nan)

        X_imputed = pd.DataFrame(
            self._imputer.transform(X_clean),
            columns=self._columns,
        )
        print(f"[DataCleaner] NaN após imputação (teste): "
              f"{X_imputed.isnull().sum().sum()}")
        return X_imputed

    @property
    def imputer(self) -> SimpleImputer:
        """Acesso ao imputador fitado para persistência."""
        return self._imputer
