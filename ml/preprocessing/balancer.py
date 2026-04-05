"""
Balanceamento de classes com SMOTE para classificacao multiclasse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from ml.config import RANDOM_STATE, SMOTE_K_NEIGHBORS, TARGET_DECODING


class ClassBalancer:
    """Aplica SMOTE apenas no conjunto de treino."""

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        k_neighbors: int = SMOTE_K_NEIGHBORS,
    ) -> None:
        self._smote = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    def fit_resample(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_arr = np.array(y)
        self._print_distribution(y_arr, prefix="ANTES")
        X_res, y_res = self._smote.fit_resample(X, y_arr)
        self._print_distribution(y_res, prefix="DEPOIS")
        print(f"[ClassBalancer] Shape treino balanceado: {X_res.shape}")
        return X_res, y_res

    def _print_distribution(self, y: np.ndarray, prefix: str) -> None:
        print(f"[ClassBalancer] Distribuicao {prefix} do SMOTE:")
        total = len(y)
        for cls in sorted(np.unique(y)):
            count = int((y == cls).sum())
            pct = count / total * 100
            label = TARGET_DECODING.get(int(cls), str(cls))
            print(f"  {int(cls)} ({label:<9}): {count:>7,}  ({pct:.2f}%)")
