"""
Balanceamento de classes com SMOTE para classificacao multiclasse.
"""

from __future__ import annotations

from collections import Counter

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
        self._random_state = random_state
        self._k_neighbors = k_neighbors

    def fit_resample(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series, #SMOTE do imbalanced-learn trabalha internamente com arrays numpy
    ) -> tuple[np.ndarray, np.ndarray]:
        y_arr = np.array(y)
        self._print_distribution(y_arr, prefix="ANTES")

        # Verifica se há amostras suficientes para o SMOTE funcionar
        # (precisa de pelo menos 2 amostras por classe para criar vizinhos)
        class_counts = Counter(y_arr.tolist())
        min_class_count = min(class_counts.values())
        if min_class_count < 2:
            print(
                "[ClassBalancer] SMOTE ignorado: alguma classe tem menos de 2 amostras "
                "na subamostra atual."
            )
            print(f"[ClassBalancer] Shape treino balanceado: {np.asarray(X).shape}")
            return X, y_arr # retorna os dados sem balancear 

        # k_neighbors não pode ser maior que (min_samples - 1) — ajusta automaticamente
        effective_k = min(self._k_neighbors, min_class_count - 1)
        if effective_k != self._k_neighbors:
            print(
                f"[ClassBalancer] Ajustando k_neighbors do SMOTE: "
                f"{self._k_neighbors} -> {effective_k}"
            )

        # SMOTE -> cria amostras sintéticas da classe minoritária interpolando 
        smote = SMOTE(
            random_state=self._random_state,
            k_neighbors=effective_k,
        )
        X_res, y_res = smote.fit_resample(X, y_arr)
        self._print_distribution(y_res, prefix="DEPOIS")
        print(f"[ClassBalancer] Shape treino balanceado: {X_res.shape}")
        return X_res, y_res # arrays com o conjunto de treino balanceado

    def _print_distribution(self, y: np.ndarray, prefix: str) -> None:
        print(f"[ClassBalancer] Distribuicao {prefix} do SMOTE:")
        total = len(y)
        for cls in sorted(np.unique(y)):
            count = int((y == cls).sum())
            pct = count / total * 100
            label = TARGET_DECODING.get(int(cls), str(cls))
            print(f"  {int(cls)} ({label:<9}): {count:>7,}  ({pct:.2f}%)")
