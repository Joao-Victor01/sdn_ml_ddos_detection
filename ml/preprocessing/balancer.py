"""
Balanceamento de classes com SMOTE.

SRP: responsável exclusivamente pelo balanceamento do conjunto de treino.

Regra obrigatória (boas práticas do curso):
  SMOTE gera instâncias SINTÉTICAS — aplique SOMENTE no treino.
  O teste deve representar a distribuição real do mundo, não dados artificiais.

O dataset insdn8 é heavily imbalanced (majoritariamente classe 1 = DDoS),
o que justifica o uso de SMOTE para equilibrar as classes antes do treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from ml.config import RANDOM_STATE


class ClassBalancer:
    """
    Aplica SMOTE para sobreamostrar a classe minoritária no conjunto de treino.

    Uso:
        balancer = ClassBalancer()
        X_bal, y_bal = balancer.fit_resample(X_train_scaled, y_train)
        # X_test NÃO passa pelo balancer
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self._smote = SMOTE(random_state=random_state)

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_resample(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica SMOTE ao treino e retorna arrays balanceados.

        Registra no stdout a distribuição antes/depois para auditoria.

        Returns
        -------
        X_resampled : np.ndarray
        y_resampled : np.ndarray
        """
        y_arr = np.array(y)

        counts_before = {cls: int((y_arr == cls).sum()) for cls in np.unique(y_arr)}
        pcts_before   = {cls: cnt / len(y_arr) * 100 for cls, cnt in counts_before.items()}

        print("[ClassBalancer] Distribuição ANTES do SMOTE:")
        for cls, cnt in counts_before.items():
            label = "Benigno" if cls == 0 else "Ataque DDoS"
            print(f"  {cls} ({label}): {cnt:>7,}  ({pcts_before[cls]:.1f}%)")

        X_res, y_res = self._smote.fit_resample(X, y_arr)

        counts_after = {cls: int((y_res == cls).sum()) for cls in np.unique(y_res)}
        pcts_after   = {cls: cnt / len(y_res) * 100 for cls, cnt in counts_after.items()}

        print("[ClassBalancer] Distribuição DEPOIS do SMOTE:")
        for cls, cnt in counts_after.items():
            label = "Benigno" if cls == 0 else "Ataque DDoS"
            print(f"  {cls} ({label}): {cnt:>7,}  ({pcts_after[cls]:.1f}%)")

        print(f"[ClassBalancer] Shape treino balanceado: {X_res.shape}")

        return X_res, y_res
