"""
Validação semântica do BOTNET no test_set.

A validação semântica (seção 8.11 do plano) é a evidência mais robusta
do que métricas agregadas: verifica se o modelo reconhece especificamente
o padrão de beacon/C2 dos 164 fluxos BOTNET reais.

Se recall ≥ 80% → as features log_duration e pkt_uniformity funcionam
como diferenciadores BOTNET vs DDoS, mesmo com volume reduzido.

SRP: único propósito é calcular e reportar o recall semântico do BOTNET.

Referência: plano_triclasse_insdn_v4.md, Seção 8.11
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.triclass.config import BOTNET_MIN_RECALL


@dataclass
class BotnetValidationResult:
    """Resultado da validação semântica do BOTNET."""
    n_botnet_test: int
    n_classified_as_class2: int
    n_classified_as_class1: int
    n_classified_as_class0: int
    recall: float
    passed: bool

    def print_report(self) -> None:
        print("\n" + "=" * 55)
        print("  Validação Semântica — BOTNET (ground truth)")
        print("=" * 55)
        if self.n_botnet_test == 0:
            print("  ⚠ Nenhuma amostra BOTNET no test_set.")
            print("  Aumente test_size ou mude random_state.")
            return
        print(f"  Amostras BOTNET no teste : {self.n_botnet_test}")
        print(f"  → Classe 2 (correto)     : {self.n_classified_as_class2} "
              f"({100*self.recall:.1f}%)")
        print(f"  → Classe 1 (confundido)  : {self.n_classified_as_class1}")
        print(f"  → Classe 0 (confundido)  : {self.n_classified_as_class0}")
        print(f"\n  Recall semântico BOTNET  : {self.recall:.4f}")
        print(f"  Threshold mínimo         : {BOTNET_MIN_RECALL:.2f}")
        if self.passed:
            print("\n  ✓ Modelo reconhece padrão beacon/C2 corretamente.")
            print("    log_duration e pkt_uniformity funcionam como diferenciadores.")
        else:
            print("\n  ⚠ Modelo confunde BOTNET com outra classe.")
            print("    Verificar: log_duration diferencia 31ms de 1-19µs?")
            print("    Possível causa: SMOTE gerou exemplos que se misturaram com DDoS.")
        print("=" * 55)


class BotnetSemanticValidator:
    """
    Valida se o modelo classifica corretamente os fluxos BOTNET reais.

    Requer acesso ao DataFrame original para identificar quais amostras
    do test_set têm label original == 'BOTNET'.

    Uso:
        validator = BotnetSemanticValidator(data_original, test_indices)
        result = validator.validate(model, X_test_vt)
        result.print_report()
    """

    def __init__(
        self,
        data_original: pd.DataFrame,
        test_indices: np.ndarray | pd.Index,
    ) -> None:
        """
        Parameters
        ----------
        data_original : DataFrame completo com coluna 'Label' original preservada
        test_indices  : índices das amostras que foram para o test_set
        """
        self._data   = data_original
        self._test_ix = np.array(test_indices)

    # ── API pública ────────────────────────────────────────────────────────────

    def validate(
        self,
        model,
        X_test_vt: np.ndarray | pd.DataFrame,
        min_recall: float = BOTNET_MIN_RECALL,
    ) -> BotnetValidationResult:
        """
        Avalia o recall do modelo especificamente nos fluxos BOTNET do teste.

        Parameters
        ----------
        model      : estimador sklearn treinado
        X_test_vt  : features do teste após VarianceThreshold
        min_recall : recall mínimo para considerar validação aprovada

        Returns
        -------
        BotnetValidationResult
        """
        # Identifica amostras BOTNET no subconjunto de teste
        label_test = self._data.loc[self._test_ix, "Label"].str.strip()
        botnet_mask = (label_test == "BOTNET").values

        n_botnet = int(botnet_mask.sum())

        if n_botnet == 0:
            return BotnetValidationResult(
                n_botnet_test=0,
                n_classified_as_class2=0,
                n_classified_as_class1=0,
                n_classified_as_class0=0,
                recall=0.0,
                passed=False,
            )

        X_botnet = (
            X_test_vt[botnet_mask]
            if isinstance(X_test_vt, np.ndarray)
            else X_test_vt.iloc[botnet_mask] if hasattr(X_test_vt, "iloc")
            else X_test_vt[botnet_mask]
        )

        y_pred_botnet = model.predict(X_botnet)

        n_cls2 = int((y_pred_botnet == 2).sum())
        n_cls1 = int((y_pred_botnet == 1).sum())
        n_cls0 = int((y_pred_botnet == 0).sum())
        recall = n_cls2 / n_botnet

        result = BotnetValidationResult(
            n_botnet_test=n_botnet,
            n_classified_as_class2=n_cls2,
            n_classified_as_class1=n_cls1,
            n_classified_as_class0=n_cls0,
            recall=recall,
            passed=recall >= min_recall,
        )
        result.print_report()
        return result
