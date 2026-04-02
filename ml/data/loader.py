"""
Camada de carregamento de dados.

Define o protocolo DataLoader (ISP/DIP — depender de abstrações) e a
implementação concreta InSDNLoader para o dataset insdn8_ddos_binary_0n1d.csv.

O dataset insdn8 é um subconjunto pré-selecionado do InSDN original
(Elsayed et al., IEEE Access 2020) com 8 features de fluxo de rede e
classificação binária: 0 = Benigno, 1 = Ataque DDoS.

Avaliação do dataset:
  - Arquivo : insdn8_ddos_binary_0n1d.csv
  - Instâncias: ~190.366 (compatível com o plano — InSDN original: 361.317)
  - Features   : 8 (já pré-selecionadas — Protocol, Flow Duration, Flow IAT Max,
                    Bwd Pkts/s, Pkt Len Std, Pkt Len Var, Bwd IAT Tot, Flow Pkts/s)
  - Label      : binária (0 = Normal, 1 = DDoS) — já binarizada
  - Desbalanceamento: majoritariamente classe 1 (ataque) — SMOTE necessário
  - Nota: a seleção de features via SHAP é realizada mesmo com poucas features
    para confirmar importância relativa e descartar features de variância zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from ml.config import DATASET_PATH, TARGET_COL


# ── Protocolo (ISP + DIP) ──────────────────────────────────────────────────────

class DataLoader(Protocol):
    """
    Interface mínima para carregadores de dataset.

    Qualquer implementação que retorne (X, y) como DataFrames é válida.
    Seguindo DIP, os módulos de pipeline dependem desta abstração,
    não de InSDNLoader diretamente.
    """

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Carrega o dataset e retorna (features, alvo).

        Returns
        -------
        X : pd.DataFrame
            Features brutas (sem transformações).
        y : pd.Series
            Coluna alvo com valores inteiros 0/1.
        """
        ...

    def describe(self) -> None:
        """Exibe resumo do dataset no stdout para fins de EDA inicial."""
        ...


# ── Implementação concreta ─────────────────────────────────────────────────────

class InSDNLoader:
    """
    Carrega e valida o dataset InSDN8 (insdn8_ddos_binary_0n1d.csv).

    Responsabilidade única: ler o CSV e retornar (X, y) sem nenhuma
    transformação — o split e o pré-processamento ficam em outras classes.
    """

    def __init__(self, path: Path | str = DATASET_PATH) -> None:
        self._path = Path(path)

    # ── API pública ────────────────────────────────────────────────────────────

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Carrega o CSV e retorna (X, y).

        EDA mínima embutida: verifica se o arquivo existe, se a coluna alvo
        está presente e se os valores são binários (0/1).

        Returns
        -------
        X : pd.DataFrame  — features brutas
        y : pd.Series     — alvo binário (0 = Benigno, 1 = Ataque)

        Raises
        ------
        FileNotFoundError  se o CSV não existir no caminho configurado.
        ValueError         se a coluna alvo estiver ausente ou não for binária.
        """
        if not self._path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {self._path}\n"
                f"Verifique ml/config.py → DATASET_PATH."
            )

        print(f"[InSDNLoader] Carregando: {self._path}")
        data = pd.read_csv(self._path, low_memory=False)
        print(f"  Shape bruto: {data.shape}")

        self._validate(data)

        X = data.drop(columns=[TARGET_COL])
        y = data[TARGET_COL].astype(int)

        return X, y

    def describe(self) -> None:
        """EDA textual: shape, tipos, distribuição do alvo, ausentes, infinitos."""
        import numpy as np

        X, y = self.load()
        data = X.copy()
        data[TARGET_COL] = y

        print("\n" + "=" * 60)
        print("  EDA — InSDN8 Dataset")
        print("=" * 60)

        print(f"\nShape       : {data.shape}")
        print(f"Memória     : {data.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        print("\n── Tipos de dados ──")
        print(data.dtypes.to_string())

        print("\n── Distribuição do alvo ──")
        counts = y.value_counts()
        pcts   = y.value_counts(normalize=True).mul(100).round(2)
        for cls in counts.index:
            label = "Benigno" if cls == 0 else "Ataque DDoS"
            print(f"  {cls} ({label}): {counts[cls]:>7,}  ({pcts[cls]:.2f}%)")

        bal = self._shannon_balance(y)
        print(f"\n  Entropia de Shannon (balanceamento): {bal:.4f}")
        print(f"  → 0 = totalmente desbalanceado | 1 = perfeitamente balanceado")

        print("\n── Valores ausentes (antes do split) ──")
        missing = data.isnull().sum()
        if missing.any():
            print(missing[missing > 0].to_string())
        else:
            print("  Nenhum valor ausente.")

        print("\n── Valores infinitos (antes do split) ──")
        inf_count = np.isinf(data.select_dtypes(include=np.number)).sum().sum()
        print(f"  Total de Inf: {inf_count}")

        print("\n── Duplicatas (antes do split) ──")
        print(f"  Linhas duplicadas: {data.duplicated().sum():,}")

        print("\n── Estatísticas descritivas (features) ──")
        print(data.drop(columns=[TARGET_COL]).describe().to_string())
        print("=" * 60 + "\n")

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _validate(self, data: pd.DataFrame) -> None:
        if TARGET_COL not in data.columns:
            raise ValueError(
                f"Coluna alvo '{TARGET_COL}' não encontrada no dataset. "
                f"Colunas disponíveis: {list(data.columns)}"
            )
        unique_vals = set(data[TARGET_COL].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"A coluna '{TARGET_COL}' deve conter apenas valores binários (0/1). "
                f"Encontrado: {unique_vals}"
            )

    @staticmethod
    def _shannon_balance(y: pd.Series) -> float:
        """Entropia de Shannon normalizada: 0 (desbalanceado) → 1 (balanceado)."""
        import numpy as np
        n_classes = y.nunique()
        if n_classes < 2:
            return 0.0
        n = len(y)
        H = sum(
            -(count / n) * np.log(count / n)
            for count in y.value_counts()
        )
        return H / np.log(n_classes)
