"""
Camada de carregamento do InSDN multiclasse.

Concatena os CSVs do diretorio InSDN_DatasetCSV, normaliza os labels
originais e devolve apenas as features selecionadas por criterio de dominio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    CLASS_GROUP_MAPPING,
    DATASET_DIR,
    DATASET_NAME,
    RANDOM_STATE,
    RELEVANT_FEATURES,
    TARGET_COL,
    TARGET_ENCODING,
    TARGET_NAMES,
)


class DataLoader(Protocol):
    """Interface minima para carregadores de dataset."""

    def load(self, sample_size: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
        ...

    def describe(self, sample_size: int | None = None) -> None:
        ...


class InSDNLoader:
    """
    Carrega o InSDN consolidado para o cenario:
      0 = Normal
      1 = Flooding  (DoS + DDoS)
      2 = Intrusao  (Probe + BFA + Web-Attack + BOTNET + U2R)
    """

    def __init__(self, dataset_dir: Path | str = DATASET_DIR) -> None:
        self._dir = Path(dataset_dir)

    def load(self, sample_size: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Carrega, concatena, filtra e mapeia o dataset.

        Parameters
        ----------
        sample_size : int | None
            Se informado, amostra estratificada do dataset consolidado.
        """
        data = self._read_all_csvs()
        self._validate_columns(data)

        data[TARGET_COL] = data[TARGET_COL].astype(str).str.strip()
        mapped_labels = data[TARGET_COL].map(CLASS_GROUP_MAPPING)
        keep_mask = mapped_labels.notna()
        dropped = int((~keep_mask).sum())
        if dropped:
            print(f"[InSDNLoader] Linhas descartadas por label nao mapeado: {dropped:,}")
        data = data.loc[keep_mask].copy()
        data["__target_group__"] = mapped_labels.loc[keep_mask]
        data["__target__"] = data["__target_group__"].map(TARGET_ENCODING)

        if sample_size is not None and 0 < sample_size < len(data):
            data, _ = train_test_split(
                data,
                train_size=sample_size,
                random_state=RANDOM_STATE,
                stratify=data["__target__"],
                shuffle=True,
            )
            data = data.reset_index(drop=True)
            print(f"[InSDNLoader] Amostra estratificada aplicada: {sample_size:,} linhas")

        X = data[RELEVANT_FEATURES].copy()
        X["__row_hash__"] = pd.util.hash_pandas_object(
            data.drop(columns=["__target_group__", "__target__"]),
            index=False,
        ).astype(str)
        y = data["__target__"].astype(int).rename(TARGET_COL)

        print(f"[InSDNLoader] Dataset carregado: {DATASET_NAME}")
        print(f"  Shape bruto consolidado : {data.shape}")
        print(f"  Features utilizadas     : {len(RELEVANT_FEATURES)}")

        return X, y

    def describe(self, sample_size: int | None = None) -> None:
        """EDA textual objetiva do dataset consolidado."""
        X, y = self.load(sample_size=sample_size)
        data = X.drop(columns=["__row_hash__"], errors="ignore").copy()
        data[TARGET_COL] = y.map({idx: label for idx, label in enumerate(TARGET_NAMES)})

        print("\n" + "=" * 72)
        print("  EDA — InSDN multiclasse")
        print("=" * 72)
        print(f"\nShape       : {data.shape}")
        print(f"Memoria     : {data.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        print("\nFeatures usadas:")
        for feature in RELEVANT_FEATURES:
            print(f"  - {feature}")

        print("\nDistribuicao do alvo agrupado:")
        counts = data[TARGET_COL].value_counts()
        pcts = data[TARGET_COL].value_counts(normalize=True).mul(100).round(2)
        for label in TARGET_NAMES:
            print(f"  {label:<10}: {counts.get(label, 0):>7,}  ({pcts.get(label, 0):.2f}%)")

        print("\nTipos de dados:")
        print(data.dtypes.to_string())

        print("\nValores ausentes:")
        missing = data.isnull().sum()
        if missing.any():
            print(missing[missing > 0].to_string())
        else:
            print("  Nenhum valor ausente.")

        print("\nDuplicatas:")
        print(f"  Linhas duplicadas: {data.duplicated().sum():,}")

        print("\nEstatisticas descritivas:")
        print(data.drop(columns=[TARGET_COL]).describe().to_string())
        print("=" * 72 + "\n")

    def _read_all_csvs(self) -> pd.DataFrame:
        if not self._dir.exists():
            raise FileNotFoundError(
                f"Diretorio do dataset nao encontrado: {self._dir}\n"
                "Verifique ml/config.py -> DATASET_DIR."
            )

        files = sorted(self._dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(
                f"Nenhum CSV encontrado em {self._dir}."
            )

        frames: list[pd.DataFrame] = []
        print(f"[InSDNLoader] Lendo {len(files)} arquivos CSV de {self._dir}")
        for path in files:
            df = pd.read_csv(path, low_memory=False)
            print(f"  - {path.name:<22} {df.shape}")
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        required = set(RELEVANT_FEATURES + [TARGET_COL])
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(
                "Colunas obrigatorias ausentes no dataset consolidado: "
                f"{missing}"
            )
