"""
Carregamento dos três arquivos InSDN para o pipeline triclasse.

SRP: carrega, concatena e valida os CSVs — sem nenhuma transformação.
     Toda transformação (limpeza, features, split) ocorre downstream.

Avaliação dos arquivos:
  Normal_data.csv    — 68.424 linhas, label='Normal' (Classe 0 completa)
  OVS.csv            — 138.722 linhas, labels: DDoS, DoS, Probe, BFA,
                       Web-Attack, BOTNET
  metasploitable-2.csv — 136.743 linhas, labels: DDoS, Probe, DoS, BFA, U2R

Apenas Normal, DDoS, DoS e BOTNET são usados — o resto é descartado
na etapa de labeling (TriclassLabeler).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.triclass.config import (
    NORMAL_CSV,
    OVS_CSV,
    META_CSV,
    RENAME_MAP,
)

# Colunas necessárias para as features do plano (após renomeação)
_REQUIRED_COLS = {
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Duration",
    "Packet Length Std",
    "Fwd Act Data Pkts",
    "Label",
}


class InSDNTriclassLoader:
    """
    Carrega os três arquivos do InSDN e retorna DataFrame concatenado.

    Uso:
        loader = InSDNTriclassLoader()
        df = loader.load()      # DataFrame bruto com Label original
        loader.describe(df)     # EDA textual (sem modificar df)
    """

    def __init__(
        self,
        normal_path=NORMAL_CSV,
        ovs_path=OVS_CSV,
        meta_path=META_CSV,
    ) -> None:
        self._paths = {
            "Normal_data": normal_path,
            "OVS":         ovs_path,
            "metasploitable": meta_path,
        }

    # ── API pública ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Carrega, concatena e renomeia colunas dos três arquivos.

        Returns
        -------
        pd.DataFrame com Label original preservado e colunas padronizadas.

        Raises
        ------
        FileNotFoundError  se algum CSV não existir.
        ValueError         se colunas obrigatórias estiverem ausentes.
        """
        frames = []
        for name, path in self._paths.items():
            path = path if hasattr(path, "exists") else __import__("pathlib").Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"[InSDNTriclassLoader] Arquivo não encontrado: {path}\n"
                    f"Verifique ml/triclass/config.py → INSDN_DIR."
                )
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            df["_source"] = name
            frames.append(df)
            print(f"[InSDNTriclassLoader] {name}: {df.shape[0]:,} linhas")

        data = pd.concat(frames, ignore_index=True)

        # Renomear colunas para padrão do plano (seção 2.2)
        data = data.rename(columns={
            k: v for k, v in RENAME_MAP.items() if k in data.columns
        })

        self._validate_columns(data)
        print(f"[InSDNTriclassLoader] Total concatenado: {data.shape[0]:,} linhas, "
              f"{data.shape[1]} colunas")
        return data

    def describe(self, data: pd.DataFrame) -> None:
        """
        EDA textual sobre o DataFrame bruto.
        Não modifica os dados — chamada segura antes do split.
        """
        print("\n" + "=" * 65)
        print("  EDA — InSDN Triclasse (dados brutos, pré-split)")
        print("=" * 65)
        print(f"\nShape       : {data.shape}")
        print(f"Memória     : {data.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        print("\n── Distribuição de Labels originais ──")
        vc = data["Label"].value_counts()
        for lbl, cnt in vc.items():
            pct = 100 * cnt / len(data)
            print(f"  {lbl:<20}: {cnt:>8,}  ({pct:.1f}%)")

        print("\n── Missing values ──")
        missing = data.isnull().sum()
        if missing.any():
            print(missing[missing > 0].to_string())
        else:
            print("  Nenhum missing value.")

        print("\n── Infinitos ──")
        num = data.select_dtypes(include=np.number)
        inf_count = np.isinf(num).sum()
        inf_cols = inf_count[inf_count > 0]
        if not inf_cols.empty:
            print(inf_cols.to_string())
        else:
            print("  Nenhum valor infinito.")

        print("\n── Duplicatas ──")
        print(f"  Linhas duplicadas: {data.duplicated().sum():,}")

        print("\n── Stats das features-chave ──")
        key_cols = [c for c in [
            "Flow Duration", "Packet Length Std",
            "Total Fwd Packets", "Total Backward Packets",
            "Fwd Act Data Pkts",
        ] if c in data.columns]
        if key_cols:
            print(data[key_cols].describe().round(4).to_string())
        print("=" * 65 + "\n")

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(data.columns)
        if missing:
            raise ValueError(
                f"[InSDNTriclassLoader] Colunas obrigatórias ausentes após renomeação:\n"
                f"  {sorted(missing)}\n"
                f"  Ajuste RENAME_MAP em ml/triclass/config.py."
            )
