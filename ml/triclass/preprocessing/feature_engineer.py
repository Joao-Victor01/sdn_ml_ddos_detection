"""
Engenharia de features comportamentais para o pipeline triclasse.

As seis features derivadas substituem o TTL (indisponível no InSDN) como
diferenciadores de comportamento entre as três classes.

SRP: apenas computa features derivadas — sem fit, sem estado, sem leakage.
     É seguro aplicar em treino e teste de forma idêntica pois não usa
     nenhuma estatística dos dados (operações puramente matemáticas).

Referência: plano_triclasse_insdn_v4.md, Seção 5.3
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.triclass.config import BEHAVIORAL_FEATURES

# Nomes das colunas-fonte (após RENAME_MAP)
_FWD   = "Total Fwd Packets"
_BWD   = "Total Backward Packets"
_FWDB  = "Total Length of Fwd Packets"
_BWDB  = "Total Length of Bwd Packets"
_DUR   = "Flow Duration"
_STD   = "Packet Length Std"
_ACT   = "Fwd Act Data Pkts"

_EPS   = 1e-9  # evitar divisão por zero


def computar_features_comportamentais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa as seis features comportamentais e as adiciona ao DataFrame.

    Feature 1 — asymmetry_pkts:
        fwd / (fwd + bwd + ε)
        ~1.0 = tráfego unidirecional (ataque).
        ~0.5 = tráfego bidirecional (legítimo).

    Feature 2 — asymmetry_bytes:
        fwd_bytes / (fwd_bytes + bwd_bytes + ε)
        Amplificação DNS/NTP tem assimetria inversa (~0) por resposta > req.

    Feature 3 — pkt_rate:
        (fwd + bwd) / (duration + ε)
        Alta taxa = ataque volumétrico.

    Feature 4 — pkt_uniformity:
        1 / (Packet_Length_Std + 1)
        Alta (~1) = gerado por script (burst).
        Baixa (~0) = tráfego humano variado.

    Feature 5 — log_duration:
        log1p(Flow_Duration)
        Normaliza distribuição de cauda longa (Aula 5 — log1p em assimétricos).
        BOTNET ~31.000 µs >> DDoS ~1-19 µs → diferenciador principal.

    Feature 6 — fwd_active_ratio:
        Fwd_Act_Data_Pkts / (Total_Fwd_Packets + ε)
        BOTNET:  ~0 (beacon sem dado real).
        DDoS:    ~0 (sem handshake completo).
        Normal:  >0.5 (dados reais trafegando).
        Nota: separa Normal de ataques, mas não BOTNET de DDoS —
        a separação BOTNET/DDoS depende de log_duration e pkt_uniformity.

    Parameters
    ----------
    df : pd.DataFrame — features numéricas com colunas renomeadas

    Returns
    -------
    pd.DataFrame com as seis novas colunas adicionadas (cópia)
    """
    df = df.copy()

    df["asymmetry_pkts"]  = df[_FWD]  / (df[_FWD]  + df[_BWD]  + _EPS)
    df["asymmetry_bytes"] = df[_FWDB] / (df[_FWDB] + df[_BWDB] + _EPS)
    df["pkt_rate"]        = (df[_FWD] + df[_BWD]) / (df[_DUR]  + _EPS)
    df["pkt_uniformity"]  = 1.0 / (df[_STD] + 1.0)
    df["log_duration"]    = np.log1p(df[_DUR])

    if _ACT in df.columns:
        df["fwd_active_ratio"] = df[_ACT] / (df[_FWD] + _EPS)
    else:
        # Coluna ausente: preencher com 0 e avisar
        import warnings
        warnings.warn(
            f"[BehavioralFeatureEngineer] Coluna '{_ACT}' ausente. "
            f"fwd_active_ratio preenchida com 0.",
            UserWarning,
            stacklevel=2,
        )
        df["fwd_active_ratio"] = 0.0

    return df


class BehavioralFeatureEngineer:
    """
    Wrapper com interface fit/transform para integração no pipeline.

    Como não há parâmetros aprendidos, fit() é um no-op — existe apenas
    para manter a interface consistente com os demais transformadores.

    Uso:
        eng = BehavioralFeatureEngineer()
        X_train = eng.fit_transform(X_train)   # computa features
        X_test  = eng.transform(X_test)         # mesma transformação
    """

    def __init__(self) -> None:
        self._source_cols_present: list[str] = []

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Registra colunas presentes e computa features."""
        self._source_cols_present = [
            c for c in [_FWD, _BWD, _FWDB, _BWDB, _DUR, _STD, _ACT]
            if c in X.columns
        ]
        result = computar_features_comportamentais(X)
        computed = [f for f in BEHAVIORAL_FEATURES if f in result.columns]
        print(f"[BehavioralFeatureEngineer] Features computadas: {computed}")
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica a mesma transformação (sem re-fit)."""
        return computar_features_comportamentais(X)

    @property
    def feature_names(self) -> list[str]:
        """Nomes das features comportamentais geradas."""
        return list(BEHAVIORAL_FEATURES)
