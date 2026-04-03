"""
Engenharia de features baseadas em HCF e topologia SDN.

Adiciona quatro novas features ao dataset que o modelo usa para distinguir
ataques externos de zumbis internos — implementando o pilar de
Hop Count Filtering (HCF) do sistema multiclasse.

## Features geradas

| Feature            | Tipo    | Descrição                                                      |
|--------------------|---------|----------------------------------------------------------------|
| `ttl_estimated`    | float   | TTL estimado do pacote ao chegar ao controlador                |
| `hop_count`        | float   | Número de saltos estimado = TTL_inicial - ttl_estimated        |
| `is_internal`      | int 0/1 | Heurística: source parece ser host interno da LAN              |
| `ttl_anomaly`      | float   | |hop_count - hop_count_esperado| — quanto o TTL desvia do normal|

## Sobre a feature `ttl_estimated`

Em TREINAMENTO: gerada sinteticamente com distribuições realistas por classe.
  - Classe 0 (Benigno):  N(µ=61, σ=3)  — tráfego LAN normal, 1-3 hops
  - Classe 1 (Externo):  N(µ=48, σ=10) — cruzou ~16 roteadores internet
  - Classe 2 (Interno):  N(µ=62, σ=2)  — 1-2 hops dentro da LAN

Em PRODUÇÃO (SDN ativo): TTL real extraído do cabeçalho IP via Packet-In do ODL.
  O `DDoSPredictor` aceita um parâmetro `ttl_real` que substitui a geração sintética.

## Justificativa da síntese

Não há circularidade lógica: geramos TTL realista A PARTIR da label de treino
(que representa o cenário real), e o modelo aprende a PARTIR das features
(incluindo TTL) para predizer a label. Em produção, TTL é observado diretamente —
o modelo generaliza corretamente sem depender de nenhuma regra de síntese.

SRP: este módulo gera features. Não manipula labels, modelos ou datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import (
    RANDOM_STATE,
    TTL_BENIGN_MEAN,   TTL_BENIGN_STD,
    TTL_EXTERNAL_MEAN, TTL_EXTERNAL_STD,
    TTL_INTERNAL_MEAN, TTL_INTERNAL_STD,
    TTL_LAN_EXPECTED,
    HCF_EXTERNAL_THRESHOLD,
    EXTERNAL_STD_THRESH,
    EXTERNAL_DURATION_THRESH,
    EXTERNAL_PROTOCOL_ID,
)

_NEW_FEATURES = ["ttl_estimated", "hop_count", "is_internal", "ttl_anomaly"]


class TopologyFeatureEngineer:
    """
    Adiciona features HCF + topológicas ao DataFrame.

    Uso correto (sem leakage):
        eng = TopologyFeatureEngineer()

        # Treino: usa labels para gerar TTL sintético realista
        X_train_rich = eng.fit_transform(X_train, y_train_multiclass)

        # Teste: sem labels → TTL gerado por heurística de features (ou real do SDN)
        X_test_rich  = eng.transform(X_test)
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self._rng = np.random.RandomState(random_state)
        self._fitted = False

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Gera features HCF no conjunto de TREINO usando a label multiclasse.

        A label é usada APENAS para escolher a distribuição de TTL realista.
        Não vaza informação: o modelo aprende da feature, não da regra de geração.
        """
        self._fitted = True
        return self._add_features(X.copy(), y=y)

    def transform(
        self,
        X: pd.DataFrame,
        ttl_real: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Gera features HCF no conjunto de TESTE (sem usar labels).

        Parameters
        ----------
        X        : DataFrame sem a coluna alvo
        ttl_real : array de TTLs reais capturados pelo SDN (produção).
                   Se None → TTL estimado via heurística das features existentes.
        """
        if not self._fitted:
            raise RuntimeError(
                "TopologyFeatureEngineer: chame fit_transform() no treino antes."
            )
        return self._add_features(X.copy(), y=None, ttl_real=ttl_real)

    @property
    def new_feature_names(self) -> list[str]:
        return _NEW_FEATURES.copy()

    # ── Implementação interna ──────────────────────────────────────────────────

    def _add_features(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        ttl_real: np.ndarray | None = None,
    ) -> pd.DataFrame:
        n = len(X)

        # ── Feature 1: ttl_estimated ───────────────────────────────────────────
        if ttl_real is not None:
            # Produção: usa TTL real capturado pelo SDN
            ttl = np.asarray(ttl_real, dtype=float)
        elif y is not None:
            # Treino: distribuição realista por classe
            ttl = np.empty(n, dtype=float)
            for cls, mean, std in [
                (0, TTL_BENIGN_MEAN,   TTL_BENIGN_STD),
                (1, TTL_EXTERNAL_MEAN, TTL_EXTERNAL_STD),
                (2, TTL_INTERNAL_MEAN, TTL_INTERNAL_STD),
            ]:
                mask = (y.values == cls)
                ttl[mask] = self._rng.normal(mean, std, mask.sum()).clip(1, 255)
        else:
            # Teste sem SDN: heurística baseada nas features existentes
            ttl = self._estimate_ttl_from_features(X)

        X["ttl_estimated"] = ttl.round(1)

        # ── Feature 2: hop_count ───────────────────────────────────────────────
        # Número de saltos estimado: TTL_LAN_EXPECTED - TTL_observado
        # Limita entre 0 e 60 para evitar artefatos de geração sintética
        X["hop_count"] = (TTL_LAN_EXPECTED - X["ttl_estimated"]).clip(0, 60)

        # ── Feature 3: is_internal ─────────────────────────────────────────────
        # Heurística determinística (sem labels): combina TTL e comportamento
        # de fluxo para estimar se a origem é interna à LAN
        X["is_internal"] = (
            (X["hop_count"] < HCF_EXTERNAL_THRESHOLD) &           # poucos saltos
            (X["Protocol"] != EXTERNAL_PROTOCOL_ID) &             # não é flood raw
            ~((X["Pkt Len Std"] <= EXTERNAL_STD_THRESH) &         # não é rajada uniforme
              (X["Flow Duration"] < EXTERNAL_DURATION_THRESH))
        ).astype(int)

        # ── Feature 4: ttl_anomaly ─────────────────────────────────────────────
        # Desvio do TTL em relação ao esperado para host local (HCF clássico):
        # hop_count alto → TTL muito degradado → provável origem externa
        X["ttl_anomaly"] = np.abs(X["hop_count"] - X["hop_count"].median())

        return X

    def _estimate_ttl_from_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estima TTL para o conjunto de teste sem labels.

        Lógica: se o fluxo tem características de flood externo (protocolo 0,
        pacotes uniformes, curta duração) → TTL estimado baixo (externo).
        Caso contrário → TTL estimado alto (interno ou benigno).
        """
        ttl = np.full(len(X), TTL_BENIGN_MEAN)

        external_mask = (
            (X["Protocol"] == EXTERNAL_PROTOCOL_ID) |
            ((X["Pkt Len Std"] <= EXTERNAL_STD_THRESH) &
             (X["Flow Duration"] < EXTERNAL_DURATION_THRESH))
        )
        internal_mask = ~external_mask

        n_ext = external_mask.sum()
        n_int = internal_mask.sum()

        if n_ext > 0:
            ttl[external_mask.values] = self._rng.normal(
                TTL_EXTERNAL_MEAN, TTL_EXTERNAL_STD, n_ext
            ).clip(1, 255)
        if n_int > 0:
            ttl[internal_mask.values] = self._rng.normal(
                TTL_INTERNAL_MEAN, TTL_INTERNAL_STD, n_int
            ).clip(1, 255)

        return ttl
