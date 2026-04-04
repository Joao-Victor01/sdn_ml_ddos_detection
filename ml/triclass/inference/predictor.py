"""
Preditor triclasse integrado — Estágio 2 do sistema de detecção.

Arquitetura dois estágios (seção 9.1 do plano):
  Estágio 1 — MLP binário existente:
    → Benigno (0): liberar. Fim.
    → Ataque (1):  passar para Estágio 2.

  Estágio 2 — RF triclasse (novo):
    → Classe 1 (Externo):  POST /manage/ip   → block global
    → Classe 2 (Interno):  POST /mitigation/isolate/{ip} → isolamento cirúrgico

  Fallback determinístico (Opção C — produção):
    → is_known_host == True + predição == 1 → forçar Classe 2
      Razão: IP registrado no SDN é evidência mais confiável que o modelo.

SRP: recebe features brutas → retorna dict com classe, label e ação.

Referência: plano_triclasse_insdn_v4.md, Seção 9.2
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.triclass.config import (
    MODELS_TRICLASS_DIR,
    RENAME_MAP,
    CLASS_NAMES,
)
from ml.triclass.preprocessing.feature_engineer import computar_features_comportamentais

# Ações de resposta do SDN por classe
_ACTIONS: dict[int, str] = {
    0: "none",
    1: "block_global",
    2: "isolate_surgical",
}


class DDoSPredictorV2:
    """
    Preditor em dois estágios para produção.

    Carrega artefatos persistidos pelo pipeline e fornece interface
    simplificada para o controlador SDN.

    Uso:
        predictor = DDoSPredictorV2()
        result = predictor.predict(X_raw_df, is_known_host=False)
        # result = {'class': 1, 'label': 'Externo', 'action': 'block_global',
        #           'confidence': 0.97}
    """

    def __init__(
        self,
        models_binary_dir: str | Path | None = None,
        models_triclass_dir: str | Path = MODELS_TRICLASS_DIR,
    ) -> None:
        tc_dir = Path(models_triclass_dir)

        self._rf          = joblib.load(tc_dir / "rf_triclass.joblib")
        self._imputer     = joblib.load(tc_dir / "imputer.joblib")
        self._vt          = joblib.load(tc_dir / "variance_filter.joblib")
        self._feat_names  = joblib.load(tc_dir / "selected_features.joblib")

        # MLP binário (Estágio 1) — opcional; se ausente, pula o Estágio 1
        self._mlp_binary = None
        if models_binary_dir is not None:
            bin_dir = Path(models_binary_dir)
            mlp_path = bin_dir / "mlp_model.joblib"
            if mlp_path.exists():
                self._mlp_binary = joblib.load(mlp_path)

    # ── API pública ────────────────────────────────────────────────────────────

    def predict(
        self,
        X_raw: pd.DataFrame,
        is_known_host: bool = False,
    ) -> dict:
        """
        Classifica um fluxo de rede nas três classes.

        Parameters
        ----------
        X_raw         : DataFrame com features brutas do CICFlowMeter/InSDN
        is_known_host : True se o IP de origem está registrado no state.ip_to_mac
                        do controlador SDN (evidência de host interno)

        Returns
        -------
        dict com keys: class (int), label (str), action (str), confidence (float),
                       e opcionalmente reason (str) para o fallback determinístico
        """
        # ── Estágio 1: filtro binário (se disponível) ─────────────────────────
        if self._mlp_binary is not None:
            binary_pred = self._mlp_binary.predict(X_raw)[0]
            if binary_pred == 0:
                return {
                    "class": 0,
                    "label": CLASS_NAMES[0],
                    "action": _ACTIONS[0],
                    "confidence": 1.0,
                }

        # ── Pré-processamento (mesmas transformações do treino) ────────────────
        X = X_raw.copy()
        X = X.rename(columns={k: v for k, v in RENAME_MAP.items() if k in X.columns})
        X = computar_features_comportamentais(X)

        X_num = X.select_dtypes(include=np.number)
        X_imp = pd.DataFrame(
            self._imputer.transform(X_num.replace([np.inf, -np.inf], np.nan)),
            columns=X_num.columns,
        )
        # Alinhar com features selecionadas no treino
        available = [f for f in self._feat_names if f in X_imp.columns]
        X_vt = X_imp[available].values

        # ── Estágio 2: predição triclasse ─────────────────────────────────────
        classe     = int(self._rf.predict(X_vt)[0])
        confianca  = float(self._rf.predict_proba(X_vt)[0][classe])

        # ── Fallback determinístico (Opção C) ─────────────────────────────────
        # IP registrado no SDN é evidência mais forte que o modelo ML
        if is_known_host and classe == 1:
            return {
                "class": 2,
                "label": CLASS_NAMES[2],
                "action": _ACTIONS[2],
                "confidence": 0.95,
                "reason": "IP registrado no SDN — reclassificado como Interno",
            }

        return {
            "class": classe,
            "label": CLASS_NAMES.get(classe, f"Classe {classe}"),
            "action": _ACTIONS.get(classe, "unknown"),
            "confidence": confianca,
        }

    def predict_batch(
        self,
        X_raw: pd.DataFrame,
        is_known_hosts: list[bool] | None = None,
    ) -> list[dict]:
        """
        Classifica um batch de fluxos de rede.

        Parameters
        ----------
        X_raw          : DataFrame com múltiplas linhas (um fluxo por linha)
        is_known_hosts : lista de bools, um por linha; None = todos False

        Returns
        -------
        Lista de dicts (um resultado por fluxo)
        """
        if is_known_hosts is None:
            is_known_hosts = [False] * len(X_raw)

        return [
            self.predict(X_raw.iloc[[i]], is_known_host=is_known_hosts[i])
            for i in range(len(X_raw))
        ]
