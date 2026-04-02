"""
Inferência em produção: DDoSPredictor.

SRP: responsável exclusivamente pela aplicação do pipeline de transformações
e predição em novos dados (tráfego de rede capturado em tempo real).

Garante que as mesmas transformações usadas no treino sejam aplicadas
na mesma ordem durante a inferência — usando os MESMOS objetos fitados.

Pipeline de inferência (ordem obrigatória):
  1. replace Inf → NaN
  2. imputer.transform()          (mediana do treino)
  3. variance_filter.transform()  (remove colunas de variância zero)
  4. selecionar features          (top-N do SHAP)
  5. scaler.transform()           (z-score do treino)
  6. model.predict() / predict_proba()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.persistence.model_io import ModelIO, PipelineArtifacts


class DDoSPredictor:
    """
    Aplica o pipeline completo de inferência a novos dados de rede.

    Uso:
        predictor = DDoSPredictor()
        predictor.load()
        predictions = predictor.predict(df_novos_fluxos)
        probabilities = predictor.predict_proba(df_novos_fluxos)

    O DataFrame de entrada deve conter as mesmas colunas que o dataset
    de treinamento (sem a coluna Label).
    """

    def __init__(self, models_dir: str | None = None) -> None:
        self._io        = ModelIO(*([models_dir] if models_dir else []))
        self._artifacts: PipelineArtifacts | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    def load(self) -> "DDoSPredictor":
        """
        Carrega todos os artefatos do pipeline salvo.

        Returns
        -------
        self (fluent interface).
        """
        self._artifacts = self._io.load()
        print("[DDoSPredictor] Pipeline carregado com sucesso.")
        print(f"  Features esperadas ({len(self._artifacts.selected_features)}): "
              f"{self._artifacts.selected_features}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz a classe de cada fluxo de rede.

        Parameters
        ----------
        X : pd.DataFrame — features brutas (sem o alvo)
                           deve conter as colunas originais do dataset

        Returns
        -------
        np.ndarray de inteiros: 0 = Benigno, 1 = Ataque DDoS
        """
        X_processed = self._preprocess(X)
        return self._artifacts.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna as probabilidades de cada classe.

        Returns
        -------
        np.ndarray de shape (n_samples, 2):
          coluna 0 = P(Benigno), coluna 1 = P(Ataque DDoS)
        """
        X_processed = self._preprocess(X)
        return self._artifacts.model.predict_proba(X_processed)

    def predict_with_confidence(
        self, X: pd.DataFrame, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Prediz classe + probabilidade de ataque + flag de alta confiança.

        Parameters
        ----------
        threshold : float — limiar de probabilidade para classificar como ataque
                            (padrão 0.5; diminuir aumenta recall, aumenta falsos positivos)

        Returns
        -------
        pd.DataFrame com colunas:
          prediction  : int   — 0 ou 1
          prob_attack : float — probabilidade de ser ataque DDoS
          high_conf   : bool  — True se prob_attack > 0.9
        """
        proba       = self.predict_proba(X)
        prob_attack = proba[:, 1]
        prediction  = (prob_attack >= threshold).astype(int)

        return pd.DataFrame({
            "prediction":  prediction,
            "prob_attack": prob_attack,
            "high_conf":   prob_attack > 0.9,
        })

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """
        Aplica o pipeline de transformações na mesma ordem do treino.

        Raises
        ------
        RuntimeError se load() não foi chamado.
        ValueError   se colunas necessárias estiverem ausentes.
        """
        if self._artifacts is None:
            raise RuntimeError(
                "DDoSPredictor: chame load() antes de predict()."
            )

        a = self._artifacts

        # 1. Infinitos → NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)

        # 2. Imputação (transform apenas — sem re-fit)
        X_imputed = pd.DataFrame(
            a.imputer.transform(X_clean),
            columns=X_clean.columns,
        )

        # 3. VarianceThreshold (transform apenas)
        surviving_cols = X_imputed.columns[a.variance_filter.get_support()].tolist()
        X_var = pd.DataFrame(
            a.variance_filter.transform(X_imputed),
            columns=surviving_cols,
        )

        # 4. Selecionar features (mesmas do treino)
        missing_features = [f for f in a.selected_features if f not in X_var.columns]
        if missing_features:
            raise ValueError(
                f"DDoSPredictor: features ausentes nos dados de entrada: {missing_features}\n"
                f"Features esperadas: {a.selected_features}"
            )
        X_sel = X_var[a.selected_features]

        # 5. Escalonamento (transform apenas — sem re-fit)
        X_scaled = a.scaler.transform(X_sel)

        return X_scaled
