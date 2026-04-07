"""
Inferencia para o pipeline multiclasse treinado no InSDN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import TARGET_DECODING
from ml.persistence.model_io import ModelIO, PipelineArtifacts


class DDoSPredictor:
    """
    realiza classificacao
    multiclasse: Normal / Flooding / Intrusao.
    """

    def __init__(self, models_dir: str | None = None) -> None:
        self._io = ModelIO(*([models_dir] if models_dir else []))
        self._artifacts: PipelineArtifacts | None = None

    def load(self) -> "DDoSPredictor":
        # Carrega tudo de uma vez para a inferência não precisar conhecer detalhes do treinamento.
        self._artifacts = self._io.load()
        print("[DDoSPredictor] Pipeline carregado com sucesso.")
        print(
            f"  Features esperadas ({len(self._artifacts.selected_features)}): "
            f"{self._artifacts.selected_features}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Primeiro passamos pelo mesmo pré-processamento do treino, depois o modelo prevê.
        X_processed = self._preprocess(X)
        return self._artifacts.model.predict(X_processed)

    def predict_labels(self, X: pd.DataFrame) -> list[str]:
        return [TARGET_DECODING[int(cls)] for cls in self.predict(X)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Essa versão devolve as probabilidades por classe, útil para confiança e inspeção.
        X_processed = self._preprocess(X)
        return self._artifacts.model.predict_proba(X_processed)

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(X)
        pred = self.predict(X)
        labels = [TARGET_DECODING[int(cls)] for cls in pred]
        # Confiança = probabilidade máxima entre as classes — quanto mais perto de 1, mais certo
        confidence = proba.max(axis=1)

        return pd.DataFrame(
            {
                "prediction": pred,     # índice numérico da classe (0, 1 ou 2)
                "label": labels,        # nome legível: Normal, Flooding ou Intrusao
                "confidence": confidence,
            }
        )

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        if self._artifacts is None:
            raise RuntimeError("DDoSPredictor: chame load() antes de predict().")

        artifacts = self._artifacts

        # Replica a mesma sequência de pré-processamento do treino:
        # infinitos → NaN → imputação → filtragem de variância → seleção → escalonamento
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_imputed = pd.DataFrame(
            artifacts.imputer.transform(X_clean),  # usa as medianas aprendidas no treino
            columns=X_clean.columns,
        )

        # Remove features de variância zero (as mesmas que foram removidas no treino)
        surviving_cols = X_imputed.columns[artifacts.variance_filter.get_support()].tolist()
        X_var = pd.DataFrame(
            artifacts.variance_filter.transform(X_imputed),
            columns=surviving_cols,
        )

        # Garante que todas as features que o modelo espera estão presentes nos dados de entrada
        missing_features = [f for f in artifacts.selected_features if f not in X_var.columns]
        if missing_features:
            raise ValueError(
                "DDoSPredictor: features ausentes nos dados de entrada: "
                f"{missing_features}\n"
                f"Features esperadas: {artifacts.selected_features}"
            )

        # Seleciona e ordena as features na mesma ordem do treino (importante para o MLP)
        X_sel = X_var[artifacts.selected_features]
        return artifacts.scaler.transform(X_sel)  # z-score com parâmetros do treino
