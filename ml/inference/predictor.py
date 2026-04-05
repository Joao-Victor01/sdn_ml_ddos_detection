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
    Mantem o nome por compatibilidade, mas agora realiza classificacao
    multiclasse: Normal / Flooding / Intrusao.
    """

    def __init__(self, models_dir: str | None = None) -> None:
        self._io = ModelIO(*([models_dir] if models_dir else []))
        self._artifacts: PipelineArtifacts | None = None

    def load(self) -> "DDoSPredictor":
        self._artifacts = self._io.load()
        print("[DDoSPredictor] Pipeline carregado com sucesso.")
        print(
            f"  Features esperadas ({len(self._artifacts.selected_features)}): "
            f"{self._artifacts.selected_features}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self._preprocess(X)
        return self._artifacts.model.predict(X_processed)

    def predict_labels(self, X: pd.DataFrame) -> list[str]:
        return [TARGET_DECODING[int(cls)] for cls in self.predict(X)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self._preprocess(X)
        return self._artifacts.model.predict_proba(X_processed)

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(X)
        pred = self.predict(X)
        labels = [TARGET_DECODING[int(cls)] for cls in pred]
        confidence = proba.max(axis=1)

        return pd.DataFrame(
            {
                "prediction": pred,
                "label": labels,
                "confidence": confidence,
            }
        )

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        if self._artifacts is None:
            raise RuntimeError("DDoSPredictor: chame load() antes de predict().")

        artifacts = self._artifacts
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_imputed = pd.DataFrame(
            artifacts.imputer.transform(X_clean),
            columns=X_clean.columns,
        )

        surviving_cols = X_imputed.columns[artifacts.variance_filter.get_support()].tolist()
        X_var = pd.DataFrame(
            artifacts.variance_filter.transform(X_imputed),
            columns=surviving_cols,
        )

        missing_features = [f for f in artifacts.selected_features if f not in X_var.columns]
        if missing_features:
            raise ValueError(
                "DDoSPredictor: features ausentes nos dados de entrada: "
                f"{missing_features}\n"
                f"Features esperadas: {artifacts.selected_features}"
            )

        X_sel = X_var[artifacts.selected_features]
        return artifacts.scaler.transform(X_sel)
