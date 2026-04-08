"""
Persistência dos artefatos do pipeline ML.

O preprocessamento é compartilhado por todos os modelos treinados na mesma
configuração; cada estimador é salvo em um arquivo próprio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from ml.config import MODELS_DIR
from ml.models.registry import get_model_spec
from ml.preprocessing.scaler import FeatureScaler


@dataclass
class PipelineArtifacts:
    """
    Conjunto de artefatos que compõem o pipeline de inferência.
    """

    model_name: str
    model: Any
    imputer: SimpleImputer
    variance_filter: VarianceThreshold
    scaler: FeatureScaler
    selected_features: list[str]


class ModelIO:
    """
    Salva e carrega o pipeline completo de inferência por modelo.
    """

    _COMMON_FILENAMES = {
        "imputer": "imputer.joblib",
        "variance_filter": "variance_filter.joblib",
        "scaler": "scaler.joblib",
        "selected_features": "selected_features.joblib",
    }

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self._dir = Path(models_dir)

    def save(self, artifacts: PipelineArtifacts) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        model_spec = get_model_spec(artifacts.model_name)

        common_pairs = [
            ("imputer", artifacts.imputer),
            ("variance_filter", artifacts.variance_filter),
            ("scaler", artifacts.scaler),
            ("selected_features", artifacts.selected_features),
        ]

        print(f"\n[ModelIO] Salvando artefatos em {self._dir}/")
        for key, obj in common_pairs:
            path = self._dir / self._COMMON_FILENAMES[key]
            with open(path, "wb") as file:
                joblib.dump(obj, file)
            print(f"  ✓ {self._COMMON_FILENAMES[key]}")

        model_path = self._dir / model_spec.persistence_filename
        with open(model_path, "wb") as file:
            joblib.dump(artifacts.model, file)
        print(f"  ✓ {model_spec.persistence_filename}")

        print(f"[ModelIO] Artefatos do modelo '{artifacts.model_name}' salvos com sucesso.")

    def load(self, model_name: str = "mlp") -> PipelineArtifacts:
        print(f"[ModelIO] Carregando artefatos de {self._dir}/")
        model_spec = get_model_spec(model_name)

        loaded: dict[str, Any] = {}
        for key, fname in self._COMMON_FILENAMES.items():
            path = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Artefato não encontrado: {path}\n"
                    "Execute o pipeline de treinamento primeiro."
                )
            with open(path, "rb") as file:
                loaded[key] = joblib.load(file)
            print(f"  ✓ {fname}")

        model_path = self._dir / model_spec.persistence_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                f"Execute o pipeline com --model {model_name} primeiro."
            )
        with open(model_path, "rb") as file:
            loaded["model"] = joblib.load(file)
        print(f"  ✓ {model_spec.persistence_filename}")

        return PipelineArtifacts(
            model_name=model_name,
            model=loaded["model"],
            imputer=loaded["imputer"],
            variance_filter=loaded["variance_filter"],
            scaler=loaded["scaler"],
            selected_features=loaded["selected_features"],
        )

    def exists(self, model_name: str = "mlp") -> bool:
        model_spec = get_model_spec(model_name)
        required = list(self._COMMON_FILENAMES.values()) + [model_spec.persistence_filename]
        return all((self._dir / fname).exists() for fname in required)
