"""
Registro central dos modelos supervisionados suportados pelo pipeline.

Cada especificação encapsula apenas o que varia entre os modelos:
construtor baseline, tuning, artefato de persistência e capacidades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import ClassifierMixin

from ml.config import MLP_TUNING_PARAM_DISTRIBUTIONS
from ml.models.mlp_model import build_baseline_mlp


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    persistence_filename: str
    build_baseline: Callable[[int], ClassifierMixin]
    tracked_params: tuple[str, ...]
    param_distributions: dict | None = None
    supports_loss_curve: bool = False
    supports_permutation_importance: bool = False

    @property
    def supports_tuning(self) -> bool:
        return bool(self.param_distributions)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mlp": ModelSpec(
        key="mlp",
        display_name="MLP",
        persistence_filename="model_mlp.joblib",
        build_baseline=build_baseline_mlp,
        tracked_params=(
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "learning_rate",
            "max_iter",
        ),
        param_distributions=MLP_TUNING_PARAM_DISTRIBUTIONS,
        supports_loss_curve=True,
        supports_permutation_importance=True,
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        raise ValueError(
            f"Modelo '{model_key}' nao suportado. Opcoes: {sorted(MODEL_REGISTRY)}"
        ) from exc


def resolve_requested_models(model_key: str) -> list[ModelSpec]:
    return [get_model_spec(model_key)]
