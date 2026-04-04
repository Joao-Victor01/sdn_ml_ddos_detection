"""
Construção do Random Forest triclasse.

SRP: única responsabilidade é instanciar o modelo com os parâmetros corretos.

Por que RF como modelo principal (seção 6 do plano):
  - Invariante a escala: não afetado pela diferença de magnitude entre
    Flow Duration (µs) e asymmetry_pkts (0–1).
  - class_weight='balanced': compensa desbalanceamento sem dados sintéticos.
  - feature_importances_ nativo: revela quais features distinguem as classes.
  - Robusto a outliers: decisões por threshold em árvores, não por distância.

Referência: plano_triclasse_insdn_v4.md, Seção 6
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from ml.triclass.config import (
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_CLASS_WEIGHT,
)


def build_baseline_rf(
    n_estimators: int = RF_N_ESTIMATORS,
    class_weight: str = RF_CLASS_WEIGHT,
    random_state: int = RANDOM_STATE,
    **kwargs,
) -> RandomForestClassifier:
    """
    Instancia o Random Forest baseline para classificação triclasse.

    Parameters
    ----------
    n_estimators : int   — número de árvores (default 200, plano seção 8.8)
    class_weight : str   — 'balanced' compensa desbalanceamento (Classe 2)
    random_state : int   — semente de reprodutibilidade
    **kwargs             — parâmetros adicionais repassados ao RF

    Returns
    -------
    RandomForestClassifier com parâmetros do plano, não treinado.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
