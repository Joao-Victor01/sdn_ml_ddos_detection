"""
Modelo RandomForest para classificacao multiclasse de trafego SDN.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from ml.config import (
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MAX_FEATURES,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
)


def build_baseline_rf(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """
    Constroi o RandomForest baseline do projeto.
    """
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        random_state=random_state,
        n_jobs=-1,
    )
