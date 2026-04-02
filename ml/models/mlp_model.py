"""
Modelo MLP para detecção de DDoS.

SRP: este módulo encapsula exclusivamente a definição e configuração do
MLPClassifier. A lógica de treino, avaliação e persistência fica em
módulos separados.

Arquitetura: conforme Mehmood et al. (PLoS ONE, 2025)
  Input → Dense(128, ReLU) → Dense(64, ReLU) → Output(Softmax binário)
  Solver: ADAM, regularização L2, early stopping.
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier

from ml.config import (
    RANDOM_STATE,
    MLP_HIDDEN_LAYERS,
    MLP_ACTIVATION,
    MLP_SOLVER,
    MLP_ALPHA,
    MLP_MAX_ITER,
    MLP_LEARNING_RATE,
    MLP_EARLY_STOP,
    MLP_VAL_FRACTION,
    MLP_N_ITER_NO_CHG,
)


def build_baseline_mlp(random_state: int = RANDOM_STATE) -> MLPClassifier:
    """
    Constrói o MLP baseline conforme o artigo.

    Arquitetura: 128 → 64 → saída
    Solver     : ADAM
    Ativação   : ReLU
    Regulariz. : L2 (alpha=0.0001)
    EarlyStopping: sim (10 épocas sem melhora)

    Returns
    -------
    MLPClassifier configurado (não treinado).
    """
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        activation=MLP_ACTIVATION,
        solver=MLP_SOLVER,
        alpha=MLP_ALPHA,
        batch_size="auto",           # mini-batch (padrão sklearn)
        learning_rate=MLP_LEARNING_RATE,
        max_iter=MLP_MAX_ITER,
        random_state=random_state,
        early_stopping=MLP_EARLY_STOP,
        validation_fraction=MLP_VAL_FRACTION,
        n_iter_no_change=MLP_N_ITER_NO_CHG,
        verbose=False,               # silencioso — o Trainer imprime o progresso
    )


def build_mlp_from_params(params: dict, random_state: int = RANDOM_STATE) -> MLPClassifier:
    """
    Constrói um MLP com hiperparâmetros arbitrários (para tuning).

    Os parâmetros fixos (activation, solver, early_stopping) são mantidos
    do baseline; apenas os variáveis do espaço de busca são substituídos.

    Parameters
    ----------
    params : dict  — hiperparâmetros do RandomizedSearchCV
    """
    return MLPClassifier(
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=MLP_VAL_FRACTION,
        random_state=random_state,
        verbose=False,
        **params,
    )
