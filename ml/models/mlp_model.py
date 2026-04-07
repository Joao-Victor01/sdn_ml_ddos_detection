"""
Modelo MLP para classificacao multiclasse de trafego SDN.

O sklearn usa saida softmax automaticamente quando o alvo possui mais de
duas classes. A definicao da arquitetura permanece isolada neste modulo.
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
    Constroi o MLP baseline do projeto.

    Returns
    -------
    MLPClassifier configurado (nao treinado).
    """
    # Todos os hiperparâmetros vêm de config.py — nenhum valor hardcoded aqui
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,  # arquitetura: (128, 64) → 2 camadas ocultas
        activation=MLP_ACTIVATION,             # relu: rápido e evita gradiente zero
        solver=MLP_SOLVER,                     # adam: otimizador adaptativo
        alpha=MLP_ALPHA,                       # regularização L2 (penaliza pesos grandes)
        batch_size="auto",                     # sklearn define automaticamente (~200 amostras)
        learning_rate=MLP_LEARNING_RATE,
        max_iter=MLP_MAX_ITER,
        random_state=random_state,
        early_stopping=MLP_EARLY_STOP,         # para antes de atingir max_iter se não melhorar
        validation_fraction=MLP_VAL_FRACTION,
        n_iter_no_change=MLP_N_ITER_NO_CHG,
        verbose=False,
    )


def build_mlp_from_params(params: dict, random_state: int = RANDOM_STATE) -> MLPClassifier:
    """
    Constroi um MLP com hiperparametros arbitrarios (para tuning).

    Os parâmetros fixos (activation, solver, early_stopping) são mantidos
    do baseline; apenas os variáveis do espaço de busca são substituídos.

    Parameters
    ----------
    params : dict  — hiperparâmetros do RandomizedSearchCV
    """
    # Parâmetros fixos garantem que a busca não teste combinações absurdas
    # (ex.: solver diferente, sem early stopping etc.)
    return MLPClassifier(
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=MLP_VAL_FRACTION,
        random_state=random_state,
        verbose=False,
        **params,  # injeta os hiperparâmetros variáveis vindos do RandomizedSearchCV
    )
