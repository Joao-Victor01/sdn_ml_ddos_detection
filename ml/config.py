"""
Configurações globais do pipeline de detecção de DDoS.

Centraliza todas as constantes — caminhos, hiperparâmetros, seeds e limiares —
de forma que qualquer alteração afete apenas este módulo (SRP).
"""

from pathlib import Path

# ── Reprodutibilidade ──────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Caminhos ───────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASET_PATH  = PROJECT_ROOT / "dataset" / "insdn8_ddos_binary_0n1d.csv"
MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

# ── Coluna alvo ────────────────────────────────────────────────────────────────
TARGET_COL: str = "Label"

# ── Split ──────────────────────────────────────────────────────────────────────
# 70% treino / 30% teste — padrão do curso (Thaís Gaudencio, UFPB/LUMO)
TEST_SIZE: float = 0.30

# ── Pré-processamento ──────────────────────────────────────────────────────────
# Limiar de variância: features com variância <= threshold são removidas
VARIANCE_THRESHOLD: float = 0.0

# Estratégia de imputação: median é mais robusto para features assimétricas de rede
IMPUTER_STRATEGY: str = "median"

# ── Seleção de features (SHAP) ─────────────────────────────────────────────────
# Número máximo de instâncias para calcular SHAP (caro computacionalmente)
SHAP_SAMPLE_SIZE: int = 10_000
# Número de features a selecionar. None = manter todas (para datasets já enxutos)
N_FEATURES_TO_SELECT: int | None = None   # insdn8 já tem apenas 8 features

# ── Arquitetura MLP ────────────────────────────────────────────────────────────
# Conforme Mehmood et al. (PLoS ONE, 2025) — arquitetura baseline
MLP_HIDDEN_LAYERS: tuple = (128, 64)
MLP_ACTIVATION:    str   = "relu"
MLP_SOLVER:        str   = "adam"
MLP_ALPHA:         float = 0.0001      # regularização L2
MLP_MAX_ITER:      int   = 200
MLP_LEARNING_RATE: str   = "adaptive"  # reduz taxa se estagna
MLP_EARLY_STOP:    bool  = True
MLP_VAL_FRACTION:  float = 0.1         # 10% do treino como validação interna
MLP_N_ITER_NO_CHG: int   = 10          # ciclos sem melhora para parar

# ── Validação cruzada ──────────────────────────────────────────────────────────
CV_N_SPLITS: int = 5
CV_SCORING:  str = "f1"  # F1 equilibra precisão e recall — melhor para DDoS

# ── Hyperparameter Tuning ──────────────────────────────────────────────────────
TUNING_N_ITER: int = 30  # combinações a testar no RandomizedSearchCV
TUNING_PARAM_DISTRIBUTIONS: dict = {
    "hidden_layer_sizes": [
        (128, 64),        # baseline do artigo
        (256, 128),
        (128, 64, 32),
        (256, 128, 64),
        (512, 256, 128),
    ],
    "alpha":               [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init":  [0.001, 0.01, 0.0001],
    "learning_rate":       ["constant", "adaptive", "invscaling"],
    "max_iter":            [200, 300, 500],
}

# ── Métricas de referência (artigo Mehmood et al., 2025) ───────────────────────
PAPER_METRICS: dict = {
    "accuracy":  99.98,
    "precision": 99.99,
    "recall":    99.97,
    "f1":        99.98,
}
