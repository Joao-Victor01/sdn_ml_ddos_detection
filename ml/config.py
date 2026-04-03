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

# ── Configurações do pipeline multiclasse ──────────────────────────────────────
# Mapeamento de classes (0=Benigno, 1=Externo, 2=Interno/Zumbi)
CLASS_NAMES:  list[str] = ["Benigno", "Ataque Externo", "Zumbi Interno"]
CLASS_LABELS: list[int] = [0, 1, 2]
N_CLASSES:    int       = 3

# Scoring multiclasse: macro equilibra as três classes igualmente
CV_SCORING_MULTI: str = "f1_macro"

# ── Engenharia de labels (heurística de separação Externo vs. Interno) ─────────
# Critérios para classificar um ataque DDoS como "Externo":
#   1. Protocol = 0 (ICMP/raw — ferramenta de flood trivial)
#   2. OU pacotes uniformes (Pkt Len Std ≈ 0) E fluxo muito curto
# O complemento dos ataques que não se enquadram → "Zumbi Interno"
EXTERNAL_PROTOCOL_ID:      int   = 0       # protocolo indicativo de flood externo
EXTERNAL_DURATION_THRESH:  float = 500.0   # µs — fluxos < 500µs = rajada externa
EXTERNAL_STD_THRESH:       float = 1e-3    # Pkt Len Std ≈ 0 → pacotes uniformes

# ── Engenharia de features HCF (Hop Count Filtering) ──────────────────────────
# Distribuições realistas de TTL por tipo de tráfego (µs estimados de SO padrão):
#   Externo: TTL inicial = 64 (Linux) → chega ≈ 48 após ~16 hops internet
#   Interno: TTL inicial = 64         → chega ≈ 62 após 1-2 hops LAN
#   Benigno: TTL inicial = 64         → chega ≈ 61 (variação normal LAN)
TTL_EXTERNAL_MEAN: float = 48.0;  TTL_EXTERNAL_STD: float = 10.0
TTL_INTERNAL_MEAN: float = 62.0;  TTL_INTERNAL_STD: float = 2.0
TTL_BENIGN_MEAN:   float = 61.0;  TTL_BENIGN_STD:   float = 3.0
TTL_LAN_EXPECTED:  int   = 64     # TTL esperado para host local sem hops

# HCF: diferença de TTL que indica origem externa (passou por muitos roteadores)
HCF_EXTERNAL_THRESHOLD: int = 10   # TTL_LAN_EXPECTED - TTL < 10 → interno

# ── Modelos multiclasse (diretório separado para não sobrescrever binário) ──────
MODELS_DIR_MULTI = MODELS_DIR.parent / "models_multiclass"
