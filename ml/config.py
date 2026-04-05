"""
Configuracoes globais do pipeline multiclasse de intrusao em SDN.

Centraliza caminhos, mapeamentos de classes, features selecionadas por
criterio de dominio e hiperparametros do MLP.
"""

from pathlib import Path

# Reprodutibilidade
RANDOM_STATE: int = 42

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "InSDN_DatasetCSV"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dataset
DATASET_NAME: str = "InSDN_DatasetCSV"
TARGET_COL: str = "Label"

# Cenario multiclasse
CLASS_GROUP_MAPPING: dict[str, str] = {
    "Normal": "Normal",
    "DoS": "Flooding",
    "DDoS": "Flooding",
    "Probe": "Intrusao",
    "BFA": "Intrusao",
    "Web-Attack": "Intrusao",
    "BOTNET": "Intrusao",
    "U2R": "Intrusao",
}
TARGET_NAMES: list[str] = ["Normal", "Flooding", "Intrusao"]
TARGET_ENCODING: dict[str, int] = {
    label: idx for idx, label in enumerate(TARGET_NAMES)
}
TARGET_DECODING: dict[int, str] = {
    idx: label for label, idx in TARGET_ENCODING.items()
}

# Features selecionadas por criterio de dominio para evitar memorizacao
# de identificadores do ambiente (IP, portas, timestamps, flow id).
RELEVANT_FEATURES: list[str] = [
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "Down/Up Ratio",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "Active Mean",
    "Idle Mean",
]

# Todas as features acima sao estatisticas nao negativas por definicao.
# Valores negativos sao tratados como ruido de extracao e imputados.
NON_NEGATIVE_FEATURES: list[str] = RELEVANT_FEATURES.copy()

# Split
TEST_SIZE: float = 0.30

# Pre-processamento
VARIANCE_THRESHOLD: float = 0.0
IMPUTER_STRATEGY: str = "median"
SMOTE_K_NEIGHBORS: int = 5

# Selecao de features
SHAP_SAMPLE_SIZE: int = 10_000
N_FEATURES_TO_SELECT: int | None = None

# Arquitetura MLP
MLP_HIDDEN_LAYERS: tuple[int, ...] = (128, 64)
MLP_ACTIVATION: str = "relu"
MLP_SOLVER: str = "adam"
MLP_ALPHA: float = 0.001
MLP_MAX_ITER: int = 250
MLP_LEARNING_RATE: str = "adaptive"
MLP_EARLY_STOP: bool = True
MLP_VAL_FRACTION: float = 0.1
MLP_N_ITER_NO_CHG: int = 12

# Validacao cruzada
CV_N_SPLITS: int = 3
CV_SCORING: str = "f1_macro"

# Learning curve
LEARNING_CURVE_TRAIN_SIZES: tuple[float, ...] = (
    0.10,
    0.25,
    0.40,
    0.60,
    0.80,
    1.00,
)

# Hyperparameter tuning
TUNING_N_ITER: int = 12
TUNING_PARAM_DISTRIBUTIONS: dict = {
    "hidden_layer_sizes": [
        (64, 32),
        (128, 64),
        (256, 128),
        (128, 64, 32),
    ],
    "alpha": [0.0001, 0.001, 0.01, 0.05],
    "learning_rate_init": [0.001, 0.0005, 0.0001],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [200, 250, 350],
}
