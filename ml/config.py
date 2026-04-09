"""
Configuracoes globais do pipeline multiclasse de intrusao em SDN.

Centraliza caminhos, mapeamentos de classes, features selecionadas por
criterio de dominio e hiperparametros dos modelos suportados.
"""

from pathlib import Path

# Semente fixa para garantir reprodutibilidade
RANDOM_STATE: int = 42

# Caminhos relativos à raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "InSDN_DatasetCSV"
MODELS_DIR = PROJECT_ROOT / "models"       # onde os artefatos treinados são salvos
OUTPUTS_DIR = PROJECT_ROOT / "outputs"     # gráficos, métricas e relatórios
OUTPUTS_RUNS_DIR = OUTPUTS_DIR / "runs"   # subpasta por execução (run_id)

# Dataset
DATASET_NAME: str = "InSDN_DatasetCSV"
TARGET_COL: str = "Label"

# Agrupamento das classes originais do InSDN em 3 categorias mais amplas.
CLASS_GROUP_MAPPING: dict[str, str] = {
    "Normal": "Normal",
    "DoS": "Flooding",    # ataques de volume via um único origem
    "DDoS": "Flooding",   # ataques de volume distribuídos — mesmo grupo do DoS
    "Probe": "Intrusao",  # varredura de rede
    "BFA": "Intrusao",    # força bruta
    "Web-Attack": "Intrusao",
    "BOTNET": "Intrusao",
    "U2R": "Intrusao",    # escalada de privilégio
}
TARGET_NAMES: list[str] = ["Normal", "Flooding", "Intrusao"]  # ordem define o índice numérico

# Converte nome de classe índice (ex.: "Flooding" -> 1)
TARGET_ENCODING: dict[str, int] = {
    label: idx for idx, label in enumerate(TARGET_NAMES)
}
# Converte índice → nome (ex.: 1 -> "Flooding") — útil na inferência
TARGET_DECODING: dict[int, str] = {
    idx: label for label, idx in TARGET_ENCODING.items()
}

# Features selecionadas para evitar memorizacao
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
    "Flow IAT Mean",      # IAT = Inter-Arrival Time: tempo entre pacotes consecutivos
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Bwd IAT Tot",        # Bwd = direção de volta (resposta)
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "Down/Up Ratio",      # razão entre tráfego de download e upload
    "SYN Flag Cnt",       # flag TCP de início de conexão —> alto = possível flood
    "ACK Flag Cnt",
    "Active Mean",        # tempo médio que o fluxo ficou ativo
    "Idle Mean",          # tempo médio que o fluxo ficou ocioso
]

# Normalizar uma feature que só tem 0 e 1 não faz sentido
BINARY_PASSTHROUGH_FEATURES: list[str] = [
    "SYN Flag Cnt",
    "ACK Flag Cnt",
]

# Valores negativos sao tratados como ruido de extracao e imputados.
NON_NEGATIVE_FEATURES: list[str] = RELEVANT_FEATURES.copy()

# Split — 70% treino, 30% teste
TEST_SIZE: float = 0.30

# Pre-processamento
VARIANCE_THRESHOLD: float = 0.0   # só remove features completamente constantes (zero variância)
IMPUTER_STRATEGY: str = "median"  # outliers
SMOTE_K_NEIGHBORS: int = 5        # num. de vizinhos para o SMOTE

# Arquitetura MLP — 2 camadas ocultas (128 -> 64 neurônios)
MLP_HIDDEN_LAYERS: tuple[int, ...] = (128, 64)
MLP_ACTIVATION: str = "relu"        # relu evita problema do gradiente zero
MLP_SOLVER: str = "adam"            # bom para datasets de tamanho médio
MLP_ALPHA: float = 0.001            # penaliza pesos grandes para reduzir overfitting
MLP_MAX_ITER: int = 250             # número máximo de épocas de treinamento
MLP_LEARNING_RATE: str = "adaptive" # reduz a taxa automaticamente se a loss parar de cair
MLP_EARLY_STOP: bool = True         # interrompe o treino se a validação interna não melhorar
MLP_VAL_FRACTION: float = 0.1       # 10% do treino reservado para o critério de early stopping
MLP_N_ITER_NO_CHG: int = 12         # quantas épocas sem melhora antes de parar

# Validacao cruzada
CV_N_SPLITS: int = 3       # 3-fold. Equilibra tempo e confiabilidade
CV_SCORING: str = "f1_macro"  # f1 macro trata todas as classes com mesmo peso (bom para desbalanceamento)

# Frações do conjunto de treino usadas para plotar a curva de aprendizado
LEARNING_CURVE_TRAIN_SIZES: tuple[float, ...] = (
    0.10,
    0.25,
    0.40,
    0.60,
    0.80,
    1.00,
)

# Hyperparameter tuning -> busca aleatória no espaço de hiperparâmetros
TUNING_N_ITER: int = 12  # qtd de combinacoes testadas (mais = melhor resultado, mas mais tempo)
MLP_TUNING_PARAM_DISTRIBUTIONS: dict = {
    "hidden_layer_sizes": [   # arquiteturas candidatas a serem testadas
        (64, 32),
        (128, 64), #pode ser que o inicio ja seja a melhor escolha
        (256, 128),
        (128, 64, 32),
    ],
    "alpha": [0.0001, 0.001, 0.01, 0.05],            # força da regularização L2
    "learning_rate_init": [0.001, 0.0005, 0.0001],   # taxa de aprendizado inicial do Adam
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [200, 250, 350],
}

# Permutation importance
# Rodado no conjunto de TESTE para refletir a importância real de generalização.
PERMUTATION_IMPORTANCE_N_REPEATS: int = 10  
PERMUTATION_IMPORTANCE_SCORING: str = "f1_macro"  # medir a queda de performance
