"""
Configuracoes globais do pipeline multiclasse de intrusao em SDN.

Centraliza caminhos, mapeamentos de classes, features selecionadas por
criterio de dominio e hiperparametros dos modelos suportados.

Ponto único de configuração — mude aqui e muda em todo o projeto.
"""

from pathlib import Path

# Semente fixa para garantir reprodutibilidade: mesmo shuffle, mesma inicialização de pesos
RANDOM_STATE: int = 42

# Caminhos — todos relativos à raiz do projeto, então funcionam de qualquer diretório
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "InSDN_DatasetCSV"
MODELS_DIR = PROJECT_ROOT / "models"       # onde os artefatos treinados são salvos
OUTPUTS_DIR = PROJECT_ROOT / "outputs"     # gráficos, métricas e relatórios
OUTPUTS_RUNS_DIR = OUTPUTS_DIR / "runs"   # subpasta por execução (run_id)

# Dataset
DATASET_NAME: str = "InSDN_DatasetCSV"
TARGET_COL: str = "Label"

# Agrupamento das classes originais do InSDN em 3 categorias mais amplas.
# Fazemos isso porque as classes originais têm amostras muito desiguais e
# algumas são semanticamente parecidas (ex.: DoS e DDoS são ambas inundação).
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

# Converte nome de classe índice (ex.: "Flooding" → 1)
TARGET_ENCODING: dict[str, int] = {
    label: idx for idx, label in enumerate(TARGET_NAMES)
}
# Converte índice → nome (ex.: 1 → "Flooding") — útil na inferência
TARGET_DECODING: dict[int, str] = {
    idx: label for label, idx in TARGET_ENCODING.items()
}

# Features selecionadas por criterio de dominio para evitar memorizacao
# de identificadores do ambiente (IP, portas, timestamps, flow id).
# São estatísticas de fluxo que descrevem o *comportamento* do tráfego,
# não quem está comunicando — isso torna o modelo mais generalizável.
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
    "SYN Flag Cnt",       # flag TCP de início de conexão — alto = possível flood
    "ACK Flag Cnt",
    "Active Mean",        # tempo médio que o fluxo ficou ativo
    "Idle Mean",          # tempo médio que o fluxo ficou ocioso
]

# Colunas binárias devem ser preservadas sem padronizacao.
# Normalizar uma feature que só tem 0 e 1 não faz sentido
# e pode distorcer o comportamento da rede neural.
BINARY_PASSTHROUGH_FEATURES: list[str] = [
    "SYN Flag Cnt",
    "ACK Flag Cnt",
]

# Todas as features acima sao estatisticas nao negativas por definicao.
# Valores negativos sao tratados como ruido de extracao e imputados.
NON_NEGATIVE_FEATURES: list[str] = RELEVANT_FEATURES.copy()

# Split — 70% treino, 30% teste
TEST_SIZE: float = 0.30

# Pre-processamento
VARIANCE_THRESHOLD: float = 0.0   # só remove features completamente constantes (zero variância)
IMPUTER_STRATEGY: str = "median"  # mediana é mais robusta que média quando há outliers
SMOTE_K_NEIGHBORS: int = 5        # número de vizinhos usados pelo SMOTE para gerar amostras sintéticas

# Arquitetura MLP — 2 camadas ocultas (128 → 64 neurônios)
MLP_HIDDEN_LAYERS: tuple[int, ...] = (128, 64)
MLP_ACTIVATION: str = "relu"        # relu é o padrão moderno — evita problema do gradiente zero
MLP_SOLVER: str = "adam"            # otimizador adaptativo, bom para datasets de tamanho médio
MLP_ALPHA: float = 0.001            # regularização L2 — penaliza pesos grandes para reduzir overfitting
MLP_MAX_ITER: int = 250             # número máximo de épocas de treinamento
MLP_LEARNING_RATE: str = "adaptive" # reduz a taxa automaticamente se a loss parar de cair
MLP_EARLY_STOP: bool = True         # interrompe o treino se a validação interna não melhorar
MLP_VAL_FRACTION: float = 0.1       # 10% do treino reservado para o critério de early stopping
MLP_N_ITER_NO_CHG: int = 12         # quantas épocas sem melhora antes de parar

# Validacao cruzada
CV_N_SPLITS: int = 3       # 3-fold — equilibrio razoável entre tempo e confiabilidade
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

# Hyperparameter tuning — busca aleatória no espaço de hiperparâmetros
TUNING_N_ITER: int = 12  # quantas combinações aleatórias testar (mais = melhor, porém mais lento)
MLP_TUNING_PARAM_DISTRIBUTIONS: dict = {
    "hidden_layer_sizes": [   # arquiteturas candidatas a serem testadas
        (64, 32),
        (128, 64),
        (256, 128),
        (128, 64, 32),
    ],
    "alpha": [0.0001, 0.001, 0.01, 0.05],            # força da regularização L2
    "learning_rate_init": [0.001, 0.0005, 0.0001],   # taxa de aprendizado inicial do Adam
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [200, 250, 350],
}

# Permutation importance — mede o impacto de cada feature embaralhando-a e
# observando a queda no score do modelo. Funciona com qualquer estimador sklearn.
# Rodado no conjunto de TESTE para refletir a importância real de generalização.
PERMUTATION_IMPORTANCE_N_REPEATS: int = 10   # repetições por feature (mais = estimativa mais estável)
PERMUTATION_IMPORTANCE_SCORING: str = "f1_macro"  # métrica usada para medir a queda de performance
