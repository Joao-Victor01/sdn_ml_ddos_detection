"""
Configurações do pipeline triclasse de detecção de DDoS em SDN.

Dataset: InSDN (Elsayed et al., IEEE Access, 2020)
Classes: 0=Benigno | 1=Ataque Externo | 2=Zumbi Interno

Todas as constantes ficam aqui (SRP) — qualquer ajuste afeta apenas este arquivo.
"""

from pathlib import Path

# ── Reprodutibilidade ──────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Caminhos ───────────────────────────────────────────────────────────────────
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
INSDN_DIR           = PROJECT_ROOT / "dataset" / "InSDN_DatasetCSV"
NORMAL_CSV          = INSDN_DIR / "Normal_data.csv"
OVS_CSV             = INSDN_DIR / "OVS.csv"
META_CSV            = INSDN_DIR / "metasploitable-2.csv"
MODELS_TRICLASS_DIR = PROJECT_ROOT / "models_triclass"
OUTPUTS_TRICLASS    = PROJECT_ROOT / "outputs_triclass"

# ── Split ──────────────────────────────────────────────────────────────────────
# 70/30 — padrão do curso (Thaís Gaudencio, UFPB/LUMO)
TEST_SIZE: float = 0.30

# ── Renomeação de colunas InSDN → padrão do plano ─────────────────────────────
# InSDN usa nomenclatura CICFlowMeter diferente. Renomear antes de qualquer op.
RENAME_MAP: dict[str, str] = {
    "Tot Fwd Pkts":    "Total Fwd Packets",
    "Tot Bwd Pkts":    "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Pkt Len Std":     "Packet Length Std",
    "Fwd Pkts/s":      "Fwd Packets/s",
    "Bwd Pkts/s":      "Bwd Packets/s",
}

# ── Labels originais do InSDN usados por classe ────────────────────────────────
LABEL_BENIGN   = "Normal"
LABELS_EXTERNAL = {"DDoS"}               # somente com burst=True
LABELS_INTERNAL = {"BOTNET", "DoS"}      # BOTNET sempre; DoS somente sem burst
LABELS_DISCARD  = {"Probe", "BFA", "Web-Attack", "U2R", "UDP-lag"}

# ── Heurística de burst (seção 5.1 do plano) ──────────────────────────────────
# BOTNET tem Flow Duration ~31.000 µs >> 500 µs → is_burst() == False (correto)
# DDoS Hping3 tem duration de 1-19 µs          → is_burst() == True  (correto)
BURST_PKT_LEN_STD_MAX: float = 1.0     # Packet Length Std ≤ 1.0
BURST_FLOW_DURATION_MAX: float = 500.0  # Flow Duration < 500 µs

# ── Pré-processamento ──────────────────────────────────────────────────────────
IMPUTER_STRATEGY: str = "median"
VARIANCE_THRESHOLD: float = 0.01

# ── Features comportamentais ───────────────────────────────────────────────────
BEHAVIORAL_FEATURES: list[str] = [
    "asymmetry_pkts",
    "asymmetry_bytes",
    "pkt_rate",
    "pkt_uniformity",
    "log_duration",
    "fwd_active_ratio",
]

# ── Nomes das classes para exibição ───────────────────────────────────────────
CLASS_NAMES: dict[int, str] = {
    0: "Benigno",
    1: "Externo",
    2: "Zumbi Interno",
}

# ── SMOTE conservador (seção 8.7 do plano) ────────────────────────────────────
# Classe 2 vai para min(5x original, tamanho da Classe 0)
SMOTE_MAX_FACTOR_CLS2: int = 5
# Undersample classe 1 se for > 2x a classe 0
SMOTE_UNDERSAMPLE_RATIO_CLS1: float = 1.5

# ── Random Forest ─────────────────────────────────────────────────────────────
RF_N_ESTIMATORS: int = 200
RF_CLASS_WEIGHT: str = "balanced"

RF_TUNING_PARAM_DIST: dict = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [10, 20, 30, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}
RF_TUNING_N_ITER: int = 30

# ── MLP (comparativo) ─────────────────────────────────────────────────────────
MLP_HIDDEN_LAYERS: tuple = (128, 64)
MLP_ACTIVATION:    str   = "relu"
MLP_SOLVER:        str   = "adam"
MLP_MAX_ITER:      int   = 300
MLP_EARLY_STOP:    bool  = True
MLP_N_ITER_NO_CHG: int   = 15

# ── Validação cruzada ──────────────────────────────────────────────────────────
CV_N_SPLITS: int = 10
CV_SCORING:  str = "f1_macro"

# ── Validação semântica BOTNET ─────────────────────────────────────────────────
# Recall mínimo esperado para BOTNET no teste (seção 8.11 do plano)
BOTNET_MIN_RECALL: float = 0.80

# ── Permutation Importance / Ablation Study ────────────────────────────────────
# Features suspeitas de ser "atalhos de identidade": o modelo pode usar
# porta/protocolo como proxy do tipo de tráfego em vez de aprender padrões
# comportamentais reais. Em produção, essas features podem não estar disponíveis
# ou ter distribuição diferente (ex: porta 80 pode ser legítimo ou ataque).
IDENTITY_FEATURES: list[str] = [
    "Dst Port",
    "Src Port",
    "Protocol",
    "Bwd Header Len",
    "Init Bwd Win Byts",
    "Fwd Header Len",
]
# Número de repetições da permutação (30 é o mínimo para estimativa estável)
PERMUTATION_N_REPEATS: int = 30
