# Plano de Implementação: Detecção Triclasse de DDoS em SDN
## Benigno | Ataque Externo | Zumbi Interno (PC comprometido)

> **Versão**: 3.0 — Classificação triclasse sem dependência de TTL  
> **Dataset**: CICDDoS2019  
> **Modelo principal**: Random Forest (com MLP como comparativo)  
> **Contexto**: Extensão do sistema binário MLP existente para detecção de origem do ataque

---

## Índice

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [As Três Classes do Problema](#2-as-três-classes-do-problema)
3. [Por que Não Usar TTL Nesta Versão](#3-por-que-não-usar-ttl-nesta-versão)
4. [As Novas Heurísticas](#4-as-novas-heurísticas)
5. [Análise Exploratória do CICDDoS2019](#5-análise-exploratória-do-cicdddos2019)
6. [Escolha do Modelo](#6-escolha-do-modelo)
7. [Pipeline Completo](#7-pipeline-completo)
8. [Integração com o Sistema Existente](#8-integração-com-o-sistema-existente)
9. [Limitações a Declarar](#9-limitações-a-declarar)
10. [Referências](#10-referências)

---

## 1. Contexto e Motivação

### 1.1 O que existe hoje

O sistema atual (v1.0) realiza **detecção binária**:

```
Fluxo de rede → MLP → 0 (Benigno) | 1 (Ataque DDoS)
```

O MLP foi treinado no dataset InSDN8 e atingiu ~97.9% de acurácia na tarefa binária. A resposta do sistema é sempre a mesma independente do tipo de ataque: bloquear o IP de origem em todos os switches da topologia SDN.

### 1.2 O problema que isso cria

Um ataque DDoS pode ter duas origens completamente diferentes, e a resposta correta é diferente para cada uma:

**Origem externa**: um servidor na internet envia pacotes com IP spoofado ou participa de amplificação (NTP, DNS reflection). Bloquear o IP de origem em todos os switches de borda é a resposta correta.

**Origem interna (zumbi)**: um PC da própria rede interna — por exemplo, um computador de laboratório — foi infectado por malware e está participando de uma botnet sem que o usuário saiba. Bloquear o IP em toda a rede seria um erro: prejudicaria toda a VLAN e o computador infectado ainda pode ter dados importantes. A resposta correta é o **isolamento cirúrgico**: bloquear apenas a porta do switch onde aquele host está conectado.

### 1.3 O que esta versão adiciona

```
Fluxo de rede → RF/MLP triclasse → 0 (Benigno)
                                  → 1 (Ataque Externo) → block global
                                  → 2 (Zumbi Interno)  → isolamento cirúrgico
```

A distinção entre classe 1 e classe 2 é feita por **padrão comportamental do fluxo**, não por TTL — porque o CICDDoS2019, como a maioria dos datasets públicos de rede, não possui a coluna TTL bruto por pacote necessária para o Hop Count Filtering original.

---

## 2. As Três Classes do Problema

| Classe | Label | Origem | Comportamento típico | Resposta do sistema |
|--------|-------|--------|---------------------|---------------------|
| 0 | Benigno | Tráfego legítimo | Bidirecional, entropia alta, duração variada | Nenhuma ação |
| 1 | Ataque Externo | Internet / IP spoofado | Assimetria extrema, pacotes uniformes, amplificação | Block global em todos os switches |
| 2 | Zumbi Interno | Host da LAN comprometido | Assimetria alta, mas IP registrado no SDN, fluxos persistentes | Isolamento cirúrgico na porta de borda |

### 2.1 Como distinguir 1 de 2 sem TTL

A intuição central:

- Um ataque externo gera tráfego **de fora para dentro** — enorme volume de pacotes forward, quase sem resposta backward, pacotes uniformes (gerados por script), fluxos curtíssimos ou por amplificação UDP.
- Um zumbi interno gera tráfego **de dentro para fora** — também assimétrico, mas com características de host legítimo: IP registrado no controlador SDN, possível tentativa de handshake TCP, fluxos ligeiramente mais longos.

A diferença mais confiável é o `is_known_host`: se o IP de origem está registrado no `state.ip_to_mac` do controlador SDN (o host se comunicou anteriormente de forma legítima), é definitivamente interno.

---

## 3. Por que Não Usar TTL Nesta Versão

O Hop Count Filtering (HCF) original — proposto por Jin et al. (ACM CCS 2003) — usa o TTL observado no pacote para estimar quantos roteadores ele cruzou. TTL baixo ao chegar → origem distante → externo. TTL alto → origem próxima → interno.

O problema é que **nenhum dataset SDN público disponível contém TTL bruto por pacote** de forma utilizável:

- **CICDDoS2019**: não possui coluna TTL.
- **InSDN8**: não possui coluna TTL.
- **UNSW-NB15**: possui `sttl` e `dttl`, mas são TTL médio por estado de conexão (agregado por fluxo), não o TTL bruto de cada pacote. Além disso, não é SDN-específico.
- **HLD-DDoSDN, TCP-SYN SDN Dataset**: SDN-específicos, mas sem TTL explícito.

A abordagem desta versão substitui o TTL por **features comportamentais** extraíveis diretamente do CICDDoS2019, embasadas em literatura recente (Tian & Miyata, IoT 2023; Santos-Neto et al., 2024; Aljahdali, 2025).

---

## 4. As Novas Heurísticas

### 4.1 Heurística de Labeling (binário → triclasse)

O CICDDoS2019 é originalmente binário: `BENIGN` vs. tipo\_de\_ataque. Para criar as três classes, usamos uma heurística baseada no **tipo de ataque** (informação presente no label original) e no **comportamento de fluxo**:

```python
def criar_label_triclasse(df):
    """
    Regras:
    
    Classe 0 — Benigno:
        Label original == 'BENIGN'
    
    Classe 1 — Ataque Externo:
        Ataques de amplificação/reflexão (NTP, DNS, LDAP, MSSQL,
        NetBIOS, SNMP, SSDP, UDP amplification):
            → Tipicamente gerados externamente, IP spoofado
        OU comportamento de rajada uniforme:
            → Packet Length Std ≤ 1.0 (pacotes idênticos, gerado por script)
            AND Flow Duration < 1000 µs (fluxo curtíssimo)
    
    Classe 2 — Zumbi Interno:
        Demais ataques não classificados como externos
        (SYN floods, TFTP, WebDDoS, etc.)
            → Comportamento mais próximo de host legítimo infectado
    
    ⚠️ LIMITAÇÃO: Esta heurística usa o label original para
    determinar o tipo de ataque. As métricas de F1 para as classes
    1 e 2 refletem a capacidade do modelo de reproduzir essa
    heurística, não necessariamente a distinção real em produção.
    Declarar explicitamente na seção de limitações do paper.
    """
    
    y = pd.Series(0, index=df.index)
    attack_mask = df['Label'].str.upper() != 'BENIGN'
    
    amplification_types = [
        'ntp', 'dns', 'ldap', 'mssql',
        'netbios', 'snmp', 'ssdp', 'udp'
    ]
    external_by_type = df['Label'].str.lower().apply(
        lambda x: any(t in x for t in amplification_types)
    )
    
    burst_behavior = (
        (df['Packet Length Std'] <= 1.0) &
        (df['Flow Duration']     < 1000.0)
    )
    
    external_mask = attack_mask & (external_by_type | burst_behavior)
    internal_mask = attack_mask & ~external_mask
    
    y[external_mask] = 1
    y[internal_mask] = 2
    
    return y
```

### 4.2 Heurísticas de Features Comportamentais

Estas são as features computadas que substituem o TTL. Todas são derivadas de features originais do CICDDoS2019:

#### Feature 1 — Assimetria de pacotes (`asymmetry_pkts`)

```python
asymmetry_pkts = Total_Fwd_Packets / (Total_Fwd_Packets + Total_Bwd_Packets + 1e-9)
```

**Intuição**: tráfego legítimo é bidirecional (~0.5). Ataques DDoS enviam muito mais para a frente do que recebem (~1.0). Tanto ataques externos quanto zumbis têm assimetria alta — mas o zumbi pode ter alguma resposta por ainda tentar estabelecer conexão.

**Intervalo esperado**: 0 (só backward) a 1 (só forward).

#### Feature 2 — Assimetria de bytes (`asymmetry_bytes`)

```python
asymmetry_bytes = Total_Fwd_Bytes / (Total_Fwd_Bytes + Total_Bwd_Bytes + 1e-9)
```

**Intuição**: amplificação DNS/NTP tem assimetria inversa — o atacante envia pacotes pequenos e a amplificação gera respostas grandes para a vítima. Isso cria um padrão de assimetria distinto do flood direto de um zumbi.

#### Feature 3 — Taxa de pacotes (`pkt_rate`)

```python
pkt_rate = (Total_Fwd_Packets + Total_Bwd_Packets) / (Flow_Duration + 1e-9)
```

**Intuição**: ataques DDoS têm taxas muito mais altas que tráfego normal. A taxa absoluta distingue ataques de tráfego pesado legítimo.

#### Feature 4 — Uniformidade dos pacotes (`pkt_uniformity`)

```python
pkt_uniformity = 1.0 / (Packet_Length_Std + 1.0)
```

**Intuição**: scripts de ataque geram pacotes de tamanho idêntico (Std ≈ 0). Tráfego humano e de aplicação tem tamanhos variados. Std alto → uniformidade baixa → mais provável benigno ou humano. Std ≈ 0 → uniformidade alta → gerado por script.

#### Feature 5 — Duração logarítmica do fluxo (`log_duration`)

```python
log_duration = log1p(Flow_Duration)
```

**Intuição**: a duração do fluxo é altamente assimétrica — transformação log1p (ensinada na Aula 5 para atributos de cauda longa) normaliza a distribuição para o modelo. Ataques de amplificação têm duração muito curta; zumbis em botnet mantêm conexões por mais tempo.

#### Feature 6 — Fração de pacotes ativos forward (`fwd_active_ratio`)

```python
fwd_active_ratio = Fwd_Act_Data_Pkts / (Total_Fwd_Packets + 1e-9)
```

**Intuição**: um SYN flood (típico de zumbi em botnet) manda SYN mas nunca dados reais — `Fwd_Act_Data_Pkts` ≈ 0. Tráfego legítimo tem dados. Amplificação UDP não tem SYN, mas também tem ratio baixo.

---

## 5. Análise Exploratória do CICDDoS2019

> **⚠️ Esta seção deve ser preenchida com os resultados reais após carregar o dataset.**
> Os blocos de código abaixo executam a análise — os resultados e interpretações
> dependem do que for encontrado nos dados.

### 5.1 Carregamento e visão geral

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os

DATA_PATH = 'data/CICDDoS2019/'
all_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))

df_list = []
for filepath in all_files:
    df_tmp = pd.read_csv(filepath, low_memory=False)
    df_tmp.columns = df_tmp.columns.str.strip()
    nome = os.path.basename(filepath)
    print(f"  {nome}: {df_tmp.shape}")
    df_list.append(df_tmp)

data = pd.concat(df_list, ignore_index=True)
print(f"\nShape total: {data.shape}")
```

**[RESULTADO A PREENCHER]:**
- Shape total: ___ linhas × ___ colunas
- Arquivos carregados: ___
- Tipos de ataque presentes: ___

### 5.2 Distribuição das classes originais (label binário)

```python
print("Distribuição original (label binário CICDDoS2019):")
print(data['Label'].value_counts())
print()
print(data['Label'].value_counts(normalize=True).mul(100).round(1))

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
contagem = data['Label'].value_counts()
cores = sns.color_palette("Set2", len(contagem))
axes[0].bar(contagem.index, contagem.values, color=cores)
axes[0].set_title('Contagem por tipo de ataque')
axes[0].tick_params(axis='x', rotation=45)
axes[1].pie(contagem.values, labels=contagem.index, autopct='%1.1f%%', colors=cores)
axes[1].set_title('Proporção')
plt.tight_layout()
plt.show()
```

**[RESULTADO A PREENCHER]:**
- Proporção BENIGN: ___%
- Proporção ataques: ___%
- Tipos de ataque e suas proporções: ___

### 5.3 Verificação das colunas necessárias para as novas features

```python
colunas_necessarias = {
    'Total Fwd Packets':         'assimetria de pacotes',
    'Total Backward Packets':    'assimetria de pacotes',
    'Total Length of Fwd Packets': 'assimetria de bytes',
    'Total Length of Bwd Packets': 'assimetria de bytes',
    'Flow Duration':             'taxa e duração',
    'Packet Length Std':         'uniformidade',
    'Fwd Packets/s':             'taxa forward',
    'Fwd Act Data Pkts':         'fração ativa',
}

print("Verificação de colunas necessárias:")
for col, uso in colunas_necessarias.items():
    status = '✓' if col in data.columns else '✗ AUSENTE'
    print(f"  {status} [{uso}] {col}")
```

**[RESULTADO A PREENCHER]:**
- Colunas presentes: ___
- Colunas ausentes (ajustar nomes): ___
- Ajustes necessários nos nomes das colunas: ___

### 5.4 Verificação de valores ausentes, infinitos e duplicatas

```python
# Missing values
print(f"Missing values: {data.isnull().sum().sum()}")
missing_por_col = data.isnull().sum()
print(missing_por_col[missing_por_col > 0])

# Infinitos (comuns em features de rede — divisão por zero no CICFlowMeter)
inf_count = np.isinf(data.select_dtypes(include=np.number)).sum()
print(f"\nInfinitos por coluna:")
print(inf_count[inf_count > 0])

# Duplicatas
print(f"\nDuplicatas: {data.duplicated().sum()}")
```

**[RESULTADO A PREENCHER]:**
- Total de missing values: ___
- Colunas com infinitos: ___
- Total de duplicatas: ___
- Estratégia adotada: ___

### 5.5 Análise das features de assimetria e fluxo por tipo de ataque

> Esta análise valida empiricamente se as heurísticas comportamentais fazem sentido.
> Se os tipos de ataque não tiverem padrões distintos nas features de assimetria,
> a abordagem precisa ser revisada antes de prosseguir.

```python
# Calcular assimetria temporariamente para EDA
data['_asym_pkts_eda'] = (
    data['Total Fwd Packets'] /
    (data['Total Fwd Packets'] + data['Total Backward Packets'] + 1e-9)
)

# Boxplot de assimetria por tipo de ataque
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

data.boxplot(
    column='_asym_pkts_eda',
    by='Label',
    ax=axes[0]
)
axes[0].set_title('Assimetria de Pacotes por Tipo de Ataque')
axes[0].set_xlabel('Tipo')
axes[0].set_ylabel('asymmetry_pkts')
axes[0].tick_params(axis='x', rotation=45)

data.boxplot(
    column='Packet Length Std',
    by='Label',
    ax=axes[1]
)
axes[1].set_title('Uniformidade de Pacotes (Std) por Tipo de Ataque')
axes[1].set_xlabel('Tipo')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**[RESULTADO A PREENCHER E INTERPRETAR]:**

Para cada tipo de ataque, verificar:
- Ataques de amplificação (NTP, DNS, LDAP) têm assimetria próxima de 1.0? ___
- Ataques de flood direto (SYN, UDP) têm padrão diferente? ___
- Tráfego BENIGN tem assimetria próxima de 0.5? ___
- A uniformidade (Packet Length Std) diferencia scripts de ataques de tráfego legítimo? ___

**Conclusão da EDA (a preencher):**
> Com base nos resultados acima, a heurística de labeling [faz / não faz] sentido
> para este dataset porque ___. As features comportamentais [conseguem / não conseguem]
> distinguir os padrões porque ___.

### 5.6 Distribuição triclasse após aplicar a heurística

```python
# Aplicar heurística de labeling
data['label_3class'] = criar_label_triclasse(data)

print("Distribuição triclasse:")
counts = data['label_3class'].value_counts().sort_index()
for cls, count in counts.items():
    nomes = {0: 'Benigno', 1: 'Externo', 2: 'Zumbi Interno'}
    pct = 100 * count / len(data)
    print(f"  Classe {cls} ({nomes[cls]}): {count:,} ({pct:.1f}%)")
```

**[RESULTADO A PREENCHER]:**
- Classe 0 (Benigno): ___ amostras (___%)
- Classe 1 (Externo): ___ amostras (___%)
- Classe 2 (Zumbi Interno): ___ amostras (___%)
- Grau de desbalanceamento: ___
- Necessidade de SMOTE: ___

### 5.7 Validação da separabilidade das classes (PCA visual)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Selecionar subset de features numéricas para PCA exploratório
feats_pca = [
    'Total Fwd Packets', 'Total Backward Packets',
    'Flow Duration', 'Packet Length Std',
    'Fwd Packets/s', 'Bwd Packets/s'
]
feats_pca = [f for f in feats_pca if f in data.columns]

# Amostra para viabilidade
amostra = data.sample(n=min(10000, len(data)), random_state=42)
X_pca = amostra[feats_pca].replace([np.inf, -np.inf], np.nan).fillna(0)

scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X_pca)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled_pca)

cores_cls = {0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}
labels_cls = {0: 'Benigno', 1: 'Externo', 2: 'Zumbi Interno'}

fig, ax = plt.subplots(figsize=(10, 7))
for cls in [0, 1, 2]:
    mask = amostra['label_3class'] == cls
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=cores_cls[cls], label=labels_cls[cls],
        alpha=0.4, s=10
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)')
ax.set_title('Separabilidade das 3 Classes — PCA 2D (exploratório)')
ax.legend()
plt.tight_layout()
plt.show()
```

**[RESULTADO A PREENCHER E INTERPRETAR]:**
- Variância explicada pelos 2 componentes: ___%
- As classes estão visualmente separadas? ___
- Há sobreposição significativa entre classes 1 e 2? ___
- Isso indica que o modelo vai / não vai conseguir separar bem? ___

---

## 6. Escolha do Modelo

### 6.1 Por que Random Forest como modelo principal

Baseado no que foi ensinado nas aulas:

**Aula 8 — Random Forest:**
> *"Ensemble de DTs — reduz variância por amostragem aleatória de features a cada nó."*
> *"GradientBoosting > HistGradientBoosting em datasets pequenos (<10k); RandomForest = ensemble de DTs reduz variância."*

As features de entropia e assimetria funcionam naturalmente como **limiares de decisão**: "assimetria acima de 0.85 e uniformidade acima de 0.9 → ataque". Árvores de decisão capturam exatamente esse tipo de fronteira. O Random Forest combina centenas dessas árvores com diferentes amostras de features, produzindo fronteiras robustas.

Vantagens adicionais relevantes para este problema:

- **Invariante a escala**: não precisa de `StandardScaler`. As features de assimetria (0 a 1) e taxa de pacotes (0 a 10^6) têm escalas completamente diferentes — o RF não é afetado por isso, ao contrário do MLP.
- **`feature_importances_` nativo**: vai revelar se as novas features comportamentais realmente contribuem ou se o modelo está usando apenas as features originais do CICDDoS2019.
- **`class_weight='balanced'`**: compensa desbalanceamento sem criar dados sintéticos, como ensinado na Aula 7.

### 6.2 Por que MLP como comparativo

O MLP (Aula 10) captura relações não-lineares entre atributos — útil se a distinção entre classes não for capturável por limiares simples. Incluir o MLP como comparativo responde empiricamente se a complexidade adicional compensa.

O MLP exige `Pipeline(StandardScaler → MLP)` para evitar data leakage na normalização (Aula 5).

### 6.3 Métricas de avaliação

Seguindo a Aula 7 para problemas multiclasse com desbalanceamento:

| Métrica | Por que usar aqui |
|---|---|
| **F1 Macro** | Todas as classes pesam igual — evita que Benigno (majoritário) domine |
| **MCC** | Robusto a desbalanceamento — Aula 7: *"MCC ∈ [-1,1], 0 = aleatório"* |
| **Geometric Mean** | Penaliza modelos que ignoram classes minoritárias |
| **Recall por classe** | Recall de Zumbi Interno é mais crítico que F1 geral |

**Não usar**: Acurácia sozinha (Aula 7: *"em classes desbalanceadas, acurácia sozinha não serve"*) e F1-weighted (favorece a classe majoritária, escondendo falhas na detecção de Externo e Interno).

---

## 7. Pipeline Completo

### 7.0 Configurações globais

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, glob, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score,
                              matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
import joblib

SEED = 42
np.random.seed(SEED)
```

---

### 7.1 Carregamento do CICDDoS2019

```python
DATA_PATH = 'data/CICDDoS2019/'
all_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))

df_list = []
for filepath in all_files:
    df_tmp = pd.read_csv(filepath, low_memory=False)
    # Remover espaços dos nomes das colunas (CICFlowMeter os inclui)
    df_tmp.columns = df_tmp.columns.str.strip()
    print(f"  {os.path.basename(filepath)}: {df_tmp.shape}")
    df_list.append(df_tmp)

data = pd.concat(df_list, ignore_index=True)
print(f"\nDataset completo: {data.shape}")
print(f"Distribuição original:\n{data['Label'].value_counts()}")
```

---

### 7.2 EDA inicial (observar, sem modificar)

```python
# Ver Seção 5 — executar toda a EDA antes de prosseguir
# As decisões das etapas seguintes dependem do que for encontrado aqui
```

> **Ponto de decisão**: após a EDA, verificar se os nomes das colunas
> necessárias para as features comportamentais existem. Se não, adaptar
> os nomes antes de prosseguir.

---

### 7.3 Engenharia de Labels (binário → triclasse)

```python
# ⚠️ ETAPA CRÍTICA: criar as 3 classes ANTES do split
# LabelEngineer é determinístico — usa apenas o label original e features básicas
# Não há leakage porque não aprende parâmetros do treino

def criar_label_triclasse(df):
    """
    Converte o dataset binário CICDDoS2019 em triclasse.
    Ver Seção 4.1 para justificativa completa das regras.
    
    LIMITAÇÃO A DECLARAR: heurística baseada no tipo de ataque original.
    """
    y = pd.Series(0, index=df.index)
    attack_mask = df['Label'].str.upper().str.strip() != 'BENIGN'
    
    amplification_types = ['ntp', 'dns', 'ldap', 'mssql',
                           'netbios', 'snmp', 'ssdp', 'udp']
    external_by_type = df['Label'].str.lower().apply(
        lambda x: any(t in x for t in amplification_types)
    )
    
    # Ajustar nomes das colunas conforme resultado da EDA (Seção 5.3)
    pkt_std_col  = 'Packet Length Std'
    duration_col = 'Flow Duration'
    
    burst_behavior = (
        (df[pkt_std_col]  <= 1.0)  &
        (df[duration_col] < 1000.0)
    )
    
    external_mask = attack_mask & (external_by_type | burst_behavior)
    internal_mask = attack_mask & ~external_mask
    
    y[external_mask] = 1
    y[internal_mask] = 2
    
    return y

data['label_3class'] = criar_label_triclasse(data)

print("Distribuição triclasse:")
mapa = {0: 'Benigno', 1: 'Externo', 2: 'Zumbi Interno'}
for cls, nome in mapa.items():
    n = (data['label_3class'] == cls).sum()
    pct = 100 * n / len(data)
    print(f"  Classe {cls} ({nome}): {n:,} ({pct:.1f}%)")
```

---

### 7.4 ⚠️ SPLIT ANTES DE TUDO (regra de ouro — Aula 5)

```python
# Separar X e y ANTES de qualquer transformação
# Regra absoluta das aulas: split primeiro, transformações depois

TARGET = 'label_3class'
cols_drop = ['Label', TARGET]

X = data.drop(columns=[c for c in cols_drop if c in data.columns],
              errors='ignore')
X = X.select_dtypes(include=np.number)
y = data[TARGET]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split estratificado — stratify=y OBRIGATÓRIO para multiclasse desbalanceada
# Aula 7: StratifiedKFold preserva proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

print(f"\nTreino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
print("\nProporção treino:")
print(y_train.value_counts(normalize=True).mul(100).round(1))
print("\nProporção teste:")
print(y_test.value_counts(normalize=True).mul(100).round(1))
# As proporções devem ser idênticas — confirmar que stratify funcionou
```

---

### 7.5 Limpeza (somente no treino)

```python
# Substituir Inf por NaN — comum em features de rede (CICFlowMeter gera Inf
# quando Flow Duration = 0 e calcula bytes/s)
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

# Imputar com mediana — mais robusto para distribuições assimétricas de rede
# fit() SOMENTE no treino — Aula 5: evitar data leakage
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)

print(f"NaN treino após imputação: {X_train.isnull().sum().sum()}")
print(f"NaN teste após imputação:  {X_test.isnull().sum().sum()}")
```

---

### 7.6 Engenharia de Features Comportamentais (somente no treino → transform no teste)

```python
def computar_features_comportamentais(df):
    """
    Computa as 6 novas features baseadas em comportamento de fluxo.
    Substitui o TTL como mecanismo de distinção Externo/Interno.
    
    Ver Seção 4.2 para justificativa de cada feature.
    
    IMPORTANTE: ajustar nomes das colunas conforme resultado da EDA (Seção 5.3).
    """
    df = df.copy()
    
    # Ajustar estes nomes conforme o que foi encontrado na EDA:
    fwd_pkts = 'Total Fwd Packets'
    bwd_pkts = 'Total Backward Packets'
    fwd_byts = 'Total Length of Fwd Packets'
    bwd_byts = 'Total Length of Bwd Packets'
    duration = 'Flow Duration'
    pkt_std  = 'Packet Length Std'
    fwd_act  = 'Fwd Act Data Pkts'
    
    # Feature 1: Assimetria de pacotes
    df['asymmetry_pkts'] = (
        df[fwd_pkts] / (df[fwd_pkts] + df[bwd_pkts] + 1e-9)
    )
    
    # Feature 2: Assimetria de bytes
    df['asymmetry_bytes'] = (
        df[fwd_byts] / (df[fwd_byts] + df[bwd_byts] + 1e-9)
    )
    
    # Feature 3: Taxa de pacotes total
    df['pkt_rate'] = (
        (df[fwd_pkts] + df[bwd_pkts]) / (df[duration] + 1e-9)
    )
    
    # Feature 4: Uniformidade dos pacotes
    # 1/(Std+1): Std≈0 → uniformidade≈1 (script); Std grande → uniformidade baixa
    df['pkt_uniformity'] = 1.0 / (df[pkt_std] + 1.0)
    
    # Feature 5: Duração logarítmica (Aula 5: log1p para atributos de cauda longa)
    df['log_duration'] = np.log1p(df[duration])
    
    # Feature 6: Fração de pacotes com dados reais forward
    if fwd_act in df.columns:
        df['fwd_active_ratio'] = df[fwd_act] / (df[fwd_pkts] + 1e-9)
    
    return df

X_train = computar_features_comportamentais(X_train)
X_test  = computar_features_comportamentais(X_test)

novas_features = [
    'asymmetry_pkts', 'asymmetry_bytes',
    'pkt_rate', 'pkt_uniformity',
    'log_duration', 'fwd_active_ratio'
]
novas_features = [f for f in novas_features if f in X_train.columns]

print(f"Novas features computadas: {novas_features}")
print("\nEstatísticas das novas features no treino:")
print(X_train[novas_features].describe().round(4))
```

---

### 7.7 Seleção de Features com VarianceThreshold (somente no treino)

```python
# Aula 5: variância zero = ruído puro — remover antes de qualquer modelo
# fit() SOMENTE no treino
vt = VarianceThreshold(threshold=0.01)
X_train_vt = pd.DataFrame(
    vt.fit_transform(X_train),
    columns=X_train.columns[vt.get_support()]
)
X_test_vt = pd.DataFrame(
    vt.transform(X_test),
    columns=X_train.columns[vt.get_support()]
)

removidas = set(X_train.columns) - set(X_train_vt.columns)
print(f"Features removidas (variância ≈ 0): {len(removidas)}")
print(f"Features restantes: {X_train_vt.shape[1]}")

print("\nStatus das novas features:")
for f in novas_features:
    status = '✓ mantida' if f in X_train_vt.columns else '✗ removida (variância ≈ 0)'
    print(f"  {f}: {status}")
```

---

### 7.8 Balanceamento com SMOTE (somente no treino)

```python
# Aula 5: SMOTE somente no treino, NUNCA no teste
# imbalanced-learn suporta 3+ classes nativamente
# O teste permanece com distribuição real

print("Distribuição ANTES do SMOTE:")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train_vt, y_train)

print("\nDistribuição APÓS o SMOTE:")
print(pd.Series(y_train_bal).value_counts())
print(f"\nShape treino balanceado: {X_train_bal.shape}")
```

---

### 7.9 Treinamento e Validação Cruzada (somente no treino)

```python
# StratifiedKFold — Aula 7: preserva proporção das 3 classes em cada fold
# 10 folds conforme enunciado do projeto
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# ── Modelo 1: Random Forest ──────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',   # Aula 7: compensar desbalanceamento
    random_state=SEED,
    n_jobs=-1
)

print("Validando Random Forest (10-fold CV no treino)...")
scores_rf = cross_val_score(
    rf, X_train_bal, y_train_bal,
    cv=cv,
    scoring='f1_macro',    # F1 macro: todas as classes pesam igual
    n_jobs=-1
)
print(f"RF — F1 Macro CV: {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")

# ── Modelo 2: MLP como comparativo ───────────────────────────────────────
# Pipeline obrigatório para evitar leakage do scaler — Aula 5
mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=SEED
    ))
])

print("\nValidando MLP (10-fold CV no treino)...")
scores_mlp = cross_val_score(
    mlp_pipe, X_train_bal, y_train_bal,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)
print(f"MLP — F1 Macro CV: {scores_mlp.mean():.4f} ± {scores_mlp.std():.4f}")

# Comparação
melhor = 'Random Forest' if scores_rf.mean() > scores_mlp.mean() else 'MLP'
print(f"\nMelhor modelo na CV: {melhor}")
```

---

### 7.10 Hyperparameter Tuning (no treino, usando CV interno)

```python
# RandomizedSearchCV — Aula prática: mais eficiente que GridSearch
# para espaços de busca grandes
# Test set NÃO é tocado aqui

param_dist_rf = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_tuning = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
    param_distributions=param_dist_rf,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring='f1_macro',
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

print("Iniciando tuning do Random Forest...")
rf_tuning.fit(X_train_bal, y_train_bal)

print(f"\nMelhores hiperparâmetros:")
print(rf_tuning.best_params_)
print(f"Melhor F1 Macro (CV): {rf_tuning.best_score_:.4f}")

rf_best = rf_tuning.best_estimator_
```

---

### 7.11 Avaliação Final no Test Set

> **Regra absoluta das aulas**: o test set é usado UMA ÚNICA VEZ, apenas aqui.

```python
nomes_classes = ['Benigno', 'Externo', 'Zumbi Interno']

# Treinar modelo final com todos os dados de treino
rf_best.fit(X_train_bal, y_train_bal)
mlp_pipe.fit(X_train_bal, y_train_bal)

for nome, modelo, X_te in [
    ('Random Forest (otimizado)', rf_best, X_test_vt),
    ('MLP',                       mlp_pipe, X_test_vt),
]:
    y_pred = modelo.predict(X_te)
    
    print(f"\n{'='*60}")
    print(f"  {nome}")
    print(f"{'='*60}")
    print(f"  F1 Macro:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"  G-Mean:    {geometric_mean_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=nomes_classes
    ))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=nomes_classes
    ).plot(ax=ax, colorbar=False, cmap='Blues', values_format='.0f')
    ax.set_title(f'Matriz de Confusão — {nome}')
    plt.tight_layout()
    plt.show()
```

---

### 7.12 Importância das Features (validação da abordagem)

```python
# Feature importances do Random Forest — Aula 8
# Esta análise responde: as novas features comportamentais realmente importam?

rf_best.fit(X_train_bal, y_train_bal)

df_imp = pd.DataFrame({
    'feature':    X_train_vt.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False).head(25)

df_imp['nova_feature'] = df_imp['feature'].isin(novas_features)

fig, ax = plt.subplots(figsize=(10, 8))
cores_barras = ['darkorange' if n else 'steelblue'
                for n in df_imp['nova_feature'][::-1]]
ax.barh(df_imp['feature'][::-1], df_imp['importance'][::-1], color=cores_barras)
ax.set_xlabel('Importância (redução de impureza)')
ax.set_title('Top 25 Features — Random Forest\nLaranja = novas features comportamentais')

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='darkorange', label='Feature comportamental computada'),
    Patch(color='steelblue',  label='Feature original do CICDDoS2019')
])
plt.tight_layout()
plt.show()

print("\nTop 10 features e suas importâncias:")
for _, row in df_imp.head(10).iterrows():
    marker = ' ← NOVA FEATURE' if row['nova_feature'] else ''
    print(f"  {row['feature']:<35} {row['importance']:.5f}{marker}")
```

**[RESULTADO A PREENCHER E INTERPRETAR]:**

Duas situações possíveis e o que fazer em cada uma:

**Situação A — Novas features aparecem no top 10:**
> As features comportamentais capturaram sinal real. A abordagem é válida e pode ser
> declarada com confiança no paper. A distinção Externo/Interno tem base em padrão
> de comportamento genuíno, não apenas na heurística de labeling.

**Situação B — Novas features têm importância próxima de zero:**
> O modelo está usando as features originais do CICDDoS2019 para fazer a distinção.
> Isso indica que a heurística de labeling foi reproduzida através de features
> correlacionadas — o modelo não generalizaria para padrões não vistos.
> Declarar como limitação e reportar os resultados da Abordagem 2 (avaliação
> sem as features da heurística — Seção 4 do plano de mitigação anterior).

---

### 7.13 Salvar Artefatos

```python
# Aula 5 + práticas: salvar TODOS os transformadores
# Em produção, dados novos passam pelas mesmas transformações
os.makedirs('models_triclass/', exist_ok=True)

joblib.dump(rf_best,                         'models_triclass/rf_triclass.joblib')
joblib.dump(mlp_pipe,                        'models_triclass/mlp_triclass.joblib')
joblib.dump(imputer,                         'models_triclass/imputer.joblib')
joblib.dump(vt,                              'models_triclass/variance_filter.joblib')
joblib.dump(X_train_vt.columns.tolist(),     'models_triclass/selected_features.joblib')
joblib.dump(novas_features,                  'models_triclass/computed_features.joblib')

print("Artefatos salvos em models_triclass/:")
print("  ├── rf_triclass.joblib           (Random Forest otimizado)")
print("  ├── mlp_triclass.joblib          (MLP comparativo)")
print("  ├── imputer.joblib               (SimpleImputer mediana)")
print("  ├── variance_filter.joblib       (VarianceThreshold)")
print("  ├── selected_features.joblib     (lista de features selecionadas)")
print("  └── computed_features.joblib     (nomes das features comportamentais)")
```

---

### 7.14 Inferência em produção

```python
def detectar_classe_ataque(X_fluxo: pd.DataFrame) -> dict:
    """
    Classifica um novo fluxo de rede nas 3 classes.
    
    Args:
        X_fluxo: DataFrame com as features brutas do fluxo de rede.
    
    Returns:
        dict com 'class', 'label', 'confidence', 'action'
    """
    rf         = joblib.load('models_triclass/rf_triclass.joblib')
    imputer    = joblib.load('models_triclass/imputer.joblib')
    vt         = joblib.load('models_triclass/variance_filter.joblib')
    feat_names = joblib.load('models_triclass/selected_features.joblib')
    
    # 1. Tratar Inf
    X = X_fluxo.replace([np.inf, -np.inf], np.nan)
    
    # 2. Computar features comportamentais
    X = computar_features_comportamentais(X)
    
    # 3. Imputar
    X = pd.DataFrame(imputer.transform(X), columns=X.columns)
    
    # 4. Filtro de variância
    X = pd.DataFrame(vt.transform(X), columns=X.columns[vt.get_support()])
    
    # 5. Selecionar features corretas
    X = X[feat_names]
    
    # 6. Predição
    classe    = rf.predict(X)[0]
    proba     = rf.predict_proba(X)[0]
    confianca = proba[classe]
    
    mapa = {
        0: {'label': 'Benigno',       'action': 'none'},
        1: {'label': 'Ataque Externo','action': 'block_global'},
        2: {'label': 'Zumbi Interno', 'action': 'isolate_surgical'},
    }
    
    return {
        'class':      int(classe),
        'label':      mapa[classe]['label'],
        'confidence': float(confianca),
        'action':     mapa[classe]['action'],
        'proba':      {mapa[i]['label']: float(p) for i, p in enumerate(proba)}
    }
```

---

## 8. Integração com o Sistema Existente

### 8.1 Relação com o sistema binário atual

O sistema triclasse não substitui o MLP binário — ele o **estende**:

```
Estágio 1 (existente): MLP binário rápido
    → Se Benigno: liberar fluxo, fim.
    → Se Ataque: passar para o Estágio 2.

Estágio 2 (novo): Random Forest triclasse
    → Classe 1 (Externo): POST /manage/ip → block global
    → Classe 2 (Interno): POST /mitigation/isolate/{ip} → isolamento cirúrgico
```

Essa arquitetura em dois estágios tem vantagem computacional: o MLP binário é mais leve e rápido, triando a grande maioria do tráfego legítimo antes de invocar o RF triclasse.

### 8.2 Como integrar ao DDoSPredictor existente

```python
class DDoSPredictorV2:
    """
    Preditor triclasse — extensão do sistema binário existente.
    """
    
    def __init__(self, models_dir_binary='models/', 
                 models_dir_triclass='models_triclass/'):
        # Estágio 1: modelo binário existente
        self.mlp_binary = joblib.load(f"{models_dir_binary}/mlp_model.joblib")
        
        # Estágio 2: novo classificador triclasse
        self.rf_triclass = joblib.load(f"{models_dir_triclass}/rf_triclass.joblib")
        self.imputer     = joblib.load(f"{models_dir_triclass}/imputer.joblib")
        self.vt          = joblib.load(f"{models_dir_triclass}/variance_filter.joblib")
        self.feat_names  = joblib.load(f"{models_dir_triclass}/selected_features.joblib")
    
    def predict(self, X_fluxo: pd.DataFrame, 
                is_known_host: bool = False) -> dict:
        """
        Classificação em dois estágios.
        
        Args:
            X_fluxo:        features do fluxo de rede
            is_known_host:  True se o IP está registrado no state.ip_to_mac do SDN
        """
        # Estágio 1: binário
        pred_binary = self.mlp_binary.predict(X_fluxo)[0]
        
        if pred_binary == 0:
            return {'class': 0, 'label': 'Benigno', 'action': 'none'}
        
        # Estágio 2: triclasse (só chega aqui se for ataque)
        resultado = detectar_classe_ataque(X_fluxo)
        
        # Reforçar com informação do SDN (is_known_host é determinístico)
        if resultado['class'] == 1 and is_known_host:
            # IP registrado no SDN não pode ser externo legítimo
            # Reclassificar como interno com alta confiança
            resultado['class']      = 2
            resultado['label']      = 'Zumbi Interno'
            resultado['action']     = 'isolate_surgical'
            resultado['confidence'] = 0.95
            resultado['reason']     = 'Reclassificado: IP registrado no SDN → interno'
        
        return resultado
```

---

## 9. Limitações a Declarar

Todo paper honesto declara suas limitações. As desta implementação:

**Limitação 1 — Heurística de labeling (a mais importante)**
> "As métricas de F1 para as classes Externo e Interno refletem a capacidade do
> modelo de reproduzir a heurística de classificação por tipo de ataque, não
> necessariamente a distinção real entre ataques de origem externa e hosts
> internos comprometidos. O CICDDoS2019 não contém metadados de topologia
> que permitiriam uma rotulagem ground-truth das duas origens."

**Limitação 2 — Ausência de TTL real**
> "A abordagem HCF original (Jin et al., 2003) utiliza TTL por pacote para
> estimar o número de saltos de roteador. Esta implementação usa features
> comportamentais de fluxo como proxy, por ausência de datasets SDN+DDoS
> públicos com TTL real disponível. A validação com TTL capturado diretamente
> via Packet-In do controlador OpenDaylight é deixada como trabalho futuro."

**Limitação 3 — Dataset não SDN-específico**
> "O CICDDoS2019 foi gerado em ambiente de rede convencional, sem controlador
> SDN. As features específicas de SDN (packet-in rate, flow table usage) não
> estão presentes, o que pode limitar a generalização para ambientes SDN reais."

**Mitigação das limitações:**
- Avaliar em dois níveis separados (Benigno vs. Ataque com métricas confiáveis; Externo vs. Interno como proxy comportamental)
- Remover as features da heurística de labeling e verificar se o modelo ainda distingue as classes (Seção 7.12 — Situação A vs. B)
- Reportar `is_known_host` do controlador SDN como o sinal mais confiável para distinção interna/externa em produção

---

## 10. Referências

- **CICDDoS2019**: Sharafaldin, I. et al. (2019). "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy." ICCST 2019.
- **HCF**: Jin, C. et al. (2003). "Hop-count filtering: an effective defense against spoofed DDoS traffic." ACM CCS 2003.
- **Entropia em SDN**: Tian, Q. & Miyata, S. (2023). "A DDoS Attack Detection Method Using Conditional Entropy Based on SDN Traffic." *IoT*, 4(2), 95-111.
- **Assimetria de fluxo**: Santos-Neto et al. (2024). "DDoS attack detection in SDN: Enhancing entropy-based detection with machine learning." *Concurrency and Computation: Practice and Experience*.
- **Random Forest**: Breiman, L. (2001). "Random Forests." *Machine Learning*, 45, 5-32.
- **SMOTE**: Chawla, N.V. et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*.
- **MLP**: Mehmood et al. (2025). "DDoS detection in SDN using Optimizer-equipped CNN-MLP." *PLoS ONE* 20(1): e0312425.
- **InSDN**: ElSayed, M. et al. (2021). "InSDN: A Novel SDN Intrusion Detection Dataset." *IEEE Access*.
- **UNSW-NB15**: Moustafa, N. & Slay, J. (2015). "UNSW-NB15: a comprehensive dataset for NIDS." *MilCIS 2015*.
