# Plano de Implementação: Detecção Triclasse de DDoS em SDN
## Benigno | Ataque Externo | Zumbi Interno (PC comprometido)

> **Versão**: 4.0 — Dataset único: InSDN (Elsayed et al., IEEE Access, 2020)  
> **CICDDoS2019**: descartado (BENIGN irrisório, 22GB inviáveis, domain mismatch)  
> **Modelo principal**: Random Forest com comparativo MLP  
> **Contexto**: Extensão do sistema binário MLP existente

---

## Índice

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [Estrutura Real do InSDN](#2-estrutura-real-do-insdn)
3. [As Três Classes — Definição Revisada](#3-as-três-classes--definição-revisada)
4. [O Problema Central: Classe 2 tem 164 amostras](#4-o-problema-central-classe-2-tem-164-amostras)
5. [As Heurísticas Adotadas](#5-as-heurísticas-adotadas)
6. [Por que Random Forest](#6-por-que-random-forest)
7. [Análise Exploratória do InSDN](#7-análise-exploratória-do-insdn)
8. [Pipeline Completo](#8-pipeline-completo)
9. [Integração com o Sistema Existente](#9-integração-com-o-sistema-existente)
10. [Limitações a Declarar](#10-limitações-a-declarar)
11. [Referências](#11-referências)

---

## 1. Contexto e Motivação

### 1.1 Por que o CICDDoS2019 foi descartado

Três problemas simultâneos tornaram o CICDDoS2019 inviável:

**Volume incompatível:** 22GB de dados com arquivo TFTP.csv de 20M linhas sem nenhum registro BENIGN.

**BENIGN irrisório:** o tráfego legítimo estava diluído nos arquivos de ataque em proporções de 0% a 4%. Equilibrar a Classe 0 exigiria gerar 95k+ amostras sintéticas via SMOTE a partir de apenas ~5k reais — fabricar a classe majoritária, não balanceá-la.

**Domain mismatch:** combinar o BENIGN do InSDN (Mininet/OpenFlow, hosts virtuais) com ataques do CICDDoS2019 (laboratório físico, 25 usuários reais em Windows) criaria duas distribuições de features distintas para a mesma classe — o modelo aprenderia a separar ambientes, não tipos de tráfego.

### 1.2 Por que o InSDN permanece como dataset principal

O InSDN atende simultaneamente a três requisitos não encontrados em conjunto em nenhum outro dataset público:

- **SDN-nativo:** gerado em Mininet com controlador OpenFlow, mesma plataforma do sistema em produção.
- **BENIGN isolado e abundante:** `Normal_data.csv` com 68.424 amostras limpas — arquivo separado, sem mistura com ataques.
- **Label BOTNET com semântica real:** 164 fluxos representando hosts internos comprometidos comunicando com servidor C2 externo — correspondência direta com a Classe 2 do plano.

O sistema binário atual já foi treinado no InSDN. A extensão triclasse preserva consistência de domínio.

---

## 2. Estrutura Real do InSDN

| Arquivo | Linhas | Labels presentes | Uso no plano |
|---|---|---|---|
| `Normal_data.csv` | 68.424 | Só `Normal` | Classe 0 completa |
| `OVS.csv` | 138.722 | DDoS, DoS, Probe, BFA, Web-Attack, BOTNET | Classes 1 e 2 |
| `metasploitable-2.csv` | 136.743 | DDoS, Probe, DoS, BFA, U2R | Classe 1 (DDoS) |

### 2.1 O que cada label representa

**DDoS** (OVS.csv e metasploitable): ataques distribuídos lançados pelos hosts h1 e h2 com Hping3. `Pkt Len Std` mediana = 0, `Flow Duration` mediana de 1–19 µs — rajadas sintéticas uniformíssimas. 99.4% disparam o critério de burst.

**DoS** (OVS.csv e metasploitable): ataque de origem única. Comportamento de fluxo mais variado — duração maior, nem sempre burst uniforme.

**BOTNET** (OVS.csv, 164 amostras): hosts internos `192.168.20.131` e `192.168.20.132` comunicando com servidor C2 externo `200.175.2.130:8081`. Padrão de heartbeat/beacon:
- Conexões TCP periódicas, 4 pacotes com SYN
- Servidor C2 responde com 1 ACK apenas
- Sem FIN flags (conexões encerradas por RST/timeout)
- `Fwd Act Data Pkts` = 0 ou 1
- Duração média ~31ms — muito acima do threshold de burst (500µs)
- Hosts internos registrados no SDN (`state.ip_to_mac`)

**Probe, BFA, Web-Attack, U2R:** fora do escopo. Descartados.

### 2.2 Diferença de nomes de colunas — ajuste obrigatório

O InSDN usa nomenclatura CICFlowMeter diferente. A renomeação é feita antes de qualquer operação:

```python
RENAME_MAP = {
    'Tot Fwd Pkts':    'Total Fwd Packets',
    'Tot Bwd Pkts':    'Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
    'Pkt Len Std':     'Packet Length Std',
    'Fwd Pkts/s':      'Fwd Packets/s',
    'Bwd Pkts/s':      'Bwd Packets/s',
}
```

> Verificar os nomes reais com `df.columns.tolist()` e ajustar conforme
> a versão dos arquivos baixados.

---

## 3. As Três Classes — Definição Revisada

| Classe | Label | Base no dataset | Qualidade do label |
|---|---|---|---|
| 0 — Benigno | `Normal` | `Normal_data.csv` completo | ✓ Ground truth real |
| 1 — Externo | `DDoS` com burst | OVS + metasploitable | ✓ Comportamento claro e confirmável |
| 2 — Zumbi Interno | `BOTNET` + `DoS` sem burst | OVS | ⚠️ BOTNET real (164); DoS é proxy |

### 3.1 Por que manter a Classe 2 com 164 amostras

O label `BOTNET` é o único label semântico de tráfego de host comprometido encontrado em toda a pesquisa de datasets públicos SDN. Suas 164 amostras têm qualidade superior a qualquer heurística comportamental — representam comunicação C2 real capturada no testbed. Descartá-las por volume seria descartá-la como evidência direta.

A estratégia é usar BOTNET como **âncora semântica** e complementar com `DoS sem burst` como proxy, sendo explícito sobre essa distinção nas métricas e nas limitações.

---

## 4. O Problema Central: Classe 2 tem 164 amostras

164 amostras são insuficientes para treinar um classificador robusto. Três opções foram avaliadas:

### 4.1 Opção A — Classificação binária em dois estágios (adotada para ML)

```
Estágio 1: MLP binário existente → Benigno | Ataque
Estágio 2: RF binário novo       → Externo | Interno
```

Cada estágio é um problema binário bem definido. O Estágio 2 usa DDoS+burst vs. BOTNET+DoS sem burst, com volume razoável após incluir o proxy comportamental.

### 4.2 Opção B — Triclasse com SMOTE agressivo (implementada para comparação)

Inclui BOTNET + DoS sem burst na Classe 2 e aplica SMOTE. Com 164 amostras reais, o SMOTE gera interpolações entre pontos próximos — risco de região artificial no espaço de features. Implementada com ressalva explícita.

### 4.3 Opção C — Regra determinística pós-predição (adotada para produção)

```
Ataque predito + IP registrado no SDN → Classe 2 (Interno)
Ataque predito + IP não registrado    → Classe 1 (Externo)
```

Não treina a distinção — determina por lógica de negócio. É a mais honesta com os dados disponíveis e usa informação real e confiável do controlador SDN.

### 4.4 Decisão: implementar A + C em paralelo

- **A** para métricas de ML comparáveis com a literatura.
- **C** como mecanismo de produção, mais confiável por usar informação real do SDN.

---

## 5. As Heurísticas Adotadas

### 5.1 Heurística de burst

```python
def is_burst(df):
    """
    Identifica rajada sintética uniforme — padrão de ferramenta de ataque.
    
    Pkt Len Std <= 1.0  → pacotes com tamanho praticamente idêntico
    Flow Duration < 500 → fluxo com duração < 500 µs (extremamente curto)
    
    BOTNET tem Flow Duration ~31ms = 31.000 µs → is_burst() == False (correto)
    DDoS Hping3 tem duration de 1-19 µs → is_burst() == True (correto)
    """
    return (
        (df['Packet Length Std'] <= 1.0) &
        (df['Flow Duration']     < 500.0)
    )
```

### 5.2 Heurística de labeling triclasse

```python
def criar_label_triclasse_insdn(df):
    """
    Converte labels do InSDN em 3 classes.

    Classe 0 — Benigno:
        Label == 'Normal'

    Classe 1 — Ataque Externo:
        Label == 'DDoS' AND is_burst() == True
        (Burst confirma ferramenta de ataque externo)

    Classe 2 — Zumbi Interno:
        Label == 'BOTNET'              (ground truth real — âncora semântica)
        Label == 'DoS' AND ~is_burst() (proxy comportamental)

    Descartados (ambíguos ou fora do escopo):
        DDoS sem burst, DoS com burst, Probe, BFA, Web-Attack, U2R
    """
    label = df['Label'].str.strip()
    burst = is_burst(df)
    y = pd.Series(-1, index=df.index)

    y[label == 'Normal']          = 0
    y[(label == 'DDoS') & burst]  = 1
    y[label == 'BOTNET']          = 2
    y[(label == 'DoS') & ~burst]  = 2

    return y
```

### 5.3 Features comportamentais

Seis features derivadas de comportamento de fluxo. Papel aqui: **reforçar** o modelo com sinal adicional, não substituir os labels.

```python
def computar_features_comportamentais(df):
    """
    Feature 1 — asymmetry_pkts:
        fwd / (fwd + bwd). ~1.0 = unidirecional (ataque); ~0.5 = legítimo.

    Feature 2 — asymmetry_bytes:
        Idem para bytes. Amplificação pode inverter (~0) por resposta > requisição.

    Feature 3 — pkt_rate:
        (fwd + bwd) / duration. Alta taxa = ataque volumétrico.

    Feature 4 — pkt_uniformity:
        1 / (Std + 1). Alta = gerado por script (burst); baixa = tráfego variado.

    Feature 5 — log_duration:
        log1p(duration). Normaliza distribuição de cauda longa — Aula 5.

    Feature 6 — fwd_active_ratio:
        Fwd Act Data Pkts / Total Fwd Pkts.
        BOTNET: ~0 (beacon sem dado real).
        DDoS:   ~0 (sem handshake completo).
        Normal: >0.5 (dados reais trafegando).
        DIFERENCIADOR: esta feature separa BOTNET de tráfego normal,
        mas não separa BOTNET de DDoS — ambos têm ratio ~0.
        A separação BOTNET vs DDoS depende de log_duration e pkt_uniformity.
    """
    df = df.copy()

    fwd  = 'Total Fwd Packets'
    bwd  = 'Total Backward Packets'
    fwdb = 'Total Length of Fwd Packets'
    bwdb = 'Total Length of Bwd Packets'
    dur  = 'Flow Duration'
    std  = 'Packet Length Std'
    act  = 'Fwd Act Data Pkts'

    df['asymmetry_pkts']  = df[fwd]  / (df[fwd]  + df[bwd]  + 1e-9)
    df['asymmetry_bytes'] = df[fwdb] / (df[fwdb] + df[bwdb] + 1e-9)
    df['pkt_rate']        = (df[fwd] + df[bwd])  / (df[dur] + 1e-9)
    df['pkt_uniformity']  = 1.0 / (df[std] + 1.0)
    df['log_duration']    = np.log1p(df[dur])

    if act in df.columns:
        df['fwd_active_ratio'] = df[act] / (df[fwd] + 1e-9)

    return df
```

---

## 6. Por que Random Forest

Seguindo as aulas:

**Aula 8:**
> *"Ensemble de DTs — reduz variância por amostragem aleatória de features a cada nó."*
> *"RandomForest = ensemble de DTs reduz variância."*

**Vantagens específicas:**

- **Invariante a escala:** `Flow Duration` (µs) e `asymmetry_pkts` (0–1) têm escalas completamente diferentes. RF não é afetado, ao contrário do MLP que exige `StandardScaler`.
- **`feature_importances_` nativo:** vai revelar se `log_duration` e `pkt_uniformity` realmente diferenciam BOTNET de DDoS.
- **`class_weight='balanced'`:** compensa desbalanceamento sem criar dados adicionais.

**MLP como comparativo:** captura relações não-lineares (Aula 10). Exige `Pipeline(StandardScaler → MLP)` obrigatoriamente para evitar leakage (Aula 5).

**Métricas (Aula 7):**
- F1 Macro, MCC, Geometric Mean
- Recall da Classe 2 é o mais crítico em produção
- NÃO usar: acurácia sozinha, F1-weighted

---

## 7. Análise Exploratória do InSDN

> **⚠️ Executar e preencher os resultados antes de prosseguir.**

### 7.1 Carregamento e verificação de colunas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'data/InSDN/'
normal = pd.read_csv(PATH + 'Normal_data.csv',      low_memory=False)
ovs    = pd.read_csv(PATH + 'OVS.csv',              low_memory=False)
meta   = pd.read_csv(PATH + 'metasploitable-2.csv', low_memory=False)

for nome, df in [('Normal_data', normal), ('OVS', ovs), ('metasploitable', meta)]:
    print(f"\n=== {nome}: {df.shape} ===")
    if 'Label' in df.columns:
        print(df['Label'].value_counts())

# Verificar nomes das colunas necessárias no OVS
COLUNAS_ALVO = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
                'TotLen Bwd Pkts', 'Flow Duration', 'Pkt Len Std',
                'Fwd Act Data Pkts', 'Label']
print("\nVerificação de colunas no OVS.csv:")
for col in COLUNAS_ALVO:
    print(f"  {'✓' if col in ovs.columns else '✗'} {col}")
```

**[RESULTADO A PREENCHER]:**
- Nomes reais das colunas: ___
- Ajustes no RENAME\_MAP: ___

### 7.2 Validar que is_burst separa corretamente DDoS de BOTNET

```python
# Esta é a verificação mais crítica — confirmar que o threshold funciona
data_temp = pd.concat([ovs, meta], ignore_index=True)
data_temp = data_temp.rename(columns={k: v for k, v in RENAME_MAP.items()
                                       if k in data_temp.columns})
data_temp = data_temp[data_temp['Label'].isin(['DDoS', 'BOTNET', 'DoS'])]
data_temp['burst'] = is_burst(data_temp)

print("Distribuição de is_burst por label:")
print(pd.crosstab(data_temp['Label'], data_temp['burst'],
                  margins=True, margins_name='Total'))
```

**[RESULTADO ESPERADO E A CONFIRMAR]:**
```
burst     False    True
Label
BOTNET      164       0   ← BOTNET NUNCA é burst (31ms >> 500µs)
DDoS          ~       ~   ← DDoS deve ser ~99% True
DoS           ~       ~   ← DoS deve ter distribuição mista
```

> ⚠️ Se BOTNET tiver amostras com burst=True, o threshold de 500µs precisa
> ser ajustado para a mediana de Flow Duration do BOTNET observada nos dados.

### 7.3 Distribuição triclasse após aplicar heurística

```python
data_all = pd.concat([normal, ovs, meta], ignore_index=True)
data_all = data_all.rename(columns={k: v for k, v in RENAME_MAP.items()
                                     if k in data_all.columns})
data_all['label_3class'] = criar_label_triclasse_insdn(data_all)

n_total = len(data_all)
n_descartados = (data_all['label_3class'] == -1).sum()
data_valido = data_all[data_all['label_3class'] != -1]

print(f"Descartados (fora do escopo): {n_descartados:,}")
print(f"\nDistribuição triclasse:")
nomes = {0: 'Benigno', 1: 'Externo', 2: 'Zumbi Interno'}
for cls, nome in nomes.items():
    n = (data_valido['label_3class'] == cls).sum()
    pct = 100 * n / len(data_valido)
    if cls == 2:
        n_bot = ((data_valido['Label'] == 'BOTNET') &
                 (data_valido['label_3class'] == 2)).sum()
        n_dos = n - n_bot
        print(f"  Classe {cls} ({nome}): {n:,} ({pct:.1f}%)")
        print(f"    ├── BOTNET (ground truth): {n_bot}")
        print(f"    └── DoS sem burst (proxy): {n_dos:,}")
    else:
        print(f"  Classe {cls} ({nome}): {n:,} ({pct:.1f}%)")
```

**[RESULTADO A PREENCHER]:**
- Classe 0: ___ amostras (___%)
- Classe 1: ___ amostras (___%)
- Classe 2 total: ___ amostras (___%)
  - BOTNET: 164
  - DoS sem burst: ___
- Ratio de desbalanceamento (maior/menor): ___

### 7.4 Separabilidade com PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

feats_pca = ['Flow Duration', 'Packet Length Std',
             'Total Fwd Packets', 'Total Backward Packets']
feats_pca = [f for f in feats_pca if f in data_valido.columns]

amostra = data_valido.groupby('label_3class').apply(
    lambda x: x.sample(n=min(2000, len(x)), random_state=42)
).reset_index(drop=True)

X_pca = amostra[feats_pca].replace([np.inf, -np.inf], np.nan).fillna(0)
X_2d  = PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(X_pca))

cores = {0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}
fig, ax = plt.subplots(figsize=(10, 7))
for cls, nome in {0: 'Benigno', 1: 'Externo', 2: 'Zumbi Interno'}.items():
    mask = amostra['label_3class'] == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=cores[cls], label=f"{nome} (n={mask.sum()})",
               alpha=0.4, s=10)
ax.set_title('PCA 2D — 3 Classes do InSDN')
ax.legend(); plt.tight_layout(); plt.show()
```

**[RESULTADO A PREENCHER]:**
- Classes visualmente separáveis? ___
- Classe 2 (BOTNET+DoS) forma cluster próprio? ___
- Sobreposição Externo/Interno? ___

---

## 8. Pipeline Completo

### 8.0 Importações

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
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
from imblearn.under_sampling import RandomUnderSampler
import joblib

SEED = 42
np.random.seed(SEED)
PATH = 'data/InSDN/'
```

---

### 8.1 Carregamento e renomeação

```python
normal = pd.read_csv(PATH + 'Normal_data.csv',      low_memory=False)
ovs    = pd.read_csv(PATH + 'OVS.csv',              low_memory=False)
meta   = pd.read_csv(PATH + 'metasploitable-2.csv', low_memory=False)

data = pd.concat([normal, ovs, meta], ignore_index=True)

# Renomear colunas para padrão do plano (ajustar conforme EDA Seção 7.1)
data = data.rename(columns={k: v for k, v in RENAME_MAP.items()
                              if k in data.columns})

print(f"Shape total: {data.shape}")
print(f"\nLabels originais:\n{data['Label'].value_counts()}")
```

---

### 8.2 Labeling triclasse

```python
data['label_3class'] = criar_label_triclasse_insdn(data)
n_antes = len(data)
data = data[data['label_3class'] != -1].reset_index(drop=True)
print(f"Descartados: {n_antes - len(data):,}")
print(f"Dataset válido: {len(data):,}")
```

---

### 8.3 ⚠️ SPLIT ANTES DE TUDO — Regra de Ouro (Aula 5)

```python
X = data.drop(columns=['Label', 'label_3class'], errors='ignore')
X = X.select_dtypes(include=np.number)
y = data['label_3class']

# stratify=y OBRIGATÓRIO — Aula 7: preserva proporção das 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

print(f"Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
print("\nProporção treino:")
print(y_train.value_counts(normalize=True).mul(100).round(1))
```

---

### 8.4 Limpeza — somente no treino

```python
# Substituir Inf por NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

# fit() SOMENTE no treino — Aula 5: evitar data leakage
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)
```

---

### 8.5 Features comportamentais

```python
X_train = computar_features_comportamentais(X_train)
X_test  = computar_features_comportamentais(X_test)

novas_features = ['asymmetry_pkts', 'asymmetry_bytes', 'pkt_rate',
                  'pkt_uniformity', 'log_duration', 'fwd_active_ratio']
novas_features = [f for f in novas_features if f in X_train.columns]
```

---

### 8.6 Seleção de features com VarianceThreshold — somente no treino

```python
# Aula 5: variância zero = ruído puro
vt = VarianceThreshold(threshold=0.01)
X_train_vt = pd.DataFrame(
    vt.fit_transform(X_train),
    columns=X_train.columns[vt.get_support()]
)
X_test_vt = pd.DataFrame(
    vt.transform(X_test),
    columns=X_train.columns[vt.get_support()]
)

print(f"Features removidas: {(~vt.get_support()).sum()}")
print(f"Features restantes: {X_train_vt.shape[1]}")
```

---

### 8.7 Balanceamento — SMOTE controlado

```python
# Aula 5: SMOTE somente no treino, NUNCA no teste
# Estratégia conservadora: SMOTE máximo 5x o volume real da Classe 2
# para não fabricar uma classe inteira sinteticamente

print("Distribuição ANTES do balanceamento:")
print(pd.Series(y_train).value_counts())

n_cls0 = (y_train == 0).sum()
n_cls1 = (y_train == 1).sum()
n_cls2 = (y_train == 2).sum()

# Target conservador: Classe 2 vai para min(5x original, tamanho da Classe 0)
target_cls2 = min(n_cls2 * 5, n_cls0)
strategy_over = {2: target_cls2}

smote = SMOTE(sampling_strategy=strategy_over, random_state=SEED)
X_res, y_res = smote.fit_resample(X_train_vt, y_train)

# Se Classe 1 for muito maior que Classe 0, undersample opcional
if n_cls1 > n_cls0 * 2:
    under = RandomUnderSampler(
        sampling_strategy={1: int(n_cls0 * 1.5)}, random_state=SEED
    )
    X_train_bal, y_train_bal = under.fit_resample(X_res, y_res)
else:
    X_train_bal, y_train_bal = X_res, y_res

print("\nDistribuição APÓS balanceamento:")
print(pd.Series(y_train_bal).value_counts())
```

---

### 8.8 Treinamento e Validação Cruzada — somente no treino

```python
# StratifiedKFold 10 folds — Aula 7
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)
scores_rf = cross_val_score(
    rf, X_train_bal, y_train_bal,
    cv=cv, scoring='f1_macro', n_jobs=-1
)
print(f"RF  — F1 Macro CV: {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")

# MLP — Pipeline obrigatório: Aula 5 (StandardScaler dentro do CV)
mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu',
        solver='adam', max_iter=300,
        early_stopping=True, n_iter_no_change=15,
        random_state=SEED
    ))
])
scores_mlp = cross_val_score(
    mlp_pipe, X_train_bal, y_train_bal,
    cv=cv, scoring='f1_macro', n_jobs=-1
)
print(f"MLP — F1 Macro CV: {scores_mlp.mean():.4f} ± {scores_mlp.std():.4f}")
```

---

### 8.9 Hyperparameter Tuning

```python
param_dist_rf = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_tuning = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
    param_distributions=param_dist_rf,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring='f1_macro',
    random_state=SEED, n_jobs=-1, verbose=1
)
rf_tuning.fit(X_train_bal, y_train_bal)

print(f"Melhores parâmetros: {rf_tuning.best_params_}")
print(f"Melhor F1 Macro CV:  {rf_tuning.best_score_:.4f}")
rf_best = rf_tuning.best_estimator_
```

---

### 8.10 Avaliação Final no Test Set

> **Regra absoluta — Aula 7: test set usado UMA ÚNICA VEZ.**

```python
nomes_classes = ['Benigno', 'Externo', 'Zumbi Interno']

rf_best.fit(X_train_bal, y_train_bal)
mlp_pipe.fit(X_train_bal, y_train_bal)

for nome, modelo in [('Random Forest (otimizado)', rf_best),
                     ('MLP', mlp_pipe)]:
    y_pred = modelo.predict(X_test_vt)

    print(f"\n{'='*60}")
    print(f"  {nome}")
    print(f"{'='*60}")
    print(f"  F1 Macro:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"  G-Mean:    {geometric_mean_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=nomes_classes))

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

### 8.11 Validação semântica do BOTNET — verificação mais crítica

```python
# O modelo classifica o BOTNET (ground truth) corretamente como Classe 2?
# Se sim, a abordagem tem fundamento mesmo com 164 amostras.

mask_botnet = (data.loc[X_test.index, 'Label'].str.strip() == 'BOTNET')
n_botnet_test = mask_botnet.sum()

if n_botnet_test > 0:
    X_botnet = X_test_vt[mask_botnet.values]
    y_botnet_pred = rf_best.predict(X_botnet)

    acerto = (y_botnet_pred == 2).sum()
    print(f"\n=== Validação Semântica do BOTNET ===")
    print(f"Amostras BOTNET no teste: {n_botnet_test}")
    print(f"Classificadas como Classe 2: {acerto} ({100*acerto/n_botnet_test:.1f}%)")
    print(f"Classificadas como Classe 1: {(y_botnet_pred == 1).sum()}")
    print(f"Classificadas como Classe 0: {(y_botnet_pred == 0).sum()}")

    if acerto / n_botnet_test >= 0.80:
        print("\n✓ Modelo reconhece padrão de beacon/C2 corretamente.")
        print("  log_duration e pkt_uniformity estão funcionando como diferenciadores.")
    else:
        print("\n⚠ Modelo confunde BOTNET com outra classe.")
        print("  Verificar se log_duration diferencia 31ms de 1-19µs.")
        print("  Possível causa: SMOTE gerou exemplos que se misturaram com DDoS.")
else:
    print("⚠ Nenhum BOTNET no teste — aumentar test_size ou mudar random_state.")
```

---

### 8.12 Importância das features

```python
rf_best.fit(X_train_bal, y_train_bal)

df_imp = pd.DataFrame({
    'feature':    X_train_vt.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False).head(25)

df_imp['nova'] = df_imp['feature'].isin(novas_features)

fig, ax = plt.subplots(figsize=(10, 8))
cores = ['darkorange' if n else 'steelblue' for n in df_imp['nova'][::-1]]
ax.barh(df_imp['feature'][::-1], df_imp['importance'][::-1], color=cores)
ax.set_xlabel('Importância (redução de impureza)')
ax.set_title('Top 25 Features — Random Forest\nLaranja = features comportamentais')

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='darkorange', label='Feature comportamental (nova)'),
    Patch(color='steelblue',  label='Feature original do InSDN')
])
plt.tight_layout()
plt.show()

print("\nTop 15 features:")
for _, row in df_imp.head(15).iterrows():
    marker = ' ← NOVA' if row['nova'] else ''
    print(f"  {row['feature']:<35} {row['importance']:.5f}{marker}")
```

---

### 8.13 Salvar artefatos

```python
# Aula 5 + práticas: salvar TODOS os transformadores junto com o modelo
os.makedirs('models_triclass/', exist_ok=True)

joblib.dump(rf_best,                     'models_triclass/rf_triclass.joblib')
joblib.dump(mlp_pipe,                    'models_triclass/mlp_triclass.joblib')
joblib.dump(imputer,                     'models_triclass/imputer.joblib')
joblib.dump(vt,                          'models_triclass/variance_filter.joblib')
joblib.dump(X_train_vt.columns.tolist(), 'models_triclass/selected_features.joblib')
joblib.dump(novas_features,              'models_triclass/computed_features.joblib')
joblib.dump(RENAME_MAP,                  'models_triclass/rename_map.joblib')

print("Artefatos salvos em models_triclass/:")
print("  ├── rf_triclass.joblib")
print("  ├── mlp_triclass.joblib")
print("  ├── imputer.joblib")
print("  ├── variance_filter.joblib")
print("  ├── selected_features.joblib")
print("  ├── computed_features.joblib")
print("  └── rename_map.joblib")
```

---

## 9. Integração com o Sistema Existente

### 9.1 Arquitetura em dois estágios

```
Estágio 1 — MLP binário (sistema atual):
  → Benigno: liberar. Fim.
  → Ataque:  passar para Estágio 2.

Estágio 2 — Random Forest triclasse (novo):
  → Classe 1 (Externo):  POST /manage/ip → block global
  → Classe 2 (Interno):  POST /mitigation/isolate/{ip} → isolamento cirúrgico

Fallback determinístico (Opção C — produção):
  → is_known_host == True  → forçar Classe 2, independente do RF
  → is_known_host == False → usar predição do RF
```

### 9.2 Código de integração

```python
class DDoSPredictorV2:
    def __init__(self, models_binary='models/', models_triclass='models_triclass/'):
        self.mlp_binary = joblib.load(f"{models_binary}/mlp_model.joblib")
        self.rf          = joblib.load(f"{models_triclass}/rf_triclass.joblib")
        self.imputer     = joblib.load(f"{models_triclass}/imputer.joblib")
        self.vt          = joblib.load(f"{models_triclass}/variance_filter.joblib")
        self.feat_names  = joblib.load(f"{models_triclass}/selected_features.joblib")
        self.rename_map  = joblib.load(f"{models_triclass}/rename_map.joblib")

    def predict(self, X_raw: pd.DataFrame,
                is_known_host: bool = False) -> dict:
        # Estágio 1
        if self.mlp_binary.predict(X_raw)[0] == 0:
            return {'class': 0, 'label': 'Benigno', 'action': 'none'}

        # Estágio 2
        X = X_raw.rename(columns=self.rename_map)
        X = computar_features_comportamentais(X)
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        X = pd.DataFrame(self.vt.transform(X),
                         columns=X.columns[self.vt.get_support()])
        X = X[self.feat_names]

        classe    = int(self.rf.predict(X)[0])
        confianca = float(self.rf.predict_proba(X)[0][classe])

        # Fallback determinístico (Opção C)
        if is_known_host and classe == 1:
            return {
                'class': 2, 'label': 'Zumbi Interno',
                'action': 'isolate_surgical', 'confidence': 0.95,
                'reason': 'IP registrado no SDN — reclassificado como interno'
            }

        mapa = {
            0: ('Benigno',        'none'),
            1: ('Ataque Externo', 'block_global'),
            2: ('Zumbi Interno',  'isolate_surgical'),
        }
        label, action = mapa[classe]
        return {'class': classe, 'label': label,
                'action': action, 'confidence': confianca}
```

---

## 10. Limitações a Declarar

**Limitação 1 — Volume da Classe 2 (a mais importante)**
> "A Classe 2 (Zumbi Interno) é composta por 164 amostras de tráfego BOTNET real
> (heartbeat/beacon para servidor C2) e por fluxos DoS sem padrão de burst, usados
> como proxy comportamental. O SMOTE foi aplicado com fator conservador (máximo 5x
> o volume real). As métricas da Classe 2 devem ser interpretadas com cautela —
> o recall pode estar inflado pelo SMOTE. A validação semântica na Seção 8.11
> fornece evidência mais robusta do que as métricas agregadas."

**Limitação 2 — DoS sem burst como proxy**
> "A heurística que classifica DoS sem burst como Zumbi Interno é uma aproximação
> comportamental. O InSDN não fornece metadados de topologia suficientes para
> confirmar que esses fluxos originaram de hosts internos comprometidos. A única
> evidência semântica real de host comprometido são as 164 amostras BOTNET."

**Limitação 3 — Ausência de TTL real**
> "A distinção externa/interna usa padrão de burst como proxy do Hop Count
> Filtering original (Jin et al., 2003). Nenhum dataset SDN público com TTL bruto
> por pacote foi encontrado. A validação com TTL real via Packet-In do
> OpenDaylight é deixada como trabalho futuro."

**Mitigação das limitações:**
- Seção 8.11 valida semanticamente o BOTNET — se o recall for alto, a abordagem tem fundamento independente do volume.
- O fallback `is_known_host` fornece o mecanismo mais confiável em produção.
- InSDN e sistema em produção compartilham o mesmo domínio (Mininet/OpenFlow) — ausência de domain mismatch.

---

## 11. Referências

- **InSDN**: Elsayed, M. et al. (2020). "InSDN: A Novel SDN Intrusion Detection Dataset." *IEEE Access*, 8, 165263-165284.
- **HCF**: Jin, C. et al. (2003). "Hop-count filtering: an effective defense against spoofed DDoS traffic." *ACM CCS 2003*.
- **Random Forest**: Breiman, L. (2001). "Random Forests." *Machine Learning*, 45, 5-32.
- **SMOTE**: Chawla, N.V. et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*.
- **MLP / Mehmood**: Mehmood et al. (2025). "DDoS detection in SDN using Optimizer-equipped CNN-MLP." *PLoS ONE* 20(1): e0312425.
- **Entropia em SDN**: Tian, Q. & Miyata, S. (2023). "A DDoS Attack Detection Method Using Conditional Entropy Based on SDN Traffic." *IoT*, 4(2), 95-111.
- **Assimetria de fluxo**: Santos-Neto et al. (2024). "DDoS attack detection in SDN: Enhancing entropy-based detection with machine learning." *Concurrency and Computation*.
