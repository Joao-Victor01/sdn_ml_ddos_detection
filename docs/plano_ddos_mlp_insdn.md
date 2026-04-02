# Plano de Implementação: Detecção de DDoS em SDN com MLP
### Replicação de Mehmood et al. (PLoS ONE, 2025) — Variante MLP com InSDN

> **Contexto:** Você já possui o ambiente SDN com Open vSwitch (OVS) e Docker funcionando,
> com os cenários C1 (com SDN + health score), C2 (sem SDN) e C3 (rerouting) estabelecidos.
> Este plano adiciona uma camada de detecção de DDoS usando MLP treinado no dataset InSDN,
> seguindo rigorosamente as boas práticas do curso (Thaís Gaudencio, UFPB/LUMO).

---

## Visão Geral do Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Configurações e Dependências                                    │
│  2. Obtenção do Dataset InSDN                                       │
│  3. EDA (sobre o dataset completo, SEM transformar)                 │
│  4. ⚠️  SPLIT treino/teste (ANTES de qualquer transformação)  ⚠️   │
│  5. Limpeza e Preparação (somente no treino)                        │
│  6. Seleção de Features com SHAP                                    │
│  7. Escalonamento (fit só no treino)                                │
│  8. Balanceamento de Classes (só no treino)                         │
│  9. Treinamento do MLP Baseline                                     │
│ 10. Avaliação com Métricas Completas                                │
│ 11. Melhoria: Hyperparameter Tuning                                 │
│ 12. Avaliação Final no Test Set                                     │
│ 13. Salvando o Modelo                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Etapa 1 — Configurações e Dependências

### 1.1 Instalação dos pacotes

```bash
pip install pandas numpy scikit-learn matplotlib seaborn \
            imbalanced-learn shap joblib ydata-profiling \
            scikit-optimize
```

### 1.2 Imports e configurações iniciais

```python
# ── Imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Sklearn — pré-processamento
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Sklearn — métricas
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay, roc_auc_score,
                              matthews_corrcoef, RocCurveDisplay)
from imblearn.metrics import geometric_mean_score

# Balanceamento
from imblearn.over_sampling import SMOTE

# ── Configurações ─────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.30       # 70/30 conforme padrão do curso
TARGET_COL   = 'Label'    # nome da coluna alvo no InSDN

pd.set_option('display.max_columns', 7000)
pd.set_option('display.max_rows', 90000)

print("Ambiente configurado com sucesso.")
```

---

## Etapa 2 — Obtenção do Dataset InSDN

### 2.1 Sobre o InSDN

O InSDN foi construído especificamente para ambientes SDN usando quatro máquinas
virtuais. Contém **361.317 instâncias** e **84 features**, divididas em:
- **292.893** instâncias de tráfego de ataque (~81%)
- **68.424** instâncias de tráfego benigno (~19%)

O dataset pode ser obtido em:
- Repositório original: `https://github.com/ElsayedMustafa/InSDN`
- Artigo original: Elsayed et al., *IEEE Access*, 2020 (doi: 10.1109/ACCESS.2020.3025534)

O dataset geralmente vem em múltiplos arquivos CSV (Normal.csv, OFswitch.csv, etc.)
que precisam ser concatenados.

### 2.2 Carregamento e concatenação

```python
import os
import glob

# Caminho onde os arquivos CSV do InSDN foram salvos
DATA_PATH = "data/InSDN/"

# Carregar e concatenar todos os arquivos CSV
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
df_list = []

for filepath in all_files:
    df_tmp = pd.read_csv(filepath, low_memory=False)
    nome_arquivo = os.path.basename(filepath).replace('.csv', '')
    print(f"  {nome_arquivo}: {df_tmp.shape}")
    df_list.append(df_tmp)

data = pd.concat(df_list, ignore_index=True)
print(f"\nDataset completo: {data.shape}")
print(f"Colunas: {list(data.columns)}")
```

### 2.3 Verificação inicial

```python
data.head()
data.info()
data.describe()
```

---

## Etapa 3 — EDA (Análise Exploratória dos Dados)

> ⚠️ **Regra do curso:** EDA é feita sobre o dataset completo, mas as
> TRANSFORMAÇÕES só acontecem APÓS o split. Aqui apenas observamos.

### 3.1 Overview rápido com ydata-profiling

```python
# Gera relatório completo automaticamente
# Útil para identificar: missing values, correlações, distribuições, outliers
from ydata_profiling import ProfileReport

profile = ProfileReport(data, minimal=True)  # minimal=True para datasets grandes
profile.to_notebook_iframe()
# Ou salvar em HTML:
# profile.to_file("eda_insdn_report.html")
```

### 3.2 Análise do atributo alvo (classes)

```python
# Verificar os valores únicos da coluna alvo
print("Valores únicos da coluna alvo:")
print(data[TARGET_COL].value_counts())
print()
print(data[TARGET_COL].value_counts(normalize=True).mul(100).round(2))

# Visualizar distribuição
plt.figure(figsize=(8, 4))
sns.countplot(x=data[TARGET_COL], palette='Set2')
plt.title("Distribuição das Classes — InSDN")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()
```

### 3.3 Medir balanceamento com Entropia de Shannon

```python
# Retorna 0 (totalmente desbalanceado) a 1 (perfeitamente balanceado)
def balance_score(df, target, n_classes):
    from numpy import log
    n = len(df)
    H = 0
    for classe in df[target].unique():
        p = len(df[df[target] == classe]) / n
        if p > 0:
            H += p * log(p)
    return (-H) / log(n_classes)

n_classes = data[TARGET_COL].nunique()
bal = balance_score(data, TARGET_COL, n_classes)
print(f"Entropia de Shannon (balanceamento): {bal:.4f}")
print(f"  → 0 = totalmente desbalanceado | 1 = perfeitamente balanceado")
```

### 3.4 Análise de distribuição das features numéricas

```python
# Histogramas para entender assimetria
numeric_cols = data.select_dtypes(include='number').columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

# Amostragem para visualização (dataset grande)
sample = data[numeric_cols].sample(n=min(5000, len(data)), random_state=RANDOM_STATE)

fig, ax = plt.subplots(1, 1, figsize=(16, 16))
sample.hist(ax=ax, bins=30)
plt.suptitle("Distribuição das Features Numéricas — InSDN", y=1.02)
plt.tight_layout()
plt.show()
```

### 3.5 Análise de valores ausentes e duplicados

```python
# Valores ausentes
print("Valores ausentes por coluna:")
missing = data.isnull().sum()
print(missing[missing > 0])
print(f"\nTotal de missing: {data.isnull().sum().sum()}")

# Duplicados
print(f"\nRegistros duplicados: {data.duplicated().sum()}")

# Infinitos (comum em datasets de rede — divisões por zero em features calculadas)
inf_count = np.isinf(data.select_dtypes(include=np.number)).sum().sum()
print(f"Valores infinitos: {inf_count}")
```

### 3.6 Correlação entre features (mapa de calor — amostra)

```python
# Com 84 features, o heatmap completo é ilegível — usar amostra
top_features_for_eda = numeric_cols[:20]  # primeiras 20 para visualização

plt.figure(figsize=(14, 10))
sns.heatmap(
    data[top_features_for_eda].corr(),
    annot=False,
    cmap='RdBu_r',
    center=0,
    linewidths=0.5
)
plt.title("Correlação entre Features (primeiras 20) — InSDN")
plt.tight_layout()
plt.show()
```

---

## Etapa 4 — A Regra de Ouro: Split Antes de Tudo

> ⚠️ **Este é o passo mais crítico do pipeline.**
> Nenhuma transformação — normalização, imputação, encoding, SMOTE,
> remoção de outliers — pode acontecer antes deste split.
> Fazer antes constitui **vazamento de dados** e invalida o experimento.

### 4.1 Preparação da coluna alvo

```python
# O InSDN pode ter a coluna alvo como string ('Benign', 'Attack', etc.)
# ou como múltiplas classes de ataque. Verificar:
print(data[TARGET_COL].unique())

# Para classificação binária (Benign vs. Attack):
# Se houver múltiplas categorias de ataque, unificar em 'Attack'
data['label_binary'] = data[TARGET_COL].apply(
    lambda x: 0 if str(x).strip().lower() in ['benign', 'normal'] else 1
)

print("\nDistribuição binária:")
print(data['label_binary'].value_counts())
print(f"  0 = Benigno | 1 = Ataque")

TARGET_BINARY = 'label_binary'
```

### 4.2 Separar X e y ANTES do split

```python
# Remover a coluna alvo original e a binária dos features
cols_to_drop = [TARGET_COL, TARGET_BINARY]

X = data.drop(columns=cols_to_drop, errors='ignore')
y = data[TARGET_BINARY]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Proporção das classes:\n{y.value_counts(normalize=True).mul(100).round(2)}")
```

### 4.3 Split estratificado 70/30

```python
# stratify=y garante que treino e teste tenham a mesma proporção de classes
# SEMPRE definir random_state para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=y          # OBRIGATÓRIO para classificação
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"\nDistribuição treino:\n{y_train.value_counts(normalize=True).mul(100).round(2)}")
print(f"\nDistribuição teste:\n{y_test.value_counts(normalize=True).mul(100).round(2)}")

# As proporções devem ser iguais — confirmar que stratify funcionou
```

---

## Etapa 5 — Limpeza e Preparação

> A partir daqui, TUDO é feito sobre X_train.
> X_test só recebe .transform(), nunca .fit().

### 5.1 Remover duplicatas

```python
# Reconstruir DataFrame temporário para remoção de duplicatas
train_df = X_train.copy()
train_df['__target__'] = y_train.values

test_df = X_test.copy()
test_df['__target__'] = y_test.values

# Verificar e remover
print(f"Duplicados no treino antes: {train_df.duplicated().sum()}")
train_df = train_df.drop_duplicates(keep='first')
print(f"Duplicados no treino depois: {train_df.duplicated().sum()}")

# Reconstruir X_train e y_train limpos
X_train = train_df.drop(columns=['__target__'])
y_train = train_df['__target__']

print(f"Shape do treino após limpeza: {X_train.shape}")
```

### 5.2 Tratar valores infinitos

```python
# Datasets de rede frequentemente têm Inf gerados por features como
# "bytes por segundo" quando a duração do fluxo é zero
# Substituir Inf por NaN para depois imputar

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

print(f"NaN no treino após substituição de Inf: {X_train.isnull().sum().sum()}")
```

### 5.3 Tratar valores ausentes

```python
from sklearn.impute import SimpleImputer

# Imputar com a mediana (mais robusto que a média para features assimétricas de rede)
# fit() SOMENTE no treino
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)

print(f"NaN no treino após imputação: {X_train.isnull().sum().sum()}")
print(f"NaN no teste após imputação:  {X_test.isnull().sum().sum()}")
```

### 5.4 Remover features com variância zero

```python
from sklearn.feature_selection import VarianceThreshold

# Features com variância zero não contribuem com nenhuma informação
var_thresh = VarianceThreshold(threshold=0.0)
var_thresh.fit(X_train)

# Identificar colunas removidas
cols_before = X_train.columns.tolist()
X_train = pd.DataFrame(
    var_thresh.transform(X_train),
    columns=X_train.columns[var_thresh.get_support()]
)
X_test = pd.DataFrame(
    var_thresh.transform(X_test),
    columns=X_test.columns[var_thresh.get_support()]
)

cols_removed = set(cols_before) - set(X_train.columns)
print(f"Features removidas (variância zero): {len(cols_removed)}")
print(f"Features restantes: {X_train.shape[1]}")
```

---

## Etapa 6 — Seleção de Features com SHAP

> O artigo usa SHAP (Shapley Additive Explanations) para identificar as features
> mais importantes. Seguindo o artigo, os atributos mais relevantes no InSDN são
> relacionados a tamanho de pacotes e características de fluxo.
> Aqui fazemos isso em duas fases: um modelo rápido para obter os SHAP values,
> depois filtramos as features.

### 6.1 Modelo inicial rápido para calcular importância

```python
from sklearn.ensemble import RandomForestClassifier

# RandomForest rápido para calcular feature importance como proxy antes do SHAP
# Usar amostra para acelerar (SHAP é computacionalmente caro para datasets grandes)
sample_size = min(10000, len(X_train))
X_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
y_sample = y_train.loc[X_sample.index]

rf_for_shap = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_for_shap.fit(X_sample, y_sample)
print("Modelo auxiliar treinado para SHAP.")
```

### 6.2 Calcular SHAP values

```python
# TreeExplainer é eficiente para modelos baseados em árvore
explainer = shap.TreeExplainer(rf_for_shap)
shap_values = explainer.shap_values(X_sample)

# Para classificação binária, pegar os valores da classe 1 (ataque)
if isinstance(shap_values, list):
    shap_vals_attack = shap_values[1]
else:
    shap_vals_attack = shap_values

# Importância média absoluta por feature
feature_importance_shap = pd.DataFrame({
    'feature': X_train.columns,
    'shap_importance': np.abs(shap_vals_attack).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("Top 20 features mais importantes (SHAP):")
print(feature_importance_shap.head(20).to_string(index=False))
```

### 6.3 Visualizar SHAP (summary plot)

```python
# Beeswarm plot — mostra direção e magnitude do impacto de cada feature
shap.summary_plot(
    shap_vals_attack,
    X_sample,
    max_display=20,
    show=True
)

# Bar plot — importância média absoluta
shap.summary_plot(
    shap_vals_attack,
    X_sample,
    plot_type='bar',
    max_display=20,
    show=True
)
```

### 6.4 Selecionar top features

```python
# Selecionar as N features mais importantes
# Comparar com as do artigo para o InSDN:
# Fwd Pkt Len Mean, Pkt Size Avg, Pkt Len Mean, Fwd Seg Size Avg,
# Dst Port, Subflow Fwd Byts, Flow Duration, etc.

N_FEATURES = 20  # ajustar conforme análise — o artigo usou ~20 features principais

top_features = feature_importance_shap.head(N_FEATURES)['feature'].tolist()

print(f"\nFeatures selecionadas ({N_FEATURES}):")
for i, f in enumerate(top_features, 1):
    print(f"  {i:2d}. {f}")

# Filtrar datasets
X_train_sel = X_train[top_features].copy()
X_test_sel  = X_test[top_features].copy()

print(f"\nShape treino após seleção: {X_train_sel.shape}")
print(f"Shape teste após seleção:  {X_test_sel.shape}")
```

---

## Etapa 7 — Escalonamento (StandardScaler)

> Regra absoluta: fit() SOMENTE no treino.
> MLP é especialmente sensível à escala dos dados —
> sem escalonamento, o treinamento pode divergir.

```python
scaler = StandardScaler()

# fit_transform no treino
X_train_scaled = scaler.fit_transform(X_train_sel)

# Apenas transform no teste (sem re-fit — evita data leakage)
X_test_scaled = scaler.transform(X_test_sel)

print("Escalonamento aplicado.")
print(f"Média das features no treino (deve ser ~0): {X_train_scaled.mean(axis=0).mean():.4f}")
print(f"Std das features no treino (deve ser ~1):   {X_train_scaled.std(axis=0).mean():.4f}")
```

---

## Etapa 8 — Balanceamento de Classes com SMOTE

> O InSDN tem ~81% ataque vs ~19% benigno.
> SMOTE gera instâncias sintéticas da classe minoritária (benigno).
> OBRIGATÓRIO: aplicar SOMENTE no treino, NUNCA no teste.
> O teste deve representar a realidade (dados desequilibrados).

```python
print(f"Distribuição ANTES do SMOTE:")
print(f"  Benigno (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Ataque  (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

# SMOTE — aplica SOMENTE no treino
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"\nDistribuição DEPOIS do SMOTE:")
print(f"  Benigno (0): {(y_train_bal == 0).sum()} ({(y_train_bal == 0).mean()*100:.1f}%)")
print(f"  Ataque  (1): {(y_train_bal == 1).sum()} ({(y_train_bal == 1).mean()*100:.1f}%)")
print(f"\nShape treino balanceado: {X_train_bal.shape}")
```

---

## Etapa 9 — Treinamento do MLP Baseline

> Baseline = modelo com parâmetros padrão (ou próximos do artigo).
> Arquitetura do artigo: 128 → 64 → saída, ativação ReLU, ADAM.

### 9.1 Definir e treinar o baseline

```python
# Arquitetura exata do artigo para MLP
mlp_baseline = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # camadas ocultas: 128 neurônios → 64 neurônios
    activation='relu',              # ReLU nas camadas ocultas
    solver='adam',                  # ADAM optimizer (com momentum adaptativo)
    alpha=0.0001,                   # regularização L2
    batch_size='auto',              # mini-batch (padrão sklearn)
    learning_rate='adaptive',       # reduz taxa se o treino estagna
    max_iter=200,                   # épocas máximas
    random_state=RANDOM_STATE,
    early_stopping=True,            # para automaticamente quando a val_loss não melhora
    validation_fraction=0.1,        # 10% do treino como validação interna
    n_iter_no_change=10,            # critério de parada: 10 épocas sem melhora
    verbose=True
)

print("Treinando MLP baseline...")
mlp_baseline.fit(X_train_bal, y_train_bal)
print("Treinamento concluído.")
```

### 9.2 Validação cruzada no conjunto de treino

```python
# NUNCA usar X_test aqui — validação cruzada é feita SOMENTE no treino
# Usar StratifiedKFold para preservar proporção de classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Avaliar com múltiplas métricas
for metric in ['accuracy', 'f1', 'precision', 'recall']:
    scores = cross_val_score(
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=RANDOM_STATE
        ),
        X_train_bal, y_train_bal,
        cv=cv,
        scoring=metric,
        n_jobs=-1
    )
    print(f"CV {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 9.3 Curva de Loss (convergência)

```python
# Verificar se o modelo convergiu adequadamente
plt.figure(figsize=(10, 4))
plt.plot(mlp_baseline.loss_curve_, label='Loss de Treino', color='steelblue')
if hasattr(mlp_baseline, 'validation_scores_'):
    plt.plot(mlp_baseline.validation_scores_, label='Score de Validação',
             color='orange', linestyle='--')
plt.xlabel('Épocas')
plt.ylabel('Loss / Score')
plt.title('Curva de Convergência do MLP Baseline')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Etapa 10 — Métricas de Avaliação no Test Set

> O test set é usado APENAS aqui — NUNCA durante o desenvolvimento.
> Representa dados "do mundo real" que o modelo nunca viu.

### 10.1 Predições

```python
predictions = mlp_baseline.predict(X_test_scaled)
predictions_proba = mlp_baseline.predict_proba(X_test_scaled)[:, 1]
```

### 10.2 Métricas completas

```python
# Métricas principais
acc  = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions)
rec  = recall_score(y_test, predictions)
f1   = f1_score(y_test, predictions)
mcc  = matthews_corrcoef(y_test, predictions)
gm   = geometric_mean_score(y_test, predictions)
auc  = roc_auc_score(y_test, predictions_proba)

print("=" * 55)
print(f"{'RESULTADOS — MLP BASELINE (InSDN)':^55}")
print("=" * 55)
print(f"  Acurácia          : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precisão          : {prec:.4f}  ({prec*100:.2f}%)")
print(f"  Recall/Sensibil.  : {rec:.4f}  ({rec*100:.2f}%)")
print(f"  F1-Score          : {f1:.4f}  ({f1*100:.2f}%)")
print(f"  MCC               : {mcc:.4f}")
print(f"  Geometric Mean    : {gm:.4f}")
print(f"  ROC-AUC           : {auc:.4f}")
print("=" * 55)

# Comparação com o artigo
print("\nComparação com o artigo (Mehmood et al., 2025):")
print(f"  Acurácia artigo   : 99.98%  | Obtido: {acc*100:.2f}%")
print(f"  Precisão artigo   : 99.99%  | Obtido: {prec*100:.2f}%")
print(f"  Recall artigo     : 99.97%  | Obtido: {rec*100:.2f}%")
print(f"  F1 artigo         : 99.98%  | Obtido: {f1*100:.2f}%")
```

### 10.3 Relatório completo de classificação

```python
print("\nRelatório de Classificação:")
print(classification_report(
    y_test, predictions,
    target_names=['Benigno (0)', 'Ataque (1)']
))
```

### 10.4 Matriz de confusão

```python
cm = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = cm.ravel()

print(f"Verdadeiro Positivo  (TP — ataque detectado): {TP}")
print(f"Verdadeiro Negativo  (TN — benigno correto):  {TN}")
print(f"Falso Positivo       (FP — falso alarme):     {FP}")
print(f"Falso Negativo       (FN — ataque perdido):   {FN}")

# Visualização
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Benigno', 'Ataque']
).plot(values_format='.0f', ax=ax, cmap='Blues')
ax.set_title('Matriz de Confusão — MLP Baseline (InSDN)')
plt.tight_layout()
plt.show()
```

### 10.5 Curva ROC

```python
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(
    y_test,
    predictions_proba,
    name=f"MLP (AUC = {auc:.4f})",
    ax=ax
)
ax.plot([0, 1], [0, 1], 'k--', label='Aleatório')
ax.set_title('Curva ROC — MLP Baseline (InSDN)')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Etapa 11 — Melhoria: Hyperparameter Tuning

> O artigo usa Otimização Bayesiana.
> Aqui usamos RandomizedSearchCV (mais simples, mesma ideia)
> com StratifiedKFold no CONJUNTO DE TREINO.
> O test set NÃO é tocado durante o tuning.

### 11.1 Espaço de busca

```python
from sklearn.model_selection import RandomizedSearchCV

# Espaço de hiperparâmetros baseado no artigo e nas aulas
param_distributions = {
    # Arquitetura da rede
    'hidden_layer_sizes': [
        (128, 64),          # arquitetura do artigo
        (256, 128),
        (128, 64, 32),
        (256, 128, 64),
        (512, 256, 128),
    ],
    # Regularização
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    # Taxa de aprendizagem inicial
    'learning_rate_init': [0.001, 0.01, 0.0001],
    # Estratégia de taxa de aprendizagem
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    # Épocas
    'max_iter': [200, 300, 500],
}
```

### 11.2 Busca aleatória com validação cruzada

```python
# StratifiedKFold garante proporção de classes em cada fold
cv_tuning = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

mlp_tuning_base = MLPClassifier(
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_STATE
)

# RandomizedSearchCV — mais eficiente que GridSearch para espaços grandes
random_search = RandomizedSearchCV(
    estimator=mlp_tuning_base,
    param_distributions=param_distributions,
    n_iter=30,                    # número de combinações a testar
    cv=cv_tuning,
    scoring='f1',                 # F1 como métrica principal (balanceia precisão e recall)
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=2,
    return_train_score=True
)

print("Iniciando busca de hiperparâmetros (pode demorar alguns minutos)...")
random_search.fit(X_train_bal, y_train_bal)

print(f"\nMelhores hiperparâmetros encontrados:")
print(random_search.best_params_)
print(f"\nMelhor F1 score (CV): {random_search.best_score_:.4f}")
```

### 11.3 Treinar modelo final com os melhores parâmetros

```python
best_mlp = random_search.best_estimator_

print("Modelo otimizado treinado com os melhores parâmetros.")
```

---

## Etapa 12 — Avaliação Final no Test Set

> Esta é a avaliação definitiva — feita UMA ÚNICA VEZ com o melhor modelo.

```python
# Predições do modelo otimizado
pred_best = best_mlp.predict(X_test_scaled)
pred_best_proba = best_mlp.predict_proba(X_test_scaled)[:, 1]

# Métricas finais
acc_best  = accuracy_score(y_test, pred_best)
prec_best = precision_score(y_test, pred_best)
rec_best  = recall_score(y_test, pred_best)
f1_best   = f1_score(y_test, pred_best)
mcc_best  = matthews_corrcoef(y_test, pred_best)
gm_best   = geometric_mean_score(y_test, pred_best)
auc_best  = roc_auc_score(y_test, pred_best_proba)

print("=" * 60)
print(f"{'RESULTADOS FINAIS — MLP OTIMIZADO (InSDN)':^60}")
print("=" * 60)
print(f"  Acurácia          : {acc_best:.4f}  ({acc_best*100:.2f}%)")
print(f"  Precisão          : {prec_best:.4f}  ({prec_best*100:.2f}%)")
print(f"  Recall            : {rec_best:.4f}  ({rec_best*100:.2f}%)")
print(f"  F1-Score          : {f1_best:.4f}  ({f1_best*100:.2f}%)")
print(f"  MCC               : {mcc_best:.4f}")
print(f"  Geometric Mean    : {gm_best:.4f}")
print(f"  ROC-AUC           : {auc_best:.4f}")
print("=" * 60)

# Tabela comparativa: Baseline vs. Otimizado vs. Artigo
print("\nTabela Comparativa:")
print(f"{'Métrica':<20} {'Baseline':>10} {'Otimizado':>10} {'Artigo':>10}")
print("-" * 52)
print(f"{'Acurácia (%)':<20} {acc*100:>10.2f} {acc_best*100:>10.2f} {'99.98':>10}")
print(f"{'Precisão (%)':<20} {prec*100:>10.2f} {prec_best*100:>10.2f} {'99.99':>10}")
print(f"{'Recall (%)':<20} {rec*100:>10.2f} {rec_best*100:>10.2f} {'99.97':>10}")
print(f"{'F1-Score (%)':<20} {f1*100:>10.2f} {f1_best*100:>10.2f} {'99.98':>10}")
print(f"{'MCC':<20} {mcc:>10.4f} {mcc_best:>10.4f} {'—':>10}")

# Relatório completo
print("\nRelatório de Classificação — Modelo Otimizado:")
print(classification_report(
    y_test, pred_best,
    target_names=['Benigno', 'Ataque']
))

# Matriz de confusão final
cm_best = confusion_matrix(y_test, pred_best)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm_best,
    display_labels=['Benigno', 'Ataque']
).plot(values_format='.0f', ax=ax, cmap='Blues')
ax.set_title('Matriz de Confusão — MLP Otimizado (InSDN)')
plt.tight_layout()
plt.show()
```

---

## Etapa 13 — Salvando o Modelo e os Transformadores

> Importante: salvar TAMBÉM o imputer, o VarianceThreshold,
> a lista de features selecionadas e o scaler.
> Em produção, os dados passam pelas mesmas transformações
> aprendidas no treino — usando os MESMOS objetos fitados.

```python
# Criar diretório de saída
import os
os.makedirs("models/", exist_ok=True)

# Salvar todos os componentes do pipeline
with open('models/mlp_ddos_insdn.joblib', 'wb') as f:
    joblib.dump(best_mlp, f)

with open('models/imputer.joblib', 'wb') as f:
    joblib.dump(imputer, f)

with open('models/variance_filter.joblib', 'wb') as f:
    joblib.dump(var_thresh, f)

with open('models/scaler.joblib', 'wb') as f:
    joblib.dump(scaler, f)

with open('models/selected_features.joblib', 'wb') as f:
    joblib.dump(top_features, f)

print("Todos os componentes salvos em models/")
print("  ├── mlp_ddos_insdn.joblib     (modelo MLP)")
print("  ├── imputer.joblib            (imputador de missing values)")
print("  ├── variance_filter.joblib    (filtro de variância zero)")
print("  ├── scaler.joblib             (StandardScaler)")
print("  └── selected_features.joblib  (lista das features selecionadas)")
```

### Usando o modelo em produção (inferência)

```python
def carregar_pipeline_producao():
    """Carrega todos os componentes salvos durante o treinamento."""
    mlp       = joblib.load('models/mlp_ddos_insdn.joblib')
    imputer   = joblib.load('models/imputer.joblib')
    var_filt  = joblib.load('models/variance_filter.joblib')
    scaler    = joblib.load('models/scaler.joblib')
    features  = joblib.load('models/selected_features.joblib')
    return mlp, imputer, var_filt, scaler, features


def detectar_ddos(df_novos_dados: pd.DataFrame) -> np.ndarray:
    """
    Recebe um DataFrame com o tráfego de rede capturado e retorna
    as predições: 0 = Benigno, 1 = Ataque DDoS.
    """
    mlp, imputer, var_filt, scaler, features = carregar_pipeline_producao()

    # 1. Substituir infinitos
    df = df_novos_dados.replace([np.inf, -np.inf], np.nan)

    # 2. Imputar missing (TRANSFORM apenas)
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)

    # 3. Filtro de variância (TRANSFORM apenas)
    df = pd.DataFrame(var_filt.transform(df), columns=df.columns[var_filt.get_support()])

    # 4. Selecionar as features usadas no treino
    df = df[features]

    # 5. Escalonar (TRANSFORM apenas)
    df_scaled = scaler.transform(df)

    # 6. Predição
    predicoes = mlp.predict(df_scaled)
    return predicoes


# Exemplo de uso:
# novos_dados = pd.read_csv("trafego_capturado.csv")
# resultado = detectar_ddos(novos_dados)
# print(resultado)  # array de 0s e 1s
```

---

## Referência Rápida: Checklist do Pipeline

Use esta checklist antes de executar cada etapa para garantir que
nenhuma boa prática foi violada:

```
[ ] 1. EDA feita SEM modificar os dados
[ ] 2. Split realizado ANTES de qualquer transformação
[ ] 3. stratify=y definido no train_test_split
[ ] 4. random_state definido em TODAS as funções estocásticas
[ ] 5. imputer.fit() somente em X_train
[ ] 6. var_thresh.fit() somente em X_train
[ ] 7. SHAP calculado somente com dados de treino
[ ] 8. scaler.fit() somente em X_train
[ ] 9. SMOTE aplicado somente em X_train
[ ] 10. Validação cruzada feita somente em X_train_bal
[ ] 11. Hyperparameter tuning feito somente em X_train_bal
[ ] 12. X_test usado UMA ÚNICA VEZ na avaliação final
[ ] 13. Todos os objetos fitados salvos junto com o modelo
```

---

## Estrutura Final de Arquivos

```
projeto_ddos_mlp/
├── data/
│   └── InSDN/
│       ├── Normal.csv
│       ├── OFswitch.csv
│       └── ...
├── models/
│   ├── mlp_ddos_insdn.joblib
│   ├── imputer.joblib
│   ├── variance_filter.joblib
│   ├── scaler.joblib
│   └── selected_features.joblib
├── notebooks/
│   └── ddos_mlp_insdn.ipynb     ← notebook principal com todo o código
├── outputs/
│   ├── eda_insdn_report.html
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_otimizado.png
│   └── roc_curve.png
└── README.md
```

---

## Notas para Integração com o Experimento FL-SDN

Considerando que você já tem o ambiente SDN com os três cenários
(C1: com_sdn + health score, C2: sem_sdn, C3: rerouting), a integração
natural seria:

**1. Coleta de dados:** Capturar o tráfego gerado nos seus experimentos
com ferramentas como `tshark` ou `CICFlowMeter` para gerar features
compatíveis com o InSDN (mesmas 84 features) — permitindo usar o
modelo treinado diretamente para detectar anomalias no seu ambiente.

**2. Avaliação por cenário:** Aplicar o modelo treinado no InSDN ao
tráfego dos cenários C1, C2 e C3 separadamente, comparando a taxa de
detecção e o número de falsos positivos em cada configuração SDN.

**3. Demonstração do valor do SDN:** O cenário C1 (com health score)
deve apresentar menos tráfego anômalo chegando ao controlador — o
que se traduz em menos ativações do detector de DDoS — demonstrando
que o roteamento inteligente do SDN, além de acelerar a convergência
do FL, também contribui para a segurança da rede.

---

*Plano elaborado com base em:*
*Mehmood et al. (PLoS ONE, 2025) + Aulas do curso Paradigmas de*
*Aprendizagem de Máquina — Thaís Gaudencio do Rêgo, UFPB/LUMO.*
