# Guia de Boas Práticas para Implementação de Modelos de Machine Learning

> Baseado nas aulas do curso *Paradigmas de Aprendizagem de Máquina* (UFPB/LUMO) — Thaís Gaudencio do Rêgo.

---

## Sumário

1. [O Workflow Padrão](#1-o-workflow-padrão)
2. [Configurações Iniciais](#2-configurações-iniciais)
3. [Obtenção e Entendimento dos Dados (EDA)](#3-obtenção-e-entendimento-dos-dados-eda)
4. [A Regra de Ouro: Split Antes de Tudo](#4-a-regra-de-ouro-split-antes-de-tudo)
5. [Limpeza e Preparação dos Dados](#5-limpeza-e-preparação-dos-dados)
6. [Transformação de Atributos](#6-transformação-de-atributos)
7. [Análise de Balanceamento de Classes](#7-análise-de-balanceamento-de-classes)
8. [Escolha do Algoritmo](#8-escolha-do-algoritmo)
9. [Treinamento e Avaliação do Modelo Baseline](#9-treinamento-e-avaliação-do-modelo-baseline)
10. [Métricas de Avaliação](#10-métricas-de-avaliação)
11. [Melhoria do Modelo (Hyperparameter Tuning)](#11-melhoria-do-modelo-hyperparameter-tuning)
12. [Salvando o Modelo](#12-salvando-o-modelo)
13. [Referência Rápida: Armadilhas Comuns](#13-referência-rápida-armadilhas-comuns)

---

## 1. O Workflow Padrão

Todo projeto de ML segue uma sequência de etapas bem definidas. Seguir essa ordem **não é opcional** — invertê-la é a principal causa de resultados otimistas e irreais.

```
┌─────────────────────────────────────────────────────────┐
│  1. Configurações Básicas                               │
│  2. Obtenção dos Dados + EDA                            │
│  3. ⚠️  SPLIT treino/teste  ⚠️  (antes de qualquer     │
│         transformação)                                  │
│  4. Limpeza e Preparação (somente no treino!)           │
│  5. Transformação de Atributos (fit só no treino)       │
│  6. Treinamento do Modelo Baseline                      │
│  7. Avaliação e Métricas                                │
│  8. Melhoria do Modelo (tuning)                         │
│  9. Salvar o Modelo                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Configurações Iniciais

```python
# Acesso ao Google Drive (Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Aumentar a capacidade de visualização do Pandas
import pandas as pd
pd.set_option('display.max_columns', 7000)
pd.set_option('display.max_rows', 90000)

# Reprodutibilidade: sempre defina uma semente!
RANDOM_STATE = 42
```

---

## 3. Obtenção e Entendimento dos Dados (EDA)

Explore os dados **antes** de qualquer modificação. O objetivo é entender a estrutura, identificar problemas e formular hipóteses. Esta etapa é feita sobre o dataset completo — as transformações é que vêm depois do split.

### 3.1 Carga e visão geral

```python
import pandas as pd

data = pd.read_csv("caminho/para/dataset.csv")

# Estrutura geral
data.head()
data.info()          # tipos, valores não-nulos
data.describe()      # estatísticas numéricas
data.describe(include="O")  # estatísticas de colunas categóricas (Object)
```

### 3.2 EDA automatizada

```python
# ydata-profiling gera um relatório completo com uma linha
pip install ydata-profiling

from ydata_profiling import ProfileReport
profile = ProfileReport(data)
profile.to_notebook_iframe()
```

> **Por que usar?** Economiza tempo ao revelar automaticamente: correlações, distribuições, valores ausentes, duplicados e avisos sobre colunas problemáticas.

### 3.3 Análise de distribuição e outliers

```python
import matplotlib.pyplot as plt

# Histogramas: entender assimetria de cada atributo
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
data.hist(ax=ax)
plt.show()

# Boxplots: identificar outliers visualmente
data.plot(kind="box", subplots=True, layout=(4, 2),
          figsize=(8, 8), sharex=False)
plt.tight_layout()
plt.show()

# KDE (densidade): entender a forma da distribuição
data.plot(kind="density", subplots=True, layout=(4, 2),
          figsize=(8, 8), sharex=False)
plt.tight_layout()
plt.show()
```

### 3.4 Análise de correlações

```python
import seaborn as sns

# Correlação entre atributos numéricos
numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(11, 6))
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="RdBu_r")
plt.show()

# Ordenar por correlação com o alvo
numeric_data.corr()["coluna_alvo"].sort_values()
```

> **Regra prática:** correlação acima de 0.7 ou abaixo de -0.7 é considerada forte. Atributos muito correlacionados entre si podem ser redundantes.

### 3.5 Análise de atributos categóricos

```python
# Relação entre categoria e alvo (crosstab)
pd.crosstab(data['coluna_alvo'], data['coluna_categorica'])

# Visualização de categorias vs alvo
import seaborn as sns
g = sns.catplot(x="sex", hue="race", col="high_income",
                data=data, kind="count", height=4, aspect=0.7)
```

---

## 4. A Regra de Ouro: Split Antes de Tudo

> **⚠️ Esta é a etapa mais importante do pipeline.** Todo e qualquer pré-processamento que "aprenda" algo dos dados — normalização, imputação, OHE, remoção de outliers — deve ser feito **somente após** esta separação. Fazer antes é **vazamento de dados** (*data leakage*).

### Por que o vazamento de dados é problemático?

Quando você normaliza *antes* do split, o escalonador aprende a média e o desvio padrão do conjunto completo (incluindo o teste). Isso significa que o modelo viu indiretamente informações do conjunto de teste durante o treinamento. O resultado é um modelo artificialmente otimista que **não generaliza** para dados reais.

### 4.1 Para classificação (use `stratify`)

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(
    data,
    test_size=0.30,       # 70% treino / 30% teste (padrão do curso)
    random_state=35,      # SEMPRE defina! Garante reprodutibilidade.
    shuffle=True,
    stratify=data["coluna_alvo"]  # preserva a proporção de classes
)

print(f"Train: {train_set.shape} | Test: {test_set.shape}")

# Verificar se a distribuição foi preservada
print(train_set['coluna_alvo'].value_counts(normalize=True))
print(test_set['coluna_alvo'].value_counts(normalize=True))
```

> **Por que `stratify`?** Sem ele, o split aleatório pode colocar instâncias de uma classe rara quase inteiramente em um único conjunto. O `stratify` garante que treino e teste tenham a mesma proporção de classes do conjunto original.

### 4.2 Para regressão

```python
train_set, test_set = train_test_split(
    data,
    test_size=0.20,       # 80/20 é comum para regressão
    random_state=42
)
```

### 4.3 Separar X e y somente após o split

```python
# A separação X/y vem depois do split, não antes
train_X = train_set.drop("coluna_alvo", axis=1)
train_y = train_set["coluna_alvo"].copy()

test_X = test_set.drop("coluna_alvo", axis=1)
test_y = test_set["coluna_alvo"].copy()
```

---

## 5. Limpeza e Preparação dos Dados

A partir daqui, **todas as operações são feitas sobre o conjunto de treino**. As mesmas transformações são aplicadas ao teste ao final, mas nunca "fitadas" nele.

### 5.1 Verificar e remover duplicatas

```python
# Verificar
dupes = train_set.duplicated()
print(f"Duplicados no treino: {sum(dupes)}")

# Remover (keep='first' mantém a primeira ocorrência)
train_set = train_set.drop_duplicates(keep='first')
test_set = test_set.drop_duplicates(keep='first')  # aplica no teste também

# Confirmar
print(f"Duplicados após limpeza: {sum(train_set.duplicated())}")
```

### 5.2 Remover atributos desnecessários ou redundantes

```python
# Remova atributos que não contribuem para a predição
# Justifique cada remoção no seu código!

# Exemplo: 'education' e 'education_num' contêm a mesma info → manter só numérica
# Exemplo: 'fnlwgt' é um peso demográfico irrelevante para a predição
train_set = train_set.drop(columns=["fnlwgt", "education"], axis=1)
test_set  = test_set.drop(columns=["fnlwgt", "education"], axis=1)
```

### 5.3 Tratar valores ausentes

```python
# Verificar ausentes
print(train_set.isnull().sum())

# Opção 1: remover linhas com ausentes (quando são poucas)
train_set.dropna(inplace=True)

# Opção 2: imputar com a média (fit SOMENTE no treino)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
train_set[['col1', 'col2']] = imputer.fit_transform(train_set[['col1', 'col2']])
test_set[['col1', 'col2']]  = imputer.transform(test_set[['col1', 'col2']])
```

### 5.4 Análise e tratamento de outliers

> **Conceito crítico:** Dado ruidoso ≠ Outlier.
> - **Dado ruidoso:** contém erro (medição incorreta, falha de sensor). Deve ser corrigido ou removido.
> - **Outlier:** é um dado **real e válido** que está fora do padrão. Removê-lo sem justificativa pode distorcer o modelo.

```python
# Identifique outliers visualmente e por domínio, não somente por estatística
# Exemplo: Gr Liv Area > 4000 pé² distorce a regressão porque estas casas
# custam muito menos do que o esperado para seu tamanho (anomalia real do mercado)
train_set = train_set[train_set["Gr Liv Area"] < 4000]

# Para detecção multivariada: LocalOutlierFactor
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor()
outlier_mask = lof.fit_predict(train_set[colunas_numericas])  # -1 = outlier
train_set_clean = train_set[outlier_mask == 1]
```

> **Sempre compare:** treine um modelo com outliers e outro sem. Mantenha a versão com melhor desempenho no conjunto de teste — e justifique a decisão.

---

## 6. Transformação de Atributos

### 6.1 Transformação logarítmica em atributos assimétricos

```python
import numpy as np

# Use log1p (não log) para suportar valores zero sem erro
# Aplique antes do escalonamento, em treino E teste igualmente
# log1p não "aprende" nada dos dados — é pura função matemática, sem risco de leakage

train_set['Lot Area']  = np.log1p(train_set['Lot Area'])
test_set['Lot Area']   = np.log1p(test_set['Lot Area'])

train_set['SalePrice'] = np.log1p(train_set['SalePrice'])
test_set['SalePrice']  = np.log1p(test_set['SalePrice'])
```

> **Por que `log1p` e não `log`?** `log(0)` é indefinido. `log1p(x) = log(1+x)` funciona mesmo quando x=0.

> **Como interpretar o RMSE depois da transformação log?** O erro real em % é aproximado por `e^RMSE - 1`. Por exemplo, RMSE=0.143 → erro de ~15.4%.

### 6.2 Escalonamento (normalização / padronização)

> **Regra absoluta: `fit()` apenas no treino. `transform()` em treino e teste.**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# MinMaxScaler: reescala para [0, 1]
# Quando usar: algoritmos sensíveis a escala (KNN, redes neurais)
# Desvantagem: sensível a outliers

# StandardScaler: z-score (média=0, desvio=1)
# Quando usar: quando há outliers (lida melhor)
# Fórmula: (x - média) / desvio_padrão

colunas_para_escalonar = ['age', 'education_num', 'capital_gain', 'hours_per_week']

scaler = StandardScaler()

# SOMENTE fit no treino — aprender média e desvio APENAS dos dados de treino
train_set[colunas_para_escalonar] = scaler.fit_transform(train_set[colunas_para_escalonar])

# Aplicar a mesma transformação no teste (sem re-fit!)
test_set[colunas_para_escalonar]  = scaler.transform(test_set[colunas_para_escalonar])
```

> **⚠️ Nunca escalone atributos binários (0/1).** Eles já estão na escala correta e escaloná-los distorce seu significado.

### 6.3 Codificação de variáveis categóricas (One-Hot Encoding)

```python
from sklearn.preprocessing import OneHotEncoder

# sparse_output=False → retorna array denso (não matriz esparsa)
# drop='first' → remove a primeira categoria para evitar multicolinearidade
# fit() apenas no treino!

colunas_categoricas = train_set.select_dtypes("object").columns.tolist()

for coluna in colunas_categoricas:
    ohe = OneHotEncoder(sparse_output=False, drop='first')

    ohe.fit(train_set[coluna].values.reshape(-1, 1))

    train_set[ohe.get_feature_names_out([coluna])] = ohe.transform(
        train_set[coluna].values.reshape(-1, 1))
    test_set[ohe.get_feature_names_out([coluna])]  = ohe.transform(
        test_set[coluna].values.reshape(-1, 1))

# Remover colunas originais (object) após o encoding
train_set.drop(labels=colunas_categoricas, axis=1, inplace=True)
test_set.drop(labels=colunas_categoricas,  axis=1, inplace=True)
```

> **Por que `drop='first'`?** Com k categorias, bastam k-1 colunas binárias para representar todas — a categoria removida é inferida quando todas as outras são zero. Sem isso, há multicolinearidade perfeita que pode prejudicar modelos lineares.

---

## 7. Análise de Balanceamento de Classes

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizar distribuição das classes
sns.countplot(x=train_set["coluna_alvo"], palette='Set1')
plt.show()

# Calcular percentuais
contagem = train_set["coluna_alvo"].value_counts()
print(contagem / len(train_set) * 100)

# Medir balanceamento com Entropia de Shannon
# Retorna 0 (totalmente desbalanceado) a 1 (perfeitamente balanceado)
def balance(df, target, n_classes):
    from numpy import log
    n = len(df)
    H = 0
    for classe in df[target].unique():
        p = len(df[df[target] == classe]) / n
        H += p * log(p)
    return (-H) / log(n_classes)

print(f"Balanceamento (Shannon): {balance(train_set, 'coluna_alvo', 2):.3f}")
```

### Quando usar SMOTE?

```python
# Use SMOTE quando o dataset for MUITO desbalanceado
# e a entropia de Shannon estiver próxima de 0
# Aplique SOMENTE no treino, NUNCA no teste!

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
train_X_res, train_y_res = sm.fit_resample(train_X, train_y)
```

> **Por que não aplicar SMOTE no teste?** O SMOTE gera instâncias sintéticas. O conjunto de teste deve representar o mundo real, não dados gerados artificialmente.

---

## 8. Escolha do Algoritmo

A escolha do algoritmo deve ser **justificada**. A tabela abaixo resume as diretrizes adotadas no curso:

| Situação | Algoritmo Recomendado | Justificativa |
|---|---|---|
| Baseline rápido, qualquer problema | KNN | Não-paramétrico, não assume distribuição; instância-based |
| Regressão, interpretabilidade importante | `LinearRegression` | Paramétrico, aprende função matemática explícita |
| Regressão, dataset pequeno (<10k linhas) | `GradientBoostingRegressor` | Melhor que HistGB em datasets pequenos |
| Regressão, dataset grande (>10k linhas) | `HistGradientBoostingRegressor` | Mais eficiente computacionalmente em grandes volumes |
| Classificação tabular, sem deep learning | `DecisionTreeClassifier` ou `RandomForestClassifier` | DT admite exceções; RF reduz variância via ensemble |
| Classificação tabular, dados pequenos | `MLPClassifier` (sklearn) | Mais simples, menor risco de overfitting que Keras/TF |
| Classificação de imagens | CNN (TensorFlow/Keras) | Especializado em dados espaciais/visuais |
| Classificação de imagens, poucos dados | VGG16 congelada (Transfer Learning) | Aproveita filtros pré-treinados no ImageNet |

---

## 9. Treinamento e Avaliação do Modelo Baseline

```python
# Exemplo com RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. Instanciar com parâmetros padrão (baseline)
model = RandomForestClassifier(random_state=42)

# 2. Validação cruzada NO CONJUNTO DE TREINO
#    Nunca use o test_set para ajustar o modelo!
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, train_X, train_y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# 3. Treinar o modelo final no treino completo
model.fit(train_X, train_y)

# 4. Avaliar no conjunto de TESTE (apenas ao final)
predictions = model.predict(test_X)
```

---

## 10. Métricas de Avaliação

### 10.1 Regressão

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# RMSE
rmse = np.sqrt(mean_squared_error(test_y, predictions))
print(f"RMSE: {rmse:.4f}")

# Comparar com o desvio padrão do alvo
std_alvo = test_y.std()
print(f"Std(y): {std_alvo:.4f}")
print(f"RMSE < Std(y)? {rmse < std_alvo}")  # True = melhor que prever a média
```

> **Regra de ouro para regressão:** `RMSE < std(y)` indica que o modelo performa melhor do que simplesmente prever a média para todos. Se não for o caso, o modelo não aprendeu nada relevante.

### 10.2 Classificação

```python
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              matthews_corrcoef, f1_score)
from imblearn.metrics import geometric_mean_score

# Acurácia
accuracy = accuracy_score(test_y, predictions)
print(f"Acurácia: {accuracy:.4f}")

# Relatório completo: Precision, Recall, F1 por classe
print(classification_report(test_y, predictions))

# Matriz de confusão
cm = confusion_matrix(test_y, predictions)
TP = cm[1][1]; TN = cm[0][0]; FP = cm[0][1]; FN = cm[1][0]

print(f"Sensibilidade (Recall): {TP/(TP+FN):.4f}")   # VP / (VP+FN)
print(f"Especificidade:         {TN/(TN+FP):.4f}")   # VN / (VN+FP)
print(f"Precisão:               {TP/(TP+FP):.4f}")   # VP / (VP+FP)
print(f"MCC:                    {matthews_corrcoef(test_y, predictions):.4f}")
print(f"Geometric Mean:         {geometric_mean_score(test_y, predictions):.4f}")

# Visualizar matriz de confusão
fig, ax = plt.subplots(figsize=(7, 4))
ConfusionMatrixDisplay(cm).plot(values_format=".0f", ax=ax)
plt.show()
```

### Quando usar cada métrica?

| Situação | Métricas prioritárias |
|---|---|
| Classes balanceadas | Acurácia, F1 |
| Classes desbalanceadas | MCC, Geometric Mean, F1 por classe |
| Custo alto de falso positivo | Precisão |
| Custo alto de falso negativo | Recall/Sensibilidade |
| Detecção de objetos em imagem | IoU, Dice Score, mAP |
| Regressão | RMSE comparado ao std(y) |

> **⚠️ Em classes desbalanceadas, acurácia sozinha é enganosa!** Um modelo que sempre prevê a classe majoritária pode ter acurácia de 95% em um dataset 95/5 — e ainda assim ser inútil.

---

## 11. Melhoria do Modelo (Hyperparameter Tuning)

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# GridSearchCV: testa TODAS as combinações (bom para espaços pequenos)
# RandomizedSearchCV: amostra aleatoriamente (bom para espaços grandes)

# Exemplo com RandomForest
search_space = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None]
}

# Use cv no CONJUNTO DE TREINO (nunca inclua o test_set aqui)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch para espaços menores
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    search_space,
    cv=cv,
    scoring='f1',       # use a métrica mais relevante para o problema
    n_jobs=-1,
    verbose=1
)
grid_search.fit(train_X, train_y)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score CV: {grid_search.best_score_:.4f}")

# Avaliar o melhor modelo no test_set (uma única vez)
best_model = grid_search.best_estimator_
predictions = best_model.predict(test_X)
```

> **Por que validar no treino e avaliar no teste?** O `GridSearchCV` já faz validação cruzada interna. O `test_set` é reservado exclusivamente para a avaliação final — como se fosse dados do mundo real que o modelo nunca viu.

---

## 12. Salvando o Modelo

```python
import joblib

# Salvar
with open('modelo_final.joblib', 'wb') as f:
    joblib.dump(best_model, f)

# Carregar (em produção)
with open('modelo_final.joblib', 'rb') as f:
    modelo_carregado = joblib.load(f)

# Usar em produção
nova_predicao = modelo_carregado.predict(novos_dados)
```

> **Boas práticas ao salvar:** salve também o `scaler` e o `ohe` fitados no treino! Em produção, os novos dados devem passar pelas mesmas transformações, usando os mesmos parâmetros aprendidos no treino.

---

## 13. Referência Rápida: Armadilhas Comuns

| ❌ Armadilha | ✅ Solução Correta |
|---|---|
| Fazer `scaler.fit()` em todo o dataset | `fit()` somente no treino; `transform()` em treino e teste |
| Fazer `scaler.fit()` no teste | Nunca! Use apenas `transform()` no teste |
| Escalonar colunas binárias (0/1) | Não escalonar binárias — já estão na escala correta |
| Remover outliers antes do split | Remover outliers **após** o split, somente no treino |
| Fazer OHE antes do split | OHE com `fit()` somente no treino, após o split |
| Usar `log()` em colunas com zeros | Usar `log1p()` — suporta valor zero |
| Usar acurácia em dados desbalanceados | Usar MCC, Geometric Mean ou F1 por classe |
| Usar HistGradientBoosting em datasets pequenos | Preferir GradientBoostingRegressor (<10k linhas) |
| Usar TensorFlow/Keras para dados tabulares pequenos | Preferir `MLPClassifier` do sklearn |
| Aplicar SMOTE no conjunto de teste | SMOTE somente no treino |
| Ajustar hiperparâmetros com o test_set | Tuning com CV no treino; test_set apenas na avaliação final |
| Não definir `random_state` | Sempre definir para garantir reprodutibilidade |
| Não usar `stratify` na classificação | Sempre usar `stratify=y` para preservar proporção de classes |
| Confundir dado ruidoso com outlier | Dado ruidoso = erro (tratar); outlier = dado real válido (avaliar antes de remover) |

---

## Apêndice A: Tipos de Gradiente Descendente

Relevante ao entender o comportamento do treinamento de redes neurais e modelos lineares:

| Tipo | Atualiza pesos com | Vantagem | Desvantagem |
|---|---|---|---|
| **Batch (lote completo)** | Todo o dataset | Gradiente preciso, estável | Lento; pode prender em mínimo local |
| **Estocástico (SGD)** | 1 exemplo por vez | Rápido; escapa de mínimos locais | Ruidoso; convergência instável |
| **Mini-batch** | Subconjunto do dataset | Equilíbrio entre os dois | Escolha do tamanho do batch afeta convergência |

> **Padrão prático:** Mini-batch é o modo de treinamento padrão em redes neurais modernas.

---

## Apêndice B: Redução de Dimensionalidade

| Técnica | Tipo | Quando usar | Limitação |
|---|---|---|---|
| **PCA** | Linear | Muitos atributos correlacionados; reduzir multicolinearidade | Sensível a outliers; falha se atributos pouco correlacionados |
| **T-SNE** | Não-linear | **Apenas para visualização** (2D ou 3D) | Não deve ser usado como entrada de modelos |

---

*Guia gerado com base nas aulas práticas e teóricas do curso Paradigmas de Aprendizagem de Máquina — UFPB/LUMO.*
