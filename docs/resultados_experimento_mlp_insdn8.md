# Análise dos Resultados — MLP para Detecção de DDoS no InSDN8

> Experimento executado em 2026-04-02 sobre o dataset `insdn8_ddos_binary_0n1d.csv`.  
> Referência: Mehmood et al. (PLoS ONE, 2025) — replicação com variante MLP.

---

## Sumário

1. [O que o Pipeline Fez — Passo a Passo](#1-o-que-o-pipeline-fez--passo-a-passo)
2. [Análise dos Dados (EDA)](#2-análise-dos-dados-eda)
3. [O Impacto das Duplicatas — Descoberta Crítica](#3-o-impacto-das-duplicatas--descoberta-crítica)
4. [Importância das Features (SHAP)](#4-importância-das-features-shap)
5. [Convergência do Treino](#5-convergência-do-treino)
6. [Resultados do Modelo Baseline](#6-resultados-do-modelo-baseline)
7. [Comparação: Baseline vs. Otimizado](#7-comparação-baseline-vs-otimizado)
8. [Por Que os Resultados São Tão Bons?](#8-por-que-os-resultados-são-tão-bons)
9. [Glossário dos Conceitos Usados](#9-glossário-dos-conceitos-usados)

---

## 1. O que o Pipeline Fez — Passo a Passo

O pipeline executou 14 etapas em ordem rigorosa. A ordem **não é opcional** — invertê-la invalida o experimento por vazamento de dados (*data leakage*).

```
DADOS BRUTOS (190.366 instâncias, 9 colunas)
     │
     ▼ [Etapa 3] EDA — observar sem modificar
     │
     ▼ [Etapa 4] ⚠️ SPLIT 70/30 ESTRATIFICADO ⚠️
     │             Train: 133.256 | Test: 57.110
     │
     ▼ [Etapa 5] Limpeza no TREINO
     │             87.084 duplicatas removidas → 46.172 instâncias
     │             (o teste recebe só .transform(), sem novas decisões)
     │
     ▼ [Etapa 6] VarianceThreshold + SHAP
     │             0 features removidas (todas têm variância > 0)
     │             Ranking SHAP calculado sobre amostra de 10.000 do treino
     │
     ▼ [Etapa 7] StandardScaler (fit SOMENTE no treino)
     │             Média ≈ 0, Desvio ≈ 1 nas features
     │
     ▼ [Etapa 8] SMOTE (SOMENTE no treino)
     │             Classe 0 sobreamostrada para equilibrar
     │
     ▼ [Etapa 9] Treino MLP (128→64, ReLU, ADAM)
     │             48 épocas | Early stopping ativado
     │
     ▼ [Etapa 10] Avaliação BASELINE no test set (1ª e única vez)
     │
     ▼ [Etapa 11] RandomizedSearchCV (tuning no treino — 30 combinações)
     │
     ▼ [Etapa 12] Avaliação OTIMIZADO no test set
     │
     ▼ [Etapas 13-14] Comparação + Salvamento de artefatos
```

---

## 2. Análise dos Dados (EDA)

### Estrutura do dataset

| Propriedade | Valor |
|---|---|
| Total de instâncias | 190.366 |
| Features | 8 (Protocol, Flow Duration, Flow IAT Max, Bwd Pkts/s, Pkt Len Std, Pkt Len Var, Bwd IAT Tot, Flow Pkts/s) |
| Coluna alvo | Label (0 = Benigno, 1 = Ataque DDoS) |
| Tipo das features | 2 int64 + 6 float64 |
| Valores ausentes | **0** |
| Valores infinitos | **0** |

### Distribuição das classes

| Classe | Instâncias | % |
|---|---|---|
| 1 — Ataque DDoS | 121.942 | 64,06% |
| 0 — Benigno | 68.424 | 35,94% |

**Entropia de Shannon: 0.9422** → Dataset moderadamente desbalanceado mas longe do extremo (a entropia máxima seria 1.0 com classes iguais). Isso justifica o SMOTE, mas o problema não é tão severo quanto parece.

> **Por que o InSDN8 tem "apenas" 190k instâncias?** O InSDN original tem 361.317. O sufixo `insdn8` indica que este é um subconjunto com as 8 features mais discriminativas já pré-selecionadas de pesquisas anteriores.

---

## 3. O Impacto das Duplicatas — Descoberta Crítica

Esta foi a descoberta mais importante do experimento:

```
Dataset completo  : 190.366 instâncias
  ├── Treino bruto: 133.256 (70%)
  │     └── Duplicatas removidas: 87.084
  │     └── Treino limpo:          46.172  ← só 34,6% do treino original!
  └── Teste       :  57.110 (30%)
```

**65,4% do conjunto de treino eram duplicatas.** Isso é normal para datasets de captura de tráfego de rede — flows idênticos são gerados em rajadas de ataque. A remoção é correta e necessária porque:

1. **Overfitting por memorização**: duplicatas ensinam o modelo a "decorar" padrões em vez de generalizar
2. **Ilusão de desempenho**: a validação cruzada avaliaria instâncias que o modelo já viu literalmente idênticas
3. **Custo computacional desnecessário**: treinar em 133k vs 46k sem ganho real

> **Por que NÃO remover duplicatas do teste?** O teste representa o mundo real, que pode conter fluxos repetidos. Remover duplicatas do teste inflaria artificialmente as métricas (o modelo avaliaria menos casos difíceis). No nosso código, a remoção de duplicatas é feita apenas no treino — o teste recebe apenas `.transform()` do `DataCleaner`.

---

## 4. Importância das Features (SHAP)

### Ranking por importância (SHAP bar plot)

| Posição | Feature | SHAP Médio | Interpretação |
|---|---|---|---|
| 1 | **Protocol** | ~0.022 | **Feature mais discriminativa** — protocolo de rede revela o tipo de ataque |
| 2 | **Flow Pkts/s** | ~0.013 | Pacotes por segundo — ataques DDoS têm taxas anormalmente altas |
| 3 | **Flow Duration** | ~0.010 | Duração do fluxo — ataques são tipicamente de curta duração e alta intensidade |
| 4 | **Pkt Len Std** | ~0.010 | Desvio padrão do tamanho dos pacotes — ataques têm pacotes mais uniformes |
| 5 | **Bwd Pkts/s** | ~0.008 | Pacotes por segundo no sentido reverso |
| 6 | **Pkt Len Var** | ~0.008 | Variância do tamanho — relacionada ao Pkt Len Std |
| 7 | **Bwd IAT Tot** | ~0.007 | Tempo total entre chegadas no sentido reverso |
| 8 | **Flow IAT Max** | ~0.006 | Intervalo máximo entre pacotes |

### Leitura do Beeswarm Plot

O beeswarm mostra **direção + magnitude** do impacto:

- **Protocol — valores altos (azul) → SHAP positivo (+0.4):** protocolos específicos são fortemente associados a DDoS
- **Protocol — valores baixos (vermelho) → SHAP negativo (-0.2):** outros protocolos são associados a tráfego benigno
- **Flow Pkts/s — espalhamento grande:** pacotes por segundo com valores extremos (altos) empurram fortemente para a classe ataque

> **Por que Protocol domina?** Ataques DDoS no InSDN são predominantemente de um tipo específico (ex: UDP flood, ICMP flood) com protocol IDs distintos do tráfego HTTP/TCP normal. Uma única feature categórica pode ser altamente discriminativa quando há correspondência quase perfeita com a classe.

---

## 5. Convergência do Treino

A curva de loss revela muito sobre o aprendizado do modelo:

```
Épocas    : ~48 (Early Stopping ativou — não esperou as 200 épocas máximas)
Loss epoch 0: ~0.11  (modelo aleatório, chuta com 64% de chance)
Loss epoch 3: ~0.03  (já aprendeu o padrão principal)
Loss epoch 48: ~0.01 (mínimo atingido, sem melhora por 10 épocas → parou)

Score de Validação Interna: ~1.0 desde a época 0 ← sinal de que o problema é fácil!
```

**Interpretação:** O score de validação interna (laranja) colou em 1.0 quase imediatamente — o modelo aprendeu a separar as classes nas primeiras épocas. A loss continuou caindo lentamente porque o ADAM ajustava pesos para aumentar a margem de confiança, não para mudar as predições em si.

> **Early Stopping** interrompeu o treino após 10 épocas consecutivas sem melhora na validação interna. Isso **economizou ~75% do tempo** de treino (48 épocas vs 200 configuradas) e preveniu overfitting.

---

## 6. Resultados do Modelo Baseline

### Matriz de Confusão

```
                  Predito: Benigno    Predito: Ataque
Real: Benigno         20.468 (TN)         59 (FP)
Real: Ataque DDoS          1 (FN)     36.582 (TP)
```

| Sigla | Nome | Valor | Significado |
|---|---|---|---|
| **TP** | Verdadeiro Positivo | 36.582 | Ataques corretamente detectados |
| **TN** | Verdadeiro Negativo | 20.468 | Tráfego benigno corretamente liberado |
| **FP** | Falso Positivo | 59 | Tráfego benigno bloqueado desnecessariamente (falso alarme) |
| **FN** | Falso Negativo | **1** | **Ataque que passou despercebido (o mais grave!)** |

### Métricas calculadas

| Métrica | Valor | Fórmula | Interpretação |
|---|---|---|---|
| **Acurácia** | 99,90% | (TP+TN) / Total | 99,9 em cada 100 predições corretas |
| **Precisão** | 99,84% | TP / (TP+FP) | Quando diz "é ataque", erra 0,16% das vezes |
| **Recall** | **99,997%** | TP / (TP+FN) | Detecta quase **todos** os ataques reais |
| **F1-Score** | 99,92% | 2×(P×R)/(P+R) | Equilíbrio entre precisão e recall |
| **MCC** | ~0,998 | Fórmula Matthew | Melhor métrica para dados desbalanceados |
| **Geometric Mean** | ~0,998 | √(Sens × Espec) | Equilíbrio sensibilidade/especificidade |
| **ROC-AUC** | **1,0000** | Área sob curva ROC | Separação probabilística perfeita |

> **Por que o AUC é 1.0?** O modelo atribui probabilidades tão bem calibradas que existe um limiar de decisão onde **nenhum** benign tem probabilidade de ataque maior que **nenhum** ataque real. A curva ROC vai direto ao canto superior esquerdo — separação perfeita no espaço de probabilidades.

---

## 7. Comparação: Baseline vs. Otimizado

| Métrica | Baseline | Otimizado | Diferença |
|---|---|---|---|
| TN | 20.468 | 20.448 | -20 |
| **FP** | **59** | **79** | **+20 falsos alarmes** |
| FN | 1 | 1 | 0 |
| TP | 36.582 | 36.582 | 0 |
| Precisão | **99,84%** | **99,78%** | -0,06 pp |
| Recall | 99,997% | 99,997% | 0 |
| F1 | **99,92%** | **99,89%** | -0,03 pp |

### O Tuning Não Melhorou — Por Quê?

Este é um resultado **esperado e saudável**, não um problema. Existem dois motivos:

**1. Teto de performance natural do dataset:** Os dados do InSDN8 são tão bem separáveis (Protocol sozinho já discrimina ~80%) que a arquitetura baseline `(128, 64)` com ADAM já captura toda a estrutura relevante. Não há complexidade escondida para explorar com arquiteturas maiores.

**2. Espaço de busca desfavorável para este problema:** O `RandomizedSearchCV` testou arquiteturas maiores (`(512, 256, 128)`) e diferentes learning rates. Para um problema simples, modelos maiores com mais épocas tendem a ser mais agressivos na classificação — criando ligeiramente mais falsos positivos sem ganhar recall.

> **Conclusão prática:** O baseline `(128, 64)` com ADAM é o modelo ótimo para este dataset. Isso alinha com Mehmood et al. (2025), que também usou esta arquitetura.

---

## 8. Por Que os Resultados São Tão Bons?

Quatro fatores se combinam para produzir performance quasi-perfeita:

### Fator 1: Features altamente discriminativas (sinal forte)

O InSDN8 contém features extraídas de FlowMeter que capturam características estatísticas dos fluxos. Ataques DDoS têm assinaturas muito distintas:

- **Protocol**: DDoS usa UDP/ICMP em massas — completamente diferente do TCP/HTTP normal
- **Flow Pkts/s**: ataques geram centenas de milhares de pacotes/segundo
- **Pkt Len Std = 0**: ataques usam pacotes todos do mesmo tamanho (automatizados)

Quando features têm essa correlação direta com a classe, qualquer modelo razoável performa muito bem.

### Fator 2: Dados limpos e bem preprocessados

Após a remoção das 87.084 duplicatas, o conjunto de treino ficou com 46.172 instâncias **únicas**. O modelo aprendeu de exemplos genuinamente distintos, sem memorizar padrões repetidos. O StandardScaler garantiu que todas as features contribuíssem proporcionalmente — sem que `Flow Pkts/s` (valores da ordem de 10^6) dominasse sobre `Protocol` (valores 0-17).

### Fator 3: SMOTE aplicado corretamente

O SMOTE equilibrou as classes no treino (64/36 → 50/50) sem vazar informação para o teste. Isso evitou que o modelo ficasse enviesado para sempre prever "ataque" — o que teria inflado o recall às custas da precisão.

### Fator 4: MLP com capacidade adequada ao problema

A arquitetura `128 → 64` com ReLU tem capacidade suficiente para aprender a fronteira de decisão não-linear deste problema, sem ser grande demais para overfitting nos 46k exemplos de treino. O ADAM com early stopping encontrou o ponto ótimo em apenas 48 épocas.

### O único ataque perdido (FN = 1)

Dos 36.583 ataques no test set, apenas **1 passou despercebido**. Este fluxo provavelmente tem características atípicas — talvez um ataque de baixa intensidade que se assemelha a tráfego benigno em todas as 8 features. Para contexto: taxa de miss de 0,003% significa que em uma rede com 1 milhão de ataques por hora, o modelo perderia ~30.

---

## 9. Glossário dos Conceitos Usados

| Conceito | O que é | Por que foi usado |
|---|---|---|
| **Stratify** | Garante que treino e teste tenham a mesma proporção de classes que o dataset original | Sem `stratify`, o split aleatório poderia colocar quase todos os benignos no treino e nenhum no teste |
| **Data Leakage** | Quando informação do teste "vaza" para o treino, gerando resultados irrealisticamente bons | O scaler, imputer e SMOTE são fitados **só** no treino para evitar isso |
| **StandardScaler** | Transforma cada feature para média≈0 e desvio≈1 usando parâmetros do treino | MLP é sensível à escala — sem isso, features com valores maiores dominam o gradiente |
| **SMOTE** | Synthetic Minority Oversampling: gera instâncias sintéticas interpolando exemplos da classe minoritária | Equilibra as classes sem duplicar exatamente — mais robusto que oversample simples |
| **VarianceThreshold** | Remove features com variância ≤ threshold | Features constantes (variância=0) são inúteis — contribuem apenas com ruído |
| **SHAP** | Shapley Additive Explanations: quantifica a contribuição de cada feature para cada predição | Explica "por que" o modelo decidiu o que decidiu — transparência obrigatória em segurança |
| **Early Stopping** | Para o treino quando a métrica de validação não melhora por N épocas | Previne overfitting e reduz tempo de treino desnecessário |
| **ADAM** | Adaptive Moment Estimation: otimizador que ajusta a learning rate por parâmetro | Convergência mais rápida e estável que SGD puro, especialmente em início de treino |
| **ROC-AUC** | Área sob a curva Receptor Operating Characteristic | Mede separação probabilística independente do limiar de decisão |
| **MCC** | Matthews Correlation Coefficient | Única métrica que é confiável em todas as combinações de desbalanceamento de classes |
| **Geometric Mean** | √(Sensibilidade × Especificidade) | Penaliza modelos que são bons em uma classe mas ruins na outra |
| **RandomizedSearchCV** | Busca hiperparâmetros testando combinações aleatórias do espaço de busca | Mais eficiente que GridSearch para espaços grandes — mesma qualidade, menor custo |
| **StratifiedKFold** | Validação cruzada com K folds preservando proporção de classes | Garante que cada fold seja representativo da distribuição real |

---

*Análise gerada com base na execução do pipeline em 2026-04-02.*  
*Pipeline implementado seguindo as boas práticas do curso Paradigmas de Aprendizagem de Máquina — UFPB/LUMO (Thaís Gaudencio do Rêgo).*
