# Documentacao Tecnica do Modulo de Machine Learning

## Visao Geral

O diretorio `ml/` implementa um pipeline completo de aprendizado de maquina para classificacao multiclasse de trafego em ambiente SDN usando o dataset `InSDN_DatasetCSV`.

O problema modelado hoje e:

- `Normal`
- `Flooding`
- `Intrusao`

Mapeamento dos labels originais:

- `Normal -> Normal`
- `DoS -> Flooding`
- `DDoS -> Flooding`
- `Probe -> Intrusao`
- `BFA -> Intrusao`
- `Web-Attack -> Intrusao`
- `BOTNET -> Intrusao`
- `U2R -> Intrusao`

O modelo principal e um `MLPClassifier` da `scikit-learn`.

O pipeline foi implementado respeitando as boas praticas descritas em `docs/guia_boas_praticas_ml.md`, especialmente:

- `split` treino/teste antes de qualquer transformacao que aprenda dos dados;
- `fit` de imputacao, escalonamento e selecao apenas no treino;
- `SMOTE` apenas no treino;
- avaliacao final no teste sem realimentacao para o treinamento.

---

## Objetivo do Pacote `ml`

O modulo `ml` tem quatro responsabilidades principais:

1. Carregar e consolidar o dataset bruto.
2. Preparar os dados para treinamento de maneira metodologicamente correta.
3. Treinar, validar e avaliar o modelo.
4. Salvar o pipeline treinado e disponibilizar inferencia posterior.

Em outras palavras, ele transforma arquivos CSV do InSDN em:

- um classificador treinado;
- artefatos de preprocessamento reutilizaveis;
- metricas;
- graficos de interpretacao e diagnostico;
- um mecanismo de inferencia em novos fluxos.

---

## Estrutura de Diretorios

Arquivos relevantes do pacote:

```text
ml/
├── config.py
├── pipeline.py
├── data/
│   └── loader.py
├── preprocessing/
│   ├── cleaner.py
│   ├── scaler.py
│   └── balancer.py
├── features/
│   └── selector.py
├── models/
│   └── mlp_model.py
├── training/
│   ├── trainer.py
│   └── tuner.py
├── evaluation/
│   └── evaluator.py
├── persistence/
│   └── model_io.py
├── inference/
│   └── predictor.py
└── utils/
    ├── metrics_logger.py
    ├── metrics_plotter.py
    └── training_diagnostics.py
```

---

## Arquitetura Geral

O pipeline segue uma arquitetura em camadas:

1. `config.py`
   Centraliza configuracoes, caminhos, mapeamentos de classes, features selecionadas por dominio e hiperparametros.

2. `data/loader.py`
   Le os CSVs, concatena o dataset, normaliza os labels e entrega `X` e `y`.

3. `preprocessing/`
   Limpa os dados, trata ruido numerico, imputacao, escalonamento e balanceamento.

4. `features/selector.py`
   Faz filtragem por variancia e calcula importancia por SHAP ou fallback com RandomForest.

5. `models/mlp_model.py`
   Define a arquitetura do MLP baseline.

6. `training/`
   Faz treinamento, validacao cruzada e tuning opcional.

7. `evaluation/evaluator.py`
   Calcula metricas multiclasse e gera matriz de confusao.

8. `persistence/model_io.py`
   Salva e carrega o modelo e os transformadores.

9. `inference/predictor.py`
   Reaplica o pipeline salvo em dados novos.

10. `utils/`
    Registra historico de metricas, plota dashboards e gera diagnosticos de overfitting.

---

## Fluxo de Execucao do Pipeline

O arquivo principal e `ml/pipeline.py`.

A funcao `run_pipeline()` coordena todas as etapas:

1. Inicializacao e reproducibilidade.
2. Carregamento do dataset consolidado.
3. EDA opcional.
4. `train_test_split` estratificado.
5. Limpeza do treino.
6. Aplicacao da mesma limpeza ao teste.
7. Selecao de features.
8. Escalonamento.
9. Balanceamento com `SMOTE`.
10. Treinamento do MLP baseline.
11. Validacao cruzada no treino.
12. Avaliacao em treino e teste.
13. Diagnosticos de generalizacao.
14. Tuning opcional.
15. Persistencia dos artefatos.
16. Registro de metricas e outputs.

Essa ordem nao e acidental: ela existe para evitar `data leakage`.

---

## Configuracoes Centrais

Arquivo: [config.py](/home/jv/sdn_ml_ddos_detection/ml/config.py)

As configuracoes mais importantes sao:

- `DATASET_DIR`
  Diretorio com os CSVs do InSDN.

- `CLASS_GROUP_MAPPING`
  Dicionario que converte labels originais para o cenario `Normal/Flooding/Intrusao`.

- `TARGET_NAMES`
  Ordem oficial das classes:
  `["Normal", "Flooding", "Intrusao"]`

- `TARGET_ENCODING` e `TARGET_DECODING`
  Mapeamento entre labels textuais e codigos inteiros.

- `RELEVANT_FEATURES`
  Lista fixa das 26 features usadas no treinamento.

- `NON_NEGATIVE_FEATURES`
  Lista de features que, por definicao, nao deveriam assumir valores negativos.

- `TEST_SIZE`
  Fracao reservada ao conjunto de teste.

- `MLP_*`
  Hiperparametros do modelo baseline.

- `CV_N_SPLITS`
  Numero de dobras na validacao cruzada.

- `TUNING_PARAM_DISTRIBUTIONS`
  Espaco de busca do `RandomizedSearchCV`.

### Decisao de projeto importante

As features nao sao escolhidas automaticamente a partir de todas as 84 colunas do dataset. O projeto usa uma lista curada em `RELEVANT_FEATURES` para evitar que o modelo aprenda atalhos indevidos, como:

- IP de origem/destino;
- portas especificas do ambiente;
- `Flow ID`;
- `Timestamp`.

Isso reduz risco de memorizacao do cenario e melhora a capacidade de generalizacao.

---

## Camada de Dados

Arquivo: [loader.py](/home/jv/sdn_ml_ddos_detection/ml/data/loader.py)

### O que faz

O `InSDNLoader`:

- verifica se o diretorio do dataset existe;
- lista todos os CSVs;
- le cada arquivo com `pandas`;
- concatena os `DataFrame`s;
- normaliza `Label` com `str.strip()`;
- aplica o agrupamento de classes;
- seleciona as 26 features relevantes;
- devolve:
  - `X` como `DataFrame`;
  - `y` como `Series` codificada em inteiros.

### O que entra

- diretorio `dataset/InSDN_DatasetCSV`

### O que sai

- `X`: matriz de atributos
- `y`: vetor alvo

### Detalhe tecnico importante

O loader cria uma coluna auxiliar `__row_hash__` para representar de forma estavel cada linha original. Isso e usado no `DataCleaner` para que a remocao de duplicatas seja feita sobre a linha real do dataset, e nao apenas sobre as features selecionadas.

Sem isso, dois fluxos diferentes com a mesma assinatura estatistica poderiam ser removidos indevidamente.

### EDA textual

O metodo `describe()` produz:

- shape;
- memoria;
- tipos de dados;
- distribuicao do alvo;
- valores ausentes;
- duplicatas;
- estatisticas descritivas.

---

## Camada de Preprocessamento

## 1. Limpeza

Arquivo: [cleaner.py](/home/jv/sdn_ml_ddos_detection/ml/preprocessing/cleaner.py)

### Responsabilidades

- remover duplicatas reais;
- substituir `Inf` e `-Inf` por `NaN`;
- converter valores negativos fisicamente invalidos em `NaN`;
- imputar `NaN` com `SimpleImputer(strategy="median")`.

### Como funciona

#### `fit_transform(X, y)`

- usado apenas no treino;
- remove duplicatas;
- sanitiza ruido numerico;
- ajusta o imputador;
- retorna `X_clean` e `y_clean`.

#### `transform(X, y=None)`

- usado no teste ou inferencia;
- reaplica as mesmas regras de limpeza;
- usa apenas `transform()` no imputador ja ajustado.

### Por que mediana?

A mediana e robusta a outliers, o que e desejavel em dados de rede, onde taxas e duracoes podem ter distribuicoes muito assimetricas.

### Regra metodologica

O imputador nunca aprende nada do teste. Ele so e ajustado no treino.

---

## 2. Escalonamento

Arquivo: [scaler.py](/home/jv/sdn_ml_ddos_detection/ml/preprocessing/scaler.py)

### Responsabilidade

Aplicar `StandardScaler`.

### Como funciona

- `fit_transform(X_train)`
  ajusta o `scaler` e transforma o treino;

- `transform(X_test)`
  aplica os parametros aprendidos no treino ao teste.

### Por que isso importa?

O `MLPClassifier` e sensivel a escala dos atributos. Se uma feature tem amplitude muito maior que outra, ela pode dominar os gradientes e prejudicar a convergencia.

O `StandardScaler` coloca os dados aproximadamente em:

- media `0`
- desvio padrao `1`

---

## 3. Balanceamento

Arquivo: [balancer.py](/home/jv/sdn_ml_ddos_detection/ml/preprocessing/balancer.py)

### Responsabilidade

Balancear as classes com `SMOTE`.

### Como funciona

- recebe `X_train_scaled` e `y_train`;
- imprime a distribuicao antes;
- aplica `SMOTE`;
- imprime a distribuicao depois.

### Regra metodologica

O `SMOTE` e aplicado **somente no treino**.

Motivo:

- ele gera amostras sinteticas;
- o conjunto de teste deve representar o mundo real e nao dados artificiais.

---

## Camada de Selecao de Features

Arquivo: [selector.py](/home/jv/sdn_ml_ddos_detection/ml/features/selector.py)

### Etapas implementadas

1. `VarianceThreshold`
2. importancia por SHAP

### Como funciona

#### Passo 1. VarianceThreshold

Remove features com variancia menor ou igual ao limiar configurado.

Na configuracao atual, o threshold e `0.0`, entao ele remove apenas colunas constantes.

#### Passo 2. SHAP

O seletor:

- treina um `RandomForestClassifier` auxiliar;
- calcula `shap_values` sobre uma amostra do treino;
- reduz o SHAP multiclasse para uma importancia media absoluta por feature;
- produz um ranking ordenado.

### Fallback

Se a biblioteca `shap` nao estiver disponivel, o codigo usa `feature_importances_` do `RandomForestClassifier`.

### O que este modulo entrega

- conjunto filtrado de features;
- lista de features selecionadas;
- ranking de importancia;
- grafico salvo em `outputs/feature_importance_multiclass.png`.

### Decisao de projeto

Mesmo usando uma lista curada de 26 atributos, o SHAP foi mantido para:

- interpretabilidade;
- auditoria;
- verificacao da intuicao de dominio.

---

## Definicao do Modelo

Arquivo: [mlp_model.py](/home/jv/sdn_ml_ddos_detection/ml/models/mlp_model.py)

### Modelo baseline

Funcao:

- `build_baseline_mlp()`

Configuracao principal:

- `hidden_layer_sizes=(128, 64)`
- `activation="relu"`
- `solver="adam"`
- `alpha=0.001`
- `max_iter=250`
- `early_stopping=True`
- `validation_fraction=0.1`

### Modelo para tuning

Funcao:

- `build_mlp_from_params(params)`

Permite reconstruir um MLP com hiperparametros fornecidos pelo tuning.

### Observacao

Como o problema tem tres classes, o `MLPClassifier` usa internamente uma saida multiclasse adequada, sem que isso precise ser programado manualmente.

---

## Camada de Treinamento

## 1. Treino baseline

Arquivo: [trainer.py](/home/jv/sdn_ml_ddos_detection/ml/training/trainer.py)

### Responsabilidades

- treinar o baseline;
- executar validacao cruzada no treino;
- gerar curva de loss.

### `train()`

Recebe:

- `X_train_bal`
- `y_train_bal`

Executa:

- `model.fit()`
- registro de epocas executadas;
- registro da `loss_`;
- plot da curva de convergencia.

### `cross_validate()`

Executa `cross_validate` com `StratifiedKFold`.

Metricas atuais:

- `accuracy`
- `balanced_accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`

Essas metricas sao mais adequadas ao problema multiclasse e desbalanceado do que apenas acuracia.

---

## 2. Tuning

Arquivo: [tuner.py](/home/jv/sdn_ml_ddos_detection/ml/training/tuner.py)

### Responsabilidade

Executar busca de hiperparametros com `RandomizedSearchCV`.

### Como funciona

- usa `StratifiedKFold`;
- usa o scoring configurado em `CV_SCORING`;
- testa combinacoes do espaco definido em `config.py`;
- retorna `best_estimator_`.

### Parametros ajustados

Atualmente a busca cobre, entre outros:

- `hidden_layer_sizes`
- `alpha`
- `learning_rate_init`
- `learning_rate`
- `max_iter`

### Regra metodologica

O tuning usa apenas o treino balanceado.

O teste continua reservado para avaliacao final.

---

## Camada de Avaliacao

Arquivo: [evaluator.py](/home/jv/sdn_ml_ddos_detection/ml/evaluation/evaluator.py)

### O que faz

O `ModelEvaluator` calcula metricas multiclasse e gera a matriz de confusao.

### Metricas implementadas

- `accuracy`
- `balanced_accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `f1_weighted`
- `mcc`
- `gm` (`geometric_mean_score`)
- `roc_auc_ovr_macro`

### Por que essas metricas?

#### Accuracy

Mede acerto global.

#### Balanced Accuracy

Mede o desempenho medio por classe, compensando desbalanceamento.

#### F1 Macro

Trata todas as classes com o mesmo peso.

#### F1 Weighted

Leva em conta o tamanho das classes.

#### MCC

Mede qualidade global da classificacao de forma rigorosa.

#### G-Mean

Resume o equilibrio entre desempenhos de classes.

#### ROC-AUC OVR Macro

Avalia separabilidade multiclasse em esquema `one-vs-rest`.

### Outputs

- resumo textual;
- `classification_report`;
- matriz de confusao em `outputs/confusion_matrix_*.png`.

---

## Diagnostico de Overfitting

Arquivo: [training_diagnostics.py](/home/jv/sdn_ml_ddos_detection/ml/utils/training_diagnostics.py)

### Ferramentas disponiveis

#### `plot_learning_curve()`

Gera curva de aprendizado a partir de:

- diferentes tamanhos de treino;
- validacao cruzada estratificada;
- metrica principal, atualmente `f1_macro`.

Serve para verificar:

- se o modelo melhora com mais dados;
- se treino e validacao convergem;
- se existe gap grande entre ambos.

#### `plot_generalization_gap()`

Compara treino e teste lado a lado para:

- `Accuracy`
- `Balanced Accuracy`
- `F1 Macro`
- `F1 Weighted`
- `MCC`
- `ROC-AUC`

Serve para identificar sobreajuste.

#### `save_gap_report()`

Salva um JSON numerico com os gaps entre treino e teste.

### Interpretacao tecnica

Se os gaps forem pequenos, o modelo generaliza bem.

Se treino estiver muito acima do teste, ha suspeita de overfitting.

---

## Persistencia dos Artefatos

Arquivo: [model_io.py](/home/jv/sdn_ml_ddos_detection/ml/persistence/model_io.py)

### O que e salvo

- `mlp_ddos_insdn.joblib`
- `imputer.joblib`
- `variance_filter.joblib`
- `scaler.joblib`
- `selected_features.joblib`

### Por que salvar tudo?

Porque o modelo sozinho nao basta. Em producao, os novos dados precisam passar pelas **mesmas** transformacoes aprendidas no treino, na mesma ordem.

### Estrutura

O dataclass `PipelineArtifacts` encapsula:

- `model`
- `imputer`
- `variance_filter`
- `scaler`
- `selected_features`

Isso permite reproduzir o pipeline de inferencia de forma deterministica.

---

## Camada de Inferencia

Arquivo: [predictor.py](/home/jv/sdn_ml_ddos_detection/ml/inference/predictor.py)

### Observacao importante

O nome da classe ainda e `DDoSPredictor` por compatibilidade com o historico do projeto, mas ela hoje opera no cenario multiclasse.

### Fluxo de inferencia

1. `load()`
   carrega todos os artefatos salvos.

2. `_preprocess(X)`
   reaplica:
   - imputacao;
   - `VarianceThreshold`;
   - selecao das features finais;
   - escalonamento.

3. `predict(X)`
   retorna as classes codificadas.

4. `predict_labels(X)`
   converte os codigos para:
   - `Normal`
   - `Flooding`
   - `Intrusao`

5. `predict_proba(X)`
   retorna probabilidades por classe.

6. `predict_with_confidence(X)`
   devolve:
   - predicao inteira;
   - label textual;
   - confianca maxima.

### Requisito de entrada

O `DataFrame` de inferencia deve conter as mesmas colunas brutas do treino, sem a coluna alvo.

---

## Registro e Visualizacao de Experimentos

## 1. Historico de metricas

Arquivo: [metrics_logger.py](/home/jv/sdn_ml_ddos_detection/ml/utils/metrics_logger.py)

### O que faz

Registra cada run em:

- `outputs/metrics_history.json`
- `outputs/metrics_history.csv`

### O que e armazenado

- `run_id`
- `timestamp`
- label da avaliacao
- nomes das classes
- metricas agregadas
- matriz de confusao
- hiperparametros
- informacoes do dataset
- observacoes sobre gap treino/teste

Isso permite comparar experimentos ao longo do tempo.

## 2. Plotter de historico

Arquivo: [metrics_plotter.py](/home/jv/sdn_ml_ddos_detection/ml/utils/metrics_plotter.py)

### Graficos suportados

- evolucao das metricas por run;
- comparacao entre duas runs;
- radar chart;
- heatmap da matriz de confusao;
- dashboard consolidado.

---

## Explicacao do `pipeline.py`

Arquivo: [pipeline.py](/home/jv/sdn_ml_ddos_detection/ml/pipeline.py)

A funcao `run_pipeline()` e o orquestrador do sistema.

### Parametros de execucao

- `run_tuning`
  ativa ou desativa tuning.

- `run_eda`
  ativa ou desativa EDA textual.

- `verbose`
  reservado para controle de verbosidade.

- `run_id`
  identificador da execucao para o historico.

- `sample_size`
  permite amostra estratificada para experimentos rapidos.

### Sequencia interna

#### 1. Inicializacao

- configura `numpy.random.seed`;
- configura exibicao do `pandas`;
- garante que `outputs/` existe.

#### 2. Carregamento

- instancia `InSDNLoader`;
- le o dataset;
- opcionalmente executa `describe()`.

#### 3. Split

- usa `train_test_split`;
- `stratify=y`;
- preserva a distribuicao de classes.

#### 4. Limpeza

- instancia `DataCleaner`;
- limpa o treino com `fit_transform`;
- limpa o teste com `transform`.

#### 5. Selecao de features

- instancia `FeatureSelector`;
- gera ranking de importancia;
- produz `X_train_sel` e `X_test_sel`.

#### 6. Escalonamento

- instancia `FeatureScaler`;
- ajusta no treino;
- reaplica no teste.

#### 7. Balanceamento

- instancia `ClassBalancer`;
- aplica `SMOTE` apenas no treino.

#### 8. Treinamento baseline

- instancia `ModelTrainer`;
- treina o baseline;
- executa validacao cruzada.

#### 9. Avaliacao

- instancia `ModelEvaluator`;
- avalia em treino;
- avalia em teste.

#### 10. Diagnostico

- instancia `TrainingDiagnostics`;
- gera:
  - `learning_curve`;
  - `generalization_gap`;
  - `generalization_report`.

#### 11. Tuning opcional

- instancia `HyperparameterTuner`;
- busca melhor configuracao;
- reavalia treino e teste.

#### 12. Persistencia

- monta `PipelineArtifacts`;
- salva com `ModelIO`.

#### 13. Historico

- registra resultados com `MetricsLogger`.

---

## Features Utilizadas no Treinamento

As features atuais sao:

- `Protocol`
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`
- `Flow Byts/s`
- `Flow Pkts/s`
- `Flow IAT Mean`
- `Flow IAT Std`
- `Flow IAT Max`
- `Flow IAT Min`
- `Bwd IAT Tot`
- `Bwd IAT Mean`
- `Bwd IAT Std`
- `Bwd IAT Max`
- `Bwd IAT Min`
- `Bwd Pkts/s`
- `Pkt Len Mean`
- `Pkt Len Std`
- `Pkt Len Var`
- `Down/Up Ratio`
- `SYN Flag Cnt`
- `ACK Flag Cnt`
- `Active Mean`
- `Idle Mean`

### Racional de escolha

Essas features foram escolhidas por serem:

- estatisticas de fluxo;
- relativamente estaveis entre ambientes;
- coerentes com deteccao de ataque por comportamento;
- menos propensas a memorizar o cenario especifico do laboratorio.

---

## Entradas, Saidas e Artefatos

## Entradas

- CSVs do InSDN em `dataset/InSDN_DatasetCSV/`

## Saidas principais

Em `models/`:

- modelo treinado
- imputer
- filtro de variancia
- scaler
- lista de features selecionadas

Em `outputs/`:

- curvas de loss
- curva de aprendizado
- gap de generalizacao
- matrizes de confusao
- importancia de features
- historico de metricas

---

## Como Executar

## Experimento rapido

```bash
python3 -m ml.pipeline --no-tuning --no-eda --sample-size 12000 --run-id experimento_rapido
```

## Baseline no dataset completo

```bash
python3 -m ml.pipeline --no-tuning --run-id baseline_full
```

## Com tuning

```bash
python3 -m ml.pipeline --run-id tuned_full
```

## Visualizacao do historico

```bash
python3 -m ml.utils.metrics_plotter --list
python3 -m ml.utils.metrics_plotter --dashboard
python3 -m ml.utils.metrics_plotter --compare baseline_full tuned_full
```

---

## Como Interpretar os Resultados

### `feature_importance_multiclass.png`

Mostra quais atributos mais influenciaram o modelo.

### `loss_curve_baseline.png`

Mostra convergencia do MLP ao longo das epocas.

### `learning_curve_baseline.png`

Mostra como o desempenho evolui com mais dados.

### `generalization_gap_baseline.png`

Compara treino e teste para detectar overfitting.

### `generalization_report_baseline.json`

Versiona numericamente o gap entre treino e teste.

### `confusion_matrix_*.png`

Mostra exatamente onde o modelo confunde as classes.

### `metrics_history.json` e `metrics_history.csv`

Guardam o historico dos experimentos.

---

## Garantias Metodologicas do Projeto

O codigo atual garante:

- reproducibilidade via `RANDOM_STATE`;
- `split` antes do preprocessamento;
- imputacao aprendida apenas no treino;
- escalonamento aprendido apenas no treino;
- selecao de features baseada apenas no treino;
- `SMOTE` apenas no treino;
- avaliacao final isolada no teste;
- persistencia de todos os transformadores necessarios para inferencia.

Essas garantias sao o nucleo da confiabilidade metodologica do pacote.

---

## Limitacoes Atuais

Mesmo estando bem implementado, o sistema tem limitacoes naturais:

- o baseline usa uma arquitetura fixa de MLP;
- a qualidade do agrupamento `Intrusao` depende da coerencia entre tipos de ataque diferentes;
- os resultados de amostras rapidas nao substituem validacao completa no dataset inteiro;
- o nome `DDoSPredictor` ainda nao reflete o escopo multiclasse atual;
- alguns arquivos antigos de `outputs/` podem coexistir com novos graficos, dependendo do historico do diretorio.

---

## Melhorias Futuras Recomendadas

- renomear `DDoSPredictor` para algo neutro, como `TrafficClassifierPredictor`;
- implementar testes automatizados unitarios e de integracao para o pipeline;
- adicionar validacao por grupo/cenario, se desejado;
- comparar o MLP com outros baselines tabulares;
- executar tuning completo no dataset integral;
- separar melhor subtipos de `Intrusao` em experimentos posteriores.

---

## Conclusao

O modulo `ml/` implementa um pipeline de ML tabular completo, reprodutivel e metodologicamente correto para classificar trafego SDN em `Normal`, `Flooding` e `Intrusao`.

Ele foi projetado para:

- minimizar vazamento de dados;
- reduzir risco de memorizacao do ambiente;
- preservar interpretabilidade;
- permitir inferencia com os mesmos transformadores do treino;
- oferecer diagnosticos objetivos de generalizacao e overfitting.

Do ponto de vista tecnico, a implementacao esta organizada por responsabilidades, com modulos especializados para cada etapa do fluxo.
