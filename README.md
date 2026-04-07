# SDN ML DDoS Detection

Este projeto junta duas frentes que se complementam:

- um orquestrador SDN para gerenciar uma topologia com Open vSwitch e OpenDaylight;
- um pipeline de Machine Learning para classificar fluxos de rede como `Normal`, `Flooding` ou `Intrusao`.

Na prática, a ideia é ter uma base de rede programável e observável do lado `sdn/` e um módulo analítico do lado `ml/` que aprende, avalia e reaplica um classificador sobre estatísticas de fluxo.

## O que o projeto se propõe a fazer

O lado SDN cuida da operação da rede:

- descobre topologia e hosts;
- monitora carga dos enlaces;
- instala flows OpenFlow de forma proativa;
- expõe uma API REST para inspeção e gerenciamento.

O lado ML cuida da análise do tráfego:

- consolida o dataset `InSDN_DatasetCSV`;
- remapeia os rótulos originais para o cenário multiclasse do projeto;
- prepara os dados com limpeza, seleção de atributos, escalonamento e balanceamento;
- treina um `MLPClassifier`;
- salva artefatos para inferência futura;
- gera gráficos e relatórios de interpretação e diagnóstico.

## Como o repositório se organiza

```text
.
├── cientific_base/          # Base de artigos e materiais de apoio
├── dataset/                 # CSVs do InSDN usados no treinamento
├── docs/                    # Documentação principal do módulo ML e visão geral do problema
├── ml/                      # Pipeline de treinamento, avaliação e inferência
├── models/                  # Artefatos treinados salvos pelo pipeline
├── outputs/                 # Métricas, gráficos e histórico de experimentos
├── sdn/                     # Orquestrador SDN, API REST e utilitários
├── requirements_ml.txt      # Dependências do módulo de ML
└── requirements_sdn.txt     # Dependências do módulo SDN
```

## Estrutura principal do SDN

O pacote `sdn/` segue uma separação bem clara entre domínio, casos de uso, infraestrutura e API:

```text
sdn/
├── sdn_orchestrator.py
├── orchestrator/
│   ├── config.py
│   ├── main.py
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   ├── presentation/
│   └── utils/
├── tests/
└── docs/
```

Resumo rápido dos papéis:

- `domain/`: estado compartilhado da rede e modelos da API;
- `application/`: topologia, hosts, tráfego e roteamento;
- `infrastructure/`: integração com Docker, OVS e construção de flows;
- `presentation/`: endpoints FastAPI;
- `utils/`: métricas e ferramentas de verificação.

## Estrutura principal do ML

O pacote `ml/` organiza o pipeline em blocos pequenos e reutilizáveis:

```text
ml/
├── config.py
├── pipeline.py
├── data/
├── preprocessing/
├── features/
├── models/
├── training/
├── evaluation/
├── persistence/
├── inference/
└── utils/
```

Resumo rápido dos papéis:

- `data/`: leitura e consolidação dos CSVs;
- `preprocessing/`: limpeza, imputação, escalonamento e balanceamento;
- `features/`: filtro por variância e importância com SHAP/RandomForest auxiliar;
- `models/`: definição do MLP baseline;
- `training/`: treino, validação cruzada e tuning;
- `evaluation/`: métricas e matriz de confusão;
- `persistence/`: salvar e recarregar artefatos;
- `inference/`: reaplicar o pipeline treinado em novos dados;
- `utils/`: histórico de métricas, plots comparativos e diagnósticos.

## Fluxo geral do módulo de ML

O pipeline principal está em `ml/pipeline.py` e segue esta ordem:

1. carregar o dataset consolidado;
2. fazer `train_test_split` estratificado antes de qualquer transformação que aprenda dos dados;
3. limpar o treino e reaplicar a mesma limpeza no teste;
4. selecionar features por `VarianceThreshold` + importância com `SHAP` ou fallback em `RandomForest`;
5. escalonar apenas as variáveis contínuas;
6. aplicar `SMOTE` somente no treino;
7. treinar o `MLPClassifier`;
8. rodar validação cruzada e curva de aprendizado com preprocessamento refeito dentro de cada dobra;
9. avaliar treino e teste;
10. salvar modelo, transformadores, métricas e gráficos.

As classes usadas hoje são:

- `0 = Normal`
- `1 = Flooding`
- `2 = Intrusao`

Os rótulos originais do InSDN são agrupados assim:

- `Normal -> Normal`
- `DoS` e `DDoS -> Flooding`
- `Probe`, `BFA`, `Web-Attack`, `BOTNET` e `U2R -> Intrusao`

## Requisitos

Crie e ative um ambiente virtual antes de rodar qualquer parte do projeto.

Para ML:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_ml.txt
```

Para SDN:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_sdn.txt
```

## Como executar o pipeline de ML

Experimento rápido:

```bash
python3 -m ml.pipeline --no-tuning --no-eda --sample-size 12000 --run-id experimento_rapido
```

Baseline completo:

```bash
python3 -m ml.pipeline --no-tuning --run-id baseline_full
```

Execução com tuning:

```bash
python3 -m ml.pipeline --run-id tuned_full
```

Visualizar o histórico de experimentos:

```bash
python3 -m ml.utils.metrics_plotter --list
python3 -m ml.utils.metrics_plotter --dashboard
python3 -m ml.utils.metrics_plotter --compare baseline_full tuned_full
```

## Como executar o orquestrador SDN

Com o ambiente da topologia já pronto e o OpenDaylight acessível:

```bash
cd sdn
python3 sdn_orchestrator.py
```

No boot, o orquestrador:

- testa conectividade com o controlador ODL;
- descobre os containers OVS;
- valida switches disponíveis;
- inicia a coleta de métricas;
- sobe o loop de controle periódico;
- expõe a API FastAPI na porta `8000`.

## Onde ficam os resultados

Arquivos gerados pelo ML:

- `models/`: modelo treinado e artefatos de preprocessamento;
- `outputs/metrics_history.json`: histórico consolidado das runs;
- `outputs/metrics_history.csv`: versão tabular do histórico;
- `outputs/runs/<run_id>/`: gráficos e relatórios específicos de cada execução.

Arquivos do SDN:

- métricas e utilitários ficam concentrados em `sdn/orchestrator/utils/`;
- documentação operacional e de arquitetura fica em `sdn/docs/`.

## Documentação mais importante

Se você quiser se orientar mais rápido pelo projeto, estes arquivos são os melhores pontos de partida:

- `docs/documentacao_tecnica_ml.md`
- `docs/guia_boas_praticas_ml.md`
- `docs/panorama_completo_problema_ml_sdn.md`
- `docs/sdn_documentacao.md`
- `sdn/docs/guia-startup-sdn.md`

## Observações importantes

- O pipeline de ML foi organizado para evitar `data leakage`.
- `SMOTE` é aplicado apenas no treino.
- Features binárias são preservadas sem padronização.
- Validação cruzada e curva de aprendizado refazem o preprocessamento dentro de cada dobra.
- Cada run do ML salva seus gráficos em uma pasta própria para manter rastreabilidade.
