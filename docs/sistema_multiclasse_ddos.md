# Sistema Multiclasse de Detecção de DDoS em SDN

**Versão**: 2.0 — Classificação triclasse com HCF e IP Traceback  
**Data**: 2026-04-03  
**Projeto**: SDN Orchestrator v14.0 + Pipeline ML InSDN8  

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Classes do Problema](#3-classes-do-problema)
4. [Pilar 1 — Engenharia de Features HCF](#4-pilar-1--engenharia-de-features-hcf)
5. [Pilar 2 — Hop Count Filtering (SDN)](#5-pilar-2--hop-count-filtering-sdn)
6. [Pilar 3 — IP Traceback e Isolamento Cirúrgico](#6-pilar-3--ip-traceback-e-isolamento-cirúrgico)
7. [Pipeline ML Multiclasse](#7-pipeline-ml-multiclasse)
8. [API REST — Novos Endpoints](#8-api-rest--novos-endpoints)
9. [Como Rodar](#9-como-rodar)
10. [Como Interpretar os Resultados](#10-como-interpretar-os-resultados)
11. [Decisões Arquiteturais](#11-decisões-arquiteturais)
12. [Referências](#12-referências)

---

## 1. Visão Geral

O sistema binário original detectava apenas: **Benigno** vs **Ataque DDoS**.

O sistema multiclasse v2.0 distingue **três cenários**:

| Classe | Label | Origem | Mecanismo de resposta |
|--------|-------|--------|----------------------|
| 0 | Benigno | Tráfego legítimo | Nenhuma ação |
| 1 | Ataque Externo | Internet / IP spoofing | Block global (`/manage/ip`) |
| 2 | Zumbi Interno | Host interno comprometido | Isolamento cirúrgico (`/mitigation/isolate`) |

A distinção importa porque a **resposta de mitigação é diferente** para cada caso:
- Um ataque externo deve ser bloqueado em toda a topologia (borda de entrada).
- Um zumbi interno precisa de isolamento cirúrgico — bloquear em todos os switches prejudicaria toda a rede, quando o problema está em um único host.

---

## 2. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline ML (offline)                     │
│                                                              │
│  InSDN8 CSV → LabelEngineer → TopologyFeatureEngineer →      │
│  DataCleaner → FeatureSelector → FeatureScaler →             │
│  ClassBalancer (SMOTE) → MLP 128→64 → MulticlassEvaluator   │
│                                                              │
│  Artefatos salvos: models_multiclass/                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ modelos .joblib
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  SDN Orchestrator (online)                   │
│                                                              │
│  Monitor de Tráfego (ODL stats)                              │
│       ↓                                                      │
│  DDoSPredictor.predict_with_confidence()  ← modelos ML       │
│       ↓ classe predita                                       │
│  ┌────┴────┐                                                 │
│  │ classe 1│ → /manage/ip (block global em todos switches)   │
│  │ classe 2│ → HCFAnalyzer.classify() → IPTraceback.isolate()│
│  └─────────┘                                                 │
└─────────────────────────────────────────────────────────────┘
```

### Módulos novos adicionados

| Módulo | Caminho | Responsabilidade (SRP) |
|--------|---------|------------------------|
| `LabelEngineer` | `ml/data/label_engineer.py` | Converte labels binárias → triclasse |
| `TopologyFeatureEngineer` | `ml/features/topology_features.py` | Gera 4 features HCF |
| `MulticlassEvaluator` | `ml/evaluation/evaluator.py` | Métricas macro + por classe + CM 3×3 |
| `pipeline_multiclass.py` | `ml/pipeline_multiclass.py` | Pipeline de 15 etapas |
| `HCFAnalyzer` | `sdn/orchestrator/application/hcf.py` | Classifica fluxos por TTL |
| `IPTraceback` | `sdn/orchestrator/application/traceback.py` | Rastreia e isola hosts maliciosos |

---

## 3. Classes do Problema

### 3.1 Por que três classes?

O ataque DDoS pode ter origem muito diferente, com implicações distintas:

**Ataque Externo (classe 1)**  
- Origem: IPs externos, geralmente spoofados ou da internet
- Indicadores: muitos saltos de roteador → TTL muito degradado ao chegar ao SDN
- Exemplo: SYN flood da internet, NTP amplification
- Resposta: bloquear o IP de origem em todos os switches de borda

**Zumbi Interno (classe 2)**  
- Origem: host dentro da própria LAN/campus comprometido por botnet
- Indicadores: TTL quase intacto (1-2 hops), IP registrado no SDN
- Exemplo: computador de laboratório infectado participando de botnet
- Resposta: isolar **cirurgicamente** — bloquear só a porta daquele host

### 3.2 Heurística de Engenharia de Labels

O dataset InSDN8 original é binário (0=Benigno, 1=DDoS). Para criar as 3 classes, `LabelEngineer` usa heurísticas baseadas em features existentes:

```python
# Indicadores de ataque EXTERNO (protocolo/comportamento de flood da internet):
rule_protocol = X["Protocol"] == 0           # Protocol=0: raw IP flood
rule_burst    = (
    X["Pkt Len Std"]      <= 1e-3  &         # pacotes todos iguais (rajada uniforme)
    X["Pkt Len Var"]      <= 1e-3  &
    X["Flow Duration"]    < 500.0             # duração muito curta
)
external_mask = attack_idx & (rule_protocol | rule_burst)
internal_mask = attack_idx & ~external_mask  # resto = zumbi interno
```

**Importante**: estas heurísticas são um **proxy** baseado em comportamento de fluxo. Em produção com SDN ativo, o TTL real do cabeçalho IP (capturado via Packet-In do ODL) substitui esta estimativa e torna a classificação muito mais precisa.

---

## 4. Pilar 1 — Engenharia de Features HCF

**Módulo**: `ml/features/topology_features.py` — classe `TopologyFeatureEngineer`

Adiciona 4 novas features ao dataset:

### `ttl_estimated`
TTL estimado do pacote ao chegar ao controlador.

- **Em treinamento** (via `fit_transform`): gerado sinteticamente com distribuições por classe:
  - Classe 0 (Benigno):  N(µ=61, σ=3) — tráfego LAN normal, 1-3 hops
  - Classe 1 (Externo):  N(µ=48, σ=10) — cruzou ~16 roteadores de internet  
  - Classe 2 (Interno):  N(µ=62, σ=2) — 1-2 hops dentro da LAN

- **Em teste/produção** (via `transform`): estimado por heurística de features ou TTL real do SDN (`ttl_real` param)

### `hop_count`
```python
hop_count = (TTL_LAN_EXPECTED - ttl_estimated).clip(0, 60)
# TTL_LAN_EXPECTED = 64 (TTL padrão Linux)
```
Número estimado de roteadores cruzados. Ataques externos têm hop_count alto (~16+).

### `is_internal`
```python
is_internal = (
    (hop_count < 10)              &  # poucos saltos
    (Protocol != 0)               &  # não é flood raw
    NOT (Pkt_Len_Std <= 1e-3 AND Flow_Duration < 500)
).astype(int)  # 0 ou 1
```
Heurística binária: host parece estar dentro da LAN.

### `ttl_anomaly`
```python
ttl_anomaly = |hop_count - median(hop_count)|
```
Desvio do TTL em relação ao esperado. Alta anomalia → provável origem externa.

### Sem leakage de dados

```python
# CORRETO: labels só usadas no treino para distribuição de TTL
X_train_hcf = eng.fit_transform(X_train, y_train)  # usa y para distribuição
X_test_hcf  = eng.transform(X_test)                 # sem labels → heurística
```

O modelo aprende a partir das **features**, não das regras de geração. Em produção, o TTL real substitui a estimativa — sem dependência das heurísticas de treino.

---

## 5. Pilar 2 — Hop Count Filtering (SDN)

**Módulo**: `sdn/orchestrator/application/hcf.py` — classe `HCFAnalyzer`

Componente de análise online integrado ao SDN Orchestrator. Classifica fluxos de rede em tempo real usando o TTL real dos pacotes.

### Como funciona

```
Fluxo detectado pelo Monitor de Tráfego (ODL stats)
            ↓
    HCFAnalyzer.classify(src_ip, ttl_observed, flow_pkts_s)
            ↓
    ┌─────────────────────────────────────────────┐
    │ Regra 1: flow_pkts_s < 10.000 pps?          │
    │   → BENIGN  (confiança: 0.85)               │
    ├─────────────────────────────────────────────┤
    │ Regra 2: hop_count = TTL_inicial - TTL_obs  │
    │          hop_count ≥ 10?                    │
    │   → EXTERNAL (confiança: 0.70 + 0.02/hop)  │
    ├─────────────────────────────────────────────┤
    │ Regra 3: hop_count < 10 + alta taxa         │
    │   → INTERNAL (confiança: 0.92 se IP known) │
    └─────────────────────────────────────────────┘
```

### Classes e resultado

```python
class TrafficClass(IntEnum):
    BENIGN   = 0
    EXTERNAL = 1
    INTERNAL = 2

# Exemplo de resultado:
{
    "src_ip":        "10.0.0.5",
    "class":         2,
    "label":         "Zumbi Interno",
    "ttl_observed":  63,
    "hop_count":     1,
    "is_known_host": True,   # IP registrado no SDN
    "flow_pkts_s":   45000.0,
    "confidence":    0.92,
    "reason":        "hop_count=1 < 10 → host na LAN. Taxa alta (45000 pps). IP registrado no SDN — host interno comprometido (zumbi)."
}
```

### Integração com o state SDN

`_is_known_host()` consulta `state.ip_to_mac` para verificar se o IP foi aprendido pelo controlador:
- IP conhecido → host passou pelo SDN ao menos uma vez → definitivamente interno
- IP desconhecido em ataque de alta taxa → pode ser externo spoofado ou MAC spoofing

---

## 6. Pilar 3 — IP Traceback e Isolamento Cirúrgico

**Módulo**: `sdn/orchestrator/application/traceback.py` — classe `IPTraceback`

### 6.1 IP Traceback

Usa o grafo NetworkX da topologia SDN para rastrear o caminho do ataque.

```python
tb = IPTraceback()
result = tb.traceback("10.0.0.5")

# result.attack_path = ['openflow:3', 'openflow:1']
# result.src_switch  = 'openflow:3'
# result.src_port    = '3'  (porta onde o host está plugado)
```

**Algoritmo**:
1. Consultar `state.ip_to_mac` → encontra o MAC do host
2. Consultar `state.hosts_by_mac` → encontra switch de borda e porta
3. Executar Dijkstra (`nx.shortest_path`) do switch de borda ao destino
4. Retornar a lista ordenada de switches no caminho de ataque

O SDN tem **visão global da topologia** — sabe exatamente em qual switch e porta cada host está conectado.

### 6.2 Isolamento Cirúrgico

```python
ok = tb.isolate("10.0.0.5")
```

Instala um flow OpenFlow DROP de alta prioridade **apenas na porta de borda** do host infectado:

```
priority=65400, in_port=<porta_do_zumbi>, actions=drop
```

**Por que cirúrgico?**

```
Switch openflow:3
  ├── porta 1 → Host A (legítimo)   ← continua funcionando normalmente
  ├── porta 3 → Host B (zumbi)      ← bloqueado (flow DROP in_port=3)
  └── porta 5 → uplink              ← continua funcionando normalmente
```

Apenas o tráfego **saindo da porta 3** é dropado — nenhum outro host é afetado.

**Prioridades OpenFlow** (ordem de precedência):

| Priority | Flow | Instalado por |
|----------|------|---------------|
| 65500 | DROP global por IP | `/manage/ip` |
| **65400** | **DROP cirúrgico in_port** | **`IPTraceback.isolate()`** |
| 62000 | Reroute (congestion) | Loop de controle |
| 60000 | IPv4 Dijkstra | Loop de controle |
| 1000 | ARP → controlador | Loop de controle |
| 0 | TABLE-MISS DROP | Instalado na inicialização |

### 6.3 Liberar host

```python
tb.release("10.0.0.5")  # remove o flow de isolamento
```

Usa `ovs-ofctl del-flows` para remover o flow específico do switch de borda.

### 6.4 Diferença entre os dois mecanismos de bloqueio

| | `POST /manage/ip` | `POST /mitigation/isolate/{ip}` |
|--|---|---|
| Onde instala | Todos os switches (global) | Apenas switch de borda (cirúrgico) |
| O que bloqueia | Todo tráfego com nw_src=IP | Tráfego saindo de in_port específica |
| Caso de uso | Ataque externo / IP spoofado | Zumbi interno comprometido |
| Impacto | Amplo — bloqueia o IP em toda rede | Mínimo — só o host infectado |
| Priority | 65500 | 65400 |

---

## 7. Pipeline ML Multiclasse

**Arquivo**: `ml/pipeline_multiclass.py`

Pipeline de 15 etapas seguindo rigorosamente as boas práticas de ML (sem leakage, test set usado uma vez):

```
[1]  Configurações
[2]  Carregar dataset InSDN8
[3]  EDA inicial (observar, sem modificar)
[4]  ⚠️  LabelEngineer: binário → triclasse
[5]  ⚠️  Split estratificado 70/30 (stratify=y3)
[6]  TopologyFeatureEngineer.fit_transform(treino) / transform(teste)
[7]  DataCleaner (duplicatas, Inf→NaN, imputação) — só no treino
[8]  FeatureSelector (VarianceThreshold + SHAP) — só no treino
[9]  FeatureScaler (StandardScaler) — fit só no treino
[10] ClassBalancer (SMOTE multiclasse) — só no treino
[11] ModelTrainer: MLP 128→64 + CV
[12] MulticlassEvaluator: avaliação BASELINE no test_set
[13] HyperparameterTuner (RandomizedSearchCV no treino)
[14] MulticlassEvaluator: avaliação FINAL no test_set
[15] Salvar artefatos em models_multiclass/
```

### Etapas críticas (⚠️)

**[4] Label Engineering antes do split**: correto — `LabelEngineer` é determinístico e não ajustado ao train/test. Não há leakage porque não aprende parâmetros do treino.

**[5] Split antes das transformações**: correto — todas as transformações que têm `fit()` (Scaler, Imputer, SMOTE) são ajustadas apenas ao treino após o split.

### SMOTE Multiclasse

O `imbalanced-learn` suporta 3+ classes nativamente. SMOTE sobreamostra **todas as classes minoritárias** em relação à majoritária.

```python
balancer = ClassBalancer()  # SMOTE com random_state fixo
X_train_bal, y_bal = balancer.fit_resample(X_train_sc, y_train)
# Resultado: cada classe com contagem aproximadamente igual
```

### Artefatos gerados

```
models_multiclass/
  ├── mlp_model.joblib           # modelo MLP treinado (otimizado)
  ├── imputer.joblib             # SimpleImputer (median)
  ├── variance_filter.joblib     # VarianceThreshold
  ├── scaler.joblib              # StandardScaler
  ├── selected_features.joblib   # lista de features selecionadas
  └── hcf_engineer.joblib        # TopologyFeatureEngineer (estado fitted)
```

---

## 8. API REST — Novos Endpoints

### Detecção HCF

#### `POST /detect/classify`
Classifica um único fluxo usando Hop Count Filtering.

```json
// Request
{
  "src_ip":       "10.0.0.5",
  "ttl_observed": 63,
  "flow_pkts_s":  45000,
  "ttl_initial":  64
}

// Response
{
  "src_ip":        "10.0.0.5",
  "class":         2,
  "label":         "Zumbi Interno",
  "ttl_observed":  63,
  "hop_count":     1,
  "is_known_host": true,
  "flow_pkts_s":   45000.0,
  "confidence":    0.92,
  "reason":        "hop_count=1 < 10 → host na LAN. Taxa alta (45000 pps). IP registrado no SDN — host interno comprometido (zumbi)."
}
```

#### `POST /detect/classify/batch`
Classifica múltiplos fluxos em lote.

```json
// Request
{
  "flows": [
    {"src_ip": "10.0.0.5", "ttl_observed": 63, "flow_pkts_s": 45000},
    {"src_ip": "192.168.0.1", "ttl_observed": 48, "flow_pkts_s": 90000, "ttl_initial": 64}
  ]
}

// Response
{
  "results": [
    {"src_ip": "10.0.0.5", "class": 2, "label": "Zumbi Interno", ...},
    {"src_ip": "192.168.0.1", "class": 1, "label": "Ataque Externo", ...}
  ]
}
```

### Mitigação

#### `POST /mitigation/traceback/{ip}`
Rastreia o caminho de ataque de um IP suspeito.

```json
// GET /mitigation/traceback/10.0.0.5
// Response
{
  "src_ip":      "10.0.0.5",
  "src_mac":     "00:00:00:00:00:05",
  "src_switch":  "openflow:3",
  "src_port":    "3",
  "attack_path": ["openflow:3", "openflow:1"],
  "found":       true,
  "reason":      "Host 10.0.0.5 (00:00:00:00:00:05) conectado em openflow:3:3. Caminho: openflow:3 → openflow:1."
}
```

#### `POST /mitigation/isolate/{ip}`
Isola cirurgicamente um zumbi interno.

```json
// POST /mitigation/isolate/10.0.0.5
// Response (sucesso)
{
  "status":      "isolated",
  "ip":          "10.0.0.5",
  "mac":         "00:00:00:00:00:05",
  "switch":      "openflow:3",
  "port":        "3",
  "flow":        "priority=65400,in_port=3,actions=drop",
  "attack_path": ["openflow:3", "openflow:1"]
}

// Response (já isolado)
{"status": "already_isolated", "ip": "10.0.0.5"}

// Response (IP externo/desconhecido)
{"status": "failed", "ip": "10.0.0.5", "reason": "IP 10.0.0.5 não registrado no SDN — possível IP externo/spoofado."}
```

#### `DELETE /mitigation/isolate/{ip}`
Libera um host isolado após análise/limpeza.

```json
// DELETE /mitigation/isolate/10.0.0.5
// Response
{"status": "released", "ip": "10.0.0.5", "mac": "00:00:00:00:00:05"}
```

#### `GET /mitigation/status`
Lista todos os hosts atualmente isolados.

```json
// Response
{
  "isolated_hosts": [
    {
      "ip":     "10.0.0.5",
      "mac":    "00:00:00:00:00:05",
      "switch": "openflow:3",
      "port":   "3",
      "active": true
    }
  ]
}
```

---

## 9. Como Rodar

### 9.1 Pipeline ML (treinamento offline)

**Pré-requisito**: dataset `data/insdn8_ddos_binary_0n1d.csv` presente.

```bash
cd /home/lumo/sdn_ml_ddos_detection

# Com tuning (mais lento, ~20-30 min):
python -m ml.pipeline_multiclass

# Sem tuning (mais rápido, ~5 min):
python -m ml.pipeline_multiclass --no-tuning

# Sem EDA textual:
python -m ml.pipeline_multiclass --no-eda

# Com ID de run para rastreamento:
python -m ml.pipeline_multiclass --run-id experimento_v1

# Combinando opções:
python -m ml.pipeline_multiclass --no-tuning --run-id v1_quick
```

**Saída esperada**:
```
=================================================================
  Pipeline Multiclasse — DDoS SDN
  Classes: Benigno | Ataque Externo | Zumbi Interno
  Base   : InSDN8 + HCF (TTL/hop_count/is_internal)
=================================================================
[1/15] Configurações ...
[2/15] Carregando dataset InSDN8...
[3/15] EDA inicial...
[4/15] Engenharia de labels: binário → triclasse...
  Benigno         : 45,172 (33.9%)
  Ataque Externo  : 23,840 (17.9%)
  Zumbi Interno   : 64,244 (48.2%)
...
[15/15] Salvando artefatos em models_multiclass/...
  ✓ mlp_model.joblib
  ✓ hcf_engineer.joblib
  ...
```

### 9.2 Verificar resultados (métricas e gráficos)

```bash
# Listar todas as runs registradas:
python -m ml.utils.metrics_plotter --list

# Ver evolução das métricas ao longo das runs:
python -m ml.utils.metrics_plotter --evolve

# Comparar duas runs específicas:
python -m ml.utils.metrics_plotter --compare multi_baseline_XXXX multi_tuned_XXXX

# Dashboard completo:
python -m ml.utils.metrics_plotter --dashboard
```

Os gráficos são salvos em `outputs/`:
- `cm_MLP_Otimizado.png` — Matriz de confusão 3×3
- `per_class_bars.png` — F1/Precisão/Recall por classe
- `training_loss_curve.png` — Curva de loss durante o treino

### 9.3 Inferência em produção

```python
from ml.inference.predictor import DDoSPredictor

predictor = DDoSPredictor(models_dir="models_multiclass")

# Predição com TTL real do SDN:
result = predictor.predict_with_confidence(
    X_flow,          # DataFrame com features do fluxo
    ttl_real=[63]    # TTL capturado pelo Packet-In do ODL
)

print(result["prediction"])    # 2
print(result["label"])         # "Zumbi Interno"
print(result["confidence"])    # 0.94
```

### 9.4 SDN Orchestrator com os novos endpoints

```bash
# Iniciar o orquestrador (normal):
cd /home/lumo/sdn_ml_ddos_detection/sdn
python main.py

# A API estará em http://localhost:8000
# Documentação interativa: http://localhost:8000/docs
```

#### Testar classificação HCF:
```bash
curl -X POST http://localhost:8000/detect/classify \
  -H "Content-Type: application/json" \
  -d '{"src_ip": "10.0.0.5", "ttl_observed": 63, "flow_pkts_s": 45000}'
```

#### Rastrear origem de um ataque:
```bash
curl -X POST http://localhost:8000/mitigation/traceback/10.0.0.5
```

#### Isolar um zumbi interno:
```bash
curl -X POST http://localhost:8000/mitigation/isolate/10.0.0.5
```

#### Verificar status dos isolamentos:
```bash
curl http://localhost:8000/mitigation/status
```

#### Liberar um host após limpeza:
```bash
curl -X DELETE http://localhost:8000/mitigation/isolate/10.0.0.5
```

---

## 10. Como Interpretar os Resultados

### 10.1 Métricas multiclasse

O `MulticlassEvaluator` calcula:

| Métrica | O que mede | Bom valor |
|---------|-----------|-----------|
| **Accuracy** | % de predições corretas (geral) | > 0.90 |
| **F1 Macro** | F1 médio das 3 classes (classes minoritárias pesam igualmente) | > 0.85 |
| **Precision Macro** | Precisão média (falsos positivos) | > 0.85 |
| **Recall Macro** | Recall médio (falsos negativos) | > 0.85 |
| **MCC** | Correlação de Matthews — robusto para desbalanço | > 0.80 |
| **GM** | Média geométrica dos recalls por classe | > 0.80 |

### 10.2 Análise por classe

O relatório por classe indica onde o modelo erra mais:

```
               precision    recall  f1-score
Benigno           0.98      0.97      0.97
Ataque Externo    0.94      0.93      0.93
Zumbi Interno     0.96      0.97      0.96
```

**Erros mais comuns a observar**:
- **Externo predito como Interno**: hop_count baixo mas spoofado — grave, leva a isolamento incorreto
- **Interno predito como Externo**: host interno mas com features de flood — leva a block global desnecessário
- **Ataque predito como Benigno**: miss total — o mais crítico em produção

### 10.3 Matriz de Confusão 3×3

```
                  Predito
                Ben  Ext  Int
Verdadeiro Ben  [TP  FP1  FP2]
           Ext  [FN1  TP   FP3]
           Int  [FN2  FP4  TP ]
```

Ideal: diagonal alta, off-diagonal próxima de zero.

### 10.4 Confiança do HCFAnalyzer

A confiança indica o grau de certeza da classificação:

| Confiança | Interpretação | Ação recomendada |
|-----------|---------------|-----------------|
| ≥ 0.90 | Alta certeza | Agir automaticamente |
| 0.75–0.90 | Moderada | Agir + alertar operador |
| < 0.75 | Baixa | Alertar operador, aguardar confirmação |

---

## 11. Decisões Arquiteturais

### SOLID aplicado

**SRP (Single Responsibility)**:
- `LabelEngineer`: converte labels. Não classifica, não treina.
- `TopologyFeatureEngineer`: gera features. Não treina, não avalia.
- `HCFAnalyzer`: classifica por TTL. Não instala flows, não modifica estado.
- `IPTraceback`: rastreia e isola. Não classifica, não modifica topologia.

**OCP (Open/Closed)**:
- `MulticlassEvaluator` estende o sistema sem modificar `ModelEvaluator` binário.
- Novos tipos de ataque podem ser adicionados criando novas classes sem alterar existentes.

**DIP (Dependency Inversion)**:
- `InSDNLoader` implementa o protocolo `DataLoader` — permite trocar a fonte de dados sem alterar o pipeline.

**ISP (Interface Segregation)**:
- `HCFAnalyzer` e `IPTraceback` são módulos independentes — o SDN Orchestrator não precisa importar nenhum módulo ML.

### Por que TTL sintético no treino?

O dataset InSDN8 não tem coluna TTL. A geração sintética com distribuições realistas por classe é justificada porque:

1. **Não há circularidade**: geramos TTL a partir da label de treino → o modelo aprende da **feature** → em produção usa TTL real (não a regra de geração).
2. **Distribuições são realistas**: baseadas em comportamento real de rede (Linux TTL=64, ~16 hops internet).
3. **Em produção, TTL é real**: o `DDoSPredictor` aceita `ttl_real` que substitui totalmente a síntese.

### Por que isolamento na prioridade 65400 e não 65500?

O flow de isolamento cirúrgico (65400) é **intencionalmente abaixo** do DROP global por IP (65500):

- Se um operador já bloqueou o IP via `/manage/ip` (65500), o isolamento cirúrgico (65400) não interfere.
- Se apenas o isolamento cirúrgico está ativo (65400), ele bloqueia por porta — mais preciso.
- Ao liberar o isolamento, o DROP global (se existir) continua ativo — segurança em camadas.

---

## 12. Referências

- **HCF (Hop Count Filtering)**: Jin, C. et al. (2003). "Hop-count filtering: an effective defense against spoofed DDoS traffic." ACM CCS 2003.
- **MLP Architecture**: Mehmood et al. (2025). "Federated Learning for DDoS Detection in SDN." PLoS ONE.
- **InSDN dataset**: ElSaied, M. et al. (2021). "InSDN: A Novel SDN Intrusion Detection Dataset." IEEE Access.
- **OpenDaylight**: https://www.opendaylight.org/
- **Open vSwitch**: https://www.openvswitch.org/
- **SMOTE multiclasse**: Chawla, N.V. et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." JAIR.
- **SHAP**: Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NIPS.
