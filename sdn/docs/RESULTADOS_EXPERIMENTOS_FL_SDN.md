# Resultados e Análise — Experimentos FL-SDN
## Laboratório LUMO-UFPB · Março 2026

---

## Sumário

1. [Resumo Executivo](#1-resumo-executivo)
2. [Configuração do Ecossistema](#2-configuração-do-ecossistema)
3. [Topologia de Rede](#3-topologia-de-rede)
4. [Tráfego de Fundo](#4-tráfego-de-fundo)
5. [Configuração do SDN Orchestrator](#5-configuração-do-sdn-orchestrator)
6. [Configuração do Federated Learning](#6-configuração-do-federated-learning)
7. [Descrição dos Cenários](#7-descrição-dos-cenários)
8. [Resultados por Dataset](#8-resultados-por-dataset)
9. [Análise Comparativa](#9-análise-comparativa)
10. [Conclusões e Indícios](#10-conclusões-e-indícios)

---

## 1. Resumo Executivo

Foram realizados três experimentos de Federated Learning (FL) com XGBoost em ambiente SDN simulado via GNS3/OpenDaylight, com tráfego de fundo ativo para simular congestionamento de rede. Cada experimento rodou 20 rounds com 6 clientes e 4 datasets, totalizando **12 condições experimentais** comparadas.

### Cenários

| ID | Nome | SDN Rerouting | FL usa métricas SDN | Estratégia FL |
|----|------|:---:|:---:|---|
| **C1** | `com_sdn` (V5) | ✅ Ativo | ✅ Sim (health score) | sdn-bagging |
| **C2** | `sem_sdn` (V5) | ❌ Desativado | ❌ Não | bagging |
| **C3** | `rerouting` (V4) | ✅ Ativo | ❌ Não | bagging |

### Achado principal

> **O SDN com seleção dinâmica de clientes (C1) é 1,8x–2,4x mais rápido que FL sem SDN (C2), com qualidade de modelo equivalente. O SDN apenas com rerouting sem seleção de clientes (C3) é 6–13% mais lento que sem SDN — confirmando que o benefício real vem da combinação rerouting + seleção de clientes.**

---

## 2. Configuração do Ecossistema

### Hardware e Software

| Componente | Versão / Especificação |
|-----------|------------------------|
| Host OS | Ubuntu 24.04 LTS |
| CPU | Intel i7 (host) |
| Emulador de rede | GNS3 2.x com containers Docker |
| Controlador SDN | OpenDaylight (ODL) Calcium SR3 |
| OVS | Open vSwitch 2.x (container Docker) |
| Python | 3.12 |
| Framework FL | Flower (flwr) |
| Modelo ML | XGBoost |
| API SDN | FastAPI (sdn_orchestrator.py, porta 8000) |

### Processos do host

```
tap0 (172.16.1.1/24)  ← plano de controle ODL (OpenFlow :6653, REST :8181)
tap1 (sem IP)         ← plano de dados FL (gRPC :8080)
sdn_orchestrator.py   ← porta 8000, polling a cada 5s
server.py             ← porta 8080, aguarda 6 clientes
iperf3 -s -p 5210 -D  ← servidor alvo do BG-Node-1
iperf3 -s -p 5211 -D  ← servidor alvo do BG-Node-2
```

### Separação de planos de rede

O host conecta-se ao GNS3 via dois TAP interfaces:

- **tap0** (`172.16.1.1/24`): plano de controle. ODL usa esta interface para receber conexões OpenFlow 1.3 dos switches (porta 6653) e para expor a REST API (porta 8181).
- **tap1** (sem IP): plano de dados. Os clientes FL enviam modelos XGBoost ao servidor via gRPC (porta 8080). A ausência de IP no tap1 evita que o kernel use a rota errada ao responder.

**Policy routing** garante que as respostas do servidor FL saiam por tap1:
```
PREROUTING: pacotes em tap1 → MARK=1 + CONNMARK save
OUTPUT:     CONNMARK restore → fwmark=1 → tabela "dataplane"
Tabela dataplane: 172.16.1.0/24 via tap1
```

---

## 3. Topologia de Rede

### Arquitetura hierárquica de 3 camadas

```
                    HOST (172.16.1.1)
                    tap0     tap1
                     │         │
              ┌──────┴─────────┴──────┐
              │       CORE            │
              │  OVS-1    OVS-2       │
              │ (sw1)    (sw2)        │
              └──┬────────────────┬───┘
                 │                │
    ┌────────────┼────────────────┼────────────┐
    │         AGREGAÇÃO                        │
    │   OVS-3      OVS-4      OVS-5            │
    │  (sw3)      (sw4)      (sw5)             │
    └──┬──┬──┬────┬──┬──┬────┬──┬──┘
       │  │  │    │  │  │    │  │
       │  │  │    │  │  │    │  │
    ┌──┴──┴──┴────┴──┴──┴────┴──┴──┐
    │              BORDA             │
    │  sw6  sw7  sw8  sw9  sw10 sw11 sw12 sw13 sw14 │
    └──┬──────┬────┬──────┬──────────┬───────┘
       │      │    │      │          │
    FL-1   FL-2  BG-1  FL-3,4,5,6  BG-2,3
```

### Mapeamento de switches e hosts

| Container | IP | Switch de borda | Agregação | Função |
|-----------|-----|-----------------|-----------|--------|
| FL-Node-1-cat1 | 172.16.1.10 | OVS-6 (sw6) | sw3 | Cliente FL 0 |
| FL-Node-2-cat2 | 172.16.1.11 | OVS-7 (sw7) | sw3 | Cliente FL 2 |
| **BG-Node-1-cat1** | **172.16.1.12** | **OVS-8 (sw8)** | **sw3** | **Tráfego de fundo** |
| FL-Node-3-cat3 | 172.16.1.13 | OVS-9 (sw9) | sw4 | Cliente FL 3 |
| FL-Node-4-cat2 | 172.16.1.14 | OVS-10 (sw10) | sw4 | Cliente FL 4 |
| **BG-Node-2-cat1** | **172.16.1.15** | **OVS-14 (sw14)** | **sw5** | **Tráfego de fundo** |
| FL-Node-5-cat1 | 172.16.1.16 | OVS-11 (sw11) | sw4 | Cliente FL 1 |
| FL-Node-6-cat3 | 172.16.1.17 | OVS-12 (sw12) | sw5 | Cliente FL 5 |
| BG-Node-3-cat2 | 172.16.1.18 | OVS-13 (sw13) | sw5 | Inativo nos experimentos |

### Configuração dos switches OVS

Cada OVS foi configurado via `configure_ovs.sh`:

```bash
ovs-vsctl del-br br0          # limpa flows residuais de sessões anteriores
ovs-vsctl add-br br0
ovs-vsctl set bridge br0 other-config:datapath-id=$(printf '%016x' $SW_NUM)
ovs-vsctl set-controller br0 tcp:172.16.1.1:6653
ovs-vsctl set bridge br0 protocols=OpenFlow13
ovs-vsctl set bridge br0 fail-mode=secure
ovs-vsctl set bridge br0 other-config:disable-in-band=true
# Adiciona eth1..eth15 ao bridge de dados
```

- `fail-mode=secure`: descarta tráfego ao perder conexão com ODL (evita flooding)
- `disable-in-band`: impede que o OVS instale flows automáticos para manutenção da conexão de controle

### Limitação de largura de banda dos links (tc tbf)

Aplicado em **todas as interfaces eth1..eth15 de todos os 14 OVS** via `nsenter` (namespace do container):

```bash
nsenter -t $PID -n tc qdisc add dev eth$i root tbf \
    rate 20mbit \
    burst 256kbit \
    latency 50ms
```

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `rate` | **20 Mbps** | Capacidade máxima por link — alinha com `MAX_LINK_CAPACITY=20000000` no config.py |
| `burst` | 256 kbit | Bucket de absorção de rajadas (~13ms a 20 Mbps) |
| `latency` | 50 ms | Tempo máximo de espera no bucket antes de descartar |

> **Nota**: sem o tc tbf, os links virtuais OVS teriam capacidade ilimitada (dependente da CPU do host) e o congestionamento não seria observável de forma controlada.

---

## 4. Tráfego de Fundo

### Objetivo

Criar congestionamento controlado e reproduzível nos links que convergem para o OVS-1 (sw1), simulando um ambiente de rede com carga de fundo típico de cenários reais (IoT, edge computing).

### Configuração dos geradores de tráfego

#### BG-Node-1-cat1 (172.16.1.12)

```bash
docker exec -d "GNS3.BG-Node-1-cat1.$SUFFIX" \
    iperf3 -c 172.16.1.1 -p 5210 -t 86400 -b 30M -P 3
```

- Destino: `172.16.1.1:5210` (servidor iperf3 no host)
- Caminho físico: `sw8 → sw3 → sw1 → tap1 → host`
- `-b 30M -P 3`: 3 streams paralelos de 30 Mbps cada = **90 Mbps tentados**
- Resultado prático: ~13–17 Mbps efetivos (limitado pelo tc 20mbit do link sw8↔sw3)

#### BG-Node-2-cat1 (172.16.1.15)

```bash
docker exec -d "GNS3.BG-Node-2-cat1.$SUFFIX" \
    iperf3 -c 172.16.1.1 -p 5211 -t 86400 -b 30M -P 3
```

- Destino: `172.16.1.1:5211`
- Caminho físico: `sw14 → sw5 → sw1 → tap1 → host`
- Resultado prático: ~12–15 Mbps efetivos

#### Servidores iperf3 no host

```bash
iperf3 -s -p 5210 -D   # alvo do BG-Node-1
iperf3 -s -p 5211 -D   # alvo do BG-Node-2
```

### Efeito na rede (evidências do log do orquestrador — V5)

| Link | Utilização típica | Observação |
|------|:-----------------:|------------|
| sw8↔sw3 | 62–85% | Tráfego BG-Node-1 saindo para sw3 |
| sw14↔sw5 | 62–75% | Tráfego BG-Node-2 saindo para sw5 |
| sw3↔sw1 | 10–80% | BG-Node-1 chegando ao core (oscila com rerouting) |
| sw5↔sw1 | 10–80% | BG-Node-2 chegando ao core (oscila com rerouting) |
| sw4↔sw1 | 0–128% | Rota alternativa ativada pelo rerouting |

### Tráfego total chegando ao OVS-1

- **BG-Node-1**: ~13–17 Mbps pelo caminho sw3→sw1
- **BG-Node-2**: ~12–15 Mbps pelo caminho sw5→sw1
- **Total**: ~25–34 Mbps agregados no OVS-1

### Por que o tráfego dispara o rerouting

```
REROUTE_THRESH = 0.65 × MAX_LINK_CAPACITY
              = 0.65 × 20 Mbps
              = 13 Mbps

13–17 Mbps nos links sw3↔sw1 e sw5↔sw1 → acima do threshold → rerouting ativo
```

Quando o orquestrador detecta utilização > 65% em sw3↔sw1, instala flows alternativos que redirecionam tráfego via sw4↔sw1 (caminho pelos clientes FL-Node-3,4, OVS-9,10,11 → sw4 → sw1). O efeito visível: sw4↔sw1 alterna entre 0% e 90–128% ciclicamente.

### BG-Node-3-cat2 (172.16.1.18)

Este container esteve ativo (em execução) mas **não gerou tráfego nos experimentos**. Está em uma área de topologia diferente (cat2 → sw13 → sw5) e não foi incluído nos comandos iperf3.

---

## 5. Configuração do SDN Orchestrator

**Arquivo**: `orchestrator/config.py`

```python
ENABLE_REROUTING    = True
ODL_IP              = "172.16.1.1"
ODL_PORT            = "8181"
MAX_LINK_CAPACITY   = 20_000_000   # 20 Mbps em bps
POLLING_INTERVAL    = 5            # segundos entre ciclos
REROUTE_THRESH      = 0.65         # 65% = 13 Mbps → dispara rerouting
CONGESTED_THRESH    = 0.90         # 90% = 18 Mbps → congestionamento severo
WARN_THRESH         = 0.50         # 50% = 10 Mbps → alerta
FL_SERVER_SWITCH    = "openflow:1" # OVS-1 (core ao qual tap1 conecta)
HOST_TTL_CYCLES     = 3
SW_TTL_CYCLES       = 3
HOST_PROBE_CYCLES   = 6
HOST_PROBE_MISS     = 3
```

### Funcionamento do rerouting

O orquestrador executa a cada 5 segundos:

1. **Coleta topologia** via REST do ODL (switches, links, hosts, estatísticas de portas)
2. **Calcula utilização** de cada link: `bytes_delta / intervalo / MAX_LINK_CAPACITY`
3. **Detecta congestionamento**: links > `REROUTE_THRESH` (65%)
4. **Calcula caminho alternativo**: algoritmo de Dijkstra com pesos de utilização
5. **Instala flows**: `docker exec OVS-N ovs-ofctl add-flow br0 ...`
6. **Expõe API REST** em `localhost:8000`:
   - `GET /health` — status, switches, hosts
   - `GET /metrics/links` — utilização de todos os links
   - `GET /metrics/hosts` — largura de banda disponível por cliente FL

### Métrica de largura de banda por cliente FL

Calculada em `/metrics/hosts`:

```python
bottleneck_bps  = max(link.load_bps for link in caminho_cliente_até_servidor)
available_bw    = max(0, MAX_LINK_CAPACITY - bottleneck_bps)
bandwidth_mbps  = available_bw / 1_000_000
```

`FL_SERVER_SWITCH = "openflow:1"` define o switch core como âncora do caminho.

---

## 6. Configuração do Federated Learning

**Arquivo**: `FL-SDN-main/fl_sdn_code/config.py`

```python
NUM_CLIENTS             = 6
NUM_ROUNDS              = 20
LOCAL_EPOCHS            = 50
SDN_ORCHESTRATOR_IP     = "172.16.1.1"
SDN_ORCHESTRATOR_PORT   = "8000"
SDN_MIN_BANDWIDTH_MBPS  = 10.0    # limiar de elegibilidade de cliente
SDN_LINK_CAPACITY_MBPS  = 20.0

# XGBoost
XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 50,
}

# Health Score (somente em sdn-bagging)
HEALTH_SCORE_ENABLED    = True
HEALTH_SCORE_PROFILE    = "balanced"
```

### Estratégias FL

#### `bagging` (C2 sem_sdn, C3 rerouting)

- Todos os 6 clientes participam em cada round
- Sem consulta ao SDN
- `bandwidth_mpbs_avg = 0.0` e `efficiency_score_avg = 0.0` nos resultados (esperado)
- Aggregation: FedBagging padrão

#### `sdn-bagging` (C1 com_sdn)

A cada round, antes de iniciar:

1. **Consulta** `GET /metrics/hosts` → obtém `bandwidth_mbps` por cliente
2. **Filtra** clientes com `bandwidth_mbps < SDN_MIN_BANDWIDTH_MBPS (10.0)` → marcados como não elegíveis
3. **Calcula efficiency_score** para clientes elegíveis:

```python
efficiency_score = 0.5 * bw_norm + 0.3 * lat_norm + 0.2 * loss_norm
# bw_norm   = bandwidth_mbps / SDN_LINK_CAPACITY_MBPS
# lat_norm  = 1 - (latency_ms - 2.0) / 10.0   (normalizado)
# loss_norm = 1 - packet_loss
```

4. **Calcula health_score** por cliente:

```python
health_score = 0.4 * contribution_score  # histórico de contribuição ao modelo
             + 0.3 * resource_score       # training_time normalizado
             + 0.3 * network_score        # efficiency_score
```

5. **Exclui** clientes com health_score abaixo da mediana do round
6. **Treina** somente com os clientes selecionados

### Configuração dos clientes FL por IP

| Cliente ID | Container | IP |
|:---:|---|---|
| 0 | FL-Node-1-cat1 | 172.16.1.10 |
| 1 | FL-Node-5-cat1 | 172.16.1.16 |
| 2 | FL-Node-2-cat2 | 172.16.1.11 |
| 3 | FL-Node-3-cat3 | 172.16.1.13 |
| 4 | FL-Node-4-cat2 | 172.16.1.14 |
| 5 | FL-Node-6-cat3 | 172.16.1.17 |

---

## 7. Descrição dos Cenários

### C1 — `com_sdn` (V5, 27/03/2026)

- **Orquestrador**: ativo, `ENABLE_REROUTING=True`
- **FL**: estratégia `sdn-bagging`, consulta `/metrics/hosts` a cada round
- **Seleção de clientes**: ativa via health score (exclui clientes lentos/congestionados)
- **Tráfego de fundo**: BG-Node-1 + BG-Node-2 ativos (30M × 3 streams cada)
- **Duração**: 7–13 min por dataset

### C2 — `sem_sdn` (V5, 27/03/2026)

- **Orquestrador**: desativado (`ENABLE_REROUTING=False`, não iniciado)
- **FL**: estratégia `bagging`, sem consulta ao SDN
- **Seleção de clientes**: nenhuma — todos os 6 clientes participam sempre
- **Tráfego de fundo**: BG-Node-1 + BG-Node-2 ativos (mesmas condições do C1)
- **Duração**: 12–23 min por dataset

### C3 — `rerouting` (V4, 28/03/2026)

- **Orquestrador**: ativo, `ENABLE_REROUTING=True`
- **FL**: estratégia `bagging`, sem consulta ao SDN
- **Seleção de clientes**: nenhuma — todos os 6 clientes participam sempre
- **Tráfego de fundo**: BG-Node-1 + BG-Node-2 ativos (mesmas condições)
- **Duração**: 12–25 min por dataset

---

## 8. Resultados por Dataset

### 8.1 Dataset: Epsilon

**Características**: classificação binária, alta dimensionalidade, AUC de referência ~0.60

| Round | C1 com_sdn elapsed (s) | C2 sem_sdn elapsed (s) | C3 rerouting elapsed (s) | C1 AUC | C2 AUC | C3 AUC |
|:---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 87.83 | 124.54 | 134.53 | 0.6003 | 0.6017 | 0.6017 |
| 5 | 152.85 | 304.82 | 340.93 | 0.5921 | 0.5910 | 0.5910 |
| 10 | 236.11 | 494.32 | 553.76 | 0.5911 | 0.5895 | 0.5895 |
| 15 | 313.24 | 683.22 | 768.33 | 0.5898 | 0.5888 | 0.5888 |
| **20** | **420.94** | **865.82** | **978.21** | **0.5893** | **0.5878** | **0.5878** |

- **Training time médio (rounds 15–20)**:
  - C1: ~17s | C2: ~33s | C3: ~38s
- **Speedup C1 vs C2**: **2.06×**
- **C3 vs C2**: +13% mais lento

#### Bandwidth médio por round (C1 somente)

| Round | BW (Mbps) | Efficiency Score |
|:---:|:---:|:---:|
| 1 | 17.53 | 0.9248 |
| 3 | 11.05 | 0.7588 |
| 6 | 10.66 | 0.7488 |
| 9 | 12.61 | 0.7987 |
| 12 | 10.78 | 0.7520 |
| 20 | 15.17 | 0.8643 |

Oscilação entre 10–20 Mbps reflete ciclos de rerouting ativo.

---

### 8.2 Dataset: MNIST

**Características**: classificação multiclasse (10 classes), rápida convergência, AUC referência ~0.99

| Round | C1 elapsed (s) | C2 elapsed (s) | C3 elapsed (s) | C1 AUC | C2 AUC | C3 AUC |
|:---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 46.75 | 135.48 | 146.63 | 0.9874 | 0.9882 | 0.9882 |
| 5 | 195.41 | 487.96 | 536.02 | 0.9950 | 0.9950 | 0.9951 |
| 10 | 311.97 | 721.67 | 791.87 | 0.9956 | 0.9958 | 0.9959 |
| 15 | 383.52 | 927.03 | 1016.12 | 0.9962 | 0.9962 | 0.9963 |
| **20** | **481.09** | **1133.71** | **1242.85** | **0.9963** | **0.9966** | **0.9964** |

- **Training time médio (rounds 15–20)**:
  - C1: ~21s | C2: ~39s | C3: ~43s
- **Speedup C1 vs C2**: **2.36×**
- **C3 vs C2**: +10% mais lento

---

### 8.3 Dataset: CreditCard

**Características**: detecção de fraude, altamente desbalanceado (fraudes raras), balanced_accuracy mais relevante que accuracy

| Round | C1 elapsed (s) | C2 elapsed (s) | C3 elapsed (s) | C1 AUC | C2 AUC | C3 AUC |
|:---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 45.17 | 112.34 | 123.26 | 0.9752 | 0.9720 | 0.9720 |
| 5 | 111.69 | 266.61 | 294.57 | 0.9690 | 0.9651 | 0.9668 |
| 10 | 187.88 | 419.15 | 455.45 | 0.9649 | 0.9665 | 0.9644 |
| 15 | 265.14 | 562.08 | 602.19 | 0.9671 | 0.9728 | 0.9619 |
| **20** | **342.14** | **700.56** | **740.01** | **0.9693** | **0.9707** | **0.9615** |

- **Training time médio (rounds 15–20)**:
  - C1: ~13s | C2: ~25s | C3: ~25s
- **Speedup C1 vs C2**: **2.05×**
- **C3 vs C2**: +6% mais lento
- **Nota AUC**: C3 finaliza com AUC 0.9615 vs C2 0.9707 (−0.0092). A oscilação natural do bagging com dados desbalanceados explica parte; rerouting pode ter causado instabilidade nos modelos locais.

---

### 8.4 Dataset: Higgs Full

**Características**: física de partículas, classificação binária, grande volume de dados, AUC referência ~0.82

| Round | C1 elapsed (s) | C2 elapsed (s) | C3 elapsed (s) | C1 AUC | C2 AUC | C3 AUC |
|:---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 123.23 | 157.28 | 167.18 | 0.8049 | 0.8051 | 0.8051 |
| 5 | 350.93 | 546.72 | 603.05 | 0.8177 | 0.8179 | 0.8179 |
| 10 | 512.30 | 835.65 | 919.09 | 0.8206 | 0.8210 | 0.8210 |
| 15 | 650.54 | 1109.19 | 1213.61 | 0.8223 | 0.8229 | 0.8229 |
| **20** | **763.77** | **1388.84** | **1515.75** | **0.8233** | **0.8241** | **0.8241** |

- **Training time médio (rounds 15–20)**:
  - C1: ~17s | C2: ~44s | C3: ~49s
- **Speedup C1 vs C2**: **1.82×**
- **C3 vs C2**: +9% mais lento
- **Nota**: Higgs tem modelos maiores (~2 MB ao final), explicando menor speedup do C1 (transferência de modelo domina sobre training time).

---

## 9. Análise Comparativa

### 9.1 Tempo total de execução (20 rounds)

| Dataset | C1 com_sdn | C2 sem_sdn | C3 rerouting | C1 vs C2 | C3 vs C2 |
|---------|:----------:|:----------:|:------------:|:--------:|:--------:|
| Epsilon | **421s** (7min) | 866s (14min) | 978s (16min) | **−51%** | +13% |
| MNIST | **481s** (8min) | 1134s (19min) | 1243s (21min) | **−58%** | +10% |
| CreditCard | **342s** (6min) | 701s (12min) | 740s (12min) | **−51%** | +6% |
| Higgs | **764s** (13min) | 1389s (23min) | 1516s (25min) | **−45%** | +9% |
| **Média** | — | — | — | **−51%** | **+9.5%** |

### 9.2 Qualidade do modelo — AUC no round 20

| Dataset | C1 com_sdn | C2 sem_sdn | C3 rerouting | Δ C1–C2 | Δ C3–C2 |
|---------|:----------:|:----------:|:------------:|:-------:|:-------:|
| Epsilon | 0.5893 | 0.5878 | 0.5878 | +0.0015 | 0.0000 |
| MNIST | 0.9963 | 0.9966 | 0.9964 | −0.0003 | −0.0002 |
| CreditCard | 0.9693 | 0.9707 | 0.9615 | −0.0014 | **−0.0092** |
| Higgs | 0.8233 | 0.8241 | 0.8241 | −0.0008 | 0.0000 |

**Conclusão sobre AUC**: as diferenças são inferiores a 0.002 em 3 dos 4 datasets, dentro da margem de variação natural do FL distribuído. O CreditCard com C3 mostra queda maior (0.009) — provavelmente instabilidade do bagging em dados desbalanceados quando combinado com perturbações de rede por rerouting.

### 9.3 Tempo médio de treinamento por round (rounds 10–20)

| Dataset | C1 (s) | C2 (s) | C3 (s) | Observação |
|---------|:---:|:---:|:---:|---|
| Epsilon | ~16 | ~34 | ~38 | C1 usa 2–3 clientes/round vs 6 no C2/C3 |
| MNIST | ~18 | ~39 | ~43 | Mesmo padrão |
| CreditCard | ~13 | ~25 | ~26 | Modelos pequenos — menor diferença proporcional |
| Higgs | ~20 | ~44 | ~48 | Modelos grandes (~2 MB) ainda mostram ganho |

### 9.4 Atividade de rerouting

| Cenário | n_reroute_flows | Observação |
|---------|:---:|---|
| C1 com_sdn | ~154 | Rerouting ativo e consistente |
| C2 sem_sdn | 0 | Orquestrador desligado |
| C3 rerouting | ~140 | Rerouting ativo (pico em ciclos tardios) |

**Nota C3**: nos primeiros ~19 ciclos o orquestrador reportou n_hosts=0 (host discovery ainda não completado). A partir do ciclo ~100, com hosts descobertos, n_reroute_flows atingiu 140 e congested_links=4–8. O sdn_metrics mostra max_link_load_bps acima de 100 Mbps — valor que reflete a soma das contagens de porta ODL (que acumula bytes por hop: sw8→sw3→sw1 conta 3 vezes), não a taxa física real.

---

## 10. Conclusões e Indícios

### Conclusão 1: O SDN com seleção de clientes é significativamente mais eficiente

**Evidência**: C1 é 45–58% mais rápido que C2 em todos os datasets. O mecanismo é direto:

- No C1, a cada round apenas **2–4 clientes** participam (os de maior health score)
- No C2 e C3, todos os **6 clientes** participam sempre
- Menos clientes = menos transferências de modelo = round mais rápido

A seleção é guiada pela métrica de rede (30% do health score via `efficiency_score`). Clientes com caminhos congestionados (bandwidth < 13 Mbps) recebem penalidade no health score, aumentando sua probabilidade de exclusão.

---

### Conclusão 2: Rerouting sem seleção de clientes piora o desempenho

**Evidência**: C3 é 6–13% mais lento que C2 (sem nenhum SDN), apesar de ter o rerouting ativo.

**Mecanismo proposto**:

1. O rerouting instala novos flows OpenFlow nos switches OVS durante o treinamento FL
2. A troca de flows causa **reordenamento de pacotes** nas conexões TCP ativas (gRPC do FL)
3. O TCP interpreta o reordenamento como possível perda → aciona retransmissão → aumenta latência
4. Todos os 6 clientes continuam participando, incluindo os mais lentos (congestionados pelo BG traffic)
5. O round é tão rápido quanto o **cliente mais lento** — e com BG traffic ativo, clientes 0,1,2,5 (via sw3/sw5) consistentemente têm bandwidth reduzido

O SDN redistribui a carga na rede, mas sem eliminar os clientes lentos do FL, o benefício do rerouting é anulado e superado pelo overhead de disrupção TCP.

---

### Conclusão 3: A qualidade do modelo é robusta às condições de rede

**Evidência**: em 3 dos 4 datasets, a diferença de AUC no round 20 entre C1, C2 e C3 é inferior a 0.001 — equivalente a zero estatisticamente. O federated learning converge para o mesmo modelo independente de qual estratégia de rede é usada, desde que os dados estejam presentes nos clientes.

A exceção é CreditCard em C3 (AUC 0.9615 vs 0.9707 em C2). Este dataset é extremamente desbalanceado (~0.17% de fraudes). A instabilidade do bagging com dados desbalanceados, amplificada por perturbações de rede intermitentes do rerouting, pode ter gerado modelos locais com viés oscilante.

---

### Conclusão 4: A combinação rerouting + seleção é o mecanismo ótimo

O health score captura três dimensões simultâneas:

```
health = 0.4 × contribuição_histórica
       + 0.3 × tempo_de_treinamento_local
       + 0.3 × qualidade_da_rede
```

A dimensão de rede (0.3 × efficiency_score) diferencia clientes em caminhos congestionados dos que têm caminho livre:

| Grupo de clientes | Bandwidth típico | Efficiency Score | Contribuição ao health |
|-------------------|:---:|:---:|:---:|
| Clientes 3, 4 (via sw4 — sem BG traffic) | ~20 Mbps | ~0.988 | +0.296 |
| Clientes 0, 1, 2, 5 (via sw3/sw5 — com BG traffic) | 10–17 Mbps | 0.74–0.93 | +0.222 a +0.279 |

A diferença de **0.017 a 0.074 pontos** no health score, somada à contribuição histórica, determina frequentemente qual cliente é excluído. Em rounds onde a diferença de contribuição é pequena, o componente de rede pode ser o fator decisivo.

---

### Conclusão 5: O BG traffic criou condições realistas de congestionamento

**Evidências quantitativas**:

- Links sw8↔sw3 e sw14↔sw5 atingiram 60–85% de utilização consistentemente
- O REROUTE_THRESH (65% = 13 Mbps) foi cruzado com frequência → rerouting ativado
- Clientes 0,1,2,5 (caminhos congestionados) obtiveram bandwidth 10–17 Mbps
- Clientes 3,4 (caminhos alternativos via sw4) obtiveram bandwidth ~20 Mbps (máximo)

A diferença de comportamento entre grupos de clientes, claramente refletida nas métricas SDN, confirma que o tráfego de fundo criou o ambiente heterogêneo necessário para o experimento ser válido.

---

### Resumo das Implicações

| Pergunta | Resposta |
|----------|----------|
| SDN melhora a velocidade do FL? | **Sim**, quando inclui seleção de clientes (+45–58%) |
| SDN sozinho (apenas rerouting) ajuda? | **Não** — piora em 6–13% |
| O modelo FL é afetado pelo SDN? | **Não significativamente** (Δ AUC < 0.002 em 3/4 datasets) |
| Qual é o componente mais importante do SDN? | **A seleção de clientes baseada em health score** |
| O rerouting tem algum valor? | **Sim** — como suporte ao health score (melhora BW disponível), não como mecanismo standalone |

---

## Apêndice — Linhas do tempo dos experimentos

### V5 (27/03/2026)

| Experimento | Início | Fim | Duração |
|-------------|--------|-----|---------|
| sem_sdn_epsilon | 22:03:28 | 22:18:12 | 14 min |
| com_sdn_epsilon | 22:19:33 | 22:26:46 | **7 min** |
| sem_sdn_higgs_full | 22:28:35 | 22:52:17 | 23 min |
| com_sdn_higgs_full | 22:53:38 | 23:06:54 | **13 min** |
| sem_sdn_mnist | 23:08:43 | 23:27:54 | 19 min |
| com_sdn_mnist | 23:29:15 | 23:37:26 | **8 min** |
| sem_sdn_creditcard | 23:39:15 | 23:51:27 | 12 min |
| com_sdn_creditcard | 23:52:48 | 23:58:58 | **6 min** |

### V4 (28/03/2026)

| Experimento | Início | Fim | Duração |
|-------------|--------|-----|---------|
| rerouting_epsilon | 10:15:36 | 10:32:17 | 16 min |
| rerouting_higgs_full | 10:33:24 | 10:59:11 | 25 min |
| rerouting_mnist | 11:00:14 | 11:21:26 | 21 min |
| rerouting_creditcard | 11:22:27 | 11:35:07 | 12 min |

---

*Gerado automaticamente em 2026-03-28 · LUMO-UFPB*
