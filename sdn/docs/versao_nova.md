# Documentação Completa — Experimento FL-SDN v2
## Nova Topologia Hierárquica + Estratégia sdn-bagging com Health Score

**Laboratório:** LUMO-UFPB  
**Data:** Março 2026  
**Stack:** ODL Calcium SR3 · Open vSwitch (GNS3 Docker) · Flower FL · XGBoost · FastAPI · Python 3.12

---

## Sumário

1. [Visão Geral do Sistema](#1-visão-geral-do-sistema)
2. [Nova Topologia de Rede](#2-nova-topologia-de-rede)
3. [Arquitetura do Código](#3-arquitetura-do-código)
4. [Fluxo Completo de um Round FL com Health Score](#4-fluxo-completo-de-um-round-fl-com-health-score)
5. [Bloqueio Dinâmico de Clientes](#5-bloqueio-dinâmico-de-clientes)
6. [Passo a Passo — Setup do Ambiente](#6-passo-a-passo--setup-do-ambiente)
7. [Comandos de Execução do Experimento](#7-comandos-de-execução-do-experimento)
8. [Diferenças em Relação à Topologia Anterior](#8-diferenças-em-relação-à-topologia-anterior)

---

## 1. Visão Geral do Sistema

O experimento mede o impacto do SDN na convergência do Federated Learning em condições de congestionamento. São três projetos independentes que colaboram:

```mermaid
graph TD
    subgraph HOST["Host Ubuntu 172.16.1.1"]
        ODL["ODL Karaf :8181/:6653\nDescobre topologia via LLDP\nRastreia hosts via l2switch"]
        SDN["sdn_orchestrator.py :8000\nDijkstra ponderado\nInstala flows via ovs-ofctl\nExpõe API FastAPI"]
        FL_SERVER["server.py :8080\nFlower gRPC\nAgrega modelos XGBoost\nCalcula health score"]
    end

    subgraph GNS3["GNS3 — Rede Virtual"]
        subgraph CORE["Núcleo"]
            C1["OVS-1\nopenflow:1"]
            C2["OVS-2\nopenflow:2"]
        end
        subgraph AGG["Agregação"]
            A1["OVS-3\nopenflow:3"]
            A2["OVS-4\nopenflow:4"]
            A3["OVS-5\nopenflow:5"]
        end
        subgraph EDGE["Borda"]
            E1["OVS-6"] 
            E2["OVS-7"]
            E3["OVS-8"]
            E4["OVS-9"]
            E5["OVS-10"]
            E6["OVS-11"]
            E7["OVS-12"]
            E8["OVS-13"]
            E9["OVS-14"]
        end
        subgraph HOSTS["Containers FL/BG"]
            H1["FL-Node-1-cat1\n172.16.1.10"]
            H2["FL-Node-2-cat2\n172.16.1.11"]
            H3["BG-Node-1-cat1\n172.16.1.12"]
            H4["FL-Node-3-cat3\n172.16.1.13"]
            H5["FL-Node-4-cat2\n172.16.1.14"]
            H6["BG-Node-2-cat1\n172.16.1.15"]
            H7["FL-Node-5-cat1\n172.16.1.16"]
            H8["FL-Node-6-cat3\n172.16.1.17"]
            H9["BG-Node-3-cat2\n172.16.1.18"]
        end
    end

    ODL -->|"REST :8181\nlê topologia/hosts/stats"| SDN
    SDN -->|"docker exec ovs-ofctl\n~5ms por flow"| CORE
    FL_SERVER -->|"GET /metrics/hosts\n:8000"| SDN
    FL_SERVER -->|"gRPC :8080"| HOSTS

    tap0["tap0 172.16.1.1/24"] -->|"plano controle OOB"| CORE
    tap1["tap1 sem IP"] -->|"plano dados FL"| CORE
```

**Separação de planos:**
- **tap0** → plano de controle: ODL usa para OpenFlow (porta 6653) e REST (porta 8181)
- **tap1** → plano de dados: clientes FL enviam modelos via gRPC para o servidor

---

## 2. Nova Topologia de Rede

### 2.1 Hierarquia de switches

```mermaid
graph TD
    HOST["Host 172.16.1.1\nFL Server + SDN Orchestrator + ODL"]
    
    tap0["tap0 OOB"]
    tap1["tap1 dados FL"]
    
    SW1["Switch1 GNS3\nOOB — não OVS\nnão gerenciado pelo ODL"]
    
    C1["OVS-1 DPID:1\n172.16.1.101\nNÚCLEO"]
    C2["OVS-2 DPID:2\n172.16.1.102\nNÚCLEO"]
    
    A1["OVS-3 DPID:3\n172.16.1.103\nAGREGAÇÃO"]
    A2["OVS-4 DPID:4\n172.16.1.104\nAGREGAÇÃO"]
    A3["OVS-5 DPID:5\n172.16.1.105\nAGREGAÇÃO"]
    
    E1["OVS-6\nBORDA"] 
    E2["OVS-7\nBORDA"]
    E3["OVS-8\nBORDA"]
    E4["OVS-9\nBORDA"]
    E5["OVS-10\nBORDA"]
    E6["OVS-11\nBORDA"]
    E7["OVS-12\nBORDA"]
    E8["OVS-13\nBORDA"]
    E9["OVS-14\nBORDA"]

    HOST --- tap0
    HOST --- tap1
    tap0 --> SW1
    tap1 --> C1
    
    SW1 -->|"eth0"| C1
    SW1 -->|"eth0"| C2
    SW1 -->|"eth0"| A1
    SW1 -->|"eth0"| A2
    SW1 -->|"eth0"| A3
    SW1 -->|"eth0"| E1
    SW1 -->|"eth0"| E2
    SW1 -->|"eth0"| E3
    SW1 -->|"eth0"| E4
    SW1 -->|"eth0"| E5
    SW1 -->|"eth0"| E6
    SW1 -->|"eth0"| E7
    SW1 -->|"eth0"| E8
    SW1 -->|"eth0"| E9
    
    C1 -->|"eth1+"| A1
    C1 -->|"eth2+"| A2
    C1 -->|"eth3+"| A3
    C2 -->|"eth1+"| A1
    C2 -->|"eth2+"| A2
    C2 -->|"eth3+"| A3
    
    A1 -->|"eth+"| E1
    A1 -->|"eth+"| E2
    A1 -->|"eth+"| E3
    A2 -->|"eth+"| E4
    A2 -->|"eth+"| E5
    A2 -->|"eth+"| E6
    A3 -->|"eth+"| E7
    A3 -->|"eth+"| E8
    A3 -->|"eth+"| E9
```

### 2.2 Mapa completo dos hosts de borda

| OVS | DPID | IP Gerência | Tier | Host Conectado | IP Host | Cat. | Papel | Client-ID |
|---|---|---|---|---|---|---|---|---|
| OVS-1 | 1 | 172.16.1.101 | Núcleo | — | — | — | Encaminhamento | — |
| OVS-2 | 2 | 172.16.1.102 | Núcleo | — | — | — | Encaminhamento | — |
| OVS-3 | 3 | 172.16.1.103 | Agregação | — | — | — | Encaminhamento | — |
| OVS-4 | 4 | 172.16.1.104 | Agregação | — | — | — | Encaminhamento | — |
| OVS-5 | 5 | 172.16.1.105 | Agregação | — | — | — | Encaminhamento | — |
| OVS-6 | 6 | 172.16.1.106 | Borda | FL-Node-1-cat1 | 172.16.1.10 | cat1 | **Cliente FL** | 0 |
| OVS-7 | 7 | 172.16.1.107 | Borda | FL-Node-2-cat2 | 172.16.1.11 | cat2 | **Cliente FL** | 2 |
| OVS-8 | 8 | 172.16.1.108 | Borda | BG-Node-1-cat1 | 172.16.1.12 | cat1 | **BG gerador** | — |
| OVS-9 | 9 | 172.16.1.109 | Borda | FL-Node-3-cat3 | 172.16.1.13 | cat3 | **Cliente FL** | 4 |
| OVS-10 | 10 | 172.16.1.110 | Borda | FL-Node-4-cat2 | 172.16.1.14 | cat2 | **Cliente FL** | 3 |
| OVS-11 | 11 | 172.16.1.111 | Borda | BG-Node-2-cat1 | 172.16.1.15 | cat1 | **BG receptor** | — |
| OVS-12 | 12 | 172.16.1.112 | Borda | FL-Node-5-cat1 | 172.16.1.16 | cat1 | **Cliente FL** | 1 |
| OVS-13 | 13 | 172.16.1.113 | Borda | FL-Node-6-cat3 | 172.16.1.17 | cat3 | **Cliente FL** | 5 |
| OVS-14 | 14 | 172.16.1.114 | Borda | BG-Node-3-cat2 | 172.16.1.18 | cat2 | **BG gerador** | — |

### 2.3 Estratégia de congestionamento

```mermaid
graph LR
    BG1["BG-Node-1\n172.16.1.12\nOVS-8 / Ag1"]
    BG3["BG-Node-3\n172.16.1.18\nOVS-14 / Ag3"]
    BG2["BG-Node-2\n172.16.1.15\nOVS-11 / Ag2\nservidor iperf3"]

    BG1 -->|"iperf3 15M×2\nporta 5201"| BG2
    BG3 -->|"iperf3 15M×2\nporta 5202"| BG2
```

Dois geradores em grupos de agregação diferentes apontando para o mesmo receptor. Os fluxos de background percorrem **Ag1→Core→Ag2** e **Ag3→Core→Ag2**, saturando os uplinks de núcleo — exatamente os caminhos que o tráfego FL também precisa usar para chegar ao servidor em 172.16.1.1.

---

## 3. Arquitetura do Código

### 3.1 Repositórios e responsabilidades

```mermaid
graph TD
    subgraph SDN["sdn-project-main/ — SDN Orchestrator"]
        SDN_MAIN["sdn_orchestrator.py\nEntrypoint — inicia FastAPI + loop de controle"]
        SDN_CFG["orchestrator/config.py\nMAX_LINK_CAPACITY=20Mbps\nROUTE_THRESH=75%\nDOCKER_NAME_PREFIX=GNS3.OVS-"]
        SDN_DOMAIN["domain/state.py\nNetworkState — estado global\ngraph, hosts_by_mac, link_load\nblocked_ips, active_flows"]
        SDN_TOPO["application/topology.py\nDijkstra ponderado\nflows base TABLE-MISS/LLDP/BDDP"]
        SDN_HOSTS["application/hosts.py\nDescoberta via l2switch ODL\nARP probing a cada 6 ciclos"]
        SDN_TRAFFIC["application/traffic.py\nMonitoramento de bps por enlace\nAtualiza link_load a cada 5s"]
        SDN_ROUTING["application/routing.py\nInstala flows IPv4 proativos\nReroute com idle_timeout=15s"]
        SDN_API["presentation/api.py\nGET /metrics/hosts\nGET /metrics/links\nPOST /manage/ip block/unblock"]
        SDN_OVS["infrastructure/ovs_adapter.py\ndocker exec ovs-ofctl ~5ms\nThreadPoolExecutor max_workers=20"]
    end

    subgraph FL["fl-node/ — FL Original (experimentos base)"]
        FL_SERVER["fl_simple_demo/server.py\nFlower gRPC :8080\nSimpleBagging / SimpleCycling"]
        FL_CLIENT["fl_simple_demo/client.py\nXGBoost warm start\nPickle serialização"]
        FL_CFG["fl_simple_demo/config.py\nNUM_CLIENTS=6\nLOCAL_EPOCHS_BY_CAT"]
    end

    subgraph FLSDN["FL-SDN-main/ — FL Evoluído"]
        FLSDN_SERVER["fl_sdn_code/server.py\nSDNBagging / SDNCycling\nConsulta /metrics/hosts"]
        FLSDN_CLIENT["fl_sdn_code/client.py\nResourceMonitor CPU/RAM\n12 métricas de avaliação"]
        FLSDN_HEALTH["fl_sdn_code/core/health_score.py\nClientHealthTracker\n4 perfis de pesos\nLeave-one-out"]
        FLSDN_NETWORK["fl_sdn_code/sdn/network.py\nGET /metrics/hosts do orquestrador\ncalcula efficiency_score"]
        FLSDN_CONTROLLER["fl_sdn_code/sdn/controller.py\nHTTP client para :8000\nnão fala com ODL diretamente"]
        FLSDN_CFG["fl_sdn_code/config.py\nSDN_ORCHESTRATOR_IP\nSDN_MIN_BANDWIDTH_MBPS=15\nSDN_ADAPTIVE_EPOCHS=False"]
    end

    SDN_API -->|"expõe dados já processados"| FLSDN_NETWORK
    FLSDN_CONTROLLER -->|"GET :8000"| SDN_API
    FL_SERVER -->|"gRPC :8080"| FL_CLIENT
    FLSDN_SERVER -->|"gRPC :8080"| FLSDN_CLIENT
```

### 3.2 Por que o FL-SDN não fala com o ODL diretamente

```mermaid
graph TD
    subgraph ANTES["ANTES — conflito"]
        FL_OLD["FL-SDN\nsdn/network.py"]
        SDN_OLD["SDN Orchestrator"]
        ODL_OLD["ODL :8181"]
        FL_OLD -->|"GET stats porta T"| ODL_OLD
        SDN_OLD -->|"GET stats porta T+Δ"| ODL_OLD
        ODL_OLD -->|"snapshots dessincronizados"| CONFLICT["❌ Race condition\nDecisões contraditórias"]
    end

    subgraph DEPOIS["DEPOIS — fonte única de verdade"]
        FL_NEW["FL-SDN\nsdn/network.py"]
        SDN_NEW["SDN Orchestrator\nfonte de verdade"]
        ODL_NEW["ODL :8181"]
        ODL_NEW -->|"coleta raw"| SDN_NEW
        FL_NEW -->|"GET /metrics/hosts :8000\ndados já processados"| SDN_NEW
        SDN_NEW -->|"✓ dados consistentes\nsem duplo polling"| OK["✅ Decisões alinhadas"]
    end
```

---

## 4. Fluxo Completo de um Round FL com Health Score

```mermaid
graph TD
    START["Início do Round N"]
    
    CFG["configure_fit()\nServidor SDNBagging"]
    METRICS["GET /metrics/hosts :8000\nObtém bandwidth, latência,\npacket_loss de cada cliente"]
    EFFICIENCY["Calcula efficiency_score\nbw×0.5 + lat×0.3 + loss×0.2\nNormalizado por 15 Mbps"]
    HEALTH["Calcula health_score\nW_contribution×C + W_resource×R + W_network×N\nPerfil: balanced (0.4/0.3/0.3)"]
    EXCLUDE["get_excluded_clients()\nScore < 0.30 E round ≥ 2?\nMáx 2 excluídos por round\nNunca > 50% dos clientes"]
    SEND["Envia FitIns\naos clientes elegíveis\ncom modelo warm start"]
    
    TRAIN["fit() em cada container\nXGBoost warm start\nCPU/RAM monitorados via psutil"]
    SERIALIZE["pickle.dumps(model)\n~400KB round 1\naté ~15MB round 20"]
    GRPC["gRPC FitRes\nmodel_bytes em Parameters.tensors\ntraverssal OVS com flows SDN"]
    
    AGG["aggregate_fit()\nDesserializa modelos\nIdentifica best_model"]
    LOO["compute_leave_one_out()\nRemove cliente do ensemble\nMede impacto na accuracy"]
    UPDATE["update_round()\nAtualiza contribution/resource/network scores\nGrava health_scores.csv"]
    
    EVAL["evaluate()\nEnsemble: média de probabilidades\nCalcula 12 métricas"]
    LOG["_log_round()\nGrava com_sdn_resultados.csv\n24 campos incluindo métricas de rede"]
    
    SDN_CYCLE["SDN Orchestrator\n(paralelo, a cada 5s)\nmonitor_traffic_load()\ncheck_and_reroute()"]
    REROUTE["Link > 75% de 20 Mbps?\nDijkstra com peso=9999\nInstala flows priority=62000\nidle_timeout=15s"]

    START --> CFG
    CFG --> METRICS
    METRICS --> EFFICIENCY
    EFFICIENCY --> HEALTH
    HEALTH --> EXCLUDE
    EXCLUDE -->|"clientes elegíveis"| SEND
    SEND --> TRAIN
    TRAIN --> SERIALIZE
    SERIALIZE --> GRPC
    GRPC --> AGG
    AGG --> LOO
    LOO --> UPDATE
    UPDATE --> EVAL
    EVAL --> LOG
    LOG --> START

    SDN_CYCLE --> REROUTE
    REROUTE -->|"fluxos FL desviados\ncaminhos alternativos"| GRPC
```

---

## 5. Bloqueio Dinâmico de Clientes

O sistema de health score não bloqueia clientes na rede — ele os **exclui temporariamente do round FL**. São três dimensões independentes:

```mermaid
graph TD
    subgraph CONTRIBUTION["Contribution Score\nW=0.40 no perfil balanced"]
        LOO2["Leave-one-out\nRemove cliente do ensemble\nmede queda de accuracy"]
        ACC_REL["Accuracy relativa\ncliente vs média dos outros"]
        CONSIST["Consistência histórica\nVariância dos últimos 5 rounds\nalta variância = penalizado"]
        LOO2 --> CSCORE["contribution_score\n0.0 = prejudica ensemble\n1.0 = essencial"]
        ACC_REL --> CSCORE
        CONSIST --> CSCORE
    end

    subgraph RESOURCE["Resource Score\nW=0.30 no perfil balanced"]
        TIME["Tempo de treino\n(menor = melhor)\npsutil medido no container"]
        CPU["CPU%\n(menor = melhor)"]
        RAM["RAM MB\n(menor = melhor)"]
        TIME -->|"peso 0.50"| RSCORE["resource_score\n⚠ Pouco discriminativo\nem ambiente homogêneo\n(mesmo host físico)"]
        CPU -->|"peso 0.25"| RSCORE
        RAM -->|"peso 0.25"| RSCORE
    end

    subgraph NETWORK["Network Score\nW=0.30 no perfil balanced"]
        BW["bandwidth_mbps\nGET /metrics/hosts\ndo orquestrador SDN"]
        LAT["latency_ms\nestimada por M/D/1\nutilização do enlace"]
        LOSS["packet_loss\nerros de transmissão OVS"]
        BW -->|"peso 0.50"| NSCORE["network_score\nbaixo quando link > 75%\n= cliente sob congestionamento"]
        LAT -->|"peso 0.30"| NSCORE
        LOSS -->|"peso 0.20"| NSCORE
    end

    CSCORE --> HEALTH_FINAL["health_score = 0.4×C + 0.3×R + 0.3×N"]
    RSCORE --> HEALTH_FINAL
    NSCORE --> HEALTH_FINAL

    HEALTH_FINAL --> DECISION{"score < 0.30\nE round ≥ 2?"}
    DECISION -->|"Sim"| BLOCK["Cliente excluído\nneste round\n(temporário)"]
    DECISION -->|"Não"| OK2["Cliente participa\nnormalmente"]
    BLOCK --> NEXT["Próximo round:\nrecalcula score\npode ser reincluído"]
```

### Regras de exclusão

| Regra | Valor padrão | Motivo |
|---|---|---|
| Rounds mínimos antes de excluir | 2 | Histórico insuficiente nos primeiros rounds |
| Threshold de exclusão | 0.30 | Abaixo disso o cliente prejudica mais do que contribui |
| Máximo de excluídos por round | 2 | Evita degradar o ensemble por falta de diversidade |
| Proteção de quorum | 50% | Nunca exclui mais da metade dos clientes elegíveis |

### Exemplo real do experimento anterior

No round 6, o cliente 1 foi excluído com `contribution_score=0.0` — o leave-one-out mostrou que o ensemble ficava melhor sem ele. No round 9, três exclusões simultâneas (clientes 3, 4 e o sistema aplicou a regra de quorum automaticamente limitando a 2).

---

## 6. Passo a Passo — Setup do Ambiente

### 6.1 Configurar os switches OVS

```bash
sudo bash ~/Downloads/setup_switch.sh
```

**O que faz:** itera sobre todos os containers Docker com nome `GNS3.OVS-N`, entra em cada um e executa:
- `ifconfig eth0 172.16.1.1XX` — configura IP de gerência (plano de controle OOB)
- `ovs-vsctl add-br br0` — cria a bridge OpenFlow
- `ovs-vsctl set bridge br0 other-config:datapath-id=$DPID` — DPID fixo para que o ODL sempre reconheça o mesmo switch
- `ovs-vsctl set-controller br0 tcp:172.16.1.1:6653` — registra o ODL como controlador
- `ovs-vsctl set bridge br0 protocols=OpenFlow13 fail-mode=secure` — OpenFlow 1.3, descarta pacotes se ODL desconectar
- `ovs-vsctl set bridge br0 other-config:disable-in-band=true` — evita flows ocultos de gerenciamento que conflitariam com os flows do orquestrador

**Por que DPID fixo:** sem DPID fixo, o OVS gera um DPID aleatório a cada reinicialização. O ODL registraria o switch como `openflow:XXXXXXXXX` diferente a cada boot, e o orquestrador perderia o mapeamento container→switch.

### 6.2 Configurar plano de dados e hosts de borda

```bash
sudo ./setup_experimento.sh
```

**O que faz nos passos relevantes:**

**PASSO 5 — tc tbf nos OVS:**
```bash
nsenter -t $PID -n tc qdisc add dev eth$i root tbf rate 20mbit burst 32kbit latency 10ms
```
`nsenter` entra no namespace de rede do container e executa o `tc` do host. Aplica Token Bucket Filter em todas as `eth1..eth15` de cada OVS (não em `eth0` que é o plano de controle). Sem isso, as interfaces teriam capacidade ilimitada e o congestionamento não seria observável.

**PASSO 6 — Configura containers de borda:**
```bash
sysctl -w net.ipv6.conf.all.disable_ipv6=1  # evita flood NDP no ODL
ip addr add 172.16.1.XX/24 dev eth0          # IP estático
ip route add default via 172.16.1.1          # rota para o servidor FL
ping -c 2 172.16.1.1                         # força ARP → ODL descobre o host
```

**Por que IPv6 desabilitado:** o l2switch do ODL trata pacotes NDP (Neighbor Discovery Protocol do IPv6) como novos hosts, preenchendo a tabela com endereços `fe80::` inúteis. Isso polui os logs e aumenta o tráfego de controle.

**Por que o ping é necessário:** o l2switch descobre hosts capturando ARP requests. Sem o ping, o ODL não mapeia `IP → MAC → switch → porta` e o orquestrador não consegue calcular rotas Dijkstra para aquele host.

### 6.3 Copiar o código FL-SDN para os containers

```bash
cd ~/FL-SDN-main
tar czf /tmp/fl_sdn_code.tar.gz \
    --exclude='fl_sdn_code/data' \
    --exclude='fl_sdn_code/output' \
    --exclude='fl_sdn_code/__pycache__' \
    fl_sdn_code/
```

**Por que excluir `data/` e `output/`:** os arquivos `.npy` do dataset Higgs têm 5.4 MB cada. Com 9 containers, copiar os dados aumentaria o tar de ~1 MB para ~650 MB desnecessariamente — os dados já existem em `/fl/data/` dentro de cada container desde a build da imagem `fl-node:latest`.

```bash
for container in $(sudo docker ps --format '{{.Names}}' | grep -E 'FL-Node|BG-Node'); do
    docker cp /tmp/fl_sdn_code.tar.gz "$container":/tmp/
    docker exec "$container" bash -c "cd /fl && tar xzf /tmp/fl_sdn_code.tar.gz"
done
```

**Por que `docker cp` e não volume mount:** os containers GNS3 não são criados com volumes — são containers efêmeros que o GNS3 gerencia. A única forma de inserir arquivos é via `docker cp` após o container estar rodando.

```bash
for container in $(sudo docker ps --format '{{.Names}}' | grep -E 'FL-Node|BG-Node'); do
    docker exec "$container" bash -c "
        mkdir -p /fl/fl_sdn_code/data
        ln -sf /fl/data/higgs_X.npy /fl/fl_sdn_code/data/higgs_X.npy
        ln -sf /fl/data/higgs_y.npy /fl/fl_sdn_code/data/higgs_y.npy
    "
done
```

**Por que symlink e não cópia:** o `fl_sdn_code/datasets/higgs.py` procura os `.npy` em `fl_sdn_code/data/`. Criar symlinks apontando para `/fl/data/` evita duplicar 11 MB de dados por container sem copiar os arquivos fisicamente.

### 6.4 Atualizar config.py e propagar

```bash
# Após editar ~/FL-SDN-main/fl_sdn_code/config.py
for container in $(sudo docker ps --format '{{.Names}}' | grep -E 'FL-Node|BG-Node'); do
    docker cp ~/FL-SDN-main/fl_sdn_code/config.py \
        "$container":/fl/fl_sdn_code/config.py
done
```

**Por que propagar o config:** cada cliente FL lê `config.py` para saber `NUM_CLIENTS`, `CLIENT_CATEGORIES`, `CLIENT_CONNECT_ADDRESS` e `SDN_CLIENT_IPS`. Se o config estiver desatualizado, o cliente tenta fazer download do dataset via internet (sem DNS nos containers) ou usa um número errado de clientes para o particionamento do dataset.

---

## 7. Comandos de Execução do Experimento

### Sequência completa

```bash
# ── Terminal 1: Orquestrador SDN ──────────────────────────────────────────
cd ~/sdn-project-main && source venv/bin/activate
python3 sdn_orchestrator.py
# Aguardar: "OK: 14 switches | XX enlaces"
# Aguardar: "9 host(s) conhecido(s)"

# ── Terminal 2: Servidor FL ───────────────────────────────────────────────
cd ~/FL-SDN-main/fl_sdn_code && source ~/fl-node/venv/bin/activate
EXP=com_sdn python3 server.py --model xgboost --strategy sdn-bagging
# Aguardar: "Aguardando 6 cliente(s) conectarem..."

# ── BG-Node-2-cat1 (172.16.1.15) — servidor iperf3 ───────────────────────
iperf3 -s -D          # porta padrão 5201 (daemon)
iperf3 -s -p 5202 -D  # segunda porta para segundo gerador

# ── BG-Node-1-cat1 (172.16.1.12) — gerador de congestionamento ───────────
iperf3 -c 172.16.1.15 -p 5201 -t 9999 -b 15M -P 2 &
# 2 fluxos × 15 Mbps = 30 Mbps tentados em link de 20 Mbps

# ── BG-Node-3-cat2 (172.16.1.18) — segundo gerador ───────────────────────
iperf3 -c 172.16.1.15 -p 5202 -t 9999 -b 15M -P 2 &
# Fluxo via Ag3 → Core → Ag2, caminho diferente do BG-Node-1

# ── 6 Clientes FL (após servidor aguardar conexões) ──────────────────────

# FL-Node-1-cat1 (172.16.1.10) — client-id 0, cat1, 50 épocas
python3 /fl/fl_sdn_code/client.py --client-id 0 --model xgboost

# FL-Node-5-cat1 (172.16.1.16) — client-id 1, cat1, 50 épocas
python3 /fl/fl_sdn_code/client.py --client-id 1 --model xgboost

# FL-Node-2-cat2 (172.16.1.11) — client-id 2, cat2, 100 épocas
python3 /fl/fl_sdn_code/client.py --client-id 2 --model xgboost

# FL-Node-4-cat2 (172.16.1.14) — client-id 3, cat2, 100 épocas
python3 /fl/fl_sdn_code/client.py --client-id 3 --model xgboost

# FL-Node-3-cat3 (172.16.1.13) — client-id 4, cat3, 150 épocas
python3 /fl/fl_sdn_code/client.py --client-id 4 --model xgboost

# FL-Node-6-cat3 (172.16.1.17) — client-id 5, cat3, 150 épocas
python3 /fl/fl_sdn_code/client.py --client-id 5 --model xgboost
```

### Experimento SEM SDN (controle)

```bash
# Parar orquestrador (Ctrl+C)
# Limpar flows de reroute
cd ~/sdn-project-main && python3 sdn_tools.py clean

# Rodar servidor sem SDN
EXP=sem_sdn python3 server.py --model xgboost --strategy bagging
# Manter background traffic e clientes FL iguais
```

### Analisar resultados após os experimentos

```bash
cd ~/FL-SDN-main/fl_sdn_code

# Exclusões por cliente
python3 -c "
import pandas as pd, glob
df = pd.read_csv(glob.glob('output/*/health_scores.csv')[0])
print(df[df.excluded].groupby('client_id').size())
print(df[df.excluded][['round','client_id','health_score','contribution_score','network_score']])
"

# Gráficos comparativos
python3 plot_resultados.py \
    --com output/*/com_sdn_resultados.csv \
    --sem output/*/sem_sdn_resultados.csv
```

---

## 8. Diferenças em Relação à Topologia Anterior

| Aspecto | Topologia v1 (10 OVS flat) | Topologia v2 (14 OVS hierárquica) |
|---|---|---|
| Switches | 10 (sem hierarquia) | 14 (2 núcleo + 3 agregação + 9 borda) |
| Redundância | Links entre switches de mesmo nível | Uplinks duplos Ag→Core |
| Hosts por switch | 1-2 hosts por switch de borda | 1 host por switch de borda |
| Hosts totais | 8 (6 FL + 2 BG) | 9 (6 FL + 3 BG) |
| Background traffic | 2 fluxos mesmo caminho | 2 fluxos caminhos diferentes (Ag1 e Ag3) |
| Congestionamento | Localizado em poucos links | Distribuído em múltiplos uplinks |
| Caminhos alternativos | Limitados pela topologia flat | Múltiplos via Core1↔Core2 |
| IPs hosts | 172.16.1.20-.100 | 172.16.1.10-.18 |
| Start command OVS | `/usr/share/openvswitch/scripts/ovs-ctl start && tail -f /dev/null` | vazio (usa CMD da imagem: `/etc/openvswitch/init.sh; bash`) |
| Prefixo Docker | `GNS3.OpenvSwitchLocal-` | `GNS3.OVS-` |
| FL utilizado | `fl_simple_demo/` (simples) | `fl_sdn_code/` (com health score) |
| Estratégia FL | `bagging` | `sdn-bagging` |
| Métricas CSV | 7 colunas | 24 colunas (+ recursos + rede) |
| Arquivos de saída | `com_sdn_resultados.csv` | `com_sdn_resultados.csv` + `health_scores.csv` + `sdn_metricas.csv` |

---

*Documentação gerada para o experimento FL-SDN v2 — Topologia hierárquica 2-3-9 — LUMO-UFPB 2026.*