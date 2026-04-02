# Documentação do SDN Orchestrator (v14.0)

> Baseado no código em `sdn/orchestrator/` — Projeto SDN com Open vSwitch + OpenDaylight.

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura e Estrutura de Diretórios](#2-arquitetura-e-estrutura-de-diretórios)
3. [Configuração Global](#3-configuração-global)
4. [Estado da Rede (NetworkState)](#4-estado-da-rede-networkstate)
5. [Loop de Controle (6 Etapas)](#5-loop-de-controle-6-etapas)
   - [Etapa 1 — fetch_topology()](#etapa-16-fetch_topology)
   - [Etapa 2 — fetch_hosts() e probe_hosts()](#etapa-26-fetch_hosts-e-probe_hosts)
   - [Etapa 3 — monitor_traffic_load()](#etapa-36-monitor_traffic_load)
   - [Etapa 4 — install_ipv4_routes()](#etapa-46-install_ipv4_routes)
   - [Etapa 5 — install_arp_spanning_tree()](#etapa-56-install_arp_spanning_tree)
   - [Etapa 6 — check_and_reroute()](#etapa-66-check_and_reroute)
6. [Watchdogs Periódicos](#6-watchdogs-periódicos)
7. [Infraestrutura (OVS e Docker)](#7-infraestrutura-ovs-e-docker)
8. [Especificação de Flows OpenFlow](#8-especificação-de-flows-openflow)
9. [API REST (FastAPI)](#9-api-rest-fastapi)
10. [Coleta de Métricas](#10-coleta-de-métricas)
11. [Modelo de Threads e Concorrência](#11-modelo-de-threads-e-concorrência)
12. [Inicialização e Execução](#12-inicialização-e-execução)
13. [Princípios de Design (SOLID)](#13-princípios-de-design-solid)

---

## 1. Visão Geral

O SDN Orchestrator é um **controlador de rede proativo** que gerencia uma rede de switches Open vSwitch (OVS) com OpenDaylight (ODL) como backend. Ele executa um loop de controle periódico a cada 5 segundos, descobrindo a topologia, calculando rotas ótimas (Dijkstra), instalando flows OpenFlow 1.3 nos switches via `docker exec ovs-ofctl`, e aplicando roteamento dinâmico em caso de congestionamento.

```
┌────────────────────────────────────────────────────────┐
│                  SDN Orchestrator v14.0                 │
│                                                         │
│  ┌──────────────┐    ┌────────────────────────────────┐ │
│  │ Control Loop │◄──►│  NetworkState (singleton)      │ │
│  │  (daemon     │    │  graph, hosts, flows, metrics  │ │
│  │   thread)    │    └────────────────────────────────┘ │
│  └──────┬───────┘                  ▲                    │
│         │                          │                    │
│  ┌──────▼───────────────────────── ┤ ──────────────┐   │
│  │         6-Stage Pipeline        │               │   │
│  │  ODL REST ──► Dijkstra ──► OVS (docker exec)   │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  FastAPI REST Server (porta 8000)                │  │
│  │  /health  /state  /flows  /metrics  /manage      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Tecnologias-chave:**
- **OpenDaylight (ODL)** — controlador SDN que fornece a visão da topologia via REST (RESTS v2)
- **Open vSwitch (OVS)** — switches virtuais rodando em containers Docker (GNS3/GNS3.OVS-*)
- **NetworkX** — grafo da topologia para cálculo de rotas (Dijkstra, MST)
- **FastAPI + Uvicorn** — servidor REST para gerenciamento e consulta de estado
- **ThreadPoolExecutor** — instalação paralela de flows (max 20 workers simultâneos)

---

## 2. Arquitetura e Estrutura de Diretórios

O projeto segue **Clean Architecture** com separação em quatro camadas:

```
sdn/
├── sdn_orchestrator.py          # Entry point (importa e chama main())
└── orchestrator/
    ├── config.py                # [CONFIG] Constantes globais e URLs
    ├── main.py                  # [CONTROLE] Loop principal + inicialização
    ├── domain/
    │   ├── state.py             # [DOMÍNIO] NetworkState + singletons
    │   └── models.py            # [DOMÍNIO] Pydantic models para API
    ├── application/
    │   ├── topology.py          # [APP] Etapa 1/6: topologia e watchdogs
    │   ├── hosts.py             # [APP] Etapa 2/6: hosts + ARP probing
    │   ├── traffic.py           # [APP] Etapa 3/6: monitoramento de tráfego
    │   └── routing.py           # [APP] Etapas 4-6/6: rotas, MST, reroute
    ├── infrastructure/
    │   ├── docker_adapter.py    # [INFRA] Mapeamento container Docker ↔ switch
    │   ├── ovs_adapter.py       # [INFRA] Comunicação com OVS via docker exec
    │   └── flow_specs.py        # [INFRA] Construção de strings de flows OVS
    ├── presentation/
    │   └── api.py               # [API] Endpoints FastAPI
    └── utils/
        ├── metrics_collector.py # [UTIL] Coleta e escrita de métricas CSV
        ├── sdn_tools.py         # [UTIL] CLI de manutenção
        └── sdn_verify.py        # [UTIL] Validação e diagnóstico
```

**Camadas e suas responsabilidades:**

| Camada | Responsabilidade | Pode depender de |
|---|---|---|
| **domain** | Entidades e estado central | Ninguém |
| **application** | Casos de uso do negócio | domain |
| **infrastructure** | Acesso a sistemas externos | domain |
| **presentation** | Entrada/saída (HTTP) | application, domain |
| **config** | Constantes de ambiente | Ninguém |

---

## 3. Configuração Global

**Arquivo:** `orchestrator/config.py`

Todas as constantes de ambiente ficam centralizadas aqui (princípio SRP — uma única fonte de verdade para configuração):

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `ODL_IP` | `172.16.1.1` | IP do controlador OpenDaylight |
| `ODL_PORT` | `8181` | Porta REST do ODL |
| `ODL_USER` / `ODL_PASS` | `admin/admin` | Credenciais do ODL |
| `TOPOLOGY_ID` | `flow:1` | ID da topologia no ODL |
| `POLLING_INTERVAL` | `5` s | Intervalo do loop de controle |
| `MAX_LINK_CAPACITY` | `20.000.000` bps | Capacidade máxima dos enlaces (20 Mbps) |
| `WARN_THRESH` | `0.50` | 50% — limiar de aviso |
| `REROUTE_THRESH` | `0.65` | 65% — limiar de rerouting |
| `CONGESTED_THRESH` | `0.90` | 90% — limiar crítico de congestionamento |
| `HOST_TTL_CYCLES` | `3` | Ciclos até remover host ausente |
| `SW_TTL_CYCLES` | `3` | Ciclos até remover switch ausente |
| `HOST_PROBE_CYCLES` | `6` | A cada N ciclos, envia ARP probe |
| `HOST_PROBE_MISS` | `3` | Probes sem resposta até remover host |
| `FL_SERVER_SWITCH` | `openflow:1` | Switch core do servidor FL |
| `DOCKER_NAME_PREFIX` | `GNS3.OVS-` | Prefixo dos containers Docker |
| `ENABLE_REROUTING` | `True` | Habilita/desabilita rerouting dinâmico |

**URLs construídas automaticamente:**
```
BASE         = http://172.16.1.1:8181
URL_TOPO     = BASE/rests/data/network-topology:network-topology/topology=flow:1
URL_NODES    = BASE/rests/data/opendaylight-inventory:nodes
URL_NODES_OP = BASE/.../nodes?content=nonconfig   (dados operacionais)
URL_TOPO_OP  = BASE/.../topology=flow:1?content=nonconfig
```

---

## 4. Estado da Rede (NetworkState)

**Arquivo:** `orchestrator/domain/state.py`

O `NetworkState` é um **singleton** que centraliza todo o estado mutável da rede. O acesso concorrente é protegido por um único `threading.Lock`.

```python
state: NetworkState = NetworkState()   # singleton global
CYCLE_COUNT: int = 0                   # contador de ciclos (incrementado pelo loop)
```

### Atributos do NetworkState

| Atributo | Tipo | Descrição |
|---|---|---|
| `lock` | `threading.Lock` | Mutex para acesso thread-safe |
| `graph` | `nx.Graph` | Grafo da topologia: nós = switches, arestas = enlaces |
| `hosts_by_mac` | `dict[str, dict]` | `mac → {mac, ips, switch, port}` |
| `ip_to_mac` | `dict[str, str]` | Índice reverso: `ip → mac` |
| `edge_ports` | `dict[str, set]` | Portas de borda por switch: `sw_id → set(port_nums)` |
| `sw_to_container` | `dict[str, str]` | `openflow:X → nome_container_docker` |
| `active_flows` | `dict[tuple, str]` | `(sw_id, flow_id) → ovs_flow_str` |
| `blocked_switches` | `list[str]` | Switches isolados via API |
| `blocked_ips` | `list[str]` | IPs bloqueados via API |
| `pending_unblocks` | `set[str]` | IPs em processo de desbloqueio (evita race condition) |
| `port_stats` | `dict` | Contagens de bytes por porta ODL (ciclo atual) |
| `link_load` | `dict[tuple, float]` | `(u,v) → bps atual` |
| `link_costs` | `dict[tuple, int]` | `(u,v) → custo Dijkstra` |
| `_valid_switches` | `set[str]` | Switches com tabelas de flow confirmadas no ODL |
| `_guard_done` | `set[str]` | Switches com flows base já instalados (TABLE-MISS, LLDP, BDDP) |
| `_prev_edges` | `frozenset` | Arestas do ciclo anterior (detecção de mudança) |
| `topo_changed` | `bool` | Flag: topologia mudou desde o último cálculo de MST |
| `_flood_blocks` | `set[tuple]` | Flows DROP anti-storm ativos: `{(sw_id, flow_id)}` |
| `host_missing_cycles` | `dict[str, int]` | Ciclos consecutivos sem ver o host no ODL |
| `_sw_missing_cycles` | `dict[str, int]` | Ciclos consecutivos sem ver o switch no ODL |
| `_host_probe_sent` | `dict[str, int]` | Ciclo do último ARP probe enviado por MAC |

### Padrão de uso thread-safe

```python
# Leitura atômica do estado (sempre com lock)
with state.lock:
    switches = list(state.graph.nodes)
    hosts    = dict(state.hosts_by_mac)

# Escrita de múltiplos atributos (mantém consistência)
with state.lock:
    state.graph = new_graph
    state.topo_changed = True
```

---

## 5. Loop de Controle (6 Etapas)

**Arquivo:** `orchestrator/main.py`

O loop executa em uma **thread daemon** a cada `POLLING_INTERVAL` (5 s). A thread principal fica ocupada rodando o servidor FastAPI (Uvicorn).

```
CICLO N
  │
  ├─ [watchdog a cada 3 ciclos] remove_rogue_arp_flows()
  ├─ [watchdog a cada 5 ciclos] discover_containers()
  ├─ [watchdog a cada 10 ciclos] refresh_valid_switches()
  │
  ├─ [1/6] fetch_topology()           → atualiza state.graph
  ├─ [2/6] fetch_hosts()              → atualiza state.hosts_by_mac
  ├─ [2b]  probe_hosts()              → envia ARP probe se necessário
  ├─ [3/6] monitor_traffic_load()     → atualiza state.link_load
  ├─ [4/6] install_ipv4_routes()      → instala flows IPv4 via OVS
  ├─ [5/6] install_arp_spanning_tree()→ instala flows ARP (MST)
  ├─ [6/6] check_and_reroute()        → instala flows de desvio
  │
  ├─ MetricsCollector.collect()       → escreve linha no CSV
  └─ time.sleep(POLLING_INTERVAL)
```

---

### Etapa 1/6: fetch_topology()

**Arquivo:** `orchestrator/application/topology.py`

**Responsabilidade:** Buscar switches e enlaces da topologia via REST do ODL.

**Fluxo:**
1. `GET URL_TOPO` — obtém a topologia do ODL (switches como nós, enlaces como arestas)
2. Constrói novo `nx.Graph` com switches como nós e enlaces com atributos `src_port`/`dst_port`
3. Detecta mudanças comparando com `state._prev_edges`
4. Gerencia TTL de switches: incrementa `_sw_missing_cycles[sw]` para switches que sumiram; remove após `SW_TTL_CYCLES` ciclos
5. Identifica switches novos e instala flows base: `TABLE-MISS → controller`, `LLDP → controller`, `BDDP → controller`
6. Atualiza `state.graph`, `state.topo_changed`, `state.edge_ports`

**Custos de enlace (para Dijkstra):**

| Utilização | Custo atribuído |
|---|---|
| < 20% | 1 (idle) |
| 20–50% | 3 (moderado) |
| 50–65% | 10 (aviso) |
| > 90% | 100 (congestionado) |

---

### Etapa 2/6: fetch_hosts() e probe_hosts()

**Arquivo:** `orchestrator/application/hosts.py`

**fetch_hosts():**
1. `GET URL_NODES_OP` — obtém hosts registrados pelo l2switch do ODL
2. Para cada host: extrai MAC, IPs, switch de acesso e porta
3. Gerencia TTL: `host_missing_cycles[mac]` — remove após `HOST_TTL_CYCLES` ciclos
4. Atualiza `state.hosts_by_mac` e `state.ip_to_mac`

**probe_hosts():**
- A cada `HOST_PROBE_CYCLES` ciclos, envia ARP probe (via `ovs-ofctl packet-out`) para todos os hosts conhecidos
- Se o host não responder após `HOST_PROBE_MISS` probes consecutivos, é removido do estado
- Evita manter hosts "fantasmas" que desconectaram sem notificar o ODL

---

### Etapa 3/6: monitor_traffic_load()

**Arquivo:** `orchestrator/application/traffic.py`

**Responsabilidade:** Calcular a utilização de largura de banda de cada enlace.

**Fluxo:**
1. `GET URL_NODES_OP` — lê contadores de bytes por porta de cada switch
2. Calcula delta de bytes entre ciclos: `Δbytes = bytes_atual - bytes_anterior`
3. Converte para bps: `bps = Δbytes × 8 / POLLING_INTERVAL`
4. Normaliza: `utilização = bps / MAX_LINK_CAPACITY`
5. Atualiza `state.link_load[(u,v)]` e `state.link_costs[(u,v)]` baseado nos limiares

---

### Etapa 4/6: install_ipv4_routes()

**Arquivo:** `orchestrator/application/routing.py`

**Responsabilidade:** Calcular e instalar rotas IPv4 proativas usando Dijkstra.

**Fluxo:**
1. Para cada host de destino (IP → switch):
   - Executa `nx.single_source_dijkstra(graph, dst_switch, weight="weight")`
   - Para cada switch no caminho: constrói flow `priority=60000, ip, nw_dst=DST_IP, actions=output:iface`
2. Acumula todos os flows de todos os switches em lista de tasks
3. Submete para `install_flows_parallel()` — todos instalados em paralelo com ThreadPoolExecutor
4. Flows são permanentes (sem `idle_timeout`); só sobrescritos quando a topologia muda

**Nota de performance:** Com 10 switches × 15 flows cada = 150 `docker exec` sequenciais levaria ~2-3 min. Com `max_workers=20`, o tempo cai para ~3s (tempo de 1 exec).

---

### Etapa 5/6: install_arp_spanning_tree()

**Arquivo:** `orchestrator/application/routing.py`

**Responsabilidade:** Evitar tempestades de broadcast ARP usando Spanning Tree (MST).

**Fluxo:**
1. Recalcula MST apenas se `topo_changed = True`: `nx.minimum_spanning_tree(graph)`
2. Para cada switch e cada porta de entrada:
   - Se a porta está no MST: instala `priority=5000, arp, in_port=X, actions=FLOOD`
   - Se a porta não está no MST: instala `priority=3, arp, in_port=X, actions=DROP`
3. Remove flows DROP de portas que agora estão no MST (quando topologia muda)
4. Instala `priority=4999, arp, actions=CONTROLLER:65535` como catch-all ARP

**Por que MST para ARP?** Sem MST, um pacote ARP poderia circular indefinidamente em topologias com loops, consumindo toda a largura de banda.

---

### Etapa 6/6: check_and_reroute()

**Arquivo:** `orchestrator/application/routing.py`

**Responsabilidade:** Detectar enlaces congestionados e instalar rotas alternativas temporárias.

**Fluxo:**
1. Verifica `link_load` para enlaces com utilização > `REROUTE_THRESH` (65%)
2. Para cada enlace congestionado: adiciona custo temporário ao grafo e recalcula Dijkstra
3. Instala flows de desvio: `priority=62000, idle_timeout=15s` — maior prioridade que os flows permanentes (60000), expiram automaticamente em 15 s
4. Após 15 s sem tráfego no flow, o switch remove automaticamente o flow de desvio e o tráfego volta ao caminho original

---

## 6. Watchdogs Periódicos

Executados pelo loop de controle em intervalos fixos:

| Frequência | Função | Propósito |
|---|---|---|
| **A cada 3 ciclos** | `remove_rogue_arp_flows()` | Remove flows `priority=65000,arp` instalados pelo arphandler do ODL que causam flood massivo e CPU 99% |
| **A cada 5 ciclos** | `discover_containers()` | Re-descobre containers Docker para o caso de reinicialização de switches |
| **A cada 10 ciclos** | `refresh_valid_switches()` | Atualiza o conjunto de switches com tabelas de flow confirmadas no ODL |

**Watchdog anti-rogue (detalhe):**
O arphandler do ODL instala automaticamente `priority=65000, arp, actions=CONTROLLER:65535, output:ALL`, que gera flood massivo. O watchdog verifica e remove esses flows a cada 3 ciclos usando `ovs-ofctl dump-flows | grep priority=65000` seguido de `ovs-ofctl del-flows`.

---

## 7. Infraestrutura (OVS e Docker)

### 7.1 Docker Adapter

**Arquivo:** `orchestrator/infrastructure/docker_adapter.py`

Mapeia switches OpenFlow (`openflow:X`) para containers Docker (`GNS3.OVS-X`):

```python
# Descoberta: docker ps → filtra por DOCKER_NAME_PREFIX → extrai DPID
# Mapeamento: state.sw_to_container["openflow:1"] = "GNS3.OVS-1"
discover_containers()   # chama no boot e a cada 5 ciclos
container_for(sw_id)    # retorna nome do container ou None
```

### 7.2 OVS Adapter

**Arquivo:** `orchestrator/infrastructure/ovs_adapter.py`

Toda comunicação com o OVS passa por este módulo (SRP). Executa `docker exec <container> ovs-ofctl <cmd> br0 <args> -O OpenFlow13`.

**Funções principais:**

```python
install_flow(sw_id, flow_id, ovs_flow_str)      # instala 1 flow
delete_flow(sw_id, flow_id, ovs_flow_str)        # remove 1 flow
install_flows_parallel(tasks)                    # instala N flows em paralelo
delete_flows_parallel(tasks)                     # remove N flows em paralelo
verify_table_miss(sw_id)                         # verifica TABLE-MISS instalado
port_to_iface(port)                              # "1" → "eth1"
```

**ThreadPoolExecutor:**
```python
FLOW_EXECUTOR = ThreadPoolExecutor(max_workers=20, thread_name_prefix="flow-install")
```

---

## 8. Especificação de Flows OpenFlow

**Arquivo:** `orchestrator/infrastructure/flow_specs.py`

Funções que constroem as strings no formato `ovs-ofctl` (OpenFlow 1.3):

| Função | Prioridade | Descrição | Exemplo de flow |
|---|---|---|---|
| `flow_table_miss()` | 0 | TABLE-MISS → controller | `priority=0,actions=CONTROLLER:65535` |
| `flow_lldp()` | 1000 | LLDP → controller | `priority=1000,dl_type=0x88cc,actions=CONTROLLER:65535` |
| `flow_bddp()` | 1000 | BDDP → controller | `priority=1000,dl_type=0xbaf4,actions=CONTROLLER:65535` |
| `flow_arp_mst(port)` | 5000 | ARP loop-free via MST | `priority=5000,arp,in_port=1,actions=FLOOD` |
| `flow_flood_block(port)` | 3 | DROP flooding em porta não-MST | `priority=3,arp,in_port=2,actions=DROP` |
| `flow_ipv4_forward(dst_ip, iface, prio)` | 60000/62000 | Rota IPv4 proativa | `priority=60000,ip,nw_dst=10.0.0.2,actions=output:eth2` |
| `flow_ip_drop(ip)` | 65500 | Bloqueia IP (DROP) | `priority=65500,ip,nw_src=10.0.0.5,actions=DROP` |

**Tabela de prioridades completa:**

| Prioridade | Tipo | Descrição |
|---|---|---|
| 65500 | IP DROP | Bloqueio manual de IP via API |
| 62000 | IPv4 reroute | Desvio temporário (idle_timeout=15s) |
| 60000 | IPv4 proativo | Rota Dijkstra permanente |
| 5000 | ARP MST | ARP loop-free (porta MST) |
| 4999 | ARP catch-all | → CONTROLLER (qualquer ARP restante) |
| 3 | ARP DROP | Bloqueia flood em porta não-MST |
| 1000 | LLDP/BDDP | → CONTROLLER (descoberta de topologia) |
| 0 | TABLE-MISS | → CONTROLLER (qualquer pacote não classificado) |

---

## 9. API REST (FastAPI)

**Arquivo:** `orchestrator/presentation/api.py`  
**Porta:** `0.0.0.0:8000`

### Endpoints de Gerenciamento

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/manage/switch` | Bloqueia ou desbloqueia um switch |
| `POST` | `/manage/ip` | Bloqueia ou desbloqueia um IP (instala/remove flow DROP) |

### Endpoints de Consulta

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/health` | Status: `{cycle_count, n_switches, n_hosts, n_containers, uptime}` |
| `GET` | `/state` | Snapshot completo do NetworkState em JSON |
| `GET` | `/flows/{sw_id}` | Dump dos flows OpenFlow de um switch específico |
| `GET` | `/metrics/links` | Utilização atual dos enlaces em bps e % |
| `GET` | `/metrics/hosts` | Análise de bottleneck por host (caminho até FL server) |

### Endpoints FL/QoS

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/qos/apply` | Aplica marcação DSCP para clientes FL (priorização de tráfego) |
| `DELETE` | `/qos/{client_id}` | Remove flows de QoS de um cliente FL |
| `POST` | `/fl/training/start` | Inicia CSV separado de métricas para sessão FL |
| `POST` | `/fl/training/stop` | Finaliza CSV de sessão FL |

**Exemplo de resposta `/health`:**
```json
{
  "status": "running",
  "cycle_count": 42,
  "n_switches": 6,
  "n_hosts": 8,
  "n_containers": 6,
  "uptime_seconds": 210.5
}
```

**QoS — Categorias de prioridade FL:**

| Categoria | DSCP | Prioridade OpenFlow |
|---|---|---|
| cat1 (modelos pequenos) | EF (46) | Máxima |
| cat2 (modelos médios) | AF31 (26) | Média |
| cat3 (modelos grandes) | BE (0) | Best-effort |

---

## 10. Coleta de Métricas

**Arquivo:** `orchestrator/utils/metrics_collector.py`

A cada ciclo, o `MetricsCollector` escreve uma linha em `sdn_metrics_{timestamp}.csv`:

| Campo | Descrição |
|---|---|
| `timestamp` | Timestamp ISO 8601 |
| `cycle` | Número do ciclo |
| `elapsed_sec` | Segundos desde o início |
| `cycle_duration_sec` | Duração do ciclo (tempo de execução das 6 etapas) |
| `n_switches` | Número de switches na topologia |
| `n_hosts` | Número de hosts descobertos |
| `n_flows` | Total de flows ativos no estado |
| `n_reroute_flows` | Flows de rerouting ativos (priority=62000) |
| `max_link_load_bps` | Pico de carga entre todos os enlaces |
| `avg_link_load_bps` | Média de carga dos enlaces |
| `congested_links` | Número de enlaces com utilização > CONGESTED_THRESH |
| `warn_links` | Número de enlaces com utilização > WARN_THRESH |

**Sessão FL separada:** Os endpoints `/fl/training/start` e `/fl/training/stop` criam um CSV adicional isolado para correlacionar métricas de rede com o treinamento federado.

---

## 11. Modelo de Threads e Concorrência

```
Thread MAIN (uvicorn)
  └── FastAPI HTTP server (porta 8000)
       └── Lê state com state.lock

Thread DAEMON (control_loop)
  └── Loop principal a cada 5s
       ├── Lê e escreve state com state.lock
       └── Submete tasks ao FLOW_EXECUTOR

ThreadPool FLOW_EXECUTOR (max_workers=20)
  └── install_flow() → docker exec (I/O bound)
       └── Escreve state.active_flows com state.lock
```

**Regras de concorrência:**
1. Todo acesso a `NetworkState` é feito dentro de `with state.lock:` — garante atomicidade
2. O loop de controle lê o estado, realiza cálculos fora do lock, e só reentra no lock para escrita
3. O `FLOW_EXECUTOR` usa o lock apenas para atualizar `state.active_flows` (write path curto)
4. A API FastAPI só lê o estado (exceto `/manage/*`) — contenção mínima

---

## 12. Inicialização e Execução

**Arquivo:** `sdn_orchestrator.py` → `orchestrator/main.py`

```bash
python sdn_orchestrator.py
```

**Sequência de boot:**
```
1. Teste de conectividade com ODL (GET URL_TOPO, timeout=5s)
2. discover_containers() — mapeia GNS3.OVS-* → openflow:*
3. refresh_valid_switches() — popula state._valid_switches
4. MetricsCollector() — inicia CSV de métricas
5. Thread daemon: control_loop() — inicia o loop periódico
6. uvicorn.run(app, 0.0.0.0:8000) — sobe o servidor FastAPI
```

**Dependências Python:**
```
requests        # REST API ODL
uvicorn         # ASGI server para FastAPI
fastapi         # Framework REST
networkx        # Topologia (Dijkstra, MST)
pandas          # CSV de métricas
pydantic        # Modelos da API
```

---

## 13. Princípios de Design (SOLID)

| Princípio | Aplicação no projeto |
|---|---|
| **S** — Single Responsibility | Cada módulo tem uma responsabilidade. `topology.py` gerencia apenas topologia. `ovs_adapter.py` gerencia apenas comunicação OVS. `config.py` gerencia apenas configuração. |
| **O** — Open/Closed | Novos algoritmos de roteamento podem ser adicionados criando novas funções em `routing.py` sem modificar o loop de controle em `main.py`. |
| **L** — Liskov Substitution | As funções de aplicação recebem o estado via singletons e retornam resultados consistentes — qualquer implementação alternativa pode substituí-las no loop. |
| **I** — Interface Segregation | A API REST expõe endpoints pequenos e especializados. A camada de infraestrutura tem funções pequenas e focadas (`install_flow`, `delete_flow`, `container_for`). |
| **D** — Dependency Inversion | As camadas de aplicação dependem das abstrações do domínio (`state`, `config`) e não das implementações concretas de infraestrutura diretamente — `routing.py` não conhece `docker exec`, só chama `install_flows_parallel()`. |

---

*Documentação gerada com base no código em `sdn/orchestrator/` — SDN Orchestrator v14.0.*
