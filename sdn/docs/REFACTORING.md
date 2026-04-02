# Refatoração do SDN Orchestrator v14.0
## Clean Architecture + Princípios SOLID

---

## Contexto

O arquivo original `sdn_orchestrator.py` (~1480 linhas) implementava o orquestrador
SDN inteiramente em um único módulo monolítico. Toda a lógica — configuração,
estado global, comunicação com o ODL, instalação de flows no OVS, roteamento e
API REST — estava acoplada em funções globais sem separação de responsabilidades.

A refatoração **não alterou nenhuma linha de lógica**. O objetivo foi exclusivamente
reorganizar o código em camadas bem definidas, seguindo Clean Architecture e os
cinco princípios SOLID.

---

## Estrutura Antes e Depois

### Antes — monolítico

```
sdn_orchestrator.py  (1480 linhas)
├── Constantes globais (ODL_IP, URLs, thresholds…)
├── class NetworkState
├── class SwitchRequest / IPBlockRequest (Pydantic)
├── _discover_containers(), _container_for()
├── _install_flow(), _install_flows_parallel(), _delete_flow()…
├── _flow_table_miss(), _flow_lldp(), _flow_ipv4_forward()…
├── _link_key(), _out_port()
├── fetch_topology(), _install_base_flows(), _remove_rogue_arp_flows()
├── fetch_hosts(), probe_hosts(), _send_arp_probe()
├── monitor_traffic_load()
├── install_ipv4_routes(), install_arp_spanning_tree(), check_and_reroute()
├── API REST (@app.post, @app.get…)
└── control_loop() + __main__
```

### Depois — Clean Architecture em 4 camadas

```
sdn_orchestrator.py          ← entry point (5 linhas úteis)
orchestrator/
├── config.py                ← [Configuração]
├── domain/
│   ├── state.py             ← [Domínio] Entidade + Singleton
│   └── models.py            ← [Domínio] Contratos da API
├── infrastructure/
│   ├── flow_specs.py        ← [Infra] Regras de flow OVS
│   ├── docker_adapter.py    ← [Infra] Adaptador Docker
│   └── ovs_adapter.py       ← [Infra] Adaptador OVS
├── application/
│   ├── topology.py          ← [App] Caso de uso: topologia
│   ├── hosts.py             ← [App] Caso de uso: hosts
│   ├── traffic.py           ← [App] Caso de uso: tráfego
│   └── routing.py           ← [App] Caso de uso: roteamento
├── presentation/
│   └── api.py               ← [Apresentação] Rotas FastAPI
└── main.py                  ← Loop de controle + inicialização
```

---

## Mapeamento Detalhado: O Que Foi Para Onde

### `orchestrator/config.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 67–88 | Todas as constantes globais (`ODL_IP`, `ODL_PORT`, `MAX_LINK_CAPACITY`, `POLLING_INTERVAL`, thresholds, URLs, `AUTH`, `HEADERS_JSON`) | `orchestrator/config.py` |
| 91 | `DOCKER_NAME_PREFIX` | `orchestrator/config.py` |

---

### `orchestrator/domain/state.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 101–128 | `class NetworkState` (todos os atributos) | `orchestrator/domain/state.py` |
| 131 | `state = NetworkState()` (singleton global) | `orchestrator/domain/state.py` |
| 98 | `CYCLE_COUNT = 0` (global) | `orchestrator/domain/state.py` (como variável de módulo) |

**Por que CYCLE_COUNT está aqui?**
Ele precisa ser compartilhado entre `topology.py` (`install_base_flows` o lê),
`hosts.py` (`probe_hosts` o lê) e `main.py` (o incrementa). Colocá-lo em
`domain/state.py` fornece um único ponto de importação para todos, sem dependência
circular.

---

### `orchestrator/domain/models.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 137–144 | `class SwitchRequest(BaseModel)` | `orchestrator/domain/models.py` |
| 147–149 | `class IPBlockRequest(BaseModel)` | `orchestrator/domain/models.py` |

---

### `orchestrator/infrastructure/flow_specs.py`

Funções **puras** (sem I/O, sem estado) que encapsulam as regras de sintaxe
do ovs-ofctl. Correspondem diretamente à tabela de prioridades do cabeçalho original.

| Origem (linha) | O que era | Destino |
|---|---|---|
| 304–305 | `_flow_table_miss()` | `flow_table_miss()` |
| 307–308 | `_flow_lldp()` | `flow_lldp()` |
| 310–311 | `_flow_bddp()` | `flow_bddp()` |
| 313–327 | `_flow_ipv4_forward()` | `flow_ipv4_forward()` |
| 329–335 | `_flow_arp_mst()` | `flow_arp_mst()` |
| 337–338 | `_flow_ip_drop()` | `flow_ip_drop()` |
| 340–349 | `_flow_flood_block()` | `flow_flood_block()` |

> O prefixo `_` foi removido porque as funções agora são a API pública do módulo.

---

### `orchestrator/infrastructure/docker_adapter.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 150–195 | `_discover_containers()` | `discover_containers()` |
| 198–200 | `_container_for()` | `container_for()` |

---

### `orchestrator/infrastructure/ovs_adapter.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 95 | `FLOW_EXECUTOR = ThreadPoolExecutor(...)` | `orchestrator/infrastructure/ovs_adapter.py` |
| 206–208 | `_port_to_iface()` | `port_to_iface()` |
| 211–239 | `_install_flow()` | `install_flow()` |
| 242–269 | `_install_flows_parallel()` | `install_flows_parallel()` |
| 272–287 | `_delete_flow()` | `delete_flow()` |
| 290–298 | `_delete_flows_parallel()` | `delete_flows_parallel()` |
| 428–441 | `_verify_table_miss()` | `verify_table_miss()` |
| 1319–1345 | `_delete_ip_block_direct()` | `delete_ip_block_direct()` |

> `_delete_ip_block_direct` foi movida da seção de API REST para a camada de
> infraestrutura, pois é uma operação direta de I/O no OVS — não pertence à
> camada de apresentação.

---

### `orchestrator/application/topology.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 355–356 | `_link_key()` | `link_key()` |
| 358–363 | `_out_port()` | `out_port()` |
| 369–386 | `_refresh_valid_switches()` | `refresh_valid_switches()` |
| 389–424 | `_remove_rogue_arp_flows()` | `remove_rogue_arp_flows()` |
| 444–494 | `_install_base_flows()` | `install_base_flows()` |
| 498–640 | `fetch_topology()` | `fetch_topology()` |

> `link_key` e `out_port` estão aqui pois são utilitários de topologia
> também usados por `routing.py`, que importa do mesmo módulo.

---

### `orchestrator/application/hosts.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 644–688 | `_send_arp_probe()` | `_send_arp_probe()` (privado ao módulo) |
| 691–734 | `probe_hosts()` | `probe_hosts()` |
| 738–894 | `fetch_hosts()` | `fetch_hosts()` |

---

### `orchestrator/application/traffic.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 899 | `_last_traffic_ts: float = 0.0` | variável de módulo em `traffic.py` |
| 901–1009 | `monitor_traffic_load()` | `monitor_traffic_load()` |

---

### `orchestrator/application/routing.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 1013–1105 | `install_ipv4_routes()` | `install_ipv4_routes()` |
| 1109–1245 | `install_arp_spanning_tree()` | `install_arp_spanning_tree()` |
| 1249–1301 | `check_and_reroute()` | `check_and_reroute()` |

---

### `orchestrator/presentation/api.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 97 | `app = FastAPI(...)` | `orchestrator/presentation/api.py` |
| 1306–1316 | `@app.post("/manage/switch")` | `manage_switch()` |
| 1348–1388 | `@app.post("/manage/ip")` | `manage_ip()` |
| 1391–1400 | `@app.get("/health")` | `health()` |
| 1403–1413 | `@app.get("/state")` | `get_state()` |
| 1416–1430 | `@app.get("/flows/{sw_id}")` | `get_flows()` |

---

### `orchestrator/main.py`

| Origem (linha) | O que era | Destino |
|---|---|---|
| 1436–1473 | `control_loop()` | `control_loop()` |
| 1476–1479 | bloco `if __name__ == "__main__"` | função `main()` |

---

## Como o Funcionamento Foi Preservado

### 1. Zero alteração de lógica
Cada linha de código de negócio foi copiada integralmente para o módulo
correspondente. Nenhuma condição, loop, chamada de função ou efeito colateral
foi modificado.

### 2. Estado global compartilhado via singleton
O objeto `state = NetworkState()` continua sendo **uma única instância**
compartilhada por todas as camadas. Todos os módulos importam do mesmo lugar:

```python
from orchestrator.domain.state import state
```

Como Python garante que um módulo é executado apenas uma vez (cached em
`sys.modules`), todos os imports recebem a mesma instância.

### 3. `CYCLE_COUNT` como variável de módulo
O contador de ciclos precisa ser incrementado em `main.py` e lido em
`topology.py` e `hosts.py`. A solução foi manter como variável de módulo
em `domain/state.py` e usar o padrão de import de módulo para escrita:

```python
# Em main.py — incrementa
import orchestrator.domain.state as state_module
state_module.CYCLE_COUNT += 1

# Em topology.py — lê
import orchestrator.domain.state as state_module
if state_module.CYCLE_COUNT % 15 == 1: ...
```

Isso preserva a semântica de "variável global" sem introduzir objetos mutáveis
desnecessários.

### 4. `FLOW_EXECUTOR` compartilhado
O `ThreadPoolExecutor` continua sendo **uma única instância** (definida em
`ovs_adapter.py`). A camada de apresentação (`api.py`) importa diretamente:

```python
from orchestrator.infrastructure.ovs_adapter import FLOW_EXECUTOR
```

O pool de 20 workers é o mesmo utilizado por todas as operações paralelas.

### 5. `sdn_orchestrator.py` como thin wrapper
O arquivo original foi reduzido a um entry point de compatibilidade:

```python
from orchestrator.main import main
if __name__ == "__main__":
    main()
```

O comando `python sdn_orchestrator.py` continua funcionando exatamente como antes.

---

## Princípios SOLID Aplicados

### S — Single Responsibility Principle
Cada módulo tem exatamente **uma razão para mudar**:

| Módulo | Única responsabilidade |
|---|---|
| `config.py` | Alterar parâmetros de ambiente |
| `domain/state.py` | Alterar a estrutura de estado da rede |
| `infrastructure/flow_specs.py` | Alterar a sintaxe de flows OVS |
| `infrastructure/docker_adapter.py` | Alterar a forma de descobrir containers |
| `infrastructure/ovs_adapter.py` | Alterar o mecanismo de instalação de flows |
| `application/topology.py` | Alterar a lógica de descoberta de topologia |
| `application/hosts.py` | Alterar a lógica de descoberta/TTL de hosts |
| `application/traffic.py` | Alterar a lógica de monitoramento de tráfego |
| `application/routing.py` | Alterar a lógica de roteamento e MST |
| `presentation/api.py` | Alterar os contratos da API REST |

### O — Open/Closed Principle
As camadas de aplicação dependem de **interfaces funcionais estáveis** da
infraestrutura. Para adicionar, por exemplo, um novo mecanismo de instalação
de flows (ex: gRPC em vez de docker exec), basta criar um novo módulo em
`infrastructure/` e atualizar os imports em `ovs_adapter.py` — os casos de
uso em `application/` **não precisam ser modificados**.

### L — Liskov Substitution Principle
As funções construtoras de flow (`flow_specs.py`) têm assinaturas estáveis e
retornam sempre `str`. Qualquer implementação alternativa que respeite esses
contratos pode substituí-las sem quebrar os chamadores.

### I — Interface Segregation Principle
Cada módulo importa **apenas o que precisa**. Por exemplo, `traffic.py` importa
somente `link_key` de `topology.py`, sem depender de `fetch_topology` ou
`install_base_flows`. Nenhum módulo carrega dependências que não usa.

### D — Dependency Inversion Principle
A camada de **aplicação** não conhece detalhes de Docker ou OVS diretamente —
ela depende das funções exportadas pela **infraestrutura** (`install_flows_parallel`,
`delete_flows_parallel`, `port_to_iface`). Se a tecnologia de transporte mudar,
apenas a camada de infraestrutura precisa ser reescrita.

---

## Grafo de Dependências (sem ciclos)

```
config.py
    ↑
domain/state.py ← domain/models.py
    ↑
infrastructure/flow_specs.py
infrastructure/docker_adapter.py  ← config + domain/state
infrastructure/ovs_adapter.py     ← config + domain/state + docker_adapter
    ↑
application/topology.py  ← config + domain/state + infrastructure/*
application/hosts.py     ← config + domain/state + infrastructure/ovs_adapter
application/traffic.py   ← config + domain/state + application/topology
application/routing.py   ← config + domain/state + infrastructure/* + application/topology
    ↑
presentation/api.py  ← domain/* + infrastructure/ovs_adapter + infrastructure/docker_adapter
    ↑
main.py  ← config + domain/state + infrastructure/docker_adapter + application/* + presentation/api
    ↑
sdn_orchestrator.py  ← main
```

Cada seta representa "depende de". **Não há ciclos** — cada camada depende
apenas das camadas abaixo dela.

---

## Como Executar

```bash
# Sem mudança no comando original
python sdn_orchestrator.py

# Ou diretamente pelo pacote
python -m orchestrator.main
```

---

## Arquivos de Referência

| Arquivo | Descrição |
|---|---|
| `backup/sdn_orchestrator.py` | Versão monolítica original (v14.0) — não modificada |
| `sdn_orchestrator.py` | Entry point atual (delega para `orchestrator/main.py`) |
| `requirements.txt` | Dependências pip do projeto |
| `orchestrator/` | Pacote refatorado (Clean Architecture) |
