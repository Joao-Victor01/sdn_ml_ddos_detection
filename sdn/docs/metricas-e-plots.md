# Métricas e Plots — SDN Orchestrator para TCC

Documentação do sistema de coleta de métricas e geração de gráficos implementado
sobre o SDN Orchestrator, sem alterar seu funcionamento original.

---

## Visão geral

Foram adicionados três arquivos ao projeto:

| Arquivo | Responsabilidade |
|---|---|
| `orchestrator/utils/metrics_collector.py` | Coleta métricas do orquestrador a cada ciclo e persiste em CSV |
| `plot_results.py` | Gera os 5 gráficos do TCC a partir dos CSVs do FL e do SDN |
| `overhead_monitor.sh` | Captura tráfego de controle da interface de rede durante o experimento |

E quatro linhas foram adicionadas em `orchestrator/main.py` — nenhuma lógica existente foi
modificada.

---

## 1. Coleta automática — `sdn_metrics.csv`

O orquestrador grava uma linha no CSV ao final de **cada ciclo de controle** (a cada ~5 s).
O arquivo é criado automaticamente no diretório onde o orquestrador for iniciado.

### Colunas

| Coluna | Unidade | O que representa |
|---|---|---|
| `timestamp` | Unix s | Momento do fim do ciclo |
| `cycle` | # | Número sequencial do ciclo |
| `elapsed_sec` | s | Segundos desde o início do orquestrador |
| `cycle_duration_sec` | s | Tempo gasto nas 6 etapas (sem contar o sleep) |
| `n_switches` | # | Switches ativos na topologia ODL |
| `n_hosts` | # | Hosts conhecidos pelo orquestrador |
| `n_flows` | # | Total de flows instalados em `active_flows` |
| `n_reroute_flows` | # | Flows de desvio anti-congestionamento ativos (prioridade 62000, prefixo `LB_`) |
| `max_link_load_bps` | bps | Maior carga de enlace observada no ciclo |
| `avg_link_load_bps` | bps | Carga média entre todos os enlaces |
| `congested_links` | # | Número de enlaces acima de 75% da capacidade (REROUTE_THRESH) |
| `warn_links` | # | Número de enlaces acima de 50% da capacidade (WARN_THRESH) |

### Como identificar que o SDN atuou

Qualquer linha com `n_reroute_flows > 0` indica que o orquestrador detectou congestionamento
e instalou flows de desvio temporário (idle_timeout=15 s). Esses ciclos aparecerão como
linhas verticais verdes nos gráficos.

### Exemplo de saída

```
timestamp,cycle,elapsed_sec,cycle_duration_sec,n_switches,n_hosts,n_flows,n_reroute_flows,max_link_load_bps,avg_link_load_bps,congested_links,warn_links
1742300100.123,1,5.21,2.14,6,12,144,0,1250000,430000,0,1
1742300110.456,2,10.43,1.98,6,12,144,0,3800000,1100000,0,2
1742300120.789,3,15.67,2.31,6,12,156,2,8200000,3500000,1,3
```

O ciclo 3 acima instalou 2 flows de reroute em resposta a 1 enlace congestionado.

---

## 2. Coleta FL — CSVs do experimento

O `plot_results.py` espera dois CSVs gerados pelo **cliente FL** (um por experimento):

| Coluna | Obrigatória | Descrição |
|---|---|---|
| `round` | sim | Número do round de federação |
| `elapsed_sec` | sim | Segundos desde o início do treinamento FL |
| `accuracy` | sim | Accuracy global do modelo no round |
| `f1` | não | F1-Score global (gera fig2 se presente) |
| `auc` | não | AUC-ROC global (gera fig4 se presente) |

Nomeie os arquivos como `com_sdn_resultados.csv` e `sem_sdn_resultados.csv`
(ou passe os caminhos via argumento — veja seção 3).

---

## 3. Gerando os gráficos — `plot_results.py`

### Pré-requisitos

```bash
pip install matplotlib pandas
```

### Uso básico

```bash
python3 plot_results.py \
  --com com_sdn_resultados.csv \
  --sem sem_sdn_resultados.csv
```

### Com overlay de eventos SDN e figura 5

```bash
python3 plot_results.py \
  --com         com_sdn_resultados.csv \
  --sem         sem_sdn_resultados.csv \
  --sdn-metrics sdn_metrics.csv \
  --output-dir  plots/
```

### Todos os argumentos

| Argumento | Padrão | Descrição |
|---|---|---|
| `--com` | obrigatório | CSV do experimento com SDN |
| `--sem` | obrigatório | CSV do experimento sem SDN |
| `--sdn-metrics` | — | CSV do MetricsCollector (ativa fig5 e overlay) |
| `--output-dir` | `./plots/` | Pasta de saída dos PNGs |

---

## 4. Gráficos gerados

### Fig 1 — `fig1_accuracy_tempo.png`

**Accuracy × Tempo** — duas curvas (com/sem SDN) com linha de limiar em 95% do máximo
e linhas verticais indicando quando cada experimento cruzou esse limiar.
O título exibe automaticamente a **redução percentual de tempo** — o número para o abstract.

> Exemplo: "Redução de tempo com SDN: 47.3%"

Se `--sdn-metrics` for fornecido, linhas verticais verdes indicam os instantes em que o
orquestrador instalou flows de reroute.

---

### Fig 2 — `fig2_f1_tempo.png`

**F1-Score × Tempo** — mesma estrutura da fig1, usando a coluna `f1`.
Gerada apenas se ambos os CSVs contiverem a coluna `f1`.

---

### Fig 3 — `fig3_duracao_round.png`

**Duração por round** — gráfico de barras lado a lado (vermelho = sem SDN, azul = com SDN).
Rounds onde o orquestrador fez reroute recebem anotação `↓SDN` acima da barra.

Evidência direta do impacto ciclo a ciclo: barras azuis mais curtas nos rounds com tráfego
intenso mostram que o SDN reduziu a latência de transferência de gradientes.

---

### Fig 4 — `fig4_auc_round.png`

**AUC-ROC × Round** — curvas por número de round (não por tempo).
Mostra que as duas curvas convergem para o mesmo AUC final, **provando que o SDN
não degrada a qualidade do modelo** — apenas acelera o processo.
Gerada apenas se ambos os CSVs contiverem a coluna `auc`.

---

### Fig 5 — `fig5_overhead_sdn.png`

**Overhead e atuação do SDN Orchestrator** — gerada apenas com `--sdn-metrics`.

Dois subplots:
- **Superior**: carga máxima e média dos enlaces ao longo do tempo, com linhas de limiar
  (laranja = REROUTE_THRESH 75%, amarelo = WARN_THRESH 50%).
- **Inferior**: número de flows de reroute ativos (barras verdes) e duração do ciclo
  de controle (linha cinza tracejada).

Replica o conceito das Figuras 7–8 do artigo original: mostra o trade-off entre
overhead de controle e benefício de desempenho.

---

## 5. Captura de overhead de rede — `overhead_monitor.sh`

Mede o tráfego da interface de controle SDN (geralmente `tap0`) em KB/s usando
`/proc/net/dev`. Não requer `ifstat` nem nenhuma dependência extra.

```bash
# Tornar executável (primeira vez)
chmod +x overhead_monitor.sh

# Monitorar tap0 por 300 s (padrão)
./overhead_monitor.sh

# Monitorar outra interface por 600 s
./overhead_monitor.sh eth0 600
```

Gera `overhead_tap0_<timestamp>.csv` com colunas:

| Coluna | Descrição |
|---|---|
| `elapsed_sec` | Segundo da amostra |
| `rx_kb_s` | Bytes recebidos nesse segundo (KB) |
| `tx_kb_s` | Bytes transmitidos nesse segundo (KB) |
| `total_kb_s` | Soma RX + TX |

Ao final exibe média, máximo e mínimo. O artigo de referência reporta valores entre
**425–916 KB/s** dependendo do intervalo de polling.

---

## 6. Sequência de experimento recomendada

```
# Experimento SEM SDN
1. Iniciar servidor FL
2. Iniciar clientes FL (sem orquestrador rodando)
3. Aguardar conclusão → renomear CSV para sem_sdn_resultados.csv

# Experimento COM SDN
1. Iniciar o orquestrador:
      python3 -m orchestrator
   (o sdn_metrics.csv começa a ser gerado automaticamente)

2. Em paralelo, capturar overhead de controle:
      ./overhead_monitor.sh tap0 600

3. Iniciar servidor FL + clientes FL

4. Ao concluir:
   - renomear CSV do FL para com_sdn_resultados.csv
   - sdn_metrics.csv já está no diretório do orquestrador

# Gerar gráficos
python3 plot_results.py \
  --com         com_sdn_resultados.csv \
  --sem         sem_sdn_resultados.csv \
  --sdn-metrics sdn_metrics.csv \
  --output-dir  plots/
```

---

## 7. Saída textual do `plot_results.py`

Além dos PNGs, o script imprime um resumo no terminal:

```
============================================================
  RESUMO DOS RESULTADOS
============================================================

  Rounds totais:   sem SDN=20  |  com SDN=20
  Tempo total:     sem SDN=412.3s  |  com SDN=238.7s

  Accuracy máx:   sem SDN=0.7124  |  com SDN=0.7118

  Tempo p/ 95% accuracy:
    Sem SDN: 389.1s
    Com SDN: 205.4s
    Redução: 47.2%  ← número para o abstract

  F1 máx:          sem SDN=0.6891  |  com SDN=0.6874
  AUC final:       sem SDN=0.8203  |  com SDN=0.8197

  Ciclos SDN com reroute ativo: 8/84
  Carga máx. de enlace observada: 8.43 Mbps
  Duração média do ciclo SDN: 2.18s
============================================================
```

---

## 8. Arquivos do projeto

```
sdn-project-main/
├── orchestrator/
│   ├── main.py                          ← +4 linhas (import + instância + coleta)
│   └── utils/
│       └── metrics_collector.py         ← novo — coleta SDN por ciclo
├── plot_results.py                      ← novo — gera os 5 gráficos
├── overhead_monitor.sh                  ← novo — captura tráfego da interface
└── docs/
    └── metricas-e-plots.md              ← este arquivo
```
