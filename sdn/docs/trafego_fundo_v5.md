# Tráfego de Fundo --- Análise V5

## Como funcionou o tráfego de fundo no V5

### Quais nós geraram tráfego

Dois containers BG-Node enviavam iperf3 para o host (172.16.1.1),
passando pelo OVS-1:

  --------------------------------------------------------------------------
  Container        Caminho                  Evidência no log
  ---------------- ------------------------ --------------------------------
  BG-Node-1-cat1   sw8 → OVS-3 (sw3) →      sw8↔sw3 consistentemente 13--17
                   OVS-1 (sw1)              Mbps

  BG-Node-2-cat1   sw14 → OVS-5 (sw5) →     sw14↔sw5 consistentemente 12--17
                   OVS-1 (sw1)              Mbps
  --------------------------------------------------------------------------

O BG-Node-3-cat2 estava rodando mas não gerou tráfego relevante (cat2
está em outra área da topologia).

------------------------------------------------------------------------

### Quanto tráfego foi gerado

O comando usado:

    -b 30M -P 3

Isso significa:

-   3 streams paralelos
-   30 Mbps cada

Total teórico por BG-Node:

    90 Mbps

Na prática:

-   Links virtuais OVS limitaram para \~13--17 Mbps por BG-Node
-   Total agregado chegando ao OVS-1: \~25--34 Mbps

------------------------------------------------------------------------

### Impacto na rede

Esse volume de tráfego causou:

-   Saturação dos links:
    -   sw3 ↔ sw1
    -   sw5 ↔ sw1

Esses links ultrapassaram o threshold de rerouting:

    REROUTE_THRESH = 65% × 20 Mbps ≈ 13 Mbps

Consequência:

-   Ativação de rerouting
-   Tráfego redirecionado para:

```{=html}
<!-- -->
```
    sw4 ↔ sw1

Isso explica os picos observados:

-   90% até 128% de utilização no link sw4

------------------------------------------------------------------------

### Problema com o comando original

O comando foi executado sem a flag `-d`:

    docker exec ...

Consequência:

-   Rodou em foreground
-   Dependia da sessão do terminal

Funcionou porque a sessão permaneceu aberta, mas:

⚠️ Não é robusto

------------------------------------------------------------------------

### Correção para versões futuras (V4+)

Usar execução em background:

    docker exec -d ...

Benefícios:

-   Processo continua rodando mesmo após fechar o terminal
-   Maior confiabilidade para experimentos longos
