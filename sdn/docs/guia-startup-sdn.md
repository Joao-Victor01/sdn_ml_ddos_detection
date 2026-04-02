# Guia de Startup — Lab SDN (ODL Calcium + OVS + GNS3)
> Execute na ordem exata. Cada etapa depende da anterior.

---

## ETAPA 1 — Iniciar o Karaf (ODL)

```bash
cd ~/karaf-0.23.0/bin
./karaf
```

**Aguarde ~60 segundos** até o log parar de mostrar mensagens de bundle.  
Você saberá que está pronto quando o prompt `opendaylight-user@root>` ficar estável.

### 1.1 — Parar os bundles problemáticos (FRM + reconciliation)

```
bundle:stop 300
bundle:stop 297
bundle:stop 298
bundle:stop 303
bundle:stop 411
bundle:stop 412
```

Confirme que todos estão `Resolved` (não `Active`):

```
bundle:list | grep -i "forwardingrules\|reconcil\|arbitrator\|loopremover"
```

### 1.2 — Confirmar bundles LLDP ativos

```
bundle:list | grep -i lldp
```

Esperado: bundles **301**, **304** e **311** com status `Active`.  
Se algum estiver `Resolved`, inicie na ordem:

```
bundle:start 311
bundle:start 304
bundle:start 301
```

---

## ETAPA 2 — Iniciar os switches OVS no GNS3

Ligue o projeto no GNS3 e execute o script de configuração em **cada um dos 4 containers OVS**.  
Substitua `SW_NUM` pelo número do switch: `2`, `3`, `4` e `11`.

```bash
SW_NUM="4"   # ← mude para cada switch

ifconfig eth0 172.16.1.$((100 + SW_NUM)) netmask 255.255.255.0 up

ovs-vsctl del-br br0 2>/dev/null
ovs-vsctl add-br br0

DPID=$(printf "%016d" $SW_NUM)
ovs-vsctl set bridge br0 other-config:datapath-id=$DPID

for i in {1..10}; do
    ip link set eth$i up
    ovs-vsctl add-port br0 eth$i 2>/dev/null
done

ovs-vsctl set-controller br0 tcp:172.16.1.1:6653
ovs-vsctl set bridge br0 protocols=OpenFlow13
ovs-vsctl set bridge br0 fail-mode=secure
ovs-vsctl set bridge br0 other-config:disable-in-band=true

echo "✅ OVS $SW_NUM configurado"
```

### 2.1 — Confirmar conexão em cada switch

```bash
ovs-vsctl show | grep is_connected
# Esperado: is_connected: true
```

---

## ETAPA 3 — Instalar flows LLDP manualmente nos switches

> **Por quê?** O FRM está parado, então o ODL não empurra flows aos switches automaticamente.  
> Estes 3 flows mínimos permitem que o LLDP e o ARP funcionem enquanto o orquestrador não está rodando.

Execute em **cada um dos 4 OVS**:

```bash
ovs-ofctl add-flow br0 "priority=1000,dl_type=0x88cc,actions=controller:65535" -O OpenFlow13
ovs-ofctl add-flow br0 "priority=1000,dl_type=0x8942,actions=controller:65535" -O OpenFlow13
ovs-ofctl add-flow br0 "priority=0,actions=controller:65535" -O OpenFlow13
```

Confirme:

```bash
ovs-ofctl dump-flows br0 -O OpenFlow13 | grep priority=1000
# Esperado: 2 flows com dl_type=0x88cc e dl_type=0x8942
```

---

## ETAPA 4 — Reiniciar os bundles LLDP no Karaf

De volta ao console do Karaf, force um ciclo limpo nos bundles de topologia:

```
bundle:stop 301
bundle:stop 304
bundle:start 304
bundle:start 301
```

Aguarde 15 segundos e confirme no log que o LLDP está enviando:

```
log:tail
```

Você deve ver: `Sending LLDP frames to total 40 ports`  
Pressione **Ctrl+C** para sair do log.

---

## ETAPA 5 — Confirmar topologia no ODL (terminal Linux)

```bash
# Verificar switches conectados
curl -s -u admin:admin \
  "http://172.16.1.1:8181/rests/data/network-topology:network-topology/topology=flow:1?content=nonconfig" \
  -H "Accept: application/json" | python3 -m json.tool | grep '"node-id"'

# Verificar enlaces inter-switch (deve retornar ~8-12 link-ids)
curl -s -u admin:admin \
  "http://172.16.1.1:8181/rests/data/network-topology:network-topology/topology=flow:1?content=nonconfig" \
  -H "Accept: application/json" | python3 -m json.tool | grep '"link-id"' | grep -v host
```

Se os links aparecerem, a topologia está pronta. Prossiga para a Etapa 6.

---

## ETAPA 6 — Iniciar o orquestrador Python

```bash
cd ~/sdn-project
source venv/bin/activate
python3 sdn_orchestrator.py
```

**Aguarde o ciclo #1 completo.** O orquestrador vai:
- Instalar flows de infraestrutura em todos os switches
- Descobrir os enlaces LLDP
- Aguardar tráfego ARP das VPCs para descobrir os hosts

### 6.1 — Gerar tráfego ARP nas VPCs

No GNS3, acesse cada VPC e configure o IP:

```
PC1> ip 172.16.1.10
PC2> ip 172.16.1.20
```

Após configurar os IPs, as VPCs enviam ARP gratuitous automaticamente.  
O orquestrador deve reportar `🔍 Novo host` dentro de 1-2 ciclos.

### 6.2 — Testar conectividade

```
PC1> ping 172.16.1.20
```

---

## Referência rápida — Números dos bundles

| Bundle | Nome | Ação no startup |
|--------|------|-----------------|
| 300 | forwardingrules-manager | **PARAR** |
| 297 | arbitratorreconciliation-api | **PARAR** |
| 298 | arbitratorreconciliation-impl | **PARAR** |
| 303 | reconciliation-framework | **PARAR** |
| 411 | loopremover-impl | **PARAR** |
| 412 | loopremover-model | **PARAR** |
| 311 | liblldp | manter Active |
| 304 | topology-lldp-discovery | manter Active |
| 301 | lldp-speaker | manter Active |

---

## Diagnóstico rápido — o que checar se algo der errado

**Switches não aparecem no ODL:**
```bash
# No OVS
ovs-vsctl show | grep is_connected
```

**Links não aparecem (0 enlaces):**
```
# No Karaf
bundle:list | grep -i lldp
# No OVS
ovs-ofctl dump-flows br0 -O OpenFlow13 | grep 88cc
```

**Hosts não são descobertos:**
```bash
# Confirmar que o ARP chega ao ODL
curl -s -u admin:admin \
  "http://172.16.1.1:8181/rests/data/network-topology:network-topology/topology=flow:1?content=nonconfig" \
  -H "Accept: application/json" | python3 -m json.tool | grep '"host:'
```

**Orquestrador mostra links mas ping não funciona:**
```bash
# Ver flows instalados num switch
ovs-ofctl dump-flows br0 -O OpenFlow13 | grep IPv4
```
