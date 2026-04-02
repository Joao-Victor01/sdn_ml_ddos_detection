"""
Camada de Aplicação — casos de uso do orquestrador SDN.

Cada módulo representa um caso de uso coeso do ciclo de controle:
  topology.py — [1/6] Descoberta de switches e instalação de flows base
  hosts.py    — [2/6] Descoberta de hosts e ARP probing
  traffic.py  — [3/6] Monitoramento de tráfego e cálculo de custos
  routing.py  — [4-6/6] Rotas IPv4, ARP Spanning-Tree e reroute
"""
