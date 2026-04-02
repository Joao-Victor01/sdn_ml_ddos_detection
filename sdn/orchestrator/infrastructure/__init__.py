"""
Camada de Infraestrutura.

Adaptadores de I/O que isolam o sistema de suas dependências externas:
  flow_specs.py     — Construtores de strings de flow OVS (funções puras)
  docker_adapter.py — Descoberta e mapeamento de containers Docker
  ovs_adapter.py    — Instalação e remoção de flows via docker exec ovs-ofctl
"""
