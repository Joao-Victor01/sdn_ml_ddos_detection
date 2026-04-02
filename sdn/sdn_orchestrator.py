"""
SDN Orchestrator v14.0 — Entry Point
======================================
Este arquivo é o ponto de entrada mantido para compatibilidade com o
comando original `python sdn_orchestrator.py`.

O código foi refatorado para o pacote `orchestrator/` seguindo
Clean Architecture e princípios SOLID:

  orchestrator/
  ├── config.py              — Configurações e constantes
  ├── domain/
  │   ├── state.py           — Entidade NetworkState + singleton + CYCLE_COUNT
  │   └── models.py          — Modelos Pydantic da API (SwitchRequest, IPBlockRequest)
  ├── infrastructure/
  │   ├── flow_specs.py      — Construtores de strings de flow OVS (funções puras)
  │   ├── docker_adapter.py  — Descoberta e mapeamento de containers Docker
  │   └── ovs_adapter.py     — Instalação/remoção de flows via docker exec ovs-ofctl
  ├── application/
  │   ├── topology.py        — [1/6] Topologia de switches e flows base
  │   ├── hosts.py           — [2/6] Descoberta de hosts e ARP probing
  │   ├── traffic.py         — [3/6] Monitoramento de tráfego
  │   └── routing.py         — [4-6/6] Rotas IPv4, ARP spanning-tree e reroute
  ├── presentation/
  │   └── api.py             — API REST (FastAPI): /manage/switch, /manage/ip, etc.
  └── main.py                — Loop de controle e função main()

Para iniciar: python sdn_orchestrator.py  (sem alteração no comando)
"""

from orchestrator.main import main

if __name__ == "__main__":
    main()
