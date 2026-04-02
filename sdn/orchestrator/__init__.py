"""
SDN Orchestrator v14.0 — Pacote principal.

Arquitetura em camadas (Clean Architecture + SOLID):
  config.py            — Configurações e constantes globais
  domain/              — Entidades e modelos do domínio (sem dependências externas)
  infrastructure/      — Adaptadores de I/O: OVS, Docker, especificações de flow
  application/         — Casos de uso: topologia, hosts, tráfego, roteamento
  presentation/        — API REST (FastAPI)
  main.py              — Loop de controle e ponto de entrada
"""
