"""
Adaptador Docker — descoberta e mapeamento de containers OVS.

Responsabilidade única (SRP): descobrir quais containers Docker
correspondem a quais switches OpenFlow e manter esse mapeamento
atualizado no estado global.

O mapeamento é feito executando 'ovs-vsctl get bridge br0 other-config:datapath-id'
em cada container e correlacionando o DPID com os IDs OpenFlow conhecidos pelo ODL.
"""

import subprocess

from orchestrator.config import DOCKER_NAME_PREFIX
from orchestrator.domain.state import state


def discover_containers() -> None:
    """
    Descobre o mapeamento openflow:X → container Docker.

    Executa 'docker ps' para listar containers ativos com o prefixo GNS3,
    depois consulta o DPID de cada um via ovs-vsctl e constrói o mapeamento.
    Atualiza state.sw_to_container ao final.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        containers = [
            c.strip() for c in result.stdout.splitlines()
            if DOCKER_NAME_PREFIX in c
        ]
    except Exception as e:
        print(f"  ❌ docker ps: {e}")
        return

    mapping = {}
    for c in containers:
        try:
            r = subprocess.run(
                ["docker", "exec", c,
                 "ovs-vsctl", "get", "bridge", "br0",
                 "other-config:datapath-id"],
                capture_output=True, text=True, timeout=3
            )
            dpid_raw = r.stdout.strip().strip('"')   # ex: "0000000000000002"
            if not dpid_raw or len(dpid_raw) != 16:
                continue
            dpid_int = int(dpid_raw, 16)              # 2
            sw_id    = f"openflow:{dpid_int}"         # "openflow:2"
            mapping[sw_id] = c
        except Exception:
            continue

    with state.lock:
        state.sw_to_container = mapping

    if mapping:
        print(f"  🐳 Containers mapeados:")
        for sw, c in sorted(mapping.items()):
            print(f"     {sw} → {c.split('.')[1]}")
    else:
        print("  ⚠️  Nenhum container mapeado — verifique DOCKER_NAME_PREFIX")


def container_for(sw_id: str) -> str | None:
    """Retorna o nome do container Docker associado ao switch, ou None."""
    with state.lock:
        return state.sw_to_container.get(sw_id)
