"""
Adaptador OVS — instalação e remoção de flows via docker exec ovs-ofctl.

Responsabilidade única (SRP): toda a comunicação com o Open vSwitch
passa por este módulo. As funções de instalação paralela usam um
ThreadPoolExecutor compartilhado (FLOW_EXECUTOR) para reduzir o tempo
de ciclo de O(N×switches) para O(1×exec_mais_lento).

FIX v14 — Instalação paralela:
  Com 10 switches × ~15 flows cada, as ~150 chamadas sequenciais
  levavam 2-3 minutos por ciclo (cada exec ≈ 0.5-1s).
  Com max_workers=20, o tempo cai para ~3s independente do volume.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from orchestrator.domain.state import state
from orchestrator.infrastructure.docker_adapter import container_for

# Pool de threads para instalação paralela de flows.
# max_workers=20 permite instalar flows em até 20 switches simultaneamente.
FLOW_EXECUTOR = ThreadPoolExecutor(max_workers=20, thread_name_prefix="flow-install")


def port_to_iface(port: str) -> str:
    """Converte número de porta OpenFlow para nome de interface OVS."""
    return f"eth{port}"


def install_flow(sw_id: str, flow_id: str, ovs_flow_str: str,
                 silent: bool = False) -> bool:
    """
    Instala um flow via docker exec ovs-ofctl add-flow.

    Thread-safe: cada chamada usa apenas variáveis locais e um lock pontual
    para atualizar state.active_flows ao final.
    """
    container = container_for(sw_id)
    if not container:
        if not silent:
            print(f"  ⚠️  {sw_id}: container não mapeado")
        return False

    cmd = ["docker", "exec", container,
           "ovs-ofctl", "add-flow", "br0", ovs_flow_str, "-O", "OpenFlow13"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if r.returncode != 0:
            if not silent:
                print(f"  ⚠️  {flow_id}@{sw_id}: {r.stderr.strip()[:120]}")
            return False
        with state.lock:
            state.active_flows[(sw_id, flow_id)] = ovs_flow_str
        return True
    except Exception as e:
        if not silent:
            print(f"  ❌ {flow_id}@{sw_id}: {e}")
        return False


def install_flows_parallel(tasks: list[tuple]) -> tuple[int, int]:
    """
    Instala múltiplos flows em paralelo via ThreadPoolExecutor.

    tasks: lista de (sw_id, flow_id, ovs_flow_str, silent)
    Retorna: (ok_count, fail_count)

    Todos os flows de todos os switches são submetidos de uma vez ao pool.
    Com max_workers=20, até 20 docker exec correm simultaneamente.
    Tempo típico: max(tempo_de_um_exec) ≈ 0.5-1s independente do volume.
    """
    if not tasks:
        return 0, 0

    futures = {
        FLOW_EXECUTOR.submit(install_flow, sw_id, fid, fstr, silent): (sw_id, fid)
        for sw_id, fid, fstr, silent in tasks
    }
    ok = fail = 0
    for future in as_completed(futures, timeout=15):
        try:
            if future.result():
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    return ok, fail


def delete_flow(sw_id: str, flow_id: str) -> None:
    """Remove um flow via docker exec ovs-ofctl del-flows."""
    container = container_for(sw_id)
    if not container:
        return
    with state.lock:
        flow_str = state.active_flows.pop((sw_id, flow_id), None)
    if not flow_str:
        return
    match_part = flow_str.split(",actions=")[0]
    cmd = ["docker", "exec", container,
           "ovs-ofctl", "del-flows", "br0", match_part, "-O", "OpenFlow13"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except Exception:
        pass


def delete_flows_parallel(tasks: list[tuple]) -> None:
    """Remove múltiplos flows em paralelo. tasks: lista de (sw_id, flow_id)."""
    futures = [FLOW_EXECUTOR.submit(delete_flow, sw_id, fid)
               for sw_id, fid in tasks]
    for f in as_completed(futures, timeout=15):
        try:
            f.result()
        except Exception:
            pass


def verify_table_miss(sw_id: str) -> bool:
    """Verifica se o flow TABLE-MISS existe de verdade no switch via dump-flows."""
    container = container_for(sw_id)
    if not container:
        return False
    try:
        r = subprocess.run(
            ["docker", "exec", container,
             "ovs-ofctl", "dump-flows", "br0", "-O", "OpenFlow13"],
            capture_output=True, text=True, timeout=8
        )
        return "priority=0" in r.stdout
    except Exception:
        return False


def delete_ip_block_direct(sw_id: str, ip: str) -> None:
    """
    Remove flow de bloqueio de IP diretamente no switch, SEM depender de
    state.active_flows. Usa --strict para garantir que a prioridade seja
    respeitada na busca do flow. Idempotente: não falha se o flow não existir.
    """
    container = container_for(sw_id)
    if not container:
        return
    # Limpa cache independente de sucesso físico
    with state.lock:
        state.active_flows.pop((sw_id, f"Block_{ip}"), None)
    # --strict: exige correspondência exata de prioridade + match
    cmd = ["docker", "exec", container,
           "ovs-ofctl", "--strict", "del-flows", "br0",
           f"priority=65500,ip,nw_src={ip}", "-O", "OpenFlow13"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if r.returncode != 0 and r.stderr.strip():
            # Fallback sem prioridade (cobre edge cases de versão de OVS)
            subprocess.run(
                ["docker", "exec", container, "ovs-ofctl", "del-flows", "br0",
                 f"ip,nw_src={ip}", "-O", "OpenFlow13"],
                capture_output=True, text=True, timeout=8
            )
    except Exception:
        pass
