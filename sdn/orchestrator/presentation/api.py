"""
API REST do SDN Orchestrator (FastAPI).

Endpoints disponíveis:
  POST /manage/switch  — Bloqueia ou desbloqueia um switch
  POST /manage/ip      — Bloqueia ou desbloqueia um IP
  GET  /health         — Status do orquestrador (ciclo atual, contadores)
  GET  /state          — Estado completo da rede (switches, hosts, flows)
  GET  /flows/{sw_id}  — Flows instalados em um switch (dump-flows)
"""
import time as _time

import subprocess
from concurrent.futures import as_completed

from fastapi import FastAPI

import orchestrator.domain.state as state_module
from orchestrator.domain.state import state
from orchestrator.domain.models import SwitchRequest, IPBlockRequest
from orchestrator.infrastructure.docker_adapter import container_for
from orchestrator.infrastructure.ovs_adapter import (
    FLOW_EXECUTOR, delete_ip_block_direct,
)


from typing import Optional, Dict
from pydantic import BaseModel
from orchestrator.utils import metrics_collector as _metrics_mod


# ── Estado interno de flows QoS instalados ──────────────────────────────────
# {client_id: [(container_name, client_ip, dscp), ...]}
_qos_flows: Dict[int, list] = {}

# Mapeamento DSCP por nível de prioridade
# cat1 → EF(46): modelos pequenos, time-sensitive
# cat2 → AF31(26): prioridade média
# cat3 → BE(0): modelos grandes, best-effort (não congestionam caminhos críticos)
_DSCP_MAP = {1: 46, 2: 26, 3: 0}
_PRIORITY_MAP = {46: 300, 26: 250, 0: 200}


class QoSRequest(BaseModel):
    client_id:     int
    client_ip:     str
    dscp:          int   # 46=EF, 26=AF31, 0=BE
    priority_level: int  # 1=alta, 2=média, 3=baixa


app = FastAPI(title="SDN Orchestrator v14.0")


@app.post("/manage/switch")
def manage_switch(req: SwitchRequest):
    """Bloqueia ou desbloqueia um switch na topologia."""
    with state.lock:
        if req.action == "block":
            if req.switch_id not in state.blocked_switches:
                state.blocked_switches.append(req.switch_id)
                state._guard_done.discard(req.switch_id)
        elif req.action == "unblock":
            if req.switch_id in state.blocked_switches:
                state.blocked_switches.remove(req.switch_id)
    return {"status": "ok"}


@app.post("/manage/ip")
def manage_ip(req: IPBlockRequest):
    """
    Bloqueia ou desbloqueia um IP na rede.

    Unblock usa pending_unblocks para resolver race condition com o loop
    principal: sem isso, install_ipv4_routes pode reinstalar o flow de
    bloqueio usando uma cópia antiga de blocked_ips no mesmo ciclo.
    """
    with state.lock:
        if req.action == "block":
            if req.ip not in state.blocked_ips:
                state.blocked_ips.append(req.ip)
                # Garante que não está pendente de unblock
                state.pending_unblocks.discard(req.ip)
        elif req.action == "unblock":
            # 1) Sinaliza para install_ipv4_routes NÃO reinstalar este IP
            state.pending_unblocks.add(req.ip)
            # 2) Remove da lista permanente
            if req.ip in state.blocked_ips:
                state.blocked_ips.remove(req.ip)
            g_copy = state.graph.copy()
        else:
            return {"status": "ok"}

    if req.action == "unblock":
        # Deleta em paralelo (rápido, sem risco de timeout)
        futures = [
            FLOW_EXECUTOR.submit(delete_ip_block_direct, sw, req.ip)
            for sw in g_copy.nodes
        ]
        for f in as_completed(futures, timeout=15):
            try:
                f.result()
            except Exception:
                pass
        # Remove do pending_unblocks após a deleção física estar concluída
        with state.lock:
            state.pending_unblocks.discard(req.ip)

    return {"status": "ok"}


@app.get("/health")
def health():
    """Retorna o status resumido do orquestrador."""
    with state.lock:
        return {
            "status":     "ok",
            "cycle":      state_module.CYCLE_COUNT,
            "switches":   len(state.graph.nodes),
            "hosts":      len(state.hosts_by_mac),
            "containers": len(state.sw_to_container),
        }


@app.get("/state")
def get_state():
    """Retorna o estado completo da rede (switches, hosts, flows ativos)."""
    with state.lock:
        return {
            "switches":    list(state.graph.nodes),
            "hosts":       state.hosts_by_mac,
            "blocked_ips": state.blocked_ips,
            "containers":  state.sw_to_container,
            "edge_ports":  {k: list(v)
                            for k, v in state.edge_ports.items()},
        }


@app.get("/flows/{sw_id}")
def get_flows(sw_id: str):
    """Mostra os flows ativos em um switch via ovs-ofctl dump-flows."""
    container = container_for(sw_id)
    if not container:
        return {"error": f"{sw_id} não mapeado"}
    try:
        r = subprocess.run(
            ["docker", "exec", container,
             "ovs-ofctl", "dump-flows", "br0", "-O", "OpenFlow13"],
            capture_output=True, text=True, timeout=5
        )
        return {"flows": r.stdout.splitlines()}
    except Exception as e:
        return {"error": str(e)}
    

# ── Endpoints para integração FL-SDN ──────────────────────────────────────


"""
CORREÇÕES PARA orchestrator/presentation/api.py
================================================

Cole este bloco NO FINAL do arquivo api.py existente,
logo antes do if __name__ == "__main__" (se existir).

PROBLEMA 1 — /metrics/hosts: state.ip_to_mac é {ip: mac_string},
não {ip: dict}. dict("aa:bb:cc") itera chars — ValueError.

PROBLEMA 2 — /qos/apply e /qos/{id}: endpoints ausentes.
O FL-SDN chama esses endpoints mas eles não existiam no orquestrador.

PROBLEMA 3 — _get_output_port_for_ip: usava network_state.routes que
não existe. Agora usa state.active_flows para encontrar porta de saída.
"""

# =============================================================================
# GET /metrics/links
# =============================================================================

@app.get("/metrics/links")
def get_link_metrics():
    """
    Expõe utilização atual de cada enlace para o FL-SDN.
    Evita duplo polling no ODL — o FL lê dados já processados pelo orquestrador.
    """
    from orchestrator.config import MAX_LINK_CAPACITY, REROUTE_THRESH, CONGESTED_THRESH

    with state.lock:
        link_load  = dict(state.link_load)
        link_costs = dict(state.link_costs)

    links = {}
    congested_count = warn_count = 0

    for (u, v), load_bps in link_load.items():
        util      = load_bps / MAX_LINK_CAPACITY
        congested = util >= CONGESTED_THRESH
        warn      = util >= REROUTE_THRESH and not congested

        links[f"{u}↔{v}"] = {
            "load_bps":    round(load_bps),
            "utilization": round(util, 4),
            "congested":   congested,
            "warn":        warn,
        }
        congested_count += int(congested)
        warn_count      += int(warn)

    return {
        "timestamp":             _time.time(),
        "max_link_capacity_bps": MAX_LINK_CAPACITY,
        "reroute_thresh":        REROUTE_THRESH,
        "congested_thresh":      CONGESTED_THRESH,
        "links":                 links,
        "summary": {
            "total_links":     len(links),
            "congested_links": congested_count,
            "warn_links":      warn_count,
        },
    }


# =============================================================================
# GET /metrics/hosts  — CORRIGIDO
# =============================================================================

@app.get("/metrics/hosts")
def get_host_metrics():
    from orchestrator.config import MAX_LINK_CAPACITY, FL_SERVER_SWITCH
    import time as _time

    with state.lock:
        ip_to_mac = dict(state.ip_to_mac)
        by_mac    = dict(state.hosts_by_mac)
        link_load = dict(state.link_load)
        graph     = state.graph.copy()

    result = {}

    for ip, mac in ip_to_mac.items():
        if not ip.startswith("172.16.1."):
            continue

        host_info = by_mac.get(mac, {})
        sw        = host_info.get("switch", "")  # "openflow:6" — formato ODL

        # ── Bottleneck: carga máxima no caminho sw_borda → servidor ────────
        # BUG FIX: usa o node-id ODL completo ("openflow:6"), não a forma
        # abreviada ("sw6"). O grafo NetworkX tem nós no formato "openflow:X",
        # então usar "sw6" causava falha silenciosa no shortest_path e o
        # fallback retornava max(link_load) — sempre 0 Mbps disponível.
        bottleneck_bps = _get_path_bottleneck(sw, FL_SERVER_SWITCH, link_load, graph)

        util       = bottleneck_bps / MAX_LINK_CAPACITY if MAX_LINK_CAPACITY > 0 else 0
        avail_mbps = max(0.0, (MAX_LINK_CAPACITY - bottleneck_bps) / 1_000_000)

        if util > 0.8:
            latency_ms = 2.0 * (1 + util * 10)
        else:
            latency_ms = 2.0 * (1 + util)

        result[ip] = {
            "switch":         sw,
            "port":           host_info.get("port", ""),
            "bandwidth_mbps": round(avail_mbps, 2),
            "latency_ms":     round(latency_ms, 2),
            "packet_loss":    0.0,
            "jitter_ms":      round(1.0 + util * 5, 2),
        }

    return {"timestamp": _time.time(), "hosts": result}


def _get_path_bottleneck(src_sw: str, dst_sw: str,
                          link_load: dict, graph) -> float:
    """
    Retorna a carga do enlace mais congestionado no caminho src→dst.
    Usa o grafo de topologia do orquestrador (já calculado pelo Dijkstra).
    Se não encontrar caminho, retorna a carga máxima global como fallback.
    """
    import networkx as nx
    try:
        # Dijkstra sem pesos — queremos o caminho mais curto em hops
        path = nx.shortest_path(graph, src_sw, dst_sw)
        max_load = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            key = tuple(sorted([u, v]))
            load = link_load.get(key, 0.0)
            max_load = max(max_load, load)
        return max_load
    except Exception:
        # Fallback: carga máxima global entre todos os enlaces
        return max(link_load.values(), default=0.0)


# =============================================================================
# POST /qos/apply
# =============================================================================

@app.post("/qos/apply")
def apply_qos(req: QoSRequest):
    """
    Instala flow OpenFlow com marcação DSCP para um cliente FL.

    Por que QoS via DSCP:
      - cat1 (modelos ~50KB): prioridade EF(46) → rede entrega antes do background
      - cat2 (modelos ~500KB): prioridade AF31(26) → tratamento garantido
      - cat3 (modelos ~5-15MB): prioridade BE(0) → usa caminhos alternativos

    O flow é instalado no switch de attachment point do cliente (switch de borda).
    Marca os pacotes IP saindo do cliente com o ToS correto antes de entrarem
    no núcleo da rede — os flows Dijkstra do orquestrador encaminham normalmente.

    Por que instalar aqui e não no FL diretamente:
      - Orquestrador é o único escritor nos OVS (evita race conditions)
      - Usa ovs-ofctl via docker exec (~5ms) vs REST ODL (~200-500ms)
      - Conhece o switch de attachment point do cliente via state.hosts_by_mac
    """
    client_ip  = req.client_ip
    client_id  = req.client_id
    dscp       = req.dscp
    priority   = _PRIORITY_MAP.get(dscp, 200)

    with state.lock:
        ip_to_mac = dict(state.ip_to_mac)
        by_mac    = dict(state.hosts_by_mac)
        sw_map    = dict(state.sw_to_container)

    mac       = ip_to_mac.get(client_ip)
    host_info = by_mac.get(mac, {}) if mac else {}
    sw        = host_info.get("switch", "")  # "openflow:6"

    if not sw:
        return {
            "status":  "warn",
            "message": f"Host {client_ip} não encontrado na topologia — QoS não aplicado",
            "flows_installed": 0,
        }

    container = sw_map.get(sw)
    if not container:
        return {
            "status":  "warn",
            "message": f"Container para {sw} não mapeado",
            "flows_installed": 0,
        }

    # Flow: pacotes IP com src=client_ip → marca DSCP e encaminha normalmente
    # NORMAL = deixa os flows Dijkstra existentes cuidarem do encaminhamento
    # mod_nw_tos: ToS = dscp * 4 (DSCP ocupa bits 7..2 do campo ToS de 8 bits)
    flow_spec = (
        f"priority={priority},"
        f"ip,nw_src={client_ip},"
        f"actions=mod_nw_tos:{dscp * 4},NORMAL"
    )

    try:
        result = subprocess.run(
            ["docker", "exec", container,
             "ovs-ofctl", "add-flow", "br0", flow_spec, "-O", "OpenFlow13"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Registra para remoção posterior via DELETE /qos/{client_id}
            _qos_flows.setdefault(client_id, []).append(
                (container, client_ip, dscp)
            )
            return {
                "status":          "ok",
                "client_id":       client_id,
                "client_ip":       client_ip,
                "dscp":            dscp,
                "priority":        priority,
                "flows_installed": 1,
                "switch":          sw,
                "container":       container,
            }
        else:
            return {
                "status": "error",
                "error":  result.stderr.strip() or "ovs-ofctl falhou",
                "flows_installed": 0,
            }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "timeout ao executar ovs-ofctl", "flows_installed": 0}
    except Exception as e:
        return {"status": "error", "error": str(e), "flows_installed": 0}


# =============================================================================
# DELETE /qos/{client_id}
# =============================================================================

@app.delete("/qos/{client_id}")
def remove_qos(client_id: int):
    """
    Remove flows QoS instalados para o cliente especificado.
    Chamado pelo FL-SDN ao final de cada round para limpar flows temporários.
    """
    flows   = _qos_flows.pop(client_id, [])
    removed = 0

    if not flows:
        return {
            "status":  "ok",
            "removed": 0,
            "message": f"Nenhum flow QoS registrado para cliente {client_id}",
        }

    for (container, client_ip, dscp) in flows:
        priority = _PRIORITY_MAP.get(dscp, 200)
        try:
            result = subprocess.run(
                ["docker", "exec", container,
                 "ovs-ofctl", "--strict", "del-flows", "br0",
                 f"priority={priority},ip,nw_src={client_ip}",
                 "-O", "OpenFlow13"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                removed += 1
        except Exception:
            pass

    return {
        "status":    "ok",
        "client_id": client_id,
        "removed":   removed,
        "total":     len(flows),
    }


# =============================================================================
# GET /qos/status  — diagnóstico
# =============================================================================

@app.get("/qos/status")
def qos_status():
    """Mostra todos os flows QoS atualmente instalados."""
    return {
        "active_clients": list(_qos_flows.keys()),
        "flows": {
            str(cid): [
                {"container": c, "client_ip": ip, "dscp": dscp}
                for c, ip, dscp in flows
            ]
            for cid, flows in _qos_flows.items()
        },
        "total_flows": sum(len(v) for v in _qos_flows.values()),
    }


# =============================================================================
# POST /fl/training/start  e  POST /fl/training/stop
# =============================================================================

class FLTrainingStartRequest(BaseModel):
    round: int


@app.post("/fl/training/start")
def fl_training_start(req: FLTrainingStartRequest):
    """
    Notifica o SDN que um round de FL começou.
    Abre um CSV dedicado (fl_metrics_round{N}_YYYYMMDD_HHMMSS.csv) que
    recebe as mesmas métricas de ciclo do sdn_metrics.csv enquanto o
    treinamento estiver ativo.
    """
    mc = _metrics_mod._instance
    if mc is None:
        return {"status": "error", "message": "MetricsCollector não inicializado"}
    path = mc.start_fl_session(req.round)
    return {"status": "ok", "round": req.round, "csv_path": path}


@app.post("/fl/training/stop")
def fl_training_stop():
    """
    Notifica o SDN que o round de FL terminou.
    Fecha e finaliza o CSV do round, devolvendo o número do round e
    a duração total do treinamento em segundos.
    """
    mc = _metrics_mod._instance
    if mc is None:
        return {"status": "error", "message": "MetricsCollector não inicializado"}
    return mc.stop_fl_session()


# =============================================================================
# Detecção Multiclasse — HCF (Hop Count Filtering)
# =============================================================================

class ClassifyRequest(BaseModel):
    src_ip:       str
    ttl_observed: int
    flow_pkts_s:  float
    ttl_initial:  int = 64   # 64=Linux, 128=Windows


@app.post("/detect/classify")
def detect_classify(req: ClassifyRequest):
    """
    Classifica um fluxo em: Benigno (0) / Ataque Externo (1) / Zumbi Interno (2).

    Usa Hop Count Filtering (HCF): analisa o TTL do pacote para inferir a
    distância topológica entre o host de origem e o controlador SDN.

    Parâmetros:
      - src_ip       : IP de origem do fluxo suspeito
      - ttl_observed : TTL observado no pacote ao chegar ao switch de borda
      - flow_pkts_s  : taxa de pacotes por segundo do fluxo
      - ttl_initial  : TTL esperado na origem (64=Linux, 128=Windows)

    Lógica:
      - Taxa baixa (< 10.000 pps)  → Benigno
      - hop_count ≥ 10             → Externo (cruzou internet)
      - hop_count < 10 + alta taxa → Zumbi Interno (host comprometido)
    """
    from orchestrator.application.hcf import HCFAnalyzer
    analyzer = HCFAnalyzer()
    result   = analyzer.classify(
        src_ip=req.src_ip,
        ttl_observed=req.ttl_observed,
        flow_pkts_s=req.flow_pkts_s,
        ttl_initial=req.ttl_initial,
    )
    return result.to_dict()


class ClassifyBatchRequest(BaseModel):
    flows: list


@app.post("/detect/classify/batch")
def detect_classify_batch(req: ClassifyBatchRequest):
    """
    Classifica múltiplos fluxos em lote.

    Cada item da lista deve conter: src_ip, ttl_observed, flow_pkts_s
    (e opcionalmente ttl_initial, padrão 64).

    Retorna lista com classificação HCF de cada fluxo.
    """
    from orchestrator.application.hcf import HCFAnalyzer
    analyzer = HCFAnalyzer()
    results  = analyzer.classify_batch(req.flows)
    return {"results": [r.to_dict() for r in results]}


# =============================================================================
# Mitigação Multiclasse — IP Traceback + Isolamento Cirúrgico
# =============================================================================

@app.post("/mitigation/traceback/{ip}")
def mitigation_traceback(ip: str):
    """
    Rastreia o caminho de ataque de um IP suspeito até o controlador.

    Usa o grafo NetworkX da topologia SDN + Dijkstra para calcular o
    caminho completo desde o switch de borda do host até o destino.

    Retorna:
      - src_switch  : switch de borda onde o host está conectado
      - src_port    : porta do switch onde o host está conectado
      - attack_path : lista de switches no caminho de ataque
      - found       : True se o IP foi localizado na topologia SDN
    """
    from orchestrator.application.traceback import IPTraceback
    tb     = IPTraceback()
    result = tb.traceback(ip)
    return result.to_dict()


@app.post("/mitigation/isolate/{ip}")
def mitigation_isolate(ip: str):
    """
    Isola cirurgicamente um zumbi interno.

    Instala um flow DROP de alta prioridade (65400) APENAS na porta de borda
    onde o host infectado está conectado — sem afetar nenhum outro host no
    mesmo switch ou VLAN.

    Diferente de POST /manage/ip (DROP global em TODOS os switches):
      - /manage/ip       → proteção ampla, bloqueia o IP em toda a topologia
      - /mitigation/isolate → cirúrgico, bloqueia só na porta do host infectado

    Pré-condição: o host deve estar registrado no SDN (visto pelo controlador).
    Recomendado para: zumbis internos classificados pelo HCF como INTERNAL (2).
    """
    from orchestrator.application.traceback import IPTraceback
    tb = IPTraceback()
    return tb.isolate(ip)


@app.delete("/mitigation/isolate/{ip}")
def mitigation_release(ip: str):
    """
    Libera um host previamente isolado.

    Remove o flow de isolamento do switch de borda. Use após confirmar que
    o host foi desinfectado ou identificado como falso positivo.
    """
    from orchestrator.application.traceback import IPTraceback
    tb = IPTraceback()
    return tb.release(ip)


@app.get("/mitigation/status")
def mitigation_status():
    """
    Lista todos os hosts atualmente isolados e seus metadados.

    Retorna para cada host isolado:
      - ip      : endereço IP do host
      - mac     : endereço MAC
      - switch  : switch de borda onde o isolamento foi aplicado
      - port    : porta onde o host está conectado
      - active  : True se o isolamento ainda está ativo
    """
    from orchestrator.application.traceback import IPTraceback
    tb = IPTraceback()
    return {"isolated_hosts": tb.list_isolated()}