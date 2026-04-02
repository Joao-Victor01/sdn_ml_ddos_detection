#!/usr/bin/env python3
"""
sdn_tools.py — Manutenção e Testes do SDN
==========================================
Comandos:

  python3 sdn_tools.py clean          → [1] Remove flows l2switch dos switches
  python3 sdn_tools.py reroute-test   → [2] Testa desvio dinâmico por congestionamento
  python3 sdn_tools.py block <IP>     → [3] Bloqueia host na rede
  python3 sdn_tools.py unblock <IP>   → [3] Desbloqueia host
  python3 sdn_tools.py metrics        → [3] Métricas em tempo real
  python3 sdn_tools.py status         → Resumo geral
"""

import subprocess, sys, time, re, requests, argparse, threading
from collections import defaultdict
from requests.auth import HTTPBasicAuth

# ── Configuração ──────────────────────────────────────────────────────────
ORCHESTRATOR     = "http://127.0.0.1:8000"
ODL              = "http://172.16.1.1:8181"
ODL_AUTH         = HTTPBasicAuth("admin", "admin")
HEADERS          = {"Content-Type": "application/json", "Accept": "application/json"}
DOCKER_PREFIX    = "GNS3.OpenvSwitchLocal-"
MAX_CAPACITY_BPS = 10_000_000   # 10 Mbps

VPCS = {
    "172.16.1.10": "VPC-1",
    "172.16.1.20": "VPC-2",
    "172.16.1.30": "VPC-3",
    "172.16.1.40": "VPC-4",
    "172.16.1.50": "VPC-5",
}


# ── Utilitários base ──────────────────────────────────────────────────────
def get_containers() -> dict:
    try:
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=5)
        cs = [c.strip() for c in r.stdout.splitlines() if DOCKER_PREFIX in c]
        m = {}
        for c in cs:
            r2 = subprocess.run(
                ["docker", "exec", c, "ovs-vsctl", "get", "bridge", "br0",
                 "other-config:datapath-id"],
                capture_output=True, text=True, timeout=3)
            dpid = r2.stdout.strip().strip('"')
            if dpid and len(dpid) == 16:
                m[f"openflow:{int(dpid,16)}"] = c
        return dict(sorted(m.items()))
    except Exception as e:
        print(f"  ❌ docker: {e}")
        return {}


def dump_flows(container: str) -> list:
    r = subprocess.run(
        ["docker", "exec", container, "ovs-ofctl", "dump-flows", "br0", "-O", "OpenFlow13"],
        capture_output=True, text=True, timeout=5)
    return [l for l in r.stdout.splitlines() if "priority=" in l]


def prio(line: str) -> int:
    m = re.search(r"priority=(\d+)", line)
    return int(m.group(1)) if m else 0


def pkts(line: str) -> int:
    m = re.search(r"n_packets=(\d+)", line)
    return int(m.group(1)) if m else 0


def orch_get(path: str):
    try:
        return requests.get(f"{ORCHESTRATOR}{path}", timeout=5).json()
    except Exception as e:
        print(f"  ❌ Orquestrador {path}: {e}")
        return None


def orch_post(path: str, body: dict):
    try:
        return requests.post(f"{ORCHESTRATOR}{path}", json=body, timeout=5).json()
    except Exception as e:
        print(f"  ❌ Orquestrador {path}: {e}")
        return None


def sep(c="─", n=64):
    print(f"  {c*n}")


# ══════════════════════════════════════════════════════════════════════════
# [1] LIMPEZA DE FLOWS DO L2SWITCH
# ══════════════════════════════════════════════════════════════════════════
def cmd_clean():
    """
    Remove flows instalados reativamente pelo l2switch (priority 1 e 2).
    Os flows do orquestrador (priority 0, 1000, 5000, 60000+) são preservados.

    Por que esses flows existem:
      O l2switch, ao receber um PACKET_IN, aprende o MAC origem e instala um
      flow priority=2 de volta para aquela porta. Esses flows acumulam
      contadores históricos mas são inofensivos — o SDN (priority=60000)
      sempre vence. A limpeza é estética: remove os avisos do sdn_verify.

    Por que não são reinstalados:
      Após a limpeza, novos PACKET_IN que não tiverem match em priority=60000
      (ex: destinos ainda não conhecidos) farão o l2switch reinstalar flows
      de flood. Isso é normal e transitório — assim que o orquestrador
      instalar a rota SDN, ela assume o controle.
    """
    print("\n🧹 LIMPEZA DE FLOWS ANTIGOS DO L2SWITCH")
    sep("═")
    cmap = get_containers()
    if not cmap:
        print("  ❌ Nenhum container encontrado.")
        return

    total_removed = 0
    for sw_id, c in sorted(cmap.items()):
        before    = dump_flows(c)
        l2sw_lines = [l for l in before if 0 < prio(l) <= 2]
        pkts_old  = sum(pkts(l) for l in l2sw_lines)

        if not l2sw_lines:
            print(f"  ✅ {sw_id}: limpo (nenhum flow l2switch)")
            continue

        print(f"\n  {sw_id}  ({len(l2sw_lines)} flows, {pkts_old:,} pkts acumulados)")

        for p in [2, 1]:
            subprocess.run(
                ["docker", "exec", c, "ovs-ofctl", "del-flows", "br0",
                 f"priority={p}", "-O", "OpenFlow13"],
                capture_output=True, timeout=5)

        after     = dump_flows(c)
        remaining = [l for l in after if 0 < prio(l) <= 2]
        removed   = len(l2sw_lines) - len(remaining)
        total_removed += removed

        sdn_preserved = [l for l in after if prio(l) >= 1000]
        print(f"  ✅ {removed} flows removidos  |  "
              f"{len(sdn_preserved)} flows SDN preservados (pri≥1000)")

        if remaining:
            print(f"  ⚠️  {len(remaining)} flows não removidos — verifique manualmente")

    sep()
    print(f"  Total removido: {total_removed} flows")
    print(f"\n  ℹ️  O l2switch pode reinstalar flows priority=2 ao processar")
    print(f"     novos PACKET_IN, mas o SDN (priority=60000) sempre vence.")


# ══════════════════════════════════════════════════════════════════════════
# [2] TESTE DE DESVIO DINÂMICO
# ══════════════════════════════════════════════════════════════════════════
def snapshot_routes(cmap: dict) -> dict:
    """Retorna {sw_id: {(priority, dst_ip): {out, pkts}}}"""
    snap = {}
    for sw_id, c in cmap.items():
        snap[sw_id] = {}
        for line in dump_flows(c):
            p = prio(line)
            if p not in (60000, 62000):
                continue
            dst = re.search(r"nw_dst=([\d./]+)", line)
            out = re.search(r"output:(\S+)", line)
            if dst and out:
                snap[sw_id][(p, dst.group(1))] = {
                    "pkts": pkts(line), "out": out.group(1)
                }
    return snap


def cmd_reroute_test():
    """
    Testa o desvio dinâmico de caminho (reroute) do orquestrador.

    O orquestrador instala flows priority=62000 quando um link ultrapassa
    REROUTE_THRESH (75%) de utilização. Esses flows têm prioridade maior
    que as rotas normais (60000) e desviam o tráfego para outro caminho.
    Após o congestionamento, os flows 62000 expiram (idle_timeout=15s)
    e o tráfego retorna às rotas normais automaticamente.

    Para o teste funcionar é necessário:
      - Topologia com pelo menos 2 caminhos entre fonte e destino
      - Tráfego suficiente para saturar um link (>75% de 10Mbps = 7.5Mbps)
      - O orquestrador v13 rodando
    """
    print("\n🔀 TESTE DE DESVIO DINÂMICO DE CAMINHO")
    sep("═")

    cmap  = get_containers()
    state = orch_get("/state")
    if not state:
        print("  ❌ Orquestrador não respondeu.")
        return

    hosts   = state.get("hosts", {})
    switches = state.get("switches", [])

    print(f"  Rede: {len(switches)} switches | {len(hosts)} hosts")

    if len(hosts) < 2:
        print("  ❌ Menos de 2 hosts. Gere tráfego ARP nas VPCs primeiro.")
        return

    # ── Rotas antes ────────────────────────────────────────────────────
    print(f"\n  📸 Rotas SDN atuais (priority=60000):")
    sep()
    before = snapshot_routes(cmap)
    has_routes = False
    for sw_id in sorted(before):
        for (p, dst), info in sorted(before[sw_id].items(), key=lambda x: x[0][1]):
            if p == 60000:
                has_routes = True
                vpc = VPCS.get(dst, "")
                print(f"    {sw_id:14s}  dst={dst:15s} ({vpc:5s})  → {info['out']}")

    if not has_routes:
        print("  ⚠️  Nenhuma rota SDN ativa ainda.")
        return

    # ── Instruções ─────────────────────────────────────────────────────
    print(f"""
  📋 Como gerar congestionamento para o teste:

  OPÇÃO A — Ping flood nas VPCs do GNS3 (mais realista):
    VPC-1> ping 172.16.1.50 -f        (se suportado)
    VPC-1> ping 172.16.1.50 -c 10000

  OPÇÃO B — iperf entre containers (requer iperf instalado):
    docker exec <vpc-container> iperf -c 172.16.1.50 -t 30

  OPÇÃO C — Baixar MAX_LINK_CAPACITY no orquestrador:
    Edite sdn_orchestrator.py: MAX_LINK_CAPACITY = 50_000
    Reinicie — o LLDP/ARP normal já vai parecer congestionado.

  ⚠️  O reroute só funciona se houver links REDUNDANTES na topologia.
    Com 10 enlaces em 4 switches, você tem caminhos alternativos.
""")
    input("  → Inicie o tráfego e pressione Enter para começar o monitoramento...")

    # ── Monitoramento ──────────────────────────────────────────────────
    print(f"\n  ⏳ Monitorando por 90s (ciclos de 5s)...")
    sep()
    print(f"  {'t':>5}  {'Switch':14s}  {'Tipo':12s}  {'Destino':15s}  "
          f"{'Porta':6s}  {'Δpkts':>6}  {'Nota':}")
    sep()

    deadline      = time.time() + 90
    found_reroute = False
    prev_snap     = before

    while time.time() < deadline:
        time.sleep(5)
        curr = snapshot_routes(cmap)
        t    = int(90 - (deadline - time.time()))

        any_this_round = False
        for sw_id in sorted(curr):
            for (p, dst), info in sorted(curr[sw_id].items(), key=lambda x: x[0][1]):
                prev_pkts = prev_snap.get(sw_id, {}).get((p, dst), {}).get("pkts", 0)
                delta     = info["pkts"] - prev_pkts
                if delta <= 0:
                    continue

                vpc = VPCS.get(dst, dst)
                if p == 62000:
                    found_reroute  = True
                    any_this_round = True
                    orig_port = before.get(sw_id, {}).get((60000, dst), {}).get("out", "?")
                    changed   = "🔀 REROUTE" if info["out"] != orig_port else "🔀 reroute"
                    note      = f"antes→{orig_port}" if orig_port != "?" else ""
                    print(f"  {t:>5}s  {sw_id:14s}  {changed:12s}  "
                          f"{dst:15s}  {info['out']:6s}  +{delta:<6}  {note}")
                elif p == 60000:
                    any_this_round = True
                    print(f"  {t:>5}s  {sw_id:14s}  {'✅ normal':12s}  "
                          f"{dst:15s}  {info['out']:6s}  +{delta:<6}")

        prev_snap = curr

        if found_reroute and not any_this_round:
            # Reroute existiu mas parou → convergiu de volta
            print(f"\n  ✅ Reroute terminou — tráfego voltou às rotas normais (flows 62000 expiraram)")
            break

    sep()
    if found_reroute:
        print("  ✅ TESTE PASSOU: SDN detectou congestionamento e desviou o tráfego")
        print("     Os flows priority=62000 são temporários (idle_timeout=15s)")
        print("     e expiram sozinhos quando o link descongestiona.")
    else:
        print("  ⚠️  Nenhum reroute detectado em 90s.")
        print("     Verifique: MAX_LINK_CAPACITY, redundância na topologia,")
        print("     e se o tráfego gerado foi suficiente (>75% de 10Mbps = 7,5Mbps).")


# ══════════════════════════════════════════════════════════════════════════
# [3a] BLOQUEIO / DESBLOQUEIO
# ══════════════════════════════════════════════════════════════════════════
def cmd_block(ip: str):
    print(f"\n🚫 BLOQUEANDO {ip}  ({VPCS.get(ip, 'host desconhecido')})")
    sep("═")

    cmap  = get_containers()
    state = orch_get("/state")
    if state:
        known = any(ip in info.get("ips", [])
                    for info in state.get("hosts", {}).values())
        if not known:
            print(f"  ⚠️  {ip} não está na lista de hosts conhecidos")
            print(f"      O bloqueio será pré-instalado para quando o host aparecer")

    result = orch_post("/manage/ip", {"ip": ip, "action": "block"})
    if not result or result.get("status") != "ok":
        print(f"  ❌ Falha: {result}"); return
    print(f"  ✅ Orquestrador confirmou bloqueio")

    print(f"  Aguardando flows DROP nos switches (até 12s)...")
    installed = {}
    deadline  = time.time() + 12
    while time.time() < deadline:
        time.sleep(2)
        for sw_id, c in cmap.items():
            if sw_id in installed:
                continue
            for line in dump_flows(c):
                if f"nw_src={ip}" in line and prio(line) == 65500:
                    installed[sw_id] = True
                    print(f"  🛑 DROP flow instalado em {sw_id}")
        if len(installed) == len(cmap):
            break

    sep()
    if installed:
        print(f"  ✅ {ip} bloqueado em {len(installed)}/{len(cmap)} switches")
        print(f"\n  Teste: ping {ip} de outra VPC → deve retornar sem resposta")
        print(f"  Para desbloquear: python3 sdn_tools.py unblock {ip}")
    else:
        print(f"  ⚠️  Flows DROP ainda não visíveis — aguarde o próximo ciclo (~5s)")


def cmd_unblock(ip: str):
    print(f"\n✅ DESBLOQUEANDO {ip}  ({VPCS.get(ip, 'host desconhecido')})")
    sep("═")

    cmap   = get_containers()
    result = orch_post("/manage/ip", {"ip": ip, "action": "unblock"})
    if not result or result.get("status") != "ok":
        print(f"  ❌ Falha: {result}"); return
    print(f"  ✅ Orquestrador confirmou desbloqueio")

    print(f"  Verificando remoção dos flows DROP (3s)...")
    time.sleep(3)
    remaining = [sw_id for sw_id, c in cmap.items()
                 if any(f"nw_src={ip}" in l and prio(l) == 65500
                        for l in dump_flows(c))]

    if not remaining:
        print(f"  ✅ Todos os flows DROP removidos — {ip} acessível novamente")
    else:
        print(f"  ⚠️  DROP ainda em: {remaining} — aguarde o próximo ciclo")


# ══════════════════════════════════════════════════════════════════════════
# [3b] MÉTRICAS EM TEMPO REAL
# ══════════════════════════════════════════════════════════════════════════
def cmd_metrics():
    print("\n📊 MÉTRICAS EM TEMPO REAL  (Ctrl+C para parar)")
    sep("═")

    cmap             = get_containers()
    port_stats_prev  : dict = {}
    INTERVAL         = 5

    try:
        while True:
            ts = time.strftime("%H:%M:%S")
            w  = 64

            print(f"\n  ╔{'═'*w}╗")
            print(f"  ║  🕐 {ts}  —  SDN Network Dashboard"
                  f"{' '*(w - 34)}║")
            print(f"  ╠{'═'*w}╣")

            # ── Orquestrador ──────────────────────────────────────────
            health = orch_get("/health")
            state  = orch_get("/state")

            if health:
                cyc = health.get("cycle", 0)
                sw  = health.get("switches", 0)
                hs  = health.get("hosts", 0)
                line = f"  Orquestrador: ciclo #{cyc}  switches={sw}  hosts={hs}"
                print(f"  ║  {line}{' '*(w-2-len(line))}║")
            else:
                print(f"  ║  ⚠️  Orquestrador offline"
                      f"{' '*(w-26)}║")

            # ── Hosts ─────────────────────────────────────────────────
            print(f"  ╠{'─'*w}╣")
            print(f"  ║  {'HOSTS':}{' '*(w-7)}║")
            if state:
                hosts      = state.get("hosts", {})
                blocked_ips = state.get("blocked_ips", [])
                for mac, info in sorted(hosts.items(),
                                        key=lambda x: x[1].get("ips",[""])[0]):
                    ip  = info.get("ips", ["?"])[0]
                    sw  = info.get("switch", "?")
                    prt = info.get("port", "?")
                    vpc = VPCS.get(ip, "?????")
                    blk = "  🚫 BLOQUEADO" if ip in blocked_ips else ""
                    line = f"  {ip:15s} ({vpc})  {sw} p{prt}{blk}"
                    print(f"  ║    {line}{' '*(w-6-len(line))}║")

            # ── Utilização de links ───────────────────────────────────
            print(f"  ╠{'─'*w}╣")
            print(f"  ║  {'LINKS':}{' '*(w-7)}║")
            try:
                r = requests.get(
                    f"{ODL}/rests/data/opendaylight-inventory:nodes"
                    f"?content=nonconfig",
                    auth=ODL_AUTH, headers=HEADERS, timeout=4)
                nodes_data = (r.json()
                              .get("opendaylight-inventory:nodes", {})
                              .get("node", []))
                port_bw: dict = {}
                for node in nodes_data:
                    if "openflow" not in node.get("id", ""):
                        continue
                    for nc in node.get("node-connector", []):
                        pid = nc.get("id", "")
                        if "LOCAL" in pid:
                            continue
                        stats = nc.get(
                            "opendaylight-port-statistics:"
                            "flow-capable-node-connector-statistics", {})
                        if not stats:
                            continue
                        total = (int(stats.get("bytes",{}).get("transmitted",0)) +
                                 int(stats.get("bytes",{}).get("received",0)))
                        delta    = max(0, total - port_stats_prev.get(pid, total))
                        port_bw[pid]       = (delta * 8) / INTERVAL
                        port_stats_prev[pid] = total

                tr = requests.get(
                    f"{ODL}/rests/data/network-topology:network-topology"
                    f"/topology=flow:1?content=nonconfig",
                    auth=ODL_AUTH, headers=HEADERS, timeout=4)
                links = (tr.json()
                         .get("network-topology:network-topology",{})
                         .get("topology",[{}])[0]
                         .get("link",[]))

                seen: set = set()
                for lk in links:
                    src_n = lk["source"]["source-node"]
                    dst_n = lk["destination"]["dest-node"]
                    if src_n.startswith("host:") or dst_n.startswith("host:"):
                        continue
                    pair = tuple(sorted([src_n, dst_n]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    bps  = max(port_bw.get(lk["source"]["source-tp"], 0),
                               port_bw.get(lk["destination"]["dest-tp"], 0))
                    pct  = min(bps / MAX_CAPACITY_BPS * 100, 100)
                    bar  = "█" * int(pct/5) + "░" * (20 - int(pct/5))
                    warn = " ⚠️" if pct > 75 else ("  " if pct > 50 else "  ")
                    a    = src_n.replace("openflow:", "sw")
                    b    = dst_n.replace("openflow:", "sw")
                    line = f"  {a}↔{b}  [{bar}] {bps/1000:6.1f}kbps {pct:5.1f}%{warn}"
                    print(f"  ║    {line}{' '*(w-6-len(line))}║")

            except Exception as e:
                err = f"  ⚠️  ODL stats: {str(e)[:40]}"
                print(f"  ║    {err}{' '*(w-6-len(err))}║")

            # ── Flows SDN vs l2switch ─────────────────────────────────
            print(f"  ╠{'─'*w}╣")
            print(f"  ║  {'FLOWS':}{' '*(w-7)}║")
            for sw_id, c in sorted(cmap.items()):
                raw    = dump_flows(c)
                sdn_p  = sum(pkts(l) for l in raw if prio(l) == 60000)
                rrt_p  = sum(pkts(l) for l in raw if prio(l) == 62000)
                l2w_p  = sum(pkts(l) for l in raw if prio(l) == 2)
                blk_n  = sum(1       for l in raw if prio(l) == 65500)
                rrt_s  = f"  🔀reroute:{rrt_p:,}" if rrt_p > 0 else ""
                blk_s  = f"  🚫{blk_n}bloq" if blk_n > 0 else ""
                sw_s   = sw_id.replace("openflow:", "sw")
                line   = f"  {sw_s}  SDN:{sdn_p:>8,}  l2sw:{l2w_p:>8,}{rrt_s}{blk_s}"
                print(f"  ║    {line}{' '*(w-6-len(line))}║")

            print(f"  ╚{'═'*w}╝")
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n  Métricas encerradas.")


# ══════════════════════════════════════════════════════════════════════════
# STATUS GERAL
# ══════════════════════════════════════════════════════════════════════════
def cmd_status():
    print("\n📋 STATUS GERAL")
    sep("═")

    health = orch_get("/health")
    state  = orch_get("/state")
    cmap   = get_containers()

    if health:
        print(f"\n  Orquestrador  : ✅ ciclo #{health['cycle']} | "
              f"{health['switches']} switches | {health['hosts']} hosts")
    else:
        print(f"\n  Orquestrador  : ❌ offline")

    if state:
        hosts       = state.get("hosts", {})
        blocked_ips = state.get("blocked_ips", [])
        print(f"\n  Hosts:")
        for mac, info in sorted(hosts.items(),
                                key=lambda x: x[1].get("ips",[""])[0]):
            ip  = info.get("ips",["?"])[0]
            sw  = info.get("switch","?")
            prt = info.get("port","?")
            vpc = VPCS.get(ip, "")
            blk = "  🚫 BLOQUEADO" if ip in blocked_ips else ""
            print(f"    {ip:15s} ({vpc:5s})  {sw} porta {prt}{blk}")
        if blocked_ips:
            print(f"\n  IPs bloqueados: {', '.join(blocked_ips)}")

    if cmap:
        print(f"\n  Flows por switch:")
        for sw_id, c in sorted(cmap.items()):
            raw  = dump_flows(c)
            sdn  = sum(1 for l in raw if prio(l) in (60000,62000,65500,5000,1000,0))
            l2sw = sum(1 for l in raw if 0 < prio(l) <= 2)
            blk  = sum(1 for l in raw if prio(l) == 65500)
            bl_s = f"  🚫 {blk} bloqueios" if blk else ""
            print(f"    {sw_id}: {len(raw)} flows total  "
                  f"(SDN={sdn}, l2sw={l2sw}){bl_s}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Manutenção e Testes do SDN",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("clean",        help="Remove flows l2switch dos switches")
    sub.add_parser("reroute-test", help="Testa desvio dinâmico por congestionamento")
    pb = sub.add_parser("block",   help="Bloqueia host pelo IP")
    pb.add_argument("ip")
    pu = sub.add_parser("unblock", help="Desbloqueia host pelo IP")
    pu.add_argument("ip")
    sub.add_parser("metrics",      help="Dashboard de métricas em tempo real")
    sub.add_parser("status",       help="Resumo geral do estado da rede")

    args = parser.parse_args()
    dispatch = {
        "clean":        cmd_clean,
        "reroute-test": cmd_reroute_test,
        "block":        lambda: cmd_block(args.ip),
        "unblock":      lambda: cmd_unblock(args.ip),
        "metrics":      cmd_metrics,
        "status":       cmd_status,
    }
    fn = dispatch.get(args.cmd)
    if fn:
        fn()
    else:
        parser.print_help()
    print()


if __name__ == "__main__":
    main()
