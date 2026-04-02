#!/usr/bin/env python3
"""
sdn_verify.py — Verificação de controle SDN
============================================
Responde duas perguntas fundamentais:
  1. O SDN instalou flows nos switches? (tabela do OVS)
  2. O tráfego de ping está passando pelos flows do SDN?

Uso:
  python3 sdn_verify.py            # diagnóstico completo
  python3 sdn_verify.py --watch    # monitoramento contínuo (Ctrl+C para parar)
  python3 sdn_verify.py --ping     # faz ping automático entre VPCs e mostra contadores
"""

import subprocess
import sys
import time
import re
import json
import argparse
from dataclasses import dataclass, field
from collections import defaultdict

# ── Configuração ───────────────────────────────────────────────────────────
DOCKER_PREFIX = "GNS3.OpenvSwitchLocal-"

# IPs das VPCs (ajuste se necessário)
VPCS = [
    "172.16.1.10",
    "172.16.1.20",
    "172.16.1.30",
    "172.16.1.40",
    "172.16.1.50",
]

# Classificação de flows por prioridade/cookie
FLOW_CLASSES = {
    65500: ("🚫 BLOCK-IP",    "Bloqueio de IP (orquestrador)"),
    62000: ("⚖️  REROUTE",    "Desvio de congestionamento (orquestrador)"),
    60000: ("✅ ROUTE-SDN",   "Rota IPv4 Dijkstra (orquestrador)"),
    5000:  ("🌳 ARP-MST",     "ARP spanning-tree (orquestrador)"),
    1000:  ("📡 LLDP/BDDP",   "LLDP/BDDP → controller (orquestrador)"),
    100:   ("📚 L2SW-UNI",    "Unicast aprendido (l2switch)"),
    2:     ("🌊 L2SW-FLOOD",  "Flood broadcast (l2switch — indesejado)"),
    1:     ("🔧 ARP-CTRL",    "ARP → controller (l2switch)"),
    0:     ("🔁 TABLE-MISS",  "Table-miss → controller (orquestrador)"),
}


# ── Estruturas de dados ────────────────────────────────────────────────────
@dataclass
class Flow:
    priority:    int
    match:       str
    actions:     str
    packets:     int
    bytes_:      int
    idle_age:    int  = 0
    cookie:      str  = "0x0"
    table:       int  = 0

    @property
    def owner(self) -> str:
        label, _ = FLOW_CLASSES.get(self.priority, ("❓ UNKNOWN", ""))
        return label

    @property
    def is_sdn_route(self) -> bool:
        return self.priority in (60000, 62000, 65500, 5000, 1000, 0)

    @property
    def is_l2switch(self) -> bool:
        return self.priority in (100, 2, 1)


@dataclass
class SwitchStats:
    sw_id:        str
    container:    str
    flows:        list[Flow] = field(default_factory=list)
    error:        str        = ""


# ── Utilitários Docker ─────────────────────────────────────────────────────
def list_containers() -> dict[str, str]:
    """Retorna {sw_id: container_name}"""
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        containers = [c.strip() for c in r.stdout.splitlines()
                      if DOCKER_PREFIX in c]
        mapping = {}
        for c in containers:
            r2 = subprocess.run(
                ["docker", "exec", c, "ovs-vsctl", "get", "bridge", "br0",
                 "other-config:datapath-id"],
                capture_output=True, text=True, timeout=3
            )
            dpid = r2.stdout.strip().strip('"')
            if dpid and len(dpid) == 16:
                sw_id = f"openflow:{int(dpid, 16)}"
                mapping[sw_id] = c
        return dict(sorted(mapping.items()))
    except Exception as e:
        print(f"❌ Erro ao listar containers: {e}")
        return {}


def dump_flows(container: str) -> list[Flow]:
    """Faz dump-flows no OVS e retorna lista de Flow."""
    try:
        r = subprocess.run(
            ["docker", "exec", container,
             "ovs-ofctl", "dump-flows", "br0", "-O", "OpenFlow13"],
            capture_output=True, text=True, timeout=5
        )
        return parse_flows(r.stdout)
    except Exception as e:
        return []


def parse_flows(raw: str) -> list[Flow]:
    flows = []
    for line in raw.splitlines():
        if "NXST_FLOW" in line or "OFP" in line:
            continue
        # Extrai campos chave
        prio  = int(re.search(r"priority=(\d+)", line).group(1)) if re.search(r"priority=(\d+)", line) else 0
        pkts  = int(re.search(r"n_packets=(\d+)", line).group(1)) if re.search(r"n_packets=(\d+)", line) else 0
        byt   = int(re.search(r"n_bytes=(\d+)", line).group(1)) if re.search(r"n_bytes=(\d+)", line) else 0
        idle  = int(re.search(r"idle_age=(\d+)", line).group(1)) if re.search(r"idle_age=(\d+)", line) else 0
        cook  = re.search(r"cookie=(0x[0-9a-f]+)", line)
        cookie = cook.group(1) if cook else "0x0"
        tbl   = int(re.search(r"table=(\d+)", line).group(1)) if re.search(r"table=(\d+)", line) else 0

        # Separa match e actions
        acts_m = re.search(r"actions=(.+)$", line)
        actions = acts_m.group(1).strip() if acts_m else ""
        match_part = re.sub(r"\s*(cookie|duration|table|n_packets|n_bytes|"
                            r"idle_age|hard_age|send_flow_rem|actions)=\S+", "", line)
        match_part = re.sub(r"\s+", " ", match_part).strip()

        flows.append(Flow(
            priority=prio, match=match_part, actions=actions,
            packets=pkts, bytes_=byt, idle_age=idle,
            cookie=cookie, table=tbl
        ))
    return sorted(flows, key=lambda f: -f.priority)


# ── Relatório 1: tabela de flows ──────────────────────────────────────────
def report_flow_tables(containers: dict[str, str]):
    print("\n" + "═" * 70)
    print("  📋  TABELAS DE FLOW DOS SWITCHES")
    print("═" * 70)

    totals = defaultdict(lambda: {"flows": 0, "pkts": 0, "bytes": 0})
    grand_sdn  = {"flows": 0, "pkts": 0}
    grand_l2sw = {"flows": 0, "pkts": 0}

    for sw_id, container in containers.items():
        flows = dump_flows(container)
        sdn_pkts  = sum(f.packets for f in flows if f.is_sdn_route)
        l2sw_pkts = sum(f.packets for f in flows if f.is_l2switch)
        total_pkts = sum(f.packets for f in flows)

        print(f"\n  ┌─ {sw_id}  ({container.split('.')[1]})")
        print(f"  │  {len(flows)} flows   |   "
              f"SDN: {sdn_pkts:,} pkts   |   "
              f"l2switch: {l2sw_pkts:,} pkts")
        print(f"  │")

        # Agrupa por prioridade
        by_prio = defaultdict(list)
        for f in flows:
            by_prio[f.priority].append(f)

        for prio in sorted(by_prio.keys(), reverse=True):
            group   = by_prio[prio]
            label   = FLOW_CLASSES.get(prio, ("❓", ""))[0]
            g_pkts  = sum(f.packets for f in group)
            g_bytes = sum(f.bytes_   for f in group)
            marker  = "⚠️ " if prio in (2, 100) and g_pkts > 0 else ""
            print(f"  │  {label:18s}  pri={prio:<6}  "
                  f"{len(group):2d} flows  "
                  f"{g_pkts:>8,} pkts  "
                  f"{g_bytes / 1024:>8.1f} KB  {marker}")

            # Mostra detalhes dos flows SDN com pacotes
            if prio in (60000, 62000) and g_pkts > 0:
                for f in sorted(group, key=lambda x: -x.packets):
                    if f.packets > 0:
                        dst = re.search(r"nw_dst=([\d./]+)", f.match)
                        out = f.actions
                        print(f"  │    └─ dst={dst.group(1) if dst else '?':18s} "
                              f"→ {out:20s}  {f.packets:,} pkts")

        grand_sdn["flows"]  += sum(1 for f in flows if f.is_sdn_route)
        grand_sdn["pkts"]   += sdn_pkts
        grand_l2sw["flows"] += sum(1 for f in flows if f.is_l2switch)
        grand_l2sw["pkts"]  += l2sw_pkts
        print(f"  └─")

    print(f"\n  {'─'*66}")
    print(f"  TOTAIS  │  SDN: {grand_sdn['flows']} flows, {grand_sdn['pkts']:,} pkts"
          f"   │   l2switch: {grand_l2sw['flows']} flows, {grand_l2sw['pkts']:,} pkts")


# ── Relatório 2: verificação de controle SDN ─────────────────────────────
def report_sdn_control(containers: dict[str, str]):
    print("\n" + "═" * 70)
    print("  🔬  VERIFICAÇÃO DE CONTROLE SDN")
    print("═" * 70)

    issues = []
    good   = []

    for sw_id, container in containers.items():
        flows = dump_flows(container)
        if not flows:
            issues.append(f"  ❌ {sw_id}: sem flows (switch desconectado?)")
            continue

        prio_set = {f.priority for f in flows}

        # Verifica flows obrigatórios
        has_table_miss = 0 in prio_set
        has_lldp       = 1000 in prio_set
        has_arp_mst    = 5000 in prio_set
        has_ipv4_route = 60000 in prio_set

        # Verifica idle_timeout nas rotas SDN
        sdn_routes = [f for f in flows if f.priority == 60000]
        for f in sdn_routes:
            if "idle_timeout" in f.match and "idle_timeout=0" not in f.match:
                idle_m = re.search(r"idle_timeout=(\d+)", f.match)
                if idle_m and int(idle_m.group(1)) > 0:
                    issues.append(
                        f"  ⚠️  {sw_id}: rota SDN com idle_timeout={idle_m.group(1)}s "
                        f"— pode expirar! Use idle_timeout=0 (v13+)"
                    )

        # Verifica se tráfego vai pelo SDN ou pelo l2switch
        sdn_pkts  = sum(f.packets for f in flows if f.priority == 60000)
        l2sw_pkts = sum(f.packets for f in flows if f.priority in (2, 100))
        flood_flows = [f for f in flows if f.priority == 2 and f.packets > 0]

        status_parts = []
        if has_table_miss: status_parts.append("table-miss ✓")
        else:              status_parts.append("table-miss ✗"); issues.append(f"  ⚠️  {sw_id}: sem table-miss flow")
        if has_lldp:       status_parts.append("LLDP ✓")
        if has_arp_mst:    status_parts.append("ARP-MST ✓")
        else:              status_parts.append("ARP-MST ✗"); issues.append(f"  ⚠️  {sw_id}: sem ARP spanning-tree flows")
        if has_ipv4_route: status_parts.append("IPv4-route ✓")
        else:              status_parts.append("IPv4-route ✗"); issues.append(f"  ⚠️  {sw_id}: sem rotas IPv4 SDN")

        if flood_flows:
            issues.append(
                f"  ⚠️  {sw_id}: {len(flood_flows)} flood flows (l2switch priority=2) "
                f"com {sum(f.packets for f in flood_flows):,} pkts — "
                f"l2switch interferindo!"
            )

        if sdn_pkts > 0:
            good.append(f"  ✅ {sw_id}: {sdn_pkts:,} pkts via rotas SDN (priority=60000)")
        elif has_ipv4_route:
            good.append(f"  ✅ {sw_id}: rotas SDN instaladas (aguardando tráfego)")

        print(f"\n  {sw_id}:  {' | '.join(status_parts)}")

    print()
    for g in good:
        print(g)
    if issues:
        print()
        for i in issues:
            print(i)
    else:
        print("\n  ✅ Nenhum problema detectado — SDN em controle total!")


# ── Relatório 3: verificação de ping via contadores ───────────────────────
def report_ping_verification(containers: dict[str, str]):
    """
    Tira snapshot dos contadores ANTES, envia um ping, e compara com DEPOIS.
    Mostra em qual flow (SDN vs l2switch) os pacotes foram contados.
    """
    print("\n" + "═" * 70)
    print("  🏓  VERIFICAÇÃO DE PING — CONTADORES DE FLOW")
    print("═" * 70)

    if len(VPCS) < 2:
        print("  Configure a lista VPCS com pelo menos 2 IPs.")
        return

    src_ip = VPCS[0]
    dst_ip = VPCS[-1]

    print(f"\n  Origem:  {src_ip}")
    print(f"  Destino: {dst_ip}")
    print(f"\n  Snapshot ANTES do ping...")

    # Snapshot antes
    before: dict[str, list[Flow]] = {}
    for sw_id, container in containers.items():
        before[sw_id] = dump_flows(container)

    # Ping (executa no host via ping ou nc)
    print(f"  Enviando ping {src_ip} → {dst_ip} (5 pacotes)...")
    try:
        r = subprocess.run(
            ["ping", "-c", "5", "-W", "2", dst_ip],
            capture_output=True, text=True, timeout=15
        )
        ping_ok   = r.returncode == 0
        ping_out  = r.stdout.strip().split("\n")[-1]  # linha de estatísticas
    except Exception as e:
        ping_ok  = False
        ping_out = str(e)

    if ping_ok:
        print(f"  ✅ Ping OK: {ping_out}")
    else:
        print(f"  ❌ Ping FALHOU: {ping_out}")
        print(f"     (Se as VPCs estão em containers separados, rode o ping")
        print(f"      manualmente e pressione Enter para capturar contadores)")
        input("  → Pressione Enter após fazer o ping manualmente...")

    time.sleep(0.5)

    # Snapshot depois
    print(f"\n  Snapshot DEPOIS do ping...")
    after: dict[str, list[Flow]] = {}
    for sw_id, container in containers.items():
        after[sw_id] = dump_flows(container)

    # Diff de contadores
    print(f"\n  {'─'*66}")
    print(f"  FLOWS QUE RECEBERAM PACOTES DURANTE O PING")
    print(f"  {'─'*66}")

    found_sdn   = False
    found_l2sw  = False

    for sw_id in sorted(containers.keys()):
        bflows = {(f.priority, f.actions): f for f in before.get(sw_id, [])}
        aflows = {(f.priority, f.actions): f for f in after.get(sw_id, [])}

        changed = []
        for key, af in aflows.items():
            bf = bflows.get(key)
            delta_pkts = af.packets - (bf.packets if bf else 0)
            if delta_pkts > 0 and af.priority in (60000, 62000, 100, 2, 1000):
                changed.append((af, delta_pkts))

        if changed:
            print(f"\n  {sw_id}:")
            for f, delta in sorted(changed, key=lambda x: -x[0].priority):
                dst  = re.search(r"nw_dst=([\d./]+)", f.match)
                src  = re.search(r"nw_src=([\d./]+)", f.match)
                mark = ""
                if f.priority == 60000:
                    mark      = "← SDN instalou esta rota ✅"
                    found_sdn = True
                elif f.priority == 62000:
                    mark      = "← Reroute SDN ✅"
                    found_sdn = True
                elif f.priority in (100, 2):
                    mark       = "← l2switch encaminhou ⚠️"
                    found_l2sw = True

                match_desc = f"dst={dst.group(1)}" if dst else f.match[:40]
                print(f"    pri={f.priority:<6}  {match_desc:30s}  "
                      f"+{delta:>4} pkts  {mark}")

    print(f"\n  {'─'*66}")
    if found_sdn and not found_l2sw:
        print("  ✅ CONCLUSÃO: Tráfego encaminhado EXCLUSIVAMENTE pelo SDN")
    elif found_sdn and found_l2sw:
        print("  ⚠️  CONCLUSÃO: Tráfego compartilhado entre SDN e l2switch")
        print("     Verifique se os flows ARP_MST estão instalados corretamente")
    elif found_l2sw and not found_sdn:
        print("  ❌ CONCLUSÃO: Tráfego via l2switch — SDN NÃO está no controle!")
        print("     Verifique se o orquestrador está rodando e instalando flows")
    else:
        print("  ⚠️  Nenhum counter incrementado — ping pode ter falhado")


# ── Modo watch: monitora contadores em tempo real ────────────────────────
def watch_mode(containers: dict[str, str], interval: int = 3):
    print("\n  📊 MODO WATCH — Contadores SDN em tempo real (Ctrl+C para parar)")
    print(f"  Intervalo: {interval}s\n")
    prev: dict[str, dict] = {}

    try:
        while True:
            ts   = time.strftime("%H:%M:%S")
            curr: dict[str, dict] = {}

            lines = []
            for sw_id, container in containers.items():
                flows    = dump_flows(container)
                sdn_pkts = sum(f.packets for f in flows if f.priority == 60000)
                l2s_pkts = sum(f.packets for f in flows if f.priority in (2, 100))
                mst_pkts = sum(f.packets for f in flows if f.priority == 5000)
                curr[sw_id] = {"sdn": sdn_pkts, "l2s": l2s_pkts, "mst": mst_pkts}

                p = prev.get(sw_id, {})
                d_sdn = sdn_pkts - p.get("sdn", sdn_pkts)
                d_l2s = l2s_pkts - p.get("l2s", l2s_pkts)
                d_mst = mst_pkts - p.get("mst", mst_pkts)

                flag = ""
                if d_l2s > 0 and d_sdn == 0:
                    flag = "⚠️  l2switch encaminhando!"
                elif d_sdn > 0:
                    flag = "✅ SDN ativo"

                lines.append(
                    f"  {sw_id:14s}  "
                    f"SDN Δ={d_sdn:>5}  "
                    f"MST Δ={d_mst:>5}  "
                    f"l2sw Δ={d_l2s:>5}  {flag}"
                )

            prev = curr
            print(f"\r[{ts}]")
            for l in lines:
                print(l)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n  Modo watch encerrado.")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verificação SDN")
    parser.add_argument("--watch", action="store_true",
                        help="Monitoramento contínuo de contadores")
    parser.add_argument("--ping",  action="store_true",
                        help="Verifica ping via contadores de flow")
    parser.add_argument("--flows", action="store_true",
                        help="Mostra apenas a tabela de flows")
    args = parser.parse_args()

    print("\n🔍 Coletando estado dos switches...")
    containers = list_containers()

    if not containers:
        print("❌ Nenhum container OVS encontrado.")
        print(f"   Verifique se DOCKER_PREFIX='{DOCKER_PREFIX}' está correto.")
        sys.exit(1)

    print(f"   Switches: {', '.join(containers.keys())}")

    if args.watch:
        watch_mode(containers)
        return

    if args.ping:
        report_ping_verification(containers)
        return

    if args.flows:
        report_flow_tables(containers)
        return

    # Relatório completo (padrão)
    report_flow_tables(containers)
    report_sdn_control(containers)

    print("\n" + "═" * 70)
    print("  💡 DICAS")
    print("═" * 70)
    print("  python3 sdn_verify.py --watch   → monitora contadores em tempo real")
    print("  python3 sdn_verify.py --ping    → testa ping e identifica qual flow encaminhou")
    print("  python3 sdn_verify.py --flows   → só tabela de flows")
    print()


if __name__ == "__main__":
    main()
