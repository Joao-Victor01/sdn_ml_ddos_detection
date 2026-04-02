#!/usr/bin/env python3
"""
Validação do endpoint /metrics/hosts do SDN Orchestrator.

Verifica se o bug de bw=0.0 foi corrigido e confirma que o caminho
Dijkstra está sendo calculado corretamente entre cada FL-Node e o
switch servidor (FL_SERVER_SWITCH).

Uso (com orquestrador rodando):
    python3 validate_network_metrics.py
    python3 validate_network_metrics.py --url http://127.0.0.1:8000
    python3 validate_network_metrics.py --loop 5  # repete 5 vezes a cada 10s
"""

import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("[ERRO] requests não instalado. Execute: pip install requests")
    sys.exit(1)

FL_CLIENT_IPS = [
    "172.16.1.10",  # FL-Node-1
    "172.16.1.16",  # FL-Node-5
    "172.16.1.11",  # FL-Node-2
    "172.16.1.14",  # FL-Node-4
    "172.16.1.13",  # FL-Node-3
    "172.16.1.17",  # FL-Node-6
]

SDN_MIN_BW   = 10.0   # Mbps
SDN_MAX_LAT  = 50.0   # ms
SDN_MAX_LOSS = 0.10


def get(url, path, timeout=5):
    try:
        r = requests.get(f"{url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [ERRO] GET {path}: {e}")
        return None


def check_health(url):
    d = get(url, "/health")
    if not d:
        return False
    print(f"  Status: {d.get('status')} | Ciclo: {d.get('cycle')} | "
          f"Switches: {d.get('switches')} | Hosts: {d.get('hosts')}")
    return d.get("status") == "ok"


def check_switches(url):
    d = get(url, "/state")
    if not d:
        return []
    switches = d.get("switches", [])
    print(f"  Switches no grafo ({len(switches)}): {switches}")
    return switches


def check_links(url):
    d = get(url, "/metrics/links")
    if not d:
        return
    links = d.get("links", {})
    summary = d.get("summary", {})
    print(f"  Total enlaces: {summary.get('total_links', 0)} | "
          f"Congestionados: {summary.get('congested_links', 0)} | "
          f"Aviso: {summary.get('warn_links', 0)}")
    cap = d.get("max_link_capacity_bps", 20_000_000)
    for name, info in sorted(links.items()):
        bps  = info.get("load_bps", 0)
        util = info.get("utilization", 0)
        bar  = "█" * int(min(util * 100, 100) / 5) + "░" * (20 - int(min(util * 100, 100) / 5))
        flag = " ⚠️ CONGESTIONADO" if info.get("congested") else (" ⚡ AVISO" if info.get("warn") else "")
        print(f"    {name:20s} [{bar}] {bps/1000:7.1f} kbps  {util*100:5.1f}%{flag}")


def check_hosts(url):
    d = get(url, "/metrics/hosts")
    if not d:
        return False

    hosts = d.get("hosts", {})
    print(f"  Hosts reportados: {len(hosts)}")

    all_ok = True
    bugs   = []

    for ip in FL_CLIENT_IPS:
        if ip not in hosts:
            print(f"  ❌  {ip}: NÃO ENCONTRADO no /metrics/hosts")
            all_ok = False
            bugs.append(f"{ip}: ausente")
            continue

        m    = hosts[ip]
        bw   = m.get("bandwidth_mbps", 0)
        lat  = m.get("latency_ms", 0)
        loss = m.get("packet_loss", 0)
        sw   = m.get("switch", "?")
        port = m.get("port", "?")

        # Calcula efficiency score (replica lógica do network.py)
        bw_cap   = 20.0
        bw_norm  = min(bw / bw_cap, 1.0)
        lat_norm = max(1.0 - (lat / SDN_MAX_LAT), 0.0)
        loss_norm = max(1.0 - (loss / SDN_MAX_LOSS), 0.0)
        eff = round(0.5 * bw_norm + 0.3 * lat_norm + 0.2 * loss_norm, 4)

        eligible = (bw >= SDN_MIN_BW and lat <= SDN_MAX_LAT and loss <= SDN_MAX_LOSS)
        status   = "✅ ELEGÍVEL" if eligible else "❌ INELEGÍVEL"

        # Detecta o bug: bw=0.0 quando há tráfego
        bug_flag = ""
        if bw == 0.0:
            bug_flag = "  ← ⚠️  BUG: bw=0.0 (path lookup falhou?)"
            bugs.append(f"{ip}: bw=0.0")
            all_ok = False

        print(f"  {status}  {ip} (sw={sw} porta={port})")
        print(f"           bw={bw:.1f}Mbps  lat={lat:.1f}ms  loss={loss:.3f}  eff={eff:.4f}{bug_flag}")

    return all_ok, bugs


def run_validation(url, iteration=None):
    label = f" [iteração {iteration}]" if iteration else ""
    print(f"\n{'='*65}")
    print(f"  VALIDAÇÃO /metrics/hosts{label}  —  {time.strftime('%H:%M:%S')}")
    print(f"{'='*65}")

    print("\n── 1. Health ──────────────────────────────────────────────────")
    if not check_health(url):
        print("  [AVISO] Orquestrador não respondeu ao /health")
        return False

    print("\n── 2. Switches no grafo ───────────────────────────────────────")
    switches = check_switches(url)
    if not switches:
        print("  [AVISO] Nenhum switch no grafo — topologia não descoberta ainda")

    print("\n── 3. Utilização dos enlaces ──────────────────────────────────")
    check_links(url)

    print("\n── 4. Métricas por host FL ────────────────────────────────────")
    result = check_hosts(url)
    if result is False:
        return False
    all_ok, bugs = result

    print(f"\n── Resultado ──────────────────────────────────────────────────")
    if all_ok:
        print("  ✅ TODOS os FL-Nodes têm bw > 0.0 — bug CORRIGIDO")
    else:
        print(f"  ❌ Problemas detectados: {bugs}")
        print("  Verifique:")
        print("    1. FL_SERVER_SWITCH em orchestrator/config.py (deve ser o")
        print("       ODL node-id do switch core — ex: 'openflow:1')")
        print("    2. Confirme com: curl -s http://localhost:8000/state | "
              "python3 -c \"import sys,json; print(json.load(sys.stdin)['switches'])\"")
        print("    3. Tráfego de fundo ativo? Se não, bw pode ser alto (link idle)")
        print("    4. Orquestrador teve ≥ 2 ciclos para estabelecer baseline de medição")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Valida /metrics/hosts do SDN Orchestrator")
    parser.add_argument("--url",  default="http://127.0.0.1:8000", help="URL base do orquestrador")
    parser.add_argument("--loop", type=int, default=1, help="Número de iterações (default: 1)")
    parser.add_argument("--interval", type=int, default=10, help="Segundos entre iterações (default: 10)")
    args = parser.parse_args()

    if args.loop == 1:
        ok = run_validation(args.url)
        sys.exit(0 if ok else 1)
    else:
        for i in range(1, args.loop + 1):
            run_validation(args.url, iteration=i)
            if i < args.loop:
                print(f"\n  Aguardando {args.interval}s...")
                time.sleep(args.interval)
        print("\nValidação concluída.")


if __name__ == "__main__":
    main()
