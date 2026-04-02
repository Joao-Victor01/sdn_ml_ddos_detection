#!/usr/bin/env python3
"""
Gerador de gráficos para o TCC — FL com e sem SDN.

Uso básico:
    python3 plot_results.py --com com_sdn_resultados.csv --sem sem_sdn_resultados.csv

Com overlay de eventos SDN (reroutes):
    python3 plot_results.py --com com_sdn_resultados.csv \
                            --sem sem_sdn_resultados.csv \
                            --sdn-metrics sdn_metrics.csv

Gráficos gerados (pasta --output-dir, padrão: ./plots/):
    fig1_accuracy_tempo.png   — Accuracy × Tempo (com vs sem SDN)
    fig2_f1_tempo.png         — F1-Score × Tempo (com vs sem SDN)
    fig3_duracao_round.png    — Duração por round (barras lado a lado)
    fig4_auc_round.png        — AUC-ROC × Round
    fig5_overhead_sdn.png     — Overhead de controle SDN (se --sdn-metrics)

Formato esperado dos CSVs de resultados FL:
    Colunas obrigatórias: round, elapsed_sec, accuracy
    Colunas opcionais:    f1, auc

Formato do CSV SDN (gerado pelo MetricsCollector):
    timestamp, cycle, elapsed_sec, cycle_duration_sec,
    n_switches, n_hosts, n_flows, n_reroute_flows,
    max_link_load_bps, avg_link_load_bps, congested_links, warn_links
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # sem display — funciona em servidor headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Paleta de cores ──────────────────────────────────────────────────────────
C_SEM = "#e74c3c"   # vermelho — sem SDN
C_COM = "#2980b9"   # azul     — com SDN
C_SDN = "#27ae60"   # verde    — eventos SDN

FONTSIZE_LABEL  = 12
FONTSIZE_TITLE  = 13
FONTSIZE_LEGEND = 11


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str, name: str) -> pd.DataFrame:
    """Carrega CSV e valida colunas mínimas."""
    if not os.path.exists(path):
        print(f"[ERRO] Arquivo não encontrado: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"round", "elapsed_sec", "accuracy"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERRO] {name} faltam colunas: {missing}")
        sys.exit(1)
    df = df.sort_values("round").reset_index(drop=True)
    return df


def tempo_para_atingir(df: pd.DataFrame, frac: float = 0.95) -> float:
    """Retorna elapsed_sec no qual accuracy atingiu frac * max_accuracy."""
    alvo  = df["accuracy"].max() * frac
    linha = df[df["accuracy"] >= alvo]
    if linha.empty:
        return float("nan")
    return linha.iloc[0]["elapsed_sec"]


def mark_reroute_cycles(ax, sdn: pd.DataFrame, x_col: str = "elapsed_sec") -> None:
    """Adiciona linhas verticais cinzas onde houve reroute SDN."""
    reroute_rows = sdn[sdn["n_reroute_flows"] > 0]
    for _, row in reroute_rows.iterrows():
        ax.axvline(row[x_col], color=C_SDN, linewidth=0.8, alpha=0.4, linestyle="--")


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Figura 1 — Accuracy × Tempo ──────────────────────────────────────────────

def fig_accuracy_tempo(com: pd.DataFrame, sem: pd.DataFrame,
                       sdn: pd.DataFrame | None, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sem["elapsed_sec"], sem["accuracy"],
            color=C_SEM, linestyle="--", linewidth=2, label="Sem SDN")
    ax.plot(com["elapsed_sec"], com["accuracy"],
            color=C_COM, linestyle="-",  linewidth=2, label="Com SDN")

    # Linha de referência: 95% do máximo sem SDN
    threshold = sem["accuracy"].max() * 0.95
    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
               label=f"Limiar 95% ({threshold:.3f})")

    # Anotações de tempo para atingir limiar
    t_sem = tempo_para_atingir(sem)
    t_com = tempo_para_atingir(com)
    if not pd.isna(t_sem):
        ax.axvline(t_sem, color=C_SEM, linestyle=":", linewidth=1, alpha=0.6)
    if not pd.isna(t_com):
        ax.axvline(t_com, color=C_COM, linestyle=":", linewidth=1, alpha=0.6)

    if sdn is not None:
        mark_reroute_cycles(ax, sdn, "elapsed_sec")

    if not pd.isna(t_sem) and not pd.isna(t_com):
        reducao = (t_sem - t_com) / t_sem * 100
        ax.set_title(
            f"Accuracy do processo FL — redução de tempo: {reducao:.1f}%",
            fontsize=FONTSIZE_TITLE
        )
    else:
        ax.set_title("Accuracy do processo FL", fontsize=FONTSIZE_TITLE)

    ax.set_xlabel("Tempo (s)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Accuracy",  fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, os.path.join(output_dir, "fig1_accuracy_tempo.png"))


# ── Figura 2 — F1 × Tempo ────────────────────────────────────────────────────

def fig_f1_tempo(com: pd.DataFrame, sem: pd.DataFrame,
                 sdn: pd.DataFrame | None, output_dir: str) -> None:
    if "f1" not in com.columns or "f1" not in sem.columns:
        print("  [aviso] Coluna 'f1' ausente — fig2 pulada")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sem["elapsed_sec"], sem["f1"],
            color=C_SEM, linestyle="--", linewidth=2, label="Sem SDN")
    ax.plot(com["elapsed_sec"], com["f1"],
            color=C_COM, linestyle="-",  linewidth=2, label="Com SDN")

    threshold = sem["f1"].max() * 0.95
    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
               label=f"Limiar 95% ({threshold:.3f})")

    if sdn is not None:
        mark_reroute_cycles(ax, sdn, "elapsed_sec")

    ax.set_xlabel("Tempo (s)",  fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("F1-Score",   fontsize=FONTSIZE_LABEL)
    ax.set_title("F1-Score do processo FL", fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, os.path.join(output_dir, "fig2_f1_tempo.png"))


# ── Figura 3 — Duração por round ─────────────────────────────────────────────

def fig_duracao_round(com: pd.DataFrame, sem: pd.DataFrame,
                      sdn: pd.DataFrame | None, output_dir: str) -> None:
    # Duração de cada round = diferença de elapsed_sec entre rounds consecutivos
    com = com.copy()
    sem = sem.copy()
    com["round_duration"] = com["elapsed_sec"].diff().fillna(com["elapsed_sec"].iloc[0])
    sem["round_duration"] = sem["elapsed_sec"].diff().fillna(sem["elapsed_sec"].iloc[0])

    rounds     = sorted(set(com["round"]) | set(sem["round"]))
    bar_width  = 0.35
    x          = range(len(rounds))
    round_idx  = {r: i for i, r in enumerate(rounds)}

    dur_sem = [sem.loc[sem["round"] == r, "round_duration"].values[0]
               if r in sem["round"].values else 0 for r in rounds]
    dur_com = [com.loc[com["round"] == r, "round_duration"].values[0]
               if r in com["round"].values else 0 for r in rounds]

    fig, ax = plt.subplots(figsize=(max(10, len(rounds) * 0.6), 5))

    bars_sem = ax.bar([i - bar_width / 2 for i in x], dur_sem,
                      width=bar_width, color=C_SEM, alpha=0.8, label="Sem SDN")
    bars_com = ax.bar([i + bar_width / 2 for i in x], dur_com,
                      width=bar_width, color=C_COM, alpha=0.8, label="Com SDN")

    # Marca rounds com reroute SDN ativo
    if sdn is not None:
        reroute_cycles = set(sdn.loc[sdn["n_reroute_flows"] > 0, "cycle"])
        for r in rounds:
            if r in reroute_cycles:
                i = round_idx[r]
                ax.annotate("↓SDN", xy=(i, max(dur_sem[i], dur_com[i])),
                            ha="center", va="bottom", fontsize=8,
                            color=C_SDN, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(r) for r in rounds], fontsize=9)
    ax.set_xlabel("Round",          fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Duração (s)",    fontsize=FONTSIZE_LABEL)
    ax.set_title("Duração por round — impacto do reroute SDN", fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, os.path.join(output_dir, "fig3_duracao_round.png"))


# ── Figura 4 — AUC-ROC × Round ───────────────────────────────────────────────

def fig_auc_round(com: pd.DataFrame, sem: pd.DataFrame, output_dir: str) -> None:
    if "auc" not in com.columns or "auc" not in sem.columns:
        print("  [aviso] Coluna 'auc' ausente — fig4 pulada")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sem["round"], sem["auc"],
            color=C_SEM, linestyle="--", marker="o", markersize=4,
            linewidth=1.5, label="Sem SDN")
    ax.plot(com["round"], com["auc"],
            color=C_COM, linestyle="-",  marker="o", markersize=4,
            linewidth=1.5, label="Com SDN")

    ax.set_xlabel("Round",   fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("AUC-ROC", fontsize=FONTSIZE_LABEL)
    ax.set_title("AUC-ROC por round — qualidade do modelo FL", fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, os.path.join(output_dir, "fig4_auc_round.png"))


# ── Figura 5 — Overhead SDN (carga por ciclo) ────────────────────────────────

def fig_overhead_sdn(sdn: pd.DataFrame, output_dir: str) -> None:
    """
    Plota carga máxima de enlace e flows de reroute ativos por ciclo.
    Replica o conceito das Figuras 7-8 do artigo (overhead × benefício do SDN).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Subplot superior: carga máxima e média de enlace
    ax1.fill_between(sdn["elapsed_sec"], sdn["avg_link_load_bps"] / 1e6,
                     alpha=0.3, color=C_COM, label="Carga média")
    ax1.plot(sdn["elapsed_sec"], sdn["max_link_load_bps"] / 1e6,
             color=C_COM, linewidth=1.5, label="Carga máxima")

    # Linhas de threshold
    from orchestrator.config import MAX_LINK_CAPACITY, REROUTE_THRESH, WARN_THRESH
    ax1.axhline(MAX_LINK_CAPACITY * REROUTE_THRESH / 1e6, color="orange",
                linestyle="--", linewidth=1, alpha=0.8,
                label=f"Reroute ({int(REROUTE_THRESH*100)}%)")
    ax1.axhline(MAX_LINK_CAPACITY * WARN_THRESH / 1e6, color="gold",
                linestyle=":", linewidth=1, alpha=0.8,
                label=f"Aviso ({int(WARN_THRESH*100)}%)")

    ax1.set_ylabel("Carga de enlace (Mbps)", fontsize=FONTSIZE_LABEL)
    ax1.set_title("Overhead e atuação do SDN Orchestrator", fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=FONTSIZE_LEGEND - 1)
    ax1.grid(True, alpha=0.3)

    # Subplot inferior: flows de reroute ativos e duração do ciclo
    ax2b = ax2.twinx()
    ax2.bar(sdn["elapsed_sec"], sdn["n_reroute_flows"],
            width=sdn["elapsed_sec"].diff().fillna(5).clip(upper=10),
            color=C_SDN, alpha=0.7, label="Flows reroute ativos")
    ax2b.plot(sdn["elapsed_sec"], sdn["cycle_duration_sec"],
              color="gray", linewidth=1, linestyle="--", alpha=0.7,
              label="Duração do ciclo (s)")

    ax2.set_xlabel("Tempo (s)",               fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel("Flows reroute (#)",        fontsize=FONTSIZE_LABEL)
    ax2b.set_ylabel("Duração do ciclo (s)",   fontsize=FONTSIZE_LABEL)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=FONTSIZE_LEGEND - 1)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save(fig, os.path.join(output_dir, "fig5_overhead_sdn.png"))


# ── Relatório textual ─────────────────────────────────────────────────────────

def print_summary(com: pd.DataFrame, sem: pd.DataFrame,
                  sdn: pd.DataFrame | None) -> None:
    print("\n" + "=" * 60)
    print("  RESUMO DOS RESULTADOS")
    print("=" * 60)

    t_sem = tempo_para_atingir(sem, 0.95)
    t_com = tempo_para_atingir(com, 0.95)

    print(f"\n  Rounds totais:   sem SDN={len(sem)}  |  com SDN={len(com)}")
    print(f"  Tempo total:     sem SDN={sem['elapsed_sec'].max():.1f}s"
          f"  |  com SDN={com['elapsed_sec'].max():.1f}s")
    print(f"\n  Accuracy máx:   sem SDN={sem['accuracy'].max():.4f}"
          f"  |  com SDN={com['accuracy'].max():.4f}")

    if not pd.isna(t_sem) and not pd.isna(t_com):
        reducao = (t_sem - t_com) / t_sem * 100
        print(f"\n  Tempo p/ 95% accuracy:")
        print(f"    Sem SDN: {t_sem:.1f}s")
        print(f"    Com SDN: {t_com:.1f}s")
        print(f"    Redução: {reducao:.1f}%  ← número para o abstract")

    if "f1" in com.columns and "f1" in sem.columns:
        print(f"\n  F1 máx:          sem SDN={sem['f1'].max():.4f}"
              f"  |  com SDN={com['f1'].max():.4f}")

    if "auc" in com.columns and "auc" in sem.columns:
        print(f"  AUC final:       sem SDN={sem['auc'].iloc[-1]:.4f}"
              f"  |  com SDN={com['auc'].iloc[-1]:.4f}")

    if sdn is not None:
        n_reroute_cycles = (sdn["n_reroute_flows"] > 0).sum()
        print(f"\n  Ciclos SDN com reroute ativo: {n_reroute_cycles}/{len(sdn)}")
        print(f"  Carga máx. de enlace observada: "
              f"{sdn['max_link_load_bps'].max() / 1e6:.2f} Mbps")
        print(f"  Duração média do ciclo SDN: "
              f"{sdn['cycle_duration_sec'].mean():.2f}s")

    print("=" * 60 + "\n")


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos comparativos FL com SDN vs sem SDN"
    )
    parser.add_argument("--com",         required=True,
                        help="CSV do experimento COM SDN (accuracy, f1, auc, round, elapsed_sec)")
    parser.add_argument("--sem",         required=True,
                        help="CSV do experimento SEM SDN")
    parser.add_argument("--sdn-metrics", default=None,
                        help="CSV do MetricsCollector (sdn_metrics.csv) — opcional")
    parser.add_argument("--output-dir",  default="plots",
                        help="Pasta de saída dos PNGs (padrão: ./plots/)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nCarregando dados...")
    com = load_csv(args.com, "com_sdn")
    sem = load_csv(args.sem, "sem_sdn")
    print(f"  com SDN: {len(com)} rounds | sem SDN: {len(sem)} rounds")

    sdn = None
    if args.sdn_metrics:
        if os.path.exists(args.sdn_metrics):
            sdn = pd.read_csv(args.sdn_metrics)
            print(f"  SDN metrics: {len(sdn)} ciclos")
        else:
            print(f"  [aviso] {args.sdn_metrics} não encontrado — overlay SDN desativado")

    print(f"\nGerando gráficos em {os.path.abspath(args.output_dir)}/")
    fig_accuracy_tempo(com, sem, sdn, args.output_dir)
    fig_f1_tempo(com, sem, sdn, args.output_dir)
    fig_duracao_round(com, sem, sdn, args.output_dir)
    fig_auc_round(com, sem, args.output_dir)

    if sdn is not None:
        fig_overhead_sdn(sdn, args.output_dir)
    else:
        print("  [info] fig5 requer --sdn-metrics — pulada")

    print_summary(com, sem, sdn)


if __name__ == "__main__":
    main()
