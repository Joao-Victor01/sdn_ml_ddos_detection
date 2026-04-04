"""
Heurística de labeling triclasse para o InSDN.

Converte os labels originais do InSDN em três classes:
  0 — Benigno      : Label == 'Normal'
  1 — Externo      : Label == 'DDoS' AND is_burst() == True
  2 — Zumbi Interno: Label == 'BOTNET'
                     Label == 'DoS' AND is_burst() == False
 -1 — Descartado   : Probe, BFA, Web-Attack, U2R, DDoS sem burst,
                     DoS com burst — ambíguos ou fora do escopo

SRP: este módulo lida exclusivamente com a construção do vetor y triclasse.

Validação crítica (seção 7.2 do plano):
  BOTNET tem Flow Duration ~31.000 µs >> 500 µs → is_burst() == False (correto)
  DDoS Hping3 tem duration de 1–19 µs           → is_burst() == True  (correto)

Referência: plano_triclasse_insdn_v4.md, Seções 3, 5.1 e 5.2
"""

from __future__ import annotations

import pandas as pd

from ml.triclass.config import (
    BURST_FLOW_DURATION_MAX,
    BURST_PKT_LEN_STD_MAX,
    LABEL_BENIGN,
)

# ── Funções puras (sem estado — reutilizáveis nos testes) ─────────────────────

def is_burst(df: pd.DataFrame) -> pd.Series:
    """
    Identifica rajada sintética uniforme — padrão de ferramenta de ataque.

    Critério (seção 5.1 do plano):
      Packet Length Std <= 1.0  → pacotes praticamente idênticos
      Flow Duration < 500 µs   → fluxo extremamente curto

    Calibração empírica:
      BOTNET  mediana de Flow Duration ~31.000 µs → is_burst=False (correto)
      DDoS    mediana de Flow Duration 1–19 µs    → is_burst=True  (correto)

    Parameters
    ----------
    df : pd.DataFrame — deve conter 'Packet Length Std' e 'Flow Duration'

    Returns
    -------
    pd.Series[bool] — True onde o fluxo tem padrão de rajada sintética
    """
    return (
        (df["Packet Length Std"] <= BURST_PKT_LEN_STD_MAX) &
        (df["Flow Duration"]     <  BURST_FLOW_DURATION_MAX)
    )


def criar_label_triclasse_insdn(df: pd.DataFrame) -> pd.Series:
    """
    Converte labels originais do InSDN em rótulo triclasse.

    Parameters
    ----------
    df : pd.DataFrame — deve conter colunas 'Label', 'Packet Length Std',
                        'Flow Duration' (já renomeadas via RENAME_MAP)

    Returns
    -------
    pd.Series[int] — valores: 0, 1, 2 ou -1 (descartado)

    Notas
    -----
    - Label == 'DDoS' SEM burst → descartado (-1), não va para Classe 1.
      Razão: DDoS sem burst é ambíguo — pode ser fragmentação ou erro de
      captura. Incluir contaminaria a Classe 1 com padrão não confirmado.
    - Label == 'DoS' COM burst → descartado (-1).
      Razão: DoS com burst é comportamento atípico; proxy DoS aplica-se
      somente a fluxos com duração maior (comportamento de host legítimo
      infectado, não de script de flood).
    """
    label = df["Label"].str.strip()
    burst = is_burst(df)

    y = pd.Series(-1, index=df.index, dtype=int)

    # Classe 0 — Benigno
    y[label == LABEL_BENIGN] = 0

    # Classe 1 — Ataque Externo: DDoS confirmado por burst
    y[(label == "DDoS") & burst] = 1

    # Classe 2 — Zumbi Interno:
    #   BOTNET = ground truth real (âncora semântica)
    #   DoS sem burst = proxy comportamental
    y[label == "BOTNET"]            = 2
    y[(label == "DoS") & ~burst]    = 2

    return y


# ── Classe com interface fit/transform (compatível com o pipeline) ─────────────

class TriclassLabeler:
    """
    Aplica a heurística triclasse e valida o threshold de burst.

    Uso:
        labeler = TriclassLabeler()
        data    = labeler.fit_transform(data)
        # data agora contém coluna 'label_3class'; linhas com -1 são descartadas

    A separação fit/transform é por consistência de interface — a função de
    labeling não aprende parâmetros dos dados; é determinística.
    """

    def __init__(self) -> None:
        self._fitted = False
        self.n_discarded_: int = 0
        self.class_counts_: dict[int, int] = {}
        self.botnet_burst_count_: int = 0

    # ── API pública ────────────────────────────────────────────────────────────

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica labeling triclasse e remove amostras descartadas (-1).

        Valida que nenhum BOTNET é classificado como burst (verificação
        crítica da seção 7.2 do plano).

        Parameters
        ----------
        data : pd.DataFrame — DataFrame completo (pré-split), com colunas
               já renomeadas pelo RENAME_MAP

        Returns
        -------
        pd.DataFrame filtrado (sem linhas -1), com coluna 'label_3class' adicionada
        """
        data = data.copy()
        data["label_3class"] = criar_label_triclasse_insdn(data)

        # Validação crítica: BOTNET nunca deve ser burst
        botnet_mask = data["Label"].str.strip() == "BOTNET"
        self.botnet_burst_count_ = int(
            (botnet_mask & is_burst(data)).sum()
        )
        if self.botnet_burst_count_ > 0:
            import warnings
            warnings.warn(
                f"[TriclassLabeler] ATENÇÃO: {self.botnet_burst_count_} amostras BOTNET "
                f"foram identificadas como burst. O threshold BURST_FLOW_DURATION_MAX "
                f"({BURST_FLOW_DURATION_MAX} µs) pode precisar ser ajustado.",
                UserWarning,
                stacklevel=2,
            )

        n_before = len(data)
        self.n_discarded_ = int((data["label_3class"] == -1).sum())
        data = data[data["label_3class"] != -1].reset_index(drop=True)

        self.class_counts_ = {
            int(cls): int((data["label_3class"] == cls).sum())
            for cls in sorted(data["label_3class"].unique())
        }
        self._fitted = True

        self._print_summary(n_before)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica labeling ao conjunto de teste (mesma lógica, sem re-fit).
        """
        if not self._fitted:
            raise RuntimeError(
                "TriclassLabeler: chame fit_transform() antes de transform()."
            )
        data = data.copy()
        data["label_3class"] = criar_label_triclasse_insdn(data)
        return data[data["label_3class"] != -1].reset_index(drop=True)

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _print_summary(self, n_before: int) -> None:
        names = {0: "Benigno", 1: "Externo", 2: "Zumbi Interno"}
        total = sum(self.class_counts_.values())

        print(f"\n[TriclassLabeler] Descartados (fora do escopo): "
              f"{self.n_discarded_:,} / {n_before:,}")
        print(f"[TriclassLabeler] Dataset válido: {total:,}")
        print(f"[TriclassLabeler] Distribuição triclasse:")
        for cls, cnt in self.class_counts_.items():
            pct = 100 * cnt / total
            print(f"  Classe {cls} ({names.get(cls, '?')}): {cnt:,} ({pct:.1f}%)")
        print(f"[TriclassLabeler] Validação BOTNET: "
              f"{self.botnet_burst_count_} amostras BOTNET com burst "
              f"({'OK' if self.botnet_burst_count_ == 0 else 'ATENÇÃO'})")
