"""
Testes unitários para ml/triclass/preprocessing/labeler.py

Valida:
  1. is_burst() — threshold correto para DDoS vs. BOTNET
  2. criar_label_triclasse_insdn() — mapeamento correto de labels
  3. TriclassLabeler.fit_transform() — descarte de fora do escopo
  4. Validação semântica BOTNET nunca é burst
"""

import numpy as np
import pandas as pd
import pytest

from ml.triclass.preprocessing.labeler import (
    is_burst,
    criar_label_triclasse_insdn,
    TriclassLabeler,
)
from ml.triclass.config import BURST_FLOW_DURATION_MAX, BURST_PKT_LEN_STD_MAX


class TestIsBurst:
    """Testes para a função is_burst()."""

    def _make(self, std: float, duration: float) -> pd.DataFrame:
        return pd.DataFrame({
            "Packet Length Std": [std],
            "Flow Duration":     [duration],
        })

    def test_ddos_hping3_e_burst(self):
        """DDoS com Duration=10µs e Std=0 deve ser burst."""
        df = self._make(std=0.0, duration=10.0)
        assert is_burst(df).iloc[0] is True or is_burst(df).iloc[0] == True

    def test_botnet_nao_e_burst(self):
        """BOTNET com Duration=31.000µs nunca deve ser burst."""
        df = self._make(std=0.0, duration=31_000.0)
        assert not is_burst(df).iloc[0]

    def test_std_acima_do_threshold_nao_e_burst(self):
        """Std > 1.0 → não é burst, independente da duration."""
        df = self._make(std=50.0, duration=5.0)
        assert not is_burst(df).iloc[0]

    def test_duration_exatamente_no_threshold_nao_e_burst(self):
        """Flow Duration == 500 (não estritamente menor) → não é burst."""
        df = self._make(std=0.0, duration=BURST_FLOW_DURATION_MAX)
        assert not is_burst(df).iloc[0]

    def test_duration_abaixo_do_threshold_e_burst(self):
        """Flow Duration = 499.9 < 500 com Std ≤ 1 → burst."""
        df = self._make(std=0.5, duration=BURST_FLOW_DURATION_MAX - 0.1)
        assert is_burst(df).iloc[0]

    def test_resultado_e_series_booleana(self, raw_insdn_df):
        """is_burst deve retornar pd.Series de bool."""
        result = is_burst(raw_insdn_df)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_todos_os_ddos_sao_burst(self, raw_insdn_df):
        """Todos os DDoS do fixture devem ser burst."""
        ddos = raw_insdn_df[raw_insdn_df["Label"] == "DDoS"]
        assert is_burst(ddos).all(), "Alguns DDoS não foram identificados como burst"

    def test_nenhum_botnet_e_burst(self, raw_insdn_df):
        """Nenhum BOTNET do fixture deve ser burst."""
        botnet = raw_insdn_df[raw_insdn_df["Label"] == "BOTNET"]
        assert not is_burst(botnet).any(), "BOTNET incorretamente identificado como burst"

    def test_normal_nunca_e_burst(self, raw_insdn_df):
        """Normal (tráfego legítimo) não deve ser burst."""
        normal = raw_insdn_df[raw_insdn_df["Label"] == "Normal"]
        assert not is_burst(normal).any(), "Normal incorretamente identificado como burst"


class TestCriarLabelTriclasse:
    """Testes para criar_label_triclasse_insdn()."""

    def test_normal_vira_classe_0(self, raw_insdn_df):
        """Label 'Normal' → Classe 0."""
        y = criar_label_triclasse_insdn(raw_insdn_df)
        normal_mask = raw_insdn_df["Label"] == "Normal"
        assert (y[normal_mask] == 0).all()

    def test_ddos_burst_vira_classe_1(self, raw_insdn_df):
        """DDoS com burst → Classe 1 (Externo)."""
        ddos_mask = raw_insdn_df["Label"] == "DDoS"
        burst_mask = is_burst(raw_insdn_df)
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert (y[ddos_mask & burst_mask] == 1).all()

    def test_botnet_vira_classe_2(self, raw_insdn_df):
        """BOTNET → Classe 2 (Zumbi Interno) — ground truth semântico."""
        botnet_mask = raw_insdn_df["Label"] == "BOTNET"
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert (y[botnet_mask] == 2).all()

    def test_dos_sem_burst_vira_classe_2(self, raw_insdn_df):
        """DoS sem burst → Classe 2 (proxy comportamental)."""
        dos_mask   = raw_insdn_df["Label"] == "DoS"
        nburst_mask = ~is_burst(raw_insdn_df)
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert (y[dos_mask & nburst_mask] == 2).all()

    def test_probe_vira_menos1(self, raw_insdn_df):
        """Probe → descartado (-1)."""
        probe_mask = raw_insdn_df["Label"] == "Probe"
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert (y[probe_mask] == -1).all()

    def test_sem_valores_inesperados(self, raw_insdn_df):
        """Labels devem ser apenas -1, 0, 1 ou 2."""
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert set(y.unique()).issubset({-1, 0, 1, 2})

    def test_retorna_series(self, raw_insdn_df):
        """Deve retornar pd.Series."""
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert isinstance(y, pd.Series)

    def test_indice_preservado(self, raw_insdn_df):
        """O índice do resultado deve corresponder ao do DataFrame de entrada."""
        y = criar_label_triclasse_insdn(raw_insdn_df)
        assert y.index.equals(raw_insdn_df.index)


class TestTriclassLabeler:
    """Testes para a classe TriclassLabeler."""

    def test_fit_transform_retorna_dataframe(self, raw_insdn_df):
        """fit_transform deve retornar pd.DataFrame."""
        labeler = TriclassLabeler()
        result = labeler.fit_transform(raw_insdn_df)
        assert isinstance(result, pd.DataFrame)

    def test_probe_descartado(self, raw_insdn_df):
        """Probe deve ser descartado após fit_transform."""
        labeler = TriclassLabeler()
        result = labeler.fit_transform(raw_insdn_df)
        assert "Probe" not in result["Label"].values

    def test_label_3class_presente(self, raw_insdn_df):
        """Coluna 'label_3class' deve ser adicionada."""
        labeler = TriclassLabeler()
        result = labeler.fit_transform(raw_insdn_df)
        assert "label_3class" in result.columns

    def test_sem_minus1_no_resultado(self, raw_insdn_df):
        """Nenhuma linha com label -1 deve sobrar no DataFrame retornado."""
        labeler = TriclassLabeler()
        result = labeler.fit_transform(raw_insdn_df)
        assert (result["label_3class"] == -1).sum() == 0

    def test_n_discarded_registrado(self, raw_insdn_df):
        """n_discarded_ deve ser o número de Probe + DDoS sem burst + DoS com burst."""
        labeler = TriclassLabeler()
        labeler.fit_transform(raw_insdn_df)
        assert labeler.n_discarded_ > 0

    def test_class_counts_correto(self, raw_insdn_df):
        """class_counts_ deve ter as 3 classes (0, 1, 2)."""
        labeler = TriclassLabeler()
        labeler.fit_transform(raw_insdn_df)
        assert set(labeler.class_counts_.keys()).issubset({0, 1, 2})

    def test_botnet_burst_count_zero(self, raw_insdn_df):
        """Nenhum BOTNET do fixture deve ser burst (validação crítica)."""
        labeler = TriclassLabeler()
        labeler.fit_transform(raw_insdn_df)
        assert labeler.botnet_burst_count_ == 0

    def test_transform_sem_fit_levanta_erro(self, raw_insdn_df):
        """transform() sem fit_transform() prévia deve levantar RuntimeError."""
        labeler = TriclassLabeler()
        with pytest.raises(RuntimeError):
            labeler.transform(raw_insdn_df)

    def test_transform_exclui_mesmos_descartados(self, raw_insdn_df):
        """transform() deve excluir os mesmos tipos de label que fit_transform."""
        labeler = TriclassLabeler()
        labeler.fit_transform(raw_insdn_df)
        result = labeler.transform(raw_insdn_df)
        assert (result["label_3class"] == -1).sum() == 0
        assert "Probe" not in result["Label"].values

    def test_indice_resetado(self, raw_insdn_df):
        """Índice do DataFrame retornado deve começar em 0 e ser contíguo."""
        labeler = TriclassLabeler()
        result = labeler.fit_transform(raw_insdn_df)
        assert result.index.tolist() == list(range(len(result)))
