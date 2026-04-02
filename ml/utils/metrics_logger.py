"""
Registro persistente de métricas de treinamento.

Salva os resultados de cada execução do pipeline em um arquivo JSON
(metrics_history.json) e exporta para CSV quando necessário.

Cada entrada inclui: timestamp, identificador do experimento, hiperparâmetros,
métricas de avaliação e metadados do dataset. Isso permite comparar múltiplas
runs, versões do modelo e experimentos ao longo do tempo.

Uso:
    from ml.utils.metrics_logger import MetricsLogger
    from ml.evaluation.evaluator import EvaluationResult

    logger = MetricsLogger()
    logger.log(
        result=result_baseline,
        run_id="baseline_v1",
        params={"hidden_layer_sizes": (128, 64), "alpha": 0.0001},
        dataset_info={"n_train": 46172, "n_test": 57110, "n_duplicates_removed": 87084},
    )
    logger.to_csv()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import OUTPUTS_DIR
from ml.evaluation.evaluator import EvaluationResult

METRICS_FILE = OUTPUTS_DIR / "metrics_history.json"


class MetricsLogger:
    """
    Registra e persiste métricas de múltiplos experimentos em JSON + CSV.
    """

    def __init__(self, path: Path | str = METRICS_FILE) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = self._load()

    # ── API pública ────────────────────────────────────────────────────────────

    def log(
        self,
        result: EvaluationResult,
        run_id: str | None = None,
        params: dict | None = None,
        dataset_info: dict | None = None,
        notes: str = "",
    ) -> dict:
        """
        Registra uma avaliação e persiste no arquivo JSON.

        Parameters
        ----------
        result       : EvaluationResult do ModelEvaluator
        run_id       : identificador único da run (ex: "baseline_v1", "tuned_run3")
                       se None, gera automaticamente com timestamp
        params       : hiperparâmetros do modelo (dict)
        dataset_info : informações do dataset (n_train, n_test, etc.)
        notes        : observações livres sobre a run

        Returns
        -------
        dict com o registro salvo.
        """
        ts = datetime.now()
        entry: dict[str, Any] = {
            "run_id":    run_id or f"run_{ts.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": ts.isoformat(),
            "label":     result.label,

            # Métricas de avaliação
            "metrics": {
                "accuracy":   round(result.accuracy,  6),
                "precision":  round(result.precision, 6),
                "recall":     round(result.recall,    6),
                "f1":         round(result.f1,        6),
                "mcc":        round(result.mcc,       6),
                "gm":         round(result.gm,        6),
                "roc_auc":    round(result.roc_auc,   6),
            },

            # Matriz de confusão
            "confusion_matrix": {
                "tp": result.tp,
                "tn": result.tn,
                "fp": result.fp,
                "fn": result.fn,
            },

            # Métricas percentuais (para leitura humana)
            "metrics_pct": {
                "accuracy_pct":  round(result.accuracy  * 100, 4),
                "precision_pct": round(result.precision * 100, 4),
                "recall_pct":    round(result.recall    * 100, 4),
                "f1_pct":        round(result.f1        * 100, 4),
            },

            # Hiperparâmetros e contexto
            "params":       params or {},
            "dataset_info": dataset_info or {},
            "notes":        notes,
        }

        self._history.append(entry)
        self._save()

        print(f"[MetricsLogger] Run '{entry['run_id']}' registrada → {self._path}")
        return entry

    def to_csv(self, path: Path | str | None = None) -> Path:
        """
        Exporta o histórico completo para CSV.

        Cada linha = uma run. Métricas são colunas achatadas.

        Returns
        -------
        Path do CSV gerado.
        """
        if not self._history:
            print("[MetricsLogger] Nenhuma run registrada ainda.")
            return Path()

        rows = []
        for entry in self._history:
            row: dict[str, Any] = {
                "run_id":    entry["run_id"],
                "timestamp": entry["timestamp"],
                "label":     entry["label"],
                "notes":     entry.get("notes", ""),
            }
            row.update(entry.get("metrics", {}))
            row.update({f"cm_{k}": v for k, v in entry.get("confusion_matrix", {}).items()})
            row.update({f"param_{k}": v for k, v in entry.get("params", {}).items()})
            row.update({f"data_{k}": v for k, v in entry.get("dataset_info", {}).items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = Path(path) if path else self._path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[MetricsLogger] CSV exportado → {csv_path}  ({len(df)} runs)")
        return csv_path

    def summary(self) -> pd.DataFrame:
        """
        Retorna um DataFrame resumido com as métricas principais de cada run.
        """
        if not self._history:
            return pd.DataFrame()

        rows = []
        for e in self._history:
            m = e.get("metrics", {})
            cm = e.get("confusion_matrix", {})
            rows.append({
                "run_id":    e["run_id"],
                "label":     e["label"],
                "timestamp": e["timestamp"][:19],
                "accuracy":  f"{m.get('accuracy',0)*100:.4f}%",
                "precision": f"{m.get('precision',0)*100:.4f}%",
                "recall":    f"{m.get('recall',0)*100:.4f}%",
                "f1":        f"{m.get('f1',0)*100:.4f}%",
                "mcc":       f"{m.get('mcc',0):.4f}",
                "roc_auc":   f"{m.get('roc_auc',0):.4f}",
                "tp":  cm.get("tp", ""),
                "tn":  cm.get("tn", ""),
                "fp":  cm.get("fp", ""),
                "fn":  cm.get("fn", ""),
            })

        df = pd.DataFrame(rows)
        return df

    def print_summary(self) -> None:
        """Exibe tabela resumida no terminal."""
        df = self.summary()
        if df.empty:
            print("[MetricsLogger] Nenhuma run registrada.")
            return
        print("\n" + "=" * 100)
        print(f"  Histórico de Experimentos — {len(df)} run(s)")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

    def __len__(self) -> int:
        return len(self._history)

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                print(f"[MetricsLogger] {len(data)} run(s) carregada(s) de {self._path}")
                return data
            except (json.JSONDecodeError, KeyError):
                print(f"[MetricsLogger] Arquivo corrompido — iniciando histórico novo.")
        return []

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._history, f, indent=2, ensure_ascii=False)
