"""
Registro persistente das metricas de treinamento e avaliacao.
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
    """Registra e persiste metricas em JSON + CSV."""

    def __init__(self, path: Path | str = METRICS_FILE) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = self._load()

    def log(
        self,
        result: EvaluationResult,
        run_id: str | None = None,
        params: dict | None = None,
        dataset_info: dict | None = None,
        notes: str = "",
    ) -> dict:
        ts = datetime.now()
        # Monta a entrada completa do experimento — tudo que pode ser útil para comparar runs
        entry: dict[str, Any] = {
            "run_id": run_id or f"run_{ts.strftime('%Y%m%d_%H%M%S')}",  # ID gerado se não passado
            "timestamp": ts.isoformat(),
            "label": result.label,
            "class_names": result.class_names,
            "metrics": {
                "accuracy": round(result.accuracy, 6),
                "balanced_accuracy": round(result.balanced_accuracy, 6),
                "precision_macro": round(result.precision_macro, 6),
                "recall_macro": round(result.recall_macro, 6),
                "f1_macro": round(result.f1_macro, 6),
                "f1_weighted": round(result.f1_weighted, 6),
                "mcc": round(result.mcc, 6),
                "gm": round(result.gm, 6),
                "roc_auc_ovr_macro": round(result.roc_auc_ovr_macro, 6),
            },
            "confusion_matrix": result.confusion_matrix,  # matriz completa para visualização futura
            "params": params or {},                        # hiperparâmetros usados nesta run
            "dataset_info": dataset_info or {},            # metadados do dataset (tamanho, distribuição etc.)
            "notes": notes,
        }

        self._history.append(entry)
        self._save()  # persiste imediatamente — não perde nada se o processo cair depois
        print(f"[MetricsLogger] Run '{entry['run_id']}' registrada -> {self._path}")
        return entry

    def to_csv(self, path: Path | str | None = None) -> Path:
        if not self._history:
            print("[MetricsLogger] Nenhuma run registrada ainda.")
            return Path()

        rows = []
        for entry in self._history:
            # O CSV fica "achatado" para abrir fácil no pandas, Excel ou LibreOffice.
            row: dict[str, Any] = {
                "run_id": entry["run_id"],
                "timestamp": entry["timestamp"],
                "label": entry["label"],
                "notes": entry.get("notes", ""),
            }
            row.update(entry.get("metrics", {}))
            row.update({f"param_{k}": v for k, v in entry.get("params", {}).items()})
            row.update({f"data_{k}": v for k, v in entry.get("dataset_info", {}).items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = Path(path) if path else self._path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[MetricsLogger] CSV exportado -> {csv_path} ({len(df)} runs)")
        return csv_path

    def summary(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()

        rows = []
        for entry in self._history:
            # Aqui formatamos para leitura humana; por isso já sai com porcentagem e arredondamento.
            metrics = entry.get("metrics", {})
            rows.append(
                {
                    "run_id": entry["run_id"],
                    "label": entry["label"],
                    "timestamp": entry["timestamp"][:19],
                    "accuracy": f"{metrics.get('accuracy', 0) * 100:.2f}%",
                    "bal_acc": f"{metrics.get('balanced_accuracy', 0) * 100:.2f}%",
                    "f1_macro": f"{metrics.get('f1_macro', 0) * 100:.2f}%",
                    "f1_weighted": f"{metrics.get('f1_weighted', 0) * 100:.2f}%",
                    "mcc": f"{metrics.get('mcc', 0):.4f}",
                    "roc_auc": f"{metrics.get('roc_auc_ovr_macro', 0):.4f}",
                }
            )
        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        df = self.summary()
        if df.empty:
            print("[MetricsLogger] Nenhuma run registrada.")
            return
        print("\n" + "=" * 110)
        print(f"  Historico de Experimentos — {len(df)} run(s)")
        print("=" * 110)
        print(df.to_string(index=False))
        print("=" * 110)

    def __len__(self) -> int:
        return len(self._history)

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                with open(self._path) as file:
                    data = json.load(file)
                print(f"[MetricsLogger] {len(data)} run(s) carregada(s) de {self._path}")
                return data
            except (json.JSONDecodeError, KeyError):
                print("[MetricsLogger] Arquivo de historico corrompido; iniciando novo.")
        return []

    def _save(self) -> None:
        with open(self._path, "w") as file:
            json.dump(self._history, file, indent=2, ensure_ascii=False)
