"""
Persistência de artefatos do pipeline ML.

responsável exclusivamente por salvar e carregar todos os objetos
fitados durante o treinamento.

Artefatos salvos:
  mlp_ddos_insdn.joblib    — modelo MLP treinado
  imputer.joblib           — SimpleImputer (fit no treino)
  variance_filter.joblib   — VarianceThreshold (fit no treino)
  scaler.joblib            — FeatureScaler (fit no treino)
  selected_features.joblib — lista de features selecionadas pelo SHAP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

from ml.config import MODELS_DIR
from ml.preprocessing.scaler import FeatureScaler


@dataclass
class PipelineArtifacts:
    """
    Conjunto de artefatos que compõem o pipeline de inferência.

    Todos os campos devem estar fitados com os dados de TREINO antes
    de serem salvos e usados em produção.
    """

    model:             MLPClassifier
    imputer:           SimpleImputer
    variance_filter:   VarianceThreshold
    scaler:            FeatureScaler
    selected_features: list[str]


class ModelIO:
    """
    Salva e carrega o pipeline completo de detecção de DDoS.

    Uso (treino):
        io = ModelIO()
        io.save(artifacts)

    Uso (produção):
        io       = ModelIO()
        pipeline = io.load()
        preds    = pipeline.model.predict(scaler.transform(...))
    """

    _FILENAMES = {
        "model":             "mlp_ddos_insdn.joblib",
        "imputer":           "imputer.joblib",
        "variance_filter":   "variance_filter.joblib",
        "scaler":            "scaler.joblib",
        "selected_features": "selected_features.joblib",
    }

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self._dir = Path(models_dir)

    # ── API pública ────────────────────────────────────────────────────────────

    def save(self, artifacts: PipelineArtifacts) -> None:
        """
        Salva todos os artefatos do pipeline em models/.

        Salva cada objeto fitado separadamente com joblib — mais eficiente que pickle
        para arrays NumPy (usa compressão e mmap). Cria o diretório se não existir.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        # Salvamos tudo separadinho porque isso facilita trocar só uma peça depois, se precisar.
        pairs = [
            ("model",             artifacts.model),
            ("imputer",           artifacts.imputer),
            ("variance_filter",   artifacts.variance_filter),
            ("scaler",            artifacts.scaler),
            ("selected_features", artifacts.selected_features),
        ]

        print(f"\n[ModelIO] Salvando artefatos em {self._dir}/")
        for key, obj in pairs:
            path = self._dir / self._FILENAMES[key]
            with open(path, "wb") as f:
                joblib.dump(obj, f)
            print(f"  ✓ {self._FILENAMES[key]}")

        print(f"[ModelIO] {len(pairs)} artefatos salvos com sucesso.")
        print("\n  Estrutura salva:")
        print(f"  {self._dir}/")
        for fname in self._FILENAMES.values():
            print(f"    ├── {fname}")

    def load(self) -> PipelineArtifacts:
        """
        Carrega todos os artefatos do pipeline a partir de models/.

        Returns
        -------
        PipelineArtifacts com todos os objetos fitados prontos para inferência.

        Raises
        ------
        FileNotFoundError se qualquer artefato estiver ausente.
        """
        print(f"[ModelIO] Carregando artefatos de {self._dir}/")

        loaded = {}
        for key, fname in self._FILENAMES.items():
            path = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Artefato não encontrado: {path}\n"
                    f"Execute o pipeline de treinamento primeiro."
                )
            with open(path, "rb") as f:
                loaded[key] = joblib.load(f)
            print(f"  ✓ {fname}")

        # No fim remontamos um objeto único para a inferência trabalhar sem se preocupar com arquivos.
        return PipelineArtifacts(
            model=loaded["model"],
            imputer=loaded["imputer"],
            variance_filter=loaded["variance_filter"],
            scaler=loaded["scaler"],
            selected_features=loaded["selected_features"],
        )

    def exists(self) -> bool:
        """Verifica se todos os artefatos já foram salvos."""
        return all(
            (self._dir / fname).exists()
            for fname in self._FILENAMES.values()
        )
