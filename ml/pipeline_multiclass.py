"""
Pipeline multiclasse: Benigno / Ataque Externo / Zumbi Interno.

Extende o pipeline binário com três novos módulos:
  1. LabelEngineer   — converte labels binárias → triclasse
  2. TopologyFeatureEngineer — adiciona features HCF (TTL, hop_count, is_internal)
  3. MulticlassEvaluator — métricas macro + por classe + CM 3×3

Ordem rigorosa (idêntica às boas práticas do curso — nenhuma etapa invertida):
  1.  Configurações
  2.  Carregar dataset InSDN8
  3.  EDA (observar, sem modificar)
  4.  ⚠️  Engenharia de labels → 3 classes  ⚠️
  5.  ⚠️  SPLIT estratificado 70/30           ⚠️
  6.  Engenharia de features HCF (fit_transform no treino)
  7.  Limpeza (duplicatas, Inf, imputação) — só no treino
  8.  VarianceThreshold + SHAP              — só no treino
  9.  StandardScaler                        — fit só no treino
  10. SMOTE multiclasse                     — só no treino
  11. Treino MLP baseline + CV
  12. Avaliação baseline no test_set
  13. Hyperparameter Tuning                 — só no treino
  14. Avaliação final no test_set
  15. Comparação + salvamento de artefatos

Nota sobre SMOTE multiclasse:
  SMOTE do imbalanced-learn suporta 3+ classes nativamente —
  sobreamostra TODAS as classes minoritárias em relação à majoritária.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    RANDOM_STATE, TEST_SIZE, TARGET_COL,
    OUTPUTS_DIR, MODELS_DIR_MULTI, CLASS_NAMES,
    CV_SCORING_MULTI,
)
from ml.data.loader       import InSDNLoader
from ml.data.label_engineer import LabelEngineer
from ml.features.topology_features import TopologyFeatureEngineer
from ml.preprocessing.cleaner  import DataCleaner
from ml.preprocessing.scaler   import FeatureScaler
from ml.preprocessing.balancer import ClassBalancer
from ml.features.selector      import FeatureSelector
from ml.training.trainer       import ModelTrainer
from ml.training.tuner         import HyperparameterTuner
from ml.evaluation.evaluator   import MulticlassEvaluator
from ml.persistence.model_io   import ModelIO, PipelineArtifacts
from ml.utils.metrics_logger   import MetricsLogger

np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 7000)


def run_multiclass_pipeline(
    run_tuning: bool = True,
    run_eda:    bool = True,
    run_id:     str | None = None,
) -> None:
    """
    Executa o pipeline completo de classificação triclasse.

    Parameters
    ----------
    run_tuning : se True, executa RandomizedSearchCV (mais lento)
    run_eda    : se True, exibe análise exploratória textual
    run_id     : identificador da run para o logger de métricas
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR_MULTI.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Pipeline Multiclasse — DDoS SDN")
    print("  Classes: Benigno | Ataque Externo | Zumbi Interno")
    print("  Base   : InSDN8 + HCF (TTL/hop_count/is_internal)")
    print("=" * 65)

    # ── [1] Configurações ──────────────────────────────────────────────────────
    print(f"\n[1/15] Configurações")
    print(f"  Classes     : {CLASS_NAMES}")
    print(f"  RANDOM_STATE: {RANDOM_STATE}  |  TEST_SIZE: {TEST_SIZE}")
    print(f"  Scoring CV  : {CV_SCORING_MULTI}")

    # ── [2] Carregamento ───────────────────────────────────────────────────────
    print(f"\n[2/15] Carregando dataset InSDN8...")
    loader = InSDNLoader()
    X, y   = loader.load()

    # ── [3] EDA ────────────────────────────────────────────────────────────────
    if run_eda:
        print(f"\n[3/15] EDA inicial (dataset binário original)...")
        loader.describe()
    else:
        print(f"\n[3/15] EDA ignorada (run_eda=False)")

    # ── [4] Engenharia de labels → 3 classes ──────────────────────────────────
    print(f"\n[4/15] Engenharia de labels: binário → triclasse...")
    label_eng = LabelEngineer()
    y3 = label_eng.transform(X, y)
    label_eng.report(y3)

    # ── [5] SPLIT estratificado (ANTES de qualquer transformação) ─────────────
    print(f"\n[5/15] Split estratificado 70/30 (stratify=y3)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y3,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y3,      # OBRIGATÓRIO: preserva proporção das 3 classes
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    for cls, name in enumerate(CLASS_NAMES):
        tr = (y_train == cls).sum(); te = (y_test == cls).sum()
        print(f"  {name:<20}: treino={tr:>6,} ({tr/len(y_train)*100:.1f}%)"
              f"  |  teste={te:>6,} ({te/len(y_test)*100:.1f}%)")

    # ── [6] Features HCF (fit_transform no treino, transform no teste) ────────
    print(f"\n[6/15] Engenharia de features HCF (TTL, hop_count, is_internal)...")
    hcf_eng     = TopologyFeatureEngineer(random_state=RANDOM_STATE)
    X_train_hcf = hcf_eng.fit_transform(X_train, y_train)
    X_test_hcf  = hcf_eng.transform(X_test)
    print(f"  Novas features: {hcf_eng.new_feature_names}")
    print(f"  Shape treino com HCF: {X_train_hcf.shape}")

    # ── [7] Limpeza (só no treino) ────────────────────────────────────────────
    print(f"\n[7/15] Limpeza (duplicatas, Inf → NaN, imputação)...")
    cleaner              = DataCleaner()
    X_train_cln, y_train = cleaner.fit_transform(X_train_hcf, y_train)
    X_test_cln           = cleaner.transform(X_test_hcf)

    # ── [8] VarianceThreshold + SHAP (só no treino) ───────────────────────────
    print(f"\n[8/15] Seleção de features (VarianceThreshold + SHAP)...")
    selector    = FeatureSelector(save_plots=True)
    X_train_sel = selector.fit_transform(X_train_cln, y_train)
    X_test_sel  = selector.transform(X_test_cln)
    print(f"  Features selecionadas ({len(selector.selected_features)}): "
          f"{selector.selected_features}")

    # ── [9] Escalonamento (fit só no treino) ──────────────────────────────────
    print(f"\n[9/15] StandardScaler (fit no treino)...")
    scaler         = FeatureScaler()
    X_train_sc     = scaler.fit_transform(X_train_sel)
    X_test_sc      = scaler.transform(X_test_sel)

    # ── [10] SMOTE multiclasse (só no treino) ─────────────────────────────────
    print(f"\n[10/15] SMOTE multiclasse (só no treino)...")
    balancer             = ClassBalancer()
    X_train_bal, y_bal   = balancer.fit_resample(X_train_sc, y_train)

    # ── [11] Treino MLP baseline + CV ─────────────────────────────────────────
    print(f"\n[11/15] Treino MLP baseline (128→64, ReLU, ADAM)...")
    trainer     = ModelTrainer(cv_scoring=CV_SCORING_MULTI, save_plots=True)
    mlp_base    = trainer.train(X_train_bal, y_bal)

    print(f"\n[11b/15] Validação cruzada ({CV_SCORING_MULTI}) no treino...")
    trainer.cross_validate(X_train_bal, y_bal)

    # ── [12] Avaliação baseline no test_set ───────────────────────────────────
    print(f"\n[12/15] Avaliação BASELINE no test_set (1ª e única vez)...")
    ev         = MulticlassEvaluator(save_plots=True)
    res_base   = ev.evaluate(mlp_base, X_test_sc, y_test.values, label="MLP Baseline")

    # ── [13] Hyperparameter Tuning ────────────────────────────────────────────
    tuner: HyperparameterTuner | None = None
    if run_tuning:
        print(f"\n[13/15] Hyperparameter Tuning (RandomizedSearchCV no treino)...")
        tuner   = HyperparameterTuner(scoring=CV_SCORING_MULTI)
        mlp_best = tuner.fit(X_train_bal, y_bal)
    else:
        print(f"\n[13/15] Tuning ignorado — usando baseline como modelo final.")
        mlp_best = mlp_base

    # ── [14] Avaliação final ───────────────────────────────────────────────────
    print(f"\n[14/15] Avaliação FINAL (modelo otimizado) no test_set...")
    res_tuned = ev.evaluate(mlp_best, X_test_sc, y_test.values, label="MLP Otimizado")

    ev.compare(res_base, res_tuned)

    # ── [15] Salvamento ────────────────────────────────────────────────────────
    print(f"\n[15/15] Salvando artefatos em {MODELS_DIR_MULTI}/...")
    artifacts = PipelineArtifacts(
        model=mlp_best,
        imputer=cleaner.imputer,
        variance_filter=selector.variance_filter,
        scaler=scaler.scaler,
        selected_features=selector.selected_features,
    )
    ModelIO(models_dir=MODELS_DIR_MULTI).save(artifacts)

    # Salvar HCF engineer separadamente (necessário na inferência)
    import joblib
    with open(MODELS_DIR_MULTI / "hcf_engineer.joblib", "wb") as f:
        joblib.dump(hcf_eng, f)
    print(f"  ✓ hcf_engineer.joblib")

    # Registrar métricas
    _log_metrics(res_base, res_tuned, tuner, cleaner, selector, y_train, run_id)

    print("\n" + "=" * 65)
    print("  Pipeline multiclasse concluído!")
    print(f"  Modelos  : {MODELS_DIR_MULTI}/")
    print(f"  Outputs  : {OUTPUTS_DIR}/")
    print("=" * 65)


# ── helpers ────────────────────────────────────────────────────────────────────

def _log_metrics(
    res_base, res_tuned, tuner, cleaner, selector, y_train, run_id
) -> None:
    """Registra métricas multiclasse no histórico."""
    from datetime import datetime
    from ml.evaluation.evaluator import MulticlassResult

    logger    = MetricsLogger(OUTPUTS_DIR / "metrics_history_multi.json")
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M")
    ds_info   = {
        "n_train_clean": len(y_train),
        "n_features":    len(selector.selected_features),
        "n_classes":     3,
    }

    def _result_to_eval(r: MulticlassResult):
        """Adaptador: converte MulticlassResult para EvaluationResult-like."""
        from ml.evaluation.evaluator import EvaluationResult
        return EvaluationResult(
            label=r.label,
            accuracy=r.accuracy,
            precision=r.precision_macro,
            recall=r.recall_macro,
            f1=r.f1_macro,
            mcc=r.mcc,
            gm=r.gm,
            roc_auc=0.0,   # ROC multiclasse requer OVR — não calculado aqui
            tp=0, tn=0, fp=0, fn=0,
            classification_report=r.classification_report,
        )

    base_params = {"hidden_layer_sizes": "(128, 64)", "solver": "adam",
                   "activation": "relu", "multiclass": True, "tuned": False}
    logger.log(_result_to_eval(res_base), run_id=run_id or f"multi_baseline_{ts_suffix}",
               params=base_params, dataset_info=ds_info, notes="Multiclasse 3-classes")

    if tuner is not None:
        tuned_params = {**base_params, **tuner.best_params_, "tuned": True}
        logger.log(_result_to_eval(res_tuned), run_id=f"multi_tuned_{ts_suffix}",
                   params=tuned_params, dataset_info=ds_info)

    logger.to_csv()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline multiclasse DDoS SDN: Benigno/Externo/Zumbi"
    )
    parser.add_argument("--no-tuning", action="store_true",
                        help="Pular hyperparameter tuning")
    parser.add_argument("--no-eda",    action="store_true",
                        help="Pular EDA textual")
    parser.add_argument("--run-id",    type=str, default=None,
                        help="ID da run para o logger (ex: 'experimento_v2')")
    args = parser.parse_args()

    run_multiclass_pipeline(
        run_tuning=not args.no_tuning,
        run_eda=not args.no_eda,
        run_id=args.run_id,
    )
