"""
Pipeline principal de detecção de DDoS em SDN com MLP.

Orquestra todas as etapas do workflow conforme as boas práticas do curso
(Thaís Gaudencio, UFPB/LUMO) e o plano de implementação baseado em
Mehmood et al. (PLoS ONE, 2025).

Ordem rigorosa do pipeline (nenhuma etapa pode ser invertida):
  1.  Configurações e reprodutibilidade
  2.  Carregamento do dataset InSDN8
  3.  EDA exploratória (sem modificar os dados)
  4.  ⚠️  SPLIT treino/teste ANTES de qualquer transformação  ⚠️
  5.  Limpeza (duplicatas, Inf → NaN, imputação) — somente no treino
  6.  Seleção de features (VarianceThreshold + SHAP) — somente no treino
  7.  Escalonamento (StandardScaler.fit no treino) — transform no teste
  8.  Balanceamento SMOTE — somente no treino
  9.  Treinamento MLP baseline + validação cruzada
  10. Avaliação baseline no test_set
  11. Hyperparameter Tuning (RandomizedSearchCV no treino)
  12. Avaliação final do modelo otimizado no test_set
  13. Comparação baseline vs. otimizado vs. artigo
  14. Salvamento de todos os artefatos

Avaliação do dataset insdn8_ddos_binary_0n1d.csv:
  ✓ Compatível com o plano — subconjunto do InSDN com 8 features pré-selecionadas
  ✓ Label já binarizada (0=Benigno, 1=Ataque DDoS)
  ✓ ~190.366 instâncias — volume adequado para MLP
  ⚠ Fortemente desbalanceado (maioria classe 1) — SMOTE obrigatório
  ⚠ Apenas 8 features (vs. 84 do InSDN original) — seleção SHAP será feita,
    mas sem redução de dimensionalidade significativa esperada
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
    OUTPUTS_DIR,
)
from ml.data.loader import InSDNLoader
from ml.preprocessing.cleaner import DataCleaner
from ml.preprocessing.scaler import FeatureScaler
from ml.preprocessing.balancer import ClassBalancer
from ml.features.selector import FeatureSelector
from ml.training.trainer import ModelTrainer
from ml.training.tuner import HyperparameterTuner
from ml.evaluation.evaluator import ModelEvaluator
from ml.persistence.model_io import ModelIO, PipelineArtifacts
from ml.utils.metrics_logger import MetricsLogger


# ── Reprodutibilidade global ───────────────────────────────────────────────────
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 7000)
pd.set_option("display.max_rows", 90000)


def run_pipeline(
    run_tuning: bool = True,
    run_eda: bool = True,
    verbose: bool = True,
    run_id: str | None = None,
) -> None:
    """
    Executa o pipeline completo de treinamento e avaliação.

    Parameters
    ----------
    run_tuning : bool — se True, executa o hyperparameter tuning (mais lento)
    run_eda    : bool — se True, exibe a análise exploratória textual
    verbose    : bool — nível de verbosidade geral
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Pipeline: Detecção de DDoS em SDN com MLP")
    print("  Dataset : InSDN8 (insdn8_ddos_binary_0n1d.csv)")
    print("  Baseline: Mehmood et al. (PLoS ONE, 2025)")
    print("=" * 60)

    # ── Etapa 1: Configurações ─────────────────────────────────────────────────
    print(f"\n[1/14] Configurações")
    print(f"  RANDOM_STATE : {RANDOM_STATE}")
    print(f"  TEST_SIZE    : {TEST_SIZE} (70/30)")
    print(f"  TARGET_COL   : {TARGET_COL}")

    # ── Etapa 2: Carregamento ──────────────────────────────────────────────────
    print(f"\n[2/14] Carregando dataset...")
    loader = InSDNLoader()
    X, y   = loader.load()

    # ── Etapa 3: EDA ──────────────────────────────────────────────────────────
    if run_eda:
        print(f"\n[3/14] EDA (sem modificar os dados)...")
        loader.describe()
    else:
        print(f"\n[3/14] EDA ignorada (run_eda=False)")

    # ── Etapa 4: SPLIT (ANTES de qualquer transformação) ──────────────────────
    print(f"\n[4/14] Split estratificado {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,       # OBRIGATÓRIO: preserva proporção de classes
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Distribuição treino:\n  {y_train.value_counts(normalize=True).mul(100).round(2).to_dict()}")
    print(f"  Distribuição teste :\n  {y_test.value_counts(normalize=True).mul(100).round(2).to_dict()}")

    # Resetar índices para evitar problemas de alinhamento nas etapas seguintes
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # ── Etapa 5: Limpeza (somente no treino) ──────────────────────────────────
    print(f"\n[5/14] Limpeza e preparação (somente no treino)...")
    cleaner          = DataCleaner()
    X_train, y_train = cleaner.fit_transform(X_train, y_train)
    X_test           = cleaner.transform(X_test)

    # ── Etapa 6: Seleção de features (somente no treino) ─────────────────────
    print(f"\n[6/14] Seleção de features (VarianceThreshold + SHAP)...")
    selector    = FeatureSelector()
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    print(f"\n  Features selecionadas: {selector.selected_features}")
    print(f"  Shape treino selecionado: {X_train_sel.shape}")
    print(f"  Shape teste  selecionado: {X_test_sel.shape}")

    # ── Etapa 7: Escalonamento (fit no treino) ────────────────────────────────
    print(f"\n[7/14] Escalonamento (StandardScaler.fit no treino)...")
    scaler         = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled  = scaler.transform(X_test_sel)

    # ── Etapa 8: Balanceamento SMOTE (somente no treino) ─────────────────────
    print(f"\n[8/14] Balanceamento SMOTE (somente no treino)...")
    balancer             = ClassBalancer()
    X_train_bal, y_train_bal = balancer.fit_resample(X_train_scaled, y_train)

    # ── Etapa 9: Treinamento MLP baseline + CV ────────────────────────────────
    print(f"\n[9/14] Treinamento MLP baseline...")
    trainer       = ModelTrainer(save_plots=True)
    model_baseline = trainer.train(X_train_bal, y_train_bal)

    print(f"\n[9b/14] Validação cruzada no treino...")
    cv_results = trainer.cross_validate(X_train_bal, y_train_bal)

    # ── Etapa 10: Avaliação baseline no test_set ──────────────────────────────
    print(f"\n[10/14] Avaliação BASELINE no test_set...")
    evaluator      = ModelEvaluator(save_plots=True)
    result_baseline = evaluator.evaluate(
        model_baseline,
        X_test_scaled,
        y_test.values,
        label="MLP Baseline",
    )

    # ── Etapa 11: Hyperparameter Tuning ───────────────────────────────────────
    tuner: HyperparameterTuner | None = None
    if run_tuning:
        print(f"\n[11/14] Hyperparameter Tuning (RandomizedSearchCV no treino)...")
        tuner      = HyperparameterTuner()
        model_best = tuner.fit(X_train_bal, y_train_bal)
    else:
        print(f"\n[11/14] Tuning ignorado (run_tuning=False) — usando baseline como modelo final.")
        model_best = model_baseline

    # ── Etapa 12: Avaliação final no test_set ────────────────────────────────
    print(f"\n[12/14] Avaliação FINAL (modelo otimizado) no test_set...")
    result_optimized = evaluator.evaluate(
        model_best,
        X_test_scaled,
        y_test.values,
        label="MLP Otimizado",
    )

    # ── Etapa 13: Comparação ─────────────────────────────────────────────────
    print(f"\n[13/14] Comparação: Baseline vs. Otimizado vs. Artigo")
    evaluator.compare(result_baseline, result_optimized)

    # ── Etapa 14: Salvamento (artefatos + métricas) ───────────────────────────
    print(f"\n[14/14] Salvando artefatos e métricas...")
    artifacts = PipelineArtifacts(
        model=model_best,
        imputer=cleaner.imputer,
        variance_filter=selector.variance_filter,
        scaler=scaler.scaler,
        selected_features=selector.selected_features,
    )
    ModelIO().save(artifacts)

    # Registrar métricas no histórico persistente
    from datetime import datetime
    logger    = MetricsLogger()
    n_dupes   = 133256 - len(X_train)   # duplicatas removidas no treino
    ds_info   = {
        "n_raw":               190366,
        "n_train_after_clean": len(X_train),
        "n_test":              len(X_test),
        "n_duplicates_removed": n_dupes,
        "n_features":          len(selector.selected_features),
    }
    baseline_params = {
        "hidden_layer_sizes": str((128, 64)),
        "alpha": 0.0001,
        "solver": "adam",
        "activation": "relu",
        "tuned": False,
    }
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M")
    logger.log(result_baseline, run_id=run_id or f"baseline_{ts_suffix}",
               params=baseline_params, dataset_info=ds_info)

    if run_tuning and tuner is not None:
        tuned_params = {**baseline_params, **tuner.best_params_, "tuned": True}
        logger.log(result_optimized, run_id=f"tuned_{ts_suffix}",
                   params=tuned_params, dataset_info=ds_info)

    logger.to_csv()
    logger.print_summary()

    print("\n" + "=" * 60)
    print("  Pipeline concluído com sucesso!")
    print(f"  Modelos  : models/")
    print(f"  Outputs  : outputs/")
    print(f"  Métricas : outputs/metrics_history.json")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de detecção de DDoS em SDN com MLP (InSDN8)"
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Pular o hyperparameter tuning (mais rápido, apenas baseline)",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Pular a análise exploratória textual",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador da run para o histórico de métricas (ex: 'experimento_v2')",
    )
    args = parser.parse_args()

    run_pipeline(
        run_tuning=not args.no_tuning,
        run_eda=not args.no_eda,
        run_id=args.run_id,
    )
