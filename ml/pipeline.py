"""
Pipeline principal de classificacao multiclasse para o InSDN.

Fluxo:
  1. configuracoes e reprodutibilidade
  2. carregamento e EDA do dataset consolidado
  3. split treino/teste antes de qualquer transformacao
  4. limpeza do treino e aplicacao consistente ao teste
  5. selecao de features por variancia + SHAP
  6. escalonamento
  7. balanceamento SMOTE apenas no treino
  8. treinamento do baseline e validacao cruzada
  9. avaliacao em treino/teste para diagnostico de overfitting
  10. tuning opcional
  11. salvamento de artefatos, metricas e graficos auxiliares
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    DATASET_NAME,
    OUTPUTS_DIR,
    OUTPUTS_RUNS_DIR,
    RANDOM_STATE,
    TARGET_DECODING,
    TARGET_NAMES,
    TEST_SIZE,
)
from ml.data.loader import InSDNLoader
from ml.evaluation.evaluator import ModelEvaluator
from ml.features.selector import FeatureSelector
from ml.persistence.model_io import ModelIO, PipelineArtifacts
from ml.preprocessing.balancer import ClassBalancer
from ml.preprocessing.cleaner import DataCleaner
from ml.preprocessing.scaler import FeatureScaler
from ml.training.trainer import ModelTrainer
from ml.training.tuner import HyperparameterTuner
from ml.utils.metrics_logger import MetricsLogger
from ml.utils.training_diagnostics import TrainingDiagnostics

warnings.filterwarnings("ignore")

np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 7000)
pd.set_option("display.max_rows", 200)


def _slugify_run_id(run_id: str) -> str:
    # Deixa o run_id seguro para virar nome de pasta sem surpresas.
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id.strip())
    slug = slug.strip("_")
    return slug or "run"


def run_pipeline(
    run_tuning: bool = True,
    run_eda: bool = True,
    verbose: bool = True,
    run_id: str | None = None,
    sample_size: int | None = None,
) -> None:
    # Cada execução ganha sua própria pasta para não misturar gráficos novos com antigos.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_run_id = run_id or f"baseline_{ts_suffix}"
    run_output_dir = OUTPUTS_RUNS_DIR / _slugify_run_id(effective_run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Pipeline: Classificacao Multiclasse de Trafego SDN com MLP")
    print(f"  Dataset : {DATASET_NAME}")
    print("  Classes : 0=Normal | 1=Flooding | 2=Intrusao")
    print("=" * 72)

    print("\n[1/11] Configuracoes")
    print(f"  RANDOM_STATE : {RANDOM_STATE}")
    print(f"  TEST_SIZE    : {TEST_SIZE} (70/30)")
    print(f"  SAMPLE_SIZE  : {sample_size if sample_size else 'dataset completo'}")
    print(f"  RUN_ID       : {effective_run_id}")
    print(f"  OUTPUT_DIR   : {run_output_dir}")

    print("\n[2/11] Carregando dataset...")
    loader = InSDNLoader()
    X, y = loader.load(sample_size=sample_size)

    if run_eda:
        print("\n[2b/11] EDA (sem modificar os dados)...")
        loader.describe(sample_size=sample_size)

    print(f"\n[3/11] Split estratificado {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(
        f"  Distribuicao treino: "
        f"{y_train.value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()}"
    )
    print(
        f"  Distribuicao teste : "
        f"{y_test.value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()}"
    )

    # Guardamos uma cópia "bruta" porque os diagnósticos limpos refazem o pipeline por dobra.
    X_train_raw = X_train.reset_index(drop=True).copy()
    y_train_raw = y_train.reset_index(drop=True).copy()
    X_test_raw = X_test.reset_index(drop=True).copy()
    y_test_raw = y_test.reset_index(drop=True).copy()

    print("\n[4/11] Limpeza e preparacao...")
    cleaner = DataCleaner()
    X_train, y_train = cleaner.fit_transform(X_train_raw, y_train_raw)
    X_test, y_test = cleaner.transform(
        X_test_raw,
        y_test_raw,
    )

    print("\n[5/11] Selecao de features (VarianceThreshold + SHAP)...")
    selector = FeatureSelector(output_dir=run_output_dir)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(f"  Shape treino selecionado: {X_train_sel.shape}")
    print(f"  Shape teste selecionado : {X_test_sel.shape}")

    print("\n[6/11] Escalonamento...")
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    print("\n[7/11] Balanceamento SMOTE no treino...")
    balancer = ClassBalancer()
    X_train_bal, y_train_bal = balancer.fit_resample(X_train_scaled, y_train)

    print("\n[8/11] Treinamento baseline + validacao cruzada...")
    trainer = ModelTrainer(save_plots=True, output_dir=run_output_dir)
    model_baseline = trainer.train(X_train_bal, y_train_bal, label="baseline")
    # A CV usa os dados crus do treino para remontar o preprocessamento em cada fold.
    cv_results = trainer.cross_validate(X_train_raw, y_train_raw)

    print("\n[9/11] Avaliacao baseline em treino/teste...")
    evaluator = ModelEvaluator(save_plots=True, output_dir=run_output_dir)
    train_result_baseline = evaluator.evaluate(
        model_baseline,
        X_train_scaled,
        y_train.values,
        label="MLP Baseline (Treino)",
        class_names=TARGET_NAMES,
    )
    test_result_baseline = evaluator.evaluate(
        model_baseline,
        X_test_scaled,
        y_test.values,
        label="MLP Baseline (Teste)",
        class_names=TARGET_NAMES,
    )

    diagnostics = TrainingDiagnostics(output_dir=run_output_dir)
    diagnostics.plot_learning_curve(
        X_train_raw,
        y_train_raw,
        label="baseline",
        estimator=model_baseline,
    )
    diagnostics.plot_generalization_gap(
        train_result_baseline,
        test_result_baseline,
        label="baseline",
    )
    diagnostics.save_gap_report(
        train_result_baseline,
        test_result_baseline,
        label="baseline",
    )

    tuner: HyperparameterTuner | None = None
    train_result_final = train_result_baseline
    test_result_final = test_result_baseline
    model_best = model_baseline

    if run_tuning:
        print("\n[10/11] Hyperparameter tuning...")
        tuner = HyperparameterTuner()
        model_best = tuner.fit(X_train_bal, y_train_bal)

        train_result_final = evaluator.evaluate(
            model_best,
            X_train_scaled,
            y_train.values,
            label="MLP Otimizado (Treino)",
            class_names=TARGET_NAMES,
        )
        test_result_final = evaluator.evaluate(
            model_best,
            X_test_scaled,
            y_test.values,
            label="MLP Otimizado (Teste)",
            class_names=TARGET_NAMES,
        )

        diagnostics.plot_learning_curve(
            X_train_raw,
            y_train_raw,
            label="otimizado",
            estimator=model_best,
        )
        diagnostics.plot_generalization_gap(
            train_result_final,
            test_result_final,
            label="otimizado",
        )
        diagnostics.save_gap_report(
            train_result_final,
            test_result_final,
            label="otimizado",
        )
        evaluator.compare(test_result_baseline, test_result_final)
    else:
        print("\n[10/11] Tuning ignorado (run_tuning=False).")

    print("\n[11/11] Salvando artefatos e metricas...")
    artifacts = PipelineArtifacts(
        model=model_best,
        imputer=cleaner.imputer,
        variance_filter=selector.variance_filter,
        scaler=scaler,
        selected_features=selector.selected_features,
    )
    ModelIO().save(artifacts)

    logger = MetricsLogger()
    # Este bloco vira o "cartão de visita" da run no histórico JSON/CSV.
    dataset_info = {
        "n_total": len(X),
        "n_train": len(X_train_scaled),
        "n_test": len(X_test_scaled),
        "n_features_before_selection": X_train.shape[1],
        "n_features_after_selection": len(selector.selected_features),
        "selected_features": selector.selected_features,
        "binary_features_passthrough": scaler.binary_columns,
        "scaled_features": scaler.scaled_columns,
        "run_output_dir": str(run_output_dir),
        "class_distribution_total": {
            TARGET_DECODING[int(cls)]: int(count)
            for cls, count in y.value_counts().sort_index().items()
        },
        "cv_results": {
            metric: {"mean": mean, "std": std}
            for metric, (mean, std) in cv_results.items()
        },
    }

    logger.log(
        test_result_baseline,
        run_id=effective_run_id,
        params={
            "hidden_layer_sizes": str(model_baseline.hidden_layer_sizes),
            "alpha": model_baseline.alpha,
            "max_iter": model_baseline.max_iter,
            "tuned": False,
        },
        dataset_info=dataset_info,
        notes=(
            "Treino vs teste — F1 macro gap: "
            f"{train_result_baseline.f1_macro - test_result_baseline.f1_macro:+.4f}"
        ),
    )

    if run_tuning and tuner is not None:
        logger.log(
            test_result_final,
            run_id=f"tuned_{effective_run_id}",
            params={**tuner.best_params_, "tuned": True},
            dataset_info=dataset_info,
            notes=(
                "Treino vs teste — F1 macro gap: "
                f"{train_result_final.f1_macro - test_result_final.f1_macro:+.4f}"
            ),
        )

    logger.to_csv()
    logger.print_summary()

    print("\n" + "=" * 72)
    print("  Pipeline concluido com sucesso!")
    print("  Modelos/artifacts : models/")
    print(f"  Graficos e relats : {Path(run_output_dir).relative_to(OUTPUTS_DIR.parent)}/")
    print("  Historico         : outputs/metrics_history.json")
    print("=" * 72)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline multiclasse para o InSDN (Normal/Flooding/Intrusao)."
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Pula o hyperparameter tuning (mais rapido).",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Pula a EDA textual.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador opcional da execucao.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Usa uma amostra estratificada do dataset consolidado para experimentos rapidos.",
    )

    args = parser.parse_args()
    run_pipeline(
        run_tuning=not args.no_tuning,
        run_eda=not args.no_eda,
        run_id=args.run_id,
        sample_size=args.sample_size,
    )
