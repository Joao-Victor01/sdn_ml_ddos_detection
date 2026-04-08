"""
Pipeline principal de classificacao multiclasse para o InSDN.

Fluxo:
  1. configuracoes e reprodutibilidade
  2. carregamento e EDA do dataset consolidado
  3. split treino/teste antes de qualquer transformacao
  4. limpeza do treino e aplicacao consistente ao teste
  5. selecao de features por variancia
  6. escalonamento
  7. balanceamento SMOTE apenas no treino
  8. treinamento e validacao cruzada por modelo
  9. avaliacao em treino/teste para diagnostico de overfitting
  10. tuning opcional por modelo
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
from ml.evaluation.evaluator import EvaluationResult, ModelEvaluator
from ml.features.rf_explainer import RandomForestExplainer
from ml.features.selector import FeatureSelector
from ml.models.registry import ModelSpec, resolve_requested_models
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
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id.strip())
    slug = slug.strip("_")
    return slug or "run"


def _extract_model_params(model, spec: ModelSpec) -> dict:
    params = model.get_params(deep=False)
    tracked = {key: params.get(key) for key in spec.tracked_params if key in params}
    return tracked


def _build_dataset_info(
    *,
    X,
    X_train,
    X_test,
    selector: FeatureSelector,
    scaler: FeatureScaler,
    y,
    cv_results,
    run_output_dir: Path,
) -> dict:
    return {
        "n_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
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


def _save_model_artifacts(
    *,
    spec: ModelSpec,
    model,
    cleaner: DataCleaner,
    selector: FeatureSelector,
    scaler: FeatureScaler,
) -> None:
    artifacts = PipelineArtifacts(
        model_name=spec.key,
        model=model,
        imputer=cleaner.imputer,
        variance_filter=selector.variance_filter,
        scaler=scaler,
        selected_features=selector.selected_features,
    )
    ModelIO().save(artifacts)


def _log_model_results(
    *,
    logger: MetricsLogger,
    spec: ModelSpec,
    effective_run_id: str,
    dataset_info: dict,
    train_result_baseline: EvaluationResult,
    test_result_baseline: EvaluationResult,
    model_baseline,
    run_tuning: bool,
    tuner: HyperparameterTuner | None,
    train_result_final: EvaluationResult,
    test_result_final: EvaluationResult,
) -> None:
    baseline_run_id = f"{spec.key}_{effective_run_id}"
    logger.log(
        test_result_baseline,
        run_id=baseline_run_id,
        params={**_extract_model_params(model_baseline, spec), "tuned": False},
        dataset_info={**dataset_info, "model_name": spec.key},
        notes=(
            "Treino vs teste — F1 macro gap: "
            f"{train_result_baseline.f1_macro - test_result_baseline.f1_macro:+.4f}"
        ),
    )

    if run_tuning and tuner is not None:
        logger.log(
            test_result_final,
            run_id=f"tuned_{spec.key}_{effective_run_id}",
            params={**tuner.best_params_, "tuned": True},
            dataset_info={**dataset_info, "model_name": spec.key},
            notes=(
                "Treino vs teste — F1 macro gap: "
                f"{train_result_final.f1_macro - test_result_final.f1_macro:+.4f}"
            ),
        )


def _run_model_flow(
    *,
    spec: ModelSpec,
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_train_bal: pd.DataFrame,
    y_train_bal: pd.Series,
    run_output_dir: Path,
    run_tuning: bool,
    dataset_info_base: dict,
    cleaner: DataCleaner,
    selector: FeatureSelector,
    scaler: FeatureScaler,
    effective_run_id: str,
    logger: MetricsLogger,
) -> None:
    print("\n" + "-" * 72)
    print(f"  Modelo em execucao: {spec.display_name}")
    print("-" * 72)

    trainer = ModelTrainer(save_plots=True, output_dir=run_output_dir)
    evaluator = ModelEvaluator(save_plots=True, output_dir=run_output_dir)
    diagnostics = TrainingDiagnostics(output_dir=run_output_dir)

    print(f"\n[8/11] Treinamento baseline + validacao cruzada ({spec.display_name})...")
    baseline_model = spec.build_baseline(RANDOM_STATE)
    model_baseline = trainer.train(
        X_train_bal,
        y_train_bal,
        model=baseline_model,
        model_name=spec.display_name,
        label=f"{spec.key}_baseline",
        supports_loss_curve=spec.supports_loss_curve,
    )
    cv_results = trainer.cross_validate(
        X_train_raw,
        y_train_raw,
        base_model=baseline_model,
        model_name=spec.display_name,
    )

    dataset_info = {**dataset_info_base, **{"cv_results": {
        metric: {"mean": mean, "std": std}
        for metric, (mean, std) in cv_results.items()
    }}}

    print(f"\n[9/11] Avaliacao baseline em treino/teste ({spec.display_name})...")
    train_result_baseline = evaluator.evaluate(
        model_baseline,
        X_train_scaled,
        y_train.values,
        label=f"{spec.display_name} Baseline (Treino)",
        class_names=TARGET_NAMES,
    )
    test_result_baseline = evaluator.evaluate(
        model_baseline,
        X_test_scaled,
        y_test.values,
        label=f"{spec.display_name} Baseline (Teste)",
        class_names=TARGET_NAMES,
    )

    diagnostics.plot_learning_curve(
        X_train_raw,
        y_train_raw,
        label=f"{spec.key}_baseline",
        estimator=baseline_model,
    )
    diagnostics.plot_generalization_gap(
        train_result_baseline,
        test_result_baseline,
        label=f"{spec.key}_baseline",
    )
    diagnostics.save_gap_report(
        train_result_baseline,
        test_result_baseline,
        label=f"{spec.key}_baseline",
    )

    tuner: HyperparameterTuner | None = None
    train_result_final = train_result_baseline
    test_result_final = test_result_baseline
    model_final = model_baseline

    if run_tuning and spec.supports_tuning:
        print(f"\n[10/11] Hyperparameter tuning ({spec.display_name})...")
        tuner = HyperparameterTuner()
        model_final = tuner.fit(
            X_train_bal,
            y_train_bal,
            estimator=spec.build_baseline(RANDOM_STATE),
            param_distributions=spec.param_distributions or {},
            model_name=spec.display_name,
        )

        train_result_final = evaluator.evaluate(
            model_final,
            X_train_scaled,
            y_train.values,
            label=f"{spec.display_name} Otimizado (Treino)",
            class_names=TARGET_NAMES,
        )
        test_result_final = evaluator.evaluate(
            model_final,
            X_test_scaled,
            y_test.values,
            label=f"{spec.display_name} Otimizado (Teste)",
            class_names=TARGET_NAMES,
        )

        diagnostics.plot_learning_curve(
            X_train_raw,
            y_train_raw,
            label=f"{spec.key}_otimizado",
            estimator=spec.build_baseline(RANDOM_STATE).set_params(**tuner.best_params_),
        )
        diagnostics.plot_generalization_gap(
            train_result_final,
            test_result_final,
            label=f"{spec.key}_otimizado",
        )
        diagnostics.save_gap_report(
            train_result_final,
            test_result_final,
            label=f"{spec.key}_otimizado",
        )
        evaluator.compare(test_result_baseline, test_result_final)
    else:
        print(f"\n[10/11] Tuning ignorado para {spec.display_name}.")

    if spec.supports_explainability:
        explainer = RandomForestExplainer(output_dir=run_output_dir)
        explain_label = f"{spec.key}_{'otimizado' if run_tuning else 'baseline'}"
        explainer.explain(model_final, X_train_scaled, explain_label)

    print(f"\n[11/11] Salvando artefatos e metricas ({spec.display_name})...")
    _save_model_artifacts(
        spec=spec,
        model=model_final,
        cleaner=cleaner,
        selector=selector,
        scaler=scaler,
    )
    _log_model_results(
        logger=logger,
        spec=spec,
        effective_run_id=effective_run_id,
        dataset_info=dataset_info,
        train_result_baseline=train_result_baseline,
        test_result_baseline=test_result_baseline,
        model_baseline=model_baseline,
        run_tuning=run_tuning,
        tuner=tuner,
        train_result_final=train_result_final,
        test_result_final=test_result_final,
    )


def run_pipeline(
    run_tuning: bool = True,
    run_eda: bool = True,
    verbose: bool = True,
    run_id: str | None = None,
    sample_size: int | None = None,
    model_key: str = "mlp",
) -> None:
    del verbose

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_run_id = run_id or f"baseline_{ts_suffix}"
    run_output_dir = OUTPUTS_RUNS_DIR / _slugify_run_id(effective_run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = resolve_requested_models(model_key)

    print("=" * 72)
    print("  Pipeline: Classificacao Multiclasse de Trafego SDN")
    print(f"  Dataset : {DATASET_NAME}")
    print("  Classes : 0=Normal | 1=Flooding | 2=Intrusao")
    print(f"  Modelos : {', '.join(spec.display_name for spec in selected_models)}")
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

    X_train_raw = X_train.reset_index(drop=True).copy()
    y_train_raw = y_train.reset_index(drop=True).copy()
    X_test_raw = X_test.reset_index(drop=True).copy()
    y_test_raw = y_test.reset_index(drop=True).copy()

    print("\n[4/11] Limpeza e preparacao...")
    cleaner = DataCleaner()
    X_train, y_train = cleaner.fit_transform(X_train_raw, y_train_raw)
    X_test, y_test = cleaner.transform(X_test_raw, y_test_raw)

    print("\n[5/11] Selecao de features (VarianceThreshold)...")
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

    logger = MetricsLogger()
    dataset_info_base = _build_dataset_info(
        X=X,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        selector=selector,
        scaler=scaler,
        y=y,
        cv_results={},
        run_output_dir=run_output_dir,
    )

    for spec in selected_models:
        _run_model_flow(
            spec=spec,
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            X_test_scaled=X_test_scaled,
            y_test=y_test,
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            run_output_dir=run_output_dir,
            run_tuning=run_tuning,
            dataset_info_base=dataset_info_base,
            cleaner=cleaner,
            selector=selector,
            scaler=scaler,
            effective_run_id=effective_run_id,
            logger=logger,
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
    parser.add_argument(
        "--model",
        choices=["mlp", "rf", "both"],
        default="mlp",
        help="Modelo a executar: mlp, rf ou both.",
    )

    args = parser.parse_args()
    run_pipeline(
        run_tuning=not args.no_tuning,
        run_eda=not args.no_eda,
        run_id=args.run_id,
        sample_size=args.sample_size,
        model_key=args.model,
    )
