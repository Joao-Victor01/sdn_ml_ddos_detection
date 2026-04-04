"""
Pipeline principal — Detecção Triclasse de DDoS em SDN (InSDN v4).

Implementa rigorosamente as 14 etapas do plano e do guia de boas práticas:

  1.  Configurações e reprodutibilidade
  2.  Carregamento dos três CSVs do InSDN
  3.  EDA (sem modificar dados)
  4.  Labeling triclasse (heurística is_burst)
  5.  ⚠️  SPLIT treino/teste (stratify=y) — ANTES de qualquer transformação
  6.  Limpeza: duplicatas, Inf→NaN, imputação (fit somente no treino)
  7.  Features comportamentais (puras — sem leakage)
  8.  VarianceThreshold (fit somente no treino)
  9.  SMOTE conservador (somente no treino)
  10. CV 10-fold no treino — RF baseline e MLP Pipeline
  11. Hyperparameter Tuning RF (RandomizedSearchCV no treino)
  12. Avaliação final no test_set (UMA ÚNICA VEZ)
  13. Validação semântica BOTNET
  14. Importância de features + salvamento de artefatos

Referência: plano_triclasse_insdn_v4.md + guia_boas_praticas_ml.md
"""

from __future__ import annotations

import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from ml.triclass.config import (
    RANDOM_STATE,
    TEST_SIZE,
    IMPUTER_STRATEGY,
    VARIANCE_THRESHOLD,
    SMOTE_MAX_FACTOR_CLS2,
    SMOTE_UNDERSAMPLE_RATIO_CLS1,
    MODELS_TRICLASS_DIR,
    OUTPUTS_TRICLASS,
    BEHAVIORAL_FEATURES,
)
from ml.triclass.data.loader import InSDNTriclassLoader
from ml.triclass.preprocessing.labeler import TriclassLabeler
from ml.triclass.preprocessing.feature_engineer import BehavioralFeatureEngineer
from ml.triclass.training.trainer import TriclassTrainer
from ml.triclass.training.tuner import TriclassTuner
from ml.triclass.evaluation.evaluator import TriclassEvaluator
from ml.triclass.evaluation.semantic_validator import BotnetSemanticValidator

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)


def run_triclass_pipeline(
    run_tuning: bool = True,
    run_eda: bool = True,
    save_plots: bool = True,
    run_id: str | None = None,
) -> dict:
    """
    Executa o pipeline triclasse completo.

    Parameters
    ----------
    run_tuning  : bool — executa RandomizedSearchCV (mais lento)
    run_eda     : bool — exibe análise exploratória textual
    save_plots  : bool — salva gráficos em outputs_triclass/
    run_id      : str  — identificador para log de métricas

    Returns
    -------
    dict com resultados: 'rf_baseline', 'rf_best', 'mlp', 'botnet_validation'
    """
    OUTPUTS_TRICLASS.mkdir(parents=True, exist_ok=True)
    MODELS_TRICLASS_DIR.mkdir(parents=True, exist_ok=True)

    _banner("Pipeline Triclasse — Detecção de DDoS em SDN (InSDN v4)")

    # ── Etapa 1: Configurações ─────────────────────────────────────────────────
    _step(1, "Configurações")
    print(f"  RANDOM_STATE     : {RANDOM_STATE}")
    print(f"  TEST_SIZE        : {TEST_SIZE} (70/30)")
    print(f"  VARIANCE_THRESH  : {VARIANCE_THRESHOLD}")
    print(f"  SMOTE fator cls2 : {SMOTE_MAX_FACTOR_CLS2}x")

    # ── Etapa 2: Carregamento ──────────────────────────────────────────────────
    _step(2, "Carregando InSDN (3 arquivos CSV)")
    loader = InSDNTriclassLoader()
    data_raw = loader.load()

    # ── Etapa 3: EDA ──────────────────────────────────────────────────────────
    if run_eda:
        _step(3, "EDA (sem modificar dados)")
        loader.describe(data_raw)
    else:
        _step(3, "EDA ignorada (run_eda=False)")

    # ── Etapa 4: Labeling triclasse ────────────────────────────────────────────
    _step(4, "Labeling triclasse (heurística is_burst)")
    labeler = TriclassLabeler()
    data = labeler.fit_transform(data_raw)

    X_all = data.drop(columns=["Label", "label_3class", "_source"], errors="ignore")
    X_all = X_all.select_dtypes(include=np.number)
    y_all = data["label_3class"]

    # Preservar DataFrame original com labels para validação semântica
    # (precisamos dos índices originais após o split)
    data_for_validation = data.copy()

    # ── Etapa 5: SPLIT (ANTES de qualquer transformação) ──────────────────────
    _step(5, "Split estratificado 70/30 (ANTES de qualquer transformação)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_all,
    )
    # Preservar índices do teste para validação semântica
    test_original_indices = X_test.index.values

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    print(f"  Treino : {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
    _print_dist("Distribuição treino", y_train)
    _print_dist("Distribuição teste ", y_test)

    # ── Etapa 6: Limpeza (somente no treino) ──────────────────────────────────
    _step(6, "Limpeza: duplicatas, Inf→NaN, imputação (fit somente no treino)")

    # 6a. Duplicatas no treino (junto com y para consistência)
    df_tmp = X_train.copy()
    df_tmp["__y__"] = y_train.values
    before = len(df_tmp)
    df_tmp = df_tmp.drop_duplicates(keep="first")
    after  = len(df_tmp)
    print(f"  Duplicatas removidas do treino: {before - after:,}")
    X_train = df_tmp.drop(columns=["__y__"])
    y_train = pd.Series(df_tmp["__y__"].values, name="label_3class")

    # 6b. Inf → NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    # 6c. Imputação com mediana (fit somente no treino)
    imputer = SimpleImputer(strategy=IMPUTER_STRATEGY)
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )
    X_test  = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns
    )
    print(f"  NaN pós-imputação treino : {X_train.isnull().sum().sum()}")
    print(f"  NaN pós-imputação teste  : {X_test.isnull().sum().sum()}")

    # ── Etapa 7: Features comportamentais ─────────────────────────────────────
    _step(7, "Features comportamentais (puras — sem leakage)")
    eng = BehavioralFeatureEngineer()
    X_train = eng.fit_transform(X_train)
    X_test  = eng.transform(X_test)

    computed_feats = [f for f in BEHAVIORAL_FEATURES if f in X_train.columns]

    # ── Etapa 8: VarianceThreshold (fit somente no treino) ────────────────────
    _step(8, "VarianceThreshold (fit somente no treino)")
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_train_vt = pd.DataFrame(
        vt.fit_transform(X_train),
        columns=X_train.columns[vt.get_support()],
    )
    X_test_vt = pd.DataFrame(
        vt.transform(X_test),
        columns=X_train.columns[vt.get_support()],
    )
    removed = (~vt.get_support()).sum()
    print(f"  Features removidas (var≤{VARIANCE_THRESHOLD}): {removed}")
    print(f"  Features restantes: {X_train_vt.shape[1]}")
    selected_features = X_train_vt.columns.tolist()

    # ── Etapa 9: SMOTE conservador (somente no treino) ────────────────────────
    _step(9, "SMOTE conservador (somente no treino)")
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    n2 = int((y_train == 2).sum())
    print(f"  Antes: Cl0={n0:,}  Cl1={n1:,}  Cl2={n2:,}")

    target_cls2 = min(n2 * SMOTE_MAX_FACTOR_CLS2, n0)
    target_cls2 = max(target_cls2, n2 + 1)  # garante ao menos 1 amostra extra

    smote = SMOTE(
        sampling_strategy={2: target_cls2},
        random_state=RANDOM_STATE,
        k_neighbors=min(5, n2 - 1),  # evita erro quando n2 é muito pequeno
    )
    X_res, y_res = smote.fit_resample(X_train_vt, y_train)

    # Undersample classe 1 se muito maior que classe 0
    n1_res = int((y_res == 1).sum())
    n0_res = int((y_res == 0).sum())
    if n1_res > n0_res * 2:
        target_cls1 = int(n0_res * SMOTE_UNDERSAMPLE_RATIO_CLS1)
        under = RandomUnderSampler(
            sampling_strategy={1: target_cls1},
            random_state=RANDOM_STATE,
        )
        X_train_bal, y_train_bal = under.fit_resample(X_res, y_res)
    else:
        X_train_bal, y_train_bal = X_res, y_res

    n0b = int((y_train_bal == 0).sum())
    n1b = int((y_train_bal == 1).sum())
    n2b = int((y_train_bal == 2).sum())
    print(f"  Depois: Cl0={n0b:,}  Cl1={n1b:,}  Cl2={n2b:,}")

    # ── Etapa 10: CV 10-fold no treino ────────────────────────────────────────
    _step(10, "Validação cruzada 10-fold no treino")
    trainer = TriclassTrainer(save_plots=save_plots)
    cv_rf  = trainer.cross_validate_rf(X_train_bal, y_train_bal)
    cv_mlp = trainer.cross_validate_mlp(X_train_bal, y_train_bal)

    # ── Treinar modelos completos no treino balanceado ─────────────────────────
    rf_baseline = trainer.train_rf(X_train_bal, y_train_bal)
    mlp_pipe    = trainer.train_mlp(X_train_bal, y_train_bal)

    # ── Etapa 11: Hyperparameter Tuning (somente no treino) ───────────────────
    rf_best = rf_baseline  # fallback se tuning desabilitado
    if run_tuning:
        _step(11, "Hyperparameter Tuning RF (RandomizedSearchCV no treino)")
        tuner   = TriclassTuner()
        rf_best = tuner.fit(X_train_bal, y_train_bal)
    else:
        _step(11, "Tuning ignorado (run_tuning=False)")

    # ── Etapa 12: Avaliação final no test_set (UMA ÚNICA VEZ) ─────────────────
    _step(12, "Avaliação final no test_set (UMA ÚNICA VEZ)")
    evaluator = TriclassEvaluator(save_plots=save_plots)

    result_rf_base = evaluator.evaluate(
        rf_baseline, X_test_vt.values, y_test.values,
        label="RF Baseline",
    )
    result_rf_best = evaluator.evaluate(
        rf_best, X_test_vt.values, y_test.values,
        label="RF Otimizado" if run_tuning else "RF Baseline",
    )
    result_mlp = evaluator.evaluate(
        mlp_pipe, X_test_vt.values, y_test.values,
        label="MLP Pipeline",
    )
    evaluator.compare(result_rf_base, result_rf_best, result_mlp)

    # ── Etapa 13: Validação semântica BOTNET ──────────────────────────────────
    _step(13, "Validação semântica BOTNET (ground truth)")
    semantic_validator = BotnetSemanticValidator(
        data_original=data_for_validation,
        test_indices=test_original_indices,
    )
    botnet_result = semantic_validator.validate(rf_best, X_test_vt.values)

    # ── Etapa 14: Importância de features + salvamento ────────────────────────
    _step(14, "Importância de features + salvamento de artefatos")

    _plot_feature_importance(
        rf_best, selected_features, computed_feats, save_plots
    )

    # Salvar todos os artefatos
    joblib.dump(rf_best,           MODELS_TRICLASS_DIR / "rf_triclass.joblib")
    joblib.dump(mlp_pipe,          MODELS_TRICLASS_DIR / "mlp_triclass.joblib")
    joblib.dump(imputer,           MODELS_TRICLASS_DIR / "imputer.joblib")
    joblib.dump(vt,                MODELS_TRICLASS_DIR / "variance_filter.joblib")
    joblib.dump(selected_features, MODELS_TRICLASS_DIR / "selected_features.joblib")
    joblib.dump(computed_feats,    MODELS_TRICLASS_DIR / "computed_features.joblib")

    print(f"\n  Artefatos salvos em {MODELS_TRICLASS_DIR}/")

    # ── Resumo final ───────────────────────────────────────────────────────────
    _banner("Pipeline concluído com sucesso!")
    print(f"  Modelos  : {MODELS_TRICLASS_DIR}/")
    print(f"  Plots    : {OUTPUTS_TRICLASS}/")
    print()
    print(f"  RF Best  — F1 Macro: {result_rf_best.f1_macro:.4f} | "
          f"MCC: {result_rf_best.mcc:.4f} | "
          f"Recall Cl2: {result_rf_best.recall_class2:.4f}")
    print(f"  MLP Pipe — F1 Macro: {result_mlp.f1_macro:.4f} | "
          f"MCC: {result_mlp.mcc:.4f} | "
          f"Recall Cl2: {result_mlp.recall_class2:.4f}")
    print(f"  BOTNET recall semântico: {botnet_result.recall:.4f} "
          f"({'PASSOU' if botnet_result.passed else 'FALHOU'})")

    return {
        "rf_baseline":       result_rf_base,
        "rf_best":           result_rf_best,
        "mlp":               result_mlp,
        "botnet_validation": botnet_result,
        "cv_rf":             cv_rf,
        "cv_mlp":            cv_mlp,
        "selected_features": selected_features,
    }


# ── Helpers internos ───────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


def _step(n: int, msg: str) -> None:
    print(f"\n[{n}/14] {msg}...")


def _print_dist(label: str, y: pd.Series) -> None:
    from ml.triclass.config import CLASS_NAMES
    vc = y.value_counts().sort_index()
    parts = [
        f"Cl{cls}({CLASS_NAMES.get(cls,'?')})={cnt:,} ({100*cnt/len(y):.1f}%)"
        for cls, cnt in vc.items()
    ]
    print(f"  {label}: {' | '.join(parts)}")


def _plot_feature_importance(
    rf,
    selected_features: list[str],
    computed_feats: list[str],
    save: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        df_imp = pd.DataFrame({
            "feature":    selected_features,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False).head(25)

        df_imp["nova"] = df_imp["feature"].isin(computed_feats)

        fig, ax = plt.subplots(figsize=(10, 8))
        cores = ["darkorange" if n else "steelblue"
                 for n in df_imp["nova"].iloc[::-1]]
        ax.barh(df_imp["feature"].iloc[::-1],
                df_imp["importance"].iloc[::-1],
                color=cores)
        ax.set_xlabel("Importância (redução de impureza)")
        ax.set_title("Top 25 Features — RF Triclasse\nLaranja = features comportamentais")
        ax.legend(handles=[
            Patch(color="darkorange", label="Feature comportamental (nova)"),
            Patch(color="steelblue",  label="Feature original InSDN"),
        ])
        plt.tight_layout()

        if save:
            path = OUTPUTS_TRICLASS / "feature_importance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Importância de features salva → {path.name}")
        plt.close()
    except Exception as e:
        print(f"  Não foi possível plotar importância: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Triclasse — Detecção de DDoS em SDN (InSDN v4)"
    )
    parser.add_argument(
        "--no-tuning", action="store_true",
        help="Pular hyperparameter tuning (mais rápido)"
    )
    parser.add_argument(
        "--no-eda", action="store_true",
        help="Pular análise exploratória"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Não salvar gráficos"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Identificador da run (ex: 'experimento_v1')"
    )
    args = parser.parse_args()

    run_triclass_pipeline(
        run_tuning=not args.no_tuning,
        run_eda=not args.no_eda,
        save_plots=not args.no_plots,
        run_id=args.run_id,
    )
