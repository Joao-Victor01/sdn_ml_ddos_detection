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

import matplotlib
matplotlib.use("Agg")  # backend não-interativo — obrigatório com n_jobs=-1

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
    IDENTITY_FEATURES,
    PERMUTATION_N_REPEATS,
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

    # Guarda: feature engineering não deve introduzir NaN
    nan_train = X_train.isnull().sum().sum()
    nan_test  = X_test.isnull().sum().sum()
    if nan_train or nan_test:
        bad_cols = X_train.columns[X_train.isnull().any()].tolist()
        raise ValueError(
            f"NaN introduzido pela feature engineering — colunas: {bad_cols}"
        )

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

    # 14a — Impurity-based importance (Gini, nativa do RF)
    _plot_feature_importance(
        rf_best, selected_features, computed_feats, save_plots
    )

    # 14b — Permutation Importance no test set
    #        Razão: importância Gini superestima features de alta cardinalidade
    #        (ex: porta, protocolo). A permutation importance no test set mede o
    #        impacto real de cada feature na métrica de generalização.
    #        Localização obrigatória: APÓS Etapa 12 (avaliação já realizada);
    #        uso do test set é apenas diagnóstico — não influencia nenhum modelo.
    print(f"\n[14b] Permutation Importance no test set "
          f"({PERMUTATION_N_REPEATS} repetições)...")
    perm_result = _run_permutation_importance(
        rf_best,
        X_test_vt.values,
        y_test.values,
        selected_features,
        computed_feats,
        n_repeats=PERMUTATION_N_REPEATS,
        save=save_plots,
    )

    # 14c — Ablation Study: retreinar sem features de identidade
    #        Mede quanto do F1 é explicado por atalhos de identidade
    #        (Dst Port, Src Port, Protocol, etc.) em vez de padrões comportamentais.
    print(f"\n[14c] Ablation Study — retreinar sem features de identidade...")
    ablation_result = _run_ablation_study(
        X_train_bal,
        y_train_bal,
        X_test_vt,
        y_test.values,
        selected_features,
        result_rf_best.f1_macro,
        save=save_plots,
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
    print(f"  Ablation F1 (sem identidade): {ablation_result['f1_ablation']:.4f} "
          f"| Queda: {ablation_result['f1_drop']:.4f} "
          f"({'OK — comportamental' if ablation_result['f1_drop'] < 0.05 else 'ALERTA — atalho de identidade'})")

    return {
        "rf_baseline":         result_rf_base,
        "rf_best":             result_rf_best,
        "mlp":                 result_mlp,
        "botnet_validation":   botnet_result,
        "cv_rf":               cv_rf,
        "cv_mlp":              cv_mlp,
        "selected_features":   selected_features,
        "permutation_importance": perm_result,
        "ablation":            ablation_result,
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


def _run_permutation_importance(
    rf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    computed_feats: list[str],
    n_repeats: int = PERMUTATION_N_REPEATS,
    save: bool = True,
) -> dict:
    """
    Permutation Importance no test set — diagnóstico de atalhos de identidade.

    Por que usar o test set aqui (e não o treino):
      - O test set representa dados "do mundo real" nunca vistos.
      - Permutação no treino mede apenas quais features o modelo memorizou.
      - Permutação no teste mede quais features contribuem para generalização.
      - Uso diagnóstico: NÃO altera nenhum modelo; é análise post-hoc.

    Como interpretar:
      - Importância alta (média >> 0) → feature genuinamente útil.
      - Importância ≈ 0              → feature irrelevante para generalização.
      - Importância negativa         → feature introduz ruído (raramente acontece).
      - Features de identidade com alta importância → risco de overfitting
        para características do dataset, não do problema real.

    Returns
    -------
    dict com 'importances_mean', 'importances_std', 'sorted_df', 'identity_warning'
    """
    from sklearn.inspection import permutation_importance

    print(f"  Calculando permutation importance ({n_repeats} repetições)...")
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=n_repeats,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature":    feature_names,
        "mean":       result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("mean", ascending=False)

    df_perm["identity"] = df_perm["feature"].isin(IDENTITY_FEATURES)
    df_perm["nova"]     = df_perm["feature"].isin(computed_feats)

    # Diagnóstico: features de identidade no top-10?
    top10 = df_perm.head(10)
    identity_in_top10 = top10[top10["identity"]]["feature"].tolist()
    identity_warning  = len(identity_in_top10) > 0

    print(f"\n  Top 15 features (permutation importance — test set):")
    print(f"  {'Feature':<35} {'Mean':>8} {'Std':>8} {'Tipo'}")
    print(f"  {'-'*65}")
    for _, row in df_perm.head(15).iterrows():
        tipo = "IDENTIDADE" if row["identity"] else ("NOVA" if row["nova"] else "original")
        print(f"  {row['feature']:<35} {row['mean']:>8.5f} {row['std']:>8.5f}  {tipo}")

    if identity_warning:
        print(f"\n  ⚠ ALERTA: features de identidade no Top-10 de generalização:")
        for f in identity_in_top10:
            row = df_perm[df_perm["feature"] == f].iloc[0]
            print(f"    {f}: mean={row['mean']:.5f} ± {row['std']:.5f}")
        print("  → Considerar ablation study (sub-passo 14c) para quantificar o impacto.")
    else:
        print("\n  ✓ Nenhuma feature de identidade no Top-10 — padrão comportamental dominante.")

    if save:
        _plot_permutation_importance(df_perm)

    return {
        "importances_mean": result.importances_mean,
        "importances_std":  result.importances_std,
        "sorted_df":        df_perm,
        "identity_warning": identity_warning,
        "identity_in_top10": identity_in_top10,
    }


def _plot_permutation_importance(df_perm: pd.DataFrame) -> None:
    """Barras horizontais com intervalo de confiança por feature."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        top = df_perm.head(25).iloc[::-1]  # ordem crescente para barh
        colors = [
            "crimson"    if r["identity"] else
            "darkorange" if r["nova"]     else
            "steelblue"
            for _, r in top.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(top["feature"], top["mean"], xerr=top["std"],
                color=colors, alpha=0.85, capsize=3, ecolor="gray")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Redução de F1 Macro ao permutar (média ± std)")
        ax.set_title(
            "Permutation Importance — RF Triclasse (test set)\n"
            "Crimson = identidade  |  Laranja = comportamental nova  |  Azul = original"
        )
        ax.legend(handles=[
            Patch(color="crimson",    label="Feature de identidade (suspeita)"),
            Patch(color="darkorange", label="Feature comportamental (nova)"),
            Patch(color="steelblue",  label="Feature original InSDN"),
        ])
        plt.tight_layout()
        path = OUTPUTS_TRICLASS / "permutation_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Permutation importance salva → {path.name}")
    except Exception as e:
        print(f"  Erro ao plotar permutation importance: {e}")


def _run_ablation_study(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_vt: pd.DataFrame,
    y_test: np.ndarray,
    selected_features: list[str],
    f1_full: float,
    save: bool = True,
) -> dict:
    """
    Ablation Study: retreinar RF sem as features de identidade.

    Mede quanto do F1 é explicado por atalhos de identidade vs. padrões
    comportamentais reais. Se a queda for pequena (< 0.05), o modelo
    generaliza por comportamento — não por memorizar porta/protocolo.

    Por que retreinar e não apenas mascarar:
      - Mascarar colunas no teste com modelo treinado no set completo não é
        ablation real — o modelo já aprendeu relações com essas features.
      - Retreinar sem elas força o modelo a aprender sem esses sinais.

    Por que usar X_train_bal (pós-SMOTE) e não X_train_vt (pré-SMOTE):
      - O modelo de referência foi treinado em X_train_bal.
      - Para comparação justa, o modelo ablation deve ter as mesmas condições
        exceto pela ausência das features de identidade.

    Returns
    -------
    dict com 'f1_full', 'f1_ablation', 'f1_drop', 'features_removed',
             'features_kept', 'identity_warning'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    # Identificar quais features de identidade estão presentes após VT
    identity_present = [f for f in IDENTITY_FEATURES if f in selected_features]
    features_kept    = [f for f in selected_features if f not in IDENTITY_FEATURES]

    if not identity_present:
        print("  Nenhuma feature de identidade presente após VarianceThreshold — ablation ignorada.")
        return {
            "f1_full":          f1_full,
            "f1_ablation":      f1_full,
            "f1_drop":          0.0,
            "features_removed": [],
            "features_kept":    features_kept,
            "identity_warning": False,
        }

    print(f"  Features de identidade a remover: {identity_present}")
    print(f"  Features restantes no ablation: {len(features_kept)}")

    # Índices das colunas a manter em X_train_bal (numpy array)
    keep_idx = [selected_features.index(f) for f in features_kept]

    X_train_abl = X_train_bal[:, keep_idx]
    X_test_abl  = X_test_vt[features_kept].values

    rf_ablation = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_ablation.fit(X_train_abl, y_train_bal)
    y_pred_abl  = rf_ablation.predict(X_test_abl)
    f1_ablation = float(f1_score(y_test, y_pred_abl, average="macro", zero_division=0))
    f1_drop     = f1_full - f1_ablation

    print(f"\n  F1 Macro com todas as features    : {f1_full:.4f}")
    print(f"  F1 Macro sem features identidade  : {f1_ablation:.4f}")
    print(f"  Queda                             : {f1_drop:+.4f}")

    if f1_drop < 0.02:
        verdict = "✓ Queda < 0.02 — modelo generaliza por comportamento, não por identidade."
        warning = False
    elif f1_drop < 0.05:
        verdict = "⚠ Queda entre 0.02–0.05 — features de identidade contribuem moderadamente."
        warning = True
    else:
        verdict = "✗ Queda > 0.05 — modelo depende fortemente de atalhos de identidade."
        warning = True
    print(f"\n  Veredito: {verdict}")

    if save:
        _plot_ablation(f1_full, f1_ablation, identity_present, features_kept)

    return {
        "f1_full":          f1_full,
        "f1_ablation":      f1_ablation,
        "f1_drop":          f1_drop,
        "features_removed": identity_present,
        "features_kept":    features_kept,
        "identity_warning": warning,
    }


def _plot_ablation(
    f1_full: float,
    f1_ablation: float,
    features_removed: list[str],
    features_kept: list[str],
) -> None:
    """Gráfico de barras comparando F1 completo vs. ablation."""
    try:
        import matplotlib.pyplot as plt

        labels = ["Modelo completo", "Sem features\nde identidade"]
        values = [f1_full, f1_ablation]
        colors = ["steelblue", "darkorange"]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=12,
                    fontweight="bold")

        ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.05))
        ax.set_ylabel("F1 Macro (test set)")
        ax.set_title(
            f"Ablation Study — Features de Identidade\n"
            f"Removidas: {', '.join(features_removed)}"
        )

        drop = f1_full - f1_ablation
        color = "darkgreen" if drop < 0.02 else ("orange" if drop < 0.05 else "crimson")
        ax.text(0.5, 0.05,
                f"Queda: {drop:+.4f}  "
                f"({'OK' if drop < 0.02 else 'MODERADA' if drop < 0.05 else 'ALTA'})",
                transform=ax.transAxes, ha="center", fontsize=11,
                color=color, fontweight="bold")

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = OUTPUTS_TRICLASS / "ablation_identity_features.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Ablation plot salvo → {path.name}")
    except Exception as e:
        print(f"  Erro ao plotar ablation: {e}")


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
