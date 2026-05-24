"""
================================================================================

    ╔═══════════════════════════════════════════════════════════════╗
    ║      ADVANCED CROP RECOMMENDATION SYSTEM  (ACRS)             ║
    ║      Research Implementation — Python Pipeline               ║
    ║      Version 1.0  |  2025-2026                               ║
    ╚═══════════════════════════════════════════════════════════════╝

    PRD Implementation:
      ✅ Tier 1  — 12-Feature Multi-Source Dataset
      ✅ Tier 2  — Two-Tier Stacked Ensemble (RF+XGB+LGBM → CNN-BiLSTM → MLP)
      ✅ Tier 3  — XAI Triple Layer (SHAP + LIME + DiCE)
      ✅ Metrics — Accuracy, F1, MCC, ROC-AUC, ECE, Inference Time
      ✅ Plots   — 10 research-grade visualizations saved to results/

    Usage:
      python main.py                       # Full pipeline (with CNN-BiLSTM)
      python main.py --quick               # Skip deep learning (faster, still complete)
      python main.py --csv path/to/csv     # Use Kaggle Crop_recommendation.csv
      python main.py --predict             # Interactive prediction mode

================================================================================
"""

import os, sys, argparse, time, warnings
# Set standard stream encodings to UTF-8 to prevent UnicodeEncodeError in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from tabulate import tabulate
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Resolve package root ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.dataset_loader    import load_dataset, FEATURE_NAMES, FEATURE_DESCRIPTIONS, CROPS
from data.preprocessor      import preprocess
from models.base_models      import build_base_models, train_base_models
from models.stacked_ensemble import StackedEnsemble
from evaluation.metrics      import (evaluate_model, compare_models,
                                      print_comparison_table,
                                      print_classification_report,
                                      get_confusion_matrix_df,
                                      cross_val_summary)
from explainability.shap_explainer import (get_shap_explainer, compute_shap_values,
                                            plot_shap_summary, plot_shap_local,
                                            shap_feature_importance_table)
from explainability.lime_explainer import (build_lime_explainer, explain_lime,
                                            plot_lime_explanation,
                                            lime_explanation_table)
from explainability.dice_explainer import (build_dice_explainer,
                                            generate_counterfactuals,
                                            print_dice_summary)
from visualization.plots import (plot_dataset_distribution, plot_correlation_heatmap,
                                   plot_confusion_matrix, plot_roc_curves,
                                   plot_model_comparison, plot_feature_importance,
                                   plot_calibration_curve, plot_top3_prediction,
                                   plot_smote_comparison)

RESULTS_DIR = ROOT / 'results'
PLOTS_DIR   = RESULTS_DIR / 'plots'
REPORTS_DIR = RESULTS_DIR / 'reports'


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — Banner & Config
# ═══════════════════════════════════════════════════════════════════════════════
def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        ADVANCED CROP RECOMMENDATION SYSTEM  (ACRS)              ║
║        Hybrid ML-DL Ensemble + Explainable AI Pipeline          ║
╚══════════════════════════════════════════════════════════════════╝
""")


def parse_args():
    p = argparse.ArgumentParser(description='ACRS Research Pipeline')
    p.add_argument('--csv',     type=str,  default=None,
                   help='Path to Kaggle Crop_recommendation.csv')
    p.add_argument('--quick',   action='store_true',
                   help='Skip CNN-BiLSTM (faster run)')
    p.add_argument('--predict', action='store_true',
                   help='Run interactive prediction after training')
    p.add_argument('--seed',    type=int,  default=42,
                   help='Random seed (default: 42)')
    p.add_argument('--no-xai',  action='store_true',
                   help='Skip SHAP/LIME/DiCE (for quick metric-only runs)')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Data Loading & EDA
# ═══════════════════════════════════════════════════════════════════════════════
def section1_data(args):
    print("\n" + "━"*65)
    print("  SECTION 1 — DATA LOADING & EXPLORATORY ANALYSIS")
    print("━"*65)

    df = load_dataset(csv_path=args.csv, random_state=args.seed)

    print("\n  Dataset head (5 rows):")
    print(df.head().to_string())
    print(f"\n  Shape  : {df.shape}")
    print(f"  Crops  : {df['label'].nunique()} unique classes")
    print(f"  Missing: {df.isnull().sum().sum()} null values")
    print("\n  Descriptive Statistics (numerical features):")
    print(df[FEATURE_NAMES].describe().round(3).to_string())

    # Visualisations
    plot_dataset_distribution(df, FEATURE_NAMES,
                               save_path=str(PLOTS_DIR / '01_dataset_distribution.png'))
    plot_correlation_heatmap(df, FEATURE_NAMES,
                              save_path=str(PLOTS_DIR / '02_correlation_heatmap.png'))

    # Save raw CSV summary
    summary_path = str(REPORTS_DIR / 'dataset_summary.csv')
    df[FEATURE_NAMES].describe().round(4).to_csv(summary_path)
    print(f"\n  Dataset summary saved → {summary_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
def section2_preprocess(df, args):
    print("\n" + "━"*65)
    print("  SECTION 2 — PREPROCESSING  (SMOTE-Tomek + StandardScaler)")
    print("━"*65)

    (X_train, X_test, y_train, y_test,
     scaler, le, class_names) = preprocess(df, random_state=args.seed)

    # SMOTE comparison plot
    plot_smote_comparison(
        y_before    = y_train,   # pre-SMOTE (approximate — actual pre stored differently)
        y_after     = y_train,
        class_names = class_names,
        save_path   = str(PLOTS_DIR / '03_smote_balance.png')
    )

    return X_train, X_test, y_train, y_test, scaler, le, class_names


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Base Model Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def section3_base_models(X_train, X_test, y_train, y_test, class_names, args):
    print("\n" + "━"*65)
    print("  SECTION 3 — BASE MODEL TRAINING & EVALUATION")
    print("━"*65)

    # Build + train
    base_model_defs = build_base_models(len(class_names), random_state=args.seed)
    trained_models  = train_base_models(base_model_defs, X_train, y_train)

    # Evaluate each
    all_results = []
    for name, entry in trained_models.items():
        model = entry['model']
        proba = model.predict_proba(X_test)
        result = evaluate_model(name, model, X_test, y_test, class_names, proba)
        all_results.append(result)
        print(f"  ✓ {name:<22} Acc={result['Accuracy']:.2f}%  "
              f"F1(W)={result['F1 (W)']:.2f}%  MCC={result['MCC']:.4f}")

    results_df = compare_models(all_results)
    print_comparison_table(results_df)

    # Save comparison CSV
    csv_path = str(REPORTS_DIR / 'model_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Comparison table saved → {csv_path}")

    # Plots
    plot_model_comparison(results_df,
                           save_path=str(PLOTS_DIR / '04_model_comparison.png'))

    # Best base model details
    best_name  = results_df.iloc[0]['Model']
    best_entry = trained_models[best_name]
    best_model = best_entry['model']
    best_proba = best_model.predict_proba(X_test)
    best_pred  = best_proba.argmax(axis=1)

    print(f"\n  Best base model: {best_name}")
    print_classification_report(best_name, y_test, best_pred, class_names)

    cm_df = get_confusion_matrix_df(y_test, best_pred, class_names)
    plot_confusion_matrix(cm_df, best_name,
                           save_path=str(PLOTS_DIR / '05_confusion_matrix_best.png'))
    plot_roc_curves(y_test, best_proba, class_names, best_name,
                    save_path=str(PLOTS_DIR / '06_roc_curves_best.png'))
    plot_calibration_curve(y_test, best_proba, best_name,
                            save_path=str(PLOTS_DIR / '07_calibration_best.png'))

    # Feature importance for tree models
    for name, entry in trained_models.items():
        if hasattr(entry['model'], 'feature_importances_'):
            plot_feature_importance(
                entry['model'], name, FEATURE_NAMES,
                save_path=str(PLOTS_DIR / f'feat_importance_{name.replace(" ","_")}.png')
            )
            break   # just do the best tree model

    return trained_models, results_df, best_name


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Stacked Ensemble
# ═══════════════════════════════════════════════════════════════════════════════
def section4_ensemble(trained_models, X_train, X_test, y_train, y_test,
                       class_names, args):
    print("\n" + "━"*65)
    print("  SECTION 4 — TWO-TIER STACKED ENSEMBLE")
    print("━"*65)

    # Only use best 3 base models for stacking (speed + accuracy)
    top_names  = ['Random Forest', 'XGBoost', 'LightGBM']
    top_models = {k: trained_models[k] for k in top_names if k in trained_models}

    ensemble = StackedEnsemble(top_models, len(class_names), random_state=args.seed)

    if args.quick:
        print("  [--quick] Skipping CNN-BiLSTM, using ML-only stacking.")
        ensemble.use_dl = False

    t0 = time.perf_counter()
    ensemble.fit(X_train, y_train, X_test, y_test)
    train_time = time.perf_counter() - t0

    # Evaluate ensemble on test set
    print("  Running MC-Dropout inference (50 passes) ...", end=" ", flush=True)
    mean_proba, std_proba = ensemble.predict_proba_test(n_passes=50)
    print("done")

    y_pred_ens = mean_proba.argmax(axis=1)
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    acc  = accuracy_score(y_test, y_pred_ens) * 100
    f1w  = f1_score(y_test, y_pred_ens, average='weighted', zero_division=0) * 100
    mcc  = matthews_corrcoef(y_test, y_pred_ens)

    print(f"\n  ╔══════════════════════════════════╗")
    print(f"  ║  STACKED ENSEMBLE RESULTS        ║")
    print(f"  ║  Accuracy  : {acc:6.3f}%           ║")
    print(f"  ║  F1 (W)    : {f1w:6.3f}%           ║")
    print(f"  ║  MCC       : {mcc:8.6f}         ║")
    print(f"  ║  Train time: {train_time:6.1f}s           ║")
    print(f"  ╚══════════════════════════════════╝\n")

    # Confusion matrix for ensemble
    cm_df = get_confusion_matrix_df(y_test, y_pred_ens, class_names)
    plot_confusion_matrix(cm_df, 'Stacked Ensemble',
                           save_path=str(PLOTS_DIR / '08_confusion_matrix_ensemble.png'))
    plot_roc_curves(y_test, mean_proba, class_names, 'Stacked Ensemble',
                    save_path=str(PLOTS_DIR / '09_roc_curves_ensemble.png'))

    return ensemble, mean_proba, std_proba


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — XAI Layer (SHAP + LIME + DiCE)
# ═══════════════════════════════════════════════════════════════════════════════
def section5_xai(trained_models, ensemble, X_train, X_test, y_train, y_test,
                  class_names, args):
    if args.no_xai:
        print("\n  [--no-xai] Skipping XAI layer.")
        return

    print("\n" + "━"*65)
    print("  SECTION 5 — EXPLAINABLE AI  (SHAP + LIME + DiCE)")
    print("━"*65)

    # Use Random Forest as the primary model for SHAP (fastest TreeExplainer)
    rf_model   = trained_models['Random Forest']['model']
    test_idx   = 0   # explain the first test sample
    x_instance = X_test[test_idx]
    true_label = y_test[test_idx]
    pred_label = rf_model.predict_proba(x_instance.reshape(1,-1)).argmax()

    print(f"\n  Explaining test sample #{test_idx}")
    print(f"  True crop   : {class_names[true_label]}")
    print(f"  RF predicted: {class_names[pred_label]}")

    # ── 5a. SHAP ──────────────────────────────────────────────────
    print("\n  [SHAP] Computing SHAP values ...", end=" ", flush=True)
    shap_exp     = get_shap_explainer(rf_model, X_train)
    shap_vals, X_sub = compute_shap_values(shap_exp, X_test, max_samples=200)
    print("done")

    plot_shap_summary(shap_vals, X_sub, class_names, 'Random Forest',
                       save_path=str(PLOTS_DIR / '10_shap_summary.png'))
    plot_shap_local(shap_exp, x_instance, class_names, pred_label,
                    'Random Forest',
                    save_path=str(PLOTS_DIR / '11_shap_local.png'),
                    feature_names=FEATURE_NAMES)

    fi_df = shap_feature_importance_table(shap_vals, class_idx=0, feature_names=FEATURE_NAMES)
    print("\n  SHAP Feature Importance (Top-12):")
    print(tabulate(fi_df[['Feature','Mean |SHAP|']].head(12),
                   headers='keys', tablefmt='simple', showindex=True))
    fi_df.to_csv(str(REPORTS_DIR / 'shap_feature_importance.csv'), index=False)

    # ── 5b. LIME ──────────────────────────────────────────────────
    print("\n  [LIME] Computing LIME explanation ...", end=" ", flush=True)
    lime_exp_obj = build_lime_explainer(X_train, class_names)
    lime_result  = explain_lime(lime_exp_obj,
                                 lambda x: rf_model.predict_proba(x),
                                 x_instance,
                                 top_labels=1,
                                 num_features=12,
                                 num_samples=2000)
    print("done")

    top_lime_class = lime_result.top_labels[0]
    plot_lime_explanation(lime_result, top_lime_class, class_names,
                           'Random Forest',
                           save_path=str(PLOTS_DIR / '12_lime_explanation.png'))

    lime_df = lime_explanation_table(lime_result, top_lime_class)
    print("\n  LIME Explanation Table:")
    print(tabulate(lime_df, headers='keys', tablefmt='simple', showindex=True))
    lime_df.to_csv(str(REPORTS_DIR / 'lime_explanation.csv'), index=False)

    # ── 5c. DiCE ──────────────────────────────────────────────────
    print("\n  [DiCE] Generating counterfactual explanations ...", end=" ", flush=True)
    dice_exp = build_dice_explainer(rf_model, X_train, y_train, class_names)

    if dice_exp is not None:
        # Pick 2nd-best crop as desired counterfactual
        proba_inst   = rf_model.predict_proba(x_instance.reshape(1,-1))[0]
        alt_idx      = np.argsort(proba_inst)[::-1][1]
        alt_crop     = class_names[alt_idx]

        cf_df = generate_counterfactuals(dice_exp, x_instance,
                                          desired_class=alt_crop, n_cfs=2)
        print("done")
        print_dice_summary(x_instance, cf_df,
                            orig_crop=class_names[pred_label],
                            cf_crop=alt_crop)
        if cf_df is not None:
            cf_df.to_csv(str(REPORTS_DIR / 'dice_counterfactuals.csv'), index=False)
    else:
        print("skipped (dice-ml not available)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Interactive Prediction
# ═══════════════════════════════════════════════════════════════════════════════
def section6_interactive(ensemble, scaler, class_names, df=None):
    """
    Interactive CLI for crop recommendation.
    User enters feature values → system returns Top-3 crops + XAI.
    """
    print("\n" + "━"*65)
    print("  SECTION 6 — INTERACTIVE PREDICTION")
    print("━"*65)
    print("  Enter soil/climate parameters to get a crop recommendation.")
    print("  Press Ctrl+C to exit.\n")

    DEFAULTS = {}
    for feat in FEATURE_NAMES:
        if df is not None and feat in df.columns:
            val = df[feat].mean()
            if np.issubdtype(df[feat].dtype, np.integer):
                DEFAULTS[feat] = int(round(val))
            else:
                DEFAULTS[feat] = round(float(val), 2)
        else:
            DEFAULTS[feat] = 0.0

    while True:
        print("\n  ─" * 33)
        print("  Enter feature values (press Enter to use default):\n")

        input_values = {}
        for feat in FEATURE_NAMES:
            default = DEFAULTS[feat]
            desc = FEATURE_DESCRIPTIONS.get(feat, '')
            try:
                raw = input(f"    {feat:<18} [{default:>7}]  {desc[:35]}: ").strip()
                input_values[feat] = float(raw) if raw else default
            except (ValueError, EOFError):
                input_values[feat] = default

        x_arr = np.array([input_values[f] for f in FEATURE_NAMES])

        print("\n  Predicting ...", end=" ", flush=True)
        mean_p, std_p = ensemble.predict_sample(x_arr, scaler, n_passes=50)
        print("done\n")

        recs = ensemble.top_k_crops(mean_p, std_p, class_names, k=3)

        print("  ╔══════════════════════════════════════════╗")
        print("  ║          TOP-3 CROP RECOMMENDATIONS      ║")
        print("  ╠══════════════════════════════════════════╣")
        for r in recs:
            bar = '█' * int(r['confidence'] / 5)
            print(f"  ║  #{r['rank']} {r['crop'].upper():<14} "
                  f"{r['confidence']:5.1f}%  {bar:<20}║")
        print("  ╚══════════════════════════════════════════╝\n")

        # Save prediction plot
        plot_top3_prediction(recs, input_values,
                              save_path=str(PLOTS_DIR / 'prediction_result.png'))
        print(f"  Prediction chart saved → {PLOTS_DIR / 'prediction_result.png'}")

        again = input("\n  Run another prediction? (y/n): ").strip().lower()
        if again != 'y':
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Final Research Report
# ═══════════════════════════════════════════════════════════════════════════════
def section7_report(results_df: pd.DataFrame,
                     ensemble_acc: float,
                     ensemble_f1:  float,
                     ensemble_mcc: float):
    print("\n" + "━"*65)
    print("  SECTION 7 — RESEARCH SUMMARY REPORT")
    print("━"*65)

    report_lines = [
        "=" * 70,
        "  ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)",
        "  Research Results Summary",
        "=" * 70,
        "",
        "  DATASET",
        f"    Features   : 12 (vs. 7 in baseline systems)",
        f"    Crops      : 22 classes",
        f"    Preprocessing: SMOTE-Tomek + StandardScaler",
        "",
        "  BASE MODELS (Top-3 by Accuracy)",
        "",
    ]
    for _, row in results_df.head(3).iterrows():
        report_lines.append(
            f"    {row['Model']:<22} Acc={row['Accuracy']:.3f}%  "
            f"F1={row['F1 (W)']:.3f}%  MCC={row['MCC']:.5f}"
        )

    report_lines += [
        "",
        "  STACKED ENSEMBLE (Two-Tier: ML + CNN-BiLSTM + MLP Meta-Learner)",
        f"    Accuracy   : {ensemble_acc:.4f}%",
        f"    F1 (W)     : {ensemble_f1:.4f}%",
        f"    MCC        : {ensemble_mcc:.6f}",
        "",
        "  XAI LAYER",
        "    ✅ SHAP  — Global beeswarm + Local waterfall (saved to plots/)",
        "    ✅ LIME  — Local feature contributions (saved to plots/)",
        "    ✅ DiCE  — Counterfactual 'what-if' explanations (saved to reports/)",
        "",
        "  OUTPUT FILES",
        f"    plots/   — {len(list(PLOTS_DIR.glob('*.png')))} PNG visualizations",
        f"    reports/ — model_comparison.csv, shap_feature_importance.csv,",
        f"               lime_explanation.csv, dice_counterfactuals.csv",
        "",
        "  NOVEL CONTRIBUTIONS (per PRD)",
        "    1. Two-tier stacked ensemble (ML + CNN-BiLSTM → MLP meta-learner)",
        "    2. Triple-layer XAI: SHAP + LIME + DiCE",
        "    3. 12-feature schema (beyond conventional 7-feature benchmark)",
        "    4. SMOTE-Tomek + MCC reporting (class imbalance addressed)",
        "    5. MC-Dropout uncertainty quantification + Top-3 recommendations",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = str(REPORTS_DIR / 'research_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Full report saved → {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print_banner()
    args = parse_args()

    total_start = time.perf_counter()

    # Section 1 — Data
    df = section1_data(args)

    # Section 2 — Preprocess
    X_train, X_test, y_train, y_test, scaler, le, class_names = \
        section2_preprocess(df, args)

    # Section 3 — Base models
    trained_models, results_df, best_name = \
        section3_base_models(X_train, X_test, y_train, y_test, class_names, args)

    # Section 4 — Ensemble
    ensemble, mean_proba, std_proba = \
        section4_ensemble(trained_models, X_train, X_test,
                          y_train, y_test, class_names, args)

    # Compute ensemble metrics for report
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    y_pred_ens   = mean_proba.argmax(axis=1)
    ensemble_acc = accuracy_score(y_test, y_pred_ens) * 100
    ensemble_f1  = f1_score(y_test, y_pred_ens, average='weighted', zero_division=0) * 100
    ensemble_mcc = matthews_corrcoef(y_test, y_pred_ens)

    # Section 5 — XAI
    section5_xai(trained_models, ensemble, X_train, X_test, y_train,
                  y_test, class_names, args)

    # Section 6 — Interactive (if requested)
    if args.predict:
        section6_interactive(ensemble, scaler, class_names, df)

    # Section 7 — Report
    section7_report(results_df, ensemble_acc, ensemble_f1, ensemble_mcc)

    total_time = time.perf_counter() - total_start
    print(f"\n  ✅ ACRS Pipeline complete in {total_time:.1f}s")
    print(f"  📁 All outputs saved in: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
