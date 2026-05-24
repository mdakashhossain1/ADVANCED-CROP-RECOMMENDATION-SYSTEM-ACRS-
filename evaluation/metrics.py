"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Evaluation Module

Computes comprehensive metrics per the PRD experimental protocol:
  - Accuracy, Precision, Recall, F1-Score (macro & weighted)
  - Matthews Correlation Coefficient (MCC)
  - ROC-AUC (macro OvR)
  - Expected Calibration Error (ECE)
  - Inference time (ms per sample)
  - Confusion matrix
  - Per-class classification report

Results are printed as formatted tables and returned as DataFrames.
================================================================================
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report
)
from tabulate import tabulate


# ── Expected Calibration Error ───────────────────────────────────────────────
def expected_calibration_error(y_true: np.ndarray,
                                proba:  np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well model confidence aligns with actual accuracy.
    Lower is better. ECE = Σ |acc_bin - conf_bin| * n_bin/N
    """
    predicted   = np.argmax(proba, axis=1)
    confidences = proba.max(axis=1)
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracy_score(y_true[mask], predicted[mask])
        bin_conf = confidences[mask].mean()
        ece += abs(bin_acc - bin_conf) * mask.sum()

    return ece / len(y_true)


# ── Inference speed benchmark ─────────────────────────────────────────────────
def measure_inference_time(model, X_sample: np.ndarray, n_repeats: int = 100) -> float:
    """
    Measure average inference time in ms per single sample.
    Warms up the model first with 10 dummy runs.
    """
    single = X_sample[:1]

    # Warmup
    for _ in range(10):
        try:
            model.predict_proba(single)
        except Exception:
            model.predict(single)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        try:
            model.predict_proba(single)
        except Exception:
            model.predict(single)
        times.append((time.perf_counter() - t0) * 1000)

    return round(np.mean(times), 4)


# ── Single model evaluation ───────────────────────────────────────────────────
def evaluate_model(name:        str,
                   model,
                   X_test:      np.ndarray,
                   y_test:      np.ndarray,
                   class_names: list,
                   proba:       np.ndarray | None = None) -> dict:
    """
    Evaluate a single model and return a metrics dictionary.

    Parameters
    ----------
    name        : Display name for the model
    model       : Fitted sklearn/keras estimator
    X_test      : Scaled test features
    y_test      : True integer labels
    class_names : List of crop name strings
    proba       : Pre-computed probability array (optional)
    """
    if proba is None:
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None

    y_pred = np.argmax(proba, axis=1) if proba is not None else model.predict(X_test)

    metrics = {
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted',
                                           zero_division=0) * 100, 4),
        'Recall':    round(recall_score(y_test, y_pred, average='weighted',
                                        zero_division=0) * 100, 4),
        'F1 (W)':    round(f1_score(y_test, y_pred, average='weighted',
                                    zero_division=0) * 100, 4),
        'F1 (M)':    round(f1_score(y_test, y_pred, average='macro',
                                    zero_division=0) * 100, 4),
        'MCC':       round(matthews_corrcoef(y_test, y_pred), 6),
    }

    if proba is not None:
        try:
            metrics['ROC-AUC'] = round(
                roc_auc_score(y_test, proba, multi_class='ovr',
                              average='macro'), 6)
        except Exception:
            metrics['ROC-AUC'] = None
        metrics['ECE'] = round(expected_calibration_error(y_test, proba), 6)

    metrics['Infer (ms)'] = measure_inference_time(model, X_test)
    return metrics


# ── Compare all models ────────────────────────────────────────────────────────
def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a sorted comparison table from a list of metrics dicts.

    Returns
    -------
    pd.DataFrame  sorted by Accuracy descending
    """
    df = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    df.index = df.index + 1   # 1-based rank
    return df


def print_comparison_table(df: pd.DataFrame):
    """Pretty-print the comparison table to stdout."""
    print("\n" + "="*90)
    print("  MODEL COMPARISON — COMPREHENSIVE EVALUATION METRICS")
    print("="*90)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid',
                   floatfmt='.4f', showindex=True))
    print()


# ── Per-class report ──────────────────────────────────────────────────────────
def print_classification_report(model_name: str,
                                  y_test:     np.ndarray,
                                  y_pred:     np.ndarray,
                                  class_names: list):
    print(f"\n  Classification Report — {model_name}")
    print("─"*60)
    print(classification_report(y_test, y_pred, target_names=class_names,
                                 zero_division=0))


# ── Confusion matrix as DataFrame ─────────────────────────────────────────────
def get_confusion_matrix_df(y_test:      np.ndarray,
                              y_pred:     np.ndarray,
                              class_names: list) -> pd.DataFrame:
    cm = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


# ── 5-Fold cross-validation summary ──────────────────────────────────────────
def cross_val_summary(cv_scores: dict) -> pd.DataFrame:
    """
    Summarise cross-validation results.

    Parameters
    ----------
    cv_scores : dict  name → array of fold scores
    """
    rows = []
    for name, scores in cv_scores.items():
        rows.append({
            'Model':   name,
            'CV Mean': round(np.mean(scores) * 100, 4),
            'CV Std':  round(np.std(scores) * 100, 4),
            'CV Min':  round(np.min(scores) * 100, 4),
            'CV Max':  round(np.max(scores) * 100, 4),
        })
    df = pd.DataFrame(rows).sort_values('CV Mean', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df
