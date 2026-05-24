"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Visualization Module

All matplotlib / seaborn plots used in the research pipeline:
  1.  Dataset distribution   — feature histograms + crop class balance
  2.  Correlation heatmap    — feature correlation matrix
  3.  Confusion matrix       — per-model heatmap
  4.  ROC curves             — multi-class OvR
  5.  Model comparison bars  — accuracy / F1 / MCC grouped bar chart
  6.  Training history       — loss & accuracy curves (Keras models)
  7.  Feature importance     — RF/XGB native importance bars
  8.  Calibration curve      — reliability diagram (ECE visualisation)
  9.  Top-3 prediction chart — confidence bar chart for a single query
  10. SMOTE class balance     — before vs. after balancing comparison
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ── Global style ─────────────────────────────────────────────────────────────
PALETTE    = sns.color_palette('tab20', 22)
ACCENT     = '#2ecc71'
DARK_BG    = '#1a1a2e'
LIGHT_TEXT = '#ecf0f1'
sns.set_theme(style='whitegrid', font_scale=1.0)


# ── 1. Dataset Distribution ───────────────────────────────────────────────────
def plot_dataset_distribution(df: pd.DataFrame,
                               feature_names: list,
                               save_path: str | None = None):
    """Plot feature histograms and crop class bar chart."""
    n_feats = len(feature_names)
    cols = 4
    rows = (n_feats + cols - 1) // cols + 1   # +1 row for class distribution

    fig = plt.figure(figsize=(cols * 4, rows * 3))
    fig.suptitle('ACRS — Dataset Distribution Analysis', fontsize=14, fontweight='bold', y=1.01)

    # Feature histograms
    for i, feat in enumerate(feature_names):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(df[feat], bins=30, color=ACCENT, edgecolor='white', alpha=0.8)
        ax.set_title(feat, fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(labelsize=7)

    # Crop class distribution
    ax_cls = fig.add_subplot(rows, 1, rows)
    counts = df['label'].value_counts()
    bars = ax_cls.bar(counts.index, counts.values,
                       color=PALETTE[:len(counts)], edgecolor='white')
    ax_cls.set_title('Crop Class Distribution (Samples per Class)', fontsize=10, fontweight='bold')
    ax_cls.set_xlabel('Crop')
    ax_cls.set_ylabel('Count')
    ax_cls.tick_params(axis='x', rotation=45, labelsize=8)
    for bar in bars:
        ax_cls.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(bar.get_height()), ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Dataset distribution → {save_path}")
    plt.close()


# ── 2. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame,
                              feature_names: list,
                              save_path: str | None = None):
    corr = df[feature_names].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Correlation heatmap → {save_path}")
    plt.close()


# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm_df: pd.DataFrame,
                           model_name: str,
                           save_path:  str | None = None):
    fig, ax = plt.subplots(figsize=(max(8, len(cm_df) * 0.6),
                                     max(6, len(cm_df) * 0.5)))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                linewidths=0.3, ax=ax, annot_kws={'size': 7})
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Confusion matrix ({model_name}) → {save_path}")
    plt.close()


# ── 4. ROC Curves (multi-class OvR) ──────────────────────────────────────────
def plot_roc_curves(y_test:      np.ndarray,
                    proba:       np.ndarray,
                    class_names: list,
                    model_name:  str,
                    save_path:   str | None = None):
    n_classes = len(class_names)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = cycle(plt.cm.tab20.colors)

    auc_scores = []
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, lw=1.2, color=col, alpha=0.7,
                label=f'{cls} (AUC={roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (One-vs-Rest) — {model_name}\n'
                 f'Macro-average AUC = {np.mean(auc_scores):.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] ROC curves ({model_name}) → {save_path}")
    plt.close()


# ── 5. Model Comparison Bar Chart ────────────────────────────────────────────
def plot_model_comparison(results_df: pd.DataFrame,
                           metrics:    list | None = None,
                           save_path:  str | None = None):
    if metrics is None:
        metrics = ['Accuracy', 'F1 (W)', 'MCC']

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 6))
    if n == 1:
        axes = [axes]

    colors = plt.cm.Set2.colors

    for ax, metric in zip(axes, metrics):
        if metric not in results_df.columns:
            continue
        vals  = results_df[metric].astype(float)
        names = results_df['Model']
        bars  = ax.bar(names, vals,
                       color=colors[:len(names)],
                       edgecolor='white', width=0.6)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(vals.max() * 1.15, 1.0))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('ACRS — Model Performance Comparison', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Model comparison → {save_path}")
    plt.close()


# ── 6. Training History (Keras) ───────────────────────────────────────────────
def plot_training_history(history,
                           model_name: str,
                           save_path:  str | None = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training History — {model_name}', fontsize=12, fontweight='bold')

    # Accuracy
    ax1.plot(history.history['accuracy'],     label='Train', color=ACCENT)
    ax1.plot(history.history['val_accuracy'], label='Val',   color='#e74c3c', linestyle='--')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'],     label='Train', color=ACCENT)
    ax2.plot(history.history['val_loss'], label='Val',   color='#e74c3c', linestyle='--')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Training history ({model_name}) → {save_path}")
    plt.close()


# ── 7. Feature Importance (native tree importance) ────────────────────────────
def plot_feature_importance(model,
                              model_name:    str,
                              feature_names: list,
                              save_path:     str | None = None):
    if not hasattr(model, 'feature_importances_'):
        print(f"  [Plot] {model_name} has no feature_importances_, skipping.")
        return

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh([feature_names[i] for i in sorted_idx],
                   importances[sorted_idx],
                   color=['#27ae60' if v > importances.mean() else '#95a5a6'
                          for v in importances[sorted_idx]])
    ax.set_xlabel('Feature Importance (Gini / Gain)')
    ax.set_title(f'Feature Importance — {model_name}', fontsize=12, fontweight='bold')
    ax.axvline(importances.mean(), color='#e74c3c', linestyle='--',
               label=f'Mean = {importances.mean():.4f}')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Feature importance ({model_name}) → {save_path}")
    plt.close()


# ── 8. Calibration Curve (Reliability Diagram) ───────────────────────────────
def plot_calibration_curve(y_test:     np.ndarray,
                            proba:      np.ndarray,
                            model_name: str,
                            n_bins:     int = 10,
                            save_path:  str | None = None):
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correctness = (predictions == y_test).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(correctness[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', lw=1.5)
    ax.bar(bin_confs, bin_accs, width=0.05, alpha=0.5,
           color=ACCENT, edgecolor='white', label='Actual Accuracy')
    ax.plot(bin_confs, bin_accs, 'o-', color='#e74c3c', lw=2, label='Model')
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction of Correct Predictions')
    ax.set_title(f'Reliability Diagram (Calibration) — {model_name}',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Calibration curve ({model_name}) → {save_path}")
    plt.close()


# ── 9. Top-3 Prediction Chart ─────────────────────────────────────────────────
def plot_top3_prediction(recommendations: list,
                          input_values:    dict,
                          save_path:       str | None = None):
    """
    Visualise Top-3 crop recommendations with confidence and uncertainty.

    Parameters
    ----------
    recommendations : list of dicts from StackedEnsemble.top_k_crops()
    input_values    : dict of feature_name → value (for annotation)
    """
    crops = [r['crop'].capitalize() for r in recommendations]
    confs = [r['confidence'] for r in recommendations]
    stds  = [r['uncertainty'] for r in recommendations]

    fig  = plt.figure(figsize=(12, 5))
    gs   = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=fig)
    ax1  = fig.add_subplot(gs[0])
    ax2  = fig.add_subplot(gs[1])

    # ── Confidence bar chart ──────────────────────────────────────
    bar_colors = ['#f39c12', '#95a5a6', '#cd7f32'][:len(crops)]
    bar_colors[0] = '#27ae60'   # top pick always green
    bars = ax1.barh(crops[::-1], confs[::-1],
                    xerr=stds[::-1], color=bar_colors[::-1],
                    edgecolor='white', height=0.5, capsize=5,
                    error_kw={'ecolor': '#2c3e50', 'lw': 1.5})
    ax1.set_xlabel('Confidence (%)')
    ax1.set_title('Top-3 Crop Recommendations', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)
    for bar, conf, std in zip(bars, confs[::-1], stds[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{conf:.1f}% ± {std:.2f}%',
                 va='center', fontsize=9, fontweight='bold')

    # ── Input feature table ───────────────────────────────────────
    ax2.axis('off')
    feat_rows = [[k, f'{v:.2f}'] for k, v in input_values.items()]
    tbl = ax2.table(cellText=feat_rows,
                    colLabels=['Feature', 'Value'],
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#ecf0f1')
    ax2.set_title('Input Parameters', fontsize=10, fontweight='bold')

    fig.suptitle('ACRS Prediction Result', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Top-3 prediction → {save_path}")
    plt.close()


# ── 10. SMOTE Class Balance Comparison ───────────────────────────────────────
def plot_smote_comparison(y_before:    np.ndarray,
                           y_after:     np.ndarray,
                           class_names: list,
                           save_path:   str | None = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SMOTE-Tomek Class Balancing Effect', fontsize=13, fontweight='bold')

    counts_before = pd.Series(y_before).value_counts().sort_index()
    counts_after  = pd.Series(y_after).value_counts().sort_index()
    labels = [class_names[i] for i in counts_before.index]

    def _bar(ax, counts, title, color):
        ax.bar(labels, counts.values, color=color, edgecolor='white', width=0.6)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.set_ylabel('Sample Count')
        ax.grid(axis='y', alpha=0.3)

    _bar(ax1, counts_before, f'Before SMOTE-Tomek  (n={len(y_before)})', '#e74c3c')
    _bar(ax2, counts_after,  f'After  SMOTE-Tomek  (n={len(y_after)})',  '#27ae60')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] SMOTE comparison → {save_path}")
    plt.close()
