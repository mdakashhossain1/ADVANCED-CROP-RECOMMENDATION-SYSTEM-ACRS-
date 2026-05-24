"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Explainability Module — SHAP

Implements:
  - Global SHAP: Summary beeswarm plot + mean |SHAP| bar chart
  - Local  SHAP: Waterfall / force plot for a single prediction
  - SHAP interaction values for feature pair analysis

Using TreeExplainer (fast, for RF/XGB/LGBM) with KernelExplainer fallback.
================================================================================
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script execution
from data.dataset_loader import FEATURE_NAMES


def get_shap_explainer(model, X_background: np.ndarray):
    """
    Build the appropriate SHAP explainer for the model type.

    TreeExplainer → sklearn tree-based, XGBoost, LightGBM
    DeepExplainer  → Keras/TF models
    KernelExplainer → all others (slower, model-agnostic)
    """
    model_type = type(model).__name__

    if model_type in ('RandomForestClassifier', 'XGBClassifier',
                      'LGBMClassifier', 'DecisionTreeClassifier',
                      'GradientBoostingClassifier'):
        explainer = shap.TreeExplainer(model)
    else:
        # Use a small background summary for KernelExplainer
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer  = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(explainer, X: np.ndarray, max_samples: int = 200):
    """
    Compute SHAP values for up to `max_samples` instances.

    Returns
    -------
    shap_values : list[np.ndarray] of shape (n_samples, n_features) per class
                  OR np.ndarray depending on explainer type
    """
    X_subset = X[:max_samples]
    try:
        sv = explainer.shap_values(X_subset, check_additivity=False)
    except TypeError:
        sv = explainer.shap_values(X_subset)
    return sv, X_subset


def plot_shap_summary(shap_values,
                       X_subset: np.ndarray,
                       class_names: list,
                       model_name:  str,
                       save_path:   str | None = None):
    """
    Plot SHAP beeswarm summary for the top predicted class.

    The beeswarm plot shows how much each feature pushes predictions
    higher or lower, colour-coded by feature value (red=high, blue=low).
    """
    # Use class-0 SHAP values for multi-output, or the array itself
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'SHAP Global Explanation — {model_name}',
                 fontsize=14, fontweight='bold')

    # ── Beeswarm summary ──────────────────────────────────────────
    plt.sca(axes[0])
    shap.summary_plot(sv, X_subset, feature_names=FEATURE_NAMES,
                      show=False, plot_size=None)
    axes[0].set_title('Beeswarm Summary Plot', fontsize=11)

    # ── Mean |SHAP| bar chart ─────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    # Ensure mean_abs is 1D
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.flatten()
    
    # Use actual feature names from data if available, otherwise use FEATURE_NAMES
    n_features = len(mean_abs)
    if hasattr(X_subset, 'columns'):
        feature_names = list(X_subset.columns)
    elif len(FEATURE_NAMES) >= n_features:
        feature_names = FEATURE_NAMES[:n_features]
    else:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    sorted_idx = np.argsort(mean_abs)
    # Convert to Python list and ensure integer indices
    sorted_idx = [int(i) for i in sorted_idx]
    # Ensure we have a simple list of feature names and values
    feature_names_sorted = [feature_names[i] for i in sorted_idx]
    mean_abs_sorted = [float(mean_abs[i]) for i in sorted_idx]
    axes[1].barh(
        feature_names_sorted,
        mean_abs_sorted,
        color=['#2ecc71' if v > np.mean(mean_abs_sorted) else '#95a5a6'
               for v in mean_abs_sorted]
    )
    axes[1].set_xlabel('Mean |SHAP value|')
    axes[1].set_title('Feature Importance (Mean |SHAP|)', fontsize=11)
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SHAP] Saved summary plot → {save_path}")
    plt.close()


def plot_shap_local(explainer,
                     x_instance:  np.ndarray,
                     class_names: list,
                     pred_class: int,
                     model_name:  str,
                     save_path:   str | None = None,
                     feature_names: list | None = None):
    """
    Plot SHAP waterfall chart for a single prediction instance.

    Shows which features increased or decreased confidence in the
    predicted crop class.
    """
    x_2d = x_instance.reshape(1, -1)
    try:
        sv   = explainer.shap_values(x_2d, check_additivity=False)
    except TypeError:
        sv   = explainer.shap_values(x_2d)

    # Extract SHAP vector for the predicted class
    if isinstance(sv, list):
        sv_class = sv[pred_class][0]
        expected = explainer.expected_value[pred_class] \
                   if hasattr(explainer.expected_value, '__iter__') \
                   else explainer.expected_value
    else:
        sv_class = sv[0]
        expected = explainer.expected_value

    # Build waterfall data
    # Ensure sv_class and x_instance are 1D arrays and convert to scalars
    sv_class = np.asarray(sv_class).flatten()
    x_instance = np.asarray(x_instance).flatten()
    
    # Use provided feature names or fallback to FEATURE_NAMES
    n_features = len(sv_class)
    if feature_names is not None and len(feature_names) >= n_features:
        feature_names = feature_names[:n_features]
    elif len(FEATURE_NAMES) >= n_features:
        feature_names = FEATURE_NAMES[:n_features]
    else:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    pairs = sorted(zip(feature_names, sv_class, x_instance),
                   key=lambda t: abs(float(t[1])), reverse=True)
    feats, vals, raw_vals = zip(*pairs)
    colours = ['#27ae60' if v > 0 else '#e74c3c' for v in vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feats, vals, color=colours, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value (impact on model output)')
    ax.set_title(
        f'SHAP Local Explanation — Predicted: {class_names[pred_class]}\n'
        f'Model: {model_name}',
        fontsize=12, fontweight='bold'
    )

    # Annotate with raw feature values
    for bar, feat, rv in zip(bars, feats, raw_vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'  {rv:.2f}', va='center', fontsize=8, color='#2c3e50')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SHAP] Saved local plot → {save_path}")
    plt.close()


def shap_feature_importance_table(shap_values, class_idx: int = 0, feature_names: list | None = None) -> pd.DataFrame:
    """
    Return a sorted DataFrame of mean absolute SHAP values.
    """
    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(sv).mean(axis=0)
    
    # Ensure mean_abs is 1D
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.flatten()
    
    # Use provided feature names or fallback to FEATURE_NAMES
    n_features = len(mean_abs)
    if feature_names is not None and len(feature_names) >= n_features:
        feature_names = feature_names[:n_features]
    elif len(FEATURE_NAMES) >= n_features:
        feature_names = FEATURE_NAMES[:n_features]
    else:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame({
        'Feature':          feature_names,
        'Mean |SHAP|':      mean_abs.round(6),
        'Description':      [f'Feature {i+1}' for i in range(len(feature_names))],
    }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df
