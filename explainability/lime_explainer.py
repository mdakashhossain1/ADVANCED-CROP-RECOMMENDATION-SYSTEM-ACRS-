"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Explainability Module — LIME

LIME (Local Interpretable Model-Agnostic Explanations):
  Perturbs the input around a single instance and fits a simple linear model
  to explain the decision boundary locally.

Output:
  - Bar chart of per-feature positive/negative contributions
  - Tabular explanation for reporting
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

from data.dataset_loader import FEATURE_NAMES


def build_lime_explainer(X_train: np.ndarray,
                          class_names: list,
                          mode: str = 'classification') -> lime.lime_tabular.LimeTabularExplainer:
    """
    Build and return a LimeTabularExplainer fitted on the training distribution.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=FEATURE_NAMES,
        class_names=class_names,
        mode=mode,
        discretize_continuous=True,
        random_state=42
    )
    return explainer


def explain_lime(explainer:       lime.lime_tabular.LimeTabularExplainer,
                  predict_fn,
                  x_instance:      np.ndarray,
                  top_labels:      int = 1,
                  num_features:    int = 12,
                  num_samples:     int = 3000) -> lime.explanation.Explanation:
    """
    Generate LIME explanation for a single instance.

    Parameters
    ----------
    explainer    : LimeTabularExplainer
    predict_fn   : Callable(X) → probability array (n_samples, n_classes)
    x_instance   : 1-D array of raw (scaled) feature values
    top_labels   : How many top classes to explain
    num_features : Max features in the explanation
    num_samples  : Perturbation samples (higher = more accurate, slower)

    Returns
    -------
    lime.explanation.Explanation object
    """
    explanation = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=predict_fn,
        top_labels=top_labels,
        num_features=num_features,
        num_samples=num_samples
    )
    return explanation


def plot_lime_explanation(explanation,
                           class_idx:   int,
                           class_names: list,
                           model_name:  str,
                           save_path:   str | None = None):
    """
    Plot a horizontal bar chart of LIME feature contributions.

    Green bars = positive contribution (towards the predicted class)
    Red bars   = negative contribution (against the predicted class)
    """
    exp_list = explanation.as_list(label=class_idx)
    if not exp_list:
        print("  [LIME] No explanation data for this class index.")
        return

    features = [e[0] for e in exp_list]
    weights  = [e[1] for e in exp_list]
    colours  = ['#27ae60' if w > 0 else '#e74c3c' for w in weights]

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.5 + 1)))
    bars = ax.barh(features, weights, color=colours, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LIME Weight (feature contribution)')
    ax.set_title(
        f'LIME Local Explanation — Predicted: {class_names[class_idx]}\n'
        f'Model: {model_name}',
        fontsize=12, fontweight='bold'
    )

    # Value annotations
    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_width() + (0.0005 if w >= 0 else -0.0005),
            bar.get_y() + bar.get_height() / 2,
            f'{w:.4f}',
            va='center',
            ha='left' if w >= 0 else 'right',
            fontsize=8
        )

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [LIME] Saved explanation plot → {save_path}")
    plt.close()


def lime_explanation_table(explanation, class_idx: int) -> pd.DataFrame:
    """
    Return LIME explanation as a sorted DataFrame for tabular reporting.
    """
    exp_list = explanation.as_list(label=class_idx)
    df = pd.DataFrame(exp_list, columns=['Feature Condition', 'Weight'])
    df['Direction'] = df['Weight'].apply(lambda w: '✅ Supports' if w > 0 else '❌ Against')
    df = df.sort_values('Weight', key=abs, ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df
