"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Explainability Module — DiCE (Diverse Counterfactual Explanations)

DiCE answers: "What minimal changes to input features would change the
              crop recommendation to a different crop?"

Example output:
  "If you reduce Nitrogen from 90 to 65 kg/ha and increase pH from 5.5
   to 6.2, the recommendation changes from Rice to Wheat."

Uses DiCE-ML library with a scikit-learn model backend.
================================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    print("[DiCE] dice-ml not installed. DiCE explanations will be skipped.")

from data.dataset_loader import FEATURE_NAMES


def build_dice_explainer(model,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          class_names: list):
    """
    Build a DiCE explainer backed by the given model.

    Returns
    -------
    exp : dice_ml.Dice  or None if DiCE not available
    """
    if not DICE_AVAILABLE:
        return None

    # Build training DataFrame for DiCE
    df_train = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    df_train['label'] = [class_names[y] for y in y_train]

    data   = dice_ml.Data(dataframe=df_train,
                           continuous_features=FEATURE_NAMES,
                           outcome_name='label')
    m      = dice_ml.Model(model=model, backend='sklearn')
    exp    = dice_ml.Dice(data, m, method='random')
    return exp


def generate_counterfactuals(explainer,
                               x_instance:      np.ndarray,
                               desired_class:   str,
                               n_cfs:           int = 3,
                               proximity_weight: float = 0.5,
                               diversity_weight: float = 1.0) -> pd.DataFrame | None:
    """
    Generate diverse counterfactual explanations for one instance.

    Parameters
    ----------
    explainer       : DiCE explainer object
    x_instance      : 1-D scaled feature array (the query point)
    desired_class   : Target crop class string (alternative recommendation)
    n_cfs           : Number of diverse counterfactuals to generate
    proximity_weight: Weight for keeping CFs close to the original
    diversity_weight: Weight for CFs being diverse among themselves

    Returns
    -------
    pd.DataFrame of counterfactual instances, or None on failure
    """
    if explainer is None:
        return None

    try:
        query_df = pd.DataFrame([x_instance], columns=FEATURE_NAMES)
        dice_exp = explainer.generate_counterfactuals(
            query_df,
            total_CFs=n_cfs,
            desired_class=desired_class,
            proximity_weight=proximity_weight,
            diversity_weight=diversity_weight,
            verbose=False
        )
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        return cf_df
    except Exception as e:
        print(f"  [DiCE] Counterfactual generation failed: {e}")
        return None


def format_counterfactual_table(original:  np.ndarray,
                                  cf_df:     pd.DataFrame,
                                  orig_crop: str,
                                  cf_crop:   str) -> pd.DataFrame:
    """
    Build a human-readable diff table comparing original vs. counterfactuals.

    Columns: Feature | Original | CF1 | CF2 | CF3 | Change Direction
    """
    if cf_df is None or cf_df.empty:
        return pd.DataFrame({'Message': ['DiCE counterfactuals not available']})

    rows = []
    for feat, orig_val in zip(FEATURE_NAMES, original):
        row = {'Feature': feat, f'Original ({orig_crop})': round(orig_val, 3)}

        changed = False
        for i, (_, cf_row) in enumerate(cf_df.iterrows()):
            if feat in cf_row:
                cf_val = round(cf_row[feat], 3)
                row[f'CF-{i+1} ({cf_crop})'] = cf_val
                if abs(cf_val - orig_val) > 0.01:
                    changed = True

        row['Changed?'] = '⬆⬇' if changed else '─'
        rows.append(row)

    return pd.DataFrame(rows)


def print_dice_summary(original: np.ndarray,
                         cf_df:     pd.DataFrame,
                         orig_crop: str,
                         cf_crop:   str):
    """Print a concise what-if summary to stdout."""
    if cf_df is None or cf_df.empty:
        print("  [DiCE] No counterfactuals generated.")
        return

    print(f"\n  DiCE Counterfactual Analysis")
    print(f"  Original prediction : {orig_crop}")
    print(f"  Target class        : {cf_crop}")
    print("  ─" * 30)
    print(f"  {'Feature':<20} {'Original':>10}  {'CF-1':>10}  {'CF-2':>10}  Change")
    print("  ─" * 30)

    cf_rows = list(cf_df.iterrows())
    for feat, orig_val in zip(FEATURE_NAMES, original):
        cf1 = round(cf_rows[0][1][feat], 3) if len(cf_rows) > 0 and feat in cf_rows[0][1] else '—'
        cf2 = round(cf_rows[1][1][feat], 3) if len(cf_rows) > 1 and feat in cf_rows[1][1] else '—'
        delta = ''
        if isinstance(cf1, float) and isinstance(orig_val, float):
            diff = cf1 - orig_val
            if abs(diff) > 0.01:
                delta = f'{"↑" if diff > 0 else "↓"} {abs(diff):.2f}'
        print(f"  {feat:<20} {orig_val:>10.3f}  {str(cf1):>10}  {str(cf2):>10}  {delta}")
    print()
