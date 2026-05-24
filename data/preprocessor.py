"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Preprocessing Module

Pipeline:
  1. Label encoding  (LabelEncoder)
  2. Feature scaling (StandardScaler)
  3. Train/test split (80/20, stratified)
  4. Class imbalance correction (SMOTE-Tomek)
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from data.dataset_loader import FEATURE_NAMES


def preprocess(df: pd.DataFrame,
               test_size: float = 0.20,
               random_state: int = 42,
               apply_smote: bool = True):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df           : Raw DataFrame (output of load_dataset)
    test_size    : Fraction of data for test set
    random_state : Reproducibility seed
    apply_smote  : Whether to apply SMOTE-Tomek class balancing

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (scaled)
    scaler       : Fitted StandardScaler
    le           : Fitted LabelEncoder
    class_names  : List[str] of crop names in encoded order
    """
    print("\n" + "="*60)
    print("  PREPROCESSING PIPELINE")
    print("="*60)

    # ── 1. Separate features / target ─────────────────────────────
    X = df[FEATURE_NAMES].values
    y_raw = df['label'].values

    # ── 2. Label encoding ─────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    print(f"  Classes      : {len(class_names)} crops")
    print(f"  Features     : {len(FEATURE_NAMES)}")
    print(f"  Total samples: {len(X)}")

    # ── 3. Train / test split (stratified) ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train size   : {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # ── 4. SMOTE-Tomek oversampling on training set ────────────────
    if apply_smote:
        print("  Applying SMOTE-Tomek class balancing ...", end=" ", flush=True)
        smt = SMOTETomek(random_state=random_state)
        X_train, y_train = smt.fit_resample(X_train, y_train)
        print(f"done → {X_train.shape[0]} training samples after resampling")

    # ── 5. Feature scaling ────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  StandardScaler applied (mean=0, std=1)")
    print("="*60 + "\n")

    return X_train, X_test, y_train, y_test, scaler, le, class_names


def inverse_transform_label(le: LabelEncoder, encoded: np.ndarray) -> np.ndarray:
    """Convert integer labels back to crop names."""
    return le.inverse_transform(encoded)
