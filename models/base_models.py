"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Base ML Models Module

Models trained:
  - Random Forest (RF)
  - XGBoost (XGB)
  - LightGBM (LGBM)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (NB)
  - Decision Tree (DT)

All models support scikit-learn API (fit / predict / predict_proba).
================================================================================
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb


def build_base_models(n_classes: int, random_state: int = 42) -> dict:
    """
    Instantiate all base classifier models with optimised hyperparameters.

    Returns
    -------
    dict mapping model name → unfitted estimator
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=random_state,
            verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=random_state
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            metric='minkowski',
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=random_state
        ),
    }
    return models


def train_base_models(models: dict,
                      X_train: np.ndarray,
                      y_train: np.ndarray) -> dict:
    """
    Train all base models and record training time.

    Returns
    -------
    dict mapping model name → {'model': fitted_model, 'train_time': float}
    """
    print("\n" + "="*60)
    print("  TRAINING BASE MODELS")
    print("="*60)

    trained = {}
    for name, model in models.items():
        print(f"  Training {name:<20}", end=" ", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        trained[name] = {'model': model, 'train_time': elapsed}
        print(f"✓  ({elapsed:.2f}s)")

    print("="*60 + "\n")
    return trained
