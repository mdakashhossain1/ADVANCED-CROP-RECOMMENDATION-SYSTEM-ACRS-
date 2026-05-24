"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Two-Tier Stacked Ensemble Module

Architecture (PRD Table 5):

  Tier 1 — Base Learners:
    ┌─────────────┬───────────┬──────────┬───────────┐
    │ Random      │ XGBoost   │ LightGBM │ CNN-BiLSTM│
    │ Forest      │           │          │           │
    └──────┬──────┴─────┬─────┴────┬─────┴─────┬─────┘
           │             │          │            │
           └─────────────┴──────────┴────────────┘
                         │ Probability stacks
  Tier 2 — Meta-Learner:
                    ┌────▼────────┐
                    │ MLP         │
                    │ (128→64→22) │
                    └────┬────────┘
                         │
                   Final prediction (Top-3 + confidence)
================================================================================
"""

import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    from models.deep_models import (build_mlp, build_cnn_bilstm,
                                     train_deep_model, mc_dropout_predict)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False



class StackedEnsemble:
    """
    Two-tier stacked ensemble combining classical ML + deep learning.

    Parameters
    ----------
    base_models  : dict  name → fitted sklearn estimator (RF, XGB, LGBM, ...)
    n_classes    : int   number of crop classes
    random_state : int
    """

    def __init__(self, base_models: dict, n_classes: int, random_state: int = 42):
        self.base_models   = base_models   # already trained base classifiers
        self.n_classes     = n_classes
        self.random_state  = random_state
        self.meta_learner_ = None          # MLP meta-learner (fitted)
        self.cnn_bilstm_   = None          # CNN-BiLSTM (fitted)
        self.use_dl        = TF_AVAILABLE  # set False if TF unavailable

    # ── 1. Build meta-features via cross-val OOF predictions ───────────────
    def _make_oof_meta_features(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 n_folds: int = 5) -> np.ndarray:
        """
        Generate Out-Of-Fold (OOF) probability predictions from base models.
        This avoids data leakage when training the meta-learner.

        Returns shape: (n_train, n_base_models * n_classes)
        """
        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=self.random_state)
        n      = X_train.shape[0]
        n_base = len(self.base_models)
        oof    = np.zeros((n, n_base * self.n_classes))

        print(f"  Building OOF meta-features ({n_folds}-fold CV) ...", flush=True)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            for m_idx, (name, entry) in enumerate(self.base_models.items()):
                model = entry['model']
                # Re-fit on fold
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_val)   # (n_val, n_classes)
                col_s = m_idx * self.n_classes
                col_e = col_s + self.n_classes
                oof[val_idx, col_s:col_e] = proba

            print(f"    Fold {fold_idx+1}/{n_folds} done", flush=True)

        return oof

    # ── 2. Build test meta-features from full-data base model predictions ──
    def _make_test_meta_features(self, X_test: np.ndarray) -> np.ndarray:
        n_base = len(self.base_models)
        meta   = np.zeros((X_test.shape[0], n_base * self.n_classes))
        for m_idx, (name, entry) in enumerate(self.base_models.items()):
            proba = entry['model'].predict_proba(X_test)
            col_s = m_idx * self.n_classes
            meta[:, col_s:col_s + self.n_classes] = proba
        return meta

    # ── 3. Fit CNN-BiLSTM and get its OOF probabilities ───────────────────
    def _fit_cnn_bilstm(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test:  np.ndarray) -> tuple:
        """Trains CNN-BiLSTM; returns (oof_proba, test_proba)."""
        print("  Training CNN-BiLSTM ...", flush=True)
        skf     = StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=self.random_state)
        n       = X_train.shape[0]
        oof_dl  = np.zeros((n, self.n_classes))
        test_dl = np.zeros((X_test.shape[0], self.n_classes))

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            tf.random.set_seed(self.random_state)
            model = build_cnn_bilstm(X_train.shape[1], self.n_classes)
            train_deep_model(
                model,
                X_train[tr_idx], y_train[tr_idx],
                X_train[val_idx], y_train[val_idx],
                epochs=60, batch_size=64, verbose=0
            )
            oof_dl[val_idx] = model.predict(X_train[val_idx], verbose=0)
            test_dl        += model.predict(X_test, verbose=0) / 3
            print(f"    CNN-BiLSTM fold {fold_idx+1}/3 done", flush=True)

        self.cnn_bilstm_ = model   # keep last fold model for inference
        return oof_dl, test_dl

    # ── 4. Public: fit the full stacked ensemble ───────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test:  np.ndarray, y_test:  np.ndarray):
        """
        Fit the two-tier stacked ensemble end-to-end.

        Returns
        -------
        meta_train : OOF meta-features (for meta-learner training)
        meta_test  : Test meta-features
        """
        print("\n" + "="*60)
        print("  TWO-TIER STACKED ENSEMBLE TRAINING")
        print("="*60)

        # ── Tier 1a: OOF meta-features from base ML models ────────
        t0 = time.perf_counter()
        ml_oof   = self._make_oof_meta_features(X_train, y_train)
        ml_test  = self._make_test_meta_features(X_test)
        print(f"  ML OOF meta-features: {ml_oof.shape}   ({time.perf_counter()-t0:.1f}s)")

        # ── Tier 1b: CNN-BiLSTM OOF ───────────────────────────────
        if self.use_dl and TF_AVAILABLE:
            try:
                dl_oof, dl_test = self._fit_cnn_bilstm(X_train, y_train, X_test)
                meta_train = np.hstack([ml_oof,  dl_oof])
                meta_test  = np.hstack([ml_test, dl_test])
                print(f"  Combined meta-features: {meta_train.shape}")
            except Exception as e:
                print(f"  [WARNING] CNN-BiLSTM failed ({e}), using ML-only meta-features")
                self.use_dl = False
                meta_train = ml_oof
                meta_test  = ml_test
        else:
            self.use_dl = False
            meta_train = ml_oof
            meta_test  = ml_test

        # ── Tier 2: MLP meta-learner ───────────────────────────────
        print("  Training MLP meta-learner ...", flush=True)
        if TF_AVAILABLE:
            tf.random.set_seed(self.random_state)
            self.meta_learner_ = build_mlp(
                input_dim=meta_train.shape[1],
                n_classes=self.n_classes,
                hidden_units=(256, 128, 64),
                dropout_rate=0.3,
                mc_dropout=True
            )
            split = int(0.85 * len(meta_train))
            train_deep_model(
                self.meta_learner_,
                meta_train[:split], y_train[:split],
                meta_train[split:], y_train[split:],
                epochs=80, batch_size=64, verbose=0
            )
        else:
            self.meta_learner_ = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=80,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.15
            )
            self.meta_learner_.fit(meta_train, y_train)
        print("  MLP meta-learner trained ✓")

        self.meta_test_ = meta_test   # cache for predict
        print("="*60 + "\n")
        return meta_train, meta_test

    # ── 5. Predict on test set (with MC-Dropout uncertainty) ──────────────
    def predict_proba_test(self, n_passes: int = 50) -> tuple:
        """
        MC-Dropout inference on cached test meta-features.

        Returns
        -------
        mean_proba : (n_test, n_classes)
        std_proba  : (n_test, n_classes)  — uncertainty
        """
        if TF_AVAILABLE:
            return mc_dropout_predict(self.meta_learner_, self.meta_test_, n_passes)
        else:
            mean_proba = self.meta_learner_.predict_proba(self.meta_test_)
            n_base = len(self.base_models)
            reshaped_meta = self.meta_test_.reshape(-1, n_base, self.n_classes)
            std_proba = reshaped_meta.std(axis=1)
            return mean_proba, std_proba

    def predict_sample(self, x: np.ndarray, scaler,
                       n_passes: int = 50) -> tuple:
        """
        Predict a single raw (unscaled) input sample.

        Returns
        -------
        mean_proba : (n_classes,)
        std_proba  : (n_classes,)
        """
        x_scaled = scaler.transform(x.reshape(1, -1))

        # Build meta-features from base models
        ml_parts = []
        for name, entry in self.base_models.items():
            ml_parts.append(entry['model'].predict_proba(x_scaled))
        meta_x = np.hstack(ml_parts)

        if self.use_dl and self.cnn_bilstm_ is not None and TF_AVAILABLE:
            dl_part = self.cnn_bilstm_.predict(x_scaled, verbose=0)
            meta_x  = np.hstack([meta_x, dl_part])

        if TF_AVAILABLE:
            mean_p, std_p = mc_dropout_predict(self.meta_learner_, meta_x, n_passes)
            return mean_p[0], std_p[0]
        else:
            mean_p = self.meta_learner_.predict_proba(meta_x)
            n_base = len(self.base_models)
            reshaped_meta = meta_x.reshape(1, n_base, self.n_classes)
            std_p = reshaped_meta.std(axis=1)
            return mean_p[0], std_p[0]

    # ── 6. Top-K recommendation ────────────────────────────────────────────
    def top_k_crops(self, proba: np.ndarray, std: np.ndarray,
                    class_names: list, k: int = 3) -> list:
        """
        Return Top-K crop recommendations.

        Returns
        -------
        list of dicts: [{'crop': str, 'confidence': float, 'uncertainty': float}, ...]
        """
        top_idx = np.argsort(proba)[::-1][:k]
        return [
            {
                'rank':        i + 1,
                'crop':        class_names[idx],
                'confidence':  round(float(proba[idx]) * 100, 2),
                'uncertainty': round(float(std[idx]) * 100, 4),
            }
            for i, idx in enumerate(top_idx)
        ]
