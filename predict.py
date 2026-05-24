"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Quick Predict Script — Standalone single-sample predictor

Usage (after running main.py at least once to train the models):
    python predict.py

Or call predict_crop() directly from another script:
    from predict import predict_crop
    result = predict_crop(N=90, P=42, K=43, temperature=20.9,
                          humidity=82, ph=6.5, rainfall=202.9)
================================================================================
"""

import os, sys, warnings
# Set standard stream encodings to UTF-8 to prevent UnicodeEncodeError in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def predict_crop(N: float          = 90.0,
                 P: float          = 42.0,
                 K: float          = 43.0,
                 temperature: float = 20.9,
                 humidity: float   = 82.0,
                 ph: float         = 6.5,
                 rainfall: float   = 202.9,
                 elevation: float  = 300.0,
                 climate_zone: int = 1,
                 soil_texture: int = 2,
                 growth_days: int  = 120,
                 market_demand: float = 0.75,
                 verbose: bool     = True) -> list:
    """
    Train models and return Top-3 crop recommendations for given parameters.

    Parameters
    ----------
    N, P, K         : Soil macronutrients (kg/ha)
    temperature     : Mean temperature (°C)
    humidity        : Relative humidity (%)
    ph              : Soil pH (3.5 – 9.5)
    rainfall        : Annual rainfall (mm)
    elevation       : Elevation (m) — default 300
    climate_zone    : 0=Tropical, 1=Subtropical, 2=Temperate, 3=Continental, 4=Arid
    soil_texture    : 0=Sandy, 1=Loamy, 2=Silty, 3=Clayey, 4=Peaty
    growth_days     : Expected crop duration (days) — default 120
    market_demand   : Market demand index 0–1 — default 0.75
    verbose         : Print results to console

    Returns
    -------
    list of dicts: [{'rank', 'crop', 'confidence', 'uncertainty'}, ...]
    """
    from data.dataset_loader    import load_dataset, FEATURE_NAMES
    from data.preprocessor      import preprocess
    from models.base_models      import build_base_models, train_base_models
    from models.stacked_ensemble import StackedEnsemble

    # ── Train on synthetic data (fast path) ──────────────────────
    if verbose:
        print("  [ACRS] Loading dataset and training models ...")
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler, le, class_names = preprocess(df)

    base_defs    = build_base_models(len(class_names))
    trained      = train_base_models(base_defs, X_train, y_train)

    top_models   = {k: trained[k] for k in ['Random Forest', 'XGBoost', 'LightGBM']
                    if k in trained}
    ensemble     = StackedEnsemble(top_models, len(class_names))
    ensemble.use_dl = False    # skip deep learning for quick predict
    ensemble.fit(X_train, y_train, X_test, y_test)

    # ── Build input vector ────────────────────────────────────────
    x_raw = np.array([N, P, K, temperature, humidity, ph, rainfall,
                       elevation, climate_zone, soil_texture,
                       growth_days, market_demand], dtype=float)

    mean_p, std_p = ensemble.predict_sample(x_raw, scaler, n_passes=30)
    recs          = ensemble.top_k_crops(mean_p, std_p, class_names, k=3)

    if verbose:
        print("\n  ╔════════════════════════════════════╗")
        print("  ║    TOP-3 CROP RECOMMENDATIONS      ║")
        print("  ╠════════════════════════════════════╣")
        for r in recs:
            bar = '█' * int(r['confidence'] / 5)
            print(f"  ║  #{r['rank']} {r['crop'].upper():<14} "
                  f"{r['confidence']:5.1f}%  {bar:<16}║")
        print("  ╚════════════════════════════════════╝\n")

    return recs


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════╗
║   ACRS — Quick Crop Predictor                ║
╚══════════════════════════════════════════════╝
""")
    # Example 1: Rice-optimal conditions
    print("  Example 1: Rice-optimal soil/climate conditions")
    result = predict_crop(
        N=82, P=45, K=40,
        temperature=23, humidity=82, ph=6.5, rainfall=210,
        elevation=50, climate_zone=0, soil_texture=3,
        growth_days=140, market_demand=0.9
    )

    # Example 2: Dry/arid conditions
    print("\n  Example 2: Arid zone, low rainfall conditions")
    result2 = predict_crop(
        N=20, P=68, K=80,
        temperature=28, humidity=16, ph=7.2, rainfall=75,
        elevation=800, climate_zone=4, soil_texture=0,
        growth_days=85, market_demand=0.6
    )
