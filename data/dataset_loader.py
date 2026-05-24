"""
================================================================================
ADVANCED CROP RECOMMENDATION SYSTEM (ACRS)
Dataset Loader Module

Loads the Kaggle Crop Recommendation Dataset or generates a realistic
synthetic dataset with the same statistical properties.

Features (12):
    N, P, K         - Soil macronutrients (kg/ha)
    temperature     - Mean temperature (°C)
    humidity        - Relative humidity (%)
    ph              - Soil pH (0-14)
    rainfall        - Annual rainfall (mm)
    elevation       - Elevation above sea level (m)
    climate_zone    - Encoded climate zone (0-4)
    soil_texture    - Encoded soil texture class (0-4)
    growth_days     - Days to maturity
    market_demand   - Market demand index (0-1)

Target: 22 crop classes (label)
================================================================================
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Crop definitions with realistic agronomic ranges ─────────────────────────
CROPS = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton',
    'jute', 'coffee'
]

CROP_PROFILES = {
    #  crop          N_mean  N_std  P_mean  P_std  K_mean  K_std  temp_m temp_s hum_m  hum_s  ph_m  ph_s  rain_m  rain_s
    'rice':         (80,15,  45,10,  40,8,   23,2,  82,5,  6.5,0.4, 200,30),
    'maize':        (70,15,  48,10,  45,8,   22,3,  65,8,  5.8,0.5, 80,15),
    'chickpea':     (40,10,  68,12,  80,12,  18,3,  16,5,  7.2,0.4, 75,10),
    'kidneybeans':  (20,8,   68,12,  78,12,  19,3,  18,5,  7.0,0.4, 100,15),
    'pigeonpeas':   (20,8,   68,12,  78,12,  26,3,  48,8,  6.0,0.5, 150,20),
    'mothbeans':    (21,8,   48,10,  24,6,   28,3,  53,8,  7.0,0.4, 50,10),
    'mungbean':     (20,8,   48,10,  20,6,   28,3,  85,8,  6.5,0.4, 55,10),
    'blackgram':    (40,10,  68,12,  38,8,   29,3,  65,8,  7.2,0.4, 65,10),
    'lentil':       (18,6,   68,12,  19,6,   24,3,  64,8,  7.0,0.4, 47,8),
    'pomegranate':  (18,6,   18,6,   40,8,   21,3,  90,8,  6.7,0.5, 108,15),
    'banana':       (100,20, 82,15,  50,10,  27,2,  80,8,  6.0,0.4, 105,15),
    'mango':        (20,8,   27,8,   30,8,   31,3,  50,8,  6.0,0.5, 94,15),
    'grapes':       (20,8,   125,20, 200,30, 24,3,  82,8,  6.0,0.5, 70,10),
    'watermelon':   (99,15,  18,6,   50,10,  25,3,  85,8,  6.5,0.4, 50,10),
    'muskmelon':    (100,15, 18,6,   50,10,  28,3,  92,8,  6.5,0.4, 25,5),
    'apple':        (21,8,   134,20, 200,30, 21,3,  92,8,  5.8,0.4, 110,20),
    'orange':       (20,8,   10,4,   10,4,   22,3,  92,8,  7.0,0.4, 110,20),
    'papaya':       (50,10,  60,10,  200,30, 34,3,  92,8,  6.5,0.4, 145,20),
    'coconut':      (22,8,   16,6,   30,8,   27,3,  95,5,  6.0,0.4, 150,20),
    'cotton':       (118,20, 46,10,  20,6,   24,3,  80,8,  7.0,0.4, 80,15),
    'jute':         (78,15,  46,10,  40,8,   25,3,  80,8,  6.5,0.4, 175,25),
    'coffee':       (101,15, 28,8,   30,8,   25,3,  58,8,  6.5,0.4, 150,20),
}

SAMPLES_PER_CROP = 100   # 100 x 22 = 2200 samples (matches Kaggle dataset size)


def generate_synthetic_dataset(n_per_crop: int = SAMPLES_PER_CROP,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic crop recommendation dataset with realistic
    agronomic parameter distributions for 22 crop classes.

    Returns
    -------
    pd.DataFrame  shape (n_per_crop * 22, 13)
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for crop, profile in CROP_PROFILES.items():
        (N_m, N_s, P_m, P_s, K_m, K_s,
         T_m, T_s, H_m, H_s, pH_m, pH_s, R_m, R_s) = profile

        N    = rng.normal(N_m,   N_s,   n_per_crop).clip(0, 200)
        P    = rng.normal(P_m,   P_s,   n_per_crop).clip(0, 200)
        K    = rng.normal(K_m,   K_s,   n_per_crop).clip(0, 250)
        temp = rng.normal(T_m,   T_s,   n_per_crop).clip(8, 45)
        hum  = rng.normal(H_m,   H_s,   n_per_crop).clip(10, 100)
        ph   = rng.normal(pH_m,  pH_s,  n_per_crop).clip(3.5, 9.5)
        rain = rng.normal(R_m,   R_s,   n_per_crop).clip(20, 300)

        # Extended features (PRD Tier-1)
        elevation    = rng.uniform(0, 2500, n_per_crop).round(1)
        climate_zone = rng.integers(0, 5, n_per_crop)   # 0=Tropical … 4=Arid
        soil_texture = rng.integers(0, 5, n_per_crop)   # 0=Sandy … 4=Clay
        growth_days  = rng.integers(60, 200, n_per_crop)
        market_demand = rng.uniform(0.1, 1.0, n_per_crop).round(3)

        for i in range(n_per_crop):
            rows.append({
                'N':            round(N[i], 2),
                'P':            round(P[i], 2),
                'K':            round(K[i], 2),
                'temperature':  round(temp[i], 2),
                'humidity':     round(hum[i], 2),
                'ph':           round(ph[i], 2),
                'rainfall':     round(rain[i], 2),
                'elevation':    elevation[i],
                'climate_zone': int(climate_zone[i]),
                'soil_texture': int(soil_texture[i]),
                'growth_days':  int(growth_days[i]),
                'market_demand': market_demand[i],
                'label':        crop,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[DatasetLoader] Generated synthetic dataset: {df.shape[0]} samples, "
          f"{df['label'].nunique()} crops")
    return df


def load_dataset(csv_path: str | None = None,
                 random_state: int = 42) -> pd.DataFrame:
    """
    Load crop recommendation dataset.

    Priority:
      1. User-supplied CSV path (Kaggle Crop_recommendation.csv or custom CRD/CRD1 datasets)
      2. Synthetic dataset generation

    Parameters
    ----------
    csv_path    : path to CSV (optional)
    random_state: reproducibility seed

    Returns
    -------
    pd.DataFrame
    """
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # Clean column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        print(f"[DatasetLoader] Loaded from CSV: {df.shape[0]} samples, {df.shape[1]} columns")

        # Rename standard columns if present for backward compatibility
        rename_dict = {}
        if 'Recommended_Crop' in df.columns:
            rename_dict['Recommended_Crop'] = 'label'
        if 'Soil_pH' in df.columns:
            rename_dict['Soil_pH'] = 'ph'
        if 'Altitude' in df.columns:
            rename_dict['Altitude'] = 'elevation'
        if 'Soil_Type' in df.columns:
            rename_dict['Soil_Type'] = 'soil_texture'
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"[DatasetLoader] Renamed columns for compatibility: {rename_dict}")

        # Check for label column
        if 'label' not in df.columns:
            # Fallback: if target label not found, take the last column as target 'label'
            last_col = df.columns[-1]
            df = df.rename(columns={last_col: 'label'})
            print(f"[DatasetLoader] Assigned last column '{last_col}' as target 'label'")

        # Automatically encode string/categorical feature columns
        from sklearn.preprocessing import LabelEncoder
        for col in df.columns:
            if col == 'label':
                continue
            # Check if column contains string values
            if df[col].dtype == object or df[col].dtype.name == 'category' or df[col].apply(type).eq(str).any():
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                print(f"[DatasetLoader] Encoded categorical feature: '{col}'")

        # Dynamically rebuild FEATURE_NAMES in-place
        new_features = [col for col in df.columns if col != 'label']
        FEATURE_NAMES.clear()
        FEATURE_NAMES.extend(new_features)

        # Update FEATURE_DESCRIPTIONS dynamically
        for feat in new_features:
            if feat not in FEATURE_DESCRIPTIONS:
                FEATURE_DESCRIPTIONS[feat] = f"Feature '{feat}' from dataset"

        print(f"[DatasetLoader] Dynamic features loaded ({len(new_features)}): {new_features}")
        return df

    print("[DatasetLoader] No CSV found or path invalid — generating synthetic dataset...")
    return generate_synthetic_dataset(random_state=random_state)


FEATURE_NAMES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    'elevation', 'climate_zone', 'soil_texture', 'growth_days', 'market_demand'
]

FEATURE_DESCRIPTIONS = {
    'N':            'Nitrogen content in soil (kg/ha)',
    'P':            'Phosphorus content in soil (kg/ha)',
    'K':            'Potassium content in soil (kg/ha)',
    'temperature':  'Mean temperature (°C)',
    'humidity':     'Relative humidity (%)',
    'ph':           'Soil pH level (0–14)',
    'rainfall':     'Annual rainfall (mm)',
    'elevation':    'Elevation above sea level (m)',
    'climate_zone': 'Climate zone (0=Tropical, 1=Subtropical, 2=Temperate, 3=Continental, 4=Arid)',
    'soil_texture': 'Soil texture class (0=Sandy, 1=Loamy, 2=Silty, 3=Clayey, 4=Peaty)',
    'growth_days':  'Crop growth duration (days)',
    'market_demand':'Market demand index (0–1)',
}
