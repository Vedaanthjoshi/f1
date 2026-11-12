"""G3: Degradation check on Bahrain 2023 using LORO-Bahrain model"""
import sys
sys.path.insert(0, 'c:/f1')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import fastf1

fastf1.Cache.enable_cache('c:/f1/f1_cache')

FEATURE_COLS_14 = [
    'tyre_age','compound_encoded','compound_hardness','track_evolution',
    'TrackTemp','fuel_mass','abrasion_index','circuit_length_km','num_corners',
    'corner_density','thermal_load','abrasion_load','wear_cliff','compound_age_interaction'
]

# ---- Load LORO model ----
model = joblib.load('models/loro_bahrain.joblib')
print("Loaded loro_bahrain model")

# ---- Load Bahrain 2023 via FastF1 and pipeline ----
print("Loading Bahrain 2023 session...")
session = fastf1.get_session(2023, 'Bahrain Grand Prix', 'R')
session.load(weather=True, messages=False)
print("Session loaded")

from pipeline.ingestion import clean_lap_data, merge_weather_data
from pipeline.features import apply_feature_engineering
from pipeline.baseline import prepare_ml_data

laps_raw = session.laps
weather_raw = session.weather_data
print(f"Raw laps: {len(laps_raw)}")

try:
    laps_clean = clean_lap_data(laps_raw)
    print(f"Clean laps: {len(laps_clean)}")
    laps_w_weather = merge_weather_data(laps_clean, weather_raw)
    print(f"After weather merge: {len(laps_w_weather)}")
    laps_feat = apply_feature_engineering(laps_w_weather)
    print(f"After feature eng: {len(laps_feat)}")
    ml_data = prepare_ml_data(laps_feat)
    print(f"ML data: {len(ml_data)}, cols={list(ml_data.columns[:20])}")
except Exception as e:
    print(f"Pipeline error: {e}")
    import traceback; traceback.print_exc()
    # Fallback: use historical data for Bahrain 2023
    print("Falling back to historical_laps.csv for Bahrain 2023")
    df_all = pd.read_csv('data/historical_laps.csv')
    ml_data = df_all[(df_all['Track'] == 'Bahrain Grand Prix') & (df_all['Season'] == 2023)].copy()
    print(f"Fallback ml_data: {len(ml_data)}")

print(f"\nml_data columns sample: {list(ml_data.columns[:30])}")
print(f"ml_data dtypes:\n{ml_data.dtypes}")

# Check needed columns
needed = FEATURE_COLS_14 + ['lap_time_delta', 'tyre_age', 'Compound']
have = [c for c in needed if c in ml_data.columns]
missing = [c for c in needed if c not in ml_data.columns]
print(f"Have: {have}")
print(f"Missing: {missing}")

# Save intermediate
ml_data.to_csv('/tmp/g3_mldata.csv', index=False)
print("Saved ml_data to /tmp/g3_mldata.csv")

