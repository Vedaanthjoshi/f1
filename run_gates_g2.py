"""G2: Train LORO model with all Bahrain GPs removed -> models/loro_bahrain.joblib"""
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor

FEATURE_COLS_14 = [
    'tyre_age','compound_encoded','compound_hardness','track_evolution',
    'TrackTemp','fuel_mass','abrasion_index','circuit_length_km','num_corners',
    'corner_density','thermal_load','abrasion_load','wear_cliff','compound_age_interaction'
]
TARGET = 'lap_time_delta'

print("Loading historical_laps.csv...")
df = pd.read_csv('data/historical_laps.csv')
print(f"Full dataset: {df.shape}")

# Remove all Bahrain GPs
df_no_bhr = df[df['Track'] != 'Bahrain Grand Prix'].copy()
print(f"Without Bahrain: {df_no_bhr.shape}")

# Check missing features
missing = [c for c in FEATURE_COLS_14 if c not in df_no_bhr.columns]
print(f"Missing feature cols: {missing}")

# Drop rows with NaN in features or target
sub = df_no_bhr[FEATURE_COLS_14 + [TARGET]].dropna()
print(f"Training rows after dropna: {sub.shape}")

X = sub[FEATURE_COLS_14]
y = sub[TARGET]

print("Training XGBoost (LORO-Bahrain)...")
model = XGBRegressor(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:quantileerror',
    quantile_alpha=0.50,  # median model for prediction
    random_state=42,
    verbosity=1,
    n_jobs=-1,
)
model.fit(X, y, eval_set=[(X, y)], verbose=50)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loro_bahrain.joblib')
print("Saved -> models/loro_bahrain.joblib")
print(f"Training complete. n_features={model.n_features_in_}")

