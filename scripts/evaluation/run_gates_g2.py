"""G2: Train LORO physics-ensemble model with all Bahrain GPs removed -> models/loro_bahrain.joblib"""
import os
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from ml.trainer import PhysicsDegradationEnsemble, FEATURE_COLS_14

TARGET = 'lap_time_delta'

print("Loading historical_laps.csv...")
df = pd.read_csv('data/historical_laps.csv')
print(f"Full dataset: {df.shape}")

# Exclude standing-start Lap 1 rows before training/evaluation
if 'LapNumber' in df.columns:
    df = df[df['LapNumber'] >= 2].copy()

df['Rainfall'] = pd.to_numeric(df.get('Rainfall', 0.0), errors='coerce').fillna(0.0)
df_no_bhr = df[df['Track'] != 'Bahrain Grand Prix'].copy()
df_bhr = df[df['Track'] == 'Bahrain Grand Prix'].dropna(subset=FEATURE_COLS_14 + [TARGET]).copy()
print(f"Without Bahrain (Lap>=2): {df_no_bhr.shape}")

# Drop rows with NaN in features or target and train on clean dry laps
sub = df_no_bhr.dropna(subset=FEATURE_COLS_14 + [TARGET]).copy()
clean_dry = (sub[TARGET] < 3.5) & (sub['Rainfall'] == 0.0)
sub_train = sub.loc[clean_dry]
print(f"Clean-dry training rows (without Bahrain): {len(sub_train):,}")

print("Training physics-regularized ensemble (LORO-Bahrain)...")
model = PhysicsDegradationEnsemble(ridge_weight=0.65, seed=42, single_output=True)
model.fit(sub_train[FEATURE_COLS_14], sub_train[TARGET])

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loro_bahrain.joblib')
print("Saved -> models/loro_bahrain.joblib")

bhr_preds = model.predict(df_bhr[FEATURE_COLS_14])
bhr_clean = (df_bhr[TARGET] < 3.5) & (df_bhr['Rainfall'] == 0.0)
print(
    f"LORO Bahrain Zero-Shot: All Lap>=2 R2={r2_score(df_bhr[TARGET], bhr_preds):+.4f} "
    f"MAE={mean_absolute_error(df_bhr[TARGET], bhr_preds):.4f}s | "
    f"Clean Dry R2={r2_score(df_bhr.loc[bhr_clean, TARGET], bhr_preds[bhr_clean]):+.4f} "
    f"MAE={mean_absolute_error(df_bhr.loc[bhr_clean, TARGET], bhr_preds[bhr_clean]):.4f}s"
)

