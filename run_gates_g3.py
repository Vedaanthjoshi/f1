"""G3: Degradation check on Bahrain 2023 using LORO-Bahrain model"""
import sys
sys.path.insert(0, 'c:/f1')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import fastf1
from sklearn.linear_model import LinearRegression

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

from pipeline.ingestion import clean_lap_data, merge_weather_data
from pipeline.features import apply_feature_engineering
from pipeline.baseline import prepare_ml_data

laps_clean = clean_lap_data(session.laps)
laps_w = merge_weather_data(laps_clean, session.weather_data)
laps_feat = apply_feature_engineering(laps_w)
ml_data = prepare_ml_data(laps_feat)
print(f"Bahrain 2023 ml_data: {len(ml_data)} laps")

# ---- Find stints with >=12 clean green laps ----
# "clean green laps" = laps where TrackStatus is clean (not under SC/VSC) and not pit laps
# From the data: TrackStatus column in FastF1 laps, "1" = green
# In ml_data we have TrackStatus column

# Filter: not pit in/out lap, clean green status, have all features
ml = ml_data.copy()

# Green lap filter: TrackStatus == "1" (green)
if 'TrackStatus' in ml.columns:
    ml['is_green'] = ml['TrackStatus'].astype(str) == '1'
else:
    ml['is_green'] = True  # assume all green if not available

# Not pit laps: PitInTime and PitOutTime should be NaT for racing laps
if 'PitInTime' in ml.columns:
    ml['is_pit_in'] = ml['PitInTime'].notna()
else:
    ml['is_pit_in'] = False
if 'PitOutTime' in ml.columns:
    ml['is_pit_out'] = ml['PitOutTime'].notna()
else:
    ml['is_pit_out'] = False

ml['is_racing'] = ml['is_green'] & ~ml['is_pit_in'] & ~ml['is_pit_out']

# Drop rows with missing features or target
feat_cols_check = FEATURE_COLS_14 + ['lap_time_delta']
ml_valid = ml[ml['is_racing'] & ml[feat_cols_check].notna().all(axis=1)].copy()
print(f"Valid racing green laps: {len(ml_valid)}")

# Identify stint column
stint_col = 'Stint' if 'Stint' in ml_valid.columns else 'stint'
driver_col = 'Driver' if 'Driver' in ml_valid.columns else 'driver'
lap_col = 'LapNumber' if 'LapNumber' in ml_valid.columns else 'lapnumber'
compound_col = 'Compound' if 'Compound' in ml_valid.columns else 'compound'

# Group by driver+stint, find stints with >=12 laps
groups = ml_valid.groupby([driver_col, stint_col])
valid_stints = []
for (drv, stnt), grp in groups:
    if len(grp) >= 12:
        valid_stints.append((drv, stnt, grp.sort_values(lap_col).reset_index(drop=True)))

print(f"Stints with >=12 clean green laps: {len(valid_stints)}")

# ---- Predict with LORO model ----
# For each valid stint, generate pairs (lap_i, lap_i+5)
records = []
for drv, stnt, grp in valid_stints:
    X_grp = grp[FEATURE_COLS_14]
    preds = model.predict(X_grp, validate_features=False)
    grp = grp.copy()
    grp['pred_delta'] = preds
    n = len(grp)
    for i in range(n - 5):
        actual_inc = grp.loc[i+5, 'lap_time_delta'] - grp.loc[i, 'lap_time_delta']
        pred_inc = grp.loc[i+5, 'pred_delta'] - grp.loc[i, 'pred_delta']
        age_i = grp.loc[i, 'tyre_age']
        age_i5 = grp.loc[i+5, 'tyre_age']
        compound = grp.loc[i, compound_col] if compound_col in grp.columns else 'UNK'
        records.append({
            'driver': drv,
            'stint': stnt,
            'lap_i': i,
            'lap_i5': i+5,
            'tyre_age_i': age_i,
            'tyre_age_i5': age_i5,
            'compound': compound,
            'actual_increment': actual_inc,
            'pred_increment': pred_inc,
            'error': actual_inc - pred_inc,
            'abs_error': abs(actual_inc - pred_inc),
        })

pairs_df = pd.DataFrame(records)
print(f"\nTotal pairs: {len(pairs_df)}")

if len(pairs_df) > 0:
    xgb_mae = pairs_df['abs_error'].mean()
    xgb_bias = pairs_df['error'].mean()
    print(f"\nXGBoost MAE on increment pairs: {xgb_mae:.4f}s")
    print(f"XGBoost Bias (mean error): {xgb_bias:.4f}s")

    # By compound
    print("\nBy compound:")
    cpd_stats = pairs_df.groupby('compound').agg(
        n=('abs_error','count'),
        MAE=('abs_error','mean'),
        bias=('error','mean')
    )
    print(cpd_stats.to_string())

    # By tyre_age bucket
    pairs_df['age_bucket'] = pd.cut(pairs_df['tyre_age_i'], bins=[0,5,10,15,20,30,50], labels=['1-5','6-10','11-15','16-20','21-30','31+'])
    print("\nBy tyre_age bucket (at start lap of pair):")
    age_stats = pairs_df.groupby('age_bucket', observed=False).agg(
        n=('abs_error','count'),
        MAE=('abs_error','mean'),
        bias=('error','mean')
    )
    print(age_stats.to_string())
else:
    xgb_mae = None
    xgb_bias = None
    cpd_stats = None
    age_stats = None
    print("No pairs found!")

# ---- Linear baseline: per-compound slope on fuel_corrected_time vs tyre_age ----
# Fit on non-Bahrain data
print("\n\nFitting linear baseline on non-Bahrain data...")
df_all = pd.read_csv('data/historical_laps.csv')
df_non_bhr = df_all[df_all['Track'] != 'Bahrain Grand Prix'].copy()

linear_slopes = {}
linear_intercepts = {}
for cpd in df_non_bhr['Compound'].unique() if 'Compound' in df_non_bhr.columns else df_non_bhr['compound_encoded'].unique():
    cpd_col = 'Compound' if 'Compound' in df_non_bhr.columns else 'compound_encoded'
    sub = df_non_bhr[df_non_bhr[cpd_col] == cpd][['tyre_age','fuel_corrected_time']].dropna()
    if len(sub) >= 10:
        lr = LinearRegression()
        lr.fit(sub[['tyre_age']], sub['fuel_corrected_time'])
        linear_slopes[cpd] = lr.coef_[0]
        linear_intercepts[cpd] = lr.intercept_
        print(f"  {cpd}: slope={lr.coef_[0]:.4f} s/lap, intercept={lr.intercept_:.3f}s")

# Apply linear model to Bahrain 2023 pairs
print("\nApplying linear model to increment pairs...")
if len(pairs_df) > 0:
    def linear_pred_increment(row):
        cpd = row['compound']
        # For compound not in linear_slopes, use median
        if cpd in linear_slopes:
            slope = linear_slopes[cpd]
        else:
            # Try uppercase
            cpd_upper = str(cpd).upper()
            if cpd_upper in linear_slopes:
                slope = linear_slopes[cpd_upper]
            else:
                slope = np.median(list(linear_slopes.values()))
        return slope * (row['tyre_age_i5'] - row['tyre_age_i'])

    pairs_df['linear_pred_increment'] = pairs_df.apply(linear_pred_increment, axis=1)
    pairs_df['linear_error'] = pairs_df['actual_increment'] - pairs_df['linear_pred_increment']
    pairs_df['linear_abs_error'] = pairs_df['linear_error'].abs()

    linear_mae = pairs_df['linear_abs_error'].mean()
    linear_bias = pairs_df['linear_error'].mean()
    print(f"\nLinear MAE on increment pairs: {linear_mae:.4f}s")
    print(f"Linear Bias: {linear_bias:.4f}s")
    print(f"\nXGBoost MAE: {xgb_mae:.4f}s vs Linear MAE: {linear_mae:.4f}s")

    if xgb_mae < linear_mae:
        decision = "XGBoost"
        print("DECISION: Use XGBoost increments (XGBoost MAE < Linear MAE)")
    else:
        decision = "Linear slopes"
        print("DECISION: Use linear slopes (Linear MAE <= XGBoost MAE)")

    # By compound comparison
    print("\nBy compound - XGBoost vs Linear MAE:")
    cpd_comp = pairs_df.groupby('compound').agg(
        n=('abs_error','count'),
        XGB_MAE=('abs_error','mean'),
        XGB_bias=('error','mean'),
        Lin_MAE=('linear_abs_error','mean'),
        Lin_bias=('linear_error','mean'),
    )
    print(cpd_comp.to_string())

    # Save results
    pairs_df.to_csv('c:/f1/g3_pairs_results.csv', index=False)
    print("\nSaved pairs to c:/f1/g3_pairs_results.csv")

    # Store for GATES.md
    print("\n=== G3 SUMMARY FOR GATES.md ===")
    print(f"Valid stints (>=12 green laps): {len(valid_stints)}")
    print(f"Total pairs (5 apart): {len(pairs_df)}")
    print(f"XGBoost increment MAE: {xgb_mae:.4f}s, Bias: {xgb_bias:+.4f}s")
    print(f"Linear slope MAE: {linear_mae:.4f}s, Bias: {linear_bias:+.4f}s")
    print(f"DECISION: {decision}")
    print("\nCompound breakdown:")
    print(cpd_comp.to_string())
    print("\nAge bucket breakdown:")
    print(age_stats.to_string())
    print("Linear slopes:")
    for k,v in linear_slopes.items():
        print(f"  {k}: {v:.4f} s/lap")
else:
    decision = "N/A - no pairs found"
    linear_mae = None
    linear_bias = None
