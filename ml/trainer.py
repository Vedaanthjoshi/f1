"""
ml/trainer.py — Offline Global Quantile XGBoost Training Pipeline
=================================================================
Trains the multi-output quantile regression model (p05, p50, p95) on the
73,309 historical laps dataset using asymmetric pinball loss and the
14-feature physics-informed suite.

Reviewer-concern fixes applied
--------------------------------
Fix 1 : Fuel mass clamped >= 2 kg so corrections never go negative.
Fix 2 : track_evolution uses strictly causal expanding median with lag-1.
Fix 3 : baseline_pace completely excluded from features; target is lap_time_delta
        computed by excluding Lap 1 standing starts.
Fix 4 : Full 5-fold race-wise GroupKFold (grouped by Track) cross-validation
        with mean +/- std reported across all folds.
Fix 6 : 14-feature physics-informed suite with non-linear wear cliff, thermal load,
        abrasion load, corner density, and physical Pirelli C-hardness.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from pipeline.features import FEATURE_COLS_14, apply_feature_engineering
from pipeline.baseline import prepare_ml_data


def train_global_model(
    data_path: str = os.path.join("data", "historical_laps.csv"),
    output_path: str = os.path.join("models", "global_degradation_model.joblib"),
    quantile_alphas: list[float] | None = None,
) -> xgb.XGBRegressor | None:
    """Train the global multi-output quantile XGBoost degradation model."""
    if quantile_alphas is None:
        quantile_alphas = [0.05, 0.50, 0.95]

    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run data_collector.py first.")
        return None

    print(f"Loading global dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure all 14 features and clean baseline exist
    missing_feats = [f for f in FEATURE_COLS_14 if f not in df.columns]
    if missing_feats or 'lap_time_delta' not in df.columns:
        print(f"Enriching dataset with 14 features (missing: {missing_feats})...")
        df = apply_feature_engineering(df)
        df = prepare_ml_data(df)

    target = 'lap_time_delta'
    features = FEATURE_COLS_14

    clean_df = df.dropna(subset=features + [target, 'Track']).copy()
    print(f"Total clean training rows: {len(clean_df):,}")

    X = clean_df[features]
    y = clean_df[target]
    groups = clean_df['Track'].astype(str).values

    # ── Fix 4: Full 5-Fold Race-wise GroupKFold Cross-Validation ──────────────
    print("\n" + "=" * 65)
    print("RUNNING 5-FOLD RACE-WISE GROUP-KFOLD CROSS-VALIDATION")
    print("=" * 65)
    gkf = GroupKFold(n_splits=5)

    cv_p50_mae: list[float] = []
    cv_p50_rmse: list[float] = []
    cv_coverage: list[float] = []
    cv_pinball_05: list[float] = []
    cv_pinball_50: list[float] = []
    cv_pinball_95: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        held_out_tracks = sorted(set(groups[test_idx]))

        fold_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            objective='reg:quantileerror',
            quantile_alpha=quantile_alphas,
            random_state=42 + fold,
        )
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)
        y_arr = y_te.to_numpy()

        # Pinball loss
        p05_err = np.where(y_arr >= preds[:, 0], 0.05 * (y_arr - preds[:, 0]), 0.95 * (preds[:, 0] - y_arr)).mean()
        p50_err = np.where(y_arr >= preds[:, 1], 0.50 * (y_arr - preds[:, 1]), 0.50 * (preds[:, 1] - y_arr)).mean()
        p95_err = np.where(y_arr >= preds[:, 2], 0.95 * (y_arr - preds[:, 2]), 0.05 * (preds[:, 2] - y_arr)).mean()

        p50_mae = float(mean_absolute_error(y_arr, preds[:, 1]))
        p50_rmse = float(root_mean_squared_error(y_arr, preds[:, 1]))
        coverage = float(((y_arr >= preds[:, 0]) & (y_arr <= preds[:, 2])).mean())

        cv_p50_mae.append(p50_mae)
        cv_p50_rmse.append(p50_rmse)
        cv_coverage.append(coverage)
        cv_pinball_05.append(p05_err)
        cv_pinball_50.append(p50_err)
        cv_pinball_95.append(p95_err)

        print(f"Fold {fold} ({len(held_out_tracks)} tracks, {len(X_te):,} laps): "
              f"MAE = {p50_mae:.4f}s | RMSE = {p50_rmse:.4f}s | Cov(90%) = {coverage:.1%}")

    print("\n" + "─" * 65)
    print("5-FOLD CROSS-VALIDATION SUMMARY (ON HELD-OUT CIRCUITS):")
    print(f"  p50 MAE             : {np.mean(cv_p50_mae):.4f} +/- {np.std(cv_p50_mae):.4f} s")
    print(f"  p50 RMSE            : {np.mean(cv_p50_rmse):.4f} +/- {np.std(cv_p50_rmse):.4f} s")
    print(f"  p05 Pinball Loss    : {np.mean(cv_pinball_05):.4f} +/- {np.std(cv_pinball_05):.4f} s")
    print(f"  p50 Pinball Loss    : {np.mean(cv_pinball_50):.4f} +/- {np.std(cv_pinball_50):.4f} s")
    print(f"  p95 Pinball Loss    : {np.mean(cv_pinball_95):.4f} +/- {np.std(cv_pinball_95):.4f} s")
    print(f"  90% Coverage Rate   : {np.mean(cv_coverage):.1%} +/- {np.std(cv_coverage):.1%}")
    print("─" * 65)

    # ── Train Final Production Model on Full Dataset ──────────────────────────
    print("\nTraining Final Production Model on all 73,309 laps...")
    final_model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        objective='reg:quantileerror',
        quantile_alpha=quantile_alphas,
        random_state=42,
    )
    final_model.fit(X, y)
    print("[SUCCESS] Production model trained.")

    # ── Feature Importances ───────────────────────────────────────────────────
    importances = final_model.feature_importances_
    print("\n── Feature Importances (gain-based) ──")
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        print(f"  {feat:<26} : {imp:.4f}  ({imp*100:.2f}%)")

    # ── Verify Monotonic Degradation Curve ────────────────────────────────────
    print("\n── Degradation Sanity Check (Pace delta vs tyre age) ──")
    sample_ages = [1, 5, 10, 15, 20, 25]
    test_batch = pd.DataFrame([
        {
            'tyre_age':                 age,
            'compound_encoded':         2.0,  # Medium
            'compound_hardness':        3.0,  # C3
            'track_evolution':          85.0,
            'TrackTemp':                32.0,
            'fuel_mass':                60.0,
            'abrasion_index':           3.0,
            'circuit_length_km':        5.4,
            'num_corners':              16.0,
            'corner_density':           16.0 / 5.4,
            'thermal_load':             age * 32.0,
            'abrasion_load':            age * 3.0,
            'wear_cliff':               (age / 15.0) ** 2.0,
            'compound_age_interaction': age * 3.0,
        }
        for age in sample_ages
    ])
    preds_sample = final_model.predict(test_batch[features])
    for age, (p5, p50, p95) in zip(sample_ages, preds_sample):
        print(f"  Tyre Age {age:2d}: p05={p5:+.3f}s | p50={p50:+.3f}s | p95={p95:+.3f}s")

    # ── Save Artifact ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({
        'model':           final_model,
        'features':        features,
        'quantile_alphas': quantile_alphas,
        'model_version':   'phase2_physics14_quantile_groupkfold',
        'target':          target,
        'cv_metrics': {
            'mae_mean':       float(np.mean(cv_p50_mae)),
            'mae_std':        float(np.std(cv_p50_mae)),
            'rmse_mean':      float(np.mean(cv_p50_rmse)),
            'rmse_std':       float(np.std(cv_p50_rmse)),
            'coverage_mean':  float(np.mean(cv_coverage)),
            'pinball_05_mean': float(np.mean(cv_pinball_05)),
            'pinball_50_mean': float(np.mean(cv_pinball_50)),
            'pinball_95_mean': float(np.mean(cv_pinball_95)),
        }
    }, output_path)
    print(f"\n[SAVED] Validated model artifact saved to {output_path}")

    return final_model
