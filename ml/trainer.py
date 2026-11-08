"""
ml/trainer.py — Offline Global Quantile XGBoost Training Pipeline
=================================================================
Trains the multi-output quantile regression model (p05, p50, p95) on the
historical laps dataset using asymmetric pinball loss.

Reviewer-concern fixes applied
--------------------------------
Fix 3 : Removed 'baseline_pace' from the feature set.
        Previously baseline_pace was both an input feature AND the
        denominator of the regression target (lap_time_delta = fuel_corrected_time
        - baseline_pace), causing direct target leakage.
        New target: fuel_corrected_time (absolute, in seconds).
        New features: tyre_age, compound_encoded, track_evolution,
                      TrackTemp, Track, Driver.

Fix 4 : Replaced random 80:20 train_test_split with race-wise GroupKFold
        (k=5, grouped by Track) so that laps from the same Grand Prix can
        NEVER appear in both the train and test sets simultaneously.
        The held-out test set is the last cross-validation fold (a full set
        of Grand Prix circuits not seen during training), giving a realistic
        estimate of out-of-sample generalisation.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GroupKFold  # Fix 4: race-wise split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_global_model(
    data_path: str = os.path.join("data", "historical_laps.csv"),
    output_path: str = os.path.join("models", "global_degradation_model.joblib"),
    quantile_alphas: list[float] | None = None,
) -> xgb.XGBRegressor | None:
    """Train the global multi-output quantile XGBoost degradation model.

    Evaluation uses race-wise GroupKFold (grouped by Track) to prevent
    laps from the same Grand Prix appearing in both train and test sets.
    """
    if quantile_alphas is None:
        quantile_alphas = [0.05, 0.50, 0.95]

    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run data_collector.py first.")
        return None

    print(f"Loading global dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # ── Fix 3: target is now absolute fuel_corrected_time, not delta ────────
    required_cols = [
        'Driver', 'Track', 'tyre_age', 'compound_encoded',
        'track_evolution', 'TrackTemp', 'fuel_corrected_time'
    ]
    df = df.dropna(subset=required_cols)
    print(f"Total clean rows for training: {len(df)}")

    df['Track']  = df['Track'].astype('category')
    df['Driver'] = df['Driver'].astype('category')

    # baseline_pace intentionally excluded (Fix 3: target leakage removal)
    features = [
        'tyre_age', 'compound_encoded',
        'track_evolution', 'TrackTemp',
        'Track', 'Driver',
    ]
    target = 'fuel_corrected_time'   # Fix 3: absolute lap time, not delta

    X = df[features]
    y = df[target]

    # ── Fix 4: Race-wise GroupKFold split ────────────────────────────────────
    # Groups are circuit names (Track column, pre-category-encoded).
    # Last fold is held out as the test set — circuits unseen during training.
    groups = df['Track'].astype(str).values
    gkf = GroupKFold(n_splits=5)

    train_idx, test_idx = None, None
    for tr, te in gkf.split(X, y, groups):
        train_idx, test_idx = tr, te   # keep the last fold as test

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    held_out_tracks = sorted(set(groups[test_idx]))
    print(f"\n[GroupKFold] Test set circuits ({len(held_out_tracks)}): {held_out_tracks}")
    print(f"  Train laps: {len(X_train):,}  |  Test laps: {len(X_test):,}")

    # ── Train quantile model ─────────────────────────────────────────────────
    print("\nTraining global XGBoost Quantile model...")
    try:
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cuda',
            enable_categorical=True,
            objective='reg:quantileerror',
            quantile_alpha=quantile_alphas,
            random_state=42,
        )
        model.fit(X_train, y_train)
        print("[SUCCESS] Trained on GPU.")
    except Exception as e:
        print(f"GPU training failed (falling back to CPU): {e}")
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            enable_categorical=True,
            objective='reg:quantileerror',
            quantile_alpha=quantile_alphas,
            random_state=42,
        )
        model.fit(X_train, y_train)
        print("[SUCCESS] Trained on CPU.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    model.set_params(device='cpu')
    preds = model.predict(X_test)
    y_arr = y_test.to_numpy()

    print("\n── Quantile Model Evaluation (race-wise GroupKFold holdout) ──")
    print(f"  Held-out circuits : {held_out_tracks}")

    for i, tau in enumerate(quantile_alphas):
        q_pred = preds[:, i]
        errors = np.where(y_arr >= q_pred, tau * (y_arr - q_pred), (1 - tau) * (q_pred - y_arr))
        print(f"  p{int(tau*100):02d} pinball loss : {errors.mean():.4f} s")

    # p50 classical regression metrics
    p50_preds = preds[:, 1]
    r2  = r2_score(y_arr, p50_preds)
    mae = mean_absolute_error(y_arr, p50_preds)
    rmse = float(np.sqrt(mean_squared_error(y_arr, p50_preds)))
    print(f"\n  p50  R²   : {r2:.4f}")
    print(f"  p50  MAE  : {mae:.4f} s")
    print(f"  p50  RMSE : {rmse:.4f} s")

    # Within-stint degradation component error (reviewer concern 3)
    test_df = df.iloc[test_idx].copy()
    test_df['p50_pred'] = p50_preds
    # Join baseline_pace for delta computation (diagnostic only)
    if 'baseline_pace' in test_df.columns:
        test_df['delta_true'] = test_df['fuel_corrected_time'] - test_df['baseline_pace']
        test_df['delta_pred'] = test_df['p50_pred']           - test_df['baseline_pace']
        delta_mae  = mean_absolute_error(test_df['delta_true'], test_df['delta_pred'])
        delta_rmse = float(np.sqrt(mean_squared_error(test_df['delta_true'], test_df['delta_pred'])))
        print(f"\n  Degradation-delta MAE  : {delta_mae:.4f} s  (reviewer concern 3 metric)")
        print(f"  Degradation-delta RMSE : {delta_rmse:.4f} s")

    coverage = float(((y_arr >= preds[:, 0]) & (y_arr <= preds[:, 2])).mean())
    print(f"\n  90% Interval Coverage : {coverage:.1%}")

    # ── Feature importance ───────────────────────────────────────────────────
    importances = model.feature_importances_
    print("\n── Feature Importances (gain-based) ──")
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        print(f"  {feat:<22} : {imp:.4f}  ({imp*100:.2f}%)")

    # ── Save artifact ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({
        'model':           model,
        'features':        features,
        'quantile_alphas': quantile_alphas,
        'model_version':   'phase3_no_leakage_groupkfold',
        'target':          target,
        'held_out_tracks': held_out_tracks,
    }, output_path)
    print(f"\n[SAVED] Model saved to {output_path}")

    return model

