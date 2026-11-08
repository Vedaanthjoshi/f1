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

    # ── Fix 3 (corrected): target remains lap_time_delta (fuel-corrected
    #    degradation above fresh-tyre baseline). baseline_pace is used ONLY to
    #    compute the target — it is NOT a model input feature. This distinction
    #    is critical: the reviewer concern was that baseline_pace appeared in
    #    both the feature vector AND the target denominator simultaneously.
    #    Keeping it as a target reference but removing it from features resolves
    #    the leakage while preserving meaningful within-stint delta prediction.
    required_cols = [
        'Driver', 'Track', 'tyre_age', 'compound_encoded',
        'baseline_pace', 'track_evolution', 'TrackTemp', 'lap_time_delta',
        'abrasion_index', 'circuit_length_km', 'num_corners', 'compound_hardness',
    ]
    # Circuit physics columns may be missing from old CSV — fill with defaults
    from pipeline.features import CIRCUIT_PHYSICS
    for col in ['abrasion_index', 'circuit_length_km', 'num_corners', 'compound_hardness']:
        if col not in df.columns:
            # Back-fill from Track name if available
            default = CIRCUIT_PHYSICS['_default'][col]
            if 'Track' in df.columns:
                df[col] = df['Track'].map(
                    lambda t, c=col: CIRCUIT_PHYSICS.get(t, CIRCUIT_PHYSICS['_default']).get(c, default)
                )
            else:
                df[col] = default
            print(f"  [INFO] '{col}' missing in CSV — derived from Track lookup.")
    df = df.dropna(subset=required_cols)
    print(f"Total clean rows for training: {len(df)}")

    df['Track']  = df['Track'].astype('category')
    df['Driver'] = df['Driver'].astype('category')

    # baseline_pace intentionally excluded from features (Fix 3).
    # It is retained in the DataFrame only to compute within-stint delta metrics
    # in the evaluation block below — the model never sees it as an input.
    features = [
        'tyre_age', 'compound_encoded',
        'track_evolution', 'TrackTemp',
        'Track', 'Driver',
        # Fix 6: circuit physics features — allow cross-circuit generalisation
        'abrasion_index', 'circuit_length_km', 'num_corners', 'compound_hardness',
    ]
    target = 'lap_time_delta'   # Fix 3: pure within-stint degradation delta

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
    y_arr = y_test.to_numpy()   # shape (n,) — within-stint degradation deltas

    print("\n── Quantile Model Evaluation (race-wise GroupKFold holdout) ──")
    print(f"  Held-out circuits : {held_out_tracks}")
    print(f"  Target            : {target}  (fuel-corrected delta above fresh-tyre baseline)")

    for i, tau in enumerate(quantile_alphas):
        q_pred = preds[:, i]
        errors = np.where(y_arr >= q_pred, tau * (y_arr - q_pred), (1 - tau) * (q_pred - y_arr))
        print(f"  p{int(tau*100):02d} pinball loss : {errors.mean():.4f} s")

    # p50 classical regression metrics on within-stint delta
    p50_preds = preds[:, 1]
    r2   = r2_score(y_arr, p50_preds)
    mae  = mean_absolute_error(y_arr, p50_preds)
    rmse = float(np.sqrt(mean_squared_error(y_arr, p50_preds)))
    print(f"\n  p50  R²   : {r2:.4f}  (on within-stint delta — not inflated by circuit identity)")
    print(f"  p50  MAE  : {mae:.4f} s  (reviewer concern 3 metric: error on degradation component)")
    print(f"  p50  RMSE : {rmse:.4f} s")

    # Per-tyre-age breakdown (reviewer concern 4: does tyre_age actually drive errors?)
    test_df = df.iloc[test_idx].copy()
    test_df['p50_pred'] = p50_preds
    test_df['abs_err']  = (test_df['lap_time_delta'] - test_df['p50_pred']).abs()
    if 'tyre_age' in test_df.columns:
        print("\n  MAE by tyre age bucket:")
        for bucket, label in [(range(1, 6), "Laps 1-5"), (range(6, 16), "Laps 6-15"),
                               (range(16, 31), "Laps 16-30"), (range(31, 99), "Laps 31+")]:
            subset = test_df[test_df['tyre_age'].isin(bucket)]
            if len(subset):
                print(f"    {label:12s}: MAE = {subset['abs_err'].mean():.4f} s  (n={len(subset):,})")

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
        'model_version':   'phase3_delta_target_no_leakage_groupkfold',
        'target':          target,
        'held_out_tracks': held_out_tracks,
    }, output_path)
    print(f"\n[SAVED] Model saved to {output_path}")

    return model

