import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split


def train_global():
    data_path = os.path.join("data", "historical_laps.csv")
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run data_collector.py first.")
        return

    print(f"Loading global dataset from {data_path}...")
    df = pd.read_csv(data_path)

    required_cols = [
        'Driver', 'Track', 'tyre_age', 'compound_encoded',
        'baseline_pace', 'track_evolution', 'TrackTemp', 'lap_time_delta'
    ]
    df = df.dropna(subset=required_cols)
    print(f"Total clean rows for training: {len(df)}")

    df['Track'] = df['Track'].astype('category')
    df['Driver'] = df['Driver'].astype('category')

    features = ['tyre_age', 'compound_encoded', 'baseline_pace', 'track_evolution', 'TrackTemp', 'Track', 'Driver']
    target   = 'lap_time_delta'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # -- Phase 2: Probabilistic Quantile Regression --------------------------
    # Single XGBRegressor with quantile_alpha=[0.05, 0.50, 0.95].
    # model.predict() returns shape (n_samples, 3): [p5, p50, p95] per row.
    # XGBoost ≥2.0 guarantees non-crossing (p5 ≤ p50 ≤ p95 always).
    # This replaces the single point-estimate model from Phase 1.
    quantile_alphas = [0.05, 0.50, 0.95]

    print("Training global XGBoost Quantile model (GPU accelerated if available)...")
    try:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
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
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            tree_method='hist',
            enable_categorical=True,
            objective='reg:quantileerror',
            quantile_alpha=quantile_alphas,
            random_state=42,
        )
        model.fit(X_train, y_train)
        print("[SUCCESS] Trained on CPU.")

    # -- Evaluation: Pinball Loss per quantile --------------------------------
    # R² is meaningless for multi-output quantile. Pinball loss = lower is better.
    # Pinball(τ, y, ŷ) = τ*(y−ŷ) if y≥ŷ else (1−τ)*(ŷ−y)
    model.set_params(device='cpu')
    preds = model.predict(X_test)           # shape: (n_test, 3)
    y_arr = y_test.to_numpy()
    print("\n-- Quantile Model Evaluation (Pinball Loss, lower = better) --")
    for i, tau in enumerate(quantile_alphas):
        q_pred = preds[:, i]
        errors = np.where(y_arr >= q_pred, tau * (y_arr - q_pred), (1 - tau) * (q_pred - y_arr))
        print(f"  p{int(tau*100):02d} pinball loss : {errors.mean():.4f}s")

    # Calibration check: what fraction of true values fall inside [p5, p95]?
    coverage = float(((y_arr >= preds[:, 0]) & (y_arr <= preds[:, 2])).mean())
    print("\n  90% Interval Coverage : {:.1%}  (target approx 90%)".format(coverage))
    if coverage < 0.80:
        print("  !  Coverage below 80% - consider post-hoc calibration.")
    elif coverage > 0.97:
        print("  !  Coverage >97% - intervals are too wide (conservative).")
    else:
        print("  OK  Calibration looks healthy.")

    # Feature importance (uses p50 split gains internally for multi-quantile)
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.4f}")

    # -- Save -----------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'global_degradation_model.joblib')
    joblib.dump({
        'model': model,
        'features': features,
        'quantile_alphas': quantile_alphas,   # [0.05, 0.50, 0.95]
        'model_version': 'phase2_quantile',
    }, model_path)
    print(f"\n[SAVED] Model saved to {model_path}")


if __name__ == "__main__":
    train_global()
