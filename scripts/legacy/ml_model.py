"""
ml_model.py — Baseline & Data Prep Facade (Delegates to pipeline/)
==================================================================
Maintained for backward compatibility. Baseline pace isolation has been
componentized into the `pipeline/baseline.py` module.
"""

from pipeline.baseline import prepare_ml_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

__all__ = ["prepare_ml_data", "train_degradation_model"]


def train_degradation_model(ml_df):
    """Legacy prototype Random Forest training."""
    features = ['tyre_age', 'compound_encoded', 'baseline_pace', 'track_evolution', 'TrackTemp']
    target = 'lap_time_delta'
    X = ml_df[features]
    y = ml_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model, features


def predict_future_laps(model, features, current_state, laps_ahead=15):
    """Legacy helper stub."""
    pass
