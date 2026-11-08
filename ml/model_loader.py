"""
ml/model_loader.py — Serialized ML Model Management
===================================================
Loads and caches the pre-trained Global XGBoost Quantile Regressor artifact.
Enforces CPU execution context for thread safety in web dashboard workers.
"""

from __future__ import annotations
import os
from typing import Any
import joblib

DEFAULT_MODEL_PATH = os.path.join('models', 'global_degradation_model.joblib')


def load_global_model(model_path: str = DEFAULT_MODEL_PATH) -> tuple[Any, list[str] | None, list[float] | None]:
    """
    Load the pre-trained XGBoost quantile model from disk.
    
    Returns:
    --------
    model           : Trained XGBRegressor object
    features        : List of feature column names
    quantile_alphas : Quantile alpha targets [0.05, 0.50, 0.95]
    """
    if not os.path.exists(model_path):
        return None, None, None

    saved_obj = joblib.load(model_path)
    model = saved_obj['model']
    features = saved_obj.get('features')
    quantile_alphas = saved_obj.get('quantile_alphas', [0.05, 0.50, 0.95])

    # Enforce CPU execution to prevent CUDA device context mismatches across web threads
    try:
        model.set_params(device='cpu')
    except Exception:
        pass

    return model, features, quantile_alphas

