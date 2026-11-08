"""
ml package — Machine Learning, Model Loading & Explainability
"""

from ml.model_loader import load_global_model
try:
    from ml.explainability import compute_shap_explanation
except ImportError:
    compute_shap_explanation = None
from ml.trainer import train_global_model

__all__ = [
    "load_global_model",
    "compute_shap_explanation",
    "train_global_model",
]

