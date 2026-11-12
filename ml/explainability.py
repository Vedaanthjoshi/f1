"""
ml/explainability.py — SHAP Decision Explainability
===================================================
Calculates Shapley additive feature attributions for model degradation
predictions using TreeExplainer on the median quantile slice.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import shap

from ml.trainer import PHYSICS_FEATURE_COLS, _PHYS_TO_BASE14, _extract_physics_features

FEATURE_LABELS = {
    "tyre_age": "Tyre Age",
    "compound_encoded": "Tyre Compound",
    "baseline_pace": "Baseline Pace",
    "track_evolution": "Track Evolution",
    "TrackTemp": "Track Temp",
    "Track": "Circuit",
    "Driver": "Driver",
    "wear_cliff": "Tyre Wear Cliff",
    "compound_hardness": "Compound Hardness",
    "fuel_mass": "Fuel Mass",
    "abrasion_index": "Track Abrasion",
    "corner_density": "Corner Density",
    "compound_age_interaction": "Compound × Age",
    "thermal_load": "Thermal Load",
    "abrasion_load": "Abrasion Load",
    "circuit_length_km": "Circuit Length",
    "num_corners": "Corner Count",
}


def compute_shap_explanation(
    model: Any,
    features: list[str],
    current_state: dict[str, Any],
    track_env: pd.DataFrame,
    sim_lap: int,
) -> pd.DataFrame:
    """
    Compute SHAP attribution values for a driver's lap state.
    
    Returns:
    --------
    pd.DataFrame with columns ['Feature', 'SHAP Value', 'Value', 'Label']
    sorted by absolute SHAP attribution magnitude ascending.
    """
    track_evo_sim = track_env[track_env["LapNumber"] == sim_lap]["track_evolution"]
    track_evo_val = track_evo_sim.iloc[0] if not track_evo_sim.empty else 0.0

    track_temp_sim = track_env[track_env["LapNumber"] == sim_lap]["TrackTemp"]
    track_temp_val = track_temp_sim.iloc[0] if not track_temp_sim.empty else 30.0

    row_data: dict[str, Any] = {
        **current_state,
        'track_evolution':  track_evo_val,
        'TrackTemp':        track_temp_val,
    }

    shap_df = pd.DataFrame([{col: row_data.get(col, np.nan) for col in features}], columns=features)

    if hasattr(model, "xgb_center"):
        Xp = _extract_physics_features(shap_df)
        explainer = shap.TreeExplainer(model.xgb_center)
        phys_shap = explainer(Xp)
        phys_vals = phys_shap[0].values
        mapped_vals = {col: 0.0 for col in features}
        for p_col, sval in zip(PHYSICS_FEATURE_COLS, phys_vals):
            parent = _PHYS_TO_BASE14.get(p_col)
            if parent in mapped_vals:
                mapped_vals[parent] += float(sval)
        vals = np.array([mapped_vals[col] for col in features], dtype=float)
    else:
        if 'Track' in shap_df.columns:
            shap_df['Track'] = shap_df['Track'].astype('category')
        if 'Driver' in shap_df.columns:
            shap_df['Driver'] = shap_df['Driver'].astype('category')

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(shap_df)

        # For multi-quantile models (shape: samples, features, quantiles), index 1 is p50
        if len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1].values
        else:
            vals = shap_values[0].values

    shap_plot_df = pd.DataFrame({
        "Feature": features,
        "SHAP Value": vals,
        "Value": [row_data.get(col, "—") for col in features],
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    shap_plot_df["Label"] = shap_plot_df.apply(
        lambda row: f"{FEATURE_LABELS.get(row['Feature'], row['Feature'])} = {row['Value']}",
        axis=1,
    )

    return shap_plot_df

