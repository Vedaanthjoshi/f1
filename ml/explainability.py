"""
ml/explainability.py — SHAP Decision Explainability
===================================================
Calculates Shapley additive feature attributions for model degradation
predictions using TreeExplainer on the median quantile slice.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import shap

FEATURE_LABELS = {
    "tyre_age": "Tyre Age",
    "compound_encoded": "Tyre Compound",
    "baseline_pace": "Baseline Pace",
    "track_evolution": "Track Evolution",
    "TrackTemp": "Track Temp",
    "Track": "Circuit",
    "Driver": "Driver",
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
        'tyre_age':         current_state['tyre_age'],
        'compound_encoded': current_state['compound_encoded'],
        'baseline_pace':    current_state['baseline_pace'],
        'track_evolution':  track_evo_val,
        'TrackTemp':        track_temp_val,
        'Track':            current_state['Track'],
        'Driver':           current_state['Driver'],
    }

    shap_df = pd.DataFrame([{col: row_data[col] for col in features if col in row_data}], columns=features)
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
        "Value": shap_df.iloc[0].values,
    }).sort_values(by="SHAP Value", key=abs, ascending=True)

    shap_plot_df["Label"] = shap_plot_df.apply(
        lambda row: f"{FEATURE_LABELS.get(row['Feature'], row['Feature'])} = {row['Value']}",
        axis=1,
    )

    return shap_plot_df

