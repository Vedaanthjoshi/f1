"""
pipeline/baseline.py — Driver Baseline Pace (retained for reference only)
==========================================================================
Previously computed baseline_pace and lap_time_delta (delta target).

Reviewer-concern fix applied
-------------------------------
Fix 3 : baseline_pace was simultaneously an INPUT feature AND the basis of
        the TARGET (lap_time_delta = fuel_corrected_time - baseline_pace).
        This is target leakage: the model learns to subtract the same value
        it was handed, making it trivially easy to fit without learning
        any real tyre-physics signal.

New design
----------
  • Target   : fuel_corrected_time (absolute, in seconds)
  • Features : tyre_age, compound_encoded, track_evolution,
               TrackTemp, Track, Driver
  • baseline_pace is NO LONGER a model feature or required column.

This function now returns the DataFrame enriched with baseline_pace as a
diagnostic/interpretability column only — it is NOT passed to the model.
"""

import pandas as pd


def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares engineered data for ML regression.

    Returns a DataFrame with:
      - fuel_corrected_time  : the ML regression TARGET (absolute lap time
                               after fuel correction, no baseline subtraction)
      - baseline_pace        : DIAGNOSTIC ONLY — first lap of each stint;
                               retained so callers can still compute deltas
                               for human-readable reporting.
      - lap_time_delta       : DIAGNOSTIC ONLY — kept for backward compatibility
                               with existing charts/reports, not used in training.
    """
    ml_df = df.copy()

    required_cols = [
        'Driver', 'Stint', 'tyre_age', 'compound_encoded',
        'fuel_corrected_time', 'track_evolution', 'TrackTemp'
    ]
    ml_df = ml_df.dropna(subset=required_cols)

    # Compute baseline_pace (diagnostic / reporting use only)
    baseline_df = ml_df.loc[ml_df.groupby(['Driver', 'Stint'])['tyre_age'].idxmin()]
    baseline_df = baseline_df[['Driver', 'Stint', 'fuel_corrected_time']].copy()
    baseline_df = baseline_df.rename(columns={'fuel_corrected_time': 'baseline_pace'})

    ml_df = pd.merge(ml_df, baseline_df, on=['Driver', 'Stint'], how='left')

    # Diagnostic delta (NOT the ML target — kept for charts)
    ml_df['lap_time_delta'] = ml_df['fuel_corrected_time'] - ml_df['baseline_pace']

    return ml_df

