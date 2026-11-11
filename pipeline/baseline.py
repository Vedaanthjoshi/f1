"""
pipeline/baseline.py — Driver Stint Baseline Pace & Clean Degradation Target
=============================================================================
Computes clean, leak-free driver stint baseline pace and within-stint degradation.

Reviewer-concern fixes applied
-------------------------------
Fix 3 & 5 (Standing Start & Target Leakage):
  • baseline_pace is strictly computed by EXCLUDING Lap 1 standing starts and
    pit in/out anomalies.
  • In Stint 1, Lap 1 takes 5-8s longer due to standing start and grid launch.
    Including Lap 1 as baseline pace previously caused 83.5% of Stint 1 laps
    to have negative deltas (car appearing to get faster as tyres age).
  • Baseline pace is now computed from early representative green-flag laps
    (tyre_age between 2 and 6) at the 20th percentile.
  • lap_time_delta = fuel_corrected_time - baseline_pace is the true degradation
    signal (how much time a tyre set has lost relative to its clean fresh pace).
  • baseline_pace is NOT passed as a model feature, preventing target leakage.
"""

import numpy as np
import pandas as pd


def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares engineered telemetry for ML regression with a clean degradation target.

    Computes:
      - baseline_pace   : Fresh-tyre representative pace for each stint (excluding Lap 1)
      - lap_time_delta  : Degradation component in seconds above fresh-tyre pace
    """
    ml_df = df.copy()

    required_cols = [
        'Driver', 'Stint', 'tyre_age', 'compound_encoded',
        'fuel_corrected_time', 'track_evolution', 'TrackTemp'
    ]
    ml_df = ml_df.dropna(subset=required_cols)

    # ── Grouping columns: handle both single-session and multi-season datasets ──
    group_cols = ['Driver', 'Stint']
    if 'Track' in ml_df.columns:
        group_cols.insert(0, 'Track')
    if 'Season' in ml_df.columns:
        group_cols.insert(0, 'Season')

    # ── 1. Filter for clean, representative early stint laps ──────────────────
    # Exclude Lap 1 (standing start), pit in/out laps, and non-green laps
    clean_mask = (ml_df['tyre_age'] >= 2)
    if 'LapNumber' in ml_df.columns:
        clean_mask = clean_mask & (ml_df['LapNumber'] > 1)
    if 'PitInTime' in ml_df.columns:
        clean_mask = clean_mask & pd.isnull(ml_df['PitInTime'])
    if 'PitOutTime' in ml_df.columns:
        clean_mask = clean_mask & pd.isnull(ml_df['PitOutTime'])
    if 'TrackStatus' in ml_df.columns:
        clean_mask = clean_mask & (ml_df['TrackStatus'].astype(str) == '1')

    # Candidate early laps (tyre_age 2..6)
    early_laps = ml_df[clean_mask & (ml_df['tyre_age'] <= 6)]

    # Compute baseline as 20th percentile of early clean laps (robust fresh tyre pace)
    if not early_laps.empty:
        baseline_early = (
            early_laps.groupby(group_cols)['fuel_corrected_time']
            .quantile(0.20)
            .reset_index()
            .rename(columns={'fuel_corrected_time': 'baseline_pace'})
        )
    else:
        baseline_early = pd.DataFrame(columns=group_cols + ['baseline_pace'])

    # Fallback for short stints or stints without laps in [2, 6]:
    # take min fuel_corrected_time from any clean lap where LapNumber > 1
    fallback_laps = ml_df[clean_mask] if clean_mask.any() else ml_df
    baseline_fallback = (
        fallback_laps.groupby(group_cols)['fuel_corrected_time']
        .min()
        .reset_index()
        .rename(columns={'fuel_corrected_time': 'baseline_pace_fb'})
    )

    baseline_df = pd.merge(baseline_fallback, baseline_early, on=group_cols, how='left')
    baseline_df['baseline_pace'] = baseline_df['baseline_pace'].fillna(baseline_df['baseline_pace_fb'])
    baseline_df = baseline_df[group_cols + ['baseline_pace']]

    # Drop any pre-existing baseline columns to avoid merge conflicts
    for old_col in ['baseline_pace', 'lap_time_delta', 'baseline_pace_fb']:
        if old_col in ml_df.columns:
            ml_df = ml_df.drop(columns=[old_col])

    # ── 2. Merge baseline pace back ──────────────────────────────────────────
    ml_df = pd.merge(ml_df, baseline_df, on=group_cols, how='left')

    # If any row is still missing baseline_pace, fallback to the row's own fuel_corrected_time
    ml_df['baseline_pace'] = ml_df['baseline_pace'].fillna(ml_df['fuel_corrected_time'])

    # ── 3. Compute degradation delta target ─────────────────────────────────
    raw_delta = ml_df['fuel_corrected_time'] - ml_df['baseline_pace']
    # Small negative values (-0.2s) can occur due to DRS/tows or track rubbering;
    # clip lower bound to -0.2s and upper bound to 12.0s (incident/traffic filter)
    ml_df['lap_time_delta'] = np.clip(raw_delta, -0.2, 12.0)

    return ml_df
