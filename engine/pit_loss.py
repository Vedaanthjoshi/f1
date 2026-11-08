"""
engine/pit_loss.py — Empirical Pit Lane Loss Calculation
========================================================
Calculates circuit-specific pit lane transit and stop time loss dynamically
from actual FastF1 session in-lap and out-lap timing transponders.
"""

import pandas as pd


def calculate_pit_loss(session) -> float:
    """
    Calculates the median time spent in the pit lane for the given session.
    Matches in-laps (Stint N) with out-laps (Stint N+1) per driver.
    
    Returns:
    --------
    float : Median pit-lane time in seconds (typically 21.0 - 27.5s).
    """
    laps = session.laps

    in_laps = laps[~laps['PitInTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitInTime']]
    out_laps = laps[~laps['PitOutTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitOutTime']]

    in_laps = in_laps.copy()
    in_laps['NextStint'] = in_laps['Stint'] + 1
    pit_stops = pd.merge(
        in_laps, out_laps,
        left_on=['Driver', 'NextStint'],
        right_on=['Driver', 'Stint'],
        suffixes=('_in', '_out'),
    )
    pit_stops['PitLaneTime'] = (pit_stops['PitOutTime'] - pit_stops['PitInTime']).dt.total_seconds()
    
    median_loss = pit_stops['PitLaneTime'].median()
    if pd.isna(median_loss):
        # Fallback to standard 24.5s if no complete pit stop pair found
        return 24.5
        
    return float(median_loss)

