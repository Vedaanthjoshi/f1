"""
engine/pit_loss.py — Empirical Pit Lane Loss Calculation
========================================================
Calculates circuit-specific pit lane transit and stop time loss dynamically
from actual FastF1 session in-lap and out-lap timing transponders.

Reviewer-concern fix applied
-------------------------------
Fix 6 (Pit loss calculation robustness):
  • Filters out red-flag stoppages (where cars sit in the pit lane for minutes)
    and multi-minute penalty/repair stops by bounding valid green-flag pit stops
    between 16.0s and 35.0s.
  • Prevents corrupting the session pit loss (which previously hit 879.8s at Australia 2023).
"""

import pandas as pd


def calculate_pit_loss(session) -> float:
    """
    Calculates the median time spent in the pit lane for the given session.
    Matches in-laps (Stint N) with out-laps (Stint N+1) per driver.
    
    Returns:
    --------
    float : Median pit-lane transit and stop time in seconds (typically 20.0 - 27.5s).
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
    
    # Filter out red flags, extended repairs, and transponder glitches (16s - 35s)
    valid_stops = pit_stops[(pit_stops['PitLaneTime'] >= 16.0) & (pit_stops['PitLaneTime'] <= 35.0)]
    
    if not valid_stops.empty:
        median_loss = valid_stops['PitLaneTime'].median()
    else:
        median_loss = 24.5
        
    return float(median_loss)
