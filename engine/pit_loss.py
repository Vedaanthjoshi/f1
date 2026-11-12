"""
engine/pit_loss.py — Empirical Relative Pit Loss Calculation (G4)
==================================================================
Calculates circuit-specific net pit-stop time loss dynamically from FastF1
session telemetry using the 2-lap (in_lap, out_lap) relative timing method:

    PitLoss = (Time[out_lap] - Time[in_lap - 1])
              - median_non_pit(Time[out_lap] - Time[in_lap - 1])

This measures the true net time lost vs. staying on track at racing speed
(already accounting for pit-lane speed limit, stationary box time, pit-lane
shortcut geometry, and out-lap cold-tyre warmup).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def calculate_pit_loss(session, max_lap: int | None = None) -> float:
    """
    Calculate the median net 2-lap relative pit-stop loss (in seconds) for a session.

    Parameters
    ----------
    session : FastF1 session object with `.laps` loaded
    max_lap : optional int, restrict pit stops and reference laps to LapNumber <= max_lap
              (prevents lookahead leakage during causal backtesting)

    Returns
    -------
    float : Median net pit loss in seconds (typically 18.0 - 27.0s).
    """
    laps = session.laps.copy()
    if max_lap is not None and 'LapNumber' in laps.columns:
        laps = laps[laps['LapNumber'] <= max_lap]

    if laps.empty or 'Time' not in laps.columns:
        return 24.5

    laps['Time_s'] = laps['Time'].dt.total_seconds()
    laps['is_in'] = pd.notnull(laps['PitInTime'])
    laps['is_out'] = pd.notnull(laps['PitOutTime'])

    # Map (Driver, LapNumber) -> (Time_s, TrackStatus, is_in, is_out)
    laps_clean = laps.dropna(subset=['Driver', 'LapNumber', 'Time_s']).copy()
    laps_clean['LapNumber'] = laps_clean['LapNumber'].astype(int)

    time_map = laps_clean.set_index(['Driver', 'LapNumber'])['Time_s'].to_dict()
    status_map = laps_clean.set_index(['Driver', 'LapNumber'])['TrackStatus'].astype(str).to_dict()

    # Identify green-flag non-pit 2-lap reference windows per (in_lap -> out_lap)
    non_pit = laps_clean[
        (~laps_clean['is_in'])
        & (~laps_clean['is_out'])
        & (laps_clean['TrackStatus'].astype(str) == '1')
    ]
    ref_windows: dict[tuple[int, int], list[float]] = {}
    for drv, lap_in in zip(non_pit['Driver'], non_pit['LapNumber']):
        t_prev = time_map.get((drv, lap_in - 1))
        t_out = time_map.get((drv, lap_in + 1))
        if t_prev is not None and t_out is not None:
            # Ensure out_lap itself wasn't a pit lap or neutralized
            if status_map.get((drv, lap_in + 1), '1') == '1':
                ref_windows.setdefault((lap_in, lap_in + 1), []).append(t_out - t_prev)

    # Evaluate each driver's pit stops (in_lap with PitInTime -> out_lap = in_lap + 1 with PitOutTime)
    in_stops = laps_clean[laps_clean['is_in'] & (laps_clean['TrackStatus'].astype(str) == '1')]
    relative_losses: list[float] = []

    for drv, lap_in in zip(in_stops['Driver'], in_stops['LapNumber']):
        lap_out = lap_in + 1
        if status_map.get((drv, lap_out), '') != '1':
            continue
        t_prev = time_map.get((drv, lap_in - 1))
        t_out = time_map.get((drv, lap_out))
        if t_prev is None or t_out is None:
            continue

        refs = ref_windows.get((lap_in, lap_out), [])
        if len(refs) < 3:
            continue

        loss = (t_out - t_prev) - float(np.median(refs))
        if 14.0 <= loss <= 40.0:
            relative_losses.append(loss)

    if len(relative_losses) >= 10:
        arr = np.asarray(relative_losses, dtype=float)
        lo, hi = np.percentile(arr, [5, 95])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) > 0:
            return float(np.median(trimmed))

    if relative_losses:
        return float(np.median(relative_losses))

    # Fallback: raw PitOutTime - PitInTime if 2-lap relative windows weren't available
    in_laps = laps[~laps['PitInTime'].isna()][['Driver', 'Stint', 'PitInTime']].copy()
    out_laps = laps[~laps['PitOutTime'].isna()][['Driver', 'Stint', 'PitOutTime']].copy()
    in_laps['NextStint'] = in_laps['Stint'] + 1
    pit_stops = pd.merge(
        in_laps, out_laps,
        left_on=['Driver', 'NextStint'],
        right_on=['Driver', 'Stint'],
    )
    if not pit_stops.empty:
        pit_stops['PitLaneTime'] = (pit_stops['PitOutTime'] - pit_stops['PitInTime']).dt.total_seconds()
        valid = pit_stops[(pit_stops['PitLaneTime'] >= 16.0) & (pit_stops['PitLaneTime'] <= 35.0)]
        if not valid.empty:
            return float(valid['PitLaneTime'].median())

    return 24.5
