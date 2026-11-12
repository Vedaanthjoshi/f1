"""
replay/load.py — Raw FastF1 Telemetry Ingestion & Replay Data Layer
===================================================================
Constructs a raw, uncleaned replay table preserving all racing, pit, and
neutralized laps. Derives gap metrics and extracts race-level events (SC/VSC,
pit stops, compound sequences).
"""

from __future__ import annotations

import os
from typing import Any
import fastf1
import numpy as np
import pandas as pd

CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# FIA / FastF1 neutralization codes:
# '4': Safety Car (SC)
# '5': Red Flag (RF)
# '6': Virtual Safety Car (VSC) Deployed
# '7': Virtual Safety Car (VSC) Ending
NEUTRALIZATION_CODES = ("4", "5", "6", "7")


def load_replay_table(year: int, gp_name: str, session: Any = None) -> pd.DataFrame:
    """
    Load raw FastF1 session laps and produce an uncleaned replay DataFrame.
    Preserves ALL laps (including in/out pit laps and non-green laps).

    Columns:
      Driver, LapNumber, Position, Time_s, LapTime_s, Compound, TyreLife,
      Stint, is_in_lap, is_out_lap, TrackStatus, is_neutralized,
      gap_ahead_s, gap_behind_s.
    """
    if session is None:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(weather=False, messages=False)

    raw_laps = session.laps.copy()

    # Base telemetry columns
    df = pd.DataFrame()
    df['Driver'] = raw_laps['Driver'].astype(str)
    if 'Team' in raw_laps.columns:
        df['Team'] = raw_laps['Team'].astype(str)
    df['LapNumber'] = raw_laps['LapNumber'].astype(float)
    df['Position'] = raw_laps['Position'].astype(float)
    df['Time_s'] = raw_laps['Time'].dt.total_seconds()
    df['LapTime_s'] = raw_laps['LapTime'].dt.total_seconds()
    df['Compound'] = raw_laps['Compound'].astype(str)
    df['TyreLife'] = raw_laps['TyreLife'].astype(float)
    df['Stint'] = raw_laps['Stint'].astype(float)

    # In-lap / out-lap flags
    df['is_in_lap'] = pd.notnull(raw_laps['PitInTime'])
    df['is_out_lap'] = pd.notnull(raw_laps['PitOutTime'])

    # Track status & neutralization
    df['TrackStatus'] = raw_laps['TrackStatus'].astype(str)
    df['is_neutralized'] = df['TrackStatus'].apply(
        lambda ts: any(code in ts for code in NEUTRALIZATION_CODES)
    )

    # Initialize gap columns
    df['gap_ahead_s'] = np.nan
    df['gap_behind_s'] = np.nan

    # Derive gaps lap-by-lap
    # "Derive gaps from Time_s differences between cars at the same LapNumber,
    # ordered by Position. Handle lapped cars and retirements explicitly (NaN, never wrong numbers)."
    for _, grp in df.groupby('LapNumber'):
        # Filter cars that completed the lap with a valid Position and Time_s
        valid = grp.dropna(subset=['Position', 'Time_s']).sort_values('Position')
        if valid.empty:
            continue

        pos_to_idx = {row.Position: idx for idx, row in valid.iterrows()}
        pos_to_time = {row.Position: row.Time_s for _, row in valid.iterrows()}

        for _, row in valid.iterrows():
            curr_pos = row.Position
            curr_time = row.Time_s
            idx = pos_to_idx[curr_pos]

            # Gap to car ahead: car must be at curr_pos - 1 on this lap
            ahead_pos = curr_pos - 1
            if ahead_pos in pos_to_time:
                gap_ahead = curr_time - pos_to_time[ahead_pos]
                df.at[idx, 'gap_ahead_s'] = gap_ahead if gap_ahead >= 0 else np.nan
            else:
                df.at[idx, 'gap_ahead_s'] = np.nan

            # Gap to car behind: car must be at curr_pos + 1 on this lap
            behind_pos = curr_pos + 1
            if behind_pos in pos_to_time:
                gap_behind = pos_to_time[behind_pos] - curr_time
                df.at[idx, 'gap_behind_s'] = gap_behind if gap_behind >= 0 else np.nan
            else:
                df.at[idx, 'gap_behind_s'] = np.nan

    return df


def _group_contiguous_laps(laps: list[int | float]) -> list[tuple[int, int]]:
    """Group sorted list of lap numbers into contiguous ranges [start_lap, end_lap]."""
    if not laps:
        return []
    sorted_unique = sorted(set(int(lap) for lap in laps))
    ranges = []
    start = sorted_unique[0]
    prev = sorted_unique[0]

    for lap in sorted_unique[1:]:
        if lap == prev + 1:
            prev = lap
        else:
            ranges.append((start, prev))
            start = lap
            prev = lap
    ranges.append((start, prev))
    return ranges


def get_race_events(table: pd.DataFrame) -> dict[str, Any]:
    """
    Extract race events from the replay table:
      - SC and VSC lap ranges
      - Per-driver pit stops (in-lap, out-lap, compounds before/after)
      - Per-driver pit lap lists
      - Per-driver compound sequences
    """
    # 1. SC and VSC lap ranges
    sc_laps = []
    vsc_laps = []
    neutralized_laps = []

    for lap_num, grp in table.groupby('LapNumber'):
        statuses = grp['TrackStatus'].tolist()
        has_sc = any('4' in ts for ts in statuses)
        has_vsc = any(('6' in ts or '7' in ts) for ts in statuses)
        is_neut = any(row.is_neutralized for _, row in grp.iterrows())

        if has_sc:
            sc_laps.append(lap_num)
        if has_vsc and not has_sc:
            vsc_laps.append(lap_num)
        if is_neut:
            neutralized_laps.append(lap_num)

    sc_ranges = _group_contiguous_laps(sc_laps)
    vsc_ranges = _group_contiguous_laps(vsc_laps)
    neutralized_ranges = _group_contiguous_laps(neutralized_laps)

    # 2. Per-driver pit stops and compound sequences
    driver_pit_laps: dict[str, list[int]] = {}
    driver_compound_seq: dict[str, list[str]] = {}
    driver_events: dict[str, Any] = {}
    all_pit_stops: list[dict[str, Any]] = []

    for driver, grp in table.groupby('Driver'):
        grp_sorted = grp.sort_values('LapNumber')

        # Compound sequence in order of stints
        stint_compounds = (
            grp_sorted.drop_duplicates(subset=['Stint'])
            .sort_values('Stint')['Compound']
            .tolist()
        )
        driver_compound_seq[driver] = stint_compounds

        # Pit laps and stop pairing
        in_laps = [int(lap) for lap in grp_sorted[grp_sorted['is_in_lap']]['LapNumber'].tolist()]
        out_laps = [int(lap) for lap in grp_sorted[grp_sorted['is_out_lap']]['LapNumber'].tolist()]
        driver_pit_laps[driver] = in_laps

        driver_stops: list[dict[str, Any]] = []
        for in_lap in in_laps:
            matching_out = [o for o in out_laps if o >= in_lap]
            out_lap = matching_out[0] if matching_out else None

            in_lap_row = grp_sorted[grp_sorted['LapNumber'] == in_lap]
            comp_before = in_lap_row['Compound'].iloc[0] if not in_lap_row.empty else None

            if out_lap is not None:
                out_lap_row = grp_sorted[grp_sorted['LapNumber'] == out_lap]
                comp_after = out_lap_row['Compound'].iloc[0] if not out_lap_row.empty else None
            else:
                comp_after = None

            stop_info = {
                'driver': driver,
                'in_lap': in_lap,
                'out_lap': out_lap,
                'compound_before': comp_before,
                'compound_after': comp_after,
            }
            driver_stops.append(stop_info)
            all_pit_stops.append(stop_info)

        driver_events[driver] = {
            'compound_sequence': stint_compounds,
            'pit_laps': in_laps,
            'pit_stops': driver_stops,
        }

    all_pit_stops.sort(key=lambda s: (s['in_lap'], s['driver']))

    return {
        'sc_lap_ranges': sc_ranges,
        'vsc_lap_ranges': vsc_ranges,
        'neutralized_lap_ranges': neutralized_ranges,
        'driver_pit_laps': driver_pit_laps,
        'driver_compound_sequence': driver_compound_seq,
        'pit_stops': all_pit_stops,
        'per_driver': driver_events,
    }
