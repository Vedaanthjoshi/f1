"""
pipeline/ingestion.py — Telemetry Ingestion, Sanitization & Cache Management
============================================================================
Handles FastF1 session retrieval, disk caching, red-flag session validation,
in/out-lap and non-green flag lap stripping, and asynchronous weather integration.
"""

from __future__ import annotations
import os
from typing import Any
import fastf1
import pandas as pd

from pipeline.features import apply_feature_engineering
from pipeline.baseline import prepare_ml_data
from engine.pit_loss import calculate_pit_loss

CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_session(season: int, race: str):
    """Load a Grand Prix race session with weather telemetry enabled."""
    session = fastf1.get_session(season, race, 'R')
    session.load(weather=True, messages=False)
    return session


def is_red_flag_race(session) -> bool:
    """
    Check if a session experienced red flag stoppages (Status '5').
    Red flag races are excluded to preserve tyre stint continuity.
    """
    statuses = session.track_status['Status'].unique()
    return '5' in statuses


def clean_lap_data(laps: fastf1.core.Laps) -> pd.DataFrame:
    """
    Sanitize raw lap telemetry:
      - Strips in-laps and out-laps (non-null PitInTime / PitOutTime)
      - Filters out non-green flag laps (TrackStatus == '1')
      - Converts LapTime timedelta into float seconds (LapTime_s)
    """
    df = laps.copy()

    # 1. Remove in-laps and out-laps
    df = df[pd.isnull(df['PitInTime']) & pd.isnull(df['PitOutTime'])]

    # 2. Keep strictly green-flag racing laps
    df = df[df['TrackStatus'] == '1']

    # 3. Convert lap times to float seconds
    df = df.copy()
    df['LapTime_s'] = df['LapTime'].dt.total_seconds()
    df = df.dropna(subset=['LapTime_s'])

    return df


def merge_weather_data(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Attach track temperature telemetry to laps using backward nearest-prior join.
    """
    laps_sorted = laps.sort_values('Time')
    weather_sorted = weather.sort_values('Time')
    return pd.merge_asof(laps_sorted, weather_sorted, on='Time', direction='backward')


def fetch_season_schedule(year: int) -> list[str]:
    """Fetch valid Grand Prix race names for a championship calendar year."""
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()


def process_telemetry_pipeline(session) -> tuple[dict[str, Any] | None, str | None]:
    """
    Execute complete backend data preparation pipeline on a loaded FastF1 session:
      1. Red flag validation
      2. Lap cleaning & weather merge
      3. Physical feature engineering
      4. Stint baseline isolation & delta target calculation
      5. Empirical pit loss calculation
    """
    if is_red_flag_race(session):
        return None, "Error: Race contains a Red Flag. Filtered out per PRD constraints to maintain valid degradation curves."

    clean_laps = clean_lap_data(session.laps)
    if clean_laps.empty:
        return None, "Error: No clean racing laps found in this session."

    final_laps = merge_weather_data(clean_laps, session.weather_data)
    engineered_data = apply_feature_engineering(final_laps)
    ml_data = prepare_ml_data(engineered_data)
    pit_loss = calculate_pit_loss(session)

    return {
        "ml_data": ml_data,
        "pit_loss": pit_loss,
    }, None
