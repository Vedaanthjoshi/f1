"""
pipeline package — Formula 1 Telemetry Ingestion & Feature Engineering Pipeline
"""

from pipeline.features import apply_feature_engineering
from pipeline.baseline import prepare_ml_data
from pipeline.ingestion import (
    load_race_session,
    is_red_flag_race,
    clean_lap_data,
    merge_weather_data,
    fetch_season_schedule,
    process_telemetry_pipeline,
)

__all__ = [
    "apply_feature_engineering",
    "prepare_ml_data",
    "load_race_session",
    "is_red_flag_race",
    "clean_lap_data",
    "merge_weather_data",
    "fetch_season_schedule",
    "process_telemetry_pipeline",
]

