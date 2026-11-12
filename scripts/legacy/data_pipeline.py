"""
data_pipeline.py — Telemetry Ingestion Facade (Delegates to pipeline/)
======================================================================
Maintained for backward compatibility. Ingestion logic has been
componentized into the `pipeline/` package.
"""

from pipeline.ingestion import (
    load_race_session,
    is_red_flag_race,
    clean_lap_data,
    merge_weather_data,
    fetch_season_schedule,
    process_telemetry_pipeline,
)

__all__ = [
    "load_race_session",
    "is_red_flag_race",
    "clean_lap_data",
    "merge_weather_data",
    "fetch_season_schedule",
    "process_telemetry_pipeline",
]

def main():
    season = 2023
    race = 'Bahrain'
    print(f"Loading {season} {race} Grand Prix...")
    session = load_race_session(season, race)
    if is_red_flag_race(session):
        print(f"Race {season} {race} is invalid due to a red flag.")
        return None
    clean_laps = clean_lap_data(session.laps)
    final_laps = merge_weather_data(clean_laps, session.weather_data)
    print("Data Pipeline complete. Sample output:")
    print(final_laps[['Driver', 'LapNumber', 'LapTime_s', 'TrackStatus', 'TrackTemp']].head())
    return final_laps

if __name__ == "__main__":
    df = main()
