"""
data_collector.py — Historical F1 Lap Data Collector
=====================================================
Fetches race session data via FastF1 for the specified seasons, applies the
full fixed feature engineering pipeline, and saves the result to
data/historical_laps.csv for model training.

Fixed pipeline (applied automatically via facade imports):
  - Fuel mass clamped to >= 2 kg (Fix 1)
  - track_evolution uses causal expanding window (Fix 2)
  - lap_time_delta computed without baseline_pace leakage (Fix 3)
"""
import os
import pandas as pd
import fastf1
from data_pipeline import load_race_session, is_red_flag_race, clean_lap_data, merge_weather_data
from feature_engineering import apply_feature_engineering
from ml_model import prepare_ml_data

os.makedirs('data', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

def collect_season_data(season):
    print(f"--- Starting data collection for the {season} season ---")
    schedule = fastf1.get_event_schedule(season)
    # Filter for valid races (excluding testing)
    races = schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()
    
    all_ml_data = []
    
    for race in races:
        print(f"\nProcessing {race}...")
        try:
            session = load_race_session(season, race)
            
            if is_red_flag_race(session):
                print(f"Skipping {race} due to red flag.")
                continue
                
            raw_laps = session.laps
            clean_laps = clean_lap_data(raw_laps)
            
            if clean_laps.empty:
                print(f"Skipping {race}: no clean laps found.")
                continue
                
            final_laps = merge_weather_data(clean_laps, session.weather_data)
            engineered_data = apply_feature_engineering(final_laps, track_name=race)
            ml_data = prepare_ml_data(engineered_data)
            
            # Add context features for the global model
            ml_data['Track'] = race
            ml_data['Season'] = season
            
            all_ml_data.append(ml_data)
            print(f"Successfully processed {len(ml_data)} laps for {race}.")
            
        except Exception as e:
            print(f"Error processing {race}: {e}")
            
    if all_ml_data:
        combined_df = pd.concat(all_ml_data, ignore_index=True)
        return combined_df
    return pd.DataFrame()

if __name__ == "__main__":
    # Collect full Ground Effect era: 2022-2025
    seasons_to_collect = [2022, 2023, 2024, 2025]
    
    out_path = os.path.join("data", "historical_laps.csv")

    # Always start fresh so old leaky data is not mixed with new clean data
    if os.path.exists(out_path):
        backup = out_path.replace(".csv", "_backup_leaky.csv")
        os.rename(out_path, backup)
        print(f"[BACKUP] Old dataset moved to {backup}")

    global_dataset = []
    for season in seasons_to_collect:
        season_df = collect_season_data(season)
        if not season_df.empty:
            global_dataset.append(season_df)
            
    if global_dataset:
        final_dataset = pd.concat(global_dataset, ignore_index=True)
        final_dataset.to_csv(out_path, index=False)
        print(f"\n[SUCCESS] Clean dataset created at {out_path} ({len(final_dataset):,} total clean racing laps)")
    else:
        print("\n[ERROR] Failed to collect any data.")

