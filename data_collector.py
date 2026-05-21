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
            engineered_data = apply_feature_engineering(final_laps)
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
    # Fetch all Ground Effect era seasons!
    seasons_to_collect = [2024, 2025] 
    
    global_dataset = []
    for season in seasons_to_collect:
        season_df = collect_season_data(season)
        if not season_df.empty:
            global_dataset.append(season_df)
            
    if global_dataset:
        final_dataset = pd.concat(global_dataset, ignore_index=True)
        out_path = os.path.join("data", "historical_laps.csv")
        
        # Append if exists, write new if not
        if os.path.exists(out_path):
            final_dataset.to_csv(out_path, mode='a', header=False, index=False)
            print(f"\n[SUCCESS] Appended new data to {out_path} ({len(final_dataset)} clean racing laps added)")
        else:
            final_dataset.to_csv(out_path, index=False)
            print(f"\n[SUCCESS] Global dataset created at {out_path} ({len(final_dataset)} total clean racing laps)")
    else:
        print("\n[ERROR] Failed to collect any data.")
