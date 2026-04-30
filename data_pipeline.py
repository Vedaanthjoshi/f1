import os
import pandas as pd
import fastf1

# Ensure cache directory exists and enable it
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

def load_race_session(season, race):
    """
    Loads the race session data for a given season and race.
    Caches the results locally via FastF1 caching.
    """
    session = fastf1.get_session(season, race, 'R')
    session.load()
    return session

def is_red_flag_race(session):
    """
    Checks if there were any red flags during the race session.
    TrackStatus '5' corresponds to a red flag.
    """
    statuses = session.track_status['Status'].unique()
    return '5' in statuses

def clean_lap_data(laps):
    """
    Cleans the raw lap data according to the PRD rules:
    - Removes in-laps and out-laps
    - Removes laps with Safety Car, VSC, or non-green track status
    - Converts lap times to seconds for easier modeling
    """
    df = laps.copy()
    
    # 1. Remove in-laps and out-laps (they have a non-null PitInTime or PitOutTime)
    df = df[pd.isnull(df['PitInTime']) & pd.isnull(df['PitOutTime'])]
    
    # 2. Remove SC, VSC, non-green laps
    # FastF1 TrackStatus = '1' means the lap was entirely under green flag conditions
    df = df[df['TrackStatus'] == '1']
    
    # 3. Ensure lap times are converted to seconds
    # LapTime is a timedelta, converted to total seconds.
    df = df.copy()
    df['LapTime_s'] = df['LapTime'].dt.total_seconds()
    
    # Drop rows where LapTime couldn't be calculated
    df = df.dropna(subset=['LapTime_s'])
    
    return df

def merge_weather_data(laps, weather):
    """
    Merges weather data (like track temperature) with lap data.
    Uses nearest-time matching to assign the right weather conditions to each lap.
    """
    laps_sorted = laps.sort_values('Time')
    weather_sorted = weather.sort_values('Time')
    
    # Perform an asof merge to find the nearest weather entry prior to/at the lap time
    merged = pd.merge_asof(laps_sorted, weather_sorted, on='Time', direction='backward')
    
    # Direction='nearest' can also be used, but F1 weather is typically broadcast every minute. 
    # 'backward' ensures we use the latest known weather info for the lap.
    return merged

def main():
    season = 2023
    race = 'Bahrain'
    
    print(f"Loading {season} {race} Grand Prix...")
    session = load_race_session(season, race)
    
    if is_red_flag_race(session):
        print(f"Race {season} {race} is invalid due to a red flag. Excluding from dataset.")
        return None
    
    print("Race valid (no red flags). Processing laps...")
    raw_laps = session.laps
    
    # Data Cleaning
    print(f"Initial lap count: {len(raw_laps)}")
    clean_laps = clean_lap_data(raw_laps)
    print(f"Lap count after filtering (green-flag, racing laps only): {len(clean_laps)}")
    
    # Weather Integration
    weather_data = session.weather_data
    final_laps = merge_weather_data(clean_laps, weather_data)
    
    print("Data Pipeline complete. Ready for feature engineering.")
    print("\nSample Output (Laps + TrackTemp):")
    print(final_laps[['Driver', 'LapNumber', 'LapTime_s', 'TrackStatus', 'TrackTemp']].head())
    
    return final_laps

if __name__ == "__main__":
    df = main()
