import pandas as pd
import numpy as pd_numpy # not strictly needed but good practice

def apply_feature_engineering(df):
    """
    Applies the feature engineering steps defined in Phase 2 of the PRD:
    - Fuel correction calculation
    - Track evolution proxy
    - Compound encoding
    - Tyre life tracking
    """
    # Work on a copy to avoid SettingWithCopy warnings
    engineered = df.copy()
    
    # 1. Tyre life tracking
    # FastF1 already tracks this via the 'TyreLife' column.
    engineered['tyre_age'] = engineered['TyreLife']
    
    # 2. Compound encoding
    # PRD constraint: Soft=1, Medium=2, Hard=3
    # FastF1 outputs compounds as uppercase strings ('SOFT', 'MEDIUM', 'HARD')
    compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
    engineered['compound_encoded'] = engineered['Compound'].map(compound_map)
    
    # Optional logic for edge cases where Compound is not mapped:
    # If the race had INTERMEDIATE/WET or UNKNOWN compounds, it would map to NaN. 
    # Since PRD scope bounds limit to 'dry races only', we're safe to just map the three dry tires.
    
    # 3. Fuel correction calculation
    # PRD Formula:
    # fuel_mass = 110 - (1.7 * lap_number)
    # fuel_correction = fuel_mass * 0.03  # ~0.03s per kg
    # corrected_lap_time = lap_time - fuel_correction
    
    fuel_mass = 110.0 - (1.7 * engineered['LapNumber'])
    fuel_correction = fuel_mass * 0.03
    engineered['fuel_corrected_time'] = engineered['LapTime_s'] - fuel_correction
    
    # 4. Track evolution proxy
    # PRD defined track evolution as:
    # track_evo = df.groupby('LapNumber')['LapTime'].median()
    # df['track_evolution'] = df['LapNumber'].map(track_evo)
    
    # We use LapTime_s (seconds) here because we want the numerical values for the model
    track_evo = engineered.groupby('LapNumber')['LapTime_s'].median()
    engineered['track_evolution'] = engineered['LapNumber'].map(track_evo)
    
    # Keep final dataset clean by only keeping what the ML model (Phase 3) needs?
    # PRD specifies 5 features for the model:
    # tyre_age, compound_encoded, fuel_corrected_time, track_evolution, track_temp
    # It's usually best to keep all columns in the dataframe for tracing, so we just return it all.
    
    return engineered

if __name__ == "__main__":
    from data_pipeline import main as run_pipeline
    print("Testing Feature Engineering Module Pipeline...")
    
    # Get the clean pipeline data
    clean_laps = run_pipeline()
    if clean_laps is not None:
        engineered_data = apply_feature_engineering(clean_laps)
        print("\nFeature Engineering Complete. Snapshot of engineering columns:")
        columns_to_show = ['Driver', 'LapNumber', 'tyre_age', 'compound_encoded', 
                           'fuel_corrected_time', 'track_evolution', 'TrackTemp']
        print(engineered_data[columns_to_show].head(10))
