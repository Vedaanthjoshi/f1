import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def prepare_ml_data(df):
    """
    Prepares the engineered data for ML modeling.
    Calculates the target variable: 'lap_time_delta', which is the difference 
    between the current lap and the fresh-tyre baseline of that stint.
    """
    ml_df = df.copy()
    
    # Let's drop NaN values from the features we need
    required_cols = ['Driver', 'Stint', 'tyre_age', 'compound_encoded', 
                     'fuel_corrected_time', 'track_evolution', 'TrackTemp']
    ml_df = ml_df.dropna(subset=required_cols)
    
    # 1. Calculate the fresh-tyre baseline for each Driver's Stint
    # We define the baseline as the lap with the minimum tyre_age in that stint.
    baseline_df = ml_df.loc[ml_df.groupby(['Driver', 'Stint'])['tyre_age'].idxmin()]
    baseline_df = baseline_df[['Driver', 'Stint', 'fuel_corrected_time']]
    baseline_df = baseline_df.rename(columns={'fuel_corrected_time': 'baseline_pace'})
    
    # Merge baseline_pace back into the main dataset
    ml_df = pd.merge(ml_df, baseline_df, on=['Driver', 'Stint'], how='left')
    
    # 2. Calculate the target variable 'lap_time_delta'
    # lap_time_delta = current fuel corrected pace - baseline pace of that stint
    ml_df['lap_time_delta'] = ml_df['fuel_corrected_time'] - ml_df['baseline_pace']
    
    return ml_df

def train_degradation_model(ml_df):
    """
    Trains a Random Forest Regressor to predict tyre degradation (lap_time_delta).
    Follows PRD guidelines: 80/20 train/test split, shared global model.
    """
    # Features mentioned in PRD: 
    # tyre_age, compound_encoded, track_evolution, track_temp, and we use baseline_pace 
    # (as the logical usage of 'fuel_corrected_time' input) to ground the delta.
    features = ['tyre_age', 'compound_encoded', 'baseline_pace', 'track_evolution', 'TrackTemp']
    target = 'lap_time_delta'
    
    X = ml_df[features]
    y = ml_df[target]
    
    print(f"Training global Random Forest model on {len(X)} clean racing laps...")
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Model definition based on PRD Hyperparameters
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate score on test set (R^2 Score)
    score = model.score(X_test, y_test)
    print(f"Model Training Complete. R^2 Score on Test Data: {score:.3f}")
    
    # Feature Importance for Validation (as requested by PRD)
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for feature, imp in zip(features, importances):
        print(f" - {feature}: {imp:.3f}")
        
    return model, features

def predict_future_laps(model, features, current_state, laps_ahead=15):
    """
    A helper function to predict lap times for 'Stay Out' vs 'Pit Now'
    Will be fully utilized in Phase 4 (Simulation logic).
    """
    # This prepares the groundwork for Phase 4
    pass

if __name__ == "__main__":
    from data_pipeline import main as run_pipeline
    from feature_engineering import apply_feature_engineering
    
    print("Executing Phase 1 and 2 to get raw data...")
    clean_laps = run_pipeline()
    if clean_laps is not None:
        engineered_data = apply_feature_engineering(clean_laps)
        
        print("\n--- Starting Phase 3 (Machine Learning) ---")
        ml_data = prepare_ml_data(engineered_data)
        
        # Train Model
        model, features = train_degradation_model(ml_data)
        
        # Output a sample of targets to verify it logically makes sense
        print("\nSample of Data + Predicted Degradation (lap_time_delta target):")
        sample_display = ml_data[['Driver', 'LapNumber', 'tyre_age', 'compound_encoded', 
                                  'baseline_pace', 'fuel_corrected_time', 'lap_time_delta']].head(5)
        print(sample_display)
