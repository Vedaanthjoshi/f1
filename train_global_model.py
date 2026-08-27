import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

def train_global():
    data_path = os.path.join("data", "historical_laps.csv")
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run data_collector.py first.")
        return

    print(f"Loading global dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Let's drop NaN values from the features we need
    required_cols = ['Driver', 'Track', 'tyre_age', 'compound_encoded', 
                     'baseline_pace', 'track_evolution', 'TrackTemp', 'lap_time_delta']
    df = df.dropna(subset=required_cols)
    
    print(f"Total clean rows for training: {len(df)}")
    
    # We will use 'Track' and 'Driver' as categorical features.
    df['Track'] = df['Track'].astype('category')
    df['Driver'] = df['Driver'].astype('category')
    
    features = ['tyre_age', 'compound_encoded', 'baseline_pace', 'track_evolution', 'TrackTemp', 'Track', 'Driver']
    target = 'lap_time_delta'
    
    X = df[features]
    y = df[target]
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print("Training global XGBoost model (GPU Accelerated)...")
    # Initialize XGBoost Regressor
    # Using 'hist' tree method and enabling categorical support.
    # We will try to use 'cuda' (GPU). If it fails (e.g. CUDA not installed), fallback to CPU.
    try:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            tree_method='hist',
            device='cuda',
            enable_categorical=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("[SUCCESS] Successfully trained on GPU!")
    except Exception as e:
        print(f"GPU training failed (falling back to CPU): {e}")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            tree_method='hist',
            enable_categorical=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("[SUCCESS] Successfully trained on CPU!")
        
    # Evaluate score on test set (R^2 Score)
    score = model.score(X_test, y_test)
    print(f"Model Training Complete. R^2 Score on Test Data: {score:.3f}")
    
    # Feature Importance
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for feature, imp in zip(features, importances):
        print(f" - {feature}: {imp:.3f}")
        
    # Save the model and the expected feature columns
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'global_degradation_model.joblib')
    
    joblib.dump({
        'model': model,
        'features': features
    }, model_path)
    print(f"\n[SAVED] Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_global()
