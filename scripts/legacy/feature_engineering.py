"""
feature_engineering.py — Feature Engineering Facade (Delegates to pipeline/)
=============================================================================
Maintained for backward compatibility. Feature engineering has been
componentized into the `pipeline/features.py` module.
"""

from pipeline.features import apply_feature_engineering

__all__ = ["apply_feature_engineering"]

if __name__ == "__main__":
    from data_pipeline import main as run_pipeline
    print("Testing Feature Engineering Module Pipeline...")
    clean_laps = run_pipeline()
    if clean_laps is not None:
        engineered_data = apply_feature_engineering(clean_laps)
        columns_to_show = ['Driver', 'LapNumber', 'tyre_age', 'compound_encoded', 
                           'fuel_corrected_time', 'track_evolution', 'TrackTemp']
        print(engineered_data[columns_to_show].head(10))
