"""Construct the first observed stint reference and the legacy delta target."""

import pandas as pd


def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the first retained lap within each event, driver and stint."""
    ml_df = df.copy()

    required_cols = [
        'Driver', 'Stint', 'tyre_age', 'compound_encoded',
        'fuel_corrected_time', 'track_evolution', 'TrackTemp'
    ]
    ml_df = ml_df.dropna(subset=required_cols)

    keys = [col for col in ("Season", "Track", "Driver", "Stint") if col in ml_df]
    ordered = ml_df.sort_values("Time" if "Time" in ml_df else "LapNumber")
    references = ordered.drop_duplicates(keys)[keys + ["fuel_corrected_time"]]
    references = references.rename(columns={"fuel_corrected_time": "baseline_pace"})
    ml_df = ml_df.merge(references, on=keys, validate="many_to_one")
    ml_df['lap_time_delta'] = ml_df['fuel_corrected_time'] - ml_df['baseline_pace']

    return ml_df

