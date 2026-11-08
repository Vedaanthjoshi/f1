"""
pipeline/features.py — Physical Feature Derivation
===================================================
Applies domain physics transformations to Formula 1 telemetry:
  - Fuel burnoff compensation (~1.7kg/lap @ 0.03s/kg), clamped ≥ 2 kg
  - Circuit rubbering-in grip accretion proxy (causal expanding median)
  - Numeric categorical compound encoding (Soft=1, Med=2, Hard=3)
  - Tyre life mapping
  - Circuit physics features (abrasion, length, corners, compound hardness)

Reviewer-concern fixes applied
--------------------------------
Fix 1 : Fuel mass clamped to minimum 2 kg so the correction never goes
        negative (was unbounded — went negative after Lap 65).
Fix 2 : track_evolution now uses an EXPANDING window over lap-level
        session medians so that at Lap L only laps 1..L contribute.
        This eliminates temporal lookahead leakage present in the original
        per-lap groupby which used all laps in the session simultaneously.
Fix 6 : Circuit physics features added (abrasion index, circuit length,
        corner count, Pirelli compound hardness). These allow the model to
        generalise to unseen circuits by their physical characteristics
        rather than relying solely on a categorical circuit name embedding.
"""

import numpy as np
import pandas as pd


# ── Circuit Physics Lookup Table ─────────────────────────────────────────────
# Sources: Pirelli race preview documents, FIA circuit data, official lap records.
# abrasion_index : Pirelli's 1-5 scale (5 = most abrasive, destroys tyres fastest)
# circuit_length_km : official FIA circuit length
# num_corners : total corners per lap
# compound_hardness : softest Pirelli C-spec available at this event (1=hardest, 5=softest)
#   Soft at Bahrain (C2) = 2;  Soft at Monaco (C5) = 5

CIRCUIT_PHYSICS: dict[str, dict] = {
    'Bahrain Grand Prix':           {'abrasion_index': 4, 'circuit_length_km': 5.412, 'num_corners': 15, 'compound_hardness': 1},
    'Saudi Arabian Grand Prix':     {'abrasion_index': 2, 'circuit_length_km': 6.174, 'num_corners': 27, 'compound_hardness': 3},
    'Australian Grand Prix':        {'abrasion_index': 2, 'circuit_length_km': 5.278, 'num_corners': 16, 'compound_hardness': 3},
    'Japanese Grand Prix':          {'abrasion_index': 3, 'circuit_length_km': 5.807, 'num_corners': 18, 'compound_hardness': 2},
    'Chinese Grand Prix':           {'abrasion_index': 3, 'circuit_length_km': 5.451, 'num_corners': 16, 'compound_hardness': 2},
    'Miami Grand Prix':             {'abrasion_index': 3, 'circuit_length_km': 5.412, 'num_corners': 19, 'compound_hardness': 3},
    'Emilia Romagna Grand Prix':    {'abrasion_index': 2, 'circuit_length_km': 4.909, 'num_corners': 19, 'compound_hardness': 3},
    'Monaco Grand Prix':            {'abrasion_index': 1, 'circuit_length_km': 3.337, 'num_corners': 19, 'compound_hardness': 5},
    'Canadian Grand Prix':          {'abrasion_index': 2, 'circuit_length_km': 4.361, 'num_corners': 14, 'compound_hardness': 3},
    'Spanish Grand Prix':           {'abrasion_index': 3, 'circuit_length_km': 4.657, 'num_corners': 16, 'compound_hardness': 2},
    'Austrian Grand Prix':          {'abrasion_index': 2, 'circuit_length_km': 4.318, 'num_corners': 10, 'compound_hardness': 3},
    'British Grand Prix':           {'abrasion_index': 4, 'circuit_length_km': 5.891, 'num_corners': 18, 'compound_hardness': 2},
    'Hungarian Grand Prix':         {'abrasion_index': 3, 'circuit_length_km': 4.381, 'num_corners': 14, 'compound_hardness': 3},
    'Belgian Grand Prix':           {'abrasion_index': 3, 'circuit_length_km': 7.004, 'num_corners': 19, 'compound_hardness': 2},
    'Dutch Grand Prix':             {'abrasion_index': 4, 'circuit_length_km': 4.259, 'num_corners': 14, 'compound_hardness': 2},
    'Italian Grand Prix':           {'abrasion_index': 1, 'circuit_length_km': 5.793, 'num_corners': 11, 'compound_hardness': 4},
    'Azerbaijan Grand Prix':        {'abrasion_index': 2, 'circuit_length_km': 6.003, 'num_corners': 20, 'compound_hardness': 4},
    'Singapore Grand Prix':         {'abrasion_index': 2, 'circuit_length_km': 4.940, 'num_corners': 23, 'compound_hardness': 3},
    'United States Grand Prix':     {'abrasion_index': 4, 'circuit_length_km': 5.513, 'num_corners': 20, 'compound_hardness': 2},
    'Mexico City Grand Prix':       {'abrasion_index': 3, 'circuit_length_km': 4.304, 'num_corners': 17, 'compound_hardness': 2},
    'São Paulo Grand Prix':         {'abrasion_index': 4, 'circuit_length_km': 4.309, 'num_corners': 15, 'compound_hardness': 2},
    'Las Vegas Grand Prix':         {'abrasion_index': 2, 'circuit_length_km': 6.201, 'num_corners': 17, 'compound_hardness': 4},
    'Qatar Grand Prix':             {'abrasion_index': 5, 'circuit_length_km': 5.419, 'num_corners': 16, 'compound_hardness': 1},
    'Abu Dhabi Grand Prix':         {'abrasion_index': 2, 'circuit_length_km': 5.281, 'num_corners': 16, 'compound_hardness': 3},
    # Fallback for any unlisted circuit — use global averages
    '_default':                     {'abrasion_index': 3, 'circuit_length_km': 5.1,   'num_corners': 16, 'compound_hardness': 3},
}

# Column names added to DataFrame (must stay consistent with trainer feature list)
CIRCUIT_FEATURE_COLS = ['abrasion_index', 'circuit_length_km', 'num_corners', 'compound_hardness']


def _get_circuit_physics(track_name: str) -> dict:
    """Lookup circuit physics, with fallback to default averages for unknown circuits."""
    return CIRCUIT_PHYSICS.get(track_name, CIRCUIT_PHYSICS['_default'])


def apply_circuit_features(df: pd.DataFrame, track_name: str | None = None) -> pd.DataFrame:
    """
    Attach circuit physics features to each row.
    If track_name is provided (e.g. 'Bahrain Grand Prix'), use it directly.
    Otherwise, try to read a 'Track' column already present in the DataFrame.
    """
    result = df.copy()
    if track_name is not None:
        physics = _get_circuit_physics(track_name)
        for col, val in physics.items():
            result[col] = val
    elif 'Track' in result.columns:
        for col in CIRCUIT_FEATURE_COLS:
            result[col] = result['Track'].map(
                lambda t, c=col: _get_circuit_physics(t).get(c, CIRCUIT_PHYSICS['_default'][c])
            )
    else:
        # No track info available — apply defaults
        physics = CIRCUIT_PHYSICS['_default']
        for col, val in physics.items():
            result[col] = val
    return result


def apply_feature_engineering(df: pd.DataFrame, track_name: str | None = None) -> pd.DataFrame:
    """
    Derive domain features required by the degradation ML model:
      - tyre_age          : consecutive stint laps on current tyre set
      - compound_encoded  : 1=SOFT, 2=MEDIUM, 3=HARD
      - fuel_corrected_time : lap time normalised for burned fuel (clamped)
      - track_evolution   : causal rubbering-in proxy (no lookahead)
      - abrasion_index    : Pirelli circuit surface roughness (1-5)
      - circuit_length_km : official circuit length
      - num_corners       : corners per lap
      - compound_hardness : softest C-spec available (1-5)

    Parameters
    ----------
    df : pd.DataFrame
        Raw lap DataFrame from FastF1 (must have LapNumber, TyreLife, Compound,
        LapTime_s, optionally TrackTemp).
    track_name : str, optional
        Grand Prix name (e.g. 'Bahrain Grand Prix'). If omitted the function
        will use a 'Track' column if present in df, else apply defaults.
    """
    engineered = df.copy()

    # ── 1. Tyre life tracking ────────────────────────────────────────────────
    engineered['tyre_age'] = engineered['TyreLife']

    # ── 2. Compound encoding (dry compounds only) ────────────────────────────
    compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
    engineered['compound_encoded'] = engineered['Compound'].map(compound_map)

    # ── 3. Fuel correction (Fix 1: clamp fuel mass to ≥ 2 kg) ───────────────
    # 110 kg starting fuel, burning ~1.7 kg/lap, correction ~0.03 s per kg.
    # Without the clamp the correction added time after lap 65 (negative mass).
    fuel_mass = np.maximum(110.0 - 1.7 * engineered['LapNumber'], 2.0)
    fuel_correction = fuel_mass * 0.03
    engineered['fuel_corrected_time'] = engineered['LapTime_s'] - fuel_correction

    # ── 4. Track evolution proxy (Fix 2: causal expanding window) ───────────
    # Step A: compute per-lap session median across all cars at that lap
    lap_medians = (
        engineered
        .groupby('LapNumber')['LapTime_s']
        .median()
        .sort_index()
    )
    # Step B: expanding (cumulative) mean of those per-lap medians.
    # At lap L the value reflects only laps 1..L — no future information.
    # Using .shift(1) means strictly past laps; keeps lap-1 as NaN → ffill.
    causal_evo = (
        lap_medians
        .expanding()
        .mean()
        .shift(1)         # strict past-only: lap L sees average of laps 1..L-1
        .ffill()          # lap 1 inherits the global session median as seed
        .bfill()
    )
    engineered['track_evolution'] = engineered['LapNumber'].map(causal_evo)

    # ── 5. Circuit physics features (Fix 6) ─────────────────────────────────
    # These let the model generalise to unseen circuits by physical properties
    # rather than relying solely on the circuit name embedding.
    engineered = apply_circuit_features(engineered, track_name=track_name)

    return engineered


