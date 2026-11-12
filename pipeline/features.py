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

CIRCUIT_PHYSICS: dict[str, dict] = {
    'Bahrain Grand Prix':           {'abrasion_index': 4, 'circuit_length_km': 5.412, 'num_corners': 15, 'race_laps': 57},
    'Saudi Arabian Grand Prix':     {'abrasion_index': 2, 'circuit_length_km': 6.174, 'num_corners': 27, 'race_laps': 50},
    'Australian Grand Prix':        {'abrasion_index': 2, 'circuit_length_km': 5.278, 'num_corners': 16, 'race_laps': 58},
    'Japanese Grand Prix':          {'abrasion_index': 3, 'circuit_length_km': 5.807, 'num_corners': 18, 'race_laps': 53},
    'Chinese Grand Prix':           {'abrasion_index': 3, 'circuit_length_km': 5.451, 'num_corners': 16, 'race_laps': 56},
    'Miami Grand Prix':             {'abrasion_index': 3, 'circuit_length_km': 5.412, 'num_corners': 19, 'race_laps': 57},
    'Emilia Romagna Grand Prix':    {'abrasion_index': 2, 'circuit_length_km': 4.909, 'num_corners': 19, 'race_laps': 63},
    'Monaco Grand Prix':            {'abrasion_index': 1, 'circuit_length_km': 3.337, 'num_corners': 19, 'race_laps': 78},
    'Canadian Grand Prix':          {'abrasion_index': 2, 'circuit_length_km': 4.361, 'num_corners': 14, 'race_laps': 70},
    'Spanish Grand Prix':           {'abrasion_index': 3, 'circuit_length_km': 4.657, 'num_corners': 16, 'race_laps': 66},
    'Austrian Grand Prix':          {'abrasion_index': 2, 'circuit_length_km': 4.318, 'num_corners': 10, 'race_laps': 71},
    'British Grand Prix':           {'abrasion_index': 4, 'circuit_length_km': 5.891, 'num_corners': 18, 'race_laps': 52},
    'Hungarian Grand Prix':         {'abrasion_index': 3, 'circuit_length_km': 4.381, 'num_corners': 14, 'race_laps': 70},
    'Belgian Grand Prix':           {'abrasion_index': 3, 'circuit_length_km': 7.004, 'num_corners': 19, 'race_laps': 44},
    'Dutch Grand Prix':             {'abrasion_index': 4, 'circuit_length_km': 4.259, 'num_corners': 14, 'race_laps': 72},
    'Italian Grand Prix':           {'abrasion_index': 1, 'circuit_length_km': 5.793, 'num_corners': 11, 'race_laps': 53},
    'Azerbaijan Grand Prix':        {'abrasion_index': 2, 'circuit_length_km': 6.003, 'num_corners': 20, 'race_laps': 51},
    'Singapore Grand Prix':         {'abrasion_index': 2, 'circuit_length_km': 4.940, 'num_corners': 23, 'race_laps': 62},
    'United States Grand Prix':     {'abrasion_index': 4, 'circuit_length_km': 5.513, 'num_corners': 20, 'race_laps': 56},
    'Mexico City Grand Prix':       {'abrasion_index': 3, 'circuit_length_km': 4.304, 'num_corners': 17, 'race_laps': 71},
    'São Paulo Grand Prix':         {'abrasion_index': 4, 'circuit_length_km': 4.309, 'num_corners': 15, 'race_laps': 71},
    'Las Vegas Grand Prix':         {'abrasion_index': 2, 'circuit_length_km': 6.201, 'num_corners': 17, 'race_laps': 50},
    'Qatar Grand Prix':             {'abrasion_index': 5, 'circuit_length_km': 5.419, 'num_corners': 16, 'race_laps': 57},
    'Abu Dhabi Grand Prix':         {'abrasion_index': 2, 'circuit_length_km': 5.281, 'num_corners': 16, 'race_laps': 58},
    # Fallback for any unlisted circuit — use global averages
    '_default':                     {'abrasion_index': 3, 'circuit_length_km': 5.1,   'num_corners': 16, 'race_laps': 60},
}

# Physical Pirelli C-Spec Allocations (C1 = 1 [hardest] to C5 = 5 [softest])
# Fixes Reviewer Concern 8: Soft/Medium/Hard are relative labels;
# this table maps the actual physical compound hardness for each event.
PIRELLI_EVENT_ALLOCATIONS: dict[str, dict[str, int]] = {
    'Bahrain Grand Prix':           {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},  # C1, C2, C3
    'Saudi Arabian Grand Prix':     {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Australian Grand Prix':        {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Japanese Grand Prix':          {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},  # C1, C2, C3
    'Chinese Grand Prix':           {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Miami Grand Prix':             {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Emilia Romagna Grand Prix':    {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Monaco Grand Prix':            {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Canadian Grand Prix':          {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Spanish Grand Prix':           {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},  # C1, C2, C3
    'Austrian Grand Prix':          {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'British Grand Prix':           {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},  # C1, C2, C3
    'Hungarian Grand Prix':         {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Belgian Grand Prix':           {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Dutch Grand Prix':             {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},  # C1, C2, C3
    'Italian Grand Prix':           {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Azerbaijan Grand Prix':        {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Singapore Grand Prix':         {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'United States Grand Prix':     {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Mexico City Grand Prix':       {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'São Paulo Grand Prix':         {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},  # C2, C3, C4
    'Las Vegas Grand Prix':         {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    'Qatar Grand Prix':             {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3},  # C1, C2, C3
    'Abu Dhabi Grand Prix':         {'HARD': 3, 'MEDIUM': 4, 'SOFT': 5},  # C3, C4, C5
    '_default':                     {'HARD': 2, 'MEDIUM': 3, 'SOFT': 4},
}

# The complete 14-Feature Physics-Informed Suite
FEATURE_COLS_14 = [
    'tyre_age', 'compound_encoded', 'compound_hardness',
    'track_evolution', 'TrackTemp', 'fuel_mass',
    'abrasion_index', 'circuit_length_km', 'num_corners',
    'corner_density', 'thermal_load', 'abrasion_load',
    'wear_cliff', 'compound_age_interaction'
]

# Column names added to DataFrame (must stay consistent with trainer feature list)
CIRCUIT_FEATURE_COLS = [
    'abrasion_index', 'circuit_length_km', 'num_corners', 'compound_hardness',
    'thermal_load', 'abrasion_load', 'corner_density', 'wear_cliff'
]


def _get_circuit_physics(track_name: str) -> dict:
    """Lookup circuit physics, with fallback to default averages for unknown circuits."""
    return CIRCUIT_PHYSICS.get(track_name, CIRCUIT_PHYSICS['_default'])


def apply_circuit_features(df: pd.DataFrame, track_name: str | None = None) -> pd.DataFrame:
    """
    Attach circuit physics features and physical Pirelli C-compound hardness to each row.
    """
    result = df.copy()
    
    # 1. Base circuit physics (abrasion, length, corners)
    for col in ['abrasion_index', 'circuit_length_km', 'num_corners']:
        if track_name is not None:
            val = _get_circuit_physics(track_name).get(col, CIRCUIT_PHYSICS['_default'][col])
            result[col] = val
        elif 'Track' in result.columns:
            result[col] = result['Track'].map(
                lambda t, c=col: _get_circuit_physics(t).get(c, CIRCUIT_PHYSICS['_default'][c])
            )
        else:
            result[col] = CIRCUIT_PHYSICS['_default'][col]

    # 2. Physical Pirelli C-Spec Compound Hardness (1 = C1 hardest, 5 = C5 softest)
    # Fixes Reviewer Concern 8: Differentiates physical rubber hardness from tactical 1/2/3 labels.
    def _lookup_c_hardness(row):
        t = track_name if track_name is not None else row.get('Track', '_default')
        alloc = PIRELLI_EVENT_ALLOCATIONS.get(t, PIRELLI_EVENT_ALLOCATIONS['_default'])
        # Try raw compound string first
        comp_str = str(row.get('Compound', '')).upper()
        if comp_str in alloc:
            return alloc[comp_str]
        # Fall back to compound_encoded: 1=SOFT, 2=MEDIUM, 3=HARD
        enc = row.get('compound_encoded', 2)
        if enc == 1:
            return alloc.get('SOFT', 4)
        elif enc == 3:
            return alloc.get('HARD', 2)
        else:
            return alloc.get('MEDIUM', 3)

    result['compound_hardness'] = result.apply(_lookup_c_hardness, axis=1)

    return result


def apply_feature_engineering(df: pd.DataFrame, track_name: str | None = None) -> pd.DataFrame:
    """
    Derive domain features required by the degradation ML model:
      - tyre_age                : consecutive stint laps on current tyre set
      - compound_encoded        : 1=SOFT, 2=MEDIUM, 3=HARD
      - compound_hardness       : Pirelli physical C-spec (1=C1 to 5=C5)
      - fuel_mass               : remaining fuel mass clamped >= 2 kg
      - fuel_corrected_time     : lap time normalised for burned fuel
      - track_evolution         : causal rubbering-in proxy (expanding lag-1 median)
      - abrasion_index          : Pirelli circuit surface roughness (1-5)
      - circuit_length_km       : official circuit length
      - num_corners             : corners per lap
      - corner_density          : num_corners / circuit_length_km
      - thermal_load            : tyre_age * TrackTemp
      - abrasion_load           : tyre_age * abrasion_index
      - wear_cliff              : (tyre_age / 15.0) ** 2.0
      - compound_age_interaction: tyre_age * compound_hardness
    """
    engineered = df.copy()

    # ── 1. Tyre life tracking ────────────────────────────────────────────────
    engineered['tyre_age'] = engineered['TyreLife']

    # ── 2. Compound encoding (dry compounds only) ────────────────────────────
    compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
    engineered['compound_encoded'] = engineered['Compound'].map(compound_map).fillna(2)

    # ── 3. Fuel correction (circuit-specific burn rate, clamped >= 2 kg) ────
    # ~105 kg typical starting race fuel burned evenly across scheduled race laps.
    if track_name is not None:
        r_laps = _get_circuit_physics(track_name).get('race_laps', 60)
        burn_per_lap = 105.0 / max(float(r_laps), 1.0)
    elif 'Track' in engineered.columns:
        burn_per_lap = engineered['Track'].map(
            lambda t: 105.0 / max(float(_get_circuit_physics(t).get('race_laps', 60)), 1.0)
        )
    else:
        burn_per_lap = 1.7

    fuel_mass = np.maximum(105.0 - burn_per_lap * engineered['LapNumber'], 2.0)
    fuel_correction = fuel_mass * 0.03
    engineered['fuel_mass'] = fuel_mass
    engineered['fuel_corrected_time'] = engineered['LapTime_s'] - fuel_correction

    # ── 4. Track evolution proxy (Fix 2: causal expanding window per race) ──
    race_group_cols: list[str] = []
    if track_name is None:
        if 'Season' in engineered.columns:
            race_group_cols.append('Season')
        if 'Track' in engineered.columns:
            race_group_cols.append('Track')

    if race_group_cols:
        lap_medians = (
            engineered
            .groupby(race_group_cols + ['LapNumber'])['LapTime_s']
            .median()
            .sort_index()
        )
        causal_evo = (
            lap_medians
            .groupby(level=list(range(len(race_group_cols))))
            .transform(lambda s: s.expanding().mean().shift(1).ffill().bfill())
        )
        evo_key = engineered.set_index(race_group_cols + ['LapNumber']).index
        engineered['track_evolution'] = evo_key.map(causal_evo).to_numpy()
    else:
        lap_medians = (
            engineered
            .groupby('LapNumber')['LapTime_s']
            .median()
            .sort_index()
        )
        causal_evo = (
            lap_medians
            .expanding()
            .mean()
            .shift(1)
            .ffill()
            .bfill()
        )
        engineered['track_evolution'] = engineered['LapNumber'].map(causal_evo)

    # ── 5. Circuit physics features & compound hardness ─────────────────────
    engineered = apply_circuit_features(engineered, track_name=track_name)

    # ── 6. Physics Interactions & Non-linear Wear (14-Feature Suite) ────────
    engineered['thermal_load'] = engineered['tyre_age'] * engineered['TrackTemp']
    engineered['abrasion_load'] = engineered['tyre_age'] * engineered['abrasion_index']
    engineered['corner_density'] = engineered['num_corners'] / engineered['circuit_length_km']
    engineered['wear_cliff'] = (engineered['tyre_age'] / 15.0) ** 2.0
    engineered['compound_age_interaction'] = engineered['tyre_age'] * engineered['compound_hardness']

    return engineered


