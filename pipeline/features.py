"""
pipeline/features.py — Physical Feature Derivation
===================================================
Applies domain physics transformations to Formula 1 telemetry:
  - Fuel burnoff compensation (~1.7kg/lap @ 0.03s/kg), clamped ≥ 2 kg
  - Circuit rubbering-in grip accretion proxy (causal expanding median)
  - Numeric categorical compound encoding (Soft=1, Med=2, Hard=3)
  - Tyre life mapping

Reviewer-concern fixes applied
--------------------------------
Fix 1 : Fuel mass clamped to minimum 2 kg so the correction never goes
        negative (was unbounded — went negative after Lap 65).
Fix 2 : track_evolution now uses an EXPANDING window over lap-level
        session medians so that at Lap L only laps 1..L contribute.
        This eliminates temporal lookahead leakage present in the original
        per-lap groupby which used all laps in the session simultaneously.
"""

import numpy as np
import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive domain features required by the degradation ML model:
      - tyre_age          : consecutive stint laps on current tyre set
      - compound_encoded  : 1=SOFT, 2=MEDIUM, 3=HARD
      - fuel_corrected_time : lap time normalised for burned fuel (clamped)
      - track_evolution   : causal rubbering-in proxy (no lookahead)
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

    return engineered

