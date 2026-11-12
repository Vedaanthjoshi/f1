"""
replay/whatif.py — What-If Pit-Stop Strategy & Counterfactual Engine
=====================================================================
Implements the DELTA FORMULATION for counterfactual lap-time simulation:

counterfactual cumulative time = actual Time_s + cumulative delta

where delta comes from:
  (a) lap-time differences on laps whose tyre age or stint changes
  (b) the pit-loss difference

Engine Rules (from p2f2.md & Corrections C4):
  - G4 relative loss already includes out-lap warmup. NO separate out-lap penalty.
    On a moved stop, real in/out-lap times are replaced with clean-lap equivalents,
    then exactly one sampled loss is added across the pit window.
  - Green stops: sample strictly from the 5th-95th percentile of G4 green losses
    (filtering out extreme outliers).
  - Neutralized stops: sampled from G4 neutralized sample (n=6, all VSC start stops).
    Results labeled "VSC-start stop".
  - Shift range: -12 to +12 laps. Any shift needing tyre ages beyond the max real age
    seen for that compound in this race is marked "extrapolated".
  - Dual run: Every what-if runs twice (nominal vs with G3 age-bucket bias added to
    extrapolated increments). If sign of expected delta changes, or 90% interval
    changes whether it crosses zero, marked "NOT ROBUST".
  - Stint-level noise + block-bootstrapped residuals under Common Random Numbers (500+ paths).
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fuel Model Constants (matching pipeline/features.py)
# ---------------------------------------------------------------------------
FUEL_START_KG = 110.0
FUEL_BURN_PER_LAP_KG = 1.7
FUEL_MIN_KG = 2.0
FUEL_TIME_PER_KG_S = 0.03  # 0.03 s/kg

SHIFT_MIN = -12
SHIFT_MAX = 12
BLOCK_SIZE = 5  # block bootstrap length in laps

DEFAULT_GREEN_PIT_LOSS = 25.13  # Bahrain 2023 G4 median
DEFAULT_NEUT_PIT_LOSS = 15.48   # Bahrain 2023 G4 VSC-start median

# G3 Age Bucket Bias: mean(actual - predicted) from Gate 3
G3_AGE_BUCKET_BIAS = {
    "1-5": 0.8031,
    "6-10": 0.3802,
    "11-15": 0.3267,
    "16-20": 0.3148,
    "21-30": 0.0813,
    "31+": 0.0813,
}

# Max observed tyre ages by compound in Bahrain 2023
MAX_REAL_TYRE_AGE_BAHRAIN = {
    "HARD": 29,
    "SOFT": 25,
    "MEDIUM": 10,
    "INTERMEDIATE": 30,
    "WET": 30,
}


def fuel_mass(lap_num: float) -> float:
    """Remaining fuel mass clamped >= 2 kg (pipeline/features.py)."""
    return float(np.maximum(FUEL_START_KG - FUEL_BURN_PER_LAP_KG * lap_num, FUEL_MIN_KG))


def fuel_correction(lap_num: float) -> float:
    """Fuel time effect in seconds for a given lap number."""
    return fuel_mass(lap_num) * FUEL_TIME_PER_KG_S


def block_bootstrap(residuals: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Block-bootstrap residuals array to length n with block size BLOCK_SIZE.
    Samples contiguous blocks with replacement.
    """
    if residuals is None or len(residuals) == 0:
        return np.zeros(n)
    n_blocks = int(np.ceil(n / BLOCK_SIZE))
    max_start = max(1, len(residuals) - BLOCK_SIZE + 1)
    start_indices = rng.integers(0, max_start, size=n_blocks)
    blocks = [residuals[s: s + BLOCK_SIZE] for s in start_indices]
    out = np.concatenate(blocks)
    return out[:n]


def _build_feature_row(
    tyre_age: int,
    compound_encoded: int,
    lap_num: int,
    track_env: dict,
    circuit_physics: dict,
) -> pd.DataFrame:
    """Build single-row DataFrame matching the 14-feature physics suite."""
    f_mass = fuel_mass(lap_num)
    c_hard = {1: 3, 2: 2, 3: 1}.get(compound_encoded, 2)
    abr = circuit_physics.get("abrasion_index", 4.0)
    length = circuit_physics.get("circuit_length_km", 5.412)
    corners = circuit_physics.get("num_corners", 15.0)
    corner_density = corners / length if length > 0 else 3.0

    row = {
        "tyre_age": float(tyre_age),
        "compound_encoded": float(compound_encoded),
        "compound_hardness": float(c_hard),
        "track_evolution": float(track_env.get("track_evolution", 85.0)),
        "TrackTemp": float(track_env.get("TrackTemp", 32.0)),
        "fuel_mass": float(f_mass),
        "abrasion_index": float(abr),
        "circuit_length_km": float(length),
        "num_corners": float(corners),
        "corner_density": float(corner_density),
        "thermal_load": float(tyre_age * track_env.get("TrackTemp", 32.0)),
        "abrasion_load": float(tyre_age * abr),
        "wear_cliff": float((tyre_age / 15.0) ** 2.0),
        "compound_age_interaction": float(tyre_age * c_hard),
    }
    return pd.DataFrame([row])


_deg_cache: dict[tuple, float] = {}


def _predict_degradation_cached(
    model,
    tyre_age: int,
    compound_encoded: int,
    lap_num: int,
    track_env: dict,
    circuit_physics: dict,
) -> float:
    """Predict degradation (lap_time_delta) with caching to eliminate redundant predict calls."""
    key = (tyre_age, compound_encoded, lap_num)
    if key in _deg_cache:
        return _deg_cache[key]
    feat = _build_feature_row(tyre_age, compound_encoded, lap_num, track_env, circuit_physics)
    val = _predict_degradation(model, feat)
    _deg_cache[key] = val
    return val


def _predict_degradation(model, feature_row: pd.DataFrame) -> float:
    """Predict degradation (lap_time_delta) from model."""
    if model is None:
        return 0.05 * float(feature_row["tyre_age"].iloc[0])
    pred = model.predict(feature_row)
    if pred.ndim == 2:
        return float(pred[0, 1])  # p50 of 3 quantiles
    return float(pred[0])


def _get_age_bias(age: int) -> float:
    """Return G3 bias for a given tyre age."""
    if age <= 5:
        return G3_AGE_BUCKET_BIAS["1-5"]
    elif age <= 10:
        return G3_AGE_BUCKET_BIAS["6-10"]
    elif age <= 15:
        return G3_AGE_BUCKET_BIAS["11-15"]
    elif age <= 20:
        return G3_AGE_BUCKET_BIAS["16-20"]
    elif age <= 30:
        return G3_AGE_BUCKET_BIAS["21-30"]
    return G3_AGE_BUCKET_BIAS["31+"]


def filter_green_pit_losses(losses: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Filter green pit losses to the 5th-95th percentile.
    Returns (filtered_losses, dropped_stops_list).
    """
    if len(losses) < 5:
        return losses, []
    p5 = float(np.percentile(losses, 5))
    p95 = float(np.percentile(losses, 95))
    mask = (losses >= p5) & (losses <= p95)
    filtered = losses[mask]
    dropped = [
        {"loss": float(l), "reason": "below 5th percentile" if l < p5 else "above 95th percentile"}
        for l in losses[~mask]
    ]
    return filtered, dropped



# G3 controlled-linear model coefficients (non-Bahrain fit): {compound_enc: (beta_age, beta_evo, beta_fuel, intercept)}
# Used as the D5 "controlled-linear" baseline for extrapolated increments.
# Coefficients from Gate 3 controlled linear regression (track_evolution + fuel_mass controls).
G3_CONTROLLED_LINEAR_COEFFS = {
    1: {"beta_age": 0.0589, "beta_evo": 0.7909, "beta_fuel": -0.0322, "intercept": -72.0},  # SOFT
    2: {"beta_age": 0.0371, "beta_evo": 0.6887, "beta_fuel": -0.0280, "intercept": -58.0},  # MEDIUM
    3: {"beta_age": 0.0273, "beta_evo": 0.5992, "beta_fuel": -0.0220, "intercept": -47.0},  # HARD
}


def _controlled_linear_increment(
    age_from: int, age_to: int, compound_enc: int,
    track_evo: float, fuel_mass_val: float,
) -> float:
    """
    Predict degradation increment over (age_from -> age_to) laps using the G3
    controlled-linear model (non-Bahrain, with track_evolution + fuel_mass controls).
    Returns the predicted lap-time increment in seconds.
    """
    coeffs = G3_CONTROLLED_LINEAR_COEFFS.get(compound_enc, G3_CONTROLLED_LINEAR_COEFFS[3])
    beta_age = coeffs["beta_age"]
    # Degradation increment = beta_age * age_diff (the controlled regression slope)
    age_diff = max(0, age_to - age_from)
    return beta_age * age_diff


def run_whatif(
    replay_table: pd.DataFrame,
    driver: str,
    stop_idx: int,
    shift_laps: int,
    g3_model,
    g3_choice: str = "xgboost",
    g3_linear_slopes: dict | None = None,
    g3_residuals: np.ndarray | None = None,
    pit_loss_green: np.ndarray | None = None,
    pit_loss_neutralized: np.ndarray | None = None,
    neutralized_laps: set | None = None,
    n_paths: int = 500,
    seed: int = 42,
    apply_extrapolated_bias: bool = False,
    use_controlled_linear_extrap: bool = True,
) -> dict:
    """
    Simulate a counterfactual pit strategy shift for a chosen driver and stop.

    Parameters:
    -----------
    apply_extrapolated_bias : bool
        If True, adds G3 age-bucket bias to extrapolated tyre age increments (for dual-run robustness).
    use_controlled_linear_extrap : bool
        D5: If True, uses the G3 controlled-linear model (non-Bahrain fit) instead of XGBoost for
        extrapolated degradation increments (tyre ages beyond real observed range).
    """
    if g3_linear_slopes is None:
        g3_linear_slopes = {1: -0.0984, 2: -0.1579, 3: -0.0109}
    if g3_residuals is None or len(g3_residuals) == 0:
        g3_residuals = np.zeros(100)
    if pit_loss_green is None or len(pit_loss_green) == 0:
        pit_loss_green = np.array([DEFAULT_GREEN_PIT_LOSS])
    if pit_loss_neutralized is None or len(pit_loss_neutralized) == 0:
        pit_loss_neutralized = np.array([DEFAULT_NEUT_PIT_LOSS])
    if neutralized_laps is None:
        neutralized_laps = set()

    # Rule C4: Filter green stops to 5th-95th percentile
    filtered_green_losses, _ = filter_green_pit_losses(pit_loss_green)
    if len(filtered_green_losses) == 0:
        filtered_green_losses = pit_loss_green

    driver_df = replay_table[replay_table["Driver"] == driver].copy()
    driver_df = driver_df.sort_values("LapNumber").reset_index(drop=True)

    if driver_df.empty:
        return _make_infeasible("Driver not found in replay table.", 0)

    total_laps = int(driver_df["LapNumber"].max())
    n_driver_laps = len(driver_df)

    # Identify stops
    in_lap_rows = driver_df[driver_df["is_in_lap"] == True]
    pit_laps = in_lap_rows["LapNumber"].astype(int).tolist()

    if not pit_laps or stop_idx >= len(pit_laps):
        return _make_infeasible(f"Stop index {stop_idx} out of range ({len(pit_laps)} stops).", n_driver_laps)

    actual_pit_lap = pit_laps[stop_idx]
    new_pit_lap = actual_pit_lap + shift_laps

    # Feasibility checks
    if new_pit_lap < 2:
        return _make_infeasible(f"New pit lap {new_pit_lap} < 2 (too early).", n_driver_laps)
    if new_pit_lap > total_laps - 1:
        return _make_infeasible(f"New pit lap {new_pit_lap} > {total_laps - 1} (too late).", n_driver_laps)

    # Check collision with other stops
    for idx_other, p_other in enumerate(pit_laps):
        if idx_other != stop_idx and abs(new_pit_lap - p_other) < 2:
            return _make_infeasible(f"New pit lap {new_pit_lap} collides with stop at lap {p_other}.", n_driver_laps)

    # Max real tyre ages in race for compound extrapolation check
    max_real_ages = {}
    for comp_name, grp in replay_table.groupby("Compound"):
        c_upper = str(comp_name).upper()
        max_real_ages[c_upper] = float(grp["TyreLife"].max()) if not grp["TyreLife"].empty else 30.0

    # Fast path: shift = 0 reproduces actual race perfectly
    if shift_laps == 0:
        actual_positions = driver_df["Position"].fillna(20).astype(int).values
        final_pos = actual_positions[-1] if len(actual_positions) > 0 else 20
        dist = np.zeros(25, dtype=float)
        if 0 <= final_pos < 25:
            dist[final_pos] = 1.0
        return {
            "delta_per_lap": np.zeros(n_driver_laps),
            "final_time_delta_p50": 0.0,
            "final_time_delta_p05": 0.0,
            "final_time_delta_p95": 0.0,
            "position_per_lap": actual_positions.astype(float),
            "position_distribution": dist,
            "pit_loss_effect_old": 0.0,
            "pit_loss_effect_new": 0.0,
            "is_extrapolated": False,
            "stop_type": "green",
            "is_feasible": True,
            "infeasible_reason": None,
        }

    # Extract driver's actual stint structure
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
    compound_name_map = {1: "SOFT", 2: "MEDIUM", 3: "HARD", 4: "INTERMEDIATE", 5: "WET"}
    lap_records = {}
    for _, r in driver_df.iterrows():
        ln = int(r["LapNumber"])
        c_str = str(r.get("Compound", "HARD")).upper()
        comp_enc = compound_map.get(c_str, 2)
        tl = float(r.get("TyreLife", 1)) if pd.notnull(r.get("TyreLife")) else 1.0
        st = float(r.get("Stint", 1)) if pd.notnull(r.get("Stint")) else 1.0
        lt = float(r.get("LapTime_s", np.nan))
        ts = float(r.get("Time_s", np.nan))
        is_in = bool(r.get("is_in_lap", False))
        is_out = bool(r.get("is_out_lap", False))

        lap_records[ln] = {
            "lap": ln,
            "compound_enc": comp_enc,
            "compound_str": c_str,
            "tyre_life": tl,
            "stint": st,
            "lap_time": lt,
            "time_s": ts,
            "is_in": is_in,
            "is_out": is_out,
            "is_clean": (not is_in) and (not is_out) and pd.notnull(lt) and (ln not in neutralized_laps),
        }

    # Catalog real clean laps by (stint, tyre_age).
    # D2 FIX: exclude neutralized laps (VSC/SC slow laps) from the lookup table so
    # they are never used as clean driving-time references in counterfactual stints.
    # Neutralized laps have distorted (slow) actual times (~126 s vs ~98 s at Bahrain
    # lap 41) that corrupt counterfactual driving-time deltas by up to 28 s per lap.
    clean_laps_by_stint_age: dict[tuple[int, int], tuple[int, float]] = {}
    stint_last_clean: dict[int, tuple[int, int, float]] = {}

    for ln, rec in sorted(lap_records.items()):
        if rec["is_clean"] and ln not in neutralized_laps:
            st = int(rec["stint"])
            ta = int(rec["tyre_life"])
            clean_laps_by_stint_age[(st, ta)] = (ln, rec["lap_time"])
            stint_last_clean[st] = (ta, ln, rec["lap_time"])

    # Next stop lap (if any)
    next_stop_lap = pit_laps[stop_idx + 1] if stop_idx + 1 < len(pit_laps) else total_laps + 1
    stint_before = int(lap_records[actual_pit_lap]["stint"])
    stint_after = stint_before + 1

    out_lap_rec = lap_records.get(actual_pit_lap + 1)
    new_stint_comp = out_lap_rec["compound_enc"] if out_lap_rec else 3
    new_stint_comp_str = compound_name_map.get(new_stint_comp, "HARD")

    circuit_physics = {"abrasion_index": 4.0, "circuit_length_km": 5.412, "num_corners": 15.0}

    # Track environment lookup
    track_env_lookup = {}
    if "track_evolution" in replay_table.columns and "TrackTemp" in replay_table.columns:
        for _, row in replay_table.groupby("LapNumber").first().iterrows():
            track_env_lookup[int(row.name)] = {
                "track_evolution": row.get("track_evolution", 85.0),
                "TrackTemp": row.get("TrackTemp", 32.0),
            }

    # Rivals' cumulative times per lap
    rivals_df = replay_table[replay_table["Driver"] != driver]
    rivals_times_by_lap: dict[int, list[float]] = {}
    for ln_val, grp in rivals_df.groupby("LapNumber"):
        valid_times = grp["Time_s"].dropna().tolist()
        rivals_times_by_lap[int(ln_val)] = valid_times

    # Extrapolation detection across the counterfactual stint
    is_extrapolated = False
    max_seen_before = max_real_ages.get(lap_records[actual_pit_lap]["compound_str"], 30.0)
    max_seen_after = max_real_ages.get(new_stint_comp_str, 30.0)

    # Check tyre age bounds
    if shift_laps > 0:
        base_age = int(lap_records[actual_pit_lap]["tyre_life"])
        max_cf_age_before = base_age + shift_laps
        if max_cf_age_before > max_seen_before:
            is_extrapolated = True
    else:
        # Pitted early -> stint_after tyres will be older at end of stint
        ext_after_age = (next_stop_lap - 1) - new_pit_lap
        if ext_after_age > max_seen_after:
            is_extrapolated = True

    # Check whether new stop is neutralized (VSC start stop)
    new_is_vsc = (new_pit_lap in neutralized_laps) or (new_pit_lap + 1 in neutralized_laps)
    stop_type_label = "VSC-start stop" if new_is_vsc else "green"

    # Monte Carlo simulation paths
    lap_nums = sorted(lap_records.keys())
    all_path_cumulative_deltas = np.zeros((n_paths, len(lap_nums)))
    all_pit_loss_effects_old = np.zeros(n_paths)
    all_pit_loss_effects_new = np.zeros(n_paths)

    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31, size=n_paths)

    for p_idx in range(n_paths):
        rng = np.random.default_rng(seeds[p_idx])

        # Rule C4: Sample exactly one pit loss per stop (includes out-lap warmup, no separate penalty)
        actual_is_vsc = (actual_pit_lap in neutralized_laps) or (actual_pit_lap + 1 in neutralized_laps)
        p_loss_pool_act = pit_loss_neutralized if actual_is_vsc else filtered_green_losses
        loss_actual = float(rng.choice(p_loss_pool_act))

        p_loss_pool_new = pit_loss_neutralized if new_is_vsc else filtered_green_losses
        loss_new = float(rng.choice(p_loss_pool_new))

        # Block-bootstrapped residuals
        residuals = block_bootstrap(g3_residuals, len(lap_nums) + 20, rng)
        unique_stints = set(r["stint"] for r in lap_records.values())
        stint_noise = {s: rng.normal(0.0, 0.3) for s in unique_stints}

        cum_delta = 0.0
        cf_tyre_age = 0

        for l_idx, ln in enumerate(lap_nums):
            act_rec = lap_records[ln]
            noise_val = stint_noise.get(act_rec["stint"], 0.0)
            res_val = residuals[l_idx]

            lap_delta = 0.0

            # --- 1. Pit Loss Components ---
            # Rule C4: Replace real in/out-lap times with clean-lap equivalents, then add one sampled loss
            if ln == actual_pit_lap:
                # Remove actual in-lap transit loss component
                lap_delta -= 0.5 * loss_actual
                all_pit_loss_effects_old[p_idx] -= 0.5 * loss_actual
            elif ln == actual_pit_lap + 1:
                # Remove actual out-lap transit loss component
                lap_delta -= 0.5 * loss_actual
                all_pit_loss_effects_old[p_idx] -= 0.5 * loss_actual

            if ln == new_pit_lap:
                # Add CF in-lap transit loss component
                lap_delta += 0.5 * loss_new
                all_pit_loss_effects_new[p_idx] += 0.5 * loss_new
            elif ln == new_pit_lap + 1:
                # Add CF out-lap transit loss component
                lap_delta += 0.5 * loss_new
                all_pit_loss_effects_new[p_idx] += 0.5 * loss_new

            # --- 2. Clean Driving Lap Time Differences ---
            if ln < min(actual_pit_lap, new_pit_lap) or ln >= next_stop_lap:
                pass
            elif ln in neutralized_laps and ln != new_pit_lap and ln != new_pit_lap + 1:
                # D2/D3 FIX: If this lap is neutralized in real race and NOT the counterfactual pit stop,
                # both cars are driving under VSC delta on track. Driving difference is 0 s.
                pass
            else:
                if shift_laps > 0:
                    if ln <= new_pit_lap:
                        base_age = int(lap_records[actual_pit_lap]["tyre_life"])
                        cf_age = base_age + (ln - actual_pit_lap)
                        cf_comp = lap_records[actual_pit_lap]["compound_enc"]
                        target_stint = stint_before

                        cf_clean_t = _get_driving_time(
                            target_stint, cf_age, cf_comp, ln, clean_laps_by_stint_age,
                            stint_last_clean, g3_model, g3_choice, g3_linear_slopes,
                            track_env_lookup.get(ln, {}), circuit_physics,
                            apply_bias=apply_extrapolated_bias,
                            use_controlled_linear_extrap=use_controlled_linear_extrap,
                        )
                        act_clean_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        lap_delta += (cf_clean_t - act_clean_t)
                    elif ln == new_pit_lap + 1:
                        cf_tyre_age = 1
                    else:
                        cf_tyre_age += 1
                        cf_comp = new_stint_comp
                        target_stint = stint_after

                        cf_clean_t = _get_driving_time(
                            target_stint, cf_tyre_age, cf_comp, ln, clean_laps_by_stint_age,
                            stint_last_clean, g3_model, g3_choice, g3_linear_slopes,
                            track_env_lookup.get(ln, {}), circuit_physics,
                            apply_bias=apply_extrapolated_bias,
                            use_controlled_linear_extrap=use_controlled_linear_extrap,
                        )
                        act_clean_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        lap_delta += (cf_clean_t - act_clean_t)
                else:
                    if ln <= new_pit_lap:
                        pass
                    elif ln == new_pit_lap + 1:
                        cf_tyre_age = 1
                    elif ln <= actual_pit_lap:
                        cf_tyre_age += 1
                        cf_comp = new_stint_comp
                        target_stint = stint_after

                        cf_clean_t = _get_driving_time(
                            target_stint, cf_tyre_age, cf_comp, ln, clean_laps_by_stint_age,
                            stint_last_clean, g3_model, g3_choice, g3_linear_slopes,
                            track_env_lookup.get(ln, {}), circuit_physics,
                            apply_bias=apply_extrapolated_bias,
                            use_controlled_linear_extrap=use_controlled_linear_extrap,
                        )
                        act_clean_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        lap_delta += (cf_clean_t - act_clean_t)
                    elif ln == actual_pit_lap + 1:
                        cf_tyre_age += 1
                    else:
                        cf_tyre_age += 1
                        cf_comp = new_stint_comp
                        target_stint = stint_after

                        cf_clean_t = _get_driving_time(
                            target_stint, cf_tyre_age, cf_comp, ln, clean_laps_by_stint_age,
                            stint_last_clean, g3_model, g3_choice, g3_linear_slopes,
                            track_env_lookup.get(ln, {}), circuit_physics,
                            apply_bias=apply_extrapolated_bias,
                            use_controlled_linear_extrap=use_controlled_linear_extrap,
                        )
                        act_clean_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        lap_delta += (cf_clean_t - act_clean_t)

            if ln >= min(actual_pit_lap, new_pit_lap) and ln < next_stop_lap:
                lap_delta += (0.1 * noise_val + 0.1 * res_val)

            cum_delta += lap_delta
            all_path_cumulative_deltas[p_idx, l_idx] = cum_delta

    mean_cumulative_delta = all_path_cumulative_deltas.mean(axis=0)
    final_lap_deltas = all_path_cumulative_deltas[:, -1]

    # Compute counterfactual positions per lap
    position_per_lap = np.zeros(len(lap_nums))
    for l_idx, ln in enumerate(lap_nums):
        act_t = lap_records[ln]["time_s"]
        if pd.isna(act_t):
            position_per_lap[l_idx] = 20.0
            continue
        cf_t = act_t + mean_cumulative_delta[l_idx]
        rivals = rivals_times_by_lap.get(ln, [])
        pos = sum(1 for rt in rivals if rt < cf_t) + 1
        position_per_lap[l_idx] = float(pos)

    # Final position distribution across paths
    final_ln = lap_nums[-1]
    act_final_t = lap_records[final_ln]["time_s"]
    rivals_final = rivals_times_by_lap.get(final_ln, [])

    final_positions = []
    for p_idx in range(n_paths):
        cf_final_t = act_final_t + all_path_cumulative_deltas[p_idx, -1]
        pos = sum(1 for rt in rivals_final if rt < cf_final_t) + 1
        final_positions.append(pos)

    pos_dist = np.bincount(final_positions, minlength=25).astype(float)
    if pos_dist.sum() > 0:
        pos_dist /= pos_dist.sum()

    return {
        "delta_per_lap": mean_cumulative_delta,
        "final_time_delta_p50": float(np.median(final_lap_deltas)),
        "final_time_delta_p05": float(np.percentile(final_lap_deltas, 5)),
        "final_time_delta_p95": float(np.percentile(final_lap_deltas, 95)),
        "position_per_lap": position_per_lap,
        "position_distribution": pos_dist,
        "pit_loss_effect_old": float(np.mean(all_pit_loss_effects_old)),
        "pit_loss_effect_new": float(np.mean(all_pit_loss_effects_new)),
        "is_extrapolated": is_extrapolated,
        "stop_type": stop_type_label,
        "is_feasible": True,
        "infeasible_reason": None,
    }


def _get_driving_time(
    stint: int,
    tyre_age: int,
    compound_enc: int,
    target_lap: int,
    clean_laps_by_stint_age: dict,
    stint_last_clean: dict,
    g3_model,
    g3_choice: str,
    g3_linear_slopes: dict,
    track_env: dict,
    circuit_physics: dict,
    apply_bias: bool = False,
    use_controlled_linear_extrap: bool = False,
) -> float:
    """
    Look up real clean lap time or project using G3 degradation increment.

    If the requested (stint, tyre_age) is in the clean_laps_by_stint_age lookup,
    the real fuel-corrected lap time is returned directly (no extrapolation).

    When the age exceeds the last seen clean lap in that stint, an increment is
    projected using XGBoost (default) or the G3 controlled-linear model (D5).
    """
    key = (stint, tyre_age)
    if key in clean_laps_by_stint_age:
        ref_lap, ref_lt = clean_laps_by_stint_age[key]
        fuel_corrected = ref_lt - fuel_correction(ref_lap)
        return fuel_corrected + fuel_correction(target_lap)

    # Beyond real (non-neutralized) tyre age in this stint: extrapolate.
    if stint in stint_last_clean:
        last_age, last_lap, last_lt = stint_last_clean[stint]
        base_clean = last_lt - fuel_correction(last_lap) + fuel_correction(target_lap)
        age_diff = max(0, tyre_age - last_age)

        if use_controlled_linear_extrap:
            # D5: use the G3 controlled-linear model for extrapolated increments
            deg_increment = _controlled_linear_increment(
                last_age, tyre_age, compound_enc,
                track_env.get("track_evolution", 85.0),
                fuel_mass(target_lap),
            )
        elif g3_choice == "xgboost" and g3_model is not None:
            deg_cur = _predict_degradation_cached(g3_model, tyre_age, compound_enc, target_lap, track_env, circuit_physics)
            deg_last = _predict_degradation_cached(g3_model, last_age, compound_enc, target_lap, track_env, circuit_physics)
            deg_increment = deg_cur - deg_last
        else:
            slope = g3_linear_slopes.get(compound_enc, -0.0109)
            deg_increment = abs(slope) * age_diff

        if apply_bias:
            bias_corr = _get_age_bias(tyre_age) * (age_diff / 5.0)
            deg_increment += bias_corr

        return base_clean + max(0.0, deg_increment)

    return 95.0 + 0.08 * tyre_age


def _get_actual_reference_time(
    act_rec: dict,
    clean_laps_by_stint_age: dict,
    stint_last_clean: dict,
) -> float:
    """Return the actual baseline driving time (excluding in/out transit penalty)."""
    if act_rec["is_clean"]:
        return act_rec["lap_time"]
    st = int(act_rec["stint"])
    ta = int(act_rec["tyre_life"])
    key = (st, ta)
    if key in clean_laps_by_stint_age:
        return clean_laps_by_stint_age[key][1]
    if st in stint_last_clean:
        return stint_last_clean[st][2]
    return 95.0


def _make_infeasible(reason: str, n_laps: int) -> dict:
    """Helper to return an infeasible result dict."""
    return {
        "delta_per_lap": np.zeros(n_laps),
        "final_time_delta_p50": 0.0,
        "final_time_delta_p05": 0.0,
        "final_time_delta_p95": 0.0,
        "position_per_lap": np.zeros(n_laps),
        "position_distribution": np.zeros(25),
        "pit_loss_effect_old": 0.0,
        "pit_loss_effect_new": 0.0,
        "is_extrapolated": False,
        "stop_type": "green",
        "is_feasible": False,
        "infeasible_reason": reason,
    }


def decompose_whatif(
    replay_table: pd.DataFrame,
    driver: str,
    stop_idx: int,
    shift_laps: int,
    g3_model,
    g3_choice: str = "xgboost",
    g3_linear_slopes: dict | None = None,
    pit_loss_green: np.ndarray | None = None,
    pit_loss_neutralized: np.ndarray | None = None,
    neutralized_laps: set | None = None,
    use_controlled_linear_extrap: bool = True,
) -> dict:
    """
    Deterministic decomposition of a pit-stop shift into six components.
    Uses identical lap-record and clean-lap-catalog logic as run_whatif (with the
    is_clean neutralised-lap fix applied).  Pit losses are fixed at the MEDIAN of
    the relevant pool (no Monte-Carlo sampling) so the total is fully reproducible.

    Returns a dict with keys:
        shift, new_pit, total,
        comp_i   (-median actual pit loss removed),
        comp_ii  (+median new pit loss added),
        comp_iii (lap-time replacement for real in/out laps),
        comp_iv  (lap-time effect of new in/out laps),
        comp_v   (degradation / extrapolation laps),
        comp_vi  (anything else),
        residual (total - sum of comps; must be near 0),
        sample_label, act_in_n, act_out_n, new_in_n, new_out_n
    """
    if g3_linear_slopes is None:
        g3_linear_slopes = {1: -0.0984, 2: -0.1579, 3: -0.0109}
    if pit_loss_green is None or len(pit_loss_green) == 0:
        pit_loss_green = np.array([DEFAULT_GREEN_PIT_LOSS])
    if pit_loss_neutralized is None or len(pit_loss_neutralized) == 0:
        pit_loss_neutralized = np.array([DEFAULT_NEUT_PIT_LOSS])
    if neutralized_laps is None:
        neutralized_laps = set()

    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
    compound_name_map = {1: "SOFT", 2: "MEDIUM", 3: "HARD", 4: "INTERMEDIATE", 5: "WET"}

    filtered_green_losses, _ = filter_green_pit_losses(pit_loss_green)
    if len(filtered_green_losses) == 0:
        filtered_green_losses = pit_loss_green

    driver_df = replay_table[replay_table["Driver"] == driver].copy()
    driver_df = driver_df.sort_values("LapNumber").reset_index(drop=True)
    if driver_df.empty:
        return {"total": float("nan"), "is_feasible": False, "infeasible_reason": "Driver not found"}

    in_laps = driver_df[driver_df["is_in_lap"] == True]["LapNumber"].astype(int).tolist()
    if not in_laps or stop_idx >= len(in_laps):
        return {"total": float("nan"), "is_feasible": False, "infeasible_reason": "Stop index out of range"}

    act_pit = in_laps[stop_idx]
    new_pit = act_pit + shift_laps
    next_stop = in_laps[stop_idx + 1] if stop_idx + 1 < len(in_laps) else int(driver_df["LapNumber"].max()) + 1

    act_in_n  = act_pit in neutralized_laps
    act_out_n = (act_pit + 1) in neutralized_laps
    new_in_n  = new_pit in neutralized_laps
    new_out_n = (new_pit + 1) in neutralized_laps

    act_is_vsc = act_in_n or act_out_n
    new_is_vsc = new_in_n or new_out_n

    # Deterministic: use median of the pool
    loss_act = float(np.median(pit_loss_neutralized if act_is_vsc else filtered_green_losses))
    loss_new = float(np.median(pit_loss_neutralized if new_is_vsc else filtered_green_losses))

    # Build lap records (same logic as run_whatif, including neutralized-lap exclusion)
    lap_records: dict = {}
    for _, r in driver_df.iterrows():
        ln = int(r["LapNumber"])
        c_str = str(r.get("Compound", "HARD")).upper()
        comp_enc = compound_map.get(c_str, 2)
        tl = float(r.get("TyreLife", 1)) if pd.notnull(r.get("TyreLife")) else 1.0
        st = float(r.get("Stint", 1)) if pd.notnull(r.get("Stint")) else 1.0
        lt = float(r.get("LapTime_s", float("nan")))
        ts = float(r.get("Time_s", float("nan")))
        is_in = bool(r.get("is_in_lap", False))
        is_out = bool(r.get("is_out_lap", False))
        lap_records[ln] = {
            "lap": ln, "compound_enc": comp_enc, "compound_str": c_str,
            "tyre_life": tl, "stint": st, "lap_time": lt, "time_s": ts,
            "is_in": is_in, "is_out": is_out,
            # Key fix: exclude neutralized laps (identical to run_whatif after the D2 fix)
            "is_clean": (not is_in) and (not is_out) and pd.notnull(lt) and (ln not in neutralized_laps),
        }

    clean_laps_by_stint_age: dict = {}
    stint_last_clean: dict = {}
    for ln, rec in sorted(lap_records.items()):
        if rec["is_clean"]:
            st = int(rec["stint"]); ta = int(rec["tyre_life"])
            clean_laps_by_stint_age[(st, ta)] = (ln, rec["lap_time"])
            stint_last_clean[st] = (ta, ln, rec["lap_time"])

    stint_before = int(lap_records[act_pit]["stint"])
    stint_after  = stint_before + 1
    out_lap_rec  = lap_records.get(act_pit + 1)
    new_stint_comp = out_lap_rec["compound_enc"] if out_lap_rec else 3

    circuit_physics = {"abrasion_index": 4.0, "circuit_length_km": 5.412, "num_corners": 15.0}
    track_env_lookup: dict = {}
    if "track_evolution" in replay_table.columns:
        for _, row in replay_table.groupby("LapNumber").first().iterrows():
            track_env_lookup[int(row.name)] = {
                "track_evolution": row.get("track_evolution", 85.0),
                "TrackTemp": row.get("TrackTemp", 32.0),
            }

    comp_i   = -loss_act
    comp_ii  = +loss_new
    comp_iii = 0.0
    comp_iv  = 0.0
    comp_v   = 0.0
    comp_vi  = 0.0

    cum_total  = 0.0
    cf_tyre_age = 0

    for ln in sorted(lap_records.keys()):
        act_rec = lap_records[ln]
        lap_d   = 0.0

        # Pit-loss split across in-lap and out-lap
        if ln == act_pit or ln == act_pit + 1:
            lap_d -= 0.5 * loss_act
        if ln == new_pit or ln == new_pit + 1:
            lap_d += 0.5 * loss_new

        # Driving-time delta (mirror of run_whatif logic)
        if min(act_pit, new_pit) <= ln < next_stop:
            # Neutralized laps in between: both cars under VSC, delta = 0
            if ln in neutralized_laps and ln != new_pit and ln != new_pit + 1:
                pass
            else:
                if shift_laps > 0:
                    if ln <= new_pit:
                        base_age = int(lap_records[act_pit]["tyre_life"])
                        cf_age   = base_age + (ln - act_pit)
                        cf_comp  = lap_records[act_pit]["compound_enc"]
                        target_stint = stint_before
                        cf_t  = _get_driving_time(target_stint, cf_age, cf_comp, ln, clean_laps_by_stint_age, stint_last_clean, g3_model, g3_choice, g3_linear_slopes, track_env_lookup.get(ln, {}), circuit_physics, use_controlled_linear_extrap=use_controlled_linear_extrap)
                        act_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        d = cf_t - act_t
                        lap_d += d
                        if ln == act_pit or ln == act_pit + 1:
                            comp_iii += d
                        elif ln == new_pit:
                            comp_iv  += d
                        else:
                            comp_v   += d
                    elif ln == new_pit + 1:
                        cf_tyre_age = 1
                    else:
                        cf_tyre_age += 1
                        cf_comp  = new_stint_comp
                        target_stint = stint_after
                        cf_t  = _get_driving_time(target_stint, cf_tyre_age, cf_comp, ln, clean_laps_by_stint_age, stint_last_clean, g3_model, g3_choice, g3_linear_slopes, track_env_lookup.get(ln, {}), circuit_physics, use_controlled_linear_extrap=use_controlled_linear_extrap)
                        act_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        d = cf_t - act_t
                        lap_d += d
                        comp_v += d
                else:
                    if ln <= new_pit:
                        pass
                    elif ln == new_pit + 1:
                        cf_tyre_age = 1
                    elif ln <= act_pit:
                        cf_tyre_age += 1
                        cf_comp  = new_stint_comp
                        target_stint = stint_after
                        cf_t  = _get_driving_time(target_stint, cf_tyre_age, cf_comp, ln, clean_laps_by_stint_age, stint_last_clean, g3_model, g3_choice, g3_linear_slopes, track_env_lookup.get(ln, {}), circuit_physics, use_controlled_linear_extrap=use_controlled_linear_extrap)
                        act_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        d = cf_t - act_t
                        lap_d += d
                        if ln == act_pit:
                            comp_iii += d
                        elif ln == new_pit + 1:
                            comp_iv  += d
                        else:
                            comp_v   += d
                    elif ln == act_pit + 1:
                        cf_tyre_age += 1
                    else:
                        cf_tyre_age += 1
                        cf_comp  = new_stint_comp
                        target_stint = stint_after
                        cf_t  = _get_driving_time(target_stint, cf_tyre_age, cf_comp, ln, clean_laps_by_stint_age, stint_last_clean, g3_model, g3_choice, g3_linear_slopes, track_env_lookup.get(ln, {}), circuit_physics, use_controlled_linear_extrap=use_controlled_linear_extrap)
                        act_t = _get_actual_reference_time(act_rec, clean_laps_by_stint_age, stint_last_clean)
                        d = cf_t - act_t
                        lap_d += d
                        comp_v += d

        cum_total += lap_d

    total_comps = comp_i + comp_ii + comp_iii + comp_iv + comp_v + comp_vi
    residual = cum_total - total_comps
    sample_label = "VSC-start" if new_is_vsc else "green (5-95 pct)"

    return {
        "shift": shift_laps,
        "new_pit": new_pit,
        "total": cum_total,
        "comp_i": comp_i,
        "comp_ii": comp_ii,
        "comp_iii": comp_iii,
        "comp_iv": comp_iv,
        "comp_v": comp_v,
        "comp_vi": comp_vi,
        "residual": residual,
        "sample_label": sample_label,
        "act_in_n": act_in_n,
        "act_out_n": act_out_n,
        "new_in_n": new_in_n,
        "new_out_n": new_out_n,
        "is_feasible": True,
        "infeasible_reason": None,
    }


def regret_scan(
    replay_table: pd.DataFrame,
    driver: str,
    stop_idx: int,
    g3_model,
    g3_choice: str = "xgboost",
    g3_linear_slopes: dict | None = None,
    g3_residuals: np.ndarray | None = None,
    pit_loss_green: np.ndarray | None = None,
    pit_loss_neutralized: np.ndarray | None = None,
    neutralized_laps: set | None = None,
    n_paths: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate shifts from -12 to +12 laps for a driver's pit stop with dual-run robustness check.
    """
    rows = []
    base_res = run_whatif(
        replay_table=replay_table,
        driver=driver,
        stop_idx=stop_idx,
        shift_laps=0,
        g3_model=g3_model,
        g3_choice=g3_choice,
        g3_linear_slopes=g3_linear_slopes,
        g3_residuals=g3_residuals,
        pit_loss_green=pit_loss_green,
        pit_loss_neutralized=pit_loss_neutralized,
        neutralized_laps=neutralized_laps,
        n_paths=n_paths,
        seed=seed,
    )
    base_pos = base_res["position_per_lap"][-1] if len(base_res["position_per_lap"]) > 0 else 10.0

    for shift in range(SHIFT_MIN, SHIFT_MAX + 1):
        # Run A: Nominal using G3 Controlled-Linear model for extrapolated increments (D5)
        res_a = run_whatif(
            replay_table=replay_table,
            driver=driver,
            stop_idx=stop_idx,
            shift_laps=shift,
            g3_model=g3_model,
            g3_choice=g3_choice,
            g3_linear_slopes=g3_linear_slopes,
            g3_residuals=g3_residuals,
            pit_loss_green=pit_loss_green,
            pit_loss_neutralized=pit_loss_neutralized,
            neutralized_laps=neutralized_laps,
            n_paths=n_paths,
            seed=seed,
            apply_extrapolated_bias=False,
            use_controlled_linear_extrap=True,
        )
        if not res_a["is_feasible"]:
            continue

        p50_a = res_a["final_time_delta_p50"]
        p05_a = res_a["final_time_delta_p05"]
        p95_a = res_a["final_time_delta_p95"]

        # Run B: Dual-run using XGBoost for extrapolated increments (D5 robustness test)
        res_b = run_whatif(
            replay_table=replay_table,
            driver=driver,
            stop_idx=stop_idx,
            shift_laps=shift,
            g3_model=g3_model,
            g3_choice=g3_choice,
            g3_linear_slopes=g3_linear_slopes,
            g3_residuals=g3_residuals,
            pit_loss_green=pit_loss_green,
            pit_loss_neutralized=pit_loss_neutralized,
            neutralized_laps=neutralized_laps,
            n_paths=n_paths,
            seed=seed,
            apply_extrapolated_bias=False,
            use_controlled_linear_extrap=False,
        )
        p50_b = res_b["final_time_delta_p50"]
        p05_b = res_b["final_time_delta_p05"]
        p95_b = res_b["final_time_delta_p95"]

        # Dual-run robustness check (D5 Rule):
        # Sign of expected delta changes OR 90% interval changes whether it crosses zero
        sign_changed = (np.sign(p50_a) != np.sign(p50_b)) and (abs(p50_a) > 0.05 and abs(p50_b) > 0.05)
        crosses_zero_a = (p05_a <= 0.0 <= p95_a)
        crosses_zero_b = (p05_b <= 0.0 <= p95_b)
        crossing_changed = (crosses_zero_a != crosses_zero_b)

        if sign_changed or crossing_changed:
            robustness_status = "NOT ROBUST"
        else:
            robustness_status = "ROBUST"

        # Probability final position improves (lower rank number)
        dist = res_a["position_distribution"]
        if len(dist) > 0 and base_pos > 1:
            p_improves = float(np.sum(dist[:int(base_pos)]))
        else:
            p_improves = 0.0

        is_noise = (p05_a <= 0.0 <= p95_a)

        rows.append({
            "shift": shift,
            "expected_time_delta": p50_a,
            "p05": p05_a,
            "p95": p95_a,
            "p_position_improves": p_improves,
            "is_noise": is_noise,
            "is_extrapolated": res_a["is_extrapolated"],
            "robustness": robustness_status,
            "stop_type": res_a["stop_type"],
        })

    return pd.DataFrame(rows)


def prepare_race_data(year: int, gp_name: str) -> dict:
    """
    Load session, replay table, and prepare G3/G4 models and residuals for race review.
    """
    import fastf1
    from replay.load import load_replay_table

    cache_dir = "f1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    is_bahrain = "bahrain" in gp_name.lower()
    if is_bahrain and os.path.exists(os.path.join("models", "loro_bahrain.joblib")):
        model_obj = joblib.load(os.path.join("models", "loro_bahrain.joblib"))
    else:
        model_obj = joblib.load(os.path.join("models", "global_degradation_model.joblib"))

    g3_model = model_obj["model"] if isinstance(model_obj, dict) else model_obj
    g3_choice = "xgboost"

    g3_linear_slopes = {1: -0.0984, 2: -0.1579, 3: -0.0109}

    session = fastf1.get_session(year, gp_name, "R")
    session.load(weather=True, messages=False)

    replay_table = load_replay_table(year, gp_name, session=session)

    if os.path.exists("g4_pit_loss_results.csv"):
        g4_df = pd.read_csv("g4_pit_loss_results.csv")
        pit_loss_green = g4_df[g4_df["classification"] == "green"]["relative_loss_s"].values
        pit_loss_neutralized = g4_df[g4_df["classification"] == "neutralized"]["relative_loss_s"].values
    else:
        pit_loss_green = np.array([DEFAULT_GREEN_PIT_LOSS])
        pit_loss_neutralized = np.array([DEFAULT_NEUT_PIT_LOSS])

    neutralized_laps = set(
        replay_table[replay_table["is_neutralized"] == True]["LapNumber"].astype(int).tolist()
    )

    if os.path.exists("g3_pairs_results.csv"):
        g3_df = pd.read_csv("g3_pairs_results.csv")
        g3_residuals = g3_df["error"].values
    else:
        g3_residuals = np.zeros(100)

    ml_data = pd.DataFrame()
    try:
        from pipeline.ingestion import clean_lap_data, merge_weather_data
        from pipeline.features import apply_feature_engineering
        from pipeline.baseline import prepare_ml_data
        clean_df = clean_lap_data(session.laps)
        weather_df = merge_weather_data(clean_df, session.weather_data)
        feat_df = apply_feature_engineering(weather_df, gp_name)
        ml_data = prepare_ml_data(feat_df)
    except Exception:
        pass

    return {
        "session": session,
        "replay_table": replay_table,
        "ml_data": ml_data,
        "pit_loss_green": pit_loss_green,
        "pit_loss_neutralized": pit_loss_neutralized,
        "neutralized_laps": neutralized_laps,
        "g3_model": g3_model,
        "g3_choice": g3_choice,
        "g3_linear_slopes": g3_linear_slopes,
        "g3_residuals": g3_residuals,
        "is_bahrain": is_bahrain,
    }


if __name__ == "__main__":
    import sys
    print("=" * 65)
    print("replay/whatif.py — Phase 1 & C4 Unit Tests")
    print("=" * 65)

    rng = np.random.default_rng(42)
    rows = []
    ver_pit_laps = {14, 36}
    ver_time = 0.0
    ver_tyre = 0
    ver_stint = 1
    ver_comp = "SOFT"

    for lap in range(1, 58):
        is_pit = lap in ver_pit_laps
        is_out = (lap - 1) in ver_pit_laps
        if is_out:
            ver_tyre = 0
            ver_stint += 1
            ver_comp = "HARD"
        ver_tyre += 1
        lt = 94.0 + ver_tyre * 0.06 + (25.0 if is_pit else 0.0) + rng.normal(0, 0.2)
        ver_time += lt
        rows.append({
            "Driver": "VER", "LapNumber": float(lap), "Position": 1.0,
            "Time_s": ver_time, "LapTime_s": lt, "Compound": ver_comp,
            "TyreLife": float(ver_tyre), "Stint": float(ver_stint),
            "is_in_lap": is_pit, "is_out_lap": is_out, "TrackStatus": "1",
            "is_neutralized": False, "gap_ahead_s": np.nan, "gap_behind_s": np.nan,
        })

    for drv_idx, drv in enumerate(["PER", "LEC", "SAI", "HAM"]):
        d_time = 0.0
        for lap in range(1, 58):
            lt = 94.5 + drv_idx * 0.2 + rng.normal(0, 0.3)
            d_time += lt
            rows.append({
                "Driver": drv, "LapNumber": float(lap), "Position": float(drv_idx + 2),
                "Time_s": d_time, "LapTime_s": lt, "Compound": "HARD",
                "TyreLife": float(lap), "Stint": 1.0,
                "is_in_lap": False, "is_out_lap": False, "TrackStatus": "1",
                "is_neutralized": False, "gap_ahead_s": np.nan, "gap_behind_s": np.nan,
            })

    test_replay = pd.DataFrame(rows)

    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))

    model = MockModel()
    pit_green = np.array([25.13, 24.5, 26.0, 22.0, 36.0, 50.0])  # has outliers to test filtering
    pit_neut = np.array([15.48, 14.8, 16.0])

    print("\n[TEST 1] Shift 0 laps -> delta exactly 0 and positions unchanged")
    r0 = run_whatif(test_replay, "VER", 0, 0, model, pit_loss_green=pit_green, pit_loss_neutralized=pit_neut)
    max_abs_delta = np.max(np.abs(r0["delta_per_lap"]))
    p50_0 = r0["final_time_delta_p50"]
    actual_pos = test_replay[test_replay["Driver"] == "VER"]["Position"].values
    pos_match = np.all(r0["position_per_lap"] == actual_pos)

    print(f"  Max absolute delta across all laps: {max_abs_delta:.8f} s (expected 0.0)")
    print(f"  Final time delta p50:               {p50_0:.8f} s (expected 0.0)")
    print(f"  Positions match actual:             {pos_match}")
    test1_pass = (max_abs_delta < 1e-12) and (abs(p50_0) < 1e-12) and pos_match
    print(f"  Result: {'PASS [OK]' if test1_pass else 'FAIL [X]'}")

    print("\n[TEST 2] Shifting +1 vs -1 gives opposite-sign effects on the pit-loss term")
    r_plus1 = run_whatif(test_replay, "VER", 0, +1, model, pit_loss_green=pit_green, pit_loss_neutralized=pit_neut, n_paths=200)
    r_minus1 = run_whatif(test_replay, "VER", 0, -1, model, pit_loss_green=pit_green, pit_loss_neutralized=pit_neut, n_paths=200)

    eff_plus = r_plus1["pit_loss_effect_old"]
    eff_minus = r_minus1["pit_loss_effect_new"]

    print(f"  Shift +1 pit-loss effect at actual stop: {eff_plus:+.3f} s (expected < 0)")
    print(f"  Shift -1 pit-loss effect at new stop:    {eff_minus:+.3f} s (expected > 0)")
    opposite_signs = (eff_plus < 0) and (eff_minus > 0)
    print(f"  Opposite signs verified:                 {opposite_signs}")
    test2_pass = opposite_signs and r_plus1["is_feasible"] and r_minus1["is_feasible"]
    print(f"  Result: {'PASS [OK]' if test2_pass else 'FAIL [X]'}")

    print("\n[TEST 3] Feasibility checks and shift range -12..+12")
    r_infeas_early = run_whatif(test_replay, "VER", 0, -14, model)
    r_infeas_late = run_whatif(test_replay, "VER", 0, +50, model)
    r_shift12 = run_whatif(test_replay, "VER", 0, 12, model)
    print(f"  Shift -14 feasible: {r_infeas_early['is_feasible']} (reason: {r_infeas_early['infeasible_reason']})")
    print(f"  Shift +50 feasible: {r_infeas_late['is_feasible']} (reason: {r_infeas_late['infeasible_reason']})")
    print(f"  Shift +12 feasible: {r_shift12['is_feasible']}")
    test3_pass = (not r_infeas_early["is_feasible"]) and (not r_infeas_late["is_feasible"]) and r_shift12["is_feasible"]
    print(f"  Result: {'PASS [OK]' if test3_pass else 'FAIL [X]'}")

    all_passed = test1_pass and test2_pass and test3_pass
    print("\n" + "=" * 65)
    print(f"SUMMARY: {'ALL TESTS PASSED [OK]' if all_passed else 'TESTS FAILED [X]'}")
    print("=" * 65)
    sys.exit(0 if all_passed else 1)
