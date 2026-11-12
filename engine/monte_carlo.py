"""
engine/monte_carlo.py — Probabilistic Monte Carlo Strategy Optimizer
====================================================================
Vectorized 15-lap forward simulation across 6 tactical strategies.
Samples degradation trajectories from multi-quantile predictions (p05, p50, p95),
incorporates circuit Safety Car likelihood, out-lap cold tyre warmup penalty,
and two-agent game-theoretic rival reaction dynamics to identify optimal pit windows.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from engine.safety_car import get_circuit_sc_rate, SC_PIT_LOSS_MULTIPLIER
from engine.strategies import STRATEGIES
from engine.rival_reaction import (
    calculate_cover_rate,
    get_overtaking_difficulty,
    simulate_undercut_differential,
)
from pipeline.features import CIRCUIT_PHYSICS, PIRELLI_EVENT_ALLOCATIONS, FEATURE_COLS_14


def _build_feature_batch(
    tyre_ages: np.ndarray,
    compound_encoded: int,
    current_state: dict[str, Any],
    evos: np.ndarray,
    temps: np.ndarray,
    fuel_masses: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a horizon-length DataFrame for a single vectorized model.predict() call."""
    n = len(tyre_ages)
    track_name = current_state.get('Track', '_default')
    circuit = CIRCUIT_PHYSICS.get(track_name, CIRCUIT_PHYSICS['_default'])
    alloc = PIRELLI_EVENT_ALLOCATIONS.get(track_name, PIRELLI_EVENT_ALLOCATIONS['_default'])

    # Determine physical Pirelli C-hardness for this compound
    if compound_encoded == 1:
        c_hard = alloc.get('SOFT', 4)
    elif compound_encoded == 3:
        c_hard = alloc.get('HARD', 2)
    else:
        c_hard = alloc.get('MEDIUM', 3)

    corner_density = circuit['num_corners'] / circuit['circuit_length_km']

    data: dict[str, Any] = {
        'tyre_age':                 tyre_ages,
        'compound_encoded':         np.full(n, compound_encoded, dtype=float),
        'compound_hardness':        np.full(n, c_hard, dtype=float),
        'track_evolution':          evos,
        'TrackTemp':                temps,
        'fuel_mass':                fuel_masses,
        'abrasion_index':           np.full(n, circuit['abrasion_index'],    dtype=float),
        'circuit_length_km':        np.full(n, circuit['circuit_length_km'], dtype=float),
        'num_corners':              np.full(n, circuit['num_corners'],       dtype=float),
        'corner_density':           np.full(n, corner_density,               dtype=float),
        'thermal_load':             tyre_ages * temps,
        'abrasion_load':            tyre_ages * circuit['abrasion_index'],
        'wear_cliff':               (tyre_ages / 15.0) ** 2.0,
        'compound_age_interaction': tyre_ages * c_hard,
        'Track':                    [track_name] * n,
        'Driver':                   [current_state.get('Driver', 'VER')] * n,
    }
    df = pd.DataFrame({col: data[col] for col in feature_cols if col in data})
    if 'Track' in df.columns:
        df['Track'] = df['Track'].astype('category')
    if 'Driver' in df.columns:
        df['Driver'] = df['Driver'].astype('category')
    return df


def _predict_quantiles(
    model: Any,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call model.predict() and return (p05, p50, p95) arrays of shape (n_laps,)."""
    preds = model.predict(df)
    if preds.ndim == 2 and preds.shape[1] == 3:
        return preds[:, 0], preds[:, 1], preds[:, 2]
    # Legacy point-estimate fallback
    return preds.copy(), preds.copy(), preds.copy()


def run_mc_simulation(
    model: Any,
    current_lap: int,
    current_state: dict[str, Any],
    track_env: pd.DataFrame,
    pit_loss_delta: float,
    race_name: str = "",
    n_rollouts: int = 500,
    horizon: int = 15,
    feature_cols: list[str] | None = None,
    seed: int = 42,
    rival_gap_s: float | None = None,
    total_laps: int | None = None,
    race_alpha: float = 1.0,
    race_bias: float = 0.0,
    overtaking_difficulty: str | None = None,
    defender_state: dict[str, Any] | None = None,
    compound_pace_offsets: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Run Monte Carlo strategy evaluation across all candidate tactical strategies.

    Parameters
    ----------
    model : trained multi-quantile regression model (p05, p50, p95)
    current_lap : int, current lap number
    current_state : dict containing tyre_age, compound_encoded, baseline_pace, Track, Driver
    track_env : DataFrame with LapNumber, track_evolution, TrackTemp
    pit_loss_delta : float, dynamic pit stop time loss in seconds (~20-25s)
    race_name : str, Grand Prix name
    n_rollouts : int, number of Monte Carlo sampling paths (default 500)
    horizon : int, forward prediction window in laps (default 15)
    rival_gap_s : float or None, gap in seconds to defending rival (if attempting undercut)
    total_laps : int or None, total scheduled race laps to clamp forward horizon
    race_alpha : float, causal Empirical Bayes degradation multiplier (default 1.0)
    race_bias : float, causal Empirical Bayes pace residual bias (default 0.0)
    overtaking_difficulty : str or None, circuit overtaking difficulty ('Low', 'Medium', 'High')
    """
    from ml.trainer import apply_race_eb
    if feature_cols is None:
        feature_cols = FEATURE_COLS_14

    # Dynamically clamp simulation horizon if near the scheduled race finish
    if total_laps is not None and total_laps >= current_lap:
        remaining = total_laps - current_lap + 1
        horizon = max(2, min(horizon, remaining))

    sc_rate = get_circuit_sc_rate(race_name)
    rng = np.random.default_rng(seed)
    def_age_val = defender_state.get('tyre_age') if defender_state else None
    def_comp_val = defender_state.get('compound_encoded') if defender_state else None
    cover_rate = calculate_cover_rate(
        rival_gap_s,
        def_tyre_age=def_age_val,
        def_compound_encoded=def_comp_val,
    )
    diff = overtaking_difficulty or get_overtaking_difficulty(race_name)

    # Pre-compute shared track environment arrays across horizon laps
    evo_map = track_env.set_index('LapNumber')['track_evolution'].to_dict() if 'LapNumber' in track_env.columns else {}
    temp_map = track_env.set_index('LapNumber')['TrackTemp'].to_dict() if 'LapNumber' in track_env.columns else {}
    last_evo = list(evo_map.values())[-1] if evo_map else 85.0
    last_temp = list(temp_map.values())[-1] if temp_map else 30.0

    horizon_laps = [current_lap + n for n in range(horizon)]
    evos = np.array([evo_map.get(l, last_evo) for l in horizon_laps])
    temps = np.array([temp_map.get(l, last_temp) for l in horizon_laps])
    fuel_masses = np.array([np.maximum(110.0 - 1.7 * l, 2.0) for l in horizon_laps])
    fuel_corrections = fuel_masses * 0.03

    # --- Stay-Out baseline (Driver A on current worn tyre progression) ---
    so_ages = np.array([current_state['tyre_age'] + n for n in range(horizon)])
    so_df = _build_feature_batch(
        so_ages, current_state['compound_encoded'],
        current_state, evos, temps, fuel_masses, feature_cols
    )
    so_p5, so_p50, so_p95 = _predict_quantiles(model, so_df)
    so_p50_eb = apply_race_eb(so_p50, race_alpha, race_bias)
    so_std = np.maximum((so_p95 - so_p5) / (2 * 1.645), 1e-4)
    so_delta_samples = rng.normal(loc=so_p50_eb, scale=so_std, size=(n_rollouts, horizon))
    so_lap_times = current_state['baseline_pace'] + so_delta_samples + fuel_corrections

    # --- Defender stay-out baseline (use real defender pace/tyre state when available) ---
    if defender_state is not None:
        def_ages = np.array([defender_state['tyre_age'] + n for n in range(horizon)])
        def_df = _build_feature_batch(
            def_ages, int(defender_state['compound_encoded']),
            defender_state, evos, temps, fuel_masses, feature_cols
        )
        d_p5, d_p50, d_p95 = _predict_quantiles(model, def_df)
        d_p50_eb = apply_race_eb(d_p50, race_alpha, race_bias)
        d_std = np.maximum((d_p95 - d_p5) / (2 * 1.645), 1e-4)
        def_delta_samples = rng.normal(loc=d_p50_eb, scale=d_std, size=(n_rollouts, horizon))
        def_bp_shrunk = current_state['baseline_pace'] + float(
            np.clip(defender_state['baseline_pace'] - current_state['baseline_pace'], -0.50, 0.50)
        )
        def_stay_out_times = def_bp_shrunk + def_delta_samples + fuel_corrections
        def_cover_base_pace = def_bp_shrunk
        def_ref_comp = int(defender_state['compound_encoded'])
    else:
        def_stay_out_times = so_lap_times
        def_cover_base_pace = float(current_state['baseline_pace'])
        def_ref_comp = int(current_state.get('compound_encoded', 2))

    att_ref_comp = int(current_state.get('compound_encoded', 2))
    track_name_cur = current_state.get('Track', '_default')
    alloc_cur = PIRELLI_EVENT_ALLOCATIONS.get(track_name_cur, PIRELLI_EVENT_ALLOCATIONS['_default'])
    _comp_key_map = {1: 'SOFT', 2: 'MEDIUM', 3: 'HARD'}

    def _compound_offset(from_comp: int, to_comp: int) -> float:
        if int(from_comp) == int(to_comp):
            return 0.0
        if compound_pace_offsets and int(from_comp) in compound_pace_offsets and int(to_comp) in compound_pace_offsets:
            return float(np.clip(compound_pace_offsets[int(to_comp)] - compound_pace_offsets[int(from_comp)], -0.65, 0.65))
        return 0.0

    results: list[dict[str, Any]] = []

    for strat in STRATEGIES:
        pit_offset = strat['pit_offset']
        strat_compound = strat['compound_encoded']

        # ── Stay Out baseline ─────────────────────────────────────────────────
        if pit_offset is None:
            results.append({
                'name':               'Stay Out',
                'pit_offset':         None,
                'compound_encoded':   current_state['compound_encoded'],
                'p_net_gain':         0.50,
                'expected_gain_s':    0.0,
                'gain_ci_low':        0.0,
                'gain_ci_high':       0.0,
                'crossover_lap':      None,
                'window_open':        False,
                'sc_expected_gain_s': 0.0,
                'cover_rate':         0.0,
                'p50_lap_times':      (current_state['baseline_pace'] + so_p50_eb + fuel_corrections).tolist(),
            })
            continue

        if pit_offset + 2 >= horizon:
            continue

        # Filter unviable early Soft stints when >26 race laps remain and not on Soft already
        if (
            total_laps is not None
            and strat_compound == 1
            and (total_laps - (current_lap + pit_offset)) > 26
            and int(current_state.get('compound_encoded', 2)) != 1
        ):
            continue

        # ── Pit Strategy ──────────────────────────────────────────────────────
        # In-lap (`n <= pit_offset`) stays on worn tyre; fresh rubber (`age = 1`) starts at out-lap (`pit_offset + 1`)
        pit_ages = np.array([
            current_state['tyre_age'] + n if n <= pit_offset else float(n - pit_offset)
            for n in range(horizon)
        ], dtype=float)

        pit_df = _build_feature_batch(
            pit_ages, strat_compound,
            current_state, evos, temps, fuel_masses, feature_cols
        )
        pit_p5, pit_p50, pit_p95 = _predict_quantiles(model, pit_df)
        pit_p50_eb = apply_race_eb(pit_p50, race_alpha, race_bias)
        pit_std = np.maximum((pit_p95 - pit_p5) / (2 * 1.645), 1e-4)

        att_comp_shift = _compound_offset(att_ref_comp, int(strat_compound))
        pit_delta_samples = rng.normal(loc=pit_p50_eb, scale=pit_std, size=(n_rollouts, horizon))
        pit_driving_lap_times = current_state['baseline_pace'] + att_comp_shift + pit_delta_samples + fuel_corrections
        pit_driving_lap_times[:, :pit_offset + 1] = so_lap_times[:, :pit_offset + 1]

        # Safety Car stochastic sampling on pit lap
        sc_on_pit_lap = rng.binomial(1, sc_rate, size=n_rollouts).astype(bool)
        effective_pit_loss = np.where(
            sc_on_pit_lap,
            pit_loss_delta * SC_PIT_LOSS_MULTIPLIER,
            pit_loss_delta,
        )

        # Compound- & temperature-aware cold tyre out-lap warmup penalty
        strat_c_hard = float(alloc_cur.get(_comp_key_map.get(int(strat_compound), 'MEDIUM'), 3))
        out_temp = float(temps[min(pit_offset + 1, horizon - 1)])
        OUT_LAP_PENALTY_S = float(np.clip(
            1.5 * (strat_c_hard / 3.0) * np.clip(1.0 - 0.012 * (out_temp - 30.0), 0.80, 1.25),
            0.8, 2.8,
        ))

        # Age-indexed fresh-tyre predictions (`tyre_age = 1..horizon`) for defender cover reset
        def_cover_ages = np.arange(1, horizon + 1, dtype=float)
        def_cover_state = defender_state if defender_state is not None else current_state
        def_cover_df = _build_feature_batch(
            def_cover_ages, strat_compound,
            def_cover_state, evos, temps, fuel_masses, feature_cols
        )
        def_p5, def_cover_p50, def_p95 = _predict_quantiles(model, def_cover_df)
        def_cover_p50_eb = apply_race_eb(def_cover_p50, race_alpha, race_bias)
        def_cover_std = np.maximum((def_p95 - def_p5) / (2 * 1.645), 1e-4)
        def_cover_delta_samples = rng.normal(loc=def_cover_p50_eb, scale=def_cover_std, size=(n_rollouts, horizon))
        def_comp_shift = _compound_offset(def_ref_comp, int(strat_compound))
        defender_fresh_lap_times = def_cover_base_pace + def_comp_shift + def_cover_delta_samples

        # Evaluate net advantage: two-car undercut differential vs competitor
        if rival_gap_s is not None:
            final_net_gains = simulate_undercut_differential(
                attacker_lap_times_no_pitloss=pit_driving_lap_times,
                defender_stay_out_times=def_stay_out_times,
                pit_offset=pit_offset,
                horizon=horizon,
                effective_pit_loss=effective_pit_loss,
                rival_gap_s=rival_gap_s,
                cover_rate=cover_rate,
                n_rollouts=n_rollouts,
                rng=rng,
                out_lap_penalty=OUT_LAP_PENALTY_S,
                defender_fresh_lap_times=defender_fresh_lap_times,
                fuel_corrections=fuel_corrections,
                overtaking_difficulty=diff,
            )
        else:
            # Single-car stint pace optimization against staying out forever
            full_pit_times = pit_driving_lap_times.copy()
            full_pit_times[:, pit_offset] += effective_pit_loss
            out_lap_idx = min(pit_offset + 1, horizon - 1)
            full_pit_times[:, out_lap_idx] += OUT_LAP_PENALTY_S
            final_net_gains = so_lap_times.sum(axis=1) - full_pit_times.sum(axis=1)

        p_net_gain = float((final_net_gains > 0).mean())
        expected_gain = float(final_net_gains.mean())
        gain_ci_low = float(np.percentile(final_net_gains, 5))
        gain_ci_high = float(np.percentile(final_net_gains, 95))

        # Expected gain assuming certain SC on pit lap
        sc_pit_loss_const = np.full(n_rollouts, pit_loss_delta * SC_PIT_LOSS_MULTIPLIER)
        if rival_gap_s is not None:
            sc_gains = simulate_undercut_differential(
                attacker_lap_times_no_pitloss=pit_driving_lap_times,
                defender_stay_out_times=def_stay_out_times,
                pit_offset=pit_offset,
                horizon=horizon,
                effective_pit_loss=sc_pit_loss_const,
                rival_gap_s=rival_gap_s,
                cover_rate=cover_rate,
                n_rollouts=n_rollouts,
                rng=rng,
                out_lap_penalty=OUT_LAP_PENALTY_S,
                defender_fresh_lap_times=defender_fresh_lap_times,
                fuel_corrections=fuel_corrections,
                overtaking_difficulty=diff,
            )
        else:
            full_pit_times_sc = pit_driving_lap_times.copy()
            full_pit_times_sc[:, pit_offset] += sc_pit_loss_const
            full_pit_times_sc[:, min(pit_offset + 1, horizon - 1)] += OUT_LAP_PENALTY_S
            sc_gains = so_lap_times.sum(axis=1) - full_pit_times_sc.sum(axis=1)
        sc_expected = float(sc_gains.mean())

        # Crossover lap: first lap where fresh tyre degradation is lower than stay-out degradation
        crossover_lap = None
        for n in range(pit_offset + 1, horizon):
            if (pit_p50_eb[n] - so_p50_eb[n]) < -0.15:
                crossover_lap = current_lap + n
                break

        window_open = (p_net_gain >= 0.50) and (expected_gain > 0.0)

        results.append({
            'name':               strat['name'],
            'pit_offset':         pit_offset,
            'compound_encoded':   strat_compound,
            'p_net_gain':         p_net_gain,
            'expected_gain_s':    expected_gain,
            'gain_ci_low':        gain_ci_low,
            'gain_ci_high':       gain_ci_high,
            'crossover_lap':      crossover_lap,
            'window_open':        window_open,
            'sc_expected_gain_s': sc_expected,
            'cover_rate':         cover_rate,
            'p50_lap_times':      (current_state['baseline_pace'] + pit_p50_eb + fuel_corrections).tolist(),
        })

    # Sort: Stay Out at index 0, remaining ranked by expected gain descending
    stay_out = [r for r in results if r['name'] == 'Stay Out']
    pit_strats = sorted(
        [r for r in results if r['name'] != 'Stay Out'],
        key=lambda r: r['expected_gain_s'],
        reverse=True,
    )
    return stay_out + pit_strats


def simulate_undercut_window(*args, **kwargs):
    """Facade alias for run_mc_simulation."""
    return run_mc_simulation(*args, **kwargs)
