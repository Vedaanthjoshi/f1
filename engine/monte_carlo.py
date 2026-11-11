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
from engine.rival_reaction import calculate_cover_rate, simulate_undercut_differential
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
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_14

    # Dynamically clamp simulation horizon if near the scheduled race finish
    if total_laps is not None and total_laps >= current_lap:
        remaining = total_laps - current_lap + 1
        horizon = max(2, min(horizon, remaining))

    sc_rate = get_circuit_sc_rate(race_name)
    rng = np.random.default_rng(seed)
    cover_rate = calculate_cover_rate(rival_gap_s)

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
    so_std = np.maximum((so_p95 - so_p5) / (2 * 1.645), 1e-4)
    so_delta_samples = rng.normal(loc=so_p50, scale=so_std, size=(n_rollouts, horizon))
    so_lap_times = current_state['baseline_pace'] + so_delta_samples + fuel_corrections

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
                'p50_lap_times':      (current_state['baseline_pace'] + so_p50 + fuel_corrections).tolist(),
            })
            continue

        # ── Pit Strategy ──────────────────────────────────────────────────────
        # Tyre age resets to 1 upon taking fresh rubber
        pit_ages = np.array([
            current_state['tyre_age'] + n if n < pit_offset else 1 + (n - pit_offset)
            for n in range(horizon)
        ], dtype=float)

        pit_df = _build_feature_batch(
            pit_ages, strat_compound,
            current_state, evos, temps, fuel_masses, feature_cols
        )
        pit_p5, pit_p50, pit_p95 = _predict_quantiles(model, pit_df)
        pit_std = np.maximum((pit_p95 - pit_p5) / (2 * 1.645), 1e-4)

        pit_delta_samples = rng.normal(loc=pit_p50, scale=pit_std, size=(n_rollouts, horizon))
        pit_driving_lap_times = current_state['baseline_pace'] + pit_delta_samples + fuel_corrections

        # Safety Car stochastic sampling on pit lap
        sc_on_pit_lap = rng.binomial(1, sc_rate, size=n_rollouts).astype(bool)
        effective_pit_loss = np.where(
            sc_on_pit_lap,
            pit_loss_delta * SC_PIT_LOSS_MULTIPLIER,
            pit_loss_delta,
        )

        OUT_LAP_PENALTY_S = 1.5  # cold tyre out-lap warmup penalty

        # Build defender fresh-tyre predictions for the cover/extend branches.
        # When the defender pits to cover, they go onto the same compound as the attacker.
        # We model defender's post-stop pace as fresh tyre age starting from 1,
        # using the same compound as the attacking strategy so both cars are credited
        # with equivalent fresh rubber after their respective stops.
        def_cover_ages = np.array([1 + n for n in range(horizon)], dtype=float)
        def_cover_df = _build_feature_batch(
            def_cover_ages, strat_compound,
            current_state, evos, temps, fuel_masses, feature_cols
        )
        _, def_cover_p50, _ = _predict_quantiles(model, def_cover_df)
        # Sample defender's fresh-tyre paths with the same quantile spread as attacker
        def_cover_std = np.maximum((pit_p95 - pit_p5) / (2 * 1.645), 1e-4)
        def_cover_delta_samples = rng.normal(loc=def_cover_p50, scale=def_cover_std, size=(n_rollouts, horizon))
        defender_fresh_lap_times = current_state['baseline_pace'] + def_cover_delta_samples + fuel_corrections

        # Evaluate net advantage: two-car undercut differential vs competitor
        if rival_gap_s is not None:
            final_net_gains = simulate_undercut_differential(
                attacker_lap_times_no_pitloss=pit_driving_lap_times,
                defender_stay_out_times=so_lap_times,
                pit_offset=pit_offset,
                horizon=horizon,
                effective_pit_loss=effective_pit_loss,
                rival_gap_s=rival_gap_s,
                cover_rate=cover_rate,
                n_rollouts=n_rollouts,
                rng=rng,
                out_lap_penalty=OUT_LAP_PENALTY_S,
                defender_fresh_lap_times=defender_fresh_lap_times,
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
                defender_stay_out_times=so_lap_times,
                pit_offset=pit_offset,
                horizon=horizon,
                effective_pit_loss=sc_pit_loss_const,
                rival_gap_s=rival_gap_s,
                cover_rate=cover_rate,
                n_rollouts=n_rollouts,
                rng=rng,
                out_lap_penalty=OUT_LAP_PENALTY_S,
                defender_fresh_lap_times=defender_fresh_lap_times,
            )
        else:
            full_pit_times_sc = pit_driving_lap_times.copy()
            full_pit_times_sc[:, pit_offset] += sc_pit_loss_const
            full_pit_times_sc[:, min(pit_offset + 1, horizon - 1)] += OUT_LAP_PENALTY_S
            sc_gains = so_lap_times.sum(axis=1) - full_pit_times_sc.sum(axis=1)
        sc_expected = float(sc_gains.mean())

        # Crossover lap: first lap where fresh tyre degradation is lower than stay-out degradation
        crossover_lap = None
        for n in range(pit_offset, horizon):
            if (pit_p50[n] - so_p50[n]) < -0.15:  # significant fresh tyre pace advantage
                crossover_lap = current_lap + n
                break

        # Undercut decision threshold: p_net_gain >= 50% and expected gain > 0
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
            'p50_lap_times':      (current_state['baseline_pace'] + pit_p50 + fuel_corrections).tolist(),
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
