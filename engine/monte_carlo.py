"""
engine/monte_carlo.py — Probabilistic Monte Carlo Strategy Optimizer
====================================================================
Vectorized 15-lap forward simulation across 6 tactical strategies.
Samples degradation trajectories from quantile predictions, incorporates
Safety Car likelihood and rival reaction dynamics, and identifies
optimal pit windows based on net advantage and confidence intervals.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from engine.safety_car import SC_RATES, SC_RATE_DEFAULT, SC_PIT_LOSS_MULTIPLIER, get_circuit_sc_rate
from engine.strategies import STRATEGIES
from engine.rival_reaction import calculate_cover_rate, apply_rival_reaction


def _build_feature_batch(
    tyre_ages: np.ndarray,
    compound_encoded: int,
    current_state: dict[str, Any],
    evos: np.ndarray,
    temps: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a horizon-length DataFrame for a single vectorized model.predict() call.

    Note: baseline_pace is intentionally excluded from features (Fix 3 — target
    leakage removal).  The new model predicts fuel_corrected_time directly.
    """
    n = len(tyre_ages)
    data: dict[str, Any] = {
        'tyre_age':         tyre_ages,
        'compound_encoded': np.full(n, compound_encoded, dtype=float),
        'track_evolution':  evos,
        'TrackTemp':        temps,
        'Track':            [current_state.get('Track', 'Unknown')] * n,
        'Driver':           [current_state.get('Driver', 'UNK')] * n,
    }
    df = pd.DataFrame({col: data[col] for col in feature_cols if col in data})
    if 'Track' in df.columns:
        df['Track']  = df['Track'].astype('category')
        df['Driver'] = df['Driver'].astype('category')
    return df


def _predict_quantiles(
    model: Any,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Call model.predict() and return (p05, p50, p95) arrays of shape (n_laps,).
    Handles both multi-quantile models (n, 3) and legacy point-estimate models (n,).
    """
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
) -> list[dict[str, Any]]:
    """
    Run Monte Carlo strategy evaluation across all candidate strategies.

    Returns:
    --------
    List of strategy result dictionaries sorted by expected net gain descending
    (with Stay Out kept at index 0 as baseline).
    """
    if feature_cols is None:
        # Fix 3: baseline_pace removed — model now predicts absolute
        # fuel_corrected_time, so no leakage through baseline subtraction.
        feature_cols = [
            'tyre_age', 'compound_encoded',
            'track_evolution', 'TrackTemp', 'Track', 'Driver',
        ]

    sc_rate = get_circuit_sc_rate(race_name)
    rng = np.random.default_rng(seed)
    cover_rate = calculate_cover_rate(rival_gap_s)

    # Pre-compute shared track environment arrays across horizon laps
    evo_map = track_env.set_index('LapNumber')['track_evolution'].to_dict()
    temp_map = track_env.set_index('LapNumber')['TrackTemp'].to_dict()
    last_evo = list(evo_map.values())[-1] if evo_map else 0.0
    last_temp = list(temp_map.values())[-1] if temp_map else 30.0

    horizon_laps = [current_lap + n for n in range(horizon)]
    evos = np.array([evo_map.get(l, last_evo) for l in horizon_laps])
    temps = np.array([temp_map.get(l, last_temp) for l in horizon_laps])
    # Note: fuel_corrections are NOT added here — the new model (Fix 3) predicts
    # fuel_corrected_time directly, so fuel adjustment is implicit in its output.

    # --- Stay-Out baseline (one model call -> sample n_rollouts trajectories) ---
    so_ages = np.array([current_state['tyre_age'] + n for n in range(horizon)])
    so_df = _build_feature_batch(
        so_ages, current_state['compound_encoded'],
        current_state, evos, temps, feature_cols
    )
    # Fix 3: model now predicts fuel_corrected_time directly (absolute seconds).
    # Previously: baseline_pace + delta_sample + fuel_correction.
    # Now:        p50_pred is already the absolute corrected lap time — just add
    #             stochastic noise sampled from the quantile uncertainty band.
    so_p5, so_p50, so_p95 = _predict_quantiles(model, so_df)
    so_std = np.maximum((so_p95 - so_p5) / (2 * 1.645), 1e-5)
    so_delta_samples = rng.normal(loc=so_p50, scale=so_std, size=(n_rollouts, horizon))
    so_lap_times = so_delta_samples   # already absolute fuel_corrected_time
    so_total = so_lap_times.sum(axis=1)

    results: list[dict[str, Any]] = []

    for strat in STRATEGIES:
        pit_offset = strat['pit_offset']
        strat_compound = strat['compound_encoded']

        # Stay Out baseline
        if pit_offset is None:
            results.append({
                'name':               'Stay Out',
                'pit_offset':         None,
                'compound_encoded':   current_state['compound_encoded'],
                'p_net_gain':         0.5,
                'expected_gain_s':    0.0,
                'gain_ci_low':        0.0,
                'gain_ci_high':       0.0,
                'crossover_lap':      None,
                'window_open':        False,
                'sc_expected_gain_s': 0.0,
                'cover_rate':         0.0,
                'p50_lap_times':      so_p50.tolist(),   # absolute fuel_corrected_time
            })
            continue

        # Pit Strategy: build tyre-age profile around the pit lap
        pit_ages = np.array([
            current_state['tyre_age'] + n if n < pit_offset else 1 + (n - pit_offset)
            for n in range(horizon)
        ], dtype=float)

        pit_df = _build_feature_batch(
            pit_ages, strat_compound,
            current_state, evos, temps, feature_cols
        )
        pit_p5, pit_p50, pit_p95 = _predict_quantiles(model, pit_df)
        pit_std = np.maximum((pit_p95 - pit_p5) / (2 * 1.645), 1e-5)

        pit_delta_samples = rng.normal(loc=pit_p50, scale=pit_std, size=(n_rollouts, horizon))
        # Fix 3: predictions are absolute fuel_corrected_time — no baseline addition.
        pit_lap_times = pit_delta_samples

        # Fix 5 — Out-lap tyre warmup penalty
        # The lap immediately after the pit stop (out-lap) incurs a ~1.5 s pace
        # deficit from cold tyre warmup before the rubber reaches operating
        # temperature.  Previously this cost was not modelled, making pit
        # strategies appear artificially attractive.
        OUT_LAP_PENALTY_S = 1.5   # representative F1 out-lap delta (seconds)
        out_lap_col = pit_offset + 1 if pit_offset + 1 < horizon else pit_offset
        pit_lap_times[:, out_lap_col] += OUT_LAP_PENALTY_S

        # Safety Car stochastic events
        sc_on_pit_lap = rng.binomial(1, sc_rate, size=n_rollouts).astype(bool)
        effective_pit_loss = np.where(
            sc_on_pit_lap,
            pit_loss_delta * SC_PIT_LOSS_MULTIPLIER,
            pit_loss_delta,
        )
        pit_lap_times[:, pit_offset] += effective_pit_loss

        # Per-lap net advantage before reaction
        base_net_gains_per_lap = so_lap_times - pit_lap_times

        # Apply rival reaction clamping
        final_net_gains = apply_rival_reaction(
            base_net_gains_per_lap, pit_offset, horizon, cover_rate, n_rollouts, rng
        )

        p_net_gain = float((final_net_gains > 0).mean())
        expected_gain = float(final_net_gains.mean())
        gain_ci_low = float(np.percentile(final_net_gains, 5))
        gain_ci_high = float(np.percentile(final_net_gains, 95))

        # Expected gain assuming certain SC this lap
        pit_lap_times_sc = pit_lap_times.copy()
        pit_lap_times_sc[:, pit_offset] -= effective_pit_loss
        pit_lap_times_sc[:, pit_offset] += pit_loss_delta * SC_PIT_LOSS_MULTIPLIER
        sc_net_gains = so_total - pit_lap_times_sc.sum(axis=1)
        sc_expected = float(sc_net_gains.mean())

        # Crossover lap: first lap where fresh-tyre p50 is faster than stay-out p50
        lap_delta_p50 = pit_p50 - so_p50
        crossover_rel = None
        for n in range(pit_offset, horizon):
            if lap_delta_p50[n] < 0:
                crossover_rel = n
                break
        crossover_lap = (current_lap + crossover_rel) if crossover_rel is not None else None

        # Adaptive decision rule: window open if >= 55% confidence and positive expected gain
        window_open = (p_net_gain >= 0.55) and (expected_gain > 0)

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
            'p50_lap_times':      pit_p50.tolist(),   # absolute fuel_corrected_time
        })

    # Sort: Stay Out always at index 0, remaining ranked by expected gain descending
    stay_out = [r for r in results if r['name'] == 'Stay Out']
    pit_strats = sorted(
        [r for r in results if r['name'] != 'Stay Out'],
        key=lambda r: r['expected_gain_s'],
        reverse=True,
    )
    return stay_out + pit_strats


def simulate_undercut_window(
    model: Any,
    current_lap: int,
    current_state: dict[str, Any],
    track_env: pd.DataFrame,
    pit_loss_delta: float,
    next_compound_encoded: int = 3,
    feature_cols: list[str] | None = None,
    race_name: str = "",
    n_rollouts: int = 500,
) -> dict[str, Any]:
    """Phase 1-compatible wrapper returning legacy and Phase 2 keys."""
    mc_results = run_mc_simulation(
        model=model,
        current_lap=current_lap,
        current_state=current_state,
        track_env=track_env,
        pit_loss_delta=pit_loss_delta,
        race_name=race_name,
        n_rollouts=n_rollouts,
        feature_cols=feature_cols,
    )

    best_match = next(
        (r for r in mc_results if r['pit_offset'] == 0 and r['compound_encoded'] == next_compound_encoded),
        mc_results[1] if len(mc_results) > 1 else mc_results[0],
    )
    stay_out = mc_results[0]

    so_times = stay_out['p50_lap_times']
    pit_times = best_match['p50_lap_times']
    if pit_times:
        pit_times_with_loss = pit_times.copy()
        pit_times_with_loss[0] += pit_loss_delta
        lap_advantage = [so_times[i] - pit_times[i] for i in range(len(so_times))]
        net_undercut_advantage = list(pd.Series(lap_advantage).cumsum())
        stay_out_curve = list(pd.Series(so_times).cumsum())
        pit_now_curve = list(pd.Series(pit_times_with_loss).cumsum())
    else:
        net_undercut_advantage = []
        stay_out_curve = []
        pit_now_curve = []

    return {
        'stay_out_curve':         stay_out_curve,
        'pit_now_curve':          pit_now_curve,
        'net_undercut_advantage': net_undercut_advantage,
        'crossover_lap':          best_match['crossover_lap'],
        'window_open':            best_match['window_open'],
        'stay_out_lap_times':     so_times,
        'pit_now_lap_times':      pit_times,
        'mc_results':             mc_results,
        'best_strategy':          best_match,
        'p_net_gain':             best_match['p_net_gain'],
        'expected_gain_s':        best_match['expected_gain_s'],
        'gain_ci_low':            best_match['gain_ci_low'],
        'gain_ci_high':           best_match['gain_ci_high'],
        'sc_expected_gain_s':     best_match['sc_expected_gain_s'],
        'cover_rate':             best_match.get('cover_rate', 0.0),
    }

