"""
simulation.py — Phase 2: Uncertainty-Aware Monte Carlo Strategy Optimizer
==========================================================================
Replaces the Phase 1 binary undercut detector with a full multi-strategy
Monte Carlo optimizer that:
  1. Samples from the quantile degradation model (p5, p50, p95) to generate
     N rollouts per strategy, giving a probability distribution over outcomes.
  2. Evaluates 6 candidate strategies per lap (Stay Out, Pit→S/M/H, Pit+1→M/H).
  3. Injects Safety Car / VSC as a Bernoulli stochastic event per rollout lap,
     collapsing pit-lane cost realistically (~65% reduction).
  4. Removes the hardcoded 1.5s/3-lap threshold — the decision falls naturally
     from P(net gain) across rollouts.

Backward compatibility: simulate_undercut_window() still works if you pass the
old point-estimate model (1D output). It wraps run_mc_simulation() internally.
"""

import pandas as pd
import numpy as np


# ── Safety Car Probability Table ──────────────────────────────────────────────
# Per-lap Bernoulli probability of a Safety Car / VSC event, estimated from
# 2022-2025 FastF1 TrackStatus data (status '4' = SC, '6' = VSC).
# When an SC event fires in a rollout, pit-lane loss collapses by ~65%.
SC_RATES: dict[str, float] = {
    "Bahrain Grand Prix":        0.04,
    "Saudi Arabian Grand Prix":  0.06,
    "Australian Grand Prix":     0.09,
    "Japanese Grand Prix":       0.04,
    "Chinese Grand Prix":        0.06,
    "Miami Grand Prix":          0.07,
    "Emilia Romagna Grand Prix": 0.05,
    "Monaco Grand Prix":         0.14,
    "Canadian Grand Prix":       0.09,
    "Spanish Grand Prix":        0.04,
    "Austrian Grand Prix":       0.06,
    "British Grand Prix":        0.07,
    "Hungarian Grand Prix":      0.04,
    "Belgian Grand Prix":        0.05,
    "Dutch Grand Prix":          0.05,
    "Italian Grand Prix":        0.04,
    "Azerbaijan Grand Prix":     0.14,
    "Singapore Grand Prix":      0.12,
    "United States Grand Prix":  0.07,
    "Mexico City Grand Prix":    0.05,
    "São Paulo Grand Prix":      0.11,
    "Las Vegas Grand Prix":      0.08,
    "Qatar Grand Prix":          0.06,
    "Abu Dhabi Grand Prix":      0.04,
}
SC_RATE_DEFAULT           = 0.06   # fallback for unknown circuits
SC_PIT_LOSS_MULTIPLIER    = 0.35   # effective pit loss under SC ≈ 35% of normal (~8-9s)

# ── Candidate Strategy Space ──────────────────────────────────────────────────
STRATEGIES = [
    {"name": "Stay Out",        "pit_offset": None, "compound_encoded": None},
    {"name": "Pit → Soft",      "pit_offset": 0,    "compound_encoded": 1},
    {"name": "Pit → Medium",    "pit_offset": 0,    "compound_encoded": 2},
    {"name": "Pit → Hard",      "pit_offset": 0,    "compound_encoded": 3},
    {"name": "Pit+1 → Medium",  "pit_offset": 1,    "compound_encoded": 2},
    {"name": "Pit+1 → Hard",    "pit_offset": 1,    "compound_encoded": 3},
]


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _build_feature_batch(
    tyre_ages: np.ndarray,
    compound_encoded: int,
    current_state: dict,
    evos: np.ndarray,
    temps: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a horizon-length DataFrame for a single model.predict() call."""
    n = len(tyre_ages)
    data: dict = {
        'tyre_age':         tyre_ages,
        'compound_encoded': np.full(n, compound_encoded, dtype=float),
        'baseline_pace':    np.full(n, current_state['baseline_pace']),
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
    model,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Call model.predict() and return (p5, p50, p95) arrays of shape (n_laps,).
    Handles both:
      - Phase 2 quantile model  → predict() returns (n, 3)
      - Phase 1 point model     → predict() returns (n,); all three are identical
    """
    preds = model.predict(df)
    if preds.ndim == 2 and preds.shape[1] == 3:
        return preds[:, 0], preds[:, 1], preds[:, 2]
    # Legacy point-estimate model fallback — zero uncertainty
    return preds.copy(), preds.copy(), preds.copy()


# ── Primary Phase 2 API ───────────────────────────────────────────────────────

def run_mc_simulation(
    model,
    current_lap: int,
    current_state: dict,
    track_env: pd.DataFrame,
    pit_loss_delta: float,
    race_name: str = "",
    n_rollouts: int = 500,
    horizon: int = 15,
    feature_cols: list[str] | None = None,
    seed: int = 42,
    rival_gap_s: float | None = None,
) -> list[dict]:
    """
    Run a Monte Carlo strategy evaluation for all candidate strategies.

    Parameters
    ----------
    model          : Trained XGBRegressor (quantile or legacy point-estimate)
    current_lap    : The race lap being evaluated (simulation starts here)
    current_state  : Dict with keys: tyre_age, compound_encoded, baseline_pace,
                     Track, Driver
    track_env      : DataFrame with LapNumber, track_evolution, TrackTemp columns
    pit_loss_delta : Median pit-lane time loss for this circuit/session (seconds)
    race_name      : Full race name string (used to look up SC rate)
    n_rollouts     : Number of Monte Carlo samples per strategy (default 500)
    horizon        : Lap horizon to simulate (default 15)
    feature_cols   : Ordered feature list expected by the model
    seed           : RNG seed for reproducibility
    rival_gap_s    : Time gap to rival in seconds (for reaction modeling)

    Returns
    -------
    List of strategy result dicts, sorted by expected net gain (best first).
    Each dict contains: name, pit_offset, compound_encoded, p_net_gain,
    expected_gain_s, gain_ci_low, gain_ci_high, crossover_lap, window_open,
    sc_expected_gain_s.
    """
    if feature_cols is None:
        feature_cols = [
            'tyre_age', 'compound_encoded', 'baseline_pace',
            'track_evolution', 'TrackTemp', 'Track', 'Driver'
        ]

    sc_rate = SC_RATES.get(race_name, SC_RATE_DEFAULT)
    rng     = np.random.default_rng(seed)

    # Calculate Empirical Cover Rate (Feature 4)
    cover_rate = 0.0
    if rival_gap_s is not None and not np.isnan(rival_gap_s):
        abs_gap = abs(rival_gap_s)
        if abs_gap <= 1.5:   cover_rate = 0.75
        elif abs_gap <= 3.0: cover_rate = 0.45
        elif abs_gap <= 5.0: cover_rate = 0.20
        else:                cover_rate = 0.05

    # Pre-compute shared track environment arrays for all horizon laps
    evo_map  = track_env.set_index('LapNumber')['track_evolution'].to_dict()
    temp_map = track_env.set_index('LapNumber')['TrackTemp'].to_dict()
    last_evo  = list(evo_map.values())[-1]
    last_temp = list(temp_map.values())[-1]

    horizon_laps    = [current_lap + n for n in range(horizon)]
    evos            = np.array([evo_map.get(l, last_evo)  for l in horizon_laps])
    temps           = np.array([temp_map.get(l, last_temp) for l in horizon_laps])
    fuel_corrections = np.array([(110.0 - 1.7 * l) * 0.03 for l in horizon_laps])

    # ── Stay-Out baseline: one model call → sample n_rollouts trajectories ──
    so_ages = np.array([current_state['tyre_age'] + n for n in range(horizon)])
    so_df   = _build_feature_batch(so_ages, current_state['compound_encoded'],
                                    current_state, evos, temps, feature_cols)
    so_p5, so_p50, so_p95 = _predict_quantiles(model, so_df)
    # Estimated std from 90% quantile interval (≈ 1.645σ each side)
    so_std = np.maximum((so_p95 - so_p5) / (2 * 1.645), 1e-5)
    # lap_time_delta samples: (n_rollouts, horizon)
    so_delta_samples = rng.normal(loc=so_p50, scale=so_std, size=(n_rollouts, horizon))
    # Absolute lap times: baseline_pace + delta + fuel_correction
    so_lap_times = current_state['baseline_pace'] + so_delta_samples + fuel_corrections
    so_total     = so_lap_times.sum(axis=1)   # (n_rollouts,) — total time staying out

    results: list[dict] = []

    for strat in STRATEGIES:
        pit_offset       = strat['pit_offset']
        strat_compound   = strat['compound_encoded']

        # ── Stay Out: net gain is 0 by definition (it IS the baseline) ──────
        if pit_offset is None:
            # Still compute per-lap crossover for the stay-out median line
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
                'p50_lap_times':      (current_state['baseline_pace'] + so_p50 + fuel_corrections).tolist(),
            })
            continue

        # ── Pit strategy: build tyre-age profile around the pit lap ─────────
        pit_ages = np.array([
            current_state['tyre_age'] + n if n < pit_offset else 1 + (n - pit_offset)
            for n in range(horizon)
        ], dtype=float)

        pit_df   = _build_feature_batch(pit_ages, strat_compound,
                                         current_state, evos, temps, feature_cols)
        pit_p5, pit_p50, pit_p95 = _predict_quantiles(model, pit_df)
        pit_std  = np.maximum((pit_p95 - pit_p5) / (2 * 1.645), 1e-5)

        # Sample n_rollouts degradation trajectories: (n_rollouts, horizon)
        pit_delta_samples = rng.normal(loc=pit_p50, scale=pit_std, size=(n_rollouts, horizon))
        pit_lap_times     = current_state['baseline_pace'] + pit_delta_samples + fuel_corrections

        # ── Safety Car stochastic events ─────────────────────────────────────
        # Binary mask: (n_rollouts,) — did an SC occur on the pit lap?
        sc_on_pit_lap = rng.binomial(1, sc_rate, size=n_rollouts).astype(bool)
        # Effective pit loss per rollout
        effective_pit_loss = np.where(sc_on_pit_lap, pit_loss_delta * SC_PIT_LOSS_MULTIPLIER, pit_loss_delta)

        # Add pit cost to the pit lap column
        pit_lap_times[:, pit_offset] += effective_pit_loss

        # ── Outcome distribution ──────────────────────────────────────────────
        # Base net gains if rival does not react
        base_net_gains_per_lap = so_lap_times - pit_lap_times
        
        # ── Feature 4: Rival Reaction Model ───────────────────────────────────
        # Rival reacts on the lap AFTER we pit (pit_offset + 1)
        # If rival reacts, we assume pace difference goes to 0 (both have fresh tyres).
        # Therefore, Driver A's advantage is clamped to whatever they gained 
        # up to and including the undercut lap (pit_offset + 1).
        rival_reacts = rng.binomial(1, cover_rate, size=n_rollouts).astype(bool)
        
        final_net_gains = np.zeros(n_rollouts)
        for i in range(n_rollouts):
            cumulative_gain = np.cumsum(base_net_gains_per_lap[i])
            if rival_reacts[i] and (pit_offset + 1 < horizon):
                # Clamp advantage after the reaction lap
                final_net_gains[i] = cumulative_gain[pit_offset + 1]
            else:
                final_net_gains[i] = cumulative_gain[-1]

        p_net_gain    = float((final_net_gains > 0).mean())
        expected_gain = float(final_net_gains.mean())
        gain_ci_low   = float(np.percentile(final_net_gains, 5))
        gain_ci_high  = float(np.percentile(final_net_gains, 95))

        # ── SC-aware expected gain (if we assume SC is certain this lap) ─────
        pit_lap_times_sc             = pit_lap_times.copy()
        pit_lap_times_sc[:, pit_offset] -= effective_pit_loss             # undo
        pit_lap_times_sc[:, pit_offset] += pit_loss_delta * SC_PIT_LOSS_MULTIPLIER  # re-add SC cost
        sc_net_gains  = so_total - pit_lap_times_sc.sum(axis=1)
        sc_expected   = float(sc_net_gains.mean())

        # ── Crossover lap: first lap where fresh-tyre p50 pace < stay-out p50 ─
        lap_delta_p50   = pit_p50 - so_p50     # <0 means fresh tyre is faster
        crossover_rel   = None
        for n in range(pit_offset, horizon):    # crossover can only happen after pitting
            if lap_delta_p50[n] < 0:
                crossover_rel = n
                break
        crossover_lap = (current_lap + crossover_rel) if crossover_rel is not None else None

        # ── Decision: window open if pitting is net-positive with ≥55% confidence
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
            'p50_lap_times':      (current_state['baseline_pace'] + pit_p50 + fuel_corrections).tolist(),
        })

    # Sort: Stay Out always first, then rank remaining by expected gain descending
    stay_out = [r for r in results if r['name'] == 'Stay Out']
    pit_strats = sorted(
        [r for r in results if r['name'] != 'Stay Out'],
        key=lambda r: r['expected_gain_s'],
        reverse=True,
    )
    return stay_out + pit_strats


# ── Backward-Compatible Wrapper ───────────────────────────────────────────────

def simulate_undercut_window(
    model,
    current_lap: int,
    current_state: dict,
    track_env: pd.DataFrame,
    pit_loss_delta: float,
    next_compound_encoded: int = 3,
    feature_cols: list[str] | None = None,
    race_name: str = "",
    n_rollouts: int = 500,
) -> dict:
    """
    Phase 1-compatible wrapper around run_mc_simulation().
    Returns the same keys as the original Phase 1 return value, plus the full
    mc_results list for the new Phase 2 UI panels.
    """
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

    # Find the strategy matching next_compound_encoded with pit_offset=0
    best_match = next(
        (r for r in mc_results if r['pit_offset'] == 0 and r['compound_encoded'] == next_compound_encoded),
        mc_results[1] if len(mc_results) > 1 else mc_results[0],
    )
    stay_out = mc_results[0]

    # Reconstruct legacy net_undercut_advantage list from p50 lap times
    so_times  = stay_out['p50_lap_times']
    pit_times = best_match['p50_lap_times']
    if pit_times:
        pit_times_with_loss = pit_times.copy()
        pit_times_with_loss[0] += pit_loss_delta
        lap_advantage          = [so_times[i] - pit_times[i] for i in range(len(so_times))]
        net_undercut_advantage = list(pd.Series(lap_advantage).cumsum())
        stay_out_curve         = list(pd.Series(so_times).cumsum())
        pit_now_curve          = list(pd.Series(pit_times_with_loss).cumsum())
    else:
        net_undercut_advantage = []
        stay_out_curve         = []
        pit_now_curve          = []

    return {
        # Legacy Phase 1 keys (still used by old app.py chart code)
        'stay_out_curve':        stay_out_curve,
        'pit_now_curve':         pit_now_curve,
        'net_undercut_advantage': net_undercut_advantage,
        'crossover_lap':         best_match['crossover_lap'],
        'window_open':           best_match['window_open'],
        'stay_out_lap_times':    so_times,
        'pit_now_lap_times':     pit_times,
        # Phase 2 additions
        'mc_results':            mc_results,
        'best_strategy':         best_match,
        'p_net_gain':            best_match['p_net_gain'],
        'expected_gain_s':       best_match['expected_gain_s'],
        'gain_ci_low':           best_match['gain_ci_low'],
        'gain_ci_high':          best_match['gain_ci_high'],
        'sc_expected_gain_s':    best_match['sc_expected_gain_s'],
        'cover_rate':            best_match.get('cover_rate', 0.0),
    }


# ── Pit Loss Calculator (unchanged from Phase 1) ──────────────────────────────

def calculate_pit_loss(session) -> float:
    """
    Calculates the median time spent in the pit lane for this session.
    Matches in-laps (Stint N) with out-laps (Stint N+1) per driver.
    Returns the median pit-lane time in seconds.
    """
    laps = session.laps

    in_laps  = laps[~laps['PitInTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitInTime']]
    out_laps = laps[~laps['PitOutTime'].isna()][['Driver', 'Stint', 'LapNumber', 'PitOutTime']]

    in_laps = in_laps.copy()
    in_laps['NextStint'] = in_laps['Stint'] + 1
    pit_stops = pd.merge(
        in_laps, out_laps,
        left_on=['Driver', 'NextStint'],
        right_on=['Driver', 'Stint'],
        suffixes=('_in', '_out'),
    )
    pit_stops['PitLaneTime'] = (pit_stops['PitOutTime'] - pit_stops['PitInTime']).dt.total_seconds()
    return float(pit_stops['PitLaneTime'].median())


# ── CLI Self-Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib, os, fastf1
    from data_pipeline import clean_lap_data, merge_weather_data
    from feature_engineering import apply_feature_engineering
    from ml_model import prepare_ml_data

    model_path = os.path.join('models', 'global_degradation_model.joblib')
    if not os.path.exists(model_path):
        print("Global model not found. Run train_global_model.py first.")
        raise SystemExit(1)

    print("Loading global quantile model...")
    saved  = joblib.load(model_path)
    model  = saved['model']
    feats  = saved['features']
    is_q   = saved.get('model_version') == 'phase2_quantile'
    print(f"  Model type : {'Phase 2 Quantile' if is_q else 'Phase 1 Point Estimate'}")

    fastf1.Cache.enable_cache("f1_cache")
    session = fastf1.get_session(2023, 'Bahrain', 'R')
    session.load(weather=True, messages=False)

    pit_loss = calculate_pit_loss(session)
    print(f"  Pit loss   : {pit_loss:.2f}s")

    laps   = clean_lap_data(session.laps)
    laps   = merge_weather_data(laps, session.weather_data)
    eng    = apply_feature_engineering(laps)
    ml     = prepare_ml_data(eng)

    ver_row = ml[(ml['Driver'] == 'VER') & (ml['LapNumber'] == 14)].iloc[0]
    state   = {
        'tyre_age':        ver_row['tyre_age'],
        'compound_encoded': ver_row['compound_encoded'],
        'baseline_pace':   ver_row['baseline_pace'],
        'Track':           'Bahrain Grand Prix',
        'Driver':          'VER',
    }
    track_env = ml[['LapNumber', 'track_evolution', 'TrackTemp']].drop_duplicates('LapNumber')

    print("\nRunning Phase 2 Monte Carlo simulation (500 rollouts × 6 strategies)...")
    mc = run_mc_simulation(
        model=model, current_lap=14, current_state=state,
        track_env=track_env, pit_loss_delta=pit_loss,
        race_name='Bahrain Grand Prix', n_rollouts=500,
        feature_cols=feats,
    )

    print(f"\n{'Strategy':<22} {'P(gain)':>8} {'E[gain]':>8} {'CI 5%':>8} {'CI 95%':>8} {'Crossover':>10}")
    print("─" * 68)
    for r in mc:
        xo = f"Lap {r['crossover_lap']}" if r['crossover_lap'] else "None"
        print(f"  {r['name']:<20} {r['p_net_gain']:>7.1%} {r['expected_gain_s']:>+7.2f}s "
              f"{r['gain_ci_low']:>+7.2f}s {r['gain_ci_high']:>+7.2f}s {xo:>10}")
