"""
simulation.py — Monte Carlo Simulation Facade (Delegates to engine/)
===================================================================
Maintained for backward compatibility. All simulation logic has been
componentized into the `engine/` package.
"""

from engine.strategies import STRATEGIES, COMPOUND_MAP, REVERSE_COMPOUND_MAP
from engine.safety_car import SC_RATES, SC_RATE_DEFAULT, SC_PIT_LOSS_MULTIPLIER, get_circuit_sc_rate
from engine.rival_reaction import calculate_cover_rate, apply_rival_reaction
from engine.pit_loss import calculate_pit_loss
from engine.monte_carlo import (
    run_mc_simulation,
    simulate_undercut_window,
    _build_feature_batch,
    _predict_quantiles,
)

__all__ = [
    "STRATEGIES",
    "COMPOUND_MAP",
    "REVERSE_COMPOUND_MAP",
    "SC_RATES",
    "SC_RATE_DEFAULT",
    "SC_PIT_LOSS_MULTIPLIER",
    "get_circuit_sc_rate",
    "calculate_cover_rate",
    "apply_rival_reaction",
    "calculate_pit_loss",
    "run_mc_simulation",
    "simulate_undercut_window",
    "_build_feature_batch",
    "_predict_quantiles",
]

if __name__ == "__main__":
    import os, joblib, fastf1
    from pipeline.ingestion import clean_lap_data, merge_weather_data
    from pipeline.features import apply_feature_engineering
    from pipeline.baseline import prepare_ml_data

    model_path = os.path.join('models', 'global_degradation_model.joblib')
    if not os.path.exists(model_path):
        print("Global model not found. Run train_global_model.py first.")
        raise SystemExit(1)

    print("Loading global quantile model...")
    saved = joblib.load(model_path)
    model = saved['model']
    feats = saved['features']

    fastf1.Cache.enable_cache("f1_cache")
    session = fastf1.get_session(2023, 'Bahrain', 'R')
    session.load(weather=True, messages=False)

    pit_loss = calculate_pit_loss(session)
    print(f"  Pit loss   : {pit_loss:.2f}s")

    laps = clean_lap_data(session.laps)
    laps = merge_weather_data(laps, session.weather_data)
    eng = apply_feature_engineering(laps)
    ml = prepare_ml_data(eng)

    ver_laps = ml[ml['Driver'] == 'VER']
    ver_row = ver_laps[ver_laps['LapNumber'] == 13].iloc[0] if not ver_laps[ver_laps['LapNumber'] == 13].empty else ver_laps.iloc[5]
    sim_test_lap = int(ver_row['LapNumber'])
    state = {
        'tyre_age':        ver_row['tyre_age'],
        'compound_encoded': ver_row['compound_encoded'],
        'baseline_pace':   ver_row['baseline_pace'],
        'Track':           'Bahrain Grand Prix',
        'Driver':          'VER',
    }
    track_env = ml[['LapNumber', 'track_evolution', 'TrackTemp']].drop_duplicates('LapNumber')

    print(f"\nRunning Phase 2 Monte Carlo simulation from Lap {sim_test_lap} (500 rollouts × 6 strategies)...")
    mc = run_mc_simulation(
        model=model, current_lap=sim_test_lap, current_state=state,
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
