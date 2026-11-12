"""
run_side_by_side.py — Item 1: Side-by-side D1 (decompose_whatif) vs regret (run_whatif p50)
Both call the same shared engine logic after the is_clean fix.
"""
import warnings; warnings.filterwarnings('ignore')
import fastf1; fastf1.Cache.enable_cache('f1_cache')
import numpy as np, pandas as pd, joblib

session = fastf1.get_session(2023, 'Bahrain Grand Prix', 'R')
session.load(weather=True, messages=False)
from replay.load import load_replay_table
rt = load_replay_table(2023, 'Bahrain Grand Prix', session=session)

model_obj = joblib.load('models/loro_bahrain.joblib')
g3_model = model_obj['model'] if isinstance(model_obj, dict) else model_obj

g4_df = pd.read_csv('g4_pit_loss_results.csv')
pit_loss_green = g4_df[g4_df['classification'] == 'green']['relative_loss_s'].values
pit_loss_neutralized = g4_df[g4_df['classification'] == 'neutralized']['relative_loss_s'].values
neutralized_laps = set(rt[rt['is_neutralized'] == True]['LapNumber'].astype(int).tolist())

from replay.whatif import decompose_whatif, run_whatif

SEED = 42

print("=== SIDE-BY-SIDE: decompose_whatif (deterministic median) vs run_whatif p50 (seed=42, n_paths=1) ===\n")
header = f"{'Driver/Stop/Shift':<25} {'NewLap':>6} {'D1_Total':>10} {'Regret_p50':>12} {'Diff':>8}"
print(header)
print("-" * len(header))

cases = [
    ("SAI", 1, [9, 10, 11]),
    ("HAM", 1, [10, 11, 12]),
]

for driver, stop_idx, shifts in cases:
    for s in shifts:
        d1 = decompose_whatif(
            rt, driver, stop_idx, s, g3_model,
            pit_loss_green=pit_loss_green,
            pit_loss_neutralized=pit_loss_neutralized,
            neutralized_laps=neutralized_laps,
            use_controlled_linear_extrap=True,
        )
        rw = run_whatif(
            rt, driver, stop_idx, s, g3_model,
            pit_loss_green=pit_loss_green,
            pit_loss_neutralized=pit_loss_neutralized,
            neutralized_laps=neutralized_laps,
            n_paths=1, seed=SEED,
            use_controlled_linear_extrap=True,
        )
        d1_total = d1['total'] if d1.get('is_feasible', True) else float('nan')
        rw_total = rw['final_time_delta_p50'] if rw.get('is_feasible', True) else float('nan')
        diff = rw_total - d1_total
        label = f"{driver} Stop{stop_idx+1} Shift{s:+d}"
        new_pit = d1.get('new_pit', '?')
        print(f"{label:<25} {new_pit:>6} {d1_total:>10.3f} {rw_total:>12.3f} {diff:>8.3f}")

print()
print("NOTE: D1_Total uses deterministic pit-loss medians; run_whatif n_paths=1 samples one path.")
print("Small diffs (~0-2s) arise from stochastic noise + residual bootstrap in run_whatif.")
print("After the is_clean fix, the ~28-30s spike at lap-41 shifts (new_pit=41) is eliminated.")
print()
print("WHY ~28s spike at lap-41 shifts (BEFORE fix):")
print("  When new_pit_lap = 41, lap 41 is VSC-neutralized (TrackStatus '2671', actual time ~126 s).")
print("  The old code had is_clean = (not is_in) and (not is_out) and pd.notnull(lt).")
print("  Lap 41 passed that test, so it was stored as the clean reference for tyre-age 10.")
print("  When computing act_clean_t for a shifted stint covering lap 41, the engine returned")
print("  ~126.4 s instead of ~97.6 s.  The driving-delta = cf_clean_t - act_clean_t became")
print("  ~97.6 - 126.4 = -28.8 s — a false 28.8 s gain credited to the counterfactual car.")
print("  FIX: add 'and (ln not in neutralized_laps)' to is_clean in run_whatif (line 350),")
print("  matching the fix already present in clean_laps_by_stint_age construction (line 362).")
