"""
run_regret_tables.py — Regenerate stop-1 AND stop-2 regret tables for SAI, HAM, STR, RUS.
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

g3_residuals = pd.read_csv('g3_pairs_results.csv')['error'].values if pd.io.common.file_exists('g3_pairs_results.csv') else np.zeros(100)

from replay.whatif import regret_scan

DRIVERS = ["SAI", "HAM", "STR", "RUS"]
N_PATHS = 300
SEED    = 42

for driver in DRIVERS:
    for stop_idx in [0, 1]:
        driver_df = rt[rt['Driver'] == driver].sort_values('LapNumber')
        pit_laps  = driver_df[driver_df['is_in_lap'] == True]['LapNumber'].astype(int).tolist()
        if stop_idx >= len(pit_laps):
            print(f"\n{driver} Stop {stop_idx+1}: only {len(pit_laps)} stop(s) — skipped")
            continue

        act_lap = pit_laps[stop_idx]
        print(f"\n{'='*80}")
        print(f"{driver} Stop {stop_idx+1}  (actual pit lap = {act_lap})")
        print(f"{'='*80}")

        df = regret_scan(
            replay_table=rt,
            driver=driver,
            stop_idx=stop_idx,
            g3_model=g3_model,
            g3_residuals=g3_residuals,
            pit_loss_green=pit_loss_green,
            pit_loss_neutralized=pit_loss_neutralized,
            neutralized_laps=neutralized_laps,
            n_paths=N_PATHS,
            seed=SEED,
        )

        # Print the full table
        df['new_pit'] = act_lap + df['shift']
        display = df[['shift', 'new_pit', 'stop_type', 'expected_time_delta', 'p05', 'p95',
                       'p_position_improves', 'is_extrapolated', 'robustness']].copy()
        display.columns = ['shift', 'new_pit', 'stop_type', 'E[delta]_s', 'p05_s', 'p95_s',
                           'P(pos_up)', 'extrap', 'robust']
        print(display.to_string(index=False, float_format='{:+.2f}'.format))
