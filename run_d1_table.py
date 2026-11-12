"""
run_d1_table.py — D1 Decomposition Table
=========================================
Calls decompose_whatif() from replay.whatif (the SHARED function also used by
run_whatif / regret_scan) so both tables are guaranteed identical results.

No duplicate decompose_shift() logic lives here any more.
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

from replay.whatif import decompose_whatif

print("=== D1 TABLE COMPUTATION (via shared decompose_whatif) ===")
cases = [
    ("SAI", 1, [-10, -9, -8, 8, 9, 10]),
    ("HAM", 1, [8, 9, 10, 11]),
]

for driver, stop_idx, shifts in cases:
    print(f"\n--- {driver} Stop {stop_idx+1} ---")
    for s in shifts:
        r = decompose_whatif(
            rt, driver, stop_idx, s, g3_model,
            pit_loss_green=pit_loss_green,
            pit_loss_neutralized=pit_loss_neutralized,
            neutralized_laps=neutralized_laps,
            use_controlled_linear_extrap=True,
        )
        if not r.get('is_feasible', True):
            print(f"Shift {s:+3d}: INFEASIBLE — {r['infeasible_reason']}")
            continue
        print(
            f"Shift {s:+3d} (New Pit L{r['new_pit']:2d}): "
            f"Total={r['total']:+7.2f} | "
            f"(i)Remove={r['comp_i']:+6.2f} | "
            f"(ii)Add={r['comp_ii']:+6.2f} [{r['sample_label']}] | "
            f"(iii)InOutRepl={r['comp_iii']:+6.2f} | "
            f"(iv)InOutNew={r['comp_iv']:+6.2f} | "
            f"(v)DegExtrap={r['comp_v']:+6.2f} | "
            f"(vi)Other={r['comp_vi']:+5.2f} | "
            f"Resid={r['residual']:+6.4f} | "
            f"is_neut[act_in={r['act_in_n']}, act_out={r['act_out_n']}, "
            f"new_in={r['new_in_n']}, new_out={r['new_out_n']}]"
        )
