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

from replay.whatif import run_whatif

drivers = ["SAI", "HAM", "STR", "RUS"]
shifts = [-2, -1, 1, 2]

print("=== D3 CONTINUITY TABLE (SHIFTS -2, -1, +1, +2) ===")
rows = []
for drv in drivers:
    for s_idx in [0, 1]:
        # get actual lap
        d_df = rt[rt['Driver'] == drv]
        p_laps = d_df[d_df['is_in_lap'] == True]['LapNumber'].astype(int).tolist()
        act_lap = p_laps[s_idx]
        
        row_dict = {"Driver": drv, "Stop": s_idx + 1, "ActualLap": act_lap}
        for sh in shifts:
            res = run_whatif(rt, drv, s_idx, sh, g3_model, pit_loss_green=pit_loss_green, pit_loss_neutralized=pit_loss_neutralized, neutralized_laps=neutralized_laps, n_paths=200, seed=42, use_controlled_linear_extrap=True)
            row_dict[f"Shift_{sh:+d}"] = res['final_time_delta_p50']
        rows.append(row_dict)

res_df = pd.DataFrame(rows)
print(res_df.to_string(index=False))

