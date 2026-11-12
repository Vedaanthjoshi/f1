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

from replay.whatif import regret_scan

drivers = ["SAI", "HAM", "STR", "RUS"]

for drv in drivers:
    print(f"\n{'='*80}\nREGENERATED REGRET TABLE: {drv} STOP 2\n{'='*80}")
    # Stop 2 index is 1
    scan = regret_scan(
        replay_table=rt,
        driver=drv,
        stop_idx=1,
        g3_model=g3_model,
        pit_loss_green=pit_loss_green,
        pit_loss_neutralized=pit_loss_neutralized,
        neutralized_laps=neutralized_laps,
        n_paths=300,
        seed=42,
    )
    print(scan[["shift", "expected_time_delta", "p05", "p95", "p_position_improves", "is_extrapolated", "robustness", "stop_type"]].to_string(index=False))

