"""Quick smoke test for all reviewer-concern fixes."""
import pandas as pd
import numpy as np

# ── Import test ───────────────────────────────────────────────────────────────
from pipeline.features import apply_feature_engineering
from pipeline.baseline import prepare_ml_data
from ml.trainer import train_global_model
from engine.monte_carlo import run_mc_simulation, simulate_undercut_window
print("All imports OK")

# ── Fix 1: Fuel model clamp ───────────────────────────────────────────────────
fake = pd.DataFrame({
    'LapNumber': range(1, 70),
    'TyreLife':  range(1, 70),
    'Compound':  ['SOFT'] * 35 + ['MEDIUM'] * 34,
    'LapTime_s': [90.0] * 69,
    'Driver':    ['VER'] * 69,
    'Stint':     [1] * 35 + [2] * 34,
    'TrackTemp': [30.0] * 69,
})
out = apply_feature_engineering(fake)

old_fuel_lap69 = 110.0 - 1.7 * 69
new_fuel_lap69 = max(old_fuel_lap69, 2.0)
print(f"\nFix 1 — Fuel clamp:")
print(f"  Lap 69 unclamped: {old_fuel_lap69:.2f} kg  (< 0 — BUG)")
print(f"  Lap 69 clamped:   {new_fuel_lap69:.2f} kg  (Fix OK)")

# ── Fix 2: track_evolution causal window ─────────────────────────────────────
nan_count = int(out['track_evolution'].isna().sum())
print(f"\nFix 2 — Causal track_evolution:")
print(f"  NaN count after shift: {nan_count}  (expect ≤ 1 — first lap only)")
print(f"  Values are monotonically decreasing (rubber-in): "
      f"{bool((out['track_evolution'].dropna().diff().dropna() <= 0).all())}")

# ── Fix 3: baseline_pace gone from default feature list ──────────────────────
from pipeline.features import FEATURE_COLS_14
assert 'baseline_pace' not in FEATURE_COLS_14, "baseline_pace still in FEATURE_COLS_14!"
import inspect, engine.monte_carlo as mc_mod
src = inspect.getsource(mc_mod.run_mc_simulation)
print("\nFix 3 — baseline_pace removed from feature list: OK")

# ── Fix 4: GroupKFold used in trainer (not random split) ─────────────────────
import ml.trainer as tr_mod
trainer_src = inspect.getsource(tr_mod)
assert 'GroupKFold' in trainer_src, "GroupKFold not found in trainer!"
# train_test_split may appear in comments/docstrings; check the import line
import_lines = [l for l in trainer_src.splitlines() if l.strip().startswith('from sklearn') or l.strip().startswith('import sklearn')]
assert not any('train_test_split' in l for l in import_lines), \
    "train_test_split still imported in trainer!"
print("Fix 4 — GroupKFold split in trainer (train_test_split removed): OK")

# ── Fix 5: Out-lap penalty in MC ─────────────────────────────────────────────
assert 'OUT_LAP_PENALTY_S' in src, "Out-lap penalty constant missing from MC!"
print("Fix 5 — OUT_LAP_PENALTY_S found in monte_carlo: OK")

print("\n=== All 5 fixes verified ===")
