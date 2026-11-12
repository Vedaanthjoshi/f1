"""G4: Relative pit loss per stop - Bahrain 2023"""
import sys
sys.path.insert(0, 'c:/f1')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import fastf1

fastf1.Cache.enable_cache('c:/f1/f1_cache')

from replay.load import load_replay_table, get_race_events, NEUTRALIZATION_CODES

print("Loading Bahrain 2023 replay table...")
session = fastf1.get_session(2023, 'Bahrain Grand Prix', 'R')
session.load(weather=False, messages=False)
table = load_replay_table(2023, 'Bahrain Grand Prix', session=session)
print(f"Replay table: {len(table)} rows, drivers: {table['Driver'].nunique()}")
print(f"Columns: {list(table.columns)}")

events = get_race_events(table)
pit_stops = events['pit_stops']
print(f"\nTotal pit stops found: {len(pit_stops)}")
print(f"SC laps: {events['sc_lap_ranges']}")
print(f"VSC laps: {events['vsc_lap_ranges']}")
print(f"Neutralized laps: {events['neutralized_lap_ranges']}")

# Build a lookup: for each driver and lapnum, get Time_s
time_lookup = {}
for _, row in table.iterrows():
    drv = row['Driver']
    lap = int(row['LapNumber'])
    if (drv, lap) not in time_lookup:
        time_lookup[(drv, lap)] = row['Time_s']

# All drivers
all_drivers = table['Driver'].unique()

# Compute relative pit loss for each stop
# pit_loss = (pitting driver's Time_s[out_lap] - Time_s[in_lap - 1])
#           - median(non-pitting cars' Time_s[out_lap] - Time_s[in_lap - 1])
# Note: Time_s = cumulative session time for that driver at end of that lap

records = []
for stop in pit_stops:
    drv = stop['driver']
    in_lap = stop['in_lap']
    out_lap = stop['out_lap']
    if out_lap is None:
        print(f"  Skipping {drv} in_lap={in_lap}: no out_lap found")
        continue

    pre_lap = in_lap - 1  # lap before pit in

    # Get pitting driver's times
    pit_pre = time_lookup.get((drv, pre_lap), None)
    pit_out = time_lookup.get((drv, out_lap), None)
    if pit_pre is None or pit_out is None:
        print(f"  Skipping {drv} in_lap={in_lap}: missing time data (pre={pit_pre}, out={pit_out})")
        continue
    pit_window = pit_out - pit_pre

    # Get non-pitting cars over same window (in_lap to out_lap)
    # Non-pitting = cars that did NOT have their in_lap in [pre_lap+1 .. out_lap]
    # i.e., didn't pit during the window
    all_in_laps = {stop2['in_lap'] for stop2 in pit_stops if stop2['driver'] != drv}

    non_pit_deltas = []
    for other_drv in all_drivers:
        if other_drv == drv:
            continue
        # Check if this other driver pitted during our window
        other_stops = [s for s in pit_stops if s['driver'] == other_drv]
        pitted_in_window = any(
            (s['in_lap'] >= pre_lap + 1 and s['in_lap'] <= out_lap) or
            (s['out_lap'] is not None and s['out_lap'] >= pre_lap + 1 and s['out_lap'] <= out_lap)
            for s in other_stops
        )
        if pitted_in_window:
            continue

        other_pre = time_lookup.get((other_drv, pre_lap), None)
        other_out = time_lookup.get((other_drv, out_lap), None)
        if other_pre is None or other_out is None:
            continue
        delta = other_out - other_pre
        if np.isfinite(delta) and delta > 0:
            non_pit_deltas.append(delta)

    if len(non_pit_deltas) < 3:
        print(f"  Warning: {drv} in_lap={in_lap}: only {len(non_pit_deltas)} non-pitting reference cars")
        if len(non_pit_deltas) == 0:
            print(f"    -> Skipping")
            continue

    median_non_pit = np.median(non_pit_deltas)
    relative_loss = pit_window - median_non_pit

    # Classify neutralized vs green
    neutralized_laps_set = set()
    for lap_num, lap_grp in table.groupby('LapNumber'):
        if any(lap_grp['is_neutralized']):
            neutralized_laps_set.add(int(lap_num))

    is_neutralized = (in_lap in neutralized_laps_set) or (out_lap in neutralized_laps_set)
    classification = 'neutralized' if is_neutralized else 'green'

    records.append({
        'driver': drv,
        'in_lap': in_lap,
        'out_lap': out_lap,
        'compound_before': stop['compound_before'],
        'compound_after': stop['compound_after'],
        'pit_window_s': pit_window,
        'median_non_pit_s': median_non_pit,
        'relative_loss_s': relative_loss,
        'n_ref_cars': len(non_pit_deltas),
        'classification': classification,
    })

results_df = pd.DataFrame(records)
print(f"\nComputed relative pit loss for {len(results_df)} stops")
print(results_df.to_string(index=False))

# Summary stats
print("\n\n=== G4 SUMMARY ===")
for cls in ['green', 'neutralized']:
    sub = results_df[results_df['classification'] == cls]['relative_loss_s']
    if len(sub) > 0:
        q25, q75 = np.percentile(sub, [25, 75])
        print(f"\n{cls.upper()} stops (n={len(sub)}):")
        print(f"  Median pit loss: {sub.median():.2f}s")
        print(f"  IQR: [{q25:.2f}s, {q75:.2f}s]")
        print(f"  Mean: {sub.mean():.2f}s, Std: {sub.std():.2f}s")
    else:
        print(f"\n{cls.upper()} stops: none")

results_df.to_csv('c:/f1/g4_pit_loss_results.csv', index=False)
print("\nSaved to c:/f1/g4_pit_loss_results.csv")
print("\n=== FULL STOP LIST ===")
for _, row in results_df.iterrows():
    print(f"  {row['driver']:4s} in_lap={int(row['in_lap']):2d} out_lap={int(row['out_lap']):2d} "
          f"{row['compound_before']}->{row['compound_after']:5s} "
          f"pit_window={row['pit_window_s']:6.2f}s ref_median={row['median_non_pit_s']:6.2f}s "
          f"loss={row['relative_loss_s']:+7.2f}s [{row['classification']}]")

