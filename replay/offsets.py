"""
replay/offsets.py — Empirical Compound Offset Analysis
======================================================
Computes empirical tyre compound offsets across drivers using the existing
pipeline's per-stint baseline_pace.
"""

from __future__ import annotations

import itertools
from typing import Any
import numpy as np
import pandas as pd

from pipeline.ingestion import load_race_session, process_telemetry_pipeline


def compound_offsets(year: int, gp_name: str, session: Any = None) -> pd.DataFrame:
    """
    For each driver who ran two or more compounds, take the existing pipeline's
    per-stint baseline_pace and compute the difference between compounds for that driver.
    Report the median across drivers, the number of drivers, and the min/max for
    each compound pair. Print the table and return the DataFrame.
    """
    if session is None:
        session = load_race_session(year, gp_name)

    pipeline_res, err = process_telemetry_pipeline(session)
    if err or not pipeline_res or 'ml_data' not in pipeline_res:
        raise RuntimeError(f"Pipeline processing failed: {err}")

    ml_data = pipeline_res['ml_data']

    # 1. Isolate per-stint baseline_pace
    stint_pace = (
        ml_data.groupby(['Driver', 'Stint', 'Compound'])['baseline_pace']
        .first()
        .reset_index()
    )

    # 2. Identify drivers who ran two or more compounds
    driver_compounds = stint_pace.groupby('Driver')['Compound'].nunique()
    multi_comp_drivers = driver_compounds[driver_compounds >= 2].index.tolist()

    stint_multi = stint_pace[stint_pace['Driver'].isin(multi_comp_drivers)]

    # 3. Compute each driver's baseline pace per compound (mean across stints)
    driver_comp_pace = (
        stint_multi.groupby(['Driver', 'Compound'])['baseline_pace']
        .mean()
        .unstack()
    )

    all_compounds = sorted(driver_comp_pace.columns.tolist())
    compound_order = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    present_compounds = [c for c in compound_order if c in all_compounds] + [
        c for c in all_compounds if c not in compound_order
    ]

    results: list[dict[str, Any]] = []

    # 4. For each compound pair, compute delta for drivers running both
    for c1, c2 in itertools.combinations(present_compounds, 2):
        valid = driver_comp_pace[[c1, c2]].dropna()
        if valid.empty:
            continue

        # Delta: Pace(C2) - Pace(C1) (e.g. HARD - SOFT)
        deltas = valid[c2] - valid[c1]
        n_drivers = len(deltas)
        median_val = float(deltas.median())
        min_val = float(deltas.min())
        max_val = float(deltas.max())

        results.append({
            'Compound_1': c1,
            'Compound_2': c2,
            'Comparison': f"{c2} - {c1}",
            'N_Drivers': n_drivers,
            'Median_Offset_s': round(median_val, 4),
            'Min_Offset_s': round(min_val, 4),
            'Max_Offset_s': round(max_val, 4),
        })

    summary_df = pd.DataFrame(results)

    # 5. Print table
    print("=" * 80)
    print(f"COMPOUND OFFSETS — {year} {gp_name} Grand Prix")
    print("=" * 80)
    print(f"Total drivers running >= 2 compounds: {len(multi_comp_drivers)}")
    print(f"Drivers: {', '.join(sorted(multi_comp_drivers))}\n")

    if not summary_df.empty:
        header = f"{'Pair':<20} | {'Comparison':<16} | {'N Drivers':<10} | {'Median (s)':<12} | {'Min (s)':<10} | {'Max (s)':<10}"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for _, r in summary_df.iterrows():
            pair_name = f"{r['Compound_1']} vs {r['Compound_2']}"
            print(
                f"{pair_name:<20} | {r['Comparison']:<16} | {r['N_Drivers']:<10} | "
                f"{r['Median_Offset_s']:<+12.4f} | {r['Min_Offset_s']:<+10.4f} | {r['Max_Offset_s']:<+10.4f}"
            )
        print("=" * 80 + "\n")
    else:
        print("No multi-compound comparisons found.\n")

    return summary_df
