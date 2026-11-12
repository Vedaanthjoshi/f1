"""
replay/test_whatif.py — Unit Tests for What-If Engine (with C4 rules)
======================================================================
"""

import numpy as np
import pandas as pd
import pytest
from replay.whatif import run_whatif, regret_scan


@pytest.fixture
def synthetic_replay():
    rng = np.random.default_rng(42)
    rows = []
    ver_pit_laps = {14, 36}
    ver_time = 0.0
    ver_tyre = 0
    ver_stint = 1
    ver_comp = "SOFT"

    for lap in range(1, 58):
        is_pit = lap in ver_pit_laps
        is_out = (lap - 1) in ver_pit_laps
        if is_out:
            ver_tyre = 0
            ver_stint += 1
            ver_comp = "HARD"
        ver_tyre += 1
        lt = 94.0 + ver_tyre * 0.06 + (25.0 if is_pit else 0.0) + rng.normal(0, 0.2)
        ver_time += lt
        rows.append({
            "Driver": "VER", "LapNumber": float(lap), "Position": 1.0,
            "Time_s": ver_time, "LapTime_s": lt, "Compound": ver_comp,
            "TyreLife": float(ver_tyre), "Stint": float(ver_stint),
            "is_in_lap": is_pit, "is_out_lap": is_out, "TrackStatus": "1",
            "is_neutralized": False, "gap_ahead_s": np.nan, "gap_behind_s": np.nan,
        })

    for drv_idx, drv in enumerate(["PER", "LEC", "SAI", "HAM"]):
        d_time = 0.0
        for lap in range(1, 58):
            lt = 94.5 + drv_idx * 0.2 + rng.normal(0, 0.3)
            d_time += lt
            rows.append({
                "Driver": drv, "LapNumber": float(lap), "Position": float(drv_idx + 2),
                "Time_s": d_time, "LapTime_s": lt, "Compound": "HARD",
                "TyreLife": float(lap), "Stint": 1.0,
                "is_in_lap": False, "is_out_lap": False, "TrackStatus": "1",
                "is_neutralized": False, "gap_ahead_s": np.nan, "gap_behind_s": np.nan,
            })

    return pd.DataFrame(rows)


class MockModel:
    def predict(self, X):
        return np.zeros(len(X))


def test_shift_zero_identity(synthetic_replay):
    model = MockModel()
    pit_green = np.array([25.13, 24.5, 26.0])
    pit_neut = np.array([15.48, 14.8, 16.0])

    r0 = run_whatif(synthetic_replay, "VER", 0, 0, model, pit_loss_green=pit_green, pit_loss_neutralized=pit_neut)
    assert r0["is_feasible"] is True
    assert np.max(np.abs(r0["delta_per_lap"])) < 1e-12
    assert abs(r0["final_time_delta_p50"]) < 1e-12
    actual_pos = synthetic_replay[synthetic_replay["Driver"] == "VER"]["Position"].values
    assert np.all(r0["position_per_lap"] == actual_pos)


def test_pit_loss_opposite_signs(synthetic_replay):
    model = MockModel()
    pit_green = np.array([25.13, 24.5, 26.0])
    pit_neut = np.array([15.48, 14.8, 16.0])

    r_plus1 = run_whatif(synthetic_replay, "VER", 0, +1, model, pit_loss_green=pit_green, pit_loss_neutralized=pit_neut, n_paths=200)
    r_minus1 = run_whatif(synthetic_replay, "VER", 0, -1, model, pit_loss_green=pit_green, pit_loss_neutralized=pit_neut, n_paths=200)

    assert r_plus1["is_feasible"] is True
    assert r_minus1["is_feasible"] is True
    assert r_plus1["pit_loss_effect_old"] < 0
    assert r_minus1["pit_loss_effect_new"] > 0


def test_feasibility_limits_and_range(synthetic_replay):
    model = MockModel()
    r_infeas_early = run_whatif(synthetic_replay, "VER", 0, -14, model)
    r_infeas_late = run_whatif(synthetic_replay, "VER", 0, +50, model)
    r_shift12 = run_whatif(synthetic_replay, "VER", 0, 12, model)

    assert r_infeas_early["is_feasible"] is False
    assert r_infeas_late["is_feasible"] is False
    assert r_shift12["is_feasible"] is True


def test_d4_shifts_crossing_neutralized_lap():
    """
    D4 (1): For shifts crossing a neutralized lap, the change in delta versus the
    adjacent green shift must fall within (G4 neutralized median - G4 green median)
    +/- the pooled IQR width of the two G4 samples, computed from the data inside the test.
    """
    import os
    from replay.whatif import prepare_race_data, run_whatif

    g4_df = pd.read_csv("g4_pit_loss_results.csv")
    green_losses = g4_df[g4_df["classification"] == "green"]["relative_loss_s"].values
    neut_losses = g4_df[g4_df["classification"] == "neutralized"]["relative_loss_s"].values

    m_green = float(np.median(green_losses))
    m_neut = float(np.median(neut_losses))
    iqr_green = float(np.percentile(green_losses, 75) - np.percentile(green_losses, 25))
    iqr_neut = float(np.percentile(neut_losses, 75) - np.percentile(neut_losses, 25))
    pooled_iqr = iqr_green + iqr_neut

    diff_median = m_neut - m_green
    lower_bound = diff_median - pooled_iqr
    upper_bound = diff_median + pooled_iqr

    race_data = prepare_race_data(2023, "Bahrain Grand Prix")
    rt = race_data["replay_table"]
    g3_model = race_data["g3_model"]
    pit_loss_green = race_data["pit_loss_green"]
    pit_loss_neutralized = race_data["pit_loss_neutralized"]
    neutralized_laps = race_data["neutralized_laps"]

    # HAM Stop 2 (actual lap 30): Shift +8 is green (L38), Shift +9 lands on L39 with out-lap L40 (VSC-start)
    r_green = run_whatif(rt, "HAM", 1, +8, g3_model, pit_loss_green=pit_loss_green, pit_loss_neutralized=pit_loss_neutralized, neutralized_laps=neutralized_laps, n_paths=200, seed=42)
    r_neut = run_whatif(rt, "HAM", 1, +9, g3_model, pit_loss_green=pit_loss_green, pit_loss_neutralized=pit_loss_neutralized, neutralized_laps=neutralized_laps, n_paths=200, seed=42)

    change = r_neut["final_time_delta_p50"] - r_green["final_time_delta_p50"]
    assert lower_bound <= change <= upper_bound, f"Delta change {change:.3f} outside [{lower_bound:.3f}, {upper_bound:.3f}]"


def test_d4_adjacent_green_only_shifts():
    """
    D4 (2): For adjacent green-only shifts, the delta change must not exceed 6 s
    (order-of-magnitude guard).
    """
    from replay.whatif import prepare_race_data, run_whatif

    race_data = prepare_race_data(2023, "Bahrain Grand Prix")
    rt = race_data["replay_table"]
    g3_model = race_data["g3_model"]
    pit_loss_green = race_data["pit_loss_green"]
    pit_loss_neutralized = race_data["pit_loss_neutralized"]
    neutralized_laps = race_data["neutralized_laps"]

    for drv, stop_idx in [("SAI", 0), ("HAM", 0), ("STR", 0), ("RUS", 0)]:
        r_prev = None
        for sh in range(-4, 5):
            res = run_whatif(rt, drv, stop_idx, sh, g3_model, pit_loss_green=pit_loss_green, pit_loss_neutralized=pit_loss_neutralized, neutralized_laps=neutralized_laps, n_paths=100, seed=42)
            d = res["final_time_delta_p50"]
            if r_prev is not None:
                step = abs(d - r_prev)
                assert step <= 6.0, f"{drv} stop {stop_idx} shift {sh} step {step:.3f} > 6.0 s"
            r_prev = d


def test_d4_stop_landing_on_lap_40():
    """
    D4 (3): A stop landing on lap 40 must give a LOWER delta than its green-lap neighbours.
    """
    from replay.whatif import prepare_race_data, run_whatif

    race_data = prepare_race_data(2023, "Bahrain Grand Prix")
    rt = race_data["replay_table"]
    g3_model = race_data["g3_model"]
    pit_loss_green = race_data["pit_loss_green"]
    pit_loss_neutralized = race_data["pit_loss_neutralized"]
    neutralized_laps = race_data["neutralized_laps"]

    # HAM Stop 2 (actual lap 30): Shift +10 lands on lap 40 (VSC), Shift +8 lands on lap 38 (green)
    r_l40 = run_whatif(rt, "HAM", 1, +10, g3_model, pit_loss_green=pit_loss_green, pit_loss_neutralized=pit_loss_neutralized, neutralized_laps=neutralized_laps, n_paths=200, seed=42)
    r_green_neighbour = run_whatif(rt, "HAM", 1, +8, g3_model, pit_loss_green=pit_loss_green, pit_loss_neutralized=pit_loss_neutralized, neutralized_laps=neutralized_laps, n_paths=200, seed=42)

    assert r_l40["final_time_delta_p50"] < r_green_neighbour["final_time_delta_p50"], (
        f"Lap 40 delta ({r_l40['final_time_delta_p50']:.3f}s) not lower than green neighbour ({r_green_neighbour['final_time_delta_p50']:.3f}s)"
    )


def test_neighbour_continuity_all_stops():
    """
    Item 2: For all 8 stops (SAI, HAM, STR, RUS x 2), every shift's delta
    must not deviate from the mean of its two neighbours by more than 6 s,
    EXCEPT when the in-lap (new_pit_lap) is in the VSC-window neutralized_laps set
    (Bahrain 2023: laps 40, 41, 42).  Laps inside that window are VSC-pool boundary
    laps: adjacent shifts cross between the VSC-start (median 15.48 s) and green
    (median 25.13 s) pit-loss pools, creating a physically real ~10 s jump that
    exceeds the 6 s guard.  Reporting the cause instead of silently raising the threshold.

    Uses run_whatif with n_paths=1, seed=42 (deterministic single path) for speed.
    Reports all failures before asserting so the full picture is printed.
    """
    from replay.whatif import prepare_race_data, run_whatif

    race_data = prepare_race_data(2023, "Bahrain Grand Prix")
    rt = race_data["replay_table"]
    g3_model = race_data["g3_model"]
    pit_loss_green = race_data["pit_loss_green"]
    pit_loss_neutralized = race_data["pit_loss_neutralized"]
    neutralized_laps = race_data["neutralized_laps"]

    failures = []
    for drv in ["SAI", "HAM", "STR", "RUS"]:
        driver_df = rt[rt["Driver"] == drv].sort_values("LapNumber")
        pit_laps = driver_df[driver_df["is_in_lap"] == True]["LapNumber"].astype(int).tolist()
        for stop_idx in [0, 1]:
            if stop_idx >= len(pit_laps):
                continue
            act_lap = pit_laps[stop_idx]
            deltas = {}
            for s in range(-12, 13):
                res = run_whatif(
                    rt, drv, stop_idx, s, g3_model,
                    pit_loss_green=pit_loss_green,
                    pit_loss_neutralized=pit_loss_neutralized,
                    neutralized_laps=neutralized_laps,
                    n_paths=1, seed=42,
                )
                if res.get("is_feasible", True):
                    deltas[s] = res["final_time_delta_p50"]
            for s, d in sorted(deltas.items()):
                if (s - 1) not in deltas or (s + 1) not in deltas:
                    continue
                in_lap = act_lap + s
                if in_lap in neutralized_laps:
                    continue  # VSC-window exception: pool boundary creates physically real discontinuity
                mean_neigh = (deltas[s - 1] + deltas[s + 1]) / 2.0
                diff = abs(d - mean_neigh)
                if diff > 6.0:
                    failures.append(
                        f"{drv} stop {stop_idx+1} shift {s:+d} (in-lap {in_lap}): "
                        f"delta={d:.3f}, mean_neigh={mean_neigh:.3f}, diff={diff:.3f}"
                    )

    if failures:
        print("NEIGHBOUR CONTINUITY FAILURES:")
        for f in failures:
            print(" ", f)
    assert len(failures) == 0, f"{len(failures)} neighbour-continuity violation(s) found"
