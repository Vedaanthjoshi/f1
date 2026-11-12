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

    g4_df = pd.read_csv("results/g4_pit_loss_results.csv")
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


def test_rival_cover_and_dirty_air_reaction():
    """
    Verify that in 'cover' and 'cover_traffic' modes:
    1. A rival ahead within the pit window reacts to an early undercut by boxing +1 lap after.
    2. Dirty-air wake penalties increase counterfactual time when emerging behind a slower car.
    """
    from replay.whatif import compute_race_pit_losses, get_circuit_metadata

    rows = []
    # Build a 2-car + 3-backmarker synthetic race where PER is ~1.5s ahead of VER and pits at Lap 16,
    # while VER pits at Lap 14. Shifting VER to Lap 11 (-3) threatens PER with an undercut.
    for drv, pit_lap, base_pace, pos_init in [
        ("PER", 16, 93.8, 1.0),
        ("VER", 14, 94.0, 2.0),
        ("LEC", 25, 94.6, 3.0),
        ("SAI", 26, 94.8, 4.0),
        ("HAM", 27, 95.0, 5.0),
    ]:
        cum_t = 0.0 if drv == "PER" else (1.5 if drv == "VER" else 22.0 + pos_init)
        tyre = 0
        stint = 1
        comp = "SOFT"
        for lap in range(1, 40):
            is_in = (lap == pit_lap)
            is_out = (lap == pit_lap + 1)
            if is_out:
                tyre = 0
                stint += 1
                comp = "HARD"
            tyre += 1
            lt = base_pace + tyre * 0.10 + (24.0 if is_in else 0.0)
            cum_t += lt
            rows.append({
                "Driver": drv, "LapNumber": float(lap), "Position": pos_init,
                "Time_s": cum_t, "LapTime_s": lt, "Compound": comp,
                "TyreLife": float(tyre), "Stint": float(stint),
                "is_in_lap": is_in, "is_out_lap": is_out, "TrackStatus": "1",
                "is_neutralized": False, "gap_ahead_s": np.nan, "gap_behind_s": np.nan,
            })

    df = pd.DataFrame(rows)
    model = MockModel()
    pit_green, pit_neut = compute_race_pit_losses(df)
    assert len(pit_green) >= 2
    assert 20.0 <= float(np.median(pit_green)) <= 30.0

    phys, diff = get_circuit_metadata("Monaco Grand Prix")
    assert phys["abrasion_index"] == 1
    assert diff == "High"

    res_fixed = run_whatif(
        df, "VER", 0, -3, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=100, rival_mode="fixed",
    )
    res_cover = run_whatif(
        df, "VER", 0, -3, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=100, rival_mode="cover",
    )
    res_traffic = run_whatif(
        df, "VER", 0, -3, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=100, rival_mode="cover_traffic", overtaking_difficulty="High",
    )

    assert res_cover["is_feasible"] is True
    assert len(res_cover["rival_reactions"]) >= 1
    per_rx = [r for r in res_cover["rival_reactions"] if r["driver"] == "PER"]
    assert len(per_rx) == 1
    assert per_rx[0]["reacted_pit"] == 12  # +1 lap after VER's new Lap 11 stop
    assert res_traffic["dirty_air_laps"] >= 0.0
    assert res_traffic["final_time_delta_p50"] >= res_fixed["final_time_delta_p50"]


def test_step2_teammate_overcut_and_overtake_pace_cap():
    """
    Verify Step 2 physics:
    1. Teammates do not trigger adversarial undercut cover against each other.
    2. Staying out longer (shift > 0) triggers an opportunistic undercut from a close non-teammate behind.
    3. High overtaking difficulty (Monaco, 1.80 s/lap threshold) blocks a car whose pace advantage
       is smaller than threshold, whereas Low difficulty (Monza, 0.55 s/lap) lets it pass.
    """
    from replay.whatif import compute_race_pit_losses

    rows = []
    # PER (Red Bull, Lap 16) is 1.5s ahead of VER (Red Bull, Lap 14) -> teammates should NOT cover.
    # LEC (Ferrari, Lap 18) is 1.5s behind VER -> if VER shifts +4 (Lap 18), LEC attacks with undercut.
    # Also add slow midfield car ALB (Williams, no pit until Lap 32) ~22.5s ahead of VER on Lap 10 so
    # an early stop at Lap 11 drops VER right behind ALB with ~1.0s/lap pace advantage (blocked at High, passes at Low).
    specs = [
        ("PER", "Red Bull", 16, 93.8, 0.0, 1.0),
        ("VER", "Red Bull", 14, 94.0, 1.5, 2.0),
        ("LEC", "Ferrari", 18, 94.0, 3.0, 3.0),
        ("ALB", "Williams", 32, 94.8, 15.0, 4.0),
    ]
    for drv, team, pit_lap, base_pace, init_offset, pos_init in specs:
        cum_t = init_offset
        tyre = 0
        stint = 1
        comp = "SOFT"
        for lap in range(1, 36):
            is_in = (lap == pit_lap)
            is_out = (lap == pit_lap + 1)
            if is_out:
                tyre = 0
                stint += 1
                comp = "HARD"
            tyre += 1
            lt = base_pace + tyre * 0.08 + (24.0 if is_in else 0.0)
            cum_t += lt
            rows.append({
                "Driver": drv, "Team": team, "LapNumber": float(lap), "Position": pos_init,
                "Time_s": cum_t, "LapTime_s": lt, "Compound": comp,
                "TyreLife": float(tyre), "Stint": float(stint),
                "is_in_lap": is_in, "is_out_lap": is_out, "TrackStatus": "1",
                "is_neutralized": False, "gap_ahead_s": np.nan, "gap_behind_s": np.nan,
            })

    df = pd.DataFrame(rows)
    model = MockModel()
    pit_green, pit_neut = compute_race_pit_losses(df)

    # 1. Teammate PER should not adversarially cover VER on shift=-3
    res_team = run_whatif(
        df, "VER", 0, -3, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=50, rival_mode="cover",
    )
    assert all(r["driver"] != "PER" for r in res_team["rival_reactions"])

    # 2. Staying out longer (shift=+4 -> Lap 18) exposes VER to LEC behind attacking with undercut
    res_overcut = run_whatif(
        df, "VER", 0, +4, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=50, rival_mode="cover",
    )
    lec_rx = [r for r in res_overcut["rival_reactions"] if r["driver"] == "LEC"]
    assert len(lec_rx) == 1
    assert lec_rx[0]["reacted_pit"] < 18

    # 3. High vs Low overtaking difficulty pace-cap check
    res_monaco = run_whatif(
        df, "VER", 0, -3, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=50, rival_mode="cover_traffic", overtaking_difficulty="High",
    )
    res_monza = run_whatif(
        df, "VER", 0, -3, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=50, rival_mode="cover_traffic", overtaking_difficulty="Low",
    )
    assert res_monaco["final_time_delta_p50"] >= res_monza["final_time_delta_p50"]
    assert res_monaco["blocked_laps"] >= res_monza["blocked_laps"]


def test_step3_4_5_physics_extrapolation_pred_curve_and_regret_scan(synthetic_replay):
    """
    Verify Steps 3, 4, and 5:
    - _controlled_linear_increment scales with circuit abrasion_index and adds non-linear tyre cliff.
    - predict_stint_degradation_curve returns monotonic predictions anchored at 0.0 at anchor_age.
    - NEUTRALIZATION_CODES includes '5' (Red Flag) in replay/load.py.
    - regret_scan outputs robustness and blocked_laps columns cleanly.
    """
    from replay.whatif import _controlled_linear_increment, predict_stint_degradation_curve
    from replay.load import NEUTRALIZATION_CODES

    assert "5" in NEUTRALIZATION_CODES

    # Abrasion scaling (Qatar 5.0 > Monza 1.0)
    inc_monza = _controlled_linear_increment(10, 15, 1, 85.0, 70.0, abrasion_index=1.0)
    inc_qatar = _controlled_linear_increment(10, 15, 1, 85.0, 70.0, abrasion_index=5.0)
    assert inc_qatar > inc_monza

    # Non-linear cliff past SOFT cliff age (22)
    inc_pre_cliff = _controlled_linear_increment(10, 15, 1, 85.0, 70.0, abrasion_index=3.0)
    inc_post_cliff = _controlled_linear_increment(23, 28, 1, 85.0, 70.0, abrasion_index=3.0)
    assert inc_post_cliff > inc_pre_cliff

    # Predicted degradation curve anchored at anchor_age=2
    ages = [2, 5, 10, 15, 20, 25]
    laps = [2, 5, 10, 15, 20, 25]
    lin_curve, xgb_curve = predict_stint_degradation_curve(
        ages, laps, "SOFT", g3_model=None,
        circuit_physics={"abrasion_index": 4.0, "circuit_length_km": 5.4, "num_corners": 15.0},
        anchor_age=2,
    )
    assert abs(lin_curve[0]) < 1e-9
    assert np.all(np.diff(lin_curve) > 0)
    assert len(xgb_curve) == len(ages)

    # Regret scan columns
    model = MockModel()
    pit_green = np.array([25.13, 24.5, 26.0])
    pit_neut = np.array([15.48, 14.8, 16.0])
    scan_df = regret_scan(
        synthetic_replay, "VER", 0, model,
        pit_loss_green=pit_green, pit_loss_neutralized=pit_neut,
        n_paths=40, rival_mode="cover_traffic",
    )
    assert not scan_df.empty
    for col in ("robustness", "is_extrapolated", "blocked_laps", "dirty_air_laps", "n_rival_reactions"):
        assert col in scan_df.columns


def test_physics_ensemble_and_race_r2_stats(synthetic_replay):
    """Verify Race-Review 5-lap R² stats and non-memorizing PhysicsDegradationEnsemble invariants."""
    from replay.whatif import compute_race_degradation_stats
    from ml.trainer import PhysicsDegradationEnsemble
    from pipeline.features import FEATURE_COLS_14

    cp = {"abrasion_index": 4.0, "circuit_length_km": 5.4, "num_corners": 15.0}
    _, stats = compute_race_degradation_stats(synthetic_replay, g3_model=None, circuit_physics=cp)
    for key in ("n_pairs", "xgb_mae", "linear_mae", "xgb_r2", "linear_r2", "race_inc_bias", "race_alpha"):
        assert key in stats
    assert stats["n_pairs"] > 0

    rng = np.random.default_rng(7)
    n = 120
    ages = rng.integers(1, 28, size=n).astype(float)
    X_syn = pd.DataFrame({
        "tyre_age": ages,
        "compound_encoded": rng.choice([1.0, 2.0, 3.0], size=n),
        "compound_hardness": rng.choice([2.0, 3.0, 4.0], size=n),
        "track_evolution": rng.uniform(75.0, 95.0, size=n),
        "TrackTemp": rng.uniform(22.0, 45.0, size=n),
        "fuel_mass": rng.uniform(10.0, 100.0, size=n),
        "abrasion_index": np.full(n, 3.0),
        "circuit_length_km": rng.uniform(3.3, 6.5, size=n),
        "num_corners": rng.integers(10, 22, size=n).astype(float),
        "corner_density": np.full(n, 3.0),
        "thermal_load": ages * 30.0,
        "abrasion_load": ages * 3.0,
        "wear_cliff": (ages / 15.0) ** 2.0,
        "compound_age_interaction": ages * 3.0,
    })[FEATURE_COLS_14]
    y_syn = 0.04 * ages + rng.normal(0.0, 0.15, size=n)

    ens = PhysicsDegradationEnsemble(ridge_weight=0.65, seed=42)
    ens.fit(X_syn, y_syn)
    preds = ens.predict(X_syn)
    assert preds.shape == (n, 3)
    assert np.all(preds[:, 0] <= preds[:, 1] + 1e-9)
    assert np.all(preds[:, 1] <= preds[:, 2] + 1e-9)

    probe = pd.concat([X_syn.iloc[[0]].assign(tyre_age=a) for a in (1.0, 10.0, 20.0)], ignore_index=True)
    curve = ens.predict(probe)[:, 1]
    assert curve[2] >= curve[0]

    imp_map = dict(zip(FEATURE_COLS_14, ens.feature_importances_))
    for leaker in ("track_evolution", "circuit_length_km", "num_corners"):
        assert imp_map[leaker] == 0.0
