"""
replay/eval_benchmark.py — Strictly Out-of-Sample R² & F1 Benchmark
====================================================================
Measures genuine generalization (no in-sample memorization) across:
1. Degradation Regression R² (MAE / RMSE / R²):
   - Clean Lap>=2 5-fold GroupKFold (grouped by Track) on data/historical_laps.csv
   - Zero-Shot Physics Ensemble + Causal In-Race Empirical Bayes (Lap < L)
   - Within-stint 5-lap & 10-lap increments (Ensemble vs Smooth Linear + LODO-Driver EB)
   - LORO-Bahrain (held-out Bahrain Grand Prix) on lap_time_delta & 5-lap increments
2. Strategy Undercut Decision F1 Score:
   - Curated 15-event Historical Backtest (Pure MC vs Monaco rule)
   - Uncurated 2024 Mined Green-Flag Undercut Duels (strictly out-of-sample from 2023 tuning)
   - Race Review Counterfactual Replay (run_whatif) on historical battles
"""

from __future__ import annotations
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.CRITICAL)

import fastf1
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

fastf1.Cache.enable_cache("f1_cache")
try:
    fastf1.Cache.offline_mode(True)
except Exception:
    pass

from pipeline.features import FEATURE_COLS_14, apply_feature_engineering
from pipeline.ingestion import clean_lap_data, merge_weather_data
from pipeline.baseline import prepare_ml_data
from ml.trainer import PhysicsDegradationEnsemble, estimate_causal_race_eb, apply_race_eb
from engine.pit_loss import calculate_pit_loss
from engine.monte_carlo import run_mc_simulation
from replay.whatif import _controlled_linear_increment, prepare_race_data, run_whatif
from backtester import HISTORICAL_UNDERCUTS, evaluate_single_undercut

MINED_DUELS_PATH = os.path.join(
    os.path.expanduser("~"),
    ".gemini", "antigravity", "brain",
    "3b7a3832-6965-4b97-9f35-57c45cca9308", "scratch", "mined_duels.csv",
)


def _compute_increment_pairs(
    df: pd.DataFrame,
    pred_col: str = "pred_delta",
    pred_eb_col: str | None = "pred_eb",
    step: int = 5,
) -> pd.DataFrame:
    """Build contiguous clean within-stint step-lap increment pairs (age_to - age_from == step)."""
    df_s = df.sort_values(["Season", "Track", "Driver", "Stint", "LapNumber"]).reset_index(drop=True)
    grp_cols = ["Season", "Track", "Driver", "Stint"]
    na = df_s.groupby(grp_cols)["tyre_age"].shift(-step)
    nl = df_s.groupby(grp_cols)["LapNumber"].shift(-step)
    nd = df_s.groupby(grp_cols)["lap_time_delta"].shift(-step)
    np0 = df_s.groupby(grp_cols)[pred_col].shift(-step)
    ok = (na - df_s["tyre_age"] == step) & (nl - df_s["LapNumber"] == step)

    lin_incs = [
        _controlled_linear_increment(
            int(a1), int(a2), int(ce), float(te), float(fm),
            abrasion_index=float(ab), track_temp=float(tt),
            is_stint1=(int(st) == 1), race_inc_bias=0.0,
        )
        for a1, a2, ce, te, fm, ab, tt, st in zip(
            df_s.loc[ok, "tyre_age"], na[ok], df_s.loc[ok, "compound_encoded"],
            df_s.loc[ok, "track_evolution"], df_s.loc[ok, "fuel_mass"],
            df_s.loc[ok, "abrasion_index"], df_s.loc[ok, "TrackTemp"],
            df_s.loc[ok, "Stint"].fillna(1),
        )
    ]
    out = pd.DataFrame({
        "Season": df_s.loc[ok, "Season"].to_numpy(),
        "Track": df_s.loc[ok, "Track"].astype(str).to_numpy(),
        "Driver": df_s.loc[ok, "Driver"].astype(str).to_numpy(),
        "actual_inc": (nd[ok] - df_s.loc[ok, "lap_time_delta"]).to_numpy(dtype=float),
        "xgb_inc": (np0[ok] - df_s.loc[ok, pred_col]).to_numpy(dtype=float),
        "lin_zs_inc": np.array(lin_incs, dtype=float),
    })
    if pred_eb_col and pred_eb_col in df_s.columns:
        npeb = df_s.groupby(grp_cols)[pred_eb_col].shift(-step)
        out["xgb_eb_inc"] = (npeb[ok] - df_s.loc[ok, pred_eb_col]).to_numpy(dtype=float)

    # Strictly Leave-One-Driver-Out Empirical Bayes race bias on linear increments
    res = out["actual_inc"] - out["lin_zs_inc"]
    tot_res = out.groupby(["Season", "Track"])["actual_inc"].transform("sum") - out.groupby(["Season", "Track"])["lin_zs_inc"].transform("sum")
    tot_cnt = out.groupby(["Season", "Track"])["actual_inc"].transform("count")
    drv_res = res.groupby([out["Season"], out["Track"], out["Driver"]]).transform("sum")
    drv_cnt = res.groupby([out["Season"], out["Track"], out["Driver"]]).transform("count")
    lodo_bias = (tot_res - drv_res) / ((tot_cnt - drv_cnt) + 15.0)
    out["lin_lodo_inc"] = np.clip(out["lin_zs_inc"] + lodo_bias.to_numpy(dtype=float), 0.0, 1.5)
    return out


def evaluate_degradation_r2(csv_path: str = "data/historical_laps.csv") -> dict:
    """Strictly out-of-sample GroupKFold(Track) and LORO-Bahrain R² evaluation on Lap>=2."""
    df = pd.read_csv(csv_path)
    if "LapNumber" in df.columns:
        df = df[df["LapNumber"] >= 2].copy()
    df = df.dropna(subset=FEATURE_COLS_14 + ["lap_time_delta", "Track"]).reset_index(drop=True)
    rainfall = pd.to_numeric(df.get("Rainfall", 0.0), errors="coerce").fillna(0.0)
    clean_dry = ((df["lap_time_delta"] < 3.5) & (rainfall == 0.0)).to_numpy()

    X = df[FEATURE_COLS_14]
    y = df["lap_time_delta"].to_numpy(dtype=float)
    groups = df["Track"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=5)
    oof_zs = np.full(len(df), np.nan, dtype=float)
    fold_r2s = []
    fold_maes = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        tr_c = tr_idx[clean_dry[tr_idx]]
        m = PhysicsDegradationEnsemble(seed=42 + fold, single_output=True)
        m.fit(X.iloc[tr_c], y[tr_c])
        pred_val = np.asarray(m.predict(X.iloc[val_idx])).reshape(-1)
        oof_zs[val_idx] = pred_val
        va_c = clean_dry[val_idx]
        fold_r2s.append(float(r2_score(y[val_idx][va_c], pred_val[va_c])))
        fold_maes.append(float(mean_absolute_error(y[val_idx][va_c], pred_val[va_c])))

    # Causal Race-EB OOF
    eb_tmp = pd.DataFrame({
        "Season": df["Season"],
        "Track": df["Track"],
        "LapNumber": df["LapNumber"],
        "base": oof_zs,
        "y": y,
        "obs": clean_dry & (df["tyre_age"].to_numpy(dtype=float) >= 3.0),
    })
    eb_tmp["num"] = np.where(eb_tmp["obs"], eb_tmp["base"] * eb_tmp["y"], 0.0)
    eb_tmp["den"] = np.where(eb_tmp["obs"], eb_tmp["base"] ** 2, 0.0)
    eb_tmp["res"] = np.where(eb_tmp["obs"], eb_tmp["y"] - eb_tmp["base"], 0.0)
    eb_tmp["cnt"] = eb_tmp["obs"].astype(float)
    la = eb_tmp.groupby(["Season", "Track", "LapNumber"])[["num", "den", "res", "cnt"]].sum().reset_index().sort_values(["Season", "Track", "LapNumber"])
    for c in ["num", "den", "res", "cnt"]:
        la[f"c_{c}"] = la.groupby(["Season", "Track"])[c].cumsum() - la[c]
    la["alpha"] = ((la["c_num"] + 12.0) / (la["c_den"] + 12.0)).clip(0.15, 2.8)
    la["bias"] = la["c_res"] / (la["c_cnt"] + 25.0)
    eb_m = eb_tmp.merge(la[["Season", "Track", "LapNumber", "alpha", "bias"]], on=["Season", "Track", "LapNumber"], how="left")
    oof_eb = apply_race_eb(eb_m["base"].to_numpy(), eb_m["alpha"].fillna(1.0).to_numpy(), eb_m["bias"].fillna(0.0).to_numpy())

    df_eval = df.copy()
    df_eval["pred_delta"] = oof_zs
    df_eval["pred_eb"] = oof_eb
    df_clean = df_eval.loc[clean_dry].copy()

    pairs5 = _compute_increment_pairs(df_clean, step=5)
    pairs10 = _compute_increment_pairs(df_clean, step=10)

    loro_bhr_metrics = {}
    loro_path = os.path.join("models", "loro_bahrain.joblib")
    if os.path.exists(loro_path):
        loro_model = joblib.load(loro_path)
        bhr_mask = (df["Track"] == "Bahrain Grand Prix").to_numpy()
        bhr_df = df_eval.loc[bhr_mask].copy()
        bhr_pred = np.asarray(loro_model.predict(bhr_df[FEATURE_COLS_14], validate_features=False)).reshape(-1)
        bhr_c = clean_dry[bhr_mask]
        bhr_eb = oof_eb[bhr_mask]  # note Bahrain OOF fold also held out Bahrain
        bhr_pairs5 = pairs5[pairs5["Track"] == "Bahrain Grand Prix"]
        loro_bhr_metrics = {
            "bhr_rows": len(bhr_df),
            "bhr_clean_rows": int(bhr_c.sum()),
            "bhr_zs_all_r2": float(r2_score(bhr_df["lap_time_delta"], bhr_pred)),
            "bhr_zs_clean_r2": float(r2_score(bhr_df.loc[bhr_c, "lap_time_delta"], bhr_pred[bhr_c])),
            "bhr_eb_clean_r2": float(r2_score(bhr_df.loc[bhr_c, "lap_time_delta"], bhr_eb[bhr_c])),
            "bhr_eb_clean_mae": float(mean_absolute_error(bhr_df.loc[bhr_c, "lap_time_delta"], bhr_eb[bhr_c])),
            "bhr_inc_pairs": len(bhr_pairs5),
            "bhr_xgb_inc5_r2": float(r2_score(bhr_pairs5["actual_inc"], bhr_pairs5["xgb_inc"])),
            "bhr_lin_zs_inc5_r2": float(r2_score(bhr_pairs5["actual_inc"], bhr_pairs5["lin_zs_inc"])),
            "bhr_lin_lodo_inc5_r2": float(r2_score(bhr_pairs5["actual_inc"], bhr_pairs5["lin_lodo_inc"])),
            "bhr_lin_lodo_inc5_mae": float(mean_absolute_error(bhr_pairs5["actual_inc"], bhr_pairs5["lin_lodo_inc"])),
        }

    return {
        "n_rows": len(df),
        "n_clean_rows": int(clean_dry.sum()),
        "n_tracks": int(df["Track"].nunique()),
        "oof_zs_all_r2": float(r2_score(y, oof_zs)),
        "oof_zs_all_mae": float(mean_absolute_error(y, oof_zs)),
        "oof_zs_clean_r2": float(r2_score(y[clean_dry], oof_zs[clean_dry])),
        "oof_zs_clean_mae": float(mean_absolute_error(y[clean_dry], oof_zs[clean_dry])),
        "oof_zs_clean_rmse": float(root_mean_squared_error(y[clean_dry], oof_zs[clean_dry])),
        "oof_eb_all_r2": float(r2_score(y, oof_eb)),
        "oof_eb_clean_r2": float(r2_score(y[clean_dry], oof_eb[clean_dry])),
        "oof_eb_clean_mae": float(mean_absolute_error(y[clean_dry], oof_eb[clean_dry])),
        "fold_clean_r2_mean": float(np.mean(fold_r2s)),
        "fold_clean_r2_std": float(np.std(fold_r2s)),
        "n_inc5_pairs": len(pairs5),
        "inc5_xgb_eb_r2": float(r2_score(pairs5["actual_inc"], pairs5["xgb_eb_inc"])),
        "inc5_lin_zs_r2": float(r2_score(pairs5["actual_inc"], pairs5["lin_zs_inc"])),
        "inc5_lin_lodo_r2": float(r2_score(pairs5["actual_inc"], pairs5["lin_lodo_inc"])),
        "inc5_lin_lodo_mae": float(mean_absolute_error(pairs5["actual_inc"], pairs5["lin_lodo_inc"])),
        "n_inc10_pairs": len(pairs10),
        "inc10_xgb_eb_r2": float(r2_score(pairs10["actual_inc"], pairs10["xgb_eb_inc"])),
        "inc10_lin_zs_r2": float(r2_score(pairs10["actual_inc"], pairs10["lin_zs_inc"])),
        "inc10_lin_lodo_r2": float(r2_score(pairs10["actual_inc"], pairs10["lin_lodo_inc"])),
        "loro_bahrain": loro_bhr_metrics,
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def evaluate_backtester_f1() -> tuple[dict, dict, pd.DataFrame]:
    """Run all 15 historical undercut events and compute F1 with & without Monaco rule."""
    model_meta = joblib.load(os.path.join("models", "global_degradation_model.joblib"))
    rows = [evaluate_single_undercut(ev, model_meta, verbose=False) for ev in HISTORICAL_UNDERCUTS]
    df = pd.DataFrame(rows)
    valid = df[df["load_error"] == ""].copy()
    y_true = valid["undercut_worked"].astype(int).to_numpy()
    y_pred_rule = (valid["model_recommendation"] == "PIT").astype(int).to_numpy()
    y_pred_pure = ((valid["model_confidence"] >= 0.50) & (valid["expected_gain_s"] > 0.0)).astype(int).to_numpy()
    return _classification_metrics(y_true, y_pred_rule), _classification_metrics(y_true, y_pred_pure), valid


def evaluate_mined_2024_duels_f1() -> dict:
    """Strictly out-of-sample Live MC F1 evaluation on 2024 mined green-flag undercut duels."""
    if not os.path.exists(MINED_DUELS_PATH):
        return {"n": 0}
    duels = pd.read_csv(MINED_DUELS_PATH)
    d24 = duels[duels["season"] == 2024].copy()
    if d24.empty:
        return {"n": 0}

    meta = joblib.load(os.path.join("models", "global_degradation_model.joblib"))
    model = meta["model"]
    features = meta.get("features", FEATURE_COLS_14)

    rows = []
    for gp, grp in d24.groupby("race"):
        try:
            s = fastf1.get_session(2024, gp, "R")
            s.load(laps=True, telemetry=False, weather=True, messages=False)
            ml = prepare_ml_data(apply_feature_engineering(merge_weather_data(clean_lap_data(s.laps), s.weather_data), gp))
            pit_loss = calculate_pit_loss(s)
        except Exception:
            continue
        tenv = ml[["LapNumber", "track_evolution", "TrackTemp"]].drop_duplicates("LapNumber")
        total_laps = int(ml["LapNumber"].max())
        eb_cache = {}
        for _, r in grp.iterrows():
            att, def_drv, l_att, gap_s, worked = r["att"], r["def"], int(r["l_att"]), float(r["gap_s"]), int(r["worked"])
            ev = l_att - 1
            past_ev = ml[ml["LapNumber"] <= ev]
            dl = past_ev[past_ev["Driver"] == att].sort_values("LapNumber")
            if dl.empty:
                continue
            last = dl.iloc[-1]
            st = {
                "tyre_age": float(last["tyre_age"]),
                "compound_encoded": int(last["compound_encoded"]),
                "baseline_pace": float(last["baseline_pace"]),
                "Track": gp,
                "Driver": att,
            }
            def_st = None
            def_l = past_ev[past_ev["Driver"] == def_drv].sort_values("LapNumber")
            if not def_l.empty:
                dl_last = def_l.iloc[-1]
                def_st = {
                    "tyre_age": float(dl_last["tyre_age"]),
                    "compound_encoded": int(dl_last["compound_encoded"]),
                    "baseline_pace": float(dl_last["baseline_pace"]),
                    "Track": gp,
                    "Driver": def_drv,
                }
            if ev not in eb_cache:
                eb_cache[ev] = estimate_causal_race_eb(past_ev, model, features)
            alpha_eb, bias_eb = eb_cache[ev]
            mc = run_mc_simulation(
                model=model,
                current_lap=ev,
                current_state=st,
                track_env=tenv,
                pit_loss_delta=pit_loss,
                race_name=gp,
                n_rollouts=250,
                horizon=15,
                feature_cols=features,
                rival_gap_s=gap_s,
                total_laps=total_laps,
                race_alpha=alpha_eb,
                race_bias=bias_eb,
                defender_state=def_st,
            )
            pit_opts = [x for x in mc if x["pit_offset"] is not None]
            if not pit_opts:
                continue
            best = max(pit_opts, key=lambda x: x["expected_gain_s"])
            pred = int((best["p_net_gain"] >= 0.50) and (best["expected_gain_s"] > 0.0) and ("Monaco" not in gp))
            rows.append({"race": gp, "gap_s": gap_s, "worked": worked, "pred": pred})

    if not rows:
        return {"n": 0}
    res_df = pd.DataFrame(rows)
    return {
        "n": len(res_df),
        "races": int(res_df["race"].nunique()),
        "metrics": _classification_metrics(res_df["worked"].to_numpy(), res_df["pred"].to_numpy()),
    }


def evaluate_whatif_undercut_f1() -> dict:
    """
    Evaluate Race Review counterfactual engine (replay/whatif.py) on mined green-flag
    undercut duels via attacker delay margin vs initial gap.
    """
    if not os.path.exists(MINED_DUELS_PATH):
        return {"n": 0}
    duels = pd.read_csv(MINED_DUELS_PATH)
    sample_races = [
        (2023, "Bahrain Grand Prix"), (2023, "Spanish Grand Prix"), (2023, "Hungarian Grand Prix"),
        (2023, "British Grand Prix"), (2023, "Italian Grand Prix"), (2023, "Singapore Grand Prix"),
        (2024, "Bahrain Grand Prix"), (2024, "Japanese Grand Prix"), (2024, "Spanish Grand Prix"),
        (2024, "Austrian Grand Prix"), (2024, "Hungarian Grand Prix"), (2024, "Dutch Grand Prix"),
    ]
    sub = duels.merge(pd.DataFrame(sample_races, columns=["season", "race"]), on=["season", "race"])
    meta = joblib.load(os.path.join("models", "global_degradation_model.joblib"))
    loro_path = os.path.join("models", "loro_bahrain.joblib")
    loro_model = joblib.load(loro_path) if os.path.exists(loro_path) else meta["model"]

    records = []
    for (yr, gp), grp in sub.groupby(["season", "race"]):
        try:
            rd = prepare_race_data(int(yr), gp, include_session=False)
        except Exception:
            continue
        rt = rd["replay_table"]
        g3_m = loro_model if "Bahrain" in gp else meta["model"]
        for _, r in grp.head(10).iterrows():
            att, dfd = r["att"], r["def"]
            l_att, l_def, gap_s, worked = int(r["l_att"]), int(r["l_def"]), float(r["gap_s"]), int(r["worked"])
            att_stops = rt[(rt["Driver"] == att) & rt["is_in_lap"]].sort_values("LapNumber")
            if l_att not in set(att_stops["LapNumber"]):
                continue
            att_sidx = list(att_stops["LapNumber"]).index(l_att)  # 0-indexed stop_idx
            shift_att = (l_def - l_att) + 1
            kw = dict(
                g3_model=g3_m, g3_choice="linear", g3_linear_slopes=rd["g3_linear_slopes"],
                g3_residuals=rd["g3_residuals"], pit_loss_green=rd["pit_loss_green"],
                pit_loss_neutralized=rd["pit_loss_neutralized"], neutralized_laps=rd["neutralized_laps"],
                n_paths=40, seed=42, circuit_physics=rd["circuit_physics"],
                overtaking_difficulty=rd.get("overtaking_difficulty", "Medium"),
            )
            ra_cov = run_whatif(rt, att, att_sidx, shift_att, rival_mode="cover_traffic", **kw)
            ra_fix = run_whatif(rt, att, att_sidx, shift_att, rival_mode="fixed", **kw)
            if not (ra_cov.get("is_feasible") and ra_fix.get("is_feasible")):
                continue
            al = rt[rt["Driver"] == att].sort_values("LapNumber").reset_index(drop=True)
            chk = min(l_def + 2, int(al["LapNumber"].max()))
            ai = al[al["LapNumber"] == chk]
            if ai.empty:
                continue
            c_idx = int(ai.index[0])
            adv_cov = float(ra_cov["delta_per_lap"][c_idx])
            adv_fix = float(ra_fix["delta_per_lap"][c_idx])
            records.append({
                "season": yr, "race": gp, "att": att, "def": dfd, "gap_s": gap_s, "worked": worked,
                "margin_cov": adv_cov - gap_s, "margin_fix": adv_fix - gap_s,
                "blocked": float(ra_cov.get("blocked_laps", 0.0)),
            })

    wi_df = pd.DataFrame(records)
    if wi_df.empty:
        return {"n": 0}
    y_true = wi_df["worked"].to_numpy()
    return {
        "n": len(wi_df),
        "cover_00": _classification_metrics(y_true, (wi_df["margin_cov"] > 0.0).astype(int).to_numpy()),
        "cover_05": _classification_metrics(y_true, (wi_df["margin_cov"] > 0.5).astype(int).to_numpy()),
        "fixed_00": _classification_metrics(y_true, (wi_df["margin_fix"] > 0.0).astype(int).to_numpy()),
    }


def main():
    print("=" * 74)
    print("  1. OUT-OF-SAMPLE DEGRADATION REGRESSION BENCHMARK (Lap>=2 R² / MAE)")
    print("=" * 74)
    deg = evaluate_degradation_r2()
    print(f"Dataset: {deg['n_rows']:,} Lap>=2 rows ({deg['n_clean_rows']:,} clean dry) across {deg['n_tracks']} tracks (5-fold GroupKFold)")
    print(f"  Zero-Shot OOF (All Lap>=2)  : R² = {deg['oof_zs_all_r2']:+.4f}  | MAE = {deg['oof_zs_all_mae']:.4f}s")
    print(f"  Zero-Shot OOF (Clean Dry)   : R² = {deg['oof_zs_clean_r2']:+.4f}  | MAE = {deg['oof_zs_clean_mae']:.4f}s | RMSE = {deg['oof_zs_clean_rmse']:.4f}s")
    print(f"  Causal Race-EB (All Lap>=2) : R² = {deg['oof_eb_all_r2']:+.4f}")
    print(f"  Causal Race-EB (Clean Dry)  : R² = {deg['oof_eb_clean_r2']:+.4f}  | MAE = {deg['oof_eb_clean_mae']:.4f}s")
    print(f"  Clean Fold R² mean ± std    : {deg['fold_clean_r2_mean']:+.4f} ± {deg['fold_clean_r2_std']:.4f}")
    print(f"Within-stint 5-lap clean increments (n={deg['n_inc5_pairs']:,} pairs):")
    print(f"  PhysicsEnsemble Race-EB     : R² = {deg['inc5_xgb_eb_r2']:+.4f}")
    print(f"  Smooth Linear Zero-Shot     : R² = {deg['inc5_lin_zs_r2']:+.4f}")
    print(f"  Smooth Linear + LODO RaceEB : R² = {deg['inc5_lin_lodo_r2']:+.4f}  | MAE = {deg['inc5_lin_lodo_mae']:.4f}s")
    print(f"Within-stint 10-lap clean increments (n={deg['n_inc10_pairs']:,} pairs):")
    print(f"  PhysicsEnsemble Race-EB     : R² = {deg['inc10_xgb_eb_r2']:+.4f}")
    print(f"  Smooth Linear + LODO RaceEB : R² = {deg['inc10_lin_lodo_r2']:+.4f}")
    if deg["loro_bahrain"]:
        b = deg["loro_bahrain"]
        print(f"Held-out LORO-Bahrain (n={b['bhr_rows']} Lap>=2 rows, {b['bhr_inc_pairs']} 5-lap pairs):")
        print(f"  LORO-Bahrain Zero-Shot Clean: R² = {b['bhr_zs_clean_r2']:+.4f} (All Lap>=2: {b['bhr_zs_all_r2']:+.4f})")
        print(f"  LORO-Bahrain Causal Race-EB : R² = {b['bhr_eb_clean_r2']:+.4f}  | MAE = {b['bhr_eb_clean_mae']:.4f}s")
        print(f"  Bahrain 5-lap Smooth LODO   : R² = {b['bhr_lin_lodo_inc5_r2']:+.4f}  | MAE = {b['bhr_lin_lodo_inc5_mae']:.4f}s (XGB inc5 R² = {b['bhr_xgb_inc5_r2']:+.4f})")

    print("\n" + "=" * 74)
    print("  2. HISTORICAL UNDERCUT STRATEGY BACKTEST (15 CURATED EVENTS)")
    print("=" * 74)
    m_rule, m_pure, bt_df = evaluate_backtester_f1()
    print(f"Pure MC Model (P>=0.50 & E[Gain]>0):  F1 = {m_pure['f1']:.3f} | Prec = {m_pure['precision']:.1%} | Rec = {m_pure['recall']:.1%} | Acc = {m_pure['accuracy']:.1%} | TP={m_pure['tp']} FP={m_pure['fp']} TN={m_pure['tn']} FN={m_pure['fn']}")
    print(f"Backtester Rule (+ not is_monaco):    F1 = {m_rule['f1']:.3f} | Prec = {m_rule['precision']:.1%} | Rec = {m_rule['recall']:.1%} | Acc = {m_rule['accuracy']:.1%} | TP={m_rule['tp']} FP={m_rule['fp']} TN={m_rule['tn']} FN={m_rule['fn']}")
    cols = ["season", "race", "driver", "rival_gap_s", "undercut_worked", "best_strategy", "model_confidence", "expected_gain_s", "model_recommendation", "correct_call"]
    print(bt_df[cols].to_string(index=False))

    print("\n" + "=" * 74)
    print("  3. STRICTLY OUT-OF-SAMPLE 2024 MINED GREEN-FLAG DUELS (Live MC)")
    print("=" * 74)
    m24 = evaluate_mined_2024_duels_f1()
    if m24["n"] > 0:
        mm = m24["metrics"]
        print(f"Evaluated {m24['n']} uncurated 2024 green-flag duels across {m24['races']} races:")
        print(f"  Live MC (2023-calibrated physics): F1 = {mm['f1']:.3f} | Prec = {mm['precision']:.1%} | Rec = {mm['recall']:.1%} | Acc = {mm['accuracy']:.1%} | TP={mm['tp']} FP={mm['fp']} TN={mm['tn']} FN={mm['fn']}")

    print("\n" + "=" * 74)
    print("  4. RACE REVIEW COUNTERFACTUAL REPLAY (run_whatif PAIRWISE F1)")
    print("=" * 74)
    wi = evaluate_whatif_undercut_f1()
    if wi["n"] > 0:
        print(f"Evaluated {wi['n']} real undercut duels across 12 races:")
        for label, key in [("cover_traffic (margin > 0.0s)", "cover_00"), ("cover_traffic (margin > 0.5s)", "cover_05"), ("fixed (margin > 0.0s)", "fixed_00")]:
            m = wi[key]
            print(f"  {label:30s}: F1 = {m['f1']:.3f} | Prec = {m['precision']:.1%} | Rec = {m['recall']:.1%} | Acc = {m['accuracy']:.1%} | TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


if __name__ == "__main__":
    main()
