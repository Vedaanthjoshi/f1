"""
server.py — Comprehensive FastAPI Backend Bridge for F1 Strategy & Review Engine
================================================================================
Exposes complete REST endpoints consumed by the modern F1_V React/Tailwind frontend,
with full 100% parity to the Streamlit backend:
  1. Live Strategy Simulation with 6 candidate strategies, Two-Car Game-Theoretic Rival
     Cover Reaction, Causal Empirical Bayes scaling, and SHAP Explainability.
  2. Post-Race Review & What-If counterfactual simulation, Pit Decision Regret Scan,
     Stint Degradation Residuals, and LORO validation metadata.
  3. 3D FastF1 Circuit Coordinates for Three.js visualization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

logger = logging.getLogger("f1_server")
logging.basicConfig(level=logging.INFO)

# Phase 2 Core Engine & Pipeline modules
from ml import load_global_model, compute_shap_explanation
from ml.trainer import estimate_causal_race_eb
from pipeline import fetch_season_schedule, load_race_session, process_telemetry_pipeline
from engine import run_mc_simulation, COMPOUND_MAP
from ui.metadata import drivers, driver_profile, teams
from replay.whatif import (
    prepare_race_data,
    run_whatif,
    regret_scan,
    predict_stint_degradation_curve,
    SHIFT_MIN,
    SHIFT_MAX,
)

app = FastAPI(title="F1 Strategy AI API", version="2.5.0")

# Enable CORS for local Vite dev server and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches
CACHE_SESSIONS: dict[tuple[int, str], dict[str, Any]] = {}
CACHE_RACE_REVIEWS: dict[tuple[int, str], dict[str, Any]] = {}
CACHE_INIT_RESPONSES: dict[tuple[int, str], dict[str, Any]] = {}
CACHE_REGRET_SCANS: dict[tuple[int, str, str, int, str], dict[str, Any]] = {}
CACHE_WHATIF_RESPONSES: dict[tuple[int, str, str, int, int, str], dict[str, Any]] = {}
CACHE_STINT_DEG_RESPONSES: dict[tuple[int, str, str], dict[str, Any]] = {}
MODEL_CACHE: tuple[Any, list[str] | None, Any] | None = None

TRAINING_SET_RACES = {
    "Abu Dhabi Grand Prix", "Australian Grand Prix", "Austrian Grand Prix",
    "Azerbaijan Grand Prix", "Bahrain Grand Prix", "Belgian Grand Prix",
    "British Grand Prix", "Canadian Grand Prix", "Chinese Grand Prix",
    "Dutch Grand Prix", "Emilia Romagna Grand Prix", "French Grand Prix",
    "Hungarian Grand Prix", "Italian Grand Prix", "Japanese Grand Prix",
    "Las Vegas Grand Prix", "Mexico City Grand Prix", "Miami Grand Prix",
    "Monaco Grand Prix", "Qatar Grand Prix", "Saudi Arabian Grand Prix",
    "Singapore Grand Prix", "Spanish Grand Prix", "São Paulo Grand Prix",
    "United States Grand Prix"
}

FALLBACK_SCHEDULES: dict[str, list[str]] = {
    "2025": [
        "Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix",
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Miami Grand Prix",
        "Monaco Grand Prix", "Spanish Grand Prix", "Canadian Grand Prix",
        "Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix",
        "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
        "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix",
        "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix",
        "Qatar Grand Prix", "Abu Dhabi Grand Prix"
    ],
    "2024": [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
        "Monaco Grand Prix", "Canadian Grand Prix", "Spanish Grand Prix",
        "Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix",
        "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
        "Azerbaijan Grand Prix", "Singapore Grand Prix", "United States Grand Prix",
        "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix",
        "Qatar Grand Prix", "Abu Dhabi Grand Prix"
    ],
    "2023": [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Azerbaijan Grand Prix", "Miami Grand Prix", "Monaco Grand Prix",
        "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix",
        "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix",
        "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix",
        "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix",
        "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix",
        "Abu Dhabi Grand Prix"
    ],
    "2022": [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Emilia Romagna Grand Prix", "Miami Grand Prix", "Spanish Grand Prix",
        "Monaco Grand Prix", "Azerbaijan Grand Prix", "Canadian Grand Prix",
        "British Grand Prix", "Austrian Grand Prix", "French Grand Prix",
        "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
        "Italian Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix",
        "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
        "Abu Dhabi Grand Prix"
    ]
}


def get_cached_model():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        MODEL_CACHE = load_global_model()
    return MODEL_CACHE


# ── Request Models ────────────────────────────────────────────────────────────

class LoadRaceRequest(BaseModel):
    year: int
    race: str


class SimulateRequest(BaseModel):
    year: int
    race: str
    driver_a: str
    driver_b: str | None = None
    sim_lap: int
    next_compound: str
    n_rollouts: int = 500


class RegretScanRequest(BaseModel):
    year: int
    race: str
    driver: str
    stop_idx: int = 0
    rival_mode: str = "cover_traffic"
    n_paths: int = 500


class WhatIfRequest(BaseModel):
    year: int
    race: str
    driver: str
    stop_idx: int = 0
    shift_laps: int
    rival_mode: str = "cover_traffic"
    n_paths: int = 500


# ── Live Strategy Endpoints ───────────────────────────────────────────────────

@app.get("/api/schedules")
def get_schedules():
    """Return calendar schedules for all recent Ground Effect seasons."""
    schedules: dict[str, list[str]] = {}
    for y in [2025, 2024, 2023, 2022]:
        try:
            races = fetch_season_schedule(y)
            if races:
                schedules[str(y)] = races
            else:
                schedules[str(y)] = FALLBACK_SCHEDULES.get(str(y), [])
        except Exception:
            schedules[str(y)] = FALLBACK_SCHEDULES.get(str(y), [])
    return schedules


@app.get("/api/metadata")
def get_metadata():
    """Return complete grid driver metadata conforming to Driver TypeScript interface."""
    drivers_list = []
    for d in drivers():
        code = d.get("code", "")
        prof = driver_profile(code)
        driver_info = prof.get("driver", {})
        team_info = prof.get("team", {})
        car_info = prof.get("car", {})

        drivers_list.append({
            "code": code,
            "name": driver_info.get("name", code),
            "number": driver_info.get("number") or d.get("number") or 0,
            "team": team_info.get("name", d.get("team", "Unknown Team")),
            "car": car_info.get("name", "F1 Car"),
            "country": driver_info.get("country", d.get("country", "Global")),
            "accent": team_info.get("accentColor", "#E10600"),
        })

    return {"drivers": drivers_list}


@app.post("/api/load-race")
def load_race(req: LoadRaceRequest):
    """
    Load FastF1 session, run cleaning & physics feature pipeline, and cache payload.
    """
    key = (req.year, req.race)
    if key in CACHE_SESSIONS:
        return CACHE_SESSIONS[key]["race_info"]

    session = load_race_session(req.year, req.race)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Failed to load race session for {req.year} {req.race}")

    backend_payload, error_msg = process_telemetry_pipeline(session)
    if error_msg or not backend_payload:
        raise HTTPException(status_code=400, detail=error_msg or "Failed to process telemetry pipeline")

    ml_data: pd.DataFrame = backend_payload["ml_data"]
    pit_loss: float = float(backend_payload.get("pit_loss", 22.0))
    all_drivers = [str(x) for x in ml_data["Driver"].unique().tolist()]

    laps_records = []
    ml_sorted = ml_data.sort_values(by=["Driver", "LapNumber"])
    for driver_code, group in ml_sorted.groupby("Driver"):
        elapsed_sec = 0.0
        for _, row in group.iterrows():
            lap_time = float(row.get("LapTime_s", 90.0))
            elapsed_sec += lap_time
            laps_records.append({
                "d": str(driver_code),
                "l": int(row["LapNumber"]),
                "t": round(lap_time, 3),
                "c": str(row.get("Compound", "MEDIUM")).upper(),
                "a": int(row.get("tyre_age", 1)),
                "elapsed": round(elapsed_sec, 3),
            })

    min_lap = int(ml_data["LapNumber"].min()) if not ml_data.empty else 1
    max_lap = int(ml_data["LapNumber"].max()) if not ml_data.empty else 57

    circuit_name = getattr(session.event, "Location", req.race)
    country_name = getattr(session.event, "Country", "F1 Circuit")

    race_info = {
        "year": req.year,
        "race": req.race,
        "circuit": circuit_name,
        "country": country_name,
        "laps_count": max_lap,
        "min_lap": min_lap,
        "max_lap": max_lap,
        "distance_km": round(max_lap * 5.3, 1),
        "pit_loss": round(pit_loss, 2),
        "drivers": all_drivers,
        "laps": laps_records,
    }

    CACHE_SESSIONS[key] = {
        "backend_payload": backend_payload,
        "race_info": race_info,
        "session": session,
    }

    return race_info


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """
    Run Phase 2 Probabilistic Monte Carlo Strategy Optimizer:
    - 6 candidate tactical strategies
    - Two-Car Game-Theoretic Competitor Cover Model (Defender Reaction)
    - Causal Empirical Bayes degradation scaling from past laps
    - SHAP Explainability feature attributions
    """
    key = (req.year, req.race)
    if key not in CACHE_SESSIONS:
        load_race(LoadRaceRequest(year=req.year, race=req.race))

    session_data = CACHE_SESSIONS.get(key)
    if not session_data:
        raise HTTPException(status_code=400, detail="Race data could not be initialized")

    ml_data: pd.DataFrame = session_data["backend_payload"]["ml_data"]
    pit_loss: float = float(session_data["backend_payload"].get("pit_loss", 22.0))

    # Load Model
    model, features, _ = get_cached_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Global ML model not loaded. Run train_global_model.py first.")

    # Driver A state on sim_lap
    driver_laps = ml_data[ml_data["Driver"] == req.driver_a]
    if driver_laps.empty:
        raise HTTPException(status_code=404, detail=f"Driver {req.driver_a} not found in this race")

    driver_state_df = driver_laps[driver_laps["LapNumber"] == req.sim_lap]
    if driver_state_df.empty:
        closest_idx = (driver_laps["LapNumber"] - req.sim_lap).abs().argsort()[:1]
        driver_state_df = driver_laps.iloc[closest_idx]

    row_a = driver_state_df.iloc[0]
    current_state = {
        "tyre_age":         float(row_a.get("tyre_age", 15)),
        "compound_encoded": int(row_a.get("compound_encoded", 2)),
        "baseline_pace":    float(row_a.get("baseline_pace", 90.0)),
        "Track":            req.race,
        "Driver":           req.driver_a,
    }

    track_env = ml_data[["LapNumber", "track_evolution", "TrackTemp"]].drop_duplicates("LapNumber")
    past_laps = ml_data[ml_data["LapNumber"] <= req.sim_lap]

    # Causal Empirical Bayes adjustment
    race_alpha, race_bias = estimate_causal_race_eb(past_laps, model, features)

    # Competitor state (Driver B) & rival gap
    defender_state = None
    rival_gap_s = None
    if req.driver_b and req.driver_b != req.driver_a:
        def_state_df = ml_data[(ml_data["Driver"] == req.driver_b) & (ml_data["LapNumber"] <= req.sim_lap)].sort_values("LapNumber")
        if not def_state_df.empty:
            d_last = def_state_df.iloc[-1]
            defender_state = {
                "tyre_age":         float(d_last.get("tyre_age", 15)),
                "compound_encoded": int(d_last.get("compound_encoded", 2)),
                "baseline_pace":    float(d_last.get("baseline_pace", 90.0)),
                "Track":            req.race,
                "Driver":           req.driver_b,
            }
            # Compute current gap in seconds if available on sim_lap
            laps_a_sim = ml_data[(ml_data["Driver"] == req.driver_a) & (ml_data["LapNumber"] == req.sim_lap)]
            laps_b_sim = ml_data[(ml_data["Driver"] == req.driver_b) & (ml_data["LapNumber"] == req.sim_lap)]
            if not laps_a_sim.empty and not laps_b_sim.empty:
                t_a = laps_a_sim.iloc[0].get("Time")
                t_b = laps_b_sim.iloc[0].get("Time")
                if pd.notna(t_a) and pd.notna(t_b):
                    try:
                        rival_gap_s = float((t_b - t_a).total_seconds())
                    except Exception:
                        pass

    target_compound_encoded = COMPOUND_MAP.get(req.next_compound.upper(), 3)
    horizon = 15
    total_laps = int(driver_laps["LapNumber"].max()) if not driver_laps.empty else None

    # Run Monte Carlo rollouts across 6 candidate strategies
    mc_results = run_mc_simulation(
        model=model,
        current_lap=req.sim_lap,
        current_state=current_state,
        track_env=track_env,
        pit_loss_delta=pit_loss,
        race_name=req.race,
        n_rollouts=req.n_rollouts,
        horizon=horizon,
        feature_cols=features,
        rival_gap_s=rival_gap_s,
        total_laps=total_laps,
        race_alpha=race_alpha,
        race_bias=race_bias,
        defender_state=defender_state,
    )

    stay_out = mc_results[0]
    target_strat = next(
        (r for r in mc_results if r.get("pit_offset") == 0 and r.get("compound_encoded") == target_compound_encoded),
        mc_results[1] if len(mc_results) > 1 else mc_results[0],
    )

    so_times = stay_out.get("p50_lap_times", [])
    pit_times = target_strat.get("p50_lap_times", [])

    pit_times_with_loss = list(pit_times)
    if pit_times_with_loss:
        pit_times_with_loss[0] += pit_loss

    horizon_len = min(len(so_times), len(pit_times_with_loss))
    net_undercut_advantage = []
    cum_adv = 0.0
    for i in range(horizon_len):
        lap_delta = so_times[i] - pit_times_with_loss[i]
        cum_adv += lap_delta
        net_undercut_advantage.append(round(cum_adv, 3))

    crossover_lap = target_strat.get("crossover_lap")
    window_open = bool(target_strat.get("window_open", False))
    p_net_gain = float(target_strat.get("p_net_gain", 0.5))
    expected_gain = float(target_strat.get("expected_gain_s", 0.0))
    confidence = int(round(p_net_gain * 100))

    verdict = "open" if window_open else ("pending" if crossover_lap is not None else "closed")

    if verdict == "open":
        headline = f"Undercut window open (+{expected_gain:.1f}s net gain)"
        description = f"Box on lap {req.sim_lap} onto {req.next_compound}. Fresh tyres yield an immediate pace advantage overcoming the {pit_loss:.1f}s pit delta."
        recommendation = f"BOX {req.driver_a} · LAP {req.sim_lap} · {req.next_compound.upper()}"
    elif verdict == "pending":
        headline = f"Pace crossover pending around Lap {crossover_lap}"
        description = f"Current tyres degrading; expected pace advantage triggers within 2-3 laps. Monitor rival pit wall."
        recommendation = f"PREPARE BOX · MONITOR GAP · LAP {req.sim_lap}"
    else:
        headline = "Undercut window closed"
        description = f"Pitting now does not overcome the {pit_loss:.1f}s pit loss. Degradation rate remains sustainable."
        recommendation = f"STAY OUT · EXTEND STINT · {req.driver_a}"

    # Format all 6 tactical candidates
    all_strategies = []
    for s in mc_results:
        all_strategies.append({
            "name": s.get("name", "STRATEGY"),
            "compound": s.get("compound_name", "MEDIUM"),
            "compound_encoded": s.get("compound_encoded"),
            "pit_offset": s.get("pit_offset", 0),
            "stop_lap": req.sim_lap + (s.get("pit_offset") or 0),
            "p_net_gain": round(float(s.get("p_net_gain", 0.0)), 3),
            "expected_gain_s": round(float(s.get("expected_gain_s", 0.0)), 2),
            "p05_gain_s": round(float(s.get("p05_gain_s", 0.0)), 2),
            "p95_gain_s": round(float(s.get("p95_gain_s", 0.0)), 2),
            "crossover_lap": s.get("crossover_lap"),
            "window_open": bool(s.get("window_open", False)),
        })

    # Compute SHAP explainability
    shap_list: list[dict[str, Any]] = []
    try:
        shap_df = compute_shap_explanation(model, features, current_state, track_env, req.sim_lap)
        if isinstance(shap_df, pd.DataFrame) and not shap_df.empty:
            for _, srow in shap_df.iterrows():
                shap_list.append({
                    "feature": str(srow.get("Feature", "")),
                    "label": str(srow.get("Label", srow.get("Feature", ""))),
                    "shap_value": round(float(srow.get("SHAP Value", 0.0)), 4),
                    "value": round(float(srow.get("Value", 0.0)), 2) if pd.notna(srow.get("Value")) else 0.0,
                })
    except Exception as e:
        print(f"[SHAP] Error computing SHAP attribution: {e}")

    return {
        "sim_lap": req.sim_lap,
        "compound": req.next_compound,
        "horizon_laps": [req.sim_lap + i for i in range(horizon_len)],
        "net_undercut_advantage": net_undercut_advantage,
        "stay_out_lap_times": [round(t, 3) for t in so_times[:horizon_len]],
        "pit_now_lap_times": [round(t, 3) for t in pit_times_with_loss[:horizon_len]],
        "crossover_lap": crossover_lap,
        "window_open": window_open,
        "verdict": verdict,
        "confidence": confidence,
        "headline": headline,
        "description": description,
        "recommendation": recommendation,
        "all_strategies": all_strategies,
        "shap_values": shap_list,
        "race_alpha": round(float(race_alpha), 3),
        "race_bias": round(float(race_bias), 3),
        "rival_gap_s": round(float(rival_gap_s), 2) if rival_gap_s is not None else None,
        "rival_covered": bool(target_strat.get("rival_reaction_triggered", False)),
    }


# ── Circuit 3D Geometry ───────────────────────────────────────────────────────

@app.get("/api/circuit-geometry")
def get_circuit_geometry(year: int = 2023, race: str = "Bahrain Grand Prix", max_points: int = 160):
    """
    Return normalized 3D track coordinates [[x, z, y], ...] from FastF1 position telemetry
    for the Interactive 3D Circuit Profile Card.
    """
    key = (year, race)
    session = CACHE_SESSIONS[key]["session"] if key in CACHE_SESSIONS else load_race_session(year, race)
    if session is None:
        return {"year": year, "race": race, "source": "fallback", "points": []}

    try:
        fastest_lap = session.laps.pick_fastest()
        if fastest_lap is None:
            return {"year": year, "race": race, "source": "fallback", "points": []}

        pos = fastest_lap.get_pos_data()
        if pos is None or pos.empty:
            return {"year": year, "race": race, "source": "fallback", "points": []}

        coords = pos[["X", "Y", "Z"]].dropna()
        if len(coords) < 10:
            return {"year": year, "race": race, "source": "fallback", "points": []}

        step = max(1, len(coords) // max(24, min(max_points, 300)))
        sampled = coords.iloc[::step].to_numpy(dtype=float)

        center = sampled.mean(axis=0)
        centered = sampled - center
        xy_span = float(np.max(np.abs(centered[:, :2]))) or 1.0
        norm_x = centered[:, 0] / xy_span
        norm_z = -centered[:, 1] / xy_span  # Flip Y for Three.js ground plane (X, Z)
        norm_y = (centered[:, 2] / xy_span) * 1.4  # Subtle elevation exaggeration

        points = [
            [round(float(x), 4), round(float(y), 4), round(float(z), 4)]
            for x, y, z in zip(norm_x, norm_y, norm_z)
        ]
        return {"year": year, "race": race, "source": "fastf1", "points": points}
    except Exception:
        return {"year": year, "race": race, "source": "fallback", "points": []}


# ── Post-Race Review & What-If Endpoints ───────────────────────────────────────

def _get_race_review_data(year: int, race: str) -> dict[str, Any]:
    """Retrieve and memoize prepared race replay table, G3/G4 models, and stats with disk cache and graceful fallback."""
    key = (year, race)
    if key in CACHE_RACE_REVIEWS:
        return CACHE_RACE_REVIEWS[key]

    clean_race = race.lower().replace(" ", "_")
    disk_file = os.path.join("f1_cache", f"prepared_{year}_{clean_race}.joblib")
    if os.path.exists(disk_file):
        try:
            logger.info(f"Loading prepared race data from disk: {disk_file}")
            rd = joblib.load(disk_file)
            CACHE_RACE_REVIEWS[key] = rd
            return rd
        except Exception as e:
            logger.warning(f"Failed to load {disk_file}: {e}")

    bahrain_disk = os.path.join("f1_cache", "prepared_2023_bahrain.joblib")
    if ("bahrain" in race.lower() and year == 2023) or year >= 2025:
        if os.path.exists(bahrain_disk):
            try:
                base_rd = joblib.load(bahrain_disk)
                if year >= 2025:
                    fallback_rd = dict(base_rd)
                    fallback_rd["fallback_used"] = True
                    fallback_rd["fallback_reason"] = (
                        f"Telemetry unavailable for {year} {race} (future calendar season). "
                        f"Displaying 2023 Bahrain GP benchmark replay."
                    )
                    fallback_rd["requested_year"] = year
                    fallback_rd["requested_race"] = race
                    CACHE_RACE_REVIEWS[key] = fallback_rd
                    return fallback_rd
                CACHE_RACE_REVIEWS[key] = base_rd
                return base_rd
            except Exception as e:
                logger.warning(f"Failed to load {bahrain_disk}: {e}")

    if year >= 2025:
        logger.info(f"Year {year} has no completed telemetry. Loading 2023 Bahrain Grand Prix benchmark replay.")
        base_rd = _get_race_review_data(2023, "Bahrain Grand Prix")
        fallback_rd = dict(base_rd)
        fallback_rd["fallback_used"] = True
        fallback_rd["fallback_reason"] = (
            f"Telemetry unavailable for {year} {race} (future calendar season). "
            f"Displaying 2023 Bahrain GP benchmark replay."
        )
        fallback_rd["requested_year"] = year
        fallback_rd["requested_race"] = race
        CACHE_RACE_REVIEWS[key] = fallback_rd
        return fallback_rd

    try:
        rd = prepare_race_data(year, race, include_session=False)
        rd["fallback_used"] = False
        rd["requested_year"] = year
        rd["requested_race"] = race
        CACHE_RACE_REVIEWS[key] = rd
        try:
            joblib.dump(rd, disk_file)
        except Exception:
            pass
        return rd
    except Exception as e:
        logger.warning(f"Telemetry load failed for {year} {race}: {e}. Falling back to 2023 Bahrain Grand Prix benchmark.")
        try:
            # Load the benchmark 2023 Bahrain Grand Prix replay data
            base_rd = _get_race_review_data(2023, "Bahrain Grand Prix")
            fallback_rd = dict(base_rd)
            fallback_rd["fallback_used"] = True
            fallback_rd["fallback_reason"] = (
                f"Telemetry unavailable for {year} {race} (API rate limit or uncached session). "
                f"Showing 2023 Bahrain GP benchmark replay."
            )
            fallback_rd["requested_year"] = year
            fallback_rd["requested_race"] = race
            CACHE_RACE_REVIEWS[key] = fallback_rd
            return fallback_rd
        except Exception as inner_e:
            logger.error(f"Critical error loading fallback benchmark: {inner_e}")
            raise HTTPException(status_code=500, detail=f"Failed loading race telemetry and fallback benchmark: {e}")


@app.get("/api/race-review/init")
def race_review_init(year: int = Query(2023), race: str = Query("Bahrain Grand Prix")):
    """
    Return complete Post-Race Review metadata, grid finishing order, stint profiles,
    pit stops, circuit physics, and LORO validation status.
    Fast path returns precomputed JSON instantly (< 5ms).
    """
    key = (year, race)
    if key in CACHE_INIT_RESPONSES:
        return CACHE_INIT_RESPONSES[key]

    fast_json_path = os.path.join("f1_cache", "fast_review_2023_bahrain.json")
    if (("bahrain" in race.lower() and year == 2023) or year >= 2025) and os.path.exists(fast_json_path):
        try:
            with open(fast_json_path, "r", encoding="utf-8") as f:
                fast_data = json.load(f)
            
            init_payload = dict(fast_data["init"])
            init_payload["year"] = year
            init_payload["race"] = race
            init_payload["requested_year"] = year
            init_payload["requested_race"] = race
            if year >= 2025:
                init_payload["fallback_used"] = True
                init_payload["fallback_reason"] = (
                    f"Telemetry unavailable for {year} {race} (future calendar season). "
                    f"Displaying 2023 Bahrain GP benchmark replay."
                )
            else:
                init_payload["fallback_used"] = False
                init_payload["fallback_reason"] = ""

            init_payload["initial_driver"] = "VER"
            init_payload["initial_regret"] = fast_data.get("initial_regret")
            init_payload["initial_whatif"] = fast_data.get("initial_whatif")
            init_payload["initial_stint_deg"] = fast_data.get("initial_stint_deg")

            # Seed child caches so subsequent slider or driver interactions are instantaneous
            if fast_data.get("initial_regret"):
                for rm in ("fixed", "cover", "cover_traffic"):
                    CACHE_REGRET_SCANS[(year, race, "VER", 0, rm)] = fast_data["initial_regret"]
            if fast_data.get("initial_whatif"):
                for rm in ("fixed", "cover", "cover_traffic"):
                    CACHE_WHATIF_RESPONSES[(year, race, "VER", 0, 0, rm)] = fast_data["initial_whatif"]
            if fast_data.get("initial_stint_deg"):
                CACHE_STINT_DEG_RESPONSES[(year, race, "VER")] = fast_data["initial_stint_deg"]

            CACHE_INIT_RESPONSES[key] = init_payload
            return init_payload
        except Exception as e:
            logger.warning(f"Error loading {fast_json_path}: {e}")

    # Fallback to computing init from rd
    rd = _get_race_review_data(year, race)
    rt: pd.DataFrame = rd["replay_table"]
    actual_race_name = rd.get("requested_race", race)
    is_in_sample = race in TRAINING_SET_RACES

    total_laps = int(rt["LapNumber"].max()) if not rt.empty else 57

    drivers_data = []
    # Identify finishing positions from the highest lap number
    last_laps = rt.sort_values(["LapNumber", "Position"], ascending=[False, True]).drop_duplicates("Driver", keep="first")
    sorted_drivers = last_laps.sort_values("Position")["Driver"].tolist()

    winner_row = last_laps[last_laps["Position"] == 1]
    winner_time = float(winner_row.iloc[0]["Time_s"]) if not winner_row.empty else 0.0

    for drv_code in sorted_drivers:
        drv_laps = rt[rt["Driver"] == drv_code].sort_values("LapNumber")
        if drv_laps.empty:
            continue

        prof = driver_profile(drv_code)
        d_info = prof.get("driver", {})
        t_info = prof.get("team", {})

        raw_pos = drv_laps.iloc[-1].get("Position")
        pos_val = int(raw_pos) if pd.notna(raw_pos) else 20
        raw_time = drv_laps.iloc[-1].get("Time_s")
        final_time_s = float(raw_time) if pd.notna(raw_time) else 0.0
        gap_winner = round(final_time_s - winner_time, 2) if winner_time > 0 and final_time_s >= winner_time else 0.0

        pit_laps = drv_laps[drv_laps["is_in_lap"] == True]["LapNumber"].astype(int).tolist()

        # Build stint chips
        stints_list = []
        if "Stint" in drv_laps.columns:
            for stint_id, grp in drv_laps.groupby("Stint", sort=True):
                comp_series = grp["Compound"].dropna()
                comp = str(comp_series.iloc[0]).upper() if not comp_series.empty else "HARD"
                stints_list.append({
                    "stint": int(stint_id),
                    "compound": comp,
                    "laps_count": len(grp),
                    "start_lap": int(grp["LapNumber"].min()),
                    "end_lap": int(grp["LapNumber"].max()),
                })

        drivers_data.append({
            "code": drv_code,
            "name": d_info.get("name", drv_code),
            "number": d_info.get("number") or 0,
            "team": t_info.get("name", "F1 Team"),
            "accent": t_info.get("accentColor", "#E10600"),
            "final_pos": pos_val,
            "gap_to_winner": gap_winner,
            "pit_laps": pit_laps,
            "stints": stints_list,
        })

    pl_green = float(np.median(rd["pit_loss_green"])) if hasattr(rd.get("pit_loss_green"), "__len__") and len(rd["pit_loss_green"]) > 0 else 25.0
    pl_neut = float(np.median(rd["pit_loss_neutralized"])) if hasattr(rd.get("pit_loss_neutralized"), "__len__") and len(rd["pit_loss_neutralized"]) > 0 else 15.0

    init_res = {
        "year": year,
        "race": race,
        "fallback_used": bool(rd.get("fallback_used", False)),
        "fallback_reason": rd.get("fallback_reason", ""),
        "requested_year": rd.get("requested_year", year),
        "requested_race": rd.get("requested_race", race),
        "is_in_sample": is_in_sample,
        "total_laps": total_laps,
        "pit_loss_green": round(pl_green, 2),
        "pit_loss_neutralized": round(pl_neut, 2),
        "circuit_physics": rd.get("circuit_physics", {}),
        "drivers": drivers_data,
    }
    CACHE_INIT_RESPONSES[key] = init_res
    return init_res


@app.post("/api/race-review/regret-scan")
def race_review_regret_scan(req: RegretScanRequest):
    """
    Compute multi-stop pit decision regret scan (-8 to +8 laps) with Monte Carlo paths.
    Caches results so queries return in < 1ms.
    """
    cache_key = (req.year, req.race, req.driver, req.stop_idx, req.rival_mode)
    if cache_key in CACHE_REGRET_SCANS:
        return CACHE_REGRET_SCANS[cache_key]

    rd = _get_race_review_data(req.year, req.race)
    rt = rd["replay_table"]

    drv_laps = rt[rt["Driver"] == req.driver].sort_values("LapNumber")
    pit_laps = drv_laps[drv_laps["is_in_lap"] == True]["LapNumber"].astype(int).tolist()
    actual_stop_lap = pit_laps[req.stop_idx] if len(pit_laps) > req.stop_idx else 15

    scan_df = regret_scan(
        replay_table=rt,
        driver=req.driver,
        stop_idx=req.stop_idx,
        g3_model=rd["g3_model"],
        g3_choice=rd["g3_choice"],
        g3_linear_slopes=rd["g3_linear_slopes"],
        g3_residuals=rd["g3_residuals"],
        pit_loss_green=rd["pit_loss_green"],
        pit_loss_neutralized=rd["pit_loss_neutralized"],
        neutralized_laps=rd["neutralized_laps"],
        n_paths=min(req.n_paths, 35),
        rival_mode=req.rival_mode,
        circuit_physics=rd["circuit_physics"],
        overtaking_difficulty=rd["overtaking_difficulty"],
        shift_min=-8,
        shift_max=8,
    )

    if scan_df.empty:
        raise HTTPException(status_code=400, detail="No feasible regret scan shifts found for this stop")

    scan_records = []
    for _, row in scan_df.iterrows():
        shift_val = int(row["shift"])
        moved_lap_val = actual_stop_lap + shift_val
        exp_delta = float(row["expected_time_delta"])
        is_noise = bool(row.get("is_noise", False))
        status = "Noise / Neutral" if is_noise else ("Advantage" if exp_delta < 0 else "Loss")

        scan_records.append({
            "shift": shift_val,
            "moved_lap": moved_lap_val,
            "p05": round(float(row["p05"]), 3),
            "expected_time_delta": round(exp_delta, 3),
            "p95": round(float(row["p95"]), 3),
            "is_noise": is_noise,
            "status": status,
            "robust": bool(row.get("robustness", "ROBUST") == "ROBUST"),
            "extrapolated": bool(row.get("is_extrapolated", False)),
            "is_actual": shift_val == 0,
        })

    best_row = scan_df.loc[scan_df["expected_time_delta"].idxmin()]
    best_shift = int(best_row["shift"])
    best_lap = actual_stop_lap + best_shift
    best_delta = float(best_row["expected_time_delta"])

    actual_row = scan_df.loc[scan_df["shift"] == 0]
    actual_delta = float(actual_row.iloc[0]["expected_time_delta"]) if not actual_row.empty else 0.0
    max_regret_s = max(0.0, round(actual_delta - best_delta, 2))

    res_data = {
        "driver": req.driver,
        "stop_idx": req.stop_idx,
        "actual_pit_lap": actual_stop_lap,
        "best_pit_lap": best_lap,
        "best_shift": best_shift,
        "best_delta_s": round(best_delta, 2),
        "max_regret_s": max_regret_s,
        "scan": scan_records,
    }
    CACHE_REGRET_SCANS[cache_key] = res_data
    return res_data


@app.post("/api/race-review/whatif")
def race_review_whatif(req: WhatIfRequest):
    """
    Execute What-If counterfactual Monte Carlo simulation moving a pit stop by `shift_laps`.
    Caches results so slider scrubbing returns in < 1ms.
    """
    cache_key = (req.year, req.race, req.driver, req.stop_idx, req.shift_laps, req.rival_mode)
    if cache_key in CACHE_WHATIF_RESPONSES:
        return CACHE_WHATIF_RESPONSES[cache_key]

    rd = _get_race_review_data(req.year, req.race)
    rt = rd["replay_table"]
    drv_laps = rt[rt["Driver"] == req.driver].sort_values("LapNumber")
    pit_laps = drv_laps[drv_laps["is_in_lap"] == True]["LapNumber"].astype(int).tolist()
    actual_stop_lap = pit_laps[req.stop_idx] if len(pit_laps) > req.stop_idx else 15
    final_pos = int(drv_laps.iloc[-1].get("Position", 20)) if pd.notna(drv_laps.iloc[-1].get("Position")) else 20

    res = run_whatif(
        replay_table=rt,
        driver=req.driver,
        stop_idx=req.stop_idx,
        shift_laps=req.shift_laps,
        g3_model=rd["g3_model"],
        g3_choice=rd["g3_choice"],
        g3_linear_slopes=rd["g3_linear_slopes"],
        g3_residuals=rd["g3_residuals"],
        pit_loss_green=rd["pit_loss_green"],
        pit_loss_neutralized=rd["pit_loss_neutralized"],
        neutralized_laps=rd["neutralized_laps"],
        n_paths=min(req.n_paths, 35),
        rival_mode=req.rival_mode,
        circuit_physics=rd["circuit_physics"],
        overtaking_difficulty=rd["overtaking_difficulty"],
    )

    if not res.get("is_feasible", False):
        return {
            "is_feasible": False,
            "infeasible_reason": res.get("infeasible_reason", "Infeasible stop shift"),
        }

    cf_pos = int(res["position_per_lap"][-1]) if res.get("position_per_lap") is not None and len(res["position_per_lap"]) > 0 else final_pos
    delta_curve = [round(float(x), 3) for x in res.get("delta_per_lap", [])]
    laps_list = [int(x) for x in drv_laps["LapNumber"].values[:len(delta_curve)]]

    reactions = []
    for r in res.get("rival_reactions", []):
        reactions.append({
            "driver": str(r.get("driver")),
            "reacted_pit": int(r.get("reacted_pit", 0)),
            "actual_pit": int(r.get("actual_pit", 0)),
            "reason": str(r.get("reason", "")),
        })

    whatif_res = {
        "is_feasible": True,
        "driver": req.driver,
        "actual_pit_lap": actual_stop_lap,
        "moved_lap": actual_stop_lap + req.shift_laps,
        "shift_laps": req.shift_laps,
        "final_time_delta_p50": round(float(res.get("final_time_delta_p50", 0.0)), 2),
        "final_time_delta_p05": round(float(res.get("final_time_delta_p05", 0.0)), 2),
        "final_time_delta_p95": round(float(res.get("final_time_delta_p95", 0.0)), 2),
        "actual_pos": final_pos,
        "counterfactual_pos": cf_pos,
        "positions_gained": final_pos - cf_pos,
        "dirty_air_total_s": round(float(res.get("dirty_air_total_s", 0.0)), 2),
        "dirty_air_laps": round(float(res.get("dirty_air_laps", 0.0)), 1),
        "blocked_laps": round(float(res.get("blocked_laps", 0.0)), 1),
        "robustness": "ROBUST" if res.get("is_robust", True) else "NOT ROBUST",
        "extrapolated": bool(res.get("extrapolated", False)),
        "rival_reactions": reactions,
        "delta_per_lap": delta_curve,
        "laps": laps_list,
    }
    CACHE_WHATIF_RESPONSES[cache_key] = whatif_res
    return whatif_res


@app.get("/api/race-review/stint-degradation")
def race_review_stint_degradation(
    year: int = Query(2023),
    race: str = Query("Bahrain Grand Prix"),
    driver: str = Query("VER"),
):
    """
    Return actual observed lap-by-lap tyre degradation vs predicted curves per stint.
    Caches results so switching drivers returns in < 1ms.
    """
    cache_key = (year, race, driver)
    if cache_key in CACHE_STINT_DEG_RESPONSES:
        return CACHE_STINT_DEG_RESPONSES[cache_key]

    rd = _get_race_review_data(year, race)
    rt = rd["replay_table"]
    drv_laps = rt[rt["Driver"] == driver].sort_values("LapNumber")

    if drv_laps.empty or "Stint" not in drv_laps.columns:
        raise HTTPException(status_code=404, detail=f"No stint data found for driver {driver}")

    stints_output = []
    race_deg_stats = rd.get("race_deg_stats", {})

    for stint_id, grp in drv_laps.groupby("Stint", sort=True):
        clean_mask = (grp["is_in_lap"] == False) & (grp["is_out_lap"] == False) & (grp["is_neutralized"] == False)
        stint_clean = grp[clean_mask]
        if len(stint_clean) < 3:
            continue

        comp_series = grp["Compound"].dropna()
        comp = str(comp_series.iloc[0]).upper() if not comp_series.empty else "HARD"

        fc_times = stint_clean["LapTime_s"]
        anchor_age = int(stint_clean["TyreLife"].iloc[0])
        base_time = float(fc_times.min())
        deg_observed = (fc_times - base_time).values

        lin_pred, xgb_pred = predict_stint_degradation_curve(
            ages=stint_clean["TyreLife"].values,
            laps=stint_clean["LapNumber"].values,
            compound_str=comp,
            g3_model=rd["g3_model"],
            circuit_physics=rd["circuit_physics"],
            anchor_age=anchor_age,
            is_stint1=(int(stint_id) == 1),
            race_inc_bias=float(race_deg_stats.get("race_inc_bias", 0.0)),
            race_alpha=float(race_deg_stats.get("race_alpha", 1.0)),
        )

        stints_output.append({
            "stint": int(stint_id),
            "compound": comp,
            "ages": [int(a) for a in stint_clean["TyreLife"].values],
            "laps": [int(l) for l in stint_clean["LapNumber"].values],
            "deg_observed": [round(float(d), 3) for d in deg_observed],
            "deg_predicted_xgb": [round(float(d), 3) for d in xgb_pred],
            "deg_predicted_linear": [round(float(d), 3) for d in lin_pred],
        })

    stint_res = {
        "driver": driver,
        "stints": stints_output,
    }
    CACHE_STINT_DEG_RESPONSES[cache_key] = stint_res
    return stint_res


# Mount built React frontend static files if present
static_dist = Path(__file__).resolve().parent / "F1_V" / "dist"
if static_dist.exists():
    app.mount("/", StaticFiles(directory=str(static_dist), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("[API] Starting F1 Strategy Engine Comprehensive FastAPI Server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
