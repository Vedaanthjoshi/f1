"""
server.py — FastAPI Backend Bridge for F1 Strategy Engine
=========================================================
Exposes REST endpoints consumed by the modern F1_V React/Tailwind frontend,
connected directly to the Phase 2 Monte Carlo engine and Quantile XGBoost models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Phase 2 Core Engine & Pipeline modules
from ml import load_global_model
from pipeline import fetch_season_schedule, load_race_session, process_telemetry_pipeline
from engine import run_mc_simulation, COMPOUND_MAP
from ui.metadata import drivers, driver_profile, teams

app = FastAPI(title="F1 Strategy AI API", version="2.0.0")

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
MODEL_CACHE: tuple[Any, list[str] | None, Any] | None = None


def get_cached_model():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        MODEL_CACHE = load_global_model()
    return MODEL_CACHE


# ── Models ────────────────────────────────────────────────────────────────────
class LoadRaceRequest(BaseModel):
    year: int
    race: str


class SimulateRequest(BaseModel):
    year: int
    race: str
    driver_a: str
    sim_lap: int
    next_compound: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/schedules")
def get_schedules():
    """Return available calendar schedules for Ground Effect era."""
    return {
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
        ]
    }


@app.get("/api/metadata")
def get_metadata():
    """Return driver metadata conforming to Driver TypeScript interface."""
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
    Load FastF1 session, run processing pipeline, and return RaceInfo format.
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

    # Build LapRecords
    # Required keys: d (driver), l (lap), t (lap time s), c (compound), a (tyre age), elapsed
    laps_records = []
    # Sort by Driver and LapNumber to calculate accurate elapsed cumulative seconds
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

    # Cache payload and race_info
    CACHE_SESSIONS[key] = {
        "backend_payload": backend_payload,
        "race_info": race_info,
        "session": session,
    }

    return race_info


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """
    Run Phase 2 Monte Carlo strategy optimizer and return SimResult.
    """
    key = (req.year, req.race)
    if key not in CACHE_SESSIONS:
        # Load race automatically if not already loaded
        load_race(LoadRaceRequest(year=req.year, race=req.race))

    session_data = CACHE_SESSIONS.get(key)
    if not session_data:
        raise HTTPException(status_code=400, detail="Race data could not be initialized")

    ml_data: pd.DataFrame = session_data["backend_payload"]["ml_data"]
    pit_loss: float = float(session_data["backend_payload"].get("pit_loss", 22.0))

    # Load Model
    model, features, _ = get_cached_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Global ML model not loaded")

    # Driver state on sim_lap
    driver_laps = ml_data[ml_data["Driver"] == req.driver_a]
    if driver_laps.empty:
        raise HTTPException(status_code=404, detail=f"Driver {req.driver_a} not found in this race")

    driver_state_df = driver_laps[driver_laps["LapNumber"] == req.sim_lap]
    if driver_state_df.empty:
        # Fallback to closest available lap
        closest_lap = driver_laps.iloc[(driver_laps["LapNumber"] - req.sim_lap).abs().argsort()[:1]]
        driver_state_df = closest_lap

    row = driver_state_df.iloc[0]
    current_state = {
        "tyre_age":         int(row.get("tyre_age", 15)),
        "compound_encoded": int(row.get("compound_encoded", 2)),
        "baseline_pace":    float(row.get("baseline_pace", 90.0)),
        "Track":            req.race,
        "Driver":           req.driver_a,
    }

    track_env = ml_data[["LapNumber", "track_evolution", "TrackTemp"]].drop_duplicates("LapNumber")

    target_compound_encoded = COMPOUND_MAP.get(req.next_compound.upper(), 3)
    horizon = 15

    total_laps = int(driver_laps["LapNumber"].max()) if not driver_laps.empty else None

    # Run Monte Carlo rollouts across strategies
    mc_results = run_mc_simulation(
        model=model,
        current_lap=req.sim_lap,
        current_state=current_state,
        track_env=track_env,
        pit_loss_delta=pit_loss,
        race_name=req.race,
        n_rollouts=500,
        horizon=horizon,
        feature_cols=features,
        total_laps=total_laps,
    )

    stay_out = mc_results[0]
    target_strat = next(
        (r for r in mc_results if r.get("pit_offset") == 0 and r.get("compound_encoded") == target_compound_encoded),
        mc_results[1] if len(mc_results) > 1 else mc_results[0],
    )

    so_times = stay_out.get("p50_lap_times", [])
    pit_times = target_strat.get("p50_lap_times", [])

    # Apply pit loss to lap 0 of the pit strategy
    pit_times_with_loss = list(pit_times)
    if pit_times_with_loss:
        pit_times_with_loss[0] += pit_loss

    # Net undercut advantage cumsum (positive means ahead)
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
    }


# Mount built React frontend static files if present
static_dist = Path(__file__).resolve().parent / "F1_V" / "dist"
if static_dist.exists():
    app.mount("/", StaticFiles(directory=str(static_dist), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("[API] Starting F1 Strategy Engine FastAPI Server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

