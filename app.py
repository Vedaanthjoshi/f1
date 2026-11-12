"""
app.py — F1 Strategy Engineer Web Application
=============================================
Main application entrypoint. Coordinates the telemetry data pipeline,
probabilistic XGBoost quantile model, Monte Carlo strategy simulation,
and modular presentation components.
"""

import streamlit as st
import pandas as pd

# Core domain modules
from pipeline import fetch_season_schedule, load_race_session, process_telemetry_pipeline
from engine import run_mc_simulation, COMPOUND_MAP
from ml import load_global_model, compute_shap_explanation
from ml.trainer import estimate_causal_race_eb

# UI components & styling
from ui.styles import inject_custom_css
from ui.metadata import driver_profile
from ui.state import init_session_state, load_requested_race
from ui.components import (
    render_homepage_hero,
    render_event_strip,
    render_matchup_preview,
    render_race_setup,
    render_strategy_table,
)
from ui.charts import (
    render_degradation_chart,
    render_gap_chart,
    render_advantage_chart,
    render_shap_chart,
)

# Application Configuration
st.set_page_config(page_title="F1 Strategy Engineer", layout="wide", page_icon=":racing_car:")


# --- CACHED DATA & MODEL LOADERS ---
@st.cache_data(show_spinner=False)
def get_schedule(year: int) -> list[str]:
    """Fetch calendar schedule for the chosen season."""
    return fetch_season_schedule(year)


@st.cache_resource(show_spinner=False)
def load_race_data(year: int, race_name: str):
    """Load and cache the raw FastF1 race session."""
    return load_race_session(year, race_name)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load and memoize the pre-trained Global XGBoost model."""
    return load_global_model()


@st.cache_data(show_spinner=False)
def process_pipeline(year: int, race: str, _session):
    """Execute cleaning, weather merging, feature engineering, and baseline preparation."""
    return process_telemetry_pipeline(_session)


# --- STRATEGY DASHBOARD VIEW ---
def render_strategy_dashboard():
    model, features, _ = get_model()
    if model is None:
        st.error("Global Model not found! Please run `python train_global_model.py` first.")
        return

    with st.spinner(f"Downloading telemetry for {st.session_state['race']}..."):
        session = load_race_data(st.session_state["year"], st.session_state["race"])
        backend_payload, error_msg = process_pipeline(st.session_state["year"], st.session_state["race"], session)

    if error_msg:
        st.error(error_msg)
        return

    st.success("Telemetry processed. Global AI Model loaded and ready.")

    ml_data = backend_payload["ml_data"]
    pit_loss = backend_payload["pit_loss"]
    drivers = ml_data["Driver"].unique().tolist()

    # Sidebar: Driver Matchup
    st.sidebar.divider()
    st.sidebar.header("Driver Comparison")
    driver_a = st.sidebar.selectbox("Driver A (Target)", drivers, index=0, key="driver_a")
    driver_b = st.sidebar.selectbox("Driver B (Attacker)", drivers, index=1 if len(drivers) > 1 else 0, key="driver_b")

    # Matchup Header & HUD Cards
    st.markdown("### Matchup Preview")
    render_event_strip(st.session_state["year"], st.session_state["race"], driver_a, driver_b)
    render_matchup_preview(driver_a, driver_b, st.session_state["race"])
    st.divider()

    df_ab = ml_data[ml_data["Driver"].isin([driver_a, driver_b])]
    if df_ab.empty:
        st.warning("No clean racing laps are available for the selected driver pairing. Try another driver or race.")
        return

    driver_color_map = {
        driver_a: driver_profile(driver_a)["team"].get("accentColor", "#E10600"),
        driver_b: driver_profile(driver_b)["team"].get("accentColor", "#27F4D2"),
    }
    if driver_color_map[driver_a] == driver_color_map[driver_b]:
        driver_color_map[driver_b] = "#FFFFFF"

    # Chart 1: Raw Tyre Degradation
    render_degradation_chart(df_ab, driver_a, driver_b, driver_color_map)

    # Chart 2: Gap Evolution
    gap_df = render_gap_chart(df_ab, driver_a, driver_b, driver_color_map)

    # Simulation Controls
    st.divider()
    st.subheader(f"🎲 Monte Carlo Strategy Optimizer — {driver_a}")
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 2])

    driver_laps = df_ab[df_ab["Driver"] == driver_a]["LapNumber"]
    min_lap = int(driver_laps.min())
    max_lap = int(driver_laps.max())
    default_val = min(min_lap + 15, max_lap)

    sim_lap = sim_col1.number_input("Simulate from Lap", min_value=min_lap, max_value=max_lap, value=default_val)
    n_rollouts = sim_col2.select_slider("MC Rollouts", options=[200, 500, 1000], value=500)
    next_compound = sim_col3.selectbox("Chart Compound Detail", ["HARD", "MEDIUM", "SOFT"])

    driver_state_df = ml_data[(ml_data["Driver"] == driver_a) & (ml_data["LapNumber"] == sim_lap)]
    if driver_state_df.empty:
        st.warning(f"No valid racing data for {driver_a} on Lap {sim_lap}. Try another lap.")
        return

    # Extract gap for competitor reaction modeling
    gap_at_sim = gap_df[gap_df["LapNumber"] == sim_lap]["Gap_Seconds"] if not gap_df.empty else pd.Series()
    rival_gap_s = float(gap_at_sim.iloc[0]) if not gap_at_sim.empty else None

    current_state = {
        "tyre_age":         driver_state_df.iloc[0]["tyre_age"],
        "compound_encoded": driver_state_df.iloc[0]["compound_encoded"],
        "baseline_pace":    driver_state_df.iloc[0]["baseline_pace"],
        "Track":            st.session_state["race"],
        "Driver":           driver_a,
    }
    track_env = ml_data[["LapNumber", "track_evolution", "TrackTemp"]].drop_duplicates("LapNumber")
    past_laps = ml_data[ml_data["LapNumber"] <= sim_lap]
    race_alpha, race_bias = estimate_causal_race_eb(past_laps, model, features)

    def_state_df = ml_data[(ml_data["Driver"] == driver_b) & (ml_data["LapNumber"] <= sim_lap)].sort_values("LapNumber")
    defender_state = None
    if not def_state_df.empty and driver_b != driver_a:
        d_last = def_state_df.iloc[-1]
        defender_state = {
            "tyre_age":         float(d_last["tyre_age"]),
            "compound_encoded": int(d_last["compound_encoded"]),
            "baseline_pace":    float(d_last["baseline_pace"]),
            "Track":            st.session_state["race"],
            "Driver":           driver_b,
        }

    # Run Monte Carlo simulation across 6 strategies
    with st.spinner(f"Running {n_rollouts:,} Monte Carlo rollouts across 6 strategies..."):
        mc_results = run_mc_simulation(
            model=model,
            current_lap=sim_lap,
            current_state=current_state,
            track_env=track_env,
            pit_loss_delta=pit_loss,
            race_name=st.session_state["race"],
            n_rollouts=n_rollouts,
            feature_cols=features,
            rival_gap_s=rival_gap_s,
            total_laps=max_lap,
            race_alpha=race_alpha,
            race_bias=race_bias,
            defender_state=defender_state,
        )

    # Strategy Table
    render_strategy_table(mc_results, st.session_state["race"], n_rollouts)

    # Chart 3: Net Undercut Advantage
    st.divider()
    chart_strat = next(
        (r for r in mc_results if r.get("compound_encoded") == COMPOUND_MAP[next_compound] and r.get("pit_offset") == 0),
        mc_results[1] if len(mc_results) > 1 else mc_results[0],
    )
    render_advantage_chart(chart_strat, mc_results[0], sim_lap, next_compound)

    # Chart 4: SHAP Explainability
    with st.spinner("Computing SHAP feature importance..."):
        shap_plot_df = compute_shap_explanation(model, features, current_state, track_env, sim_lap)
        render_shap_chart(shap_plot_df, sim_lap)


# --- MAIN APPLICATION ORCHESTRATION ---
inject_custom_css(st.session_state.get("driver_a"), st.session_state.get("driver_b"))
render_homepage_hero()

default_season = 2023
with st.spinner(f"Loading {default_season} schedule..."):
    default_races = get_schedule(default_season)

default_race = default_races[0] if default_races else "Bahrain Grand Prix"
init_session_state(default_season=default_season, default_race=default_race)

tab_strategy, tab_review = st.tabs(["🏎️ Live Strategy Dashboard", "📋 Race Review & What-If"])

with tab_strategy:
    render_race_setup(
        current_season=default_season,
        races=default_races,
        current_race=default_race,
        schedule_loader=get_schedule,
        on_load_requested=load_requested_race,
    )

    if "year" in st.session_state and "race" in st.session_state and st.session_state.get("data_loaded"):
        render_strategy_dashboard()
    else:
        st.info("Choose a season and Grand Prix, then load race strategy data to open the analysis dashboard.")

with tab_review:
    from ui.pages.race_review import render_race_review
    render_race_review()
