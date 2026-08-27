import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from ui.charts import style_chart
from ui.homepage import (
    apply_homepage_theme,
    render_event_strip,
    render_homepage_hero,
    render_matchup_preview,
)
from ui.metadata import driver_profile

# Local backend modules
from data_pipeline import is_red_flag_race, clean_lap_data, merge_weather_data
from feature_engineering import apply_feature_engineering
from ml_model import prepare_ml_data
from simulation import calculate_pit_loss, simulate_undercut_window

# App Config
st.set_page_config(page_title="F1 Strategy Engineer", layout="wide", page_icon=":racing_car:")
fastf1.Cache.enable_cache("f1_cache")


# --- CACHED DATA LOADERS ---
@st.cache_data(show_spinner=False)
def get_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    # Filter for valid races (excluding testing)
    races = schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()
    return races


@st.cache_resource(show_spinner=False)
def load_race_data(year, race_name):
    # This directly loads the FastF1 session.
    session = fastf1.get_session(year, race_name, "R")
    session.load(weather=True, messages=False)
    return session

@st.cache_resource(show_spinner=False)
def load_global_model():
    model_path = os.path.join('models', 'global_degradation_model.joblib')
    if not os.path.exists(model_path):
        return None, None
    saved_obj = joblib.load(model_path)
    model = saved_obj['model']
    features = saved_obj['features']
    
    # Inference is faster on CPU and avoids device mismatch warnings since pandas is CPU-bound
    try:
        model.set_params(device='cpu')
    except:
        pass
        
    return model, features

@st.cache_data(show_spinner=False)
def process_backend_pipeline(_session):
    """
    Executes Phase 1-2 pipeline on the cached session to get the data for plotting.
    Returns the ready-to-chart data and pit loss.
    """
    # Phase 1
    if is_red_flag_race(_session):
        return None, "Error: Race contains a Red Flag. Filtered out per PRD constraints to maintain valid degradation curves."

    raw_laps = _session.laps
    clean_laps = clean_lap_data(raw_laps)
    weather_data = _session.weather_data
    final_laps = merge_weather_data(clean_laps, weather_data)

    # Phase 2
    engineered_data = apply_feature_engineering(final_laps)

    # Phase 3
    ml_data = prepare_ml_data(engineered_data)

    # Phase 4 (Pit loss)
    pit_loss = calculate_pit_loss(_session)

    return {
        "ml_data": ml_data,
        "pit_loss": pit_loss,
    }, None


def load_requested_race(year, race_name):
    st.session_state["data_loaded"] = False
    st.session_state["year"] = year
    st.session_state["race"] = race_name


def sync_season(source):
    if source == 'sidebar':
        st.session_state.home_season = st.session_state.sidebar_season
    else:
        st.session_state.sidebar_season = st.session_state.home_season

def sync_race(source):
    if source == 'sidebar':
        st.session_state.home_race = st.session_state.sidebar_race
    else:
        st.session_state.sidebar_race = st.session_state.home_race

def render_race_setup(season, races, race):
    st.sidebar.caption("F1 Strategy Engineer")
    st.sidebar.header("Race Setup")
    
    # Initialize session state if not present
    if "sidebar_season" not in st.session_state:
        st.session_state.sidebar_season = season
        st.session_state.home_season = season
    if "sidebar_race" not in st.session_state:
        st.session_state.sidebar_race = race
        st.session_state.home_race = race

    sidebar_season = st.sidebar.selectbox(
        "Season", 
        list(range(2025, 2017, -1)), 
        key="sidebar_season",
        on_change=sync_season,
        args=('sidebar',)
    )
    
    sidebar_races = races
    if sidebar_season != season:
        with st.spinner(f"Loading {sidebar_season} schedule..."):
            sidebar_races = get_schedule(sidebar_season)
            
    # Ensure the race is valid for the season
    if st.session_state.sidebar_race not in sidebar_races:
        st.session_state.sidebar_race = sidebar_races[0]
        st.session_state.home_race = sidebar_races[0]

    sidebar_race = st.sidebar.selectbox(
        "Grand Prix",
        sidebar_races,
        key="sidebar_race",
        on_change=sync_race,
        args=('sidebar',)
    )
    
    if st.sidebar.button("Load Selected Race", width='stretch'):
        load_requested_race(sidebar_season, sidebar_race)

    st.markdown("### Race Entry")
    setup_col, launch_col = st.columns([0.82, 1.18], gap="large")

    with setup_col:
        st.markdown("#### Configure the Grand Prix")
        home_season = st.selectbox(
            "Season",
            list(range(2025, 2017, -1)),
            key="home_season",
            on_change=sync_season,
            args=('home',)
        )
        
        home_races = races
        if home_season != season:
            with st.spinner(f"Loading {home_season} schedule..."):
                home_races = get_schedule(home_season)
                
        # Ensure the race is valid for the season
        if st.session_state.home_race not in home_races:
            st.session_state.home_race = home_races[0]
            st.session_state.sidebar_race = home_races[0]

        home_race = st.selectbox(
            "Grand Prix",
            home_races,
            key="home_race",
            on_change=sync_race,
            args=('home',)
        )
        if st.button("Load Race Strategy Data", width='stretch'):
            load_requested_race(home_season, home_race)

    with launch_col:
        st.markdown("#### Strategy Brief")
        st.write(
            "The race engine will prepare real telemetry data and leverage the pre-trained Global AI Model to simulate undercut strategies."
        )


def render_strategy_dashboard():
    model, features = load_global_model()
    if model is None:
        st.error("Global Model not found! Please run `python train_global_model.py` first.")
        return

    with st.spinner(f"Downloading telemetry for {st.session_state['race']}..."):
        session = load_race_data(st.session_state["year"], st.session_state["race"])
        backend_payload, error_msg = process_backend_pipeline(session)

    if error_msg:
        st.error(error_msg)
        return

    st.success("Telemetry processed. Global AI Model loaded and ready.")

    ml_data = backend_payload["ml_data"]
    pit_loss = backend_payload["pit_loss"]

    # Get Finisher List
    drivers = ml_data["Driver"].unique().tolist()

    st.sidebar.divider()
    st.sidebar.header("Driver Comparison")
    driver_a = st.sidebar.selectbox("Driver A (Target)", drivers, index=0, key="driver_a")
    driver_b = st.sidebar.selectbox("Driver B (Attacker)", drivers, index=1 if len(drivers) > 1 else 0, key="driver_b")

    st.markdown("### Matchup Preview")
    render_event_strip(st.session_state["year"], st.session_state["race"], driver_a, driver_b)
    render_matchup_preview(driver_a, driver_b, st.session_state["race"])

    st.divider()

    # 1. Tyre Degradation Chart (Scatter)
    st.subheader("Tyre Degradation (Raw Pace)")

    # Filter for the two drivers
    df_ab = ml_data[ml_data["Driver"].isin([driver_a, driver_b])]

    if df_ab.empty:
        st.warning("No clean racing laps are available for the selected driver pairing. Try another driver or race.")
        return

    color_map = {"SOFT": "#FF3333", "MEDIUM": "#FFE800", "HARD": "#E0E0E0"}
    driver_color_map = {
        driver_a: driver_profile(driver_a)["team"].get("accentColor", "#E10600"),
        driver_b: driver_profile(driver_b)["team"].get("accentColor", "#27F4D2"),
    }
    if driver_color_map[driver_a] == driver_color_map[driver_b]:
        driver_color_map[driver_b] = "#FFFFFF"

    fig1 = px.scatter(
        df_ab,
        x="LapNumber",
        y="LapTime_s",
        color="Driver",
        symbol="Compound",
        color_discrete_map=driver_color_map,
        hover_data=["tyre_age"],
        title=f"Actual Lap Times: {driver_a} vs {driver_b}",
        template="plotly_dark",
        labels={
            "LapNumber": "Race Lap",
            "LapTime_s": "Lap Time (seconds, lower is faster)",
            "tyre_age": "Tyre Age",
            "Compound": "Tyre",
        },
    )
    fig1.update_yaxes(autorange="reversed")  # In racing, shorter time is better (higher up)
    fig1.update_traces(
        marker=dict(size=8, opacity=0.9, line=dict(width=1.1, color="#FFFFFF")),
        selector=dict(mode="markers"),
    )
    fig1.update_layout(legend_title_text="")
    st.plotly_chart(style_chart(fig1), width='stretch')

    # 2. Gap Evolution Chart
    st.subheader("Gap Evolution")
    laps_a = df_ab[df_ab["Driver"] == driver_a][["LapNumber", "Time"]]
    laps_b = df_ab[df_ab["Driver"] == driver_b][["LapNumber", "Time"]]

    gap_df = pd.merge(laps_a, laps_b, on="LapNumber", suffixes=("_A", "_B"))
    gap_df["Gap_Seconds"] = (gap_df["Time_A"] - gap_df["Time_B"]).dt.total_seconds()

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=gap_df["LapNumber"],
            y=gap_df["Gap_Seconds"],
            mode="lines+markers",
            name=f"{driver_a} gap to {driver_b}",
            line=dict(color=driver_color_map[driver_a], width=4),
            marker=dict(size=6, color=driver_color_map[driver_a], line=dict(color="#FFFFFF", width=1)),
        )
    )
    fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)")
    fig2.update_layout(
        title=f"Time Gap ({driver_a} behind {driver_b})",
        xaxis_title="Race Lap",
        yaxis_title=f"Gap to {driver_b} (seconds)",
        legend_title_text="",
    )
    st.plotly_chart(style_chart(fig2), width='stretch')

    # 3. Undercut Simulation
    st.divider()
    st.subheader(f"Undercut Window Simulation ({driver_a})")

    # Let User select exactly which lap to simulate from
    min_lap = int(df_ab[df_ab["Driver"] == driver_a]["LapNumber"].min())
    max_lap = int(df_ab[df_ab["Driver"] == driver_a]["LapNumber"].max())

    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 2])
    default_val = min(min_lap + 15, max_lap)
    sim_lap = sim_col1.number_input("Simulate from Lap", min_value=min_lap, max_value=max_lap, value=default_val)
    next_compound = sim_col2.selectbox("Pit for Compound", ["HARD", "MEDIUM", "SOFT"])
    comp_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}

    # Grab driver state at sim_lap
    driver_state_df = ml_data[(ml_data["Driver"] == driver_a) & (ml_data["LapNumber"] == sim_lap)]

    if driver_state_df.empty:
        st.warning(f"No valid racing data for {driver_a} on Lap {sim_lap} (might have been in pits or yellow flag). Try another lap.")
        return

    current_state = {
        "tyre_age": driver_state_df.iloc[0]["tyre_age"],
        "compound_encoded": driver_state_df.iloc[0]["compound_encoded"],
        "baseline_pace": driver_state_df.iloc[0]["baseline_pace"],
        "Track": st.session_state["race"],
        "Driver": driver_a,
    }

    track_env = ml_data[["LapNumber", "track_evolution", "TrackTemp"]].drop_duplicates("LapNumber")

    sim_result = simulate_undercut_window(
        model=model,
        current_lap=sim_lap,
        current_state=current_state,
        track_env=track_env,
        pit_loss_delta=pit_loss,
        next_compound_encoded=comp_map[next_compound],
        feature_cols=features
    )

    horizon_laps = list(range(sim_lap, sim_lap + 15))
    sim_plot_df = pd.DataFrame(
        {
            "Lap": horizon_laps,
            "Net Advantage": sim_result["net_undercut_advantage"],
        }
    )

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=sim_plot_df["Lap"],
            y=sim_plot_df["Net Advantage"],
            mode="lines+markers",
            name="Net Time Gained (s)",
            line=dict(color="#E10600", width=3),
            marker=dict(size=8, color="#E10600", line=dict(width=1, color="white"))
        )
    )
    
    # Add zero-line to show where the advantage becomes positive
    fig3.add_hline(y=0, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)")

    if sim_result["crossover_lap"]:
        fig3.add_vline(
            x=sim_result["crossover_lap"],
            line_dash="dash",
            line_color="#FFD166",
            annotation_text="Tyres Become Slower",
        )

    fig3.update_layout(
        title="Net Undercut Advantage (Pitting Now vs Staying Out)",
        xaxis_title="Race Lap",
        yaxis_title="Seconds Gained (+) or Lost (-) per lap",
        legend_title_text="",
    )
    st.plotly_chart(style_chart(fig3), width='stretch')

    if sim_result["window_open"]:
        st.success(f"UNDERCUT WINDOW OPEN. Pitting {driver_a} now gives an immediate, compounding pace advantage over cars staying out on old tyres.")
    elif sim_result["crossover_lap"]:
        st.info(f"Pace crossover happens on Lap {sim_result['crossover_lap']}. Old tyres will become slower than fresh tyres then.")
    else:
        st.error("Window is firmly closed. Staying out is faster for the foreseeable horizon.")


apply_homepage_theme(st.session_state.get("driver_a"), st.session_state.get("driver_b"))
render_homepage_hero()

default_season = 2023
with st.spinner(f"Loading {default_season} schedule..."):
    default_races = get_schedule(default_season)

default_race = default_races[0]
render_race_setup(default_season, default_races, default_race)

if "year" in st.session_state and "race" in st.session_state:
    render_strategy_dashboard()
else:
    st.info("Choose a season and Grand Prix, then load race strategy data to open the analysis dashboard.")

