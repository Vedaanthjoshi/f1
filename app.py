import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Local backend modules
from data_pipeline import is_red_flag_race, clean_lap_data, merge_weather_data
from feature_engineering import apply_feature_engineering
from ml_model import prepare_ml_data, train_degradation_model
from simulation import calculate_pit_loss, simulate_undercut_window

# App Config
st.set_page_config(page_title="F1 Strategy Engineer", layout="wide", page_icon="🏎️")
fastf1.Cache.enable_cache('f1_cache')

# --- CACHED DATA LOADERS ---
@st.cache_data(show_spinner=False)
def get_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    # Filter for valid races (excluding testing)
    races = schedule[schedule['EventFormat'] != 'testing']['EventName'].tolist()
    return races

@st.cache_resource(show_spinner=False)
def load_race_data(year, race_name):
    # This directly loads the FastF1 session.
    session = fastf1.get_session(year, race_name, 'R')
    session.load(weather=True, messages=False)
    return session

@st.cache_data(show_spinner=False)
def process_backend_pipeline(_session):
    """
    Executes Phase 1-4 pipeline on the cached session.
    Returns the ready-to-chart data and trained ML model.
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
    model, features = train_degradation_model(ml_data)
    
    # Phase 4 (Pit loss)
    pit_loss = calculate_pit_loss(_session)
    
    return {
        'ml_data': ml_data,
        'model': model,
        'pit_loss': pit_loss
    }, None

# --- UI LAYOUT ---
st.title("🏎️ F1 Strategy Engineer")
st.markdown("Predict the exact moment a driver becomes vulnerable to an undercut using real telemetry and Machine Learning.")

# Sidebar Controls
st.sidebar.header("Race Selection")
# Per User Request: Allow till 2025
season = st.sidebar.selectbox("Season", list(range(2025, 2017, -1)), index=2) # default to 2023 for safety

with st.spinner(f"Loading {season} Schedule..."):
    races = get_schedule(season)

race = st.sidebar.selectbox("Grand Prix", races)

# Load Data Button
if st.sidebar.button("Load Race Data", use_container_width=True):
    st.session_state['data_loaded'] = False
    st.session_state['year'] = season
    st.session_state['race'] = race

# Display UI if data is requested
if 'year' in st.session_state and 'race' in st.session_state:
    with st.spinner(f"Downloading telemetry and processing ML pipeline for {st.session_state['race']}..."):
        session = load_race_data(st.session_state['year'], st.session_state['race'])
        backend_payload, error_msg = process_backend_pipeline(session)
    
    if error_msg:
        st.error(error_msg)
    else:
        st.success("Pipeline executed successfully. Mathematical model trained!")
        
        ml_data = backend_payload['ml_data']
        model = backend_payload['model']
        pit_loss = backend_payload['pit_loss']
        
        # Get Finisher List
        drivers = ml_data['Driver'].unique().tolist()
        
        st.sidebar.header("Driver Comparison")
        driver_a = st.sidebar.selectbox("Driver A (Target)", drivers, index=0)
        driver_b = st.sidebar.selectbox("Driver B (Attacker)", drivers, index=1 if len(drivers)>1 else 0)
        
        st.divider()
        
        # 1. Tyre Degradation Chart (Scatter)
        st.subheader("Tyre Degradation (Raw Pace)")
        
        # Filter for the two drivers
        df_ab = ml_data[ml_data['Driver'].isin([driver_a, driver_b])]
        
        color_map = {'SOFT': '#FF3333', 'MEDIUM': '#FFE800', 'HARD': '#E0E0E0'}
        
        fig1 = px.scatter(df_ab, x='LapNumber', y='LapTime_s', color='Compound', symbol='Driver', 
                          color_discrete_map=color_map, hover_data=['tyre_age'],
                          title=f"Actual Lap Times: {driver_a} vs {driver_b}",
                          template="plotly_dark")
        fig1.update_yaxes(autorange="reversed") # In racing, shorter time is better (higher up)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Gap Evolution Chart
        st.subheader("Gap Evolution")
        laps_a = df_ab[df_ab['Driver'] == driver_a][['LapNumber', 'Time']]
        laps_b = df_ab[df_ab['Driver'] == driver_b][['LapNumber', 'Time']]
        
        gap_df = pd.merge(laps_a, laps_b, on='LapNumber', suffixes=('_A', '_B'))
        gap_df['Gap_Seconds'] = (gap_df['Time_A'] - gap_df['Time_B']).dt.total_seconds()
        
        fig2 = px.line(gap_df, x='LapNumber', y='Gap_Seconds', 
                       title=f"Time Gap ({driver_a} behind {driver_b})",
                       labels={"Gap_Seconds": f"Gap (s) - Positive means {driver_a} is behind"},
                       template="plotly_dark")
        fig2.add_hline(y=0, line_dash="dot", line_color="white")
        st.plotly_chart(fig2, use_container_width=True)
        
        # 3. Undercut Simulation
        st.divider()
        st.subheader(f"Undercut Window Simulation ({driver_a})")
        
        # Let User select exactly which lap to simulate from
        min_lap = int(df_ab[df_ab['Driver'] == driver_a]['LapNumber'].min())
        max_lap = int(df_ab[df_ab['Driver'] == driver_a]['LapNumber'].max())
        
        sim_col1, sim_col2, sim_col3 = st.columns([1,1,2])
        default_val = min(min_lap + 15, max_lap)
        sim_lap = sim_col1.number_input("Simulate from Lap", min_value=min_lap, max_value=max_lap, value=default_val)
        next_compound = sim_col2.selectbox("Pit for Compound", ["HARD", "MEDIUM", "SOFT"])
        comp_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
        
        # Grab driver state at sim_lap
        driver_state_df = ml_data[(ml_data['Driver'] == driver_a) & (ml_data['LapNumber'] == sim_lap)]
        
        if driver_state_df.empty:
            st.warning(f"No valid racing data for {driver_a} on Lap {sim_lap} (might have been in pits or yellow flag). Try another lap.")
        else:
            current_state = {
                'tyre_age': driver_state_df.iloc[0]['tyre_age'],
                'compound_encoded': driver_state_df.iloc[0]['compound_encoded'],
                'baseline_pace': driver_state_df.iloc[0]['baseline_pace']
            }
            
            track_env = ml_data[['LapNumber', 'track_evolution', 'TrackTemp']].drop_duplicates('LapNumber')
            
            sim_result = simulate_undercut_window(
                model=model, 
                current_lap=sim_lap, 
                current_state=current_state, 
                track_env=track_env, 
                pit_loss_delta=pit_loss,
                next_compound_encoded=comp_map[next_compound]
            )
            
            horizon_laps = list(range(sim_lap, sim_lap + 15))
            sim_plot_df = pd.DataFrame({
                'Lap': horizon_laps,
                'Stay Out Pace': sim_result['stay_out_curve'],
                'Pit Now Pace': sim_result['pit_now_curve']
            })
            
            # Note: "Pace" means lap time. We want to show cumulative race time to show traditional undercut crossing!
            # Wait, PRD specifically says: "Crossover lap = first lap where Pit Now curve < Stay Out curve" which means lap time.
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=sim_plot_df['Lap'], y=sim_plot_df['Stay Out Pace'], mode='lines', name='Stay Out (Old Tyres)', line=dict(color='red')))
            fig3.add_trace(go.Scatter(x=sim_plot_df['Lap'], y=sim_plot_df['Pit Now Pace'], mode='lines', name=f'Pit Now ({next_compound})', line=dict(color='cyan')))
            
            if sim_result['crossover_lap']:
                 fig3.add_vline(x=sim_result['crossover_lap'], line_dash="dash", line_color="yellow", annotation_text="Crossover Lap")
                 
            fig3.update_layout(title="Prediction: Stay Out vs Pit Now Pace", xaxis_title="Race Lap", yaxis_title="Predicted Lap Time (s)", template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)
            
            if sim_result['window_open']:
                st.success(f"**UNDERCUT WINDOW OPEN!** Pitting {driver_a} now will produce a faster lap time on Lap {sim_result['crossover_lap']}.")
            elif sim_result['crossover_lap']:
                st.info(f"Window opens soon. Crossover point is Lap {sim_result['crossover_lap']}.")
            else:
                st.error("Window is firmly closed. Staying out is faster for the foreseeable horizon.")
