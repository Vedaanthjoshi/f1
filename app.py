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
from simulation import calculate_pit_loss, simulate_undercut_window, run_mc_simulation, SC_RATES, SC_RATE_DEFAULT

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
def process_backend_pipeline(year, race, _session):
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

    # ── Main panel: clean section header ──────────────────────────────────
    st.markdown(
        """
        <div class="f1-setup-header">
            <span class="f1-setup-kicker">&#9635; Grand Prix Configuration</span>
            <h2 class="f1-setup-title">Race Setup</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_col, brief_col = st.columns([1, 1], gap="large")

    with setup_col:
        st.markdown('<div class="f1-setup-card">', unsafe_allow_html=True)
        st.markdown('<div class="f1-setup-card-label">Configure the Grand Prix</div>', unsafe_allow_html=True)

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
        st.markdown('</div>', unsafe_allow_html=True)

    with brief_col:
        st.markdown(
            """
            <div class="f1-setup-card f1-brief-card">
                <div class="f1-setup-card-label">Strategy Brief</div>
                <p class="f1-brief-body">
                    The race engine will prepare real telemetry data and leverage
                    the pre-trained Global AI Model to simulate undercut strategies.
                </p>
                <div class="f1-brief-stats">
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">2018 – 2025</span>
                        <span class="f1-brief-stat-key">Season Range</span>
                    </div>
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">XGBoost</span>
                        <span class="f1-brief-stat-key">Degradation Model</span>
                    </div>
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">15-lap</span>
                        <span class="f1-brief-stat-key">Simulation Horizon</span>
                    </div>
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">Live Pit Loss</span>
                        <span class="f1-brief-stat-key">From FastF1 Data</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_strategy_dashboard():
    model, features = load_global_model()
    if model is None:
        st.error("Global Model not found! Please run `python train_global_model.py` first.")
        return

    with st.spinner(f"Downloading telemetry for {st.session_state['race']}..."):
        session = load_race_data(st.session_state["year"], st.session_state["race"])
        backend_payload, error_msg = process_backend_pipeline(st.session_state["year"], st.session_state["race"], session)

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

    # ── Phase 2 MC Simulation ─────────────────────────────────────────────
    st.divider()
    st.subheader(f"🎲 Monte Carlo Strategy Optimizer — {driver_a}")

    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 2])
    min_lap     = int(df_ab[df_ab["Driver"] == driver_a]["LapNumber"].min())
    max_lap     = int(df_ab[df_ab["Driver"] == driver_a]["LapNumber"].max())
    default_val = min(min_lap + 15, max_lap)
    sim_lap     = sim_col1.number_input("Simulate from Lap", min_value=min_lap, max_value=max_lap, value=default_val)
    n_rollouts  = sim_col2.select_slider("MC Rollouts", options=[200, 500, 1000], value=500)

    driver_state_df = ml_data[(ml_data["Driver"] == driver_a) & (ml_data["LapNumber"] == sim_lap)]
    if driver_state_df.empty:
        st.warning(f"No valid racing data for {driver_a} on Lap {sim_lap}. Try another lap.")
        return

    # Find gap for rival reaction modeling
    gap_at_sim = gap_df[gap_df["LapNumber"] == sim_lap]["Gap_Seconds"]
    rival_gap_s = float(gap_at_sim.iloc[0]) if not gap_at_sim.empty else None

    current_state = {
        "tyre_age":         driver_state_df.iloc[0]["tyre_age"],
        "compound_encoded": driver_state_df.iloc[0]["compound_encoded"],
        "baseline_pace":    driver_state_df.iloc[0]["baseline_pace"],
        "Track":            st.session_state["race"],
        "Driver":           driver_a,
    }
    track_env = ml_data[["LapNumber", "track_evolution", "TrackTemp"]].drop_duplicates("LapNumber")

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
        )

    # ── Strategy Ranking Cards ────────────────────────────────────────────
    sc_rate     = SC_RATES.get(st.session_state["race"], SC_RATE_DEFAULT)
    sc_pct      = f"{sc_rate:.0%}"
    best_strat  = next((r for r in mc_results if r["window_open"]), None)

    if best_strat:
        st.success(
            f"✅ **STRATEGY RECOMMENDED: {best_strat['name']}** — "
            f"{best_strat['p_net_gain']:.0%} probability of net gain, "
            f"expected +{best_strat['expected_gain_s']:.2f}s "
            f"(CI: {best_strat['gain_ci_low']:+.1f}s → {best_strat['gain_ci_high']:+.1f}s)"
        )
    else:
        st.info("No strategy currently clears the 55% confidence threshold. Stay out or wait for a better window.")

    # Render strategy table
    st.markdown(
        """
        <style>
        .mc-table {
            width: 100%; border-collapse: collapse;
            margin: 0.65rem 0 1.1rem;
            background: rgba(10,13,20,0.70);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 10px;
            overflow: hidden;
        }
        .mc-table th {
            font-size: 0.63rem; font-weight: 800; text-transform: uppercase;
            letter-spacing: 0.11em; color: rgba(255,255,255,0.38);
            padding: 0.6rem 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.09);
            text-align: left; background: rgba(255,255,255,0.03);
        }
        .mc-table td {
            font-size: 0.85rem; color: #E0E8F2;
            padding: 0.58rem 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.05);
            vertical-align: middle;
        }
        .mc-table tbody tr:last-child td { border-bottom: none; }
        .mc-table tbody tr:hover td { background: rgba(255,255,255,0.035); }
        .mc-table td:first-child { font-weight: 700; color: #FFFFFF; }
        .mc-p-high   { color: #3ECF8E; font-weight: 800; }
        .mc-p-mid    { color: #FFD166; font-weight: 700; }
        .mc-p-low    { color: #FF5C5C; font-weight: 700; }
        .mc-gain-pos { color: #3ECF8E; font-weight: 700; }
        .mc-gain-neg { color: #FF5C5C; font-weight: 700; }
        .mc-badge-open    { background:rgba(62,207,142,0.15); color:#3ECF8E;
                           border:1px solid rgba(62,207,142,0.38); border-radius:3px;
                           font-size:0.62rem; font-weight:800; padding:0.18rem 0.48rem; letter-spacing:0.08em; }
        .mc-badge-closed  { background:rgba(255,92,92,0.13); color:#FF5C5C;
                           border:1px solid rgba(255,92,92,0.30); border-radius:3px;
                           font-size:0.62rem; font-weight:800; padding:0.18rem 0.48rem; letter-spacing:0.08em; }
        .mc-badge-neutral { background:rgba(255,255,255,0.07); color:#7A8BA0;
                           border:1px solid rgba(255,255,255,0.11); border-radius:3px;
                           font-size:0.62rem; font-weight:800; padding:0.18rem 0.48rem; letter-spacing:0.08em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    rows_html = ""
    for r in mc_results:
        if r["name"] == "Stay Out":
            badge    = '<span class="mc-badge-neutral">BASELINE</span>'
            gain_str = "<span style='color:#7A8BA0'>&#8212;</span>"
            ci_str   = "<span style='color:#7A8BA0'>&#8212;</span>"
            sc_str   = "<span style='color:#7A8BA0'>&#8212;</span>"
            p_str    = "<span style='color:#7A8BA0'>&#8212;</span>"
        else:
            badge     = '<span class="mc-badge-open">OPEN</span>' if r["window_open"] else '<span class="mc-badge-closed">CLOSED</span>'
            p_val     = r['p_net_gain']
            p_cls     = 'mc-p-high' if p_val >= 0.65 else ('mc-p-mid' if p_val >= 0.50 else 'mc-p-low')
            p_str     = f'<span class="{p_cls}">{p_val:.0%}</span>'
            g_val     = r['expected_gain_s']
            g_cls     = 'mc-gain-pos' if g_val > 0 else 'mc-gain-neg'
            gain_str  = f'<span class="{g_cls}">{g_val:+.2f}s</span>'
            ci_lo, ci_hi = r['gain_ci_low'], r['gain_ci_high']
            ci_str    = f"<span style='color:#7A8BA0;font-size:0.78rem;'>{ci_lo:+.1f}s &rarr; {ci_hi:+.1f}s</span>"
            sc_val    = r['sc_expected_gain_s']
            sc_cls    = 'mc-gain-pos' if sc_val > 0 else 'mc-gain-neg'
            sc_str    = f'<span class="{sc_cls}">{sc_val:+.2f}s</span>'
        xo        = f"Lap {r['crossover_lap']}" if r.get("crossover_lap") else "<span style='color:#7A8BA0'>&#8212;</span>"
        cover_val = r.get('cover_rate', 0.0)
        cover_cls = 'mc-p-low' if cover_val >= 0.65 else ('mc-p-mid' if cover_val >= 0.35 else 'mc-gain-pos')
        cover_str = f'<span class="{cover_cls}">{cover_val:.0%}</span>'
        rows_html += (
            f"<tr>"
            f"<td><strong>{r['name']}</strong></td>"
            f"<td>{p_str}</td>"
            f"<td>{gain_str}</td>"
            f"<td>{ci_str}</td>"
            f"<td>{sc_str}</td>"
            f"<td>{cover_str}</td>"
            f"<td style='color:#AAB8C8;font-size:0.82rem;'>{xo}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="mc-table">
          <thead><tr>
            <th>Strategy</th><th>P(Net Gain)</th><th>E[Gain]</th>
            <th>90% CI</th><th>Under SC</th><th>Rival Cover</th><th>Crossover</th><th>Status</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        <p style='font-size:0.68rem;color:#5A6878;margin-top:0.3rem;line-height:1.5;'>
          &#9679; SC probability at {st.session_state['race']}: {sc_pct}
          &nbsp;&middot;&nbsp; {n_rollouts:,} MC rollouts
          &nbsp;&middot;&nbsp; Horizon: 15 laps
          &nbsp;&middot;&nbsp; Green P &ge; 65% / Amber &ge; 50% / Red &lt; 50%
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Net Advantage Chart (best pit strategy vs Stay Out) ───────────────
    st.divider()
    comp_map     = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    next_compound = sim_col3.selectbox("Chart Compound Detail", ["HARD", "MEDIUM", "SOFT"])
    # Find matching strategy in mc_results for chart
    chart_strat = next(
        (r for r in mc_results if r.get("compound_encoded") == comp_map[next_compound] and r.get("pit_offset") == 0),
        mc_results[1] if len(mc_results) > 1 else mc_results[0],
    )

    stay_out_row = mc_results[0]
    so_times     = stay_out_row["p50_lap_times"]
    pit_times    = chart_strat["p50_lap_times"]
    horizon_laps_plot = list(range(sim_lap, sim_lap + len(so_times)))

    lap_advantage          = [so_times[i] - pit_times[i] for i in range(len(so_times))]
    net_undercut_advantage = list(pd.Series(lap_advantage).cumsum())

    sim_plot_df = pd.DataFrame({
        "Lap":           horizon_laps_plot,
        "Net Advantage": net_undercut_advantage,
    })

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sim_plot_df["Lap"],
        y=sim_plot_df["Net Advantage"],
        mode="lines+markers",
        name=f"Net Time Gained vs Stay Out (p50)",
        line=dict(color="#E10600", width=3),
        marker=dict(size=8, color="#E10600", line=dict(width=1, color="white")),
    ))
    fig3.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.4)")

    if chart_strat.get("crossover_lap"):
        fig3.add_vline(
            x=chart_strat["crossover_lap"],
            line_dash="dash",
            line_color="#FFD166",
            annotation_text="Tyres Become Slower",
        )

    p_gain_pct = f"{chart_strat['p_net_gain']:.0%}"
    fig3.update_layout(
        title=f"Pit → {next_compound}: Net Cumulative Advantage (p50 median) — {p_gain_pct} P(gain)",
        xaxis_title="Race Lap",
        yaxis_title="Seconds Gained (+) or Lost (−)",
        legend_title_text="",
    )
    st.plotly_chart(style_chart(fig3), width='stretch')

    if chart_strat.get("window_open"):
        st.success(
            f"UNDERCUT WINDOW OPEN — Pit → {next_compound}: "
            f"{chart_strat['p_net_gain']:.0%} confidence, expected +{chart_strat['expected_gain_s']:.2f}s "
            f"(range {chart_strat['gain_ci_low']:+.1f}s to {chart_strat['gain_ci_high']:+.1f}s over {15} laps)"
        )
    elif chart_strat.get("crossover_lap"):
        st.info(f"Pace crossover at Lap {chart_strat['crossover_lap']}. {p_gain_pct} probability pitting is net positive.")
    else:
        st.error("Window firmly closed. Fresh tyres do not recover the pit-lane cost within the 15-lap horizon.")

    # ── Feature 6a: SHAP Explainability ────────────────────────────────────
    st.divider()
    st.subheader(f"Decision Explainability (SHAP) — Lap {sim_lap}")
    
    with st.spinner("Computing SHAP feature importance..."):
        import shap
        explainer = shap.TreeExplainer(model)
        
        track_evo_sim = track_env[track_env["LapNumber"] == sim_lap]["track_evolution"]
        track_evo_val = track_evo_sim.iloc[0] if not track_evo_sim.empty else 0
        
        track_temp_sim = track_env[track_env["LapNumber"] == sim_lap]["TrackTemp"]
        track_temp_val = track_temp_sim.iloc[0] if not track_temp_sim.empty else 30
        
        shap_df = pd.DataFrame([{
            'tyre_age': current_state['tyre_age'],
            'compound_encoded': current_state['compound_encoded'],
            'baseline_pace': current_state['baseline_pace'],
            'track_evolution': track_evo_val,
            'TrackTemp': track_temp_val,
            'Track': current_state['Track'],
            'Driver': current_state['Driver']
        }], columns=features)
        
        shap_df['Track'] = shap_df['Track'].astype('category')
        shap_df['Driver'] = shap_df['Driver'].astype('category')
        
        shap_values = explainer(shap_df)
        
        # XGBoost multi-output quantile (p50 is index 1)
        if len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1].values
            base_val = shap_values[0, :, 1].base_values
        else:
            vals = shap_values[0].values
            base_val = shap_values[0].base_values
            
        shap_df_plot = pd.DataFrame({
            "Feature": features,
            "SHAP Value": vals,
            "Value": shap_df.iloc[0].values
        }).sort_values(by="SHAP Value", key=abs, ascending=True)
        
        # Map feature names to readable labels
        feat_labels = {
            "tyre_age": "Tyre Age",
            "compound_encoded": "Tyre Compound",
            "baseline_pace": "Baseline Pace",
            "track_evolution": "Track Evolution",
            "TrackTemp": "Track Temp",
            "Track": "Circuit",
            "Driver": "Driver"
        }
        shap_df_plot["Label"] = shap_df_plot.apply(lambda row: f"{feat_labels.get(row['Feature'], row['Feature'])} = {row['Value']}", axis=1)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            y=shap_df_plot["Label"],
            x=shap_df_plot["SHAP Value"],
            orientation='h',
            marker=dict(
                color=["#FF4D5E" if v > 0 else "#36D399" for v in shap_df_plot["SHAP Value"]],
                line=dict(width=1, color="rgba(255,255,255,0.2)")
            )
        ))
        
        fig4.update_layout(
            title="What factors are driving your pace?",
            xaxis_title="Impact on predicted Lap Time (seconds)",
            yaxis_title="",
            margin=dict(l=150)
        )
        st.plotly_chart(style_chart(fig4), width='stretch')
        st.info("💡 **How to read SHAP**: Positive values (Red) mean the feature is making the lap time SLOWER (e.g. old tyres). Negative values (Green) mean the feature is making the car FASTER (e.g. track evolution or fresh soft tyres).")



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

