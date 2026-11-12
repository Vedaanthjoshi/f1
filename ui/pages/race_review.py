"""
ui/pages/race_review.py — Post-Race Review & What-If Strategy Dashboard
========================================================================
Implements Phase 2 & Phase 3 of the Post-Race Strategy Review:
  1. Race reconstruction & lap-by-lap telemetry replay.
  2. Multi-stop pit decision regret scan (-8 to +8 laps).
  3. Probabilistic Monte Carlo what-if simulator with delta formulation.
  4. Predicted vs actual tyre degradation per stint & residual summary.
  5. Honesty banners for in-sample vs out-of-sample (LORO) models.
"""

from __future__ import annotations

from html import escape
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from replay.whatif import (
    run_whatif,
    regret_scan,
    prepare_race_data,
    predict_stint_degradation_curve,
    SHIFT_MIN,
    SHIFT_MAX,
)
from pipeline import fetch_season_schedule
from ui.charts.styling import style_chart
from ui.metadata import driver_profile

# All 25 circuits present in data/historical_laps.csv (2022–2025)
TRAINING_SET_RACES = {
    "Abu Dhabi Grand Prix",
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Bahrain Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Chinese Grand Prix",
    "Dutch Grand Prix",
    "Emilia Romagna Grand Prix",
    "French Grand Prix",
    "Hungarian Grand Prix",
    "Italian Grand Prix",
    "Japanese Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Qatar Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Spanish Grand Prix",
    "São Paulo Grand Prix",
    "United States Grand Prix",
}

FALLBACK_RACES = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Japanese Grand Prix",
    "Chinese Grand Prix",
    "Miami Grand Prix",
    "Emilia Romagna Grand Prix",
    "Monaco Grand Prix",
    "Canadian Grand Prix",
    "Spanish Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix",
]

RIVAL_MODES = {
    "cover_traffic": "Rivals Cover + Dirty Air (Realistic)",
    "cover": "Rivals Cover Only (+1 Lap Reaction)",
    "fixed": "Rivals Fixed (Stopwatch Baseline)",
}


@st.cache_data(show_spinner=False)
def load_cached_schedule(year: int) -> list[str]:
    """Fetch season schedule dynamically with fallback."""
    try:
        races = fetch_season_schedule(year)
        if races:
            return races
    except Exception:
        pass
    return FALLBACK_RACES


@st.cache_data(show_spinner=False)
def load_cached_race_data(year: int, race_name: str):
    """Load and cache telemetry and model artifacts (excluding unpickleable Session)."""
    return prepare_race_data(year, race_name, include_session=False)


def _stint_pill_html(compound: str, laps_count: int) -> str:
    """Render a compact Pirelli compound pill + stint lap count."""
    comp = str(compound or "").upper()
    if "SOFT" in comp:
        cls, letter = "pirelli-soft", "S"
    elif "MEDIUM" in comp:
        cls, letter = "pirelli-medium", "M"
    elif "HARD" in comp:
        cls, letter = "pirelli-hard", "H"
    elif "INTER" in comp:
        cls, letter = "pirelli-inter", "I"
    elif "WET" in comp:
        cls, letter = "pirelli-wet", "W"
    else:
        cls, letter = "pirelli-stay", "?"
    return (
        f'<span class="f1-stint-chip">'
        f'<span class="pirelli-pill {cls}">{letter}</span>'
        f'{laps_count}L'
        f'</span>'
    )


def _render_driver_banner(
    selected_driver: str,
    driver_laps: pd.DataFrame,
    final_pos: int,
    pit_laps: list[int],
) -> None:
    """Render a compact F1 timing-tower driver summary header with stint pills."""
    profile = driver_profile(selected_driver)
    drv = profile["driver"]
    team = profile["team"]
    accent = team.get("accentColor", "#E10600")
    number = f"#{drv['number']}" if drv.get("number") else selected_driver
    full_name = drv.get("name", selected_driver)
    team_name = team.get("name", "F1 Team")

    stint_chips: list[str] = []
    if "Stint" in driver_laps.columns:
        for _, grp in driver_laps.groupby("Stint", sort=True):
            comp = grp["Compound"].dropna().iloc[0] if "Compound" in grp.columns and not grp["Compound"].dropna().empty else "HARD"
            stint_chips.append(_stint_pill_html(str(comp), len(grp)))

    stints_html = '<span class="f1-stint-arrow">&rarr;</span>'.join(stint_chips)
    stops_text = ", ".join(f"L{p}" for p in pit_laps) if pit_laps else "No Stops"

    st.markdown(
        f"""
        <div class="f1-driver-banner" style="--accent: {escape(accent)};">
            <div class="f1-driver-banner-left">
                <div class="f1-driver-banner-num">{escape(number)}</div>
                <div>
                    <div class="f1-driver-banner-name">{escape(selected_driver)} &middot; {escape(full_name)}</div>
                    <div class="f1-driver-banner-sub">{escape(team_name)} &nbsp;&bull;&nbsp; Classified P{final_pos} &nbsp;&bull;&nbsp; Pit Laps: {escape(stops_text)}</div>
                </div>
            </div>
            <div class="f1-stint-chain">{stints_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_race_review():
    """Main render function for the Post-Race Review tab."""
    st.markdown(
        """
        <div class="f1-setup-header" style="margin-top: 0.4rem;">
            <span class="f1-setup-kicker">&#9635; Counterfactual Telemetry &nbsp;//&nbsp; What-If Engine</span>
            <h2 class="f1-setup-title">Post-Race Strategy Review</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 1. Unified Race & Reaction Setup Card
    # -----------------------------------------------------------------------
    with st.container(border=True):
        col_yr, col_gp, col_mode = st.columns([1, 2, 2])
        available_seasons = [2025, 2024, 2023, 2022, 2021, 2020, 2019]
        year = col_yr.selectbox("Season", available_seasons, index=2, key="review_year")

        races_available = load_cached_schedule(year)
        default_gp_idx = races_available.index("Bahrain Grand Prix") if "Bahrain Grand Prix" in races_available else 0
        gp_name = col_gp.selectbox("Grand Prix", races_available, index=default_gp_idx, key="review_gp")

        rival_mode = col_mode.selectbox(
            "Rival Reaction Model",
            list(RIVAL_MODES.keys()),
            index=0,
            format_func=lambda k: RIVAL_MODES[k],
            key="review_rival_mode",
        )

    # Load Race Data
    with st.spinner(f"Loading telemetry & replay table for {year} {gp_name}..."):
        try:
            race_data = load_cached_race_data(year, gp_name)
        except Exception as e:
            st.error(f"Failed to load race session: {e}")
            return

    replay_table = race_data["replay_table"]
    g3_model = race_data["g3_model"]
    g3_choice = race_data["g3_choice"]
    g3_linear_slopes = race_data["g3_linear_slopes"]
    g3_residuals = race_data["g3_residuals"]
    pit_loss_green = race_data["pit_loss_green"]
    pit_loss_neutralized = race_data["pit_loss_neutralized"]
    neutralized_laps = race_data["neutralized_laps"]
    circuit_physics = race_data.get("circuit_physics")
    overtaking_difficulty = race_data.get("overtaking_difficulty", "Medium")
    race_deg_stats = race_data.get("race_deg_stats", {})

    # Compact single-row status strip (replaces stacked warning/info/caption walls)
    is_bahrain = "bahrain" in gp_name.lower()
    if is_bahrain:
        val_pill = '<span class="f1-status-pill is-loro">Validation: <strong>LORO (Out-of-Sample)</strong></span>'
    elif gp_name in TRAINING_SET_RACES and year in (2022, 2023, 2024, 2025):
        val_pill = '<span class="f1-status-pill is-insample">Validation: <strong>In-Sample (Monotonic Constrained)</strong></span>'
    else:
        val_pill = '<span class="f1-status-pill is-loro">Validation: <strong>Out-of-Sample Circuit</strong></span>'

    mode_short = {
        "cover_traffic": "Cover (+1L) + Dirty Air",
        "cover": "Cover (+1L) Only",
        "fixed": "Fixed Rivals (Stopwatch)",
    }.get(rival_mode, rival_mode)
    green_med = f"{np.median(pit_loss_green):.1f}s" if len(pit_loss_green) else "—"
    sc_cnt = len(pit_loss_neutralized)

    st.markdown(
        f"""
        <div class="f1-status-bar">
            {val_pill}
            <span class="f1-status-pill is-mode">Rivals: <strong>{escape(mode_short)}</strong></span>
            <span class="f1-status-pill is-pit">Pit Pool: <strong>{green_med} green ({len(pit_loss_green)}) &middot; {sc_cnt} SC/VSC</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    drivers = sorted(replay_table["Driver"].unique().tolist())
    def_driver_idx = drivers.index("VER") if "VER" in drivers else 0
    drv_col, _ = st.columns([1, 3])
    selected_driver = drv_col.selectbox("Driver Under Review", drivers, index=def_driver_idx, key="review_driver")

    driver_laps = replay_table[replay_table["Driver"] == selected_driver].sort_values("LapNumber").reset_index(drop=True)
    if driver_laps.empty:
        st.warning(f"No laps recorded for driver {selected_driver}.")
        return

    # Driver Stops & KPIs
    driver_stops = driver_laps[driver_laps["is_in_lap"] == True]
    pit_laps = driver_stops["LapNumber"].astype(int).tolist()
    final_pos = int(driver_laps["Position"].dropna().iloc[-1]) if not driver_laps["Position"].dropna().empty else 20
    stints_run = int(driver_laps["Stint"].dropna().max()) if not driver_laps["Stint"].dropna().empty else 1

    _render_driver_banner(selected_driver, driver_laps, final_pos, pit_laps)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Position", f"P{final_pos}")
    m2.metric("Laps Completed", f"{len(driver_laps)}")
    m3.metric("Pit Stops", f"{len(pit_laps)}")
    m4.metric("Stints Run", f"{stints_run}")

    st.divider()

    # -----------------------------------------------------------------------
    # Sub-Tabs for Review
    # -----------------------------------------------------------------------
    tab_regret, tab_whatif, tab_replay, tab_deg = st.tabs([
        "Pit Regret Scan",
        "Monte Carlo What-If",
        "Lap-by-Lap Replay",
        "Tyre Degradation",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Regret Scan
    # -----------------------------------------------------------------------
    with tab_regret:
        st.caption(
            f"Evaluates counterfactual pit shifts ({SHIFT_MIN} to +{SHIFT_MAX} laps) under {RIVAL_MODES[rival_mode]}. "
            "Shifts whose 90% CI spans 0.0s are marked indistinguishable from noise."
        )

        if not pit_laps:
            st.info(f"{selected_driver} did not make any pit stops during this race.")
        else:
            stop_labels = [f"Stop {i+1} (Lap {p})" for i, p in enumerate(pit_laps)]
            sel_col, _ = st.columns([1, 2])
            chosen_stop_idx = sel_col.selectbox("Pit Stop to Scan", range(len(pit_laps)), format_func=lambda i: stop_labels[i])
            actual_stop_lap = pit_laps[chosen_stop_idx]

            with st.spinner("Executing 500-rollout Monte Carlo regret scan across counterfactual shifts..."):
                scan_df = regret_scan(
                    replay_table=replay_table,
                    driver=selected_driver,
                    stop_idx=chosen_stop_idx,
                    g3_model=g3_model,
                    g3_choice=g3_choice,
                    g3_linear_slopes=g3_linear_slopes,
                    g3_residuals=g3_residuals,
                    pit_loss_green=pit_loss_green,
                    pit_loss_neutralized=pit_loss_neutralized,
                    neutralized_laps=neutralized_laps,
                    n_paths=500,
                    rival_mode=rival_mode,
                    circuit_physics=circuit_physics,
                    overtaking_difficulty=overtaking_difficulty,
                )

            if scan_df.empty:
                st.warning("No feasible shifts found for this stop.")
            else:
                scan_df["Status"] = scan_df.apply(
                    lambda r: "Not distinguishable from noise" if r["is_noise"]
                    else ("Faster (Advantage)" if r["expected_time_delta"] < 0 else "Slower (Loss)"),
                    axis=1,
                )
                scan_df["New In-Lap"] = actual_stop_lap + scan_df["shift"]
                scan_df = scan_df[scan_df["New In-Lap"] >= 2].copy()

                if scan_df.empty:
                    st.warning("No feasible positive pit laps found for this stop.")
                else:
                    # Highlight optimal shift in 3 clean KPI cards
                    best_row = scan_df.loc[scan_df["expected_time_delta"].idxmin()]
                    best_shift = int(best_row["shift"])
                    best_delta = float(best_row["expected_time_delta"])
                    best_lap = int(best_row["New In-Lap"])

                    r_k1, r_k2, r_k3 = st.columns(3)
                    r_k1.metric("Actual Pit Lap", f"Lap {actual_stop_lap}", delta=f"Stop {chosen_stop_idx + 1}")
                    r_k2.metric("Optimal Pit Lap", f"Lap {best_lap}", delta=f"{best_shift:+d} laps")
                    r_k3.metric(
                        "Best Expected Delta",
                        f"{best_delta:+.2f} s",
                        delta="Noise / Neutral" if best_row["is_noise"] else ("Net Gain" if best_delta < 0 else "Net Loss"),
                        delta_color="normal" if best_delta < 0 else "inverse",
                    )

                    colors = scan_df["Status"].map({
                        "Faster (Advantage)": "#3ECF8E",
                        "Not distinguishable from noise": "#7A8BA0",
                        "Slower (Loss)": "#E10600",
                    })

                    fig_regret = go.Figure()
                    fig_regret.add_trace(go.Bar(
                        x=scan_df["New In-Lap"],
                        y=scan_df["expected_time_delta"],
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=scan_df["p95"] - scan_df["expected_time_delta"],
                            arrayminus=scan_df["expected_time_delta"] - scan_df["p05"],
                            color="rgba(170, 184, 200, 0.55)",
                        ),
                        marker_color=colors,
                        name="Expected Delta",
                        hovertemplate="Pit Lap: %{x}<br>Shift: %{customdata[0]:+d} laps<br>Time Delta: %{y:+.2f} s<br>Status: %{customdata[1]}<extra></extra>",
                        customdata=scan_df[["shift", "Status"]].values,
                    ))
                    fig_regret.add_hline(y=0.0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
                    fig_regret.add_vline(
                        x=actual_stop_lap,
                        line_dash="dot",
                        line_color="#FFFFFF",
                        annotation_text=f"Actual L{actual_stop_lap}",
                    )
                    fig_regret.update_layout(
                        xaxis_title="Counterfactual Pit Lap (Positive Race Laps)",
                        yaxis_title="Expected Time Delta (s) [Negative = Faster]",
                        xaxis=dict(dtick=1),
                        height=430,
                    )
                    st.plotly_chart(
                        style_chart(fig_regret, f"Pit Regret Curve — Stop {chosen_stop_idx + 1} (Actual: Lap {actual_stop_lap})"),
                        width="stretch",
                    )

                    display_cols = [
                        "New In-Lap", "shift", "expected_time_delta", "p05", "p95",
                        "p_position_improves", "n_rival_reactions", "dirty_air_laps",
                        "blocked_laps", "is_extrapolated", "robustness", "Status",
                    ]
                    available_disp = [c for c in display_cols if c in scan_df.columns]
                    with st.expander("Detailed Shift Matrix & Robustness Metrics", expanded=False):
                        st.dataframe(
                            scan_df[available_disp].rename(columns={
                                "shift": "Shift (Laps)",
                                "expected_time_delta": "Exp. Delta (s)",
                                "p05": "5th Pct (s)",
                                "p95": "95th Pct (s)",
                                "p_position_improves": "P(Pos Improves)",
                                "n_rival_reactions": "Rivals Reacted",
                                "dirty_air_laps": "Dirty Air Laps",
                                "blocked_laps": "Blocked Laps",
                                "is_extrapolated": "Extrapolated",
                                "robustness": "Robustness",
                            }).style.format({
                                "Exp. Delta (s)": "{:+.2f}",
                                "5th Pct (s)": "{:+.2f}",
                                "95th Pct (s)": "{:+.2f}",
                                "P(Pos Improves)": "{:.1%}",
                                "Dirty Air Laps": "{:.1f}",
                                "Blocked Laps": "{:.1f}",
                            }),
                            width="stretch",
                        )

    # -----------------------------------------------------------------------
    # TAB 2: Interactive What-If Simulator
    # -----------------------------------------------------------------------
    with tab_whatif:
        if not pit_laps:
            st.info("No pit stops available to simulate.")
        else:
            max_driver_lap = int(driver_laps["LapNumber"].max())
            with st.container(border=True):
                w_col1, w_col2, w_col3 = st.columns([1.2, 2, 1])
                w_stop_idx = w_col1.selectbox("Pit Stop to Alter", range(len(pit_laps)), format_func=lambda i: f"Stop {i+1} (Lap {pit_laps[i]})", key="whatif_stop_idx")
                w_actual_lap = pit_laps[w_stop_idx]

                min_shift = max(SHIFT_MIN, 2 - w_actual_lap)
                if w_stop_idx > 0:
                    min_shift = max(min_shift, pit_laps[w_stop_idx - 1] + 2 - w_actual_lap)
                max_shift = min(SHIFT_MAX, max_driver_lap - 1 - w_actual_lap)
                if w_stop_idx + 1 < len(pit_laps):
                    max_shift = min(max_shift, pit_laps[w_stop_idx + 1] - 2 - w_actual_lap)
                min_shift = min(min_shift, 0)
                max_shift = max(max_shift, 0)

                if "whatif_shift_slider" in st.session_state:
                    st.session_state["whatif_shift_slider"] = max(
                        min_shift, min(max_shift, int(st.session_state["whatif_shift_slider"]))
                    )

                if min_shift < max_shift:
                    w_shift = w_col2.slider(
                        "Shift Lap Count",
                        min_value=int(min_shift),
                        max_value=int(max_shift),
                        value=0,
                        step=1,
                        key="whatif_shift_slider",
                    )
                else:
                    w_shift = 0
                    w_col2.caption("No feasible shift window around this stop.")

                w_paths = w_col3.select_slider("Rollout Paths", [200, 500, 1000], value=500, key="whatif_paths")

                w_new_lap = w_actual_lap + w_shift
                st.caption(f"Simulating **Lap {w_new_lap}** ({w_shift:+d} laps vs actual Lap {w_actual_lap}) under **{RIVAL_MODES[rival_mode]}**")

            with st.spinner("Computing counterfactual trajectory..."):
                sim_res = run_whatif(
                    replay_table=replay_table,
                    driver=selected_driver,
                    stop_idx=w_stop_idx,
                    shift_laps=w_shift,
                    g3_model=g3_model,
                    g3_choice=g3_choice,
                    g3_linear_slopes=g3_linear_slopes,
                    g3_residuals=g3_residuals,
                    pit_loss_green=pit_loss_green,
                    pit_loss_neutralized=pit_loss_neutralized,
                    neutralized_laps=neutralized_laps,
                    n_paths=w_paths,
                    rival_mode=rival_mode,
                    circuit_physics=circuit_physics,
                    overtaking_difficulty=overtaking_difficulty,
                )

            if not sim_res["is_feasible"]:
                st.error(f"Infeasible scenario: {sim_res['infeasible_reason']}")
            else:
                p50_val = sim_res["final_time_delta_p50"]
                p05_val = sim_res["final_time_delta_p05"]
                p95_val = sim_res["final_time_delta_p95"]

                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    "Expected Time Delta",
                    f"{p50_val:+.2f} s",
                    delta=f"{(-p50_val):+.2f}s {'gain' if p50_val < 0 else 'loss'}",
                    delta_color="normal",
                )
                k2.metric("90% Confidence Interval", f"[{p05_val:+.2f}s, {p95_val:+.2f}s]")
                k3.metric("Final Position (Expected)", f"P{int(sim_res['position_per_lap'][-1])}", delta=f"{final_pos - int(sim_res['position_per_lap'][-1]):+d} positions")
                k4.metric(
                    "Dirty Air / Traffic",
                    f"+{sim_res.get('dirty_air_total_s', 0.0):.2f} s",
                    delta=f"{sim_res.get('dirty_air_laps', 0.0):.1f} wake | {sim_res.get('blocked_laps', 0.0):.1f} blocked",
                    delta_color="inverse",
                )

                reactions = sim_res.get("rival_reactions", [])
                if reactions:
                    cards_html = "".join(
                        f'<div class="f1-reaction-card"><strong>{escape(str(r["driver"]))}</strong> '
                        f'boxed Lap {r["reacted_pit"]} <span style="color:#7A8BA0;">(actual L{r["actual_pit"]})</span>'
                        f'<br><span style="color:#AAB8C8;">{escape(str(r["reason"]))}</span></div>'
                        for r in reactions
                    )
                    st.markdown(f'<div class="f1-reaction-grid">{cards_html}</div>', unsafe_allow_html=True)
                elif rival_mode != "fixed" and w_shift != 0:
                    st.caption("No rival within the pit window needed to cover or counter this shift.")

                chart_col1, chart_col2 = st.columns([3, 2], gap="medium")

                with chart_col1:
                    lap_numbers = driver_laps["LapNumber"].values
                    delta_curve = sim_res["delta_per_lap"]

                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=lap_numbers,
                        y=delta_curve,
                        mode="lines",
                        line=dict(color="#FFD166", width=3),
                        name="Cumulative Delta",
                        hovertemplate="Lap %{x}: %{y:+.2f}s<extra></extra>",
                    ))
                    fig_cum.add_vline(x=w_actual_lap, line_dash="dash", line_color="#E10600", annotation_text="Actual Pit")
                    if w_shift != 0:
                        fig_cum.add_vline(x=w_new_lap, line_dash="dash", line_color="#00D4FF", annotation_text="New Pit")
                    fig_cum.add_hline(y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.35)")
                    fig_cum.update_layout(
                        xaxis_title="Lap Number",
                        yaxis_title="Cumulative Delta (s)",
                        height=400,
                    )
                    st.plotly_chart(style_chart(fig_cum, f"Cumulative Delta ({selected_driver})"), width="stretch")

                with chart_col2:
                    dist = sim_res["position_distribution"]
                    if len(dist) > 0 and dist.sum() > 0:
                        pos_indices = [i for i, p in enumerate(dist) if p > 0.005 and i > 0]
                        fig_dist = px.bar(
                            x=[f"P{i}" for i in pos_indices],
                            y=[dist[i] for i in pos_indices],
                            labels={"x": "Finish Position", "y": "Probability"},
                        )
                        fig_dist.update_traces(marker_color="#00D4FF")
                        fig_dist.update_layout(height=400, yaxis_tickformat=".0%")
                        st.plotly_chart(style_chart(fig_dist, f"Finish Distribution ({w_paths} paths)"), width="stretch")

    # -----------------------------------------------------------------------
    # TAB 3: Lap-by-Lap Replay Table
    # -----------------------------------------------------------------------
    with tab_replay:
        sc_laps_count = int(driver_laps["is_neutralized"].sum()) if "is_neutralized" in driver_laps.columns else 0
        st.caption(f"Historical telemetry for {selected_driver} ({year} {gp_name}) — {len(driver_laps)} laps, {len(pit_laps)} stops, {sc_laps_count} SC/VSC laps.")
        replay_cols = [
            "LapNumber", "Position", "LapTime_s", "Compound", "TyreLife",
            "Stint", "is_in_lap", "is_out_lap", "is_neutralized", "gap_ahead_s", "gap_behind_s"
        ]
        available_cols = [c for c in replay_cols if c in driver_laps.columns]

        st.dataframe(
            driver_laps[available_cols].style.format({
                "LapTime_s": "{:.3f}",
                "gap_ahead_s": "{:.2f}",
                "gap_behind_s": "{:.2f}",
                "Position": "{:.0f}",
                "LapNumber": "{:.0f}",
                "TyreLife": "{:.0f}",
                "Stint": "{:.0f}",
            }),
            height=460,
            width="stretch",
        )

    # -----------------------------------------------------------------------
    # TAB 4: Tyre Degradation (Predicted vs Actual)
    # -----------------------------------------------------------------------
    with tab_deg:
        st.caption("Fuel-corrected lap-time progression vs tyre age (clean green-flag laps only).")

        clean_driver_laps = driver_laps[
            (driver_laps["is_in_lap"] == False)
            & (driver_laps["is_out_lap"] == False)
            & (driver_laps["is_neutralized"] == False)
            & (driver_laps["LapTime_s"].notnull())
        ].copy()

        if clean_driver_laps.empty:
            st.info("No clean green-flag laps available for degradation analysis.")
        else:
            fig_deg = go.Figure()
            drv_lin_errs: list[float] = []
            drv_xgb_errs: list[float] = []
            for stint_id, grp in clean_driver_laps.groupby("Stint"):
                grp = grp.sort_values("TyreLife")
                comp = grp["Compound"].iloc[0] if "Compound" in grp.columns else "TYRE"
                color = "#E10600" if comp == "SOFT" else ("#FFE800" if comp == "MEDIUM" else ("#3ECF8E" if comp == "INTERMEDIATE" else "#E8ECEF"))

                fc_times = grp["LapTime_s"] - (np.maximum(110.0 - 1.7 * grp["LapNumber"], 2.0) * 0.03)
                early_subset = fc_times[(grp["TyreLife"] >= 2) & (grp["TyreLife"] <= 6)]
                if not early_subset.empty:
                    base_time = float(early_subset.quantile(0.20))
                    anchor_age = int(grp.loc[early_subset.idxmin(), "TyreLife"])
                else:
                    base_time = float(fc_times.min())
                    anchor_age = int(grp["TyreLife"].iloc[0])
                deg_observed = fc_times - base_time

                lin_pred, xgb_pred = predict_stint_degradation_curve(
                    ages=grp["TyreLife"].values,
                    laps=grp["LapNumber"].values,
                    compound_str=comp,
                    g3_model=g3_model,
                    circuit_physics=circuit_physics,
                    anchor_age=anchor_age,
                    is_stint1=(int(stint_id) == 1),
                    race_inc_bias=float(race_deg_stats.get("race_inc_bias", 0.0)),
                    race_alpha=float(race_deg_stats.get("race_alpha", 1.0)),
                )
                drv_lin_errs.extend(np.abs(deg_observed.values - lin_pred).tolist())
                drv_xgb_errs.extend(np.abs(deg_observed.values - xgb_pred).tolist())

                fig_deg.add_trace(go.Scatter(
                    x=grp["TyreLife"],
                    y=deg_observed,
                    mode="markers+lines",
                    name=f"S{int(stint_id)} {comp} Actual",
                    marker=dict(size=7, color=color),
                    line=dict(color=color, width=2.2),
                    hovertemplate="Actual Age %{x}: +%{y:.2f}s<extra></extra>",
                ))
                fig_deg.add_trace(go.Scatter(
                    x=grp["TyreLife"],
                    y=lin_pred,
                    mode="lines",
                    name=f"S{int(stint_id)} Linear Pred",
                    line=dict(color=color, width=1.8, dash="dash"),
                    hovertemplate="Linear Pred Age %{x}: +%{y:.2f}s<extra></extra>",
                ))
                fig_deg.add_trace(go.Scatter(
                    x=grp["TyreLife"],
                    y=xgb_pred,
                    mode="lines",
                    name=f"S{int(stint_id)} XGB Pred",
                    line=dict(color=color, width=1.8, dash="dot"),
                    hovertemplate="XGB Pred Age %{x}: +%{y:.2f}s<extra></extra>",
                ))

            fig_deg.update_layout(
                xaxis_title="Tyre Age (Laps on Tyre Set)",
                yaxis_title="Degradation Above Baseline (s)",
                height=470,
            )
            st.plotly_chart(
                style_chart(fig_deg, f"Predicted vs Actual Degradation ({selected_driver})"),
                width="stretch",
            )

            n_pairs = race_deg_stats.get("n_pairs", 0)
            lin_mae = race_deg_stats.get("linear_mae", np.nan)
            xgb_mae = race_deg_stats.get("xgb_mae", np.nan)
            lin_r2 = race_deg_stats.get("linear_r2", np.nan)
            xgb_r2 = race_deg_stats.get("xgb_r2", np.nan)
            drv_lin_mae = float(np.mean(drv_lin_errs)) if drv_lin_errs else np.nan
            drv_xgb_mae = float(np.mean(drv_xgb_errs)) if drv_xgb_errs else np.nan

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Extrapolation Model", "Smooth Physics + LODO EB")
            r2.metric(
                "Race 5-Lap Linear MAE",
                f"{lin_mae:.3f} s" if pd.notnull(lin_mae) else "N/A",
                delta=f"R² {lin_r2:+.3f} ({n_pairs} pairs)" if pd.notnull(lin_r2) else f"{n_pairs} clean pairs",
            )
            r3.metric(
                "Race 5-Lap XGB MAE",
                f"{xgb_mae:.3f} s" if pd.notnull(xgb_mae) else "N/A",
                delta=f"R² {xgb_r2:+.3f}" if pd.notnull(xgb_r2) else None,
            )
            r4.metric(
                f"{selected_driver} Stint MAE (Lin / XGB)",
                f"{drv_lin_mae:.3f}s / {drv_xgb_mae:.3f}s" if pd.notnull(drv_lin_mae) else "N/A",
            )

