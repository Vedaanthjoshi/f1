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

import os
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


def render_race_review():
    """Main render function for the Post-Race Review tab."""
    st.markdown("## 📋 Post-Race Strategy Review & Counterfactual Engine")
    st.markdown(
        "Reconstruct actual race telemetry, evaluate pit-stop timing regret, run Monte Carlo what-ifs "
        "with rival cover & dirty-air reactions, and inspect model-predicted vs actual tyre degradation."
    )

    # -----------------------------------------------------------------------
    # 1. Race & Driver Selectors (All Cached Seasons 2019–2025)
    # -----------------------------------------------------------------------
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

    # Honesty Banner
    is_bahrain = "bahrain" in gp_name.lower()
    if is_bahrain:
        st.info(
            "🛡️ **Leave-One-Race-Out Model Active:** All Bahrain GPs were strictly removed from training data "
            "(`models/loro_bahrain.joblib`). Pit loss & residuals are computed directly from this race."
        )
    elif gp_name in TRAINING_SET_RACES and year in (2022, 2023, 2024, 2025):
        st.warning(
            "⚠️ **In-Sample Sanity Check:** The global model saw 2022–2025 laps from this circuit during training. "
            "Pit loss, circuit physics, and residuals are computed dynamically for this race; "
            "monotonic wear constraints prevent curve memorization."
        )

    if rival_mode == "fixed":
        st.caption("ℹ️ Strategy Modeling Assumption: **Rivals held fixed (stopwatch baseline), no traffic or reaction model.**")
    elif rival_mode == "cover":
        st.caption("ℹ️ Strategy Modeling Assumption: **Threatened rivals cover undercuts (+1 lap pit reaction); no dirty-air penalty.**")
    else:
        st.caption("ℹ️ Strategy Modeling Assumption: **Threatened rivals cover undercuts (+1 lap) + dirty-air wake penalty in traffic.**")

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

    drivers = sorted(replay_table["Driver"].unique().tolist())
    def_driver_idx = drivers.index("VER") if "VER" in drivers else 0
    selected_driver = st.selectbox("Select Driver to Review", drivers, index=def_driver_idx, key="review_driver")

    driver_laps = replay_table[replay_table["Driver"] == selected_driver].sort_values("LapNumber").reset_index(drop=True)
    if driver_laps.empty:
        st.warning(f"No laps recorded for driver {selected_driver}.")
        return

    # Driver Stops
    driver_stops = driver_laps[driver_laps["is_in_lap"] == True]
    pit_laps = driver_stops["LapNumber"].astype(int).tolist()

    # Driver KPI Header
    final_pos = int(driver_laps["Position"].dropna().iloc[-1]) if not driver_laps["Position"].dropna().empty else 20
    total_time_m = (driver_laps["Time_s"].dropna().iloc[-1] / 60.0) if not driver_laps["Time_s"].dropna().empty else 0.0
    stints_run = int(driver_laps["Stint"].dropna().max()) if not driver_laps["Stint"].dropna().empty else 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Position", f"P{final_pos}")
    m2.metric("Total Laps Completed", f"{len(driver_laps)}")
    m3.metric("Pit Stops Completed", f"{len(pit_laps)}")
    m4.metric("Stints", f"{stints_run}")

    st.divider()

    # -----------------------------------------------------------------------
    # Sub-Tabs for Review
    # -----------------------------------------------------------------------
    tab_regret, tab_whatif, tab_replay, tab_deg = st.tabs([
        "🔍 Pit Decision Regret Scan",
        "🎲 Monte Carlo What-If Simulator",
        "📊 Lap-by-Lap Replay Table",
        "📈 Tyre Degradation (Pred vs Actual)",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Regret Scan
    # -----------------------------------------------------------------------
    with tab_regret:
        st.subheader(f"Pit Stop Regret Analysis ({SHIFT_MIN} to +{SHIFT_MAX} Laps)")
        st.markdown(
            f"For each actual pit stop, evaluate counterfactual shifts between **{SHIFT_MIN} laps earlier** and **+{SHIFT_MAX} laps later** "
            f"under **{RIVAL_MODES[rival_mode]}**. "
            "Any shift whose 90% confidence interval covers 0.0s is labeled **not distinguishable from noise**."
        )

        if not pit_laps:
            st.info(f"{selected_driver} did not make any pit stops during this race.")
        else:
            stop_labels = [f"Stop {i+1} (Lap {p})" for i, p in enumerate(pit_laps)]
            chosen_stop_idx = st.selectbox("Select Pit Stop to Scan", range(len(pit_laps)), format_func=lambda i: stop_labels[i])

            actual_stop_lap = pit_laps[chosen_stop_idx]
            st.write(f"Analyzing **Stop {chosen_stop_idx + 1}** taken at **Lap {actual_stop_lap}**")

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
                # Add human readable labels
                scan_df["Status"] = scan_df.apply(
                    lambda r: "Not distinguishable from noise" if r["is_noise"]
                    else ("Faster (Advantage)" if r["expected_time_delta"] < 0 else "Slower (Loss)"),
                    axis=1,
                )
                scan_df["New In-Lap"] = actual_stop_lap + scan_df["shift"]

                # Regret Chart
                colors = scan_df["Status"].map({
                    "Faster (Advantage)": "#00D2BE",
                    "Not distinguishable from noise": "#9E9E9E",
                    "Slower (Loss)": "#E10600",
                })

                fig_regret = go.Figure()
                fig_regret.add_trace(go.Bar(
                    x=scan_df["shift"],
                    y=scan_df["expected_time_delta"],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=scan_df["p95"] - scan_df["expected_time_delta"],
                        arrayminus=scan_df["expected_time_delta"] - scan_df["p05"],
                        color="#B0BEC5",
                    ),
                    marker_color=colors,
                    name="Expected Delta",
                    hovertemplate="Shift: %{x:+d} laps<br>New Lap: %{customdata[0]}<br>Time Delta: %{y:+.2f} s<br>Status: %{customdata[1]}<extra></extra>",
                    customdata=scan_df[["New In-Lap", "Status"]].values,
                ))
                fig_regret.add_hline(y=0.0, line_dash="dash", line_color="#FFFFFF", opacity=0.7)
                fig_regret.update_layout(
                    title=f"Pit Regret Curve — Stop {chosen_stop_idx + 1} (Actual: Lap {actual_stop_lap})",
                    xaxis_title="Pit Stop Shift (Laps: Earlier [-] / Later [+])",
                    yaxis_title="Expected Race Time Delta (s) [Negative = Faster]",
                    template="plotly_dark",
                    height=450,
                )
                st.plotly_chart(fig_regret, width="stretch")

                # Regret Summary Table
                display_cols = [
                    "shift", "New In-Lap", "expected_time_delta", "p05", "p95",
                    "p_position_improves", "n_rival_reactions", "dirty_air_laps",
                    "blocked_laps", "is_extrapolated", "robustness", "Status",
                ]
                available_disp = [c for c in display_cols if c in scan_df.columns]
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
        st.subheader("Interactive Counterfactual Strategy Simulator")
        st.markdown(
            "Slide the stop window to simulate pitting earlier or later. "
            "The model replays 500 stochastic paths with block-bootstrapped degradation residuals and empirical pit loss sampling."
        )

        if not pit_laps:
            st.info("No pit stops available to simulate.")
        else:
            w_col1, w_col2, w_col3 = st.columns([1, 2, 1])
            w_stop_idx = w_col1.selectbox("Pit Stop to Alter", range(len(pit_laps)), format_func=lambda i: f"Stop {i+1} (Lap {pit_laps[i]})", key="whatif_stop_idx")
            w_shift = w_col2.slider("Shift Lap Count", min_value=SHIFT_MIN, max_value=SHIFT_MAX, value=0, step=1, key="whatif_shift_slider")
            w_paths = w_col3.select_slider("Rollout Paths", [200, 500, 1000], value=500, key="whatif_paths")

            w_actual_lap = pit_laps[w_stop_idx]
            w_new_lap = w_actual_lap + w_shift
            st.write(f"Evaluating strategy: Pit at **Lap {w_new_lap}** ({w_shift:+d} laps relative to actual Lap {w_actual_lap}) — **{RIVAL_MODES[rival_mode]}**")

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
                    rx_lines = [f"**{r['driver']}** boxed **Lap {r['reacted_pit']}** (instead of Lap {r['actual_pit']}) — {r['reason']}" for r in reactions]
                    st.info("🔄 **Rival Pit Wall Reactions:** " + " | ".join(rx_lines))
                elif rival_mode != "fixed" and w_shift != 0:
                    st.caption("No rival within the pit window needed to cover or counter this shift.")

                # Cumulative Time Delta Evolution Chart
                lap_numbers = driver_laps["LapNumber"].values
                delta_curve = sim_res["delta_per_lap"]

                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=lap_numbers,
                    y=delta_curve,
                    mode="lines",
                    line=dict(color="#FFD700", width=3),
                    name="Counterfactual Cumulative Delta",
                    hovertemplate="Lap %{x}: %{y:+.2f}s<extra></extra>",
                ))
                fig_cum.add_vline(x=w_actual_lap, line_dash="dash", line_color="#E10600", annotation_text="Actual Pit")
                if w_shift != 0:
                    fig_cum.add_vline(x=w_new_lap, line_dash="dash", line_color="#00D2BE", annotation_text="New Pit")
                fig_cum.add_hline(y=0.0, line_dash="dot", line_color="#888888")
                fig_cum.update_layout(
                    title=f"Cumulative Race Time Delta Over Race Laps ({selected_driver})",
                    xaxis_title="Lap Number",
                    yaxis_title="Cumulative Delta (s) [Negative = Ahead of Actual]",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig_cum, width="stretch")

                # Final Position Distribution
                dist = sim_res["position_distribution"]
                if len(dist) > 0 and dist.sum() > 0:
                    pos_indices = [i for i, p in enumerate(dist) if p > 0.005 and i > 0]
                    fig_dist = px.bar(
                        x=pos_indices,
                        y=[dist[i] for i in pos_indices],
                        labels={"x": "Finishing Position", "y": "Probability"},
                        title=f"Final Position Distribution over {w_paths} Monte Carlo Paths",
                        template="plotly_dark",
                    )
                    fig_dist.update_traces(marker_color="#00D2BE")
                    st.plotly_chart(fig_dist, width="stretch")

    # -----------------------------------------------------------------------
    # TAB 3: Lap-by-Lap Replay Table
    # -----------------------------------------------------------------------
    with tab_replay:
        st.subheader(f"Historical Race Replay — {selected_driver} ({year} {gp_name})")
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
            height=450,
            width="stretch",
        )

    # -----------------------------------------------------------------------
    # TAB 4: Tyre Degradation (Predicted vs Actual)
    # -----------------------------------------------------------------------
    with tab_deg:
        st.subheader("Tyre Degradation Analysis (Predicted vs Actual)")
        st.markdown(
            "Compares observed driver stint degradation against model predictions. "
            "Evaluates fuel-corrected lap-time progression vs tyre age."
        )

        clean_driver_laps = driver_laps[
            (driver_laps["is_in_lap"] == False)
            & (driver_laps["is_out_lap"] == False)
            & (driver_laps["is_neutralized"] == False)
            & (driver_laps["LapTime_s"].notnull())
        ].copy()

        if clean_driver_laps.empty:
            st.info("No clean green-flag laps available for degradation analysis.")
        else:
            # Stint-level Predicted vs Actual degradation plot
            fig_deg = go.Figure()
            drv_lin_errs: list[float] = []
            drv_xgb_errs: list[float] = []
            for stint_id, grp in clean_driver_laps.groupby("Stint"):
                grp = grp.sort_values("TyreLife")
                comp = grp["Compound"].iloc[0] if "Compound" in grp.columns else "TYRE"
                color = "#E10600" if comp == "SOFT" else ("#FFD700" if comp == "MEDIUM" else ("#00D2BE" if comp == "INTERMEDIATE" else "#FFFFFF"))

                # Fuel-corrected time relative to early stint baseline
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
                    name=f"Stint {int(stint_id)} ({comp}) Actual",
                    marker=dict(size=7, color=color),
                    line=dict(color=color, width=2),
                    hovertemplate="Actual Age %{x}: +%{y:.2f}s<extra></extra>",
                ))
                fig_deg.add_trace(go.Scatter(
                    x=grp["TyreLife"],
                    y=lin_pred,
                    mode="lines",
                    name=f"Stint {int(stint_id)} Controlled-Linear Pred",
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate="Linear Pred Age %{x}: +%{y:.2f}s<extra></extra>",
                ))
                fig_deg.add_trace(go.Scatter(
                    x=grp["TyreLife"],
                    y=xgb_pred,
                    mode="lines",
                    name=f"Stint {int(stint_id)} Monotonic XGB Pred",
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate="XGB Pred Age %{x}: +%{y:.2f}s<extra></extra>",
                ))

            fig_deg.update_layout(
                title=f"Predicted vs Actual Fuel-Corrected Degradation ({selected_driver})",
                xaxis_title="Tyre Age (Laps on Tyre Set)",
                yaxis_title="Degradation Above Stint Baseline (s)",
                template="plotly_dark",
                height=470,
            )
            st.plotly_chart(fig_deg, width="stretch")

            # Residual summary metrics (driver stint + full race)
            st.markdown(f"#### Degradation Model Error Statistics ({year} {gp_name})")
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
            st.caption(
                f"Pit loss pools for {year} {gp_name}: {len(pit_loss_green)} green stops "
                f"(median {np.median(pit_loss_green):.2f}s), {len(pit_loss_neutralized)} neutralized stops."
            )

