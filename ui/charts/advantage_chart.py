"""
ui/charts/advantage_chart.py — Net Undercut Advantage Curve
===========================================================
Plots cumulative seconds gained (+) or lost (-) across the forward horizon
for a selected pit strategy compared to staying out on current tyres.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts.styling import style_chart


def render_advantage_chart(
    chart_strat: dict[str, Any],
    stay_out_row: dict[str, Any],
    sim_lap: int,
    compound_name: str,
) -> None:
    """Render the net undercut advantage line chart and decision status."""
    so_times = stay_out_row["p50_lap_times"]
    pit_times = chart_strat["p50_lap_times"]
    horizon_laps_plot = list(range(sim_lap, sim_lap + len(so_times)))

    lap_advantage = [so_times[i] - pit_times[i] for i in range(len(so_times))]
    net_undercut_advantage = list(pd.Series(lap_advantage).cumsum())

    sim_plot_df = pd.DataFrame({
        "Lap":           horizon_laps_plot,
        "Net Advantage": net_undercut_advantage,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_plot_df["Lap"],
        y=sim_plot_df["Net Advantage"],
        mode="lines+markers",
        name="Net Time Gained vs Stay Out (p50)",
        line=dict(color="#E10600", width=3),
        marker=dict(size=8, color="#E10600", line=dict(width=1, color="white")),
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.4)")

    if chart_strat.get("crossover_lap"):
        fig.add_vline(
            x=chart_strat["crossover_lap"],
            line_dash="dash",
            line_color="#FFD166",
            annotation_text="Tyres Become Slower",
        )

    p_gain_pct = f"{chart_strat['p_net_gain']:.0%}"
    fig.update_layout(
        title=f"Pit → {compound_name}: Net Cumulative Advantage (p50 median) — {p_gain_pct} P(gain)",
        xaxis_title="Race Lap",
        yaxis_title="Seconds Gained (+) or Lost (−)",
        legend_title_text="",
    )
    st.plotly_chart(style_chart(fig), width='stretch')

    # Decision verdict banner
    if chart_strat.get("window_open"):
        st.success(
            f"UNDERCUT WINDOW OPEN — Pit → {compound_name}: "
            f"{chart_strat['p_net_gain']:.0%} confidence, expected +{chart_strat['expected_gain_s']:.2f}s "
            f"(range {chart_strat['gain_ci_low']:+.1f}s to {chart_strat['gain_ci_high']:+.1f}s over 15 laps)"
        )
    elif chart_strat.get("crossover_lap"):
        st.info(f"Pace crossover at Lap {chart_strat['crossover_lap']}. {p_gain_pct} probability pitting is net positive.")
    else:
        st.error("Window firmly closed. Fresh tyres do not recover the pit-lane cost within the 15-lap horizon.")

