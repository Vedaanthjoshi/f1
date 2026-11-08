"""
ui/charts/gap_chart.py — Gap Evolution Line Visualization
=========================================================
Tracks on-track delta in seconds between Driver A (Target) and Driver B (Attacker)
across all mutually completed race laps.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts.styling import style_chart


def render_gap_chart(
    df_ab: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    driver_color_map: dict[str, str],
) -> pd.DataFrame:
    """
    Render on-track interval line chart.
    
    Returns:
    --------
    gap_df : pd.DataFrame containing LapNumber, Time_A, Time_B, Gap_Seconds
    """
    st.subheader("Gap Evolution")

    laps_a = df_ab[df_ab["Driver"] == driver_a][["LapNumber", "Time"]]
    laps_b = df_ab[df_ab["Driver"] == driver_b][["LapNumber", "Time"]]

    gap_df = pd.merge(laps_a, laps_b, on="LapNumber", suffixes=("_A", "_B"))
    gap_df["Gap_Seconds"] = (gap_df["Time_A"] - gap_df["Time_B"]).dt.total_seconds()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gap_df["LapNumber"],
            y=gap_df["Gap_Seconds"],
            mode="lines+markers",
            name=f"{driver_a} gap to {driver_b}",
            line=dict(color=driver_color_map[driver_a], width=4),
            marker=dict(size=6, color=driver_color_map[driver_a], line=dict(color="#FFFFFF", width=1)),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)")
    fig.update_layout(
        title=f"Time Gap ({driver_a} behind {driver_b})",
        xaxis_title="Race Lap",
        yaxis_title=f"Gap to {driver_b} (seconds)",
        legend_title_text="",
    )
    st.plotly_chart(style_chart(fig), width='stretch')
    return gap_df

