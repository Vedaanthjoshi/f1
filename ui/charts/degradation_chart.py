"""
ui/charts/degradation_chart.py — Tyre Degradation Scatter Visualization
======================================================================
Renders actual lap times for both selected drivers across all stints.
Y-axis is inverted (lower time = faster = higher up) to mirror telemetry standards.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.charts.styling import style_chart


def render_degradation_chart(
    df_ab: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    driver_color_map: dict[str, str],
) -> None:
    """Render the raw tyre pace degradation scatter plot."""
    st.subheader("Tyre Degradation (Raw Pace)")

    if df_ab.empty:
        st.warning("No clean racing laps are available for the selected driver pairing.")
        return

    fig = px.scatter(
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
    fig.update_yaxes(autorange="reversed")  # Inverted: faster lap times are higher
    fig.update_traces(
        marker=dict(size=8, opacity=0.9, line=dict(width=1.1, color="#FFFFFF")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(style_chart(fig), width='stretch')

