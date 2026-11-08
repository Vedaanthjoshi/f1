"""
ui/charts/shap_chart.py — SHAP Feature Importance Bar Chart
===========================================================
Visualizes model feature contributions:
  - Red (> 0): features that increase lap time delta (slowing car down, e.g. old tyre)
  - Green (< 0): features that decrease lap time delta (making car faster, e.g. track evo)
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts.styling import style_chart


def render_shap_chart(shap_df_plot: pd.DataFrame, sim_lap: int) -> None:
    """Render the SHAP feature importance horizontal bar chart."""
    st.divider()
    st.subheader(f"Decision Explainability (SHAP) — Lap {sim_lap}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=shap_df_plot["Label"],
        x=shap_df_plot["SHAP Value"],
        orientation='h',
        marker=dict(
            color=["#FF4D5E" if v > 0 else "#36D399" for v in shap_df_plot["SHAP Value"]],
            line=dict(width=1, color="rgba(255,255,255,0.2)"),
        ),
    ))

    fig.update_layout(
        title="What factors are driving your pace?",
        xaxis_title="Impact on predicted Lap Time (seconds)",
        yaxis_title="",
        margin=dict(l=150),
    )
    st.plotly_chart(style_chart(fig), width='stretch')
    st.info(
        "💡 **How to read SHAP**: Positive values (Red) mean the feature is making the lap time SLOWER "
        "(e.g. old tyres). Negative values (Green) mean the feature is making the car FASTER "
        "(e.g. track evolution or fresh soft tyres)."
    )

