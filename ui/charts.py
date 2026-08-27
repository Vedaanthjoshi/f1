"""Plotly styling helpers for the redesigned cinematic UI."""

from __future__ import annotations

import plotly.graph_objects as go

DARK_CHART_LAYOUT = {
    "paper_bgcolor": "rgba(10, 12, 16, 0.70)",
    "plot_bgcolor": "rgba(16, 20, 28, 0.50)",
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#F0F4F8", "size": 14},
    "margin": {"l": 72, "r": 34, "t": 78, "b": 64},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.06,
        "xanchor": "left",
        "x": 0,
        "font": {"color": "#FFFFFF", "size": 12},
        "bgcolor": "rgba(12, 15, 20, 0.85)",
        "bordercolor": "rgba(255, 255, 255, 0.15)",
        "borderwidth": 1,
        "itemsizing": "constant",
        "title": {"text": ""},
    },
    "xaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.08)",
        "zerolinecolor": "rgba(255, 255, 255, 0.22)",
        "linecolor": "rgba(255, 255, 255, 0.35)",
        "tickfont": {"color": "#D0D8E0", "size": 12},
        "title": {"font": {"color": "#FFFFFF", "size": 14}},
        "showline": True,
        "linewidth": 1.5,
        "ticks": "outside",
        "ticklen": 6,
    },
    "yaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.08)",
        "zerolinecolor": "rgba(255, 255, 255, 0.22)",
        "linecolor": "rgba(255, 255, 255, 0.35)",
        "tickfont": {"color": "#D0D8E0", "size": 12},
        "title": {"font": {"color": "#FFFFFF", "size": 14}},
        "showline": True,
        "linewidth": 1.5,
        "ticks": "outside",
        "ticklen": 6,
    },
    "hoverlabel": {
        "bgcolor": "#08090C",
        "font": {"color": "#FFFFFF", "family": "Inter, Segoe UI, sans-serif"},
        "bordercolor": "#E10600",
    },
}


def style_chart(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply shared dark motorsport chart styling without changing the analytical data."""
    fig.update_layout(template="plotly_dark")
    fig.update_layout(**DARK_CHART_LAYOUT)
    if title:
        fig.update_layout(title={"text": title, "font": {"size": 20, "color": "#FFFFFF"}})
    fig.update_xaxes(showline=True, mirror=False)
    fig.update_yaxes(showline=True, mirror=False)
    return fig
