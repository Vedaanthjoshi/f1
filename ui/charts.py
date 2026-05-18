"""Plotly styling helpers for the redesigned UI."""

from __future__ import annotations

import plotly.graph_objects as go

LIGHT_CHART_LAYOUT = {
    "paper_bgcolor": "rgba(255,255,255,0)",
    "plot_bgcolor": "#F9FAFB",
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#08090B", "size": 14},
    "margin": {"l": 72, "r": 34, "t": 78, "b": 64},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.06,
        "xanchor": "left",
        "x": 0,
        "font": {"color": "#08090B", "size": 12},
        "bgcolor": "rgba(255,255,255,0.94)",
        "bordercolor": "rgba(8,9,11,0.18)",
        "borderwidth": 1,
        "itemsizing": "constant",
        "title": {"text": ""},
    },
    "xaxis": {
        "gridcolor": "rgba(8,9,11,0.16)",
        "zerolinecolor": "rgba(8,9,11,0.36)",
        "linecolor": "rgba(8,9,11,0.55)",
        "tickfont": {"color": "#08090B", "size": 12},
        "title": {"font": {"color": "#08090B", "size": 14}},
        "showline": True,
        "linewidth": 2,
        "ticks": "outside",
        "ticklen": 7,
    },
    "yaxis": {
        "gridcolor": "rgba(8,9,11,0.16)",
        "zerolinecolor": "rgba(8,9,11,0.36)",
        "linecolor": "rgba(8,9,11,0.55)",
        "tickfont": {"color": "#08090B", "size": 12},
        "title": {"font": {"color": "#08090B", "size": 14}},
        "showline": True,
        "linewidth": 2,
        "ticks": "outside",
        "ticklen": 7,
    },
    "hoverlabel": {
        "bgcolor": "#08090B",
        "font": {"color": "#FFFFFF", "family": "Inter, Segoe UI, sans-serif"},
        "bordercolor": "#E10600",
    },
}


def style_chart(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply shared chart styling without changing the analytical data."""
    fig.update_layout(template="plotly_white")
    fig.update_layout(**LIGHT_CHART_LAYOUT)
    if title:
        fig.update_layout(title={"text": title, "font": {"size": 20, "color": "#08090B"}})
    fig.update_xaxes(showline=True, mirror=False)
    fig.update_yaxes(showline=True, mirror=False)
    return fig
