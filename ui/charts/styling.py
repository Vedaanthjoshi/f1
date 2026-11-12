"""
ui/charts/styling.py — Dark Motorsport Plotly Chart Styling
===========================================================
Shared dark layout templates, font styling, and axis formatting for all charts.
"""

from __future__ import annotations
import plotly.graph_objects as go

DARK_CHART_LAYOUT: dict = {
    "paper_bgcolor": "rgba(10, 12, 16, 0.75)",
    "plot_bgcolor": "rgba(14, 18, 26, 0.58)",
    "font": {"family": "Titillium Web, Inter, Segoe UI, sans-serif", "color": "#F0F4F8", "size": 14},
    "margin": {"l": 72, "r": 34, "t": 78, "b": 64},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.06,
        "xanchor": "left",
        "x": 0,
        "font": {"family": "JetBrains Mono, monospace", "color": "#FFFFFF", "size": 11},
        "bgcolor": "rgba(10, 13, 20, 0.88)",
        "bordercolor": "rgba(255, 255, 255, 0.14)",
        "borderwidth": 1,
        "itemsizing": "constant",
        "title": {"text": ""},
    },
    "xaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.06)",
        "zerolinecolor": "rgba(255, 255, 255, 0.22)",
        "linecolor": "rgba(255, 255, 255, 0.30)",
        "tickfont": {"family": "JetBrains Mono, monospace", "color": "#AAB8C8", "size": 11},
        "title": {"font": {"family": "Titillium Web, sans-serif", "color": "#FFFFFF", "size": 14}},
        "showline": True,
        "linewidth": 1.5,
        "ticks": "outside",
        "ticklen": 5,
    },
    "yaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.06)",
        "zerolinecolor": "rgba(255, 255, 255, 0.22)",
        "linecolor": "rgba(255, 255, 255, 0.30)",
        "tickfont": {"family": "JetBrains Mono, monospace", "color": "#AAB8C8", "size": 11},
        "title": {"font": {"family": "Titillium Web, sans-serif", "color": "#FFFFFF", "size": 14}},
        "showline": True,
        "linewidth": 1.5,
        "ticks": "outside",
        "ticklen": 5,
    },
    "hoverlabel": {
        "bgcolor": "#08090C",
        "font": {"color": "#FFFFFF", "family": "JetBrains Mono, monospace", "size": 12},
        "bordercolor": "#E10600",
    },
}


def style_chart(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply shared dark motorsport chart styling without modifying analytical data."""
    fig.update_layout(template="plotly_dark")
    fig.update_layout(**DARK_CHART_LAYOUT)
    if title:
        fig.update_layout(title={"text": title, "font": {"size": 20, "color": "#FFFFFF"}})
    fig.update_xaxes(showline=True, mirror=False)
    fig.update_yaxes(showline=True, mirror=False)
    return fig

