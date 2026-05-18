"""Shared visual theme constants for the redesigned Streamlit UI."""

APP_TITLE = "F1 Strategy Engineer"

COLORS = {
    "background": "#07090D",
    "surface": "#10141B",
    "surface_alt": "#161C25",
    "surface_high": "#202837",
    "border": "rgba(255, 255, 255, 0.12)",
    "text": "#F5F7FA",
    "muted": "#A7B0BE",
    "subtle": "#6F7A89",
    "red": "#E10600",
    "cyan": "#00D4FF",
    "yellow": "#FFD166",
    "green": "#36D399",
    "danger": "#FF4D5E",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0090FF",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Default": COLORS["red"],
}

CHART_TEMPLATE = "plotly_dark"

CHART_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": COLORS["text"]},
    "margin": {"l": 30, "r": 24, "t": 58, "b": 34},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.08)",
        "zerolinecolor": "rgba(255,255,255,0.16)",
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.08)",
        "zerolinecolor": "rgba(255,255,255,0.16)",
    },
}


def css() -> str:
    """Return global CSS for the redesigned shell."""
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --f1-bg: {COLORS["background"]};
            --f1-surface: {COLORS["surface"]};
            --f1-surface-alt: {COLORS["surface_alt"]};
            --f1-border: {COLORS["border"]};
            --f1-text: {COLORS["text"]};
            --f1-muted: {COLORS["muted"]};
            --f1-red: {COLORS["red"]};
            --f1-cyan: {COLORS["cyan"]};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(135deg, rgba(225, 6, 0, 0.13), transparent 32rem),
                radial-gradient(circle at 80% 12%, rgba(0, 212, 255, 0.12), transparent 28rem),
                var(--f1-bg);
            color: var(--f1-text);
            font-family: Inter, Segoe UI, sans-serif;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(10, 13, 19, 0.96);
            border-right: 1px solid var(--f1-border);
        }}

        .block-container {{
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}

        h1, h2, h3 {{
            letter-spacing: 0;
            color: var(--f1-text);
        }}

        .f1-shell-hero {{
            border: 1px solid var(--f1-border);
            background:
                linear-gradient(110deg, rgba(16, 20, 27, 0.96), rgba(16, 20, 27, 0.64)),
                linear-gradient(90deg, rgba(225, 6, 0, 0.20), transparent 48%);
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 22px 80px rgba(0, 0, 0, 0.36);
        }}

        .f1-kicker {{
            color: var(--f1-cyan);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .f1-title {{
            margin: 0.3rem 0 0.5rem;
            font-size: clamp(2.1rem, 4vw, 4.8rem);
            line-height: 0.95;
            font-weight: 800;
        }}

        .f1-copy {{
            color: var(--f1-muted);
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.65;
        }}

        .f1-panel {{
            border: 1px solid var(--f1-border);
            background: rgba(16, 20, 27, 0.88);
            border-radius: 8px;
            padding: 1.25rem;
        }}

        .f1-panel-title {{
            margin: 0 0 0.35rem;
            font-size: 1rem;
            font-weight: 800;
        }}

        .f1-panel-meta {{
            color: var(--f1-muted);
            font-size: 0.88rem;
        }}

        .f1-stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
            margin: 1rem 0;
        }}

        .f1-stat {{
            border: 1px solid var(--f1-border);
            background: rgba(255, 255, 255, 0.035);
            border-radius: 8px;
            padding: 0.9rem;
        }}

        .f1-stat-label {{
            color: var(--f1-muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .f1-stat-value {{
            margin-top: 0.25rem;
            font-size: 1.35rem;
            font-weight: 800;
        }}

        .stButton > button {{
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: linear-gradient(90deg, var(--f1-red), #ff3b30);
            color: white;
            font-weight: 800;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid var(--f1-border);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px 6px 0 0;
            color: var(--f1-muted);
            font-weight: 700;
        }}
    </style>
    """

