"""
ui/theme.py — Backward-compatible wrapper for UI theme constants
"""

from ui.styles.theme import COLORS, TEAM_COLORS, TYRE_COLORS
from ui.styles.loader import load_stylesheet
from ui.charts.styling import DARK_CHART_LAYOUT as CHART_LAYOUT

APP_TITLE = "F1 Strategy Engineer"
CHART_TEMPLATE = "plotly_dark"


def css() -> str:
    """Return global CSS string for backward compatibility."""
    return f"<style>\n{load_stylesheet()}\n</style>"
