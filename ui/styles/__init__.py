"""
ui/styles package — Theme Constants & Stylesheet Injection
"""

from ui.styles.theme import COLORS, TEAM_COLORS, TYRE_COLORS
from ui.styles.loader import inject_custom_css, load_stylesheet

__all__ = [
    "COLORS",
    "TEAM_COLORS",
    "TYRE_COLORS",
    "inject_custom_css",
    "load_stylesheet",
]

