"""
ui/homepage.py — Backward-compatible wrapper for homepage components
"""

from ui.styles.loader import inject_custom_css as apply_homepage_theme
from ui.components.hero import render_homepage_hero
from ui.components.event_strip import render_event_strip
from ui.components.matchup_hud import render_matchup_preview, _driver_card

__all__ = [
    "apply_homepage_theme",
    "render_homepage_hero",
    "render_event_strip",
    "render_matchup_preview",
    "_driver_card",
]
