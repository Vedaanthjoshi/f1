"""
ui/components package — Reusable UI Presentation Components
"""

from ui.components.hero import render_homepage_hero
from ui.components.event_strip import render_event_strip
from ui.components.matchup_hud import render_matchup_preview
from ui.components.setup_cards import render_race_setup
from ui.components.strategy_table import render_strategy_table

__all__ = [
    "render_homepage_hero",
    "render_event_strip",
    "render_matchup_preview",
    "render_race_setup",
    "render_strategy_table",
]

