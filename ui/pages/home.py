"""Homepage module for the future visual redesign."""

import streamlit as st

from ui.components import hero, panel, stat_grid, state_message


def render() -> None:
    """Render the Iteration 1 homepage shell without external images."""
    hero(
        "Race Intelligence Platform",
        "F1 Strategy Engineer",
        "A cinematic strategy workspace for comparing drivers, reading tyre degradation, and judging the undercut window from real Formula 1 race data.",
    )

    stat_grid(
        [
            ("Data Source", "FastF1"),
            ("Model", "Random Forest"),
            ("Simulation", "15 Laps"),
            ("Mode", "Two Drivers"),
        ]
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        panel("Race Setup", "Season, Grand Prix, and driver pairing controls live here in the future homepage flow.")
    with col_b:
        panel("Matchup Preview", "Driver photos, team colors, and car context will appear after local assets are provided.")
    with col_c:
        panel("Track Context", "Circuit metadata and pit-loss intelligence will connect the race to the strategy call.")

    state_message(
        "empty",
        "Ready For Race Data",
        "Load a race from the current app to train the model. This shell is isolated until you approve integration.",
    )

