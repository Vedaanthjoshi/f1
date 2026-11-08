"""
ui/components/event_strip.py — Event Summary Metric Strip
=========================================================
Displays responsive glassmorphic metric chips for Season, Grand Prix,
Driver Matchup, and Circuit Name.
"""

from html import escape
from textwrap import dedent
import streamlit as st

from ui.metadata import get_track


def render_event_strip(
    season: int,
    race: str,
    driver_a: str | None,
    driver_b: str | None,
) -> None:
    """Render the 4-column summary metric chip strip."""
    matchup = f"{driver_a or 'Driver A'} vs {driver_b or 'Driver B'}"
    track = get_track(race)
    circuit = track["circuit"] if track else "Circuit metadata pending"

    st.markdown(
        dedent(
            f"""
            <div class="nfs-strip">
                <div class="nfs-chip"><span>Season</span><strong>{escape(str(season))}</strong></div>
                <div class="nfs-chip"><span>Grand Prix</span><strong>{escape(race)}</strong></div>
                <div class="nfs-chip"><span>Matchup</span><strong>{escape(matchup)}</strong></div>
                <div class="nfs-chip"><span>Circuit</span><strong>{escape(circuit)}</strong></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

