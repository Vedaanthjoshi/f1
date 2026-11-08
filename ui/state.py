"""
ui/state.py — Streamlit Session State & Synchronization Manager
===============================================================
Encapsulates two-way synchronized dropdown state between the sidebar rail
and the main page configuration cards, as well as race selection persistence.
"""

from __future__ import annotations
import streamlit as st


def init_session_state(default_season: int = 2023, default_race: str | None = None) -> None:
    """Initialize necessary Streamlit session_state keys with defaults."""
    if "sidebar_season" not in st.session_state:
        st.session_state.sidebar_season = default_season
        st.session_state.home_season = default_season

    if default_race:
        if "sidebar_race" not in st.session_state:
            st.session_state.sidebar_race = default_race
            st.session_state.home_race = default_race


def sync_season(source: str) -> None:
    """Synchronize Season selection between sidebar and main panel."""
    if source == 'sidebar':
        st.session_state.home_season = st.session_state.sidebar_season
    else:
        st.session_state.sidebar_season = st.session_state.home_season


def sync_race(source: str) -> None:
    """Synchronize Grand Prix selection between sidebar and main panel."""
    if source == 'sidebar':
        st.session_state.home_race = st.session_state.sidebar_race
    else:
        st.session_state.sidebar_race = st.session_state.home_race


def load_requested_race(year: int, race_name: str) -> None:
    """Set the active race and trigger a pipeline reload."""
    st.session_state["year"] = year
    st.session_state["race"] = race_name
    st.session_state["data_loaded"] = True

