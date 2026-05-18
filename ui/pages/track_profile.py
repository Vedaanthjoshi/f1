"""Track profile page placeholder for the redesigned UI."""

import streamlit as st

from ui.components import panel, state_message
from ui.metadata import get_track, tracks


def render() -> None:
    panel("Track Profile", "Future circuit panels will display race laps, length, pit loss, sector traits, and track photo.")
    race_names = [track["raceName"] for track in tracks()]
    if race_names:
        selected_race = st.selectbox("Metadata preview race", race_names, index=0)
        st.json(get_track(selected_race))
    else:
        state_message("empty", "No Tracks", "Add track records to data/tracks.json.")
    state_message(
        "info",
        "Use Provided Images Only",
        "Track imagery will come from user-provided files, not internet downloads.",
    )
