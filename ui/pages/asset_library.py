"""Asset and metadata inspection page for Iteration 2."""

import pandas as pd
import streamlit as st

from ui.components import panel, state_message
from ui.metadata import cars, drivers, missing_assets, teams, tracks


def _table(records: list[dict], columns: list[str]) -> None:
    if records:
        st.dataframe(pd.DataFrame(records)[columns], width='stretch', hide_index=True)
    else:
        state_message("empty", "No Metadata", "No records were found for this metadata type.")


def render() -> None:
    panel(
        "Asset And Metadata System",
        "Local JSON metadata now maps FastF1-style driver codes and race names to teams, cars, tracks, and future local image files.",
    )

    missing = missing_assets()
    if missing:
        state_message(
            "info",
            f"{len(missing)} Image Files Awaiting Input",
            "These paths are configured for future local assets. Ask the user for files before filling them.",
        )
        st.dataframe(
            pd.DataFrame([item.__dict__ for item in missing]),
            width='stretch',
            hide_index=True,
        )
    else:
        state_message("success", "All Assets Found", "Every configured image path exists locally.")

    tab_drivers, tab_teams, tab_cars, tab_tracks = st.tabs(["Drivers", "Teams", "Cars", "Tracks"])
    with tab_drivers:
        _table(drivers(), ["code", "name", "number", "team", "country", "photo", "carId"])
    with tab_teams:
        _table(teams(), ["name", "shortName", "base", "accentColor", "secondaryColor"])
    with tab_cars:
        _table(cars(), ["id", "team", "name", "season", "engine", "image"])
    with tab_tracks:
        _table(tracks(), ["raceName", "country", "circuit", "laps", "lengthKm", "raceDistanceKm", "photo"])

