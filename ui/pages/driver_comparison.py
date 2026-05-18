"""Driver comparison page placeholder for the redesigned UI."""

import streamlit as st

from ui.components import panel, state_message
from ui.metadata import driver_profile, drivers


def render() -> None:
    panel("Driver Comparison", "Future driver cards will show photo, code, number, team, country, and car context.")
    codes = [driver["code"] for driver in drivers()]
    if codes:
        selected_code = st.selectbox("Metadata preview driver", codes, index=0)
        profile = driver_profile(selected_code)
        driver = profile["driver"]
        team = profile["team"]
        car = profile["car"]
        st.json(
            {
                "driver": driver,
                "team": team,
                "car": car,
                "driverPhotoFound": profile["driverPhotoPath"] is not None,
                "carImageFound": profile["carImagePath"] is not None,
            }
        )
    else:
        state_message("empty", "No Drivers", "Add driver records to data/drivers.json.")
    state_message(
        "empty",
        "Awaiting Metadata System",
        "Metadata is available now. Visual driver cards can be built in Iteration 4 after local images are provided.",
    )
