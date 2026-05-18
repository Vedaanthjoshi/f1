"""Separate Streamlit entrypoint for the redesigned UI shell.

Run with:
    streamlit run ui_app.py

The current production prototype in app.py is intentionally untouched.
"""

import streamlit as st

from ui.components import apply_global_theme, navigation
from ui.pages import asset_library, driver_comparison, home, race_setup, strategy_analysis, track_profile
from ui.theme import APP_TITLE


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="F1")
apply_global_theme()

st.sidebar.caption("F1 Strategy Engineer")
page = navigation()

if page == "Home":
    home.render()
elif page == "Asset Library":
    asset_library.render()
elif page == "Race Setup":
    race_setup.render()
elif page == "Driver Comparison":
    driver_comparison.render()
elif page == "Track Profile":
    track_profile.render()
elif page == "Strategy Analysis":
    strategy_analysis.render()
