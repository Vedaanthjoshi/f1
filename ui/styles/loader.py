"""
ui/styles/loader.py — CSS Loader & Dynamic Style Injector
=========================================================
Reads static CSS from main.css and injects dynamic radial gradients
based on the active driver matchup team colors and hero background.
"""

from __future__ import annotations
from pathlib import Path
import streamlit as st

from ui.assets import image_data_uri
from ui.metadata import driver_profile

CSS_FILE_PATH = Path(__file__).resolve().parent / "main.css"


def load_stylesheet() -> str:
    """Read the base stylesheet from disk."""
    if CSS_FILE_PATH.exists():
        return CSS_FILE_PATH.read_text(encoding="utf-8")
    return ""


def inject_custom_css(driver_a: str | None = None, driver_b: str | None = None) -> None:
    """
    Inject global styles and dynamic team color glow background into the Streamlit app.
    """
    profile_a = driver_profile(driver_a) if driver_a else None
    profile_b = driver_profile(driver_b) if driver_b else None

    accent_a = profile_a["team"].get("accentColor", "#E10600") if profile_a else "#E10600"
    accent_b = profile_b["team"].get("accentColor", "#27F4D2") if profile_b else "#27F4D2"

    hero_image = image_data_uri("assets/home/new_hero_bg.png")

    dynamic_bg = f"""
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: #07090D;
        background-image:
            linear-gradient(180deg, rgba(7, 9, 13, 0.50) 0%, rgba(7, 9, 13, 0.92) 75%, #07090D 100%),
            radial-gradient(circle at 15% -10%, {accent_a}40 0%, transparent 45%),
            radial-gradient(circle at 85% -10%, {accent_b}40 0%, transparent 45%),
            radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
            url('{hero_image}'),
            linear-gradient(135deg, #07090D 0%, #11141A 100%);
        background-size: 100% 100%, 100% 100%, 100% 100%, 24px 24px, 100% auto, 100% 100%;
        background-position: center top;
        background-repeat: no-repeat, no-repeat, no-repeat, repeat, no-repeat, no-repeat;
        background-attachment: fixed;
    }}
    """

    base_css = load_stylesheet()
    st.markdown(f"<style>\n{base_css}\n{dynamic_bg}\n</style>", unsafe_allow_html=True)

