"""Iteration 3 homepage components for the main Streamlit app."""

from __future__ import annotations

from html import escape
from textwrap import dedent

import streamlit as st

from ui.assets import image_data_uri
from ui.metadata import driver_profile, get_track

HOMEPAGE_IMAGE = "assets/home/homepage_hero.webp"


def homepage_css() -> str:
    hero_image = image_data_uri(HOMEPAGE_IMAGE)
    page_background = (
        f"linear-gradient(90deg, rgba(246, 247, 248, 0.62) 0%, rgba(246, 247, 248, 0.30) 34%, rgba(246, 247, 248, 0.38) 58%, rgba(246, 247, 248, 0.88) 82%, rgba(246, 247, 248, 0.94) 100%), "
        f"linear-gradient(180deg, rgba(246, 247, 248, 0.04) 0%, rgba(246, 247, 248, 0.38) 50%, rgba(246, 247, 248, 0.88) 78%, #f2f3f5 100%), "
        f"url('{hero_image}')"
        if hero_image
        else "linear-gradient(135deg, #f5f6f7, #dfe2e6)"
    )
    return dedent(
        f"""
        <style>
            :root {{
                --f1-red: #e10600;
                --f1-ink: #08090b;
                --f1-panel: rgba(255, 255, 255, 0.72);
                --f1-panel-strong: rgba(255, 255, 255, 0.88);
                --f1-line: rgba(8, 9, 11, 0.16);
                --f1-text: #101216;
                --f1-muted: #606873;
                --f1-soft: #f3f4f6;
                --f1-shadow: rgba(10, 11, 13, 0.14);
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                background-image:
                    repeating-linear-gradient(90deg, rgba(8, 9, 11, 0.035) 0 1px, transparent 1px 14px),
                    linear-gradient(180deg, transparent 0 50vh, rgba(242, 243, 245, 0.82) 72vh, #f2f3f5 100%),
                    {page_background};
                background-size: cover;
                background-position: 34% top;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: var(--f1-text);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="collapsedControl"] button {{
                background: transparent;
                border: 0;
                box-shadow: none;
                position: relative;
            }}

            [data-testid="collapsedControl"] button,
            [data-testid="collapsedControl"] button *,
            [data-testid="collapsedControl"] button svg,
            [data-testid="collapsedControl"] button svg *,
            [data-testid="collapsedControl"] button path {{
                color: #08090b;
                fill: #08090b;
                stroke: #08090b;
                opacity: 1;
            }}

            [data-testid="collapsedControl"] button svg {{
                opacity: 0;
            }}

            [data-testid="collapsedControl"] button::after {{
                content: ">>";
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #08090b;
                font-size: 1.35rem;
                font-weight: 950;
                line-height: 1;
                opacity: 1;
                pointer-events: none;
            }}

            [data-testid="collapsedControl"] button:hover {{
                background: transparent;
            }}

            [data-testid="collapsedControl"] button:hover,
            [data-testid="collapsedControl"] button:hover *,
            [data-testid="collapsedControl"] button:hover svg,
            [data-testid="collapsedControl"] button:hover path {{
                color: #08090b;
                fill: #08090b;
                stroke: #08090b;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(255, 255, 255, 0.94);
                border-right: 2px solid rgba(8, 9, 11, 0.22);
                box-shadow: 24px 0 70px rgba(8, 9, 11, 0.16);
                backdrop-filter: blur(26px);
            }}

            [data-testid="stSidebar"]::before {{
                content: "";
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(90deg, rgba(225, 6, 0, 0.16), transparent 13%),
                    repeating-linear-gradient(135deg, rgba(8, 9, 11, 0.052) 0 1px, transparent 1px 11px);
                pointer-events: none;
            }}

            [data-testid="stSidebar"] * {{
                color: var(--f1-text);
            }}

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label {{
                color: #08090b;
                opacity: 1;
            }}

            .block-container {{
                max-width: 1500px;
                padding-top: 0;
                padding-bottom: 4rem;
            }}

            h1, h2, h3, p {{
                letter-spacing: 0;
            }}

            h2, h3, h4, [data-testid="stMarkdownContainer"] p {{
                color: var(--f1-text);
            }}

            .nfs-hero {{
                min-height: 60vh;
                border: 0;
                border-radius: 0;
                background: transparent;
                padding: clamp(1.4rem, 4vw, 3.2rem);
                display: flex;
                align-items: flex-end;
                box-shadow: none;
                position: relative;
                margin-inline: calc(-1 * clamp(0rem, 2vw, 1.8rem));
            }}

            .nfs-hero::after {{
                display: none;
            }}

            .nfs-hero-content {{
                position: relative;
                z-index: 1;
                max-width: 570px;
                margin-left: auto;
                padding: 1.1rem 1.25rem 1.25rem;
                background: rgba(255, 255, 255, 0.62);
                border: 1px solid rgba(8, 9, 11, 0.14);
                border-radius: 8px;
                backdrop-filter: blur(14px);
                box-shadow: 0 22px 70px rgba(8, 9, 11, 0.12);
            }}

            .nfs-kicker {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.34rem 0.62rem;
                border: 1px solid rgba(8, 9, 11, 0.26);
                background: rgba(255, 255, 255, 0.88);
                color: var(--f1-red);
                font-size: 0.78rem;
                font-weight: 900;
                text-transform: uppercase;
                backdrop-filter: blur(14px);
            }}

            .nfs-title {{
                margin: 0.85rem 0 0.8rem;
                color: var(--f1-ink);
                font-size: clamp(2.35rem, 4.7vw, 4.55rem);
                line-height: 0.86;
                font-weight: 950;
                text-transform: uppercase;
                text-shadow: 0 2px 0 rgba(255,255,255,0.55);
            }}

            .nfs-copy {{
                max-width: 640px;
                color: #1a1d22;
                font-size: clamp(0.96rem, 1.4vw, 1.16rem);
                line-height: 1.65;
                font-weight: 600;
            }}

            .nfs-strip {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.65rem;
                margin: 0.8rem 0 1.2rem;
            }}

            .nfs-chip, .nfs-card {{
                border: 1px solid var(--f1-line);
                background: rgba(255, 255, 255, 0.86);
                border-radius: 8px;
                backdrop-filter: blur(18px);
                box-shadow: 0 18px 55px var(--f1-shadow);
            }}

            .nfs-chip {{
                padding: 0.8rem;
            }}

            .nfs-chip span {{
                display: block;
                color: var(--f1-muted);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
            }}

            .nfs-chip strong {{
                display: block;
                margin-top: 0.2rem;
                color: var(--f1-text);
                font-size: 1.05rem;
            }}

            .nfs-card {{
                padding: 1rem;
                min-height: 160px;
            }}

            .nfs-card-title {{
                margin: 0;
                color: var(--f1-text);
                font-size: 1.05rem;
                font-weight: 900;
                text-transform: uppercase;
            }}

            .nfs-card-meta {{
                margin-top: 0.42rem;
                color: var(--f1-muted);
                font-size: 0.9rem;
                line-height: 1.5;
            }}

            .nfs-driver {{
                border-left: 5px solid var(--accent, var(--f1-red));
            }}

            .nfs-vs {{
                display: flex;
                min-height: 160px;
                align-items: center;
                justify-content: center;
                color: var(--f1-red);
                font-size: clamp(2rem, 4vw, 4rem);
                font-weight: 950;
                text-transform: uppercase;
            }}

            .stButton > button {{
                min-height: 3rem;
                border-radius: 4px;
                border: 1px solid rgba(8, 9, 11, 0.24);
                background: linear-gradient(90deg, #e10600, #ff2e2e);
                color: #fff;
                font-weight: 900;
                text-transform: uppercase;
                box-shadow: 0 12px 30px rgba(225, 6, 0, 0.22);
            }}

            .stButton > button *,
            .stButton > button p {{
                color: #fff;
            }}

            .stButton > button:hover {{
                border-color: rgba(8, 9, 11, 0.34);
                filter: brightness(1.02);
            }}

            [data-testid="stSelectbox"] label {{
                color: var(--f1-muted);
                font-weight: 800;
                text-transform: uppercase;
                font-size: 0.76rem;
            }}

            [data-baseweb="select"] > div {{
                background: rgba(255, 255, 255, 0.92);
                border-color: rgba(8, 9, 11, 0.26);
                color: var(--f1-text);
                border-radius: 4px;
            }}

            [data-testid="stAlert"] {{
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(8, 9, 11, 0.14);
                color: var(--f1-text);
            }}

            [data-testid="stPlotlyChart"] {{
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(8, 9, 11, 0.18);
                border-radius: 8px;
                padding: 0.75rem;
                box-shadow: 0 18px 55px rgba(8, 9, 11, 0.12);
            }}

            [data-testid="stPlotlyChart"] svg text {{
                fill: #08090b;
            }}

            @media (max-width: 900px) {{
                .nfs-strip {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
                .nfs-hero {{
                    min-height: 54vh;
                }}
            }}
        </style>
        """
    )


def apply_homepage_theme() -> None:
    st.markdown(homepage_css(), unsafe_allow_html=True)


def render_homepage_hero() -> None:
    st.markdown(
        dedent(
            """
            <section class="nfs-hero">
                <div class="nfs-hero-content">
                    <div class="nfs-kicker">Live race data / ML strategy model</div>
                    <h1 class="nfs-title">F1 Strategy Engineer</h1>
                    <p class="nfs-copy">
                        Set up a Grand Prix, choose the tactical matchup, and load the race engine to expose tyre degradation,
                        gap evolution, and the exact lap where the undercut becomes a threat.
                    </p>
                </div>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )


def render_event_strip(season: int, race: str, driver_a: str | None, driver_b: str | None) -> None:
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


def _driver_card(code: str | None, label: str) -> str:
    if not code:
        return (
            '<div class="nfs-card nfs-driver">'
            f'<h3 class="nfs-card-title">{escape(label)}</h3>'
            '<div class="nfs-card-meta">Select a driver after race data is loaded.</div>'
            '</div>'
        )

    profile = driver_profile(code)
    driver = profile["driver"]
    team = profile["team"]
    car = profile["car"]
    accent = team.get("accentColor", "#E10600")
    number = f"#{driver['number']}" if driver.get("number") else "No number"
    photo_state = "Ready" if profile["driverPhotoPath"] else "Awaiting local file"
    return (
        f'<div class="nfs-card nfs-driver" style="--accent: {escape(accent)};">'
        f'<h3 class="nfs-card-title">{escape(label)} / {escape(driver.get("code", code))}</h3>'
        '<div class="nfs-card-meta">'
        f'<strong>{escape(driver.get("name", code))}</strong><br>'
        f'{escape(number)} / {escape(team.get("name", "Unknown Team"))}<br>'
        f'Car: {escape(car.get("name", "Unknown Car"))}<br>'
        f'Photo: {photo_state}'
        '</div>'
        '</div>'
    )


def render_matchup_preview(driver_a: str | None, driver_b: str | None, race: str) -> None:
    track = get_track(race)
    track_body = (
        f"{track['circuit']} · {track['country']}<br>{track['laps']} laps · {track['lengthKm']} km"
        if track
        else "Metadata fallback active. Add this circuit to data/tracks.json for richer context."
    )
    st.markdown(
        (
            '<div class="nfs-strip" style="grid-template-columns: 1fr 0.35fr 1fr 1fr;">'
            f'{_driver_card(driver_a, "Target")}'
            '<div class="nfs-card nfs-vs">VS</div>'
            f'{_driver_card(driver_b, "Attacker")}'
            '<div class="nfs-card">'
            '<h3 class="nfs-card-title">Track Preview</h3>'
            f'<div class="nfs-card-meta">{track_body}</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
