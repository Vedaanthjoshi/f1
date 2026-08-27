"""Cinematic dark motorsport homepage components for the F1 Strategy Engineer app."""

from __future__ import annotations

from html import escape
from textwrap import dedent
from pathlib import Path

import streamlit as st

from ui.assets import image_data_uri
from ui.metadata import PROJECT_ROOT, driver_profile, get_track

HOMEPAGE_IMAGE = "assets/home/homepage_hero.webp"


def homepage_css(driver_a: str | None = None, driver_b: str | None = None) -> str:
    profile_a = driver_profile(driver_a) if driver_a else None
    profile_b = driver_profile(driver_b) if driver_b else None

    car_a_uri = None
    if profile_a and profile_a.get("carImagePath"):
        try:
            rel = str(Path(profile_a["carImagePath"]).relative_to(PROJECT_ROOT)).replace("\\", "/")
            car_a_uri = image_data_uri(rel)
        except Exception:
            pass

    car_b_uri = None
    if profile_b and profile_b.get("carImagePath"):
        try:
            rel = str(Path(profile_b["carImagePath"]).relative_to(PROJECT_ROOT)).replace("\\", "/")
            car_b_uri = image_data_uri(rel)
        except Exception:
            pass

    accent_a = profile_a["team"].get("accentColor", "#3671C6") if profile_a else "#3671C6"
    accent_b = profile_b["team"].get("accentColor", "#27F4D2") if profile_b else "#27F4D2"
    hero_image = image_data_uri(HOMEPAGE_IMAGE)

    if car_a_uri or car_b_uri:
        bg_layers = []
        bg_positions = []
        bg_sizes = []
        bg_repeats = []

        # Layer 1: Dark cinematic readability vignette (darker at bottom for charts/panels)
        bg_layers.append("linear-gradient(180deg, rgba(7, 9, 13, 0.45) 0%, rgba(7, 9, 13, 0.20) 25%, rgba(7, 9, 13, 0.65) 55%, #07090D 92%)")
        bg_positions.append("center top")
        bg_sizes.append("cover")
        bg_repeats.append("no-repeat")

        # Layer 2: Center glowing vertical laser / clash divider
        bg_layers.append("linear-gradient(112deg, transparent 49.3%, rgba(225, 6, 0, 0.35) 49.6%, rgba(255, 255, 255, 0.95) 50%, rgba(39, 244, 210, 0.35) 50.4%, transparent 50.7%)")
        bg_positions.append("center top")
        bg_sizes.append("cover")
        bg_repeats.append("no-repeat")

        # Layer 3: Left Car (Driver A) - large, prominent, unclipped
        if car_a_uri:
            bg_layers.append(f"url('{car_a_uri}')")
            bg_positions.append("left 0% top 14vh")
            bg_sizes.append("50vw auto")
            bg_repeats.append("no-repeat")

        # Layer 4: Right Car (Driver B) - large, prominent, unclipped
        if car_b_uri:
            bg_layers.append(f"url('{car_b_uri}')")
            bg_positions.append("right 0% top 14vh")
            bg_sizes.append("50vw auto")
            bg_repeats.append("no-repeat")

        # Layer 5: Left team ambient lighting & smoke glow
        bg_layers.append(f"radial-gradient(circle at 16% 32%, {accent_a}50 0%, {accent_a}10 40%, transparent 65%)")
        bg_positions.append("center top")
        bg_sizes.append("cover")
        bg_repeats.append("no-repeat")

        # Layer 6: Right team ambient lighting & smoke glow
        bg_layers.append(f"radial-gradient(circle at 84% 32%, {accent_b}50 0%, {accent_b}10 40%, transparent 65%)")
        bg_positions.append("center top")
        bg_sizes.append("cover")
        bg_repeats.append("no-repeat")

        # Layer 7: Deep obsidian dark base
        bg_layers.append("linear-gradient(135deg, #07090D 0%, #0D1117 100%)")
        bg_positions.append("center top")
        bg_sizes.append("cover")
        bg_repeats.append("no-repeat")

        page_bg_css = f"""
            background-color: #07090D;
            background-image: {', '.join(bg_layers)};
            background-position: {', '.join(bg_positions)};
            background-size: {', '.join(bg_sizes)};
            background-repeat: {', '.join(bg_repeats)};
            background-attachment: fixed;
        """
    else:
        page_bg_css = f"""
            background-color: #07090D;
            background-image:
                linear-gradient(180deg, rgba(7, 9, 13, 0.45) 0%, rgba(7, 9, 13, 0.75) 60%, #07090D 100%),
                url('{hero_image}' if hero_image else ''),
                linear-gradient(135deg, #07090D 0%, #0D1117 100%);
            background-size: cover;
            background-position: center top;
            background-repeat: no-repeat;
            background-attachment: fixed;
        """

    return dedent(
        f"""
        <style>
            :root {{
                --f1-red: #E10600;
                --f1-bg: #07090D;
                --f1-card-bg: rgba(13, 17, 24, 0.65);
                --f1-card-hover: rgba(18, 24, 34, 0.85);
                --f1-line: rgba(255, 255, 255, 0.12);
                --f1-line-strong: rgba(255, 255, 255, 0.22);
                --f1-text: #F0F4F8;
                --f1-muted: #8E9BAE;
                --f1-shadow: rgba(0, 0, 0, 0.55);
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                {page_bg_css}
                color: var(--f1-text);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="collapsedControl"] button {{
                background: rgba(12, 16, 24, 0.75);
                border: 1px solid var(--f1-line);
                border-radius: 6px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.4);
                position: relative;
            }}

            [data-testid="collapsedControl"] button,
            [data-testid="collapsedControl"] button *,
            [data-testid="collapsedControl"] button svg,
            [data-testid="collapsedControl"] button path {{
                color: #FFFFFF;
                fill: #FFFFFF;
                stroke: #FFFFFF;
                opacity: 1;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(9, 12, 18, 0.94);
                border-right: 1px solid var(--f1-line-strong);
                box-shadow: 20px 0 60px rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(28px);
            }}

            [data-testid="stSidebar"]::before {{
                content: "";
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(90deg, rgba(225, 6, 0, 0.12), transparent 15%),
                    repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0 1px, transparent 1px 12px);
                pointer-events: none;
            }}

            [data-testid="stSidebar"] * {{
                color: var(--f1-text);
            }}

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label {{
                color: #FFFFFF;
                opacity: 1;
            }}

            .block-container {{
                max-width: 1540px;
                padding-top: 0;
                padding-bottom: 4rem;
            }}

            h1, h2, h3, h4, [data-testid="stMarkdownContainer"] p {{
                color: var(--f1-text);
            }}

            /* --- FLOATING CINEMATIC HERO --- */
            .nfs-hero {{
                min-height: 48vh;
                border: 0;
                background: transparent;
                padding: clamp(1.5rem, 3.5vw, 3rem);
                display: flex;
                align-items: flex-end;
                box-shadow: none;
                position: relative;
                margin-inline: calc(-1 * clamp(0rem, 2vw, 1.8rem));
            }}

            .nfs-hero-content {{
                position: relative;
                z-index: 2;
                max-width: 580px;
                margin-left: auto;
                padding: 1.25rem 1.4rem;
                background: rgba(10, 14, 20, 0.68);
                border: 1px solid var(--f1-line);
                border-radius: 10px;
                backdrop-filter: blur(20px);
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.5);
            }}

            .nfs-kicker {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.32rem 0.65rem;
                border: 1px solid rgba(225, 6, 0, 0.4);
                background: rgba(225, 6, 0, 0.15);
                color: #FF4A4A;
                font-size: 0.76rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                border-radius: 4px;
            }}

            .nfs-title {{
                margin: 0.75rem 0 0.65rem;
                color: #FFFFFF;
                font-size: clamp(2.3rem, 4.5vw, 4.2rem);
                line-height: 0.88;
                font-weight: 950;
                text-transform: uppercase;
                text-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
            }}

            .nfs-copy {{
                max-width: 600px;
                color: #C2CBD6;
                font-size: clamp(0.94rem, 1.3vw, 1.08rem);
                line-height: 1.6;
                font-weight: 500;
            }}

            /* --- FLOATING METRIC STRIP --- */
            .nfs-strip {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.75rem;
                margin: 0.8rem 0 1.2rem;
            }}

            .nfs-chip {{
                border: 1px solid var(--f1-line);
                background: rgba(12, 16, 24, 0.65);
                border-radius: 8px;
                backdrop-filter: blur(20px);
                box-shadow: 0 14px 40px rgba(0, 0, 0, 0.4);
                padding: 0.85rem 1rem;
                transition: transform 0.2s ease, border-color 0.2s ease;
            }}

            .nfs-chip:hover {{
                transform: translateY(-2px);
                border-color: var(--f1-line-strong);
            }}

            .nfs-chip span {{
                display: block;
                color: var(--f1-muted);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}

            .nfs-chip strong {{
                display: block;
                margin-top: 0.25rem;
                color: #FFFFFF;
                font-size: 1.12rem;
                font-weight: 800;
            }}

            /* --- BUTTONS & CONTROLS --- */
            .stButton > button {{
                min-height: 3.1rem;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                background: linear-gradient(90deg, #E10600, #FF2E2E);
                color: #FFFFFF;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                box-shadow: 0 12px 32px rgba(225, 6, 0, 0.4);
                transition: transform 0.2s ease, filter 0.2s ease;
            }}

            .stButton > button *,
            .stButton > button p {{
                color: #FFFFFF !important;
            }}

            .stButton > button:hover {{
                border-color: #FFFFFF;
                filter: brightness(1.1);
                transform: translateY(-2px);
            }}

            [data-testid="stSelectbox"] label {{
                color: var(--f1-muted);
                font-weight: 800;
                text-transform: uppercase;
                font-size: 0.76rem;
                letter-spacing: 0.06em;
            }}

            [data-baseweb="select"] > div {{
                background: rgba(14, 18, 26, 0.85);
                border-color: var(--f1-line);
                color: #FFFFFF;
                border-radius: 6px;
            }}

            [data-testid="stAlert"] {{
                background: rgba(12, 16, 24, 0.80);
                border: 1px solid var(--f1-line);
                color: #FFFFFF;
                backdrop-filter: blur(16px);
            }}

            [data-testid="stPlotlyChart"] {{
                background: rgba(10, 14, 20, 0.72);
                border: 1px solid var(--f1-line);
                border-radius: 10px;
                padding: 0.85rem;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(20px);
            }}

            /* --- FLOATING CINEMATIC MATCHUP HUD --- */
            @keyframes hudSlideLeft {{
                0% {{ opacity: 0; transform: translateX(-35px); }}
                100% {{ opacity: 1; transform: translateX(0); }}
            }}

            @keyframes hudSlideRight {{
                0% {{ opacity: 0; transform: translateX(35px); }}
                100% {{ opacity: 1; transform: translateX(0); }}
            }}

            @keyframes vsGlowPulse {{
                0% {{
                    text-shadow: 0 0 12px rgba(225, 6, 0, 0.4), 0 0 24px rgba(255, 255, 255, 0.2);
                    transform: scale(0.96);
                }}
                100% {{
                    text-shadow: 0 0 24px rgba(225, 6, 0, 0.9), 0 0 40px rgba(255, 255, 255, 0.6);
                    transform: scale(1.05);
                }}
            }}

            .f1-cinematic-hud {{
                display: grid;
                grid-template-columns: minmax(290px, 1fr) 80px minmax(290px, 1fr) minmax(220px, 0.85fr);
                gap: 1rem;
                align-items: stretch;
                margin: 1rem 0 1.5rem;
            }}

            @media (max-width: 1100px) {{
                .f1-cinematic-hud {{
                    grid-template-columns: 1fr;
                }}
            }}

            .f1-hud-panel {{
                position: relative;
                background: rgba(10, 14, 22, 0.68);
                border: 1px solid var(--f1-line);
                border-radius: 12px;
                padding: 1.25rem 1.4rem;
                backdrop-filter: blur(24px);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.55);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            }}

            .f1-hud-panel:hover {{
                transform: translateY(-3px);
                border-color: var(--f1-line-strong);
                box-shadow: 0 26px 70px var(--accent-glow, rgba(0, 0, 0, 0.7));
            }}

            .f1-hud-panel.target-hud {{
                border-left: 5px solid var(--accent, #E10600);
                animation: hudSlideLeft 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
            }}

            .f1-hud-panel.attacker-hud {{
                border-right: 5px solid var(--accent, #27F4D2);
                animation: hudSlideRight 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
            }}

            .f1-hud-top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.65rem;
            }}

            .f1-hud-badge {{
                font-size: 0.70rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                padding: 0.28rem 0.6rem;
                border-radius: 4px;
                color: #FFFFFF;
                background: var(--accent, #E10600);
                box-shadow: 0 4px 14px var(--accent-glow, rgba(225, 6, 0, 0.4));
            }}

            .f1-hud-number {{
                font-size: 1.85rem;
                font-weight: 950;
                font-style: italic;
                color: #FFFFFF;
                opacity: 0.92;
                line-height: 1;
            }}

            .f1-hud-body {{
                display: flex;
                align-items: center;
                gap: 1rem;
                margin: 0.4rem 0;
            }}

            .f1-hud-avatar {{
                width: 68px;
                height: 68px;
                min-width: 68px;
                border-radius: 50%;
                overflow: hidden;
                border: 2.5px solid var(--accent, #E10600);
                box-shadow: 0 0 20px var(--accent-glow, rgba(225, 6, 0, 0.35));
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #10141C, #1E2532);
            }}

            .f1-hud-avatar img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: top center;
            }}

            .f1-hud-monogram {{
                font-size: 1.35rem;
                font-weight: 950;
                letter-spacing: 0.05em;
                color: #FFFFFF;
                text-shadow: 0 0 10px var(--accent, #E10600);
            }}

            .f1-hud-name {{
                font-size: 1.35rem;
                font-weight: 950;
                text-transform: uppercase;
                color: #FFFFFF;
                line-height: 1.1;
                margin: 0.1rem 0;
                letter-spacing: 0.02em;
            }}

            .f1-hud-team {{
                font-size: 0.85rem;
                font-weight: 700;
                color: var(--f1-muted);
            }}

            .f1-hud-footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 0.65rem;
                padding-top: 0.55rem;
                border-top: 1px solid var(--f1-line);
                font-size: 0.76rem;
                color: var(--f1-muted);
                font-weight: 700;
            }}

            .f1-hud-vs {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 130px;
                user-select: none;
            }}

            .f1-hud-vs-title {{
                font-size: 2.2rem;
                font-weight: 950;
                font-style: italic;
                letter-spacing: 0.08em;
                color: #FFFFFF;
                line-height: 1;
                margin: 0.2rem 0;
                animation: vsGlowPulse 2.2s ease-in-out infinite alternate;
            }}

            .f1-hud-vs-slash {{
                font-size: 1rem;
                font-weight: 950;
                letter-spacing: 0.16em;
                color: var(--f1-red, #E10600);
            }}
        </style>
        """
    )


def apply_homepage_theme(driver_a: str | None = None, driver_b: str | None = None) -> None:
    st.markdown(homepage_css(driver_a, driver_b), unsafe_allow_html=True)


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


def _driver_card(code: str | None, label: str, role_type: str = "target") -> str:
    if not code:
        return (
            f'<div class="f1-hud-panel {role_type}-hud">'
            f'<div class="f1-hud-top"><span class="f1-hud-badge">{escape(label)}</span></div>'
            '<div class="f1-hud-body"><div class="f1-hud-team">Select a driver after telemetry loads.</div></div>'
            '</div>'
        )

    profile = driver_profile(code)
    driver = profile["driver"]
    team = profile["team"]
    car = profile["car"]
    accent = team.get("accentColor", "#E10600")
    number = f"#{driver['number']}" if driver.get("number") else ""
    country = driver.get("country", "")

    # Resolve photo Data URI if available
    photo_uri = None
    if profile.get("driverPhotoPath"):
        try:
            rel = str(Path(profile["driverPhotoPath"]).relative_to(PROJECT_ROOT)).replace("\\", "/")
            photo_uri = image_data_uri(rel)
        except Exception:
            photo_uri = None

    if not photo_uri:
        photo_uri = image_data_uri(f"assets/drivers/{code.upper()}.png")

    if photo_uri:
        avatar_html = f'<img src="{photo_uri}" alt="{escape(code)}" />'
    else:
        avatar_html = f'<span class="f1-hud-monogram">{escape(code[:3].upper())}</span>'

    role_badge_label = "ATTACKER // TARGET" if role_type == "target" else "DEFENDER // CHASER"

    return (
        f'<div class="f1-hud-panel {role_type}-hud" style="--accent: {escape(accent)}; --accent-glow: {escape(accent)}44;">'
        f'  <div class="f1-hud-top">'
        f'    <span class="f1-hud-badge">{escape(role_badge_label)}</span>'
        f'    <span class="f1-hud-number">{escape(number)}</span>'
        f'  </div>'
        f'  <div class="f1-hud-body">'
        f'    <div class="f1-hud-avatar">{avatar_html}</div>'
        f'    <div>'
        f'      <div class="f1-hud-name">{escape(driver.get("name", code))}</div>'
        f'      <div class="f1-hud-team">{escape(team.get("name", "Unknown Team"))}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="f1-hud-footer">'
        f'    <span>🏎️ {escape(car.get("name", "F1 Car"))}</span>'
        f'    <span>📍 {escape(country)}</span>'
        f'  </div>'
        f'</div>'
    )


def render_matchup_preview(driver_a: str | None, driver_b: str | None, race: str) -> None:
    profile_a = driver_profile(driver_a) if driver_a else None
    profile_b = driver_profile(driver_b) if driver_b else None

    accent_a = profile_a["team"].get("accentColor", "#3671C6") if profile_a else "#3671C6"
    accent_b = profile_b["team"].get("accentColor", "#27F4D2") if profile_b else "#27F4D2"

    track = get_track(race)
    track_circuit = track["circuit"] if track else race
    track_country = track.get("country", "Grand Prix") if track else "F1 Circuit"
    track_laps = f"{track['laps']} Laps" if track and track.get("laps") else "Race Distance"
    track_len = f"{track['lengthKm']} km" if track and track.get("lengthKm") else ""

    st.markdown(
        (
            '<div class="f1-cinematic-hud">'
            f'{_driver_card(driver_a, "Attacker", "target")}'
            '  <div class="f1-hud-vs">'
            '    <span class="f1-hud-vs-slash">//</span>'
            '    <span class="f1-hud-vs-title">VS</span>'
            '    <span class="f1-hud-vs-slash">//</span>'
            '  </div>'
            f'{_driver_card(driver_b, "Defender", "attacker")}'
            '  <div class="f1-hud-panel">'
            '    <div class="f1-hud-top">'
            '      <span class="f1-hud-badge" style="background: rgba(255,255,255,0.12); box-shadow: none;">CIRCUIT PROFILE</span>'
            '    </div>'
            '    <div class="f1-hud-body" style="flex-direction: column; align-items: flex-start; gap: 0.25rem;">'
            f'      <div class="f1-hud-name" style="font-size: 1.15rem;">{escape(track_circuit)}</div>'
            f'      <div class="f1-hud-team">{escape(track_country)}</div>'
            '    </div>'
            '    <div class="f1-hud-footer">'
            f'      <span>🏁 {escape(track_laps)}</span>'
            f'      <span>📏 {escape(track_len)}</span>'
            '    </div>'
            '  </div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

