"""Cinematic dark motorsport homepage components for the F1 Strategy Engineer app."""

from __future__ import annotations

from html import escape
from textwrap import dedent
from pathlib import Path

import streamlit as st

from ui.assets import image_data_uri
from ui.metadata import PROJECT_ROOT, driver_profile, get_track

HOMEPAGE_IMAGE = "assets/home/homepage_hero.webp"


@st.cache_data(show_spinner=False)
def homepage_css(driver_a: str | None = None, driver_b: str | None = None) -> str:
    profile_a = driver_profile(driver_a) if driver_a else None
    profile_b = driver_profile(driver_b) if driver_b else None

    # ── Background styling ────────────────────────────────────────────────────
    # Premium F1 aesthetic: subtle team color glows + engineer dot-grid + user image
    accent_a = profile_a["team"].get("accentColor", "#E10600") if profile_a else "#E10600"
    accent_b = profile_b["team"].get("accentColor", "#27F4D2") if profile_b else "#27F4D2"
    
    # Load the new user-provided background image
    hero_image = image_data_uri("assets/home/new_hero_bg.png")

    page_bg_css = f"""
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
    """

    return dedent(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

            :root {{
                --f1-red:          #E10600;
                --f1-red-dim:      rgba(225, 6, 0, 0.28);
                --f1-bg:           #07090D;
                --f1-card-bg:      rgba(13, 17, 24, 0.78);
                --f1-card-hover:   rgba(18, 24, 34, 0.94);
                --f1-line:         rgba(255, 255, 255, 0.08);
                --f1-line-mid:     rgba(255, 255, 255, 0.13);
                --f1-line-strong:  rgba(255, 255, 255, 0.22);
                --f1-text:         #EDF2F7;
                --f1-muted:        #7A8BA0;
                --f1-muted-2:      #AAB8C8;
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                {page_bg_css}
                color: var(--f1-text);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            [data-testid="stHeader"] {{
                background: rgba(7,9,13,0.70);
                border-bottom: 1px solid var(--f1-line);
                backdrop-filter: blur(18px);
            }}

            [data-testid="collapsedControl"] button {{
                background: rgba(12, 16, 24, 0.80);
                border: 1px solid var(--f1-line-mid);
                border-radius: 7px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.4);
                position: relative;
            }}

            [data-testid="collapsedControl"] button,
            [data-testid="collapsedControl"] button *,
            [data-testid="collapsedControl"] button svg,
            [data-testid="collapsedControl"] button path {{
                color: #FFFFFF; fill: #FFFFFF; stroke: #FFFFFF; opacity: 1;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(7, 10, 17, 0.97);
                border-right: 1px solid var(--f1-line-mid);
                box-shadow: 18px 0 56px rgba(0,0,0,0.55);
                backdrop-filter: blur(32px);
            }}

            [data-testid="stSidebar"]::before {{
                content: "";
                position: absolute; inset: 0;
                background:
                    linear-gradient(180deg, rgba(225,6,0,0.06) 0%, transparent 18%),
                    repeating-linear-gradient(135deg, rgba(255,255,255,0.014) 0 1px, transparent 1px 14px);
                pointer-events: none;
            }}

            [data-testid="stSidebar"] * {{ color: var(--f1-text); }}

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {{
                font-size: 0.70rem; font-weight: 800;
                letter-spacing: 0.12em; text-transform: uppercase;
                color: var(--f1-muted-2) !important; opacity: 1 !important;
            }}

            [data-testid="stSidebar"] hr {{ border-color: var(--f1-line); margin: 0.65rem 0; }}

            .block-container {{
                max-width: 1540px;
                padding-top: 0.25rem;
                padding-bottom: 5rem;
            }}

            h1, h2, h3, h4 {{ color: var(--f1-text); font-family: 'Inter', sans-serif; }}

            [data-testid="stSelectbox"] label {{
                color: var(--f1-muted); font-weight: 700;
                font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.10em;
            }}

            [data-baseweb="select"] > div {{
                background: rgba(14, 18, 28, 0.90);
                border: 1px solid var(--f1-line-mid);
                border-radius: 7px; color: #FFF;
                transition: border-color 0.18s ease;
            }}

            [data-baseweb="select"] > div:hover,
            [data-baseweb="select"] > div:focus-within {{
                border-color: rgba(225, 6, 0, 0.50);
            }}

            .stButton > button {{
                min-height: 2.75rem; border-radius: 7px;
                border: 1px solid rgba(225,6,0,0.45);
                background: linear-gradient(135deg, #BE0000, #E10600);
                color: #FFF; font-family: 'Inter', sans-serif;
                font-weight: 800; font-size: 0.72rem;
                text-transform: uppercase; letter-spacing: 0.10em;
                box-shadow: 0 4px 18px rgba(225,6,0,0.28);
                transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            }}

            .stButton > button * {{ color: #FFF !important; }}
            .stButton > button p {{ color: #FFF !important; }}

            .stButton > button:hover {{
                filter: brightness(1.13);
                box-shadow: 0 6px 26px rgba(225,6,0,0.42);
                transform: translateY(-1px);
                border-color: rgba(255,255,255,0.30);
            }}

            .stButton > button:active {{ transform: translateY(0); }}

            [data-testid="stAlert"] {{
                background: rgba(12,16,24,0.88);
                border: 1px solid var(--f1-line-mid);
                color: #FFF; backdrop-filter: blur(16px); border-radius: 9px;
            }}

            [data-testid="stPlotlyChart"] {{
                background: rgba(10,13,20,0.82);
                border: 1px solid var(--f1-line-mid);
                border-radius: 12px; padding: 1rem 0.5rem 0.5rem;
                box-shadow: 0 6px 28px rgba(0,0,0,0.38);
                backdrop-filter: blur(18px);
                transition: border-color 0.22s ease;
            }}

            [data-testid="stPlotlyChart"]:hover {{ border-color: var(--f1-line-strong); }}

            /* --- HERO --- */
            .nfs-hero {{
                min-height: 42vh;
                padding: clamp(2.5rem, 4.5vw, 4.5rem) clamp(1.5rem, 3vw, 3rem);
                display: flex; align-items: flex-end; position: relative;
                margin-inline: calc(-1 * clamp(0rem, 2vw, 1.8rem));
                border-radius: 14px; margin-top: 0.25rem;
                background:
                    linear-gradient(135deg, rgba(225,6,0,0.09) 0%, rgba(225,6,0,0.02) 45%, transparent 70%),
                    linear-gradient(180deg, transparent 15%, rgba(7,9,13,0.80) 80%);
                border: 1px solid var(--f1-line); overflow: hidden;
            }}

            .nfs-hero::before {{
                content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
                background: linear-gradient(90deg, transparent, rgba(225,6,0,0.55) 38%, rgba(255,255,255,0.18) 55%, transparent);
            }}

            .nfs-hero::after {{
                content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
                background: linear-gradient(90deg, transparent, var(--f1-line-mid), transparent);
            }}

            .nfs-hero-content {{ position: relative; z-index: 2; max-width: 560px; }}

            .nfs-kicker {{
                display: inline-flex; align-items: center; gap: 0.55rem;
                padding: 0.28rem 0.75rem;
                border: 1px solid rgba(225,6,0,0.42);
                background: rgba(225,6,0,0.10); color: #FF5C5C;
                font-size: 0.68rem; font-weight: 800;
                text-transform: uppercase; letter-spacing: 0.14em;
                border-radius: 3px; margin-bottom: 1.1rem;
            }}

            .nfs-kicker::before {{
                content: ""; width: 5px; height: 5px;
                background: var(--f1-red); border-radius: 50%;
                animation: dot-blink 2.2s ease-in-out infinite;
            }}

            @keyframes dot-blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.22; }} }}

            .nfs-title {{
                margin: 0 0 0.85rem; color: #FFFFFF;
                font-size: clamp(2.8rem, 5.5vw, 5rem);
                line-height: 0.88; font-weight: 900;
                text-transform: uppercase; letter-spacing: -0.01em;
            }}

            .nfs-title em {{ font-style: normal; color: var(--f1-red); }}

            .nfs-copy {{
                color: var(--f1-muted-2); font-size: clamp(0.88rem, 1.15vw, 0.98rem);
                line-height: 1.70; font-weight: 400; max-width: 460px;
            }}

            /* --- METRIC STRIP --- */
            .nfs-strip {{
                display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
                gap: 0.6rem; margin: 0.9rem 0 1.4rem;
            }}

            .nfs-chip {{
                border: 1px solid var(--f1-line-mid);
                background: var(--f1-card-bg); border-radius: 9px;
                backdrop-filter: blur(18px); box-shadow: 0 3px 14px rgba(0,0,0,0.28);
                padding: 0.8rem 1rem; transition: border-color 0.2s ease, transform 0.2s ease;
            }}

            .nfs-chip:hover {{ border-color: var(--f1-line-strong); transform: translateY(-1px); }}
            .nfs-chip span {{ display: block; color: var(--f1-muted); font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 0.28rem; }}
            .nfs-chip strong {{ display: block; color: #FFF; font-size: 1.02rem; font-weight: 800; }}

            /* --- HUD ANIMATIONS --- */
            @keyframes hudSlideLeft  {{ from {{ opacity:0; transform:translateX(-22px); }} to {{ opacity:1; transform:none; }} }}
            @keyframes hudSlideRight {{ from {{ opacity:0; transform:translateX( 22px); }} to {{ opacity:1; transform:none; }} }}
            @keyframes vsGlowPulse   {{
                from {{ text-shadow: 0 0 10px rgba(225,6,0,0.3), 0 0 22px rgba(255,255,255,0.08); transform:scale(0.97); }}
                to   {{ text-shadow: 0 0 22px rgba(225,6,0,0.7), 0 0 38px rgba(255,255,255,0.32); transform:scale(1.04); }}
            }}

            .f1-cinematic-hud {{
                display: grid;
                grid-template-columns: minmax(275px, 1fr) 66px minmax(275px, 1fr) minmax(195px, 0.78fr);
                gap: 0.8rem; align-items: stretch; margin: 0.4rem 0 1.5rem;
            }}

            @media (max-width: 1100px) {{ .f1-cinematic-hud {{ grid-template-columns: 1fr; }} }}

            .f1-hud-panel {{
                position: relative; background: var(--f1-card-bg);
                border: 1px solid var(--f1-line); border-radius: 12px;
                padding: 1.1rem 1.25rem; backdrop-filter: blur(24px);
                box-shadow: 0 10px 36px rgba(0,0,0,0.42);
                display: flex; flex-direction: column; justify-content: space-between;
                transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            }}

            .f1-hud-panel:hover {{ transform: translateY(-2px); border-color: var(--f1-line-mid); box-shadow: 0 18px 52px rgba(0,0,0,0.58); }}
            .f1-hud-panel.target-hud   {{ border-left: 4px solid var(--accent, #E10600); animation: hudSlideLeft  0.42s cubic-bezier(0.16,1,0.3,1) both; }}
            .f1-hud-panel.attacker-hud {{ border-right: 4px solid var(--accent, #27F4D2); animation: hudSlideRight 0.42s cubic-bezier(0.16,1,0.3,1) both; }}

            .f1-hud-top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem; }}

            .f1-hud-badge {{
                font-size: 0.62rem; font-weight: 800; text-transform: uppercase;
                letter-spacing: 0.11em; padding: 0.22rem 0.55rem; border-radius: 3px;
                color: #FFF; background: var(--accent, #E10600);
                box-shadow: 0 2px 10px var(--accent-glow, rgba(225,6,0,0.38));
            }}

            .f1-hud-number {{ font-size: 1.60rem; font-weight: 900; font-style: italic; color: rgba(255,255,255,0.88); line-height: 1; }}
            .f1-hud-body   {{ display:flex; align-items:center; gap:0.85rem; margin:0.3rem 0; }}

            .f1-hud-avatar {{
                width: 60px; height: 60px; min-width: 60px; border-radius: 50%;
                overflow: hidden; border: 2px solid var(--accent, #E10600);
                box-shadow: 0 0 16px var(--accent-glow, rgba(225,6,0,0.28));
                display: flex; align-items: center; justify-content: center;
                background: linear-gradient(135deg, #10141C, #1E2532);
            }}

            .f1-hud-avatar img {{ width:100%; height:100%; object-fit:cover; object-position:top center; }}
            .f1-hud-monogram {{ font-size:1.15rem; font-weight:900; letter-spacing:0.04em; color:#FFF; }}
            .f1-hud-name {{ font-size: 1.15rem; font-weight: 800; text-transform: uppercase; color: #FFF; line-height: 1.1; margin: 0.05rem 0; letter-spacing: 0.025em; }}
            .f1-hud-team {{ font-size:0.78rem; font-weight:600; color:var(--f1-muted); }}

            .f1-hud-footer {{
                display: flex; justify-content: space-between; align-items: center;
                margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid var(--f1-line);
                font-size: 0.70rem; color: var(--f1-muted); font-weight: 600;
            }}

            .f1-hud-vs {{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:115px; user-select:none; }}
            .f1-hud-vs-title {{ font-size:1.9rem; font-weight:900; font-style:italic; letter-spacing:0.07em; color:#FFF; line-height:1; margin:0.15rem 0; animation: vsGlowPulse 2.4s ease-in-out infinite alternate; }}
            .f1-hud-vs-slash {{ font-size:0.82rem; font-weight:900; letter-spacing:0.20em; color:var(--f1-red); }}

            /* --- RACE SETUP --- */
            .f1-setup-header {{ margin: 1.75rem 0 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--f1-line); }}
            .f1-setup-kicker {{ display: inline-block; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.15em; color: var(--f1-red); margin-bottom: 0.38rem; }}
            .f1-setup-title  {{ margin: 0; font-size: 1.45rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.03em; color: #FFF; line-height: 1; }}

            .f1-setup-card {{
                background: var(--f1-card-bg); border: 1px solid var(--f1-line-mid);
                border-radius: 12px; padding: 1.35rem 1.5rem 1.55rem;
                backdrop-filter: blur(20px); box-shadow: 0 6px 28px rgba(0,0,0,0.35);
                height: 100%; transition: border-color 0.2s ease;
            }}

            .f1-setup-card:hover {{ border-color: var(--f1-line-strong); }}

            .f1-setup-card-label {{
                font-size: 0.65rem; font-weight: 800; text-transform: uppercase;
                letter-spacing: 0.13em; color: var(--f1-muted); margin-bottom: 1.15rem;
                padding-bottom: 0.72rem; border-bottom: 1px solid var(--f1-line);
            }}

            .f1-brief-card {{ border-left: 3px solid var(--f1-red); }}
            .f1-brief-body {{ color: var(--f1-muted-2); font-size: 0.88rem; line-height: 1.70; margin: 0 0 1rem; }}
            .f1-brief-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; }}

            .f1-brief-stat {{
                background: rgba(255,255,255,0.025);
                border: 1px solid var(--f1-line); border-radius: 7px;
                padding: 0.58rem 0.8rem; transition: border-color 0.18s ease;
            }}

            .f1-brief-stat:hover {{ border-color: var(--f1-line-mid); }}
            .f1-brief-stat-val {{ font-size:0.92rem; font-weight:800; color:#FFF; line-height:1.15; }}
            .f1-brief-stat-key {{ font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--f1-muted); margin-top:0.14rem; }}
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
                    <div class="nfs-kicker">Live race data &nbsp;&middot;&nbsp; ML strategy model</div>
                    <h1 class="nfs-title">F1 <em>Strategy</em><br>Engineer</h1>
                    <p class="nfs-copy">
                        Set up a Grand Prix, choose the tactical matchup, and load the race engine
                        to expose tyre degradation, gap evolution, and the exact lap where the
                        undercut becomes a threat.
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

    svg_track = '''
    <svg width="60" height="40" viewBox="0 0 100 60" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 0.5rem;">
        <path d="M 20 50 L 80 50 Q 90 50 90 40 L 90 20 Q 90 10 80 10 L 40 10 Q 30 10 30 20 L 30 30 Q 30 40 20 40 L 10 40 Q 5 40 5 45 Q 5 50 20 50 Z" />
    </svg>
    '''

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
            f'      {svg_track}'
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

