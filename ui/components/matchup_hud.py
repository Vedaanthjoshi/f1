"""
ui/components/matchup_hud.py — Driver Matchup HUD & Circuit Profile
===================================================================
Renders glassmorphic driver cards with driver headshot, team accent borders,
racing number, animated pulsing VS badge, and SVG circuit outline.
"""

from __future__ import annotations
from html import escape
from pathlib import Path
import streamlit as st

from ui.assets import image_data_uri
from ui.metadata import PROJECT_ROOT, driver_profile, get_track


def _driver_card(code: str | None, label: str, role_type: str = "target") -> str:
    """Render an individual driver glassmorphic HUD card."""
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
    driver_code = escape(code[:3].upper())

    return (
        f'<div class="f1-hud-panel {role_type}-hud" style="--accent: {escape(accent)}; --accent-glow: {escape(accent)}44;">'
        f'  <div class="f1-hud-top">'
        f'    <span class="f1-hud-badge">{escape(role_badge_label)}</span>'
        f'    <span class="f1-hud-number">{escape(number)}</span>'
        f'  </div>'
        f'  <div class="f1-hud-body">'
        f'    <div class="f1-hud-avatar">{avatar_html}</div>'
        f'    <div>'
        f'      <div class="f1-hud-name"><span class="f1-driver-tag">{driver_code}</span>{escape(driver.get("name", code))}</div>'
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
    """Render the full cinematic matchup HUD (Attacker vs Defender + Circuit Profile)."""
    track = get_track(race)
    track_circuit = track["circuit"] if track else race
    track_country = track.get("country", "Grand Prix") if track else "F1 Circuit"
    track_laps = f"{track['laps']} Laps" if track and track.get("laps") else "Race Distance"
    track_len = f"{track['lengthKm']} km" if track and track.get("lengthKm") else ""

    ctx = track.get("strategyContext", {}) if track else {}
    tags_html = ""
    if ctx:
        tags = [
            f"Tyre: {ctx.get('tyreStress', '—')}",
            f"Pass: {ctx.get('overtakingDifficulty', '—')}",
            f"Pit: {ctx.get('pitLaneSensitivity', '—')}",
        ]
        tags_html = '<div class="f1-track-tags">' + "".join(
            f'<span class="f1-track-tag">{escape(t)}</span>' for t in tags
        ) + "</div>"

    svg_track = '''
    <svg width="56" height="36" viewBox="0 0 100 60" fill="none" stroke="rgba(225,6,0,0.65)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 0.35rem;">
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
            '    <div class="f1-hud-body" style="flex-direction: column; align-items: flex-start; gap: 0.2rem;">'
            f'      {svg_track}'
            f'      <div class="f1-hud-name" style="font-size: 1.08rem;">{escape(track_circuit)}</div>'
            f'      <div class="f1-hud-team">{escape(track_country)}</div>'
            f'      {tags_html}'
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

