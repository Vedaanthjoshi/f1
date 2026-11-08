"""
ui/styles/theme.py — Theme Constants & Color Palettes
=====================================================
Centralized color definitions for Formula 1 teams, tyre compounds,
status badges, and UI surface layers.
"""

COLORS: dict[str, str] = {
    "background": "#07090D",
    "surface": "#10141B",
    "surface_alt": "#161C25",
    "surface_high": "#202837",
    "border": "rgba(255, 255, 255, 0.12)",
    "text": "#EDF2F7",
    "muted": "#7A8BA0",
    "muted_light": "#AAB8C8",
    "red": "#E10600",
    "cyan": "#00D4FF",
    "yellow": "#FFD166",
    "green": "#3ECF8E",
    "danger": "#FF5C5C",
}

TEAM_COLORS: dict[str, str] = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0090FF",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Sauber": "#52E252",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
    "Default": COLORS["red"],
}

TYRE_COLORS: dict[str, str] = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFE800",
    "HARD": "#E0E0E0",
}

