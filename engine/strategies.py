"""
engine/strategies.py — Candidate Pit Strategy Definitions
==========================================================
Defines the tactical strategy space evaluated by the Monte Carlo optimizer.
"""

from typing import Any

STRATEGIES: list[dict[str, Any]] = [
    {"name": "Stay Out",        "pit_offset": None, "compound_encoded": None},
    {"name": "Pit → Soft",      "pit_offset": 0,    "compound_encoded": 1},
    {"name": "Pit → Medium",    "pit_offset": 0,    "compound_encoded": 2},
    {"name": "Pit → Hard",      "pit_offset": 0,    "compound_encoded": 3},
    {"name": "Pit+1 → Medium",  "pit_offset": 1,    "compound_encoded": 2},
    {"name": "Pit+1 → Hard",    "pit_offset": 1,    "compound_encoded": 3},
]

COMPOUND_MAP: dict[str, int] = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
}

REVERSE_COMPOUND_MAP: dict[int, str] = {v: k for k, v in COMPOUND_MAP.items()}

