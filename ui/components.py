"""Reusable UI components for the redesigned Streamlit interface."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from textwrap import dedent
from typing import Literal

import streamlit as st

from ui.theme import COLORS, css

StateKind = Literal["empty", "loading", "error", "success", "info"]


def apply_global_theme() -> None:
    """Apply the shared Streamlit CSS theme."""
    st.markdown(css(), unsafe_allow_html=True)


def hero(kicker: str, title: str, body: str) -> None:
    """Render the app hero without depending on future image assets."""
    st.markdown(
        dedent(f"""
        <section class="f1-shell-hero">
            <div class="f1-kicker">{escape(kicker)}</div>
            <h1 class="f1-title">{escape(title)}</h1>
            <p class="f1-copy">{escape(body)}</p>
        </section>
        """),
        unsafe_allow_html=True,
    )


def panel(title: str, body: str, meta: str | None = None) -> None:
    """Render a compact reusable panel."""
    meta_html = f'<div class="f1-panel-meta">{escape(meta)}</div>' if meta else ""
    st.markdown(
        dedent(f"""
        <section class="f1-panel">
            <h3 class="f1-panel-title">{escape(title)}</h3>
            <div class="f1-panel-meta">{escape(body)}</div>
            {meta_html}
        </section>
        """),
        unsafe_allow_html=True,
    )


def stat_grid(items: Iterable[tuple[str, str]]) -> None:
    """Render small dashboard metrics in a responsive grid."""
    cards = "\n".join(
        f'<div class="f1-stat"><div class="f1-stat-label">{escape(label)}</div><div class="f1-stat-value">{escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="f1-stat-grid">{cards}</div>', unsafe_allow_html=True)


def state_message(kind: StateKind, title: str, body: str) -> None:
    """Render consistent empty, loading, error, and success states."""
    accent = {
        "empty": COLORS["muted"],
        "loading": COLORS["cyan"],
        "error": COLORS["danger"],
        "success": COLORS["green"],
        "info": COLORS["yellow"],
    }[kind]
    st.markdown(
        dedent(f"""
        <section class="f1-panel" style="border-left: 4px solid {accent};">
            <h3 class="f1-panel-title">{escape(title)}</h3>
            <div class="f1-panel-meta">{escape(body)}</div>
        </section>
        """),
        unsafe_allow_html=True,
    )


def navigation() -> str:
    """Sidebar navigation labels for current and future UI modules."""
    return st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Asset Library",
            "Race Setup",
            "Driver Comparison",
            "Track Profile",
            "Strategy Analysis",
        ],
    )
