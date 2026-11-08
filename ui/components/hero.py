"""
ui/components/hero.py — Cinematic Homepage Hero Banner
======================================================
Renders the prominent top header with title, subtitle, and live indicator.
"""

from textwrap import dedent
import streamlit as st


def render_homepage_hero() -> None:
    """Render the cinematic dark motorsport hero header."""
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

