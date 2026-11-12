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
                    <div class="f1-gantry-row">
                        <div class="f1-lights" aria-hidden="true">
                            <span class="f1-light-dot"></span>
                            <span class="f1-light-dot"></span>
                            <span class="f1-light-dot"></span>
                            <span class="f1-light-dot"></span>
                            <span class="f1-light-dot"></span>
                        </div>
                        <div class="nfs-kicker">Pit Wall Telemetry &nbsp;//&nbsp; Live Strategy Engine</div>
                    </div>
                    <h1 class="nfs-title">F1 <em>Strategy</em><br>Engineer</h1>
                    <p class="nfs-copy">
                        Configure a Grand Prix, lock in the on-track matchup, and run the probabilistic
                        race engine to expose tyre degradation, gap evolution, and exact undercut windows.
                    </p>
                </div>
                <div class="f1-hero-telemetry">
                    <div class="f1-telem-pill">
                        <span>Model</span>
                        <strong>XGBoost Quantile</strong>
                    </div>
                    <div class="f1-telem-pill">
                        <span>Bounds</span>
                        <strong>p05 / p50 / p95</strong>
                    </div>
                    <div class="f1-telem-pill">
                        <span>Simulator</span>
                        <strong>500x Monte Carlo</strong>
                    </div>
                </div>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

