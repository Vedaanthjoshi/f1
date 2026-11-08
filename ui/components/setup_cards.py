"""
ui/components/setup_cards.py — Grand Prix Configuration Cards
=============================================================
Renders configuration selectboxes and strategy brief cards across both
the sidebar rail and the main viewport with two-way state sync.
"""

from typing import Callable
import streamlit as st

from ui.state import sync_season, sync_race


def render_race_setup(
    current_season: int,
    races: list[str],
    current_race: str,
    schedule_loader: Callable[[int], list[str]],
    on_load_requested: Callable[[int, str], None],
) -> None:
    """Render the sidebar and main setup panels."""
    st.sidebar.caption("F1 Strategy Engineer")
    st.sidebar.header("Race Setup")

    sidebar_season = st.sidebar.selectbox(
        "Season",
        list(range(2025, 2017, -1)),
        key="sidebar_season",
        on_change=sync_season,
        args=('sidebar',),
    )

    sidebar_races = races
    if sidebar_season != current_season:
        with st.spinner(f"Loading {sidebar_season} schedule..."):
            sidebar_races = schedule_loader(sidebar_season)

    if st.session_state.sidebar_race not in sidebar_races:
        st.session_state.sidebar_race = sidebar_races[0]
        st.session_state.home_race = sidebar_races[0]

    sidebar_race = st.sidebar.selectbox(
        "Grand Prix",
        sidebar_races,
        key="sidebar_race",
        on_change=sync_race,
        args=('sidebar',),
    )

    if st.sidebar.button("Load Selected Race", width='stretch'):
        on_load_requested(sidebar_season, sidebar_race)

    # Main panel section header
    st.markdown(
        """
        <div class="f1-setup-header">
            <span class="f1-setup-kicker">&#9635; Grand Prix Configuration</span>
            <h2 class="f1-setup-title">Race Setup</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_col, brief_col = st.columns([1, 1], gap="large")

    with setup_col:
        st.markdown('<div class="f1-setup-card">', unsafe_allow_html=True)
        st.markdown('<div class="f1-setup-card-label">Configure the Grand Prix</div>', unsafe_allow_html=True)

        home_season = st.selectbox(
            "Season",
            list(range(2025, 2017, -1)),
            key="home_season",
            on_change=sync_season,
            args=('home',),
        )

        home_races = races
        if home_season != current_season:
            with st.spinner(f"Loading {home_season} schedule..."):
                home_races = schedule_loader(home_season)

        if st.session_state.home_race not in home_races:
            st.session_state.home_race = home_races[0]
            st.session_state.sidebar_race = home_races[0]

        home_race = st.selectbox(
            "Grand Prix",
            home_races,
            key="home_race",
            on_change=sync_race,
            args=('home',),
        )
        if st.button("Load Race Strategy Data", width='stretch'):
            on_load_requested(home_season, home_race)
        st.markdown('</div>', unsafe_allow_html=True)

    with brief_col:
        st.markdown(
            """
            <div class="f1-setup-card f1-brief-card">
                <div class="f1-setup-card-label">Strategy Brief</div>
                <p class="f1-brief-body">
                    The race engine will prepare real telemetry data and leverage
                    the pre-trained Global AI Model to simulate undercut strategies.
                </p>
                <div class="f1-brief-stats">
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">2018 – 2025</span>
                        <span class="f1-brief-stat-key">Season Range</span>
                    </div>
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">XGBoost</span>
                        <span class="f1-brief-stat-key">Degradation Model</span>
                    </div>
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">15-lap</span>
                        <span class="f1-brief-stat-key">Simulation Horizon</span>
                    </div>
                    <div class="f1-brief-stat">
                        <span class="f1-brief-stat-val">Live Pit Loss</span>
                        <span class="f1-brief-stat-key">From FastF1 Data</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

