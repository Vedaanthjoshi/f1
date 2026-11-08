"""
ui/components/strategy_table.py — Monte Carlo Strategy Matrix Table
===================================================================
Renders the 6-strategy optimization matrix with color-coded confidence pills,
expected net gains, 90% confidence intervals, and Safety Car impacts.
"""

from typing import Any
import streamlit as st

from engine.safety_car import get_circuit_sc_rate


def render_strategy_table(
    mc_results: list[dict[str, Any]],
    race_name: str,
    n_rollouts: int,
) -> None:
    """Render the glassmorphic HTML strategy ranking table and status banner."""
    sc_rate = get_circuit_sc_rate(race_name)
    sc_pct = f"{sc_rate:.0%}"
    best_strat = next((r for r in mc_results if r["window_open"]), None)

    if best_strat:
        st.success(
            f"✅ **STRATEGY RECOMMENDED: {best_strat['name']}** — "
            f"{best_strat['p_net_gain']:.0%} probability of net gain, "
            f"expected +{best_strat['expected_gain_s']:.2f}s "
            f"(CI: {best_strat['gain_ci_low']:+.1f}s → {best_strat['gain_ci_high']:+.1f}s)"
        )
    else:
        st.info("No strategy currently clears the 55% confidence threshold. Stay out or wait for a better window.")

    rows_html = ""
    for r in mc_results:
        if r["name"] == "Stay Out":
            badge = '<span class="mc-badge-neutral">BASELINE</span>'
            gain_str = "<span style='color:#7A8BA0'>&#8212;</span>"
            ci_str = "<span style='color:#7A8BA0'>&#8212;</span>"
            sc_str = "<span style='color:#7A8BA0'>&#8212;</span>"
            p_str = "<span style='color:#7A8BA0'>&#8212;</span>"
        else:
            badge = '<span class="mc-badge-open">OPEN</span>' if r["window_open"] else '<span class="mc-badge-closed">CLOSED</span>'
            p_val = r['p_net_gain']
            p_cls = 'mc-p-high' if p_val >= 0.65 else ('mc-p-mid' if p_val >= 0.50 else 'mc-p-low')
            p_str = f'<span class="{p_cls}">{p_val:.0%}</span>'
            g_val = r['expected_gain_s']
            g_cls = 'mc-gain-pos' if g_val > 0 else 'mc-gain-neg'
            gain_str = f'<span class="{g_cls}">{g_val:+.2f}s</span>'
            ci_lo, ci_hi = r['gain_ci_low'], r['gain_ci_high']
            ci_str = f"<span style='color:#7A8BA0;font-size:0.78rem;'>{ci_lo:+.1f}s &rarr; {ci_hi:+.1f}s</span>"
            sc_val = r['sc_expected_gain_s']
            sc_cls = 'mc-gain-pos' if sc_val > 0 else 'mc-gain-neg'
            sc_str = f'<span class="{sc_cls}">{sc_val:+.2f}s</span>'

        xo = f"Lap {r['crossover_lap']}" if r.get("crossover_lap") else "<span style='color:#7A8BA0'>&#8212;</span>"
        cover_val = r.get('cover_rate', 0.0)
        cover_cls = 'mc-p-low' if cover_val >= 0.65 else ('mc-p-mid' if cover_val >= 0.35 else 'mc-gain-pos')
        cover_str = f'<span class="{cover_cls}">{cover_val:.0%}</span>'

        rows_html += (
            f"<tr>"
            f"<td><strong>{r['name']}</strong></td>"
            f"<td>{p_str}</td>"
            f"<td>{gain_str}</td>"
            f"<td>{ci_str}</td>"
            f"<td>{sc_str}</td>"
            f"<td>{cover_str}</td>"
            f"<td style='color:#AAB8C8;font-size:0.82rem;'>{xo}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="mc-table">
          <thead><tr>
            <th>Strategy</th><th>P(Net Gain)</th><th>E[Gain]</th>
            <th>90% CI</th><th>Under SC</th><th>Rival Cover</th><th>Crossover</th><th>Status</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        <p style='font-size:0.68rem;color:#5A6878;margin-top:0.3rem;line-height:1.5;'>
          &#9679; SC probability at {race_name}: {sc_pct}
          &nbsp;&middot;&nbsp; {n_rollouts:,} MC rollouts
          &nbsp;&middot;&nbsp; Horizon: 15 laps
          &nbsp;&middot;&nbsp; Green P &ge; 65% / Amber &ge; 50% / Red &lt; 50%
        </p>
        """,
        unsafe_allow_html=True,
    )

