"""
engine/rival_reaction.py — Two-Agent Competitor Reaction & Cover Model
======================================================================
Models game-theoretic two-car interaction during an undercut attempt:
- Estimates the empirical probability that the defending leader covers on the next lap.
- Accurately credits the covering rival's pit stop time loss, resolving Reviewer Concern 6.
"""

from __future__ import annotations
import numpy as np

# Circuit overtaking friction & wake constants shared with strategy simulation
_DIRTY_AIR_PENALTY = {"Low": 0.35, "Medium": 0.55, "High": 0.90}
_OVERTAKE_THRESHOLD = {"Low": 0.55, "Medium": 1.00, "High": 1.80}

_HIGH_DIFFICULTY_CIRCUITS = {
    "Monaco Grand Prix",
    "Singapore Grand Prix",
    "Hungarian Grand Prix",
    "Spanish Grand Prix",
    "Dutch Grand Prix",
    "Emilia Romagna Grand Prix",
}
_LOW_DIFFICULTY_CIRCUITS = {
    "Canadian Grand Prix",
    "Italian Grand Prix",
    "Belgian Grand Prix",
    "Azerbaijan Grand Prix",
    "Las Vegas Grand Prix",
    "Austrian Grand Prix",
}


def get_overtaking_difficulty(race_name: str = "") -> str:
    """Resolve circuit overtaking difficulty category ('Low', 'Medium', 'High')."""
    if race_name in _HIGH_DIFFICULTY_CIRCUITS:
        return "High"
    if race_name in _LOW_DIFFICULTY_CIRCUITS:
        return "Low"
    return "Medium"


def calculate_cover_rate(
    rival_gap_s: float | None,
    def_tyre_age: float | None = None,
    def_compound_encoded: float | None = None,
) -> float:
    """
    Compute empirical 1-2 lap defensive cover probability from 2023 F1 pit-duel data.
    When defender tyre age/compound are known, uses the 4-parameter 2023-calibrated
    logistic model; otherwise falls back to the 2023 gap-only schedule.
    """
    if rival_gap_s is None or np.isnan(rival_gap_s):
        return 0.55

    abs_gap = abs(float(rival_gap_s))
    if def_tyre_age is not None or def_compound_encoded is not None:
        age = float(def_tyre_age) if def_tyre_age is not None and not np.isnan(def_tyre_age) else 15.0
        comp = float(def_compound_encoded) if def_compound_encoded is not None and not np.isnan(def_compound_encoded) else 2.0
        logit = 1.0759 - 0.1454 * abs_gap + 0.0428 * age - 0.4118 * comp
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))
        return float(np.clip(prob, 0.15, 0.90))

    if abs_gap <= 1.5:
        return 0.64
    elif abs_gap <= 2.5:
        return 0.67
    elif abs_gap <= 3.5:
        return 0.70
    elif abs_gap <= 4.5:
        return 0.60
    else:
        return 0.52


def simulate_undercut_differential(
    attacker_lap_times_no_pitloss: np.ndarray,
    defender_stay_out_times: np.ndarray,
    pit_offset: int,
    horizon: int,
    effective_pit_loss: np.ndarray,
    rival_gap_s: float | None,
    cover_rate: float,
    n_rollouts: int,
    rng: np.random.Generator,
    out_lap_penalty: float = 1.5,
    defender_fresh_lap_times: np.ndarray | None = None,
    fuel_corrections: np.ndarray | None = None,
    overtaking_difficulty: str = "Medium",
    push_bonus_per_lap: float = 0.90,
) -> np.ndarray:
    """
    Vectorized two-car undercut differential with multi-lag defender cover reactions (+1, +2, +4),
    age-indexed fresh-tyre pace reset, undercut free-air push extraction, and symmetric
    dirty-air / overtaking resistance blocking phantom frictionless passes.
    """
    init_gap = abs(float(rival_gap_s)) if rival_gap_s is not None else 2.5
    p_cov = float(np.clip(cover_rate, 0.0, 1.0))
    # Split defensive covers into immediate (+1 lap: 62%) and reactive (+2 laps: 38%), else extend (+4)
    p_lag1 = 0.62 * p_cov
    p_lag2 = 0.38 * p_cov
    p_ext = max(0.0, 1.0 - p_lag1 - p_lag2)
    lags = rng.choice([1, 2, 4], size=n_rollouts, p=[p_lag1, p_lag2, p_ext])

    att = attacker_lap_times_no_pitloss.copy()
    if pit_offset < horizon:
        att[:, pit_offset] += effective_pit_loss
    if pit_offset + 1 < horizon:
        att[:, pit_offset + 1] += out_lap_penalty

    df_t = defender_stay_out_times.copy()
    def_pits = np.minimum(pit_offset + lags, max(pit_offset + 1, horizon - 2))

    for lag_val in (1, 2, 4):
        m = lags == lag_val
        if not np.any(m):
            continue
        dp = int(min(pit_offset + lag_val, max(pit_offset + 1, horizon - 2)))
        if dp < horizon:
            df_t[m, dp] += effective_pit_loss[m]
            od = dp + 1
            if od < horizon:
                if defender_fresh_lap_times is not None:
                    if fuel_corrections is not None:
                        # Age-indexed fresh predictions: age 1 begins cleanly at defender's out-lap `od`
                        k_len = horizon - od
                        df_t[m, od:horizon] = defender_fresh_lap_times[m, :k_len] + fuel_corrections[od:horizon]
                    else:
                        df_t[m, od:horizon] = defender_fresh_lap_times[m, od:horizon]
                df_t[m, od] += out_lap_penalty

    dirty_air_s = _DIRTY_AIR_PENALTY.get(overtaking_difficulty, 0.55)
    overtake_thresh_s = _OVERTAKE_THRESHOLD.get(overtaking_difficulty, 1.00)

    cum = np.full(n_rollouts, -init_gap, dtype=float)
    for h in range(horizon):
        a_l = att[:, h].copy()
        d_l = df_t[:, h].copy()

        # Free-air undercut push vs defender in-lap / worn-tyre management while defender stays out
        in_window = (h >= pit_offset + 1) & (h <= def_pits)
        a_l[in_window] -= push_bonus_per_lap

        # Symmetric dirty-air wake (when trailing car is inside 1.5s wake on non-pit/out laps)
        is_racing_lap = (
            (h != pit_offset)
            & (h != pit_offset + 1)
            & (h != def_pits)
            & (h != def_pits + 1)
        )
        att_in_wake = is_racing_lap & (cum < 0.0) & (-cum <= 1.5)
        def_in_wake = is_racing_lap & (cum > 0.0) & (cum <= 1.5)
        a_l[att_in_wake] += dirty_air_s
        d_l[def_in_wake] += dirty_air_s

        new_cum = cum + (d_l - a_l)

        # Symmetric overtaking pace threshold: cannot overtake on track unless pace delta >= threshold
        att_blocked = (cum < 0.0) & (new_cum > 0.0) & (h != def_pits) & ((d_l - a_l) < overtake_thresh_s)
        def_blocked = (cum > 0.0) & (new_cum < 0.0) & (h != pit_offset) & ((a_l - d_l) < overtake_thresh_s)
        new_cum = np.where(att_blocked, -0.35, new_cum)
        new_cum = np.where(def_blocked, +0.35, new_cum)
        cum = new_cum

    return cum


def apply_rival_reaction(*args, **kwargs) -> np.ndarray:
    """Backward compatibility wrapper delegating to simulate_undercut_differential."""
    return simulate_undercut_differential(*args, **kwargs)

