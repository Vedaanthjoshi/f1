"""
engine/rival_reaction.py — Two-Agent Competitor Reaction & Cover Model
======================================================================
Models game-theoretic two-car interaction during an undercut attempt:
- Estimates the empirical probability that the defending leader covers on the next lap.
- Accurately credits the covering rival's pit stop time loss, resolving Reviewer Concern 6.
"""

from __future__ import annotations
import numpy as np


def calculate_cover_rate(rival_gap_s: float | None) -> float:
    """
    Compute empirical defensive cover probability based on gap size in seconds.
    Tighter gaps induce higher cover urgency from the pit wall.
    """
    if rival_gap_s is None or np.isnan(rival_gap_s):
        return 0.35  # default moderate cover probability if gap unknown

    abs_gap = abs(rival_gap_s)
    if abs_gap <= 1.5:
        return 0.80   # high urgency: immediate reaction
    elif abs_gap <= 3.0:
        return 0.60   # standard undercut defense window
    elif abs_gap <= 5.0:
        return 0.30   # loose window: defender may choose to extend
    else:
        return 0.10   # gap too large: defender stays out


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
) -> np.ndarray:
    """
    Simulate the two-car undercut differential equation between Attacker and Defender.

    Parameters
    ----------
    attacker_lap_times_no_pitloss : shape (n_rollouts, horizon)
        Pure driving lap times of attacker (fresh tyres starting at pit_offset).
    defender_stay_out_times : shape (n_rollouts, horizon)
        Lap times of defender staying on worn tyres (pre-cover laps only).
    defender_fresh_lap_times : shape (n_rollouts, horizon) or None
        Fresh-tyre lap times for the defender if they pit to cover. When supplied,
        the defender's times from (react_pit_lap + 2) onward are replaced with these
        predictions so the defender is correctly credited with fresh rubber after
        their stop. If None, defender_stay_out_times is used for all laps (old
        behaviour — systematically overstates the attacker's advantage).
    pit_offset : int
        Horizon lap index where attacker pits (e.g. 0 for Pit This Lap).
    horizon : int
        Forward simulation horizon in laps.
    effective_pit_loss : shape (n_rollouts,)
        Pit lane transit and stop time loss in seconds.
    rival_gap_s : float or None
        Initial time gap between defender and attacker (defender ahead by rival_gap_s).
    cover_rate : float
        Probability that defender pits on lap pit_offset + 1 to cover.
    n_rollouts : int
        Number of Monte Carlo rollouts.
    rng : np.random.Generator
        Random number generator.
    out_lap_penalty : float
        Cold-tyre out-lap warmup penalty in seconds.

    Returns
    -------
    net_margins : np.ndarray of shape (n_rollouts,)
        Net time margin (+ = attacker ahead / undercut successful, - = defender ahead).
    """
    init_gap = abs(rival_gap_s) if rival_gap_s is not None else 2.5
    rival_reacts = rng.binomial(1, cover_rate, size=n_rollouts).astype(bool)
    net_margins = np.zeros(n_rollouts)

    for i in range(n_rollouts):
        pit_loss_val = effective_pit_loss[i]
        
        # Attacker lap times including pit loss and out-lap warmup
        att_times = attacker_lap_times_no_pitloss[i].copy()
        att_times[pit_offset] += pit_loss_val
        out_lap_idx = min(pit_offset + 1, horizon - 1)
        att_times[out_lap_idx] += out_lap_penalty

        def_times = defender_stay_out_times[i].copy()

        if rival_reacts[i] and (pit_offset + 1 < horizon):
            # Defender COVERS on the very next lap (pit_offset + 1).
            # The cover lap is driven on worn tyres (in-lap) + pit transit loss.
            react_pit_lap = pit_offset + 1
            def_times[react_pit_lap] += pit_loss_val
            out_lap_cover = react_pit_lap + 1

            # From the out-lap onward, the defender is now on fresh tyres.
            if defender_fresh_lap_times is not None and out_lap_cover < horizon:
                def_times[out_lap_cover:] = defender_fresh_lap_times[i, out_lap_cover:]
            if out_lap_cover < horizon:
                def_times[out_lap_cover] += out_lap_penalty

            # Evaluate over the full horizon.
            diff = np.sum(def_times) - np.sum(att_times)
            net_margins[i] = diff - init_gap
        else:
            # Defender EXTENDS (does not cover immediately).
            # Defender stays on worn tyres, then pits 5 laps after the attacker.
            def_pit_lap = min(pit_offset + 5, horizon - 2)
            if def_pit_lap < horizon:
                def_times[def_pit_lap] += pit_loss_val
                out_lap_ext = def_pit_lap + 1
                if defender_fresh_lap_times is not None and out_lap_ext < horizon:
                    def_times[out_lap_ext:] = defender_fresh_lap_times[i, out_lap_ext:]
                if out_lap_ext < horizon:
                    def_times[out_lap_ext] += out_lap_penalty

            diff = np.sum(def_times) - np.sum(att_times)
            net_margins[i] = diff - init_gap

    return net_margins


def apply_rival_reaction(*args, **kwargs) -> np.ndarray:
    """Backward compatibility wrapper delegating to simulate_undercut_differential."""
    return simulate_undercut_differential(*args, **kwargs)

