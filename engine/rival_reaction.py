"""
engine/rival_reaction.py — Competitor Reaction & Cover Rate Model
==================================================================
Models two-agent strategic interaction by estimating the probability that
a defending rival pits on the subsequent lap to "cover" an undercut attempt.
"""

import numpy as np


def calculate_cover_rate(rival_gap_s: float | None) -> float:
    """
    Compute empirical defensive cover probability based on gap size in seconds.
    Tighter gaps induce higher cover urgency from the pit wall.
    """
    if rival_gap_s is None or np.isnan(rival_gap_s):
        return 0.0

    abs_gap = abs(rival_gap_s)
    if abs_gap <= 1.5:
        return 0.75
    elif abs_gap <= 3.0:
        return 0.45
    elif abs_gap <= 5.0:
        return 0.20
    else:
        return 0.05


def apply_rival_reaction(
    base_net_gains_per_lap: np.ndarray,
    pit_offset: int,
    horizon: int,
    cover_rate: float,
    n_rollouts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply competitor reaction clamping to rollout trajectories.
    
    If rival reacts on the lap AFTER Driver A pits (pit_offset + 1), both cars
    now have fresh tyres, clamping Driver A's advantage to whatever delta was
    accumulated through the reaction lap.
    
    Returns:
    --------
    final_net_gains : np.ndarray of shape (n_rollouts,)
    """
    rival_reacts = rng.binomial(1, cover_rate, size=n_rollouts).astype(bool)
    final_net_gains = np.zeros(n_rollouts)

    for i in range(n_rollouts):
        cumulative_gain = np.cumsum(base_net_gains_per_lap[i])
        if rival_reacts[i] and (pit_offset + 1 < horizon):
            # Clamp advantage after the reaction lap
            final_net_gains[i] = cumulative_gain[pit_offset + 1]
        else:
            final_net_gains[i] = cumulative_gain[-1]

    return final_net_gains

