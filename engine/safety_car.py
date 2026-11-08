"""
engine/safety_car.py — Safety Car / VSC Stochastic Probability Engine
=====================================================================
Per-lap empirical Bernoulli probabilities for Safety Car / VSC neutralizations
derived from historical 2022-2025 FastF1 TrackStatus data.
"""

import numpy as np

# Per-lap empirical probability of an SC/VSC event per Grand Prix circuit
SC_RATES: dict[str, float] = {
    "Bahrain Grand Prix":        0.04,
    "Saudi Arabian Grand Prix":  0.06,
    "Australian Grand Prix":     0.09,
    "Japanese Grand Prix":       0.04,
    "Chinese Grand Prix":        0.06,
    "Miami Grand Prix":          0.07,
    "Emilia Romagna Grand Prix": 0.05,
    "Monaco Grand Prix":         0.14,
    "Canadian Grand Prix":       0.09,
    "Spanish Grand Prix":        0.04,
    "Austrian Grand Prix":       0.06,
    "British Grand Prix":        0.07,
    "Hungarian Grand Prix":      0.04,
    "Belgian Grand Prix":        0.05,
    "Dutch Grand Prix":          0.05,
    "Italian Grand Prix":        0.04,
    "Azerbaijan Grand Prix":     0.14,
    "Singapore Grand Prix":      0.12,
    "United States Grand Prix":  0.07,
    "Mexico City Grand Prix":    0.05,
    "São Paulo Grand Prix":      0.11,
    "Las Vegas Grand Prix":      0.08,
    "Qatar Grand Prix":          0.06,
    "Abu Dhabi Grand Prix":      0.04,
}

SC_RATE_DEFAULT: float = 0.06       # Fallback rate for unknown circuits
SC_PIT_LOSS_MULTIPLIER: float = 0.35 # Pit loss under SC collapses to ~35% of green flag loss (~8-9s)


def get_circuit_sc_rate(race_name: str) -> float:
    """Return the per-lap Safety Car probability for the given Grand Prix circuit."""
    return SC_RATES.get(race_name, SC_RATE_DEFAULT)


def sample_sc_pit_loss(
    race_name: str,
    base_pit_loss: float,
    n_rollouts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Safety Car stochastic trials for the pit lap.
    
    Returns:
    --------
    effective_pit_loss : np.ndarray of shape (n_rollouts,)
    sc_on_pit_lap      : np.ndarray of shape (n_rollouts,) boolean mask
    """
    sc_rate = get_circuit_sc_rate(race_name)
    sc_on_pit_lap = rng.binomial(1, sc_rate, size=n_rollouts).astype(bool)
    effective_pit_loss = np.where(
        sc_on_pit_lap,
        base_pit_loss * SC_PIT_LOSS_MULTIPLIER,
        base_pit_loss,
    )
    return effective_pit_loss, sc_on_pit_lap

