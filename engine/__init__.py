"""
engine package — Formula 1 Strategy & Monte Carlo Simulation Engine
"""

from engine.strategies import STRATEGIES, COMPOUND_MAP, REVERSE_COMPOUND_MAP
from engine.safety_car import SC_RATES, SC_RATE_DEFAULT, SC_PIT_LOSS_MULTIPLIER, get_circuit_sc_rate
from engine.rival_reaction import calculate_cover_rate, apply_rival_reaction, simulate_undercut_differential
from engine.pit_loss import calculate_pit_loss
from engine.monte_carlo import run_mc_simulation, simulate_undercut_window

__all__ = [
    "STRATEGIES",
    "COMPOUND_MAP",
    "REVERSE_COMPOUND_MAP",
    "SC_RATES",
    "SC_RATE_DEFAULT",
    "SC_PIT_LOSS_MULTIPLIER",
    "get_circuit_sc_rate",
    "calculate_cover_rate",
    "apply_rival_reaction",
    "simulate_undercut_differential",
    "calculate_pit_loss",
    "run_mc_simulation",
    "simulate_undercut_window",
]

