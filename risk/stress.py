"""Scenario stress testing.

Two flavours:
  - historical replay : apply a named crisis-period shock vector to today's
    positions (2008 GFC, 2020 COVID crash, 2022 rate shock).
  - hypothetical       : user-defined per-factor shocks (e.g. equities -20%,
    credit spreads +150bp mapped to a -8% credit sleeve move).

Returns the portfolio P&L under each scenario in currency and percent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Stylised per-sleeve shock vectors (fractional one-shot moves) for the
# demo asset order: [EQ_US, EQ_EU, RATES, CREDIT, COMMOD].
HISTORICAL_SCENARIOS = {
    "GFC_2008":      np.array([-0.46, -0.44,  0.08, -0.30, -0.35]),
    "COVID_2020":    np.array([-0.34, -0.32,  0.05, -0.18, -0.42]),
    "RATES_2022":    np.array([-0.19, -0.16, -0.14, -0.11, -0.05]),
    "VOL_SPIKE":     np.array([-0.12, -0.11,  0.03, -0.07, -0.09]),
}


@dataclass
class StressResult:
    scenario: str
    pnl_value: float
    pnl_pct: float


def stress_portfolio(weights, portfolio_value: float = 1_000_000.0,
                     scenarios: dict[str, np.ndarray] | None = None):
    """Apply each scenario's shock vector to the weighted portfolio."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    scenarios = scenarios or HISTORICAL_SCENARIOS
    out = []
    for name, shock in scenarios.items():
        shock = np.asarray(shock, dtype=float)[: len(w)]
        pnl_pct = float(w @ shock)
        out.append(StressResult(name, pnl_pct * portfolio_value, pnl_pct))
    return sorted(out, key=lambda s: s.pnl_pct)   # worst first


def hypothetical_shock(weights, shock_vector, portfolio_value: float = 1_000_000.0):
    """One custom scenario from a per-sleeve shock vector."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    s = np.asarray(shock_vector, dtype=float)[: len(w)]
    pnl_pct = float(w @ s)
    return StressResult("hypothetical", pnl_pct * portfolio_value, pnl_pct)
