"""Value-at-Risk and Expected Shortfall estimators.

Three industry-standard methods on a portfolio return series:
  - historical   : empirical quantile of realised P&L (no dist/ assumption)
  - parametric   : variance-covariance (Gaussian) closed form
  - monte_carlo  : simulated portfolio returns (see monte_carlo.py)

All functions return LOSS-POSITIVE VaR/ES as a fraction of portfolio value
(e.g. 0.021 == a 2.1% one-day loss at the stated confidence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class RiskResult:
    method: str
    confidence: float
    horizon_days: int
    var: float          # loss-positive fraction of portfolio value
    es: float           # expected shortfall (a.k.a. CVaR), loss-positive
    var_value: float    # VaR in currency, given portfolio_value
    es_value: float


def portfolio_returns(returns, weights) -> np.ndarray:
    """Collapse an asset return panel to a single weighted portfolio series."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    r = np.asarray(returns, dtype=float)
    return r @ w


def historical_var(
    port_rets: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    portfolio_value: float = 1_000_000.0,
) -> RiskResult:
    """Empirical-quantile VaR/ES. Scales to horizon via sqrt-time."""
    q = np.quantile(port_rets, 1.0 - confidence)      # left-tail return (neg)
    tail = port_rets[port_rets <= q]
    var_1d = -q
    es_1d = -tail.mean() if tail.size else var_1d
    scale = np.sqrt(horizon_days)
    var = var_1d * scale
    es = es_1d * scale
    return RiskResult(
        "historical", confidence, horizon_days,
        var, es, var * portfolio_value, es * portfolio_value,
    )


def parametric_var(
    port_rets: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    portfolio_value: float = 1_000_000.0,
) -> RiskResult:
    """Gaussian variance-covariance VaR with closed-form ES."""
    mu = port_rets.mean()
    sigma = port_rets.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)              # negative
    scale = np.sqrt(horizon_days)
    var_1d = -(mu + z * sigma)
    # Closed-form Gaussian ES: mu - sigma * phi(z)/(1-c)
    es_1d = -(mu - sigma * stats.norm.pdf(z) / (1.0 - confidence))
    var = var_1d * scale
    es = es_1d * scale
    return RiskResult(
        "parametric", confidence, horizon_days,
        var, es, var * portfolio_value, es * portfolio_value,
    )


def component_var(returns, weights, confidence: float = 0.99,
                  portfolio_value: float = 1_000_000.0):
    """Marginal / component VaR decomposition (Gaussian).

    Returns a dict mapping asset index -> component VaR in currency; the
    components sum to the total parametric VaR, so risk can be attributed
    to each sleeve for the executive dashboard.
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    cov = np.cov(r, rowvar=False)
    port_var = float(w @ cov @ w)
    port_sigma = np.sqrt(port_var)
    z = stats.norm.ppf(1.0 - confidence)
    # dVaR/dw_i proportional to (cov @ w) / sigma
    marginal = -(z) * (cov @ w) / port_sigma
    component = w * marginal                          # sums to total VaR (frac)
    return {
        "component_frac": component,
        "component_value": component * portfolio_value,
        "total_value": component.sum() * portfolio_value,
    }
