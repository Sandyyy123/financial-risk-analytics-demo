"""Monte Carlo simulation for portfolio VaR / Expected Shortfall.

Draws correlated one-horizon asset returns from a multivariate model fitted
to the history (Cholesky of the sample covariance), aggregates to portfolio
P&L, and reads VaR/ES off the simulated loss distribution. Supports a
Gaussian or a heavier-tailed multivariate-t innovation to stress the tails.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .var import RiskResult


def monte_carlo_var(
    returns,
    weights,
    confidence: float = 0.99,
    horizon_days: int = 1,
    n_sims: int = 100_000,
    dist: str = "t",
    t_df: int = 5,
    portfolio_value: float = 1_000_000.0,
    seed: int = 11,
) -> RiskResult:
    """Simulate portfolio returns and return loss-positive VaR/ES.

    dist='normal' -> multivariate Gaussian; dist='t' -> multivariate Student-t
    (fatter tails, a more conservative and common risk-desk choice).
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    mu = r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    chol = np.linalg.cholesky(_psd(cov))
    rng = np.random.default_rng(seed)

    k = len(w)
    if dist == "normal":
        z = rng.standard_normal((n_sims, k))
    elif dist == "t":
        # scale so the multivariate-t has the target covariance
        g = rng.standard_normal((n_sims, k))
        chi = rng.chisquare(t_df, size=(n_sims, 1))
        z = g * np.sqrt(t_df / chi)
        z *= np.sqrt((t_df - 2) / t_df)               # unit-variance scaling
    else:
        raise ValueError("dist must be 'normal' or 't'")

    sim_1d = mu + z @ chol.T                           # (n_sims x k) daily
    # sqrt-time aggregation of the horizon on the portfolio series
    port_1d = sim_1d @ w
    port_h = port_1d * np.sqrt(horizon_days)

    q = np.quantile(port_h, 1.0 - confidence)
    tail = port_h[port_h <= q]
    var = -q
    es = -tail.mean() if tail.size else var
    return RiskResult(
        f"monte_carlo_{dist}", confidence, horizon_days,
        var, es, var * portfolio_value, es * portfolio_value,
    )


def simulate_paths(
    returns,
    weights,
    horizon_days: int = 10,
    n_paths: int = 5000,
    portfolio_value: float = 1_000_000.0,
    seed: int = 11,
) -> np.ndarray:
    """Simulate cumulative portfolio value paths for a fan chart / drawdown.

    Returns an (n_paths x horizon_days+1) array of portfolio values starting
    at portfolio_value.
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu = r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    chol = np.linalg.cholesky(_psd(cov))
    rng = np.random.default_rng(seed)
    k = len(w)

    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = portfolio_value
    for t in range(1, horizon_days + 1):
        z = rng.standard_normal((n_paths, k))
        daily = (mu + z @ chol.T) @ w
        paths[:, t] = paths[:, t - 1] * np.exp(daily)
    return paths


def _psd(cov: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    w = np.clip(w, 1e-12, None)
    return v @ np.diag(w) @ v.T
