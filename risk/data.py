"""Return-series loader.

Default path builds a deterministic synthetic multi-asset panel (correlated
daily log returns with a fat-tailed shock component) so the whole demo runs
offline with no API keys. Pass real tickers to pull live history via yfinance
when a network connection is available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_ASSETS = ["EQ_US", "EQ_EU", "RATES", "CREDIT", "COMMOD"]


def synthetic_returns(
    assets: list[str] | None = None,
    n_days: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    """Deterministic correlated daily log-return panel with fat tails.

    Returns a (n_days x n_assets) DataFrame indexed by business day.
    """
    assets = assets or DEFAULT_ASSETS
    k = len(assets)
    rng = np.random.default_rng(seed)

    # Plausible annualised drifts / vols per sleeve.
    ann_mu = np.array([0.08, 0.06, 0.02, 0.05, 0.04])[:k]
    ann_vol = np.array([0.18, 0.20, 0.06, 0.10, 0.24])[:k]
    mu = ann_mu / 252.0
    vol = ann_vol / np.sqrt(252.0)

    # A realistic-ish correlation structure (equities co-move, rates hedge).
    base = np.array(
        [
            [1.00, 0.75, -0.30, 0.55, 0.35],
            [0.75, 1.00, -0.25, 0.50, 0.30],
            [-0.30, -0.25, 1.00, -0.20, -0.10],
            [0.55, 0.50, -0.20, 1.00, 0.25],
            [0.35, 0.30, -0.10, 0.25, 1.00],
        ]
    )[:k, :k]
    corr = _nearest_psd(base)
    cov = np.outer(vol, vol) * corr
    chol = np.linalg.cholesky(cov)

    # Gaussian core plus occasional Student-t style jumps for tail risk.
    z = rng.standard_normal((n_days, k))
    jumps = rng.standard_t(df=4, size=(n_days, k)) * (rng.random((n_days, k)) < 0.02)
    shocks = (z + 0.6 * jumps) @ chol.T
    rets = mu + shocks

    idx = pd.bdate_range(end="2026-08-14", periods=n_days)
    return pd.DataFrame(rets, index=idx, columns=assets)


def live_returns(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """Daily log returns from Yahoo Finance. Requires `yfinance` + network."""
    import yfinance as yf  # imported lazily so the offline demo needs no dep

    px = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all").ffill().dropna()
    return np.log(px / px.shift(1)).dropna()


def _nearest_psd(a: np.ndarray) -> np.ndarray:
    """Clip eigenvalues to make a symmetric matrix positive semi-definite."""
    a = (a + a.T) / 2
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 1e-8, None)
    b = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(b))
    return b / np.outer(d, d)
