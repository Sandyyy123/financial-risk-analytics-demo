"""VaR backtesting: Kupiec POF and Christoffersen independence/CC tests.

Given a realised P&L series and a rolling one-day VaR forecast, count the
exceptions (loss worse than VaR) and test whether:
  - the exception RATE matches the model's confidence  (Kupiec POF)
  - exceptions are INDEPENDENT / not clustered         (Christoffersen IND)
  - both jointly                                        (conditional coverage)

This is the regulatory (Basel traffic-light) style check a risk model must
pass before it is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BacktestResult:
    n: int
    exceptions: int
    expected: float
    exception_rate: float
    kupiec_stat: float
    kupiec_p: float
    christoffersen_ind_stat: float
    christoffersen_ind_p: float
    cc_stat: float
    cc_p: float
    basel_zone: str


def rolling_var_forecast(port_rets: np.ndarray, window: int = 250,
                         confidence: float = 0.99) -> np.ndarray:
    """One-day-ahead historical VaR using a trailing window.

    Returns an array aligned to port_rets where the first `window` entries are
    NaN (no forecast yet). VaR is loss-positive.
    """
    n = len(port_rets)
    out = np.full(n, np.nan)
    for t in range(window, n):
        w = port_rets[t - window:t]
        out[t] = -np.quantile(w, 1.0 - confidence)
    return out


def backtest_var(port_rets: np.ndarray, var_forecast: np.ndarray,
                 confidence: float = 0.99) -> BacktestResult:
    mask = ~np.isnan(var_forecast)
    r = port_rets[mask]
    v = var_forecast[mask]
    # exception when realised loss exceeds the forecast VaR
    hits = (r < -v).astype(int)
    n = len(hits)
    x = int(hits.sum())
    p = 1.0 - confidence
    expected = n * p

    kupiec_stat, kupiec_p = _kupiec_pof(n, x, p)
    ind_stat, ind_p = _christoffersen_ind(hits)
    cc_stat = kupiec_stat + ind_stat
    cc_p = 1.0 - stats.chi2.cdf(cc_stat, df=2)

    return BacktestResult(
        n=n, exceptions=x, expected=expected, exception_rate=x / n,
        kupiec_stat=kupiec_stat, kupiec_p=kupiec_p,
        christoffersen_ind_stat=ind_stat, christoffersen_ind_p=ind_p,
        cc_stat=cc_stat, cc_p=cc_p,
        basel_zone=_basel_zone(n, x, confidence),
    )


def _kupiec_pof(n: int, x: int, p: float):
    """Kupiec proportion-of-failures likelihood-ratio test (chi2, 1 df)."""
    if x == 0:
        lr = -2.0 * (n * np.log(1 - p))
    else:
        pi = x / n
        lr = -2.0 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - (n - x) * np.log(1 - pi) - x * np.log(pi)
        )
    return lr, 1.0 - stats.chi2.cdf(lr, df=1)


def _christoffersen_ind(hits: np.ndarray):
    """Markov independence test for exception clustering (chi2, 1 df)."""
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(hits[:-1], hits[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    t0, t1 = n00 + n01, n10 + n11
    if t0 == 0 or t1 == 0 or (n01 + n11) == 0:
        return 0.0, 1.0
    pi01 = n01 / t0
    pi11 = n11 / t1
    pi = (n01 + n11) / (t0 + t1)
    if pi in (0.0, 1.0) or pi01 in (0.0,) or pi11 in (0.0,):
        # guard log(0); fall back to no-evidence-of-clustering
        num = (1 - pi) ** (n00 + n10) * pi ** (n01 + n11)
        den = ((1 - pi01) ** n00 * pi01 ** n01
               * (1 - pi11) ** n10 * pi11 ** n11)
        if den <= 0:
            return 0.0, 1.0
        lr = -2.0 * np.log(num / den)
        return lr, 1.0 - stats.chi2.cdf(lr, df=1)
    num = (1 - pi) ** (n00 + n10) * pi ** (n01 + n11)
    den = ((1 - pi01) ** n00 * pi01 ** n01
           * (1 - pi11) ** n10 * pi11 ** n11)
    lr = -2.0 * np.log(num / den)
    return lr, 1.0 - stats.chi2.cdf(lr, df=1)


def _basel_zone(n: int, x: int, confidence: float) -> str:
    """Basel traffic-light zones, scaled from the 250-day / 99% table."""
    scaled = x * (250.0 / n)
    if scaled <= 4:
        return "GREEN"
    if scaled <= 9:
        return "YELLOW"
    return "RED"
