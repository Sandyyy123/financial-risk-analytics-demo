#!/usr/bin/env python3
"""End-to-end financial risk analytics demo.

Runs entirely offline on a deterministic synthetic multi-asset portfolio:

  1. Load correlated daily return history (synthetic, or --live tickers)
  2. Value-at-Risk / Expected Shortfall by three methods
        historical | parametric | Monte Carlo (Gaussian & Student-t)
  3. Component VaR attribution across sleeves
  4. Backtest the rolling VaR model (Kupiec + Christoffersen + Basel zone)
  5. Scenario stress tests (2008 GFC, 2020 COVID, 2022 rates, vol spike)
  6. Monte Carlo 10-day fan (5th/50th/95th percentile terminal value)

Usage:
    python main.py                       # offline synthetic demo
    python main.py --live SPY IEF LQD GLD # live data via yfinance (network)
    python main.py --value 5000000 --confidence 0.975
"""
from __future__ import annotations

import argparse

import numpy as np

import risk


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Financial risk analytics demo")
    ap.add_argument("--live", nargs="+", metavar="TICKER",
                    help="pull live daily history for these tickers (yfinance)")
    ap.add_argument("--value", type=float, default=1_000_000.0,
                    help="portfolio value in base currency")
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=1, help="VaR horizon in days")
    ap.add_argument("--sims", type=int, default=100_000)
    args = ap.parse_args()

    if args.live:
        rets = risk.live_returns(args.live)
        assets = list(rets.columns)
    else:
        rets = risk.synthetic_returns()
        assets = list(rets.columns)

    weights = np.repeat(1.0 / len(assets), len(assets))  # equal weight demo
    R = rets.to_numpy()
    port = risk.portfolio_returns(R, weights)

    c, h, pv = args.confidence, args.horizon, args.value
    print("=" * 66)
    print(f"  FINANCIAL RISK ANALYTICS  |  {len(R)} obs x {len(assets)} assets")
    print(f"  Portfolio value {_fmt(pv)}  |  {int(c*100)}% VaR  |  {h}-day horizon")
    print("=" * 66)

    hv = risk.historical_var(port, c, h, pv)
    pvr = risk.parametric_var(port, c, h, pv)
    mcn = risk.monte_carlo_var(R, weights, c, h, args.sims, "normal", portfolio_value=pv)
    mct = risk.monte_carlo_var(R, weights, c, h, args.sims, "t", portfolio_value=pv)

    print("\n[1] VALUE-AT-RISK / EXPECTED SHORTFALL")
    print(f"  {'method':<18}{'VaR %':>9}{'ES %':>9}{'VaR '+chr(36):>14}{'ES '+chr(36):>14}")
    for r in (hv, pvr, mcn, mct):
        print(f"  {r.method:<18}{r.var*100:>8.2f}%{r.es*100:>8.2f}%"
              f"{_fmt(r.var_value):>14}{_fmt(r.es_value):>14}")

    print("\n[2] COMPONENT VaR ATTRIBUTION (parametric, sums to total)")
    comp = risk.component_var(R, weights, c, pv)
    for a, cv in zip(assets, comp["component_value"]):
        share = cv / comp["total_value"] * 100 if comp["total_value"] else 0
        print(f"  {a:<10}{_fmt(cv):>14}{share:>8.1f}%")
    print(f"  {'TOTAL':<10}{_fmt(comp['total_value']):>14}")

    print("\n[3] VaR MODEL BACKTEST (rolling 250-day historical VaR)")
    vf = risk.rolling_var_forecast(port, window=250, confidence=c)
    bt = risk.backtest_var(port, vf, confidence=c)
    print(f"  observations tested : {bt.n}")
    print(f"  exceptions          : {bt.exceptions}  (expected {bt.expected:.1f})")
    print(f"  exception rate      : {bt.exception_rate*100:.2f}%  "
          f"(target {(1-c)*100:.2f}%)")
    print(f"  Kupiec POF          : LR={bt.kupiec_stat:.2f}  p={bt.kupiec_p:.3f}")
    print(f"  Christoffersen IND  : LR={bt.christoffersen_ind_stat:.2f}  "
          f"p={bt.christoffersen_ind_p:.3f}")
    print(f"  Conditional coverage: LR={bt.cc_stat:.2f}  p={bt.cc_p:.3f}")
    print(f"  Basel traffic-light : {bt.basel_zone}")

    print("\n[4] SCENARIO STRESS TESTS (worst first)")
    for s in risk.stress_portfolio(weights, pv):
        print(f"  {s.scenario:<14}{s.pnl_pct*100:>8.1f}%{_fmt(s.pnl_value):>16}")

    print("\n[5] MONTE CARLO 10-DAY TERMINAL VALUE (5000 paths)")
    paths = risk.simulate_paths(R, weights, horizon_days=10, n_paths=5000,
                                portfolio_value=pv)
    term = paths[:, -1]
    for p_ in (5, 50, 95):
        print(f"  p{p_:<3d}: {_fmt(np.percentile(term, p_))}")
    worst_dd = (paths.min(axis=1) / pv - 1).min() * 100
    print(f"  worst simulated intra-path drawdown: {worst_dd:.1f}%")

    print("\n" + "=" * 66)
    print("  Deterministic synthetic data — illustrative only. Swap in --live")
    print("  tickers or a positions file to run on a real book.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
