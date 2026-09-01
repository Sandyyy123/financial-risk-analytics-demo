> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Financial Risk Analytics Demo

A compact, runnable reference implementation of the core engine behind a
market-risk and decision-support system: **Value-at-Risk, Monte Carlo
simulation, VaR backtesting, and scenario stress testing** on a multi-asset
portfolio.

Runs **fully offline** on a deterministic synthetic portfolio (no API keys),
and swaps to **live market data** with one flag.

```bash
pip install -r requirements.txt
python main.py                          # offline synthetic demo
python main.py --live SPY IEF LQD GLD   # live daily data via yfinance
python main.py --value 5000000 --confidence 0.975 --horizon 10
```

## What it computes

| Stage | Module | Method |
|-------|--------|--------|
| 1. Value-at-Risk / Expected Shortfall | `risk/var.py` | Historical (empirical quantile), Parametric (variance-covariance, closed-form ES) |
| 2. Monte Carlo VaR / ES | `risk/monte_carlo.py` | Correlated draws via Cholesky; Gaussian **and** Student-t innovations for fat tails |
| 3. Component VaR attribution | `risk/var.py` | Marginal/component decomposition that sums to the total (risk by sleeve) |
| 4. Backtesting | `risk/backtest.py` | Rolling VaR forecast + Kupiec POF + Christoffersen independence + conditional coverage + Basel traffic-light zone |
| 5. Stress testing | `risk/stress.py` | Historical replay (2008 GFC, 2020 COVID, 2022 rates) + hypothetical shocks |
| 6. Fan / drawdown | `risk/monte_carlo.py` | Simulated multi-day value paths, percentile terminal values, worst intra-path drawdown |

## Architecture

```
                +---------------------+
   returns ---> |   risk.data         |  synthetic (offline) OR yfinance (live)
   panel        +----------+----------+
                           |
        +------------------+------------------+-------------------+
        v                  v                  v                   v
  risk.var           risk.monte_carlo    risk.backtest       risk.stress
  historical VaR     MC VaR (N & t)      Kupiec / Christof.  GFC / COVID /
  parametric VaR     value-path fan      Basel zone          rates / custom
  component VaR      ES / drawdown
        \__________________ main.py __________________/
                    executive risk summary
```

## Sample output (synthetic, seed-fixed)

```
[1] VALUE-AT-RISK / EXPECTED SHORTFALL
  method                VaR %     ES %         VaR $          ES $
  historical            1.54%    1.74%        15,363        17,357
  parametric            1.59%    1.82%        15,874        18,165
  monte_carlo_normal    1.58%    1.82%        15,809        18,209
  monte_carlo_t         1.78%    2.34%        17,784        23,440

[3] VaR MODEL BACKTEST (rolling 250-day historical VaR)
  exceptions : 10  (expected 7.5)   Kupiec p=0.383   Basel: GREEN

[4] SCENARIO STRESS TESTS (worst first)
  GFC_2008     -29.4%   -294,000
  COVID_2020   -24.2%   -242,000
```

## Design notes

- The three independent VaR methods **cross-check** each other; a divergence
  is a finding, not noise (e.g. Student-t VaR sitting well above historical
  flags tail risk the Gaussian model understates).
- Backtesting is the regulatory gate: a model is only trusted once its
  exception rate and clustering pass Kupiec + Christoffersen.
- Everything is vectorised (NumPy) and seed-deterministic, so results are
  reproducible and the same code scales from this 5-asset demo to a full book.

This is a **demonstration scaffold**, not production risk infrastructure —
positions, calendars, P&L attribution, and data-quality controls are
deliberately simplified. It shows the modelling approach and code quality I
bring to a full engagement.

---
Dr. Sandeep Grover — quantitative data scientist (PhD, clinical epidemiology;
Monte Carlo, statistical simulation, and reproducible data pipelines).
