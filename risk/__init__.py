"""Financial risk analytics toolkit: VaR, Monte Carlo, backtesting, stress."""
from .data import synthetic_returns, live_returns, DEFAULT_ASSETS
from .var import (
    portfolio_returns, historical_var, parametric_var, component_var, RiskResult,
)
from .monte_carlo import monte_carlo_var, simulate_paths
from .backtest import rolling_var_forecast, backtest_var, BacktestResult
from .stress import stress_portfolio, hypothetical_shock, HISTORICAL_SCENARIOS

__all__ = [
    "synthetic_returns", "live_returns", "DEFAULT_ASSETS",
    "portfolio_returns", "historical_var", "parametric_var", "component_var",
    "RiskResult", "monte_carlo_var", "simulate_paths",
    "rolling_var_forecast", "backtest_var", "BacktestResult",
    "stress_portfolio", "hypothetical_shock", "HISTORICAL_SCENARIOS",
]
