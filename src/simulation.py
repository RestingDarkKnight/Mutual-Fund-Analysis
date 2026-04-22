"""
simulation.py
-------------
Monte Carlo simulation of portfolio trajectories.

- Fits a multivariate normal (mu, Sigma) on daily log-returns.
- Optionally uses historical bootstrap to preserve fat tails.
- Produces terminal-value distribution and percentile paths for the horizon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class SimConfig:
    n_paths: int = 5000
    horizon_years: float = 5.0
    method: str = "mvn"              # 'mvn' or 'bootstrap'
    seed: int = 42
    percentiles: Tuple[float, ...] = (5, 25, 50, 75, 95)


def simulate_portfolio(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
    initial_investment: float,
    config: Optional[SimConfig] = None,
) -> Dict[str, object]:
    """
    Run a Monte Carlo projection of a portfolio's value.

    Returns
    -------
    {
        'terminal_values': np.ndarray,        # (n_paths,)
        'percentile_paths': pd.DataFrame,     # (days x percentiles)
        'summary': {p5, p25, p50, p75, p95, mean, std},
        'cagr': {p5, p50, p95},
    }
    """
    if config is None:
        config = SimConfig()
    rng = np.random.default_rng(config.seed)

    # Portfolio-level daily return series (from historical weighted sum)
    port_daily = daily_returns.values @ weights  # shape (T,)
    n_days = int(config.horizon_years * TRADING_DAYS)

    if config.method == "mvn" and len(port_daily) > 30:
        mu, sigma = port_daily.mean(), port_daily.std(ddof=1)
        draws = rng.normal(mu, sigma, size=(config.n_paths, n_days))
    else:
        # Block bootstrap from the historical series
        hist = port_daily
        draws = rng.choice(hist, size=(config.n_paths, n_days), replace=True)

    # Cumulative value paths (log-returns sum -> exp)
    cum = np.cumsum(draws, axis=1)
    paths = initial_investment * np.exp(cum)

    # Prepend initial value
    paths = np.concatenate([np.full((config.n_paths, 1), initial_investment), paths], axis=1)

    terminal = paths[:, -1]
    pct_paths = np.percentile(paths, config.percentiles, axis=0)
    pct_df = pd.DataFrame(pct_paths.T, columns=[f"p{int(p)}" for p in config.percentiles])
    pct_df.index.name = "day"

    summary = {
        "mean":  float(terminal.mean()),
        "std":   float(terminal.std(ddof=1)),
        "p5":    float(np.percentile(terminal, 5)),
        "p25":   float(np.percentile(terminal, 25)),
        "p50":   float(np.percentile(terminal, 50)),
        "p75":   float(np.percentile(terminal, 75)),
        "p95":   float(np.percentile(terminal, 95)),
    }
    yrs = config.horizon_years
    cagr = {
        "worst_case_p5": (summary["p5"]  / initial_investment) ** (1 / yrs) - 1,
        "median_p50":    (summary["p50"] / initial_investment) ** (1 / yrs) - 1,
        "best_case_p95": (summary["p95"] / initial_investment) ** (1 / yrs) - 1,
    }
    return {
        "terminal_values": terminal,
        "percentile_paths": pct_df,
        "summary": summary,
        "cagr": {k: float(v) for k, v in cagr.items()},
    }


def max_drawdown_from_paths(paths: np.ndarray) -> np.ndarray:
    """Max drawdown per path (positive fraction)."""
    running_max = np.maximum.accumulate(paths, axis=1)
    dd = (paths - running_max) / running_max
    return -dd.min(axis=1)
