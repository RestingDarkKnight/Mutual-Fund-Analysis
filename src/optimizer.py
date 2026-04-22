"""
optimizer.py
------------
Portfolio optimization layer.

Implements
----------
- Maximum Sharpe portfolio     (tangency portfolio)
- Minimum volatility portfolio
- Mean-variance efficient at a target return
- Risk parity portfolio        (equal risk contribution)
- Long-only, fully-invested constraints by default

Inputs
------
returns_df : pd.DataFrame  -- daily log-returns, one column per fund
cov        : pd.DataFrame  -- annualized covariance matrix
mu         : pd.Series     -- annualized expected returns

Design notes
------------
- Uses scipy.optimize.minimize with SLSQP (standard for constrained MV).
- Covariance is Ledoit-Wolf shrunk when >=2 funds to avoid near-singular issues
  on small histories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def build_return_matrix(navs: pd.DataFrame, fund_list: List[str]) -> pd.DataFrame:
    """Return wide-format daily log-returns for the given fund codes."""
    nav_wide = navs.pivot(index="date", columns="scheme_code", values="nav").sort_index()
    cols = [c for c in fund_list if c in nav_wide.columns]
    nav_wide = nav_wide[cols].ffill().dropna(how="all")
    rets = np.log(nav_wide / nav_wide.shift(1)).dropna()
    return rets


def annualized_cov(returns: pd.DataFrame, shrink: bool = True) -> pd.DataFrame:
    """Annualized covariance matrix with optional Ledoit-Wolf shrinkage."""
    if shrink and returns.shape[1] > 1 and len(returns) > 20:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        except Exception:
            cov = returns.cov()
    else:
        cov = returns.cov()
    return cov * TRADING_DAYS


def annualized_mu(returns: pd.DataFrame) -> pd.Series:
    """Annualized mean returns (from daily log-returns)."""
    return returns.mean() * TRADING_DAYS


# --------------------------------------------------------------------------- #
# Portfolio math
# --------------------------------------------------------------------------- #

def port_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def port_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def port_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    vol = port_vol(weights, cov)
    if vol == 0:
        return -1e9
    return (port_return(weights, mu) - rf) / vol


# --------------------------------------------------------------------------- #
# Optimizers
# --------------------------------------------------------------------------- #

@dataclass
class OptimizerConfig:
    risk_free_rate: float = 0.065
    w_min: float = 0.00
    w_max: float = 0.40      # cap single-fund allocation at 40% for diversification
    category_caps: Dict[str, float] = field(default_factory=lambda: {
        "Sectoral": 0.20, "Thematic": 0.25, "International": 0.25, "Commodity": 0.20,
    })


class PortfolioOptimizer:
    def __init__(self, mu: pd.Series, cov: pd.DataFrame,
                 categories: Optional[pd.Series] = None,
                 config: Optional[OptimizerConfig] = None):
        self.mu = mu
        self.cov = cov
        self.fund_codes = list(mu.index)
        self.n = len(self.fund_codes)
        self.categories = categories if categories is not None else pd.Series("Unknown", index=self.fund_codes)
        self.config = config or OptimizerConfig()

    # -- constraint builders ----------------------------------------------- #
    def _bounds(self) -> List[Tuple[float, float]]:
        return [(self.config.w_min, self.config.w_max)] * self.n

    def _constraints(self) -> List[dict]:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        for cat, cap in self.config.category_caps.items():
            idx = [i for i, c in enumerate(self.fund_codes) if self.categories.get(c) == cat]
            if idx:
                cons.append({
                    "type": "ineq",
                    "fun": (lambda w, idx=idx, cap=cap: cap - np.sum([w[i] for i in idx])),
                })
        return cons

    def _x0(self) -> np.ndarray:
        return np.ones(self.n) / self.n

    # -- objective functions ----------------------------------------------- #
    def max_sharpe(self) -> np.ndarray:
        mu_a, cov_a = self.mu.values, self.cov.values
        rf = self.config.risk_free_rate

        def neg_sharpe(w):
            return -port_sharpe(w, mu_a, cov_a, rf)

        res = minimize(neg_sharpe, self._x0(), method="SLSQP",
                       bounds=self._bounds(), constraints=self._constraints(),
                       options={"maxiter": 400, "ftol": 1e-9})
        return res.x if res.success else self._x0()

    def min_volatility(self) -> np.ndarray:
        cov_a = self.cov.values

        def vol(w):
            return port_vol(w, cov_a)

        res = minimize(vol, self._x0(), method="SLSQP",
                       bounds=self._bounds(), constraints=self._constraints(),
                       options={"maxiter": 400, "ftol": 1e-9})
        return res.x if res.success else self._x0()

    def max_return(self) -> np.ndarray:
        """Upper bound — unconstrained by vol, so respects only bounds+caps."""
        mu_a = self.mu.values

        def neg_ret(w):
            return -port_return(w, mu_a)

        res = minimize(neg_ret, self._x0(), method="SLSQP",
                       bounds=self._bounds(), constraints=self._constraints(),
                       options={"maxiter": 400, "ftol": 1e-9})
        return res.x if res.success else self._x0()

    def target_return(self, r_target: float) -> np.ndarray:
        """Min variance subject to meeting a target return."""
        mu_a, cov_a = self.mu.values, self.cov.values
        cons = self._constraints() + [{"type": "ineq",
                                       "fun": lambda w: w @ mu_a - r_target}]

        def vol(w):
            return port_vol(w, cov_a)

        res = minimize(vol, self._x0(), method="SLSQP",
                       bounds=self._bounds(), constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-9})
        return res.x if res.success else self.min_volatility()

    def risk_parity(self) -> np.ndarray:
        """
        Equal Risk Contribution (ERC):
            RC_i = w_i * (Sigma w)_i ;  minimize variance of RC across assets.
        """
        cov_a = self.cov.values

        def risk_budget_obj(w):
            w = np.abs(w)
            w = w / w.sum()
            port_variance = w @ cov_a @ w
            if port_variance <= 0:
                return 1e6
            marginal = cov_a @ w
            rc = w * marginal
            target = port_variance / self.n
            return np.sum((rc - target) ** 2)

        res = minimize(risk_budget_obj, self._x0(), method="SLSQP",
                       bounds=self._bounds(), constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                       options={"maxiter": 500, "ftol": 1e-10})
        return res.x if res.success else self._x0()

    # -- convenience ------------------------------------------------------- #
    def summarize(self, weights: np.ndarray) -> Dict[str, float]:
        mu_a, cov_a = self.mu.values, self.cov.values
        return {
            "expected_return": port_return(weights, mu_a),
            "volatility":      port_vol(weights, cov_a),
            "sharpe":          port_sharpe(weights, mu_a, cov_a, self.config.risk_free_rate),
        }
