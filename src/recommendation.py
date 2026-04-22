"""
recommendation.py
-----------------
Ties everything together. Given user input:
    - investment_amount (INR)
    - horizon_years
    - risk_appetite in {Low, Medium, High}
    - preference (optional: 'equity-heavy', 'debt-heavy', 'gold', 'balanced', 'psu', None)

produces FIVE portfolio recommendations:
    1. Maximum Returns    (highest mu, respects category caps)
    2. Safest             (min vol)
    3. Balanced           (max Sharpe)
    4. Income Stability   (debt-heavy, low vol)
    5. Aggressive Growth  (equity + thematic, higher vol tolerance)

Each portfolio ships with: allocations, expected return, vol, Sharpe, max drawdown,
historical VaR, and an MC-based projection (best / worst / median).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from feature_engineering import max_drawdown, value_at_risk, compute_log_returns
from optimizer import (
    PortfolioOptimizer, OptimizerConfig,
    annualized_cov, annualized_mu, build_return_matrix,
)
from simulation import simulate_portfolio, SimConfig

logger = logging.getLogger(__name__)

# Broad category filters per risk appetite
RISK_APPETITE_FILTERS: Dict[str, Dict[str, float]] = {
    "Low":    {"min_debt_weight": 0.50, "max_equity_weight": 0.30, "target_vol": 0.08},
    "Medium": {"min_debt_weight": 0.20, "max_equity_weight": 0.70, "target_vol": 0.15},
    "High":   {"min_debt_weight": 0.00, "max_equity_weight": 1.00, "target_vol": 0.25},
}


@dataclass
class UserInput:
    investment_amount: float
    horizon_years: float
    risk_appetite: str = "Medium"        # Low / Medium / High
    preference: Optional[str] = None     # 'equity-heavy' | 'debt-heavy' | 'gold' | 'psu' | None

    def __post_init__(self):
        if self.risk_appetite not in RISK_APPETITE_FILTERS:
            raise ValueError(f"risk_appetite must be one of {list(RISK_APPETITE_FILTERS)}")


@dataclass
class PortfolioResult:
    name: str
    allocations: pd.DataFrame           # columns: scheme_name, category, weight, amount
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    projection: Dict[str, object]       # output of simulate_portfolio


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _filter_universe(features: pd.DataFrame, user: UserInput) -> pd.DataFrame:
    """
    Pre-filter the fund universe based on user preference and risk appetite.
    Keeps enough funds across categories to let the optimizer diversify.
    """
    df = features.copy()

    # Drop funds with missing key metrics
    df = df.dropna(subset=["annualized_return", "std_dev"])

    if user.preference == "equity-heavy":
        df = df[df["category"].isin(["Equity", "Index", "International", "Thematic", "Sectoral"])]
    elif user.preference == "debt-heavy":
        df = df[df["category"].isin(["Debt", "Commodity"])]
    elif user.preference == "gold":
        df = df[df["sub_category"].str.contains("Gold|Silver", case=False, na=False)
                | (df["category"] == "Debt")]
    elif user.preference == "psu":
        df = df[df["sub_category"].str.contains("PSU", case=False, na=False)
                | (df["category"] == "Debt")]

    # Risk-appetite gates
    if user.risk_appetite == "Low":
        df = df[df["category"].isin(["Debt", "Commodity", "Index"])
                | (df["std_dev"] < 0.15)]
    elif user.risk_appetite == "Medium":
        df = df[df["std_dev"] < 0.30]

    if len(df) < 4:
        logger.warning("Filter too strict (%d funds) — relaxing", len(df))
        df = features.dropna(subset=["annualized_return", "std_dev"])
    return df.reset_index(drop=True)


def _allocation_frame(
    universe: pd.DataFrame,
    codes: List[str],
    weights: np.ndarray,
    investment_amount: float,
) -> pd.DataFrame:
    meta = universe.set_index("scheme_code").loc[codes].reset_index()
    df = pd.DataFrame({
        "scheme_code": codes,
        "scheme_name": meta["scheme_name"].values,
        "category": meta["category"].values,
        "sub_category": meta["sub_category"].values,
        "weight": weights,
        "amount_inr": weights * investment_amount,
    })
    df = df[df["weight"] > 0.015].sort_values("weight", ascending=False).reset_index(drop=True)
    # renormalize after pruning tiny weights
    df["weight"] = df["weight"] / df["weight"].sum()
    df["amount_inr"] = df["weight"] * investment_amount
    return df


def _portfolio_metrics(weights: np.ndarray,
                       daily_returns: pd.DataFrame,
                       mu: pd.Series,
                       cov: pd.DataFrame,
                       rf: float) -> Dict[str, float]:
    ann_ret = float(weights @ mu.values)
    ann_vol = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown on historical weighted portfolio path
    port_daily = daily_returns.values @ weights
    equity_curve = pd.Series(np.exp(np.cumsum(port_daily)))
    mdd = max_drawdown(equity_curve)
    var95 = value_at_risk(pd.Series(port_daily), 0.05, "historical")
    return {
        "expected_return": ann_ret, "volatility": ann_vol, "sharpe": sharpe,
        "max_drawdown": mdd, "var_95": var95,
    }


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #

class RecommendationEngine:
    def __init__(
        self,
        universe: pd.DataFrame,
        features: pd.DataFrame,
        navs: pd.DataFrame,
        risk_free_rate: float = 0.065,
    ):
        self.universe = universe
        self.features = features
        self.navs = navs
        self.rf = risk_free_rate

    def _build_optimizer(self, fund_codes: List[str],
                         cfg: OptimizerConfig) -> (PortfolioOptimizer, pd.DataFrame, pd.Series, pd.DataFrame):
        rets = build_return_matrix(self.navs, fund_codes)
        cov = annualized_cov(rets)
        mu = annualized_mu(rets)
        cats = self.universe.set_index("scheme_code")["category"]
        opt = PortfolioOptimizer(mu=mu, cov=cov, categories=cats, config=cfg)
        return opt, rets, mu, cov

    def _pack(self, name: str, fund_codes: List[str], weights: np.ndarray,
              rets: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame,
              user: UserInput) -> PortfolioResult:
        metrics = _portfolio_metrics(weights, rets, mu, cov, self.rf)
        alloc = _allocation_frame(self.universe, fund_codes, weights, user.investment_amount)

        # Re-run metrics on the pruned / renormalized weights so the UI agrees
        kept_codes = alloc["scheme_code"].tolist()
        kept_w = alloc["weight"].values
        rets_kept = rets[kept_codes]
        mu_kept = mu.loc[kept_codes]
        cov_kept = cov.loc[kept_codes, kept_codes]
        final_metrics = _portfolio_metrics(kept_w, rets_kept, mu_kept, cov_kept, self.rf)

        sim_cfg = SimConfig(n_paths=4000, horizon_years=user.horizon_years,
                            method="mvn", seed=42)
        projection = simulate_portfolio(rets_kept, kept_w, user.investment_amount, sim_cfg)

        return PortfolioResult(
            name=name,
            allocations=alloc,
            expected_return=final_metrics["expected_return"],
            volatility=final_metrics["volatility"],
            sharpe_ratio=final_metrics["sharpe"],
            max_drawdown=final_metrics["max_drawdown"],
            var_95=final_metrics["var_95"],
            projection=projection,
        )

    def recommend(self, user: UserInput) -> Dict[str, PortfolioResult]:
        filtered = _filter_universe(self.features, user)
        logger.info("Filtered universe: %d funds", len(filtered))
        fund_codes = filtered["scheme_code"].tolist()

        # -- Portfolio 1: Maximum Returns -- relaxed caps, higher single-fund cap
        cfg_max = OptimizerConfig(risk_free_rate=self.rf, w_max=0.50,
                                  category_caps={"Sectoral": 0.30, "Thematic": 0.35})
        opt_max, rets_max, mu_max, cov_max = self._build_optimizer(fund_codes, cfg_max)
        w_max = opt_max.max_return()
        max_returns = self._pack("Maximum Returns", fund_codes, w_max,
                                 rets_max, mu_max, cov_max, user)

        # -- Portfolio 2: Safest (min vol) -- use low-vol universe
        safe_filter = filtered[filtered["std_dev"] < 0.20] if user.risk_appetite != "High" else filtered
        if len(safe_filter) < 4:
            safe_filter = filtered
        safe_codes = safe_filter["scheme_code"].tolist()
        cfg_safe = OptimizerConfig(risk_free_rate=self.rf, w_max=0.40)
        opt_safe, rets_safe, mu_safe, cov_safe = self._build_optimizer(safe_codes, cfg_safe)
        w_safe = opt_safe.min_volatility()
        safest = self._pack("Safest (Min Volatility)", safe_codes, w_safe,
                            rets_safe, mu_safe, cov_safe, user)

        # -- Portfolio 3: Balanced (max Sharpe)
        cfg_bal = OptimizerConfig(risk_free_rate=self.rf, w_max=0.35)
        opt_bal, rets_bal, mu_bal, cov_bal = self._build_optimizer(fund_codes, cfg_bal)
        w_bal = opt_bal.max_sharpe()
        balanced = self._pack("Balanced (Max Sharpe)", fund_codes, w_bal,
                              rets_bal, mu_bal, cov_bal, user)

        # -- Portfolio 4: Income Stability -- debt + liquid heavy
        income_mask = filtered["category"].isin(["Debt", "Commodity"]) | (filtered["std_dev"] < 0.12)
        income_filter = filtered[income_mask]
        if len(income_filter) < 3:
            income_filter = filtered[filtered["std_dev"] < filtered["std_dev"].median()]
        inc_codes = income_filter["scheme_code"].tolist()
        cfg_inc = OptimizerConfig(risk_free_rate=self.rf, w_max=0.45)
        opt_inc, rets_inc, mu_inc, cov_inc = self._build_optimizer(inc_codes, cfg_inc)
        w_inc = opt_inc.min_volatility()
        income = self._pack("Income Stability", inc_codes, w_inc,
                            rets_inc, mu_inc, cov_inc, user)

        # -- Portfolio 5: Aggressive Growth -- equity/thematic, target higher return
        agg_filter = filtered[filtered["category"].isin(
            ["Equity", "Thematic", "Sectoral", "International", "Index"])]
        if len(agg_filter) < 4:
            agg_filter = filtered
        agg_codes = agg_filter["scheme_code"].tolist()
        cfg_agg = OptimizerConfig(risk_free_rate=self.rf, w_max=0.40,
                                  category_caps={"Sectoral": 0.30, "Thematic": 0.35,
                                                 "International": 0.30})
        opt_agg, rets_agg, mu_agg, cov_agg = self._build_optimizer(agg_codes, cfg_agg)
        # Target return = 75th percentile of fund returns
        r_target = float(np.nanpercentile(mu_agg.values, 75))
        w_agg = opt_agg.target_return(r_target)
        aggressive = self._pack("Aggressive Growth", agg_codes, w_agg,
                                rets_agg, mu_agg, cov_agg, user)

        return {
            "Maximum Returns":         max_returns,
            "Safest":                  safest,
            "Balanced":                balanced,
            "Income Stability":        income,
            "Aggressive Growth":       aggressive,
        }
