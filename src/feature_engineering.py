"""
feature_engineering.py
----------------------
Computes the 15 parameters mandated by the spec, grouped into:

Performance
    - Alpha (Jensen's alpha, annualized)
    - Sharpe Ratio
    - Rolling Returns (1Y/3Y/5Y CAGR)
    - Sortino Ratio
    - Treynor Ratio
    - Assets Under Management (AUM) -- metadata passthrough
    - Capture Ratio (Up / Down)

Cost
    - Turnover Ratio       (from metadata; computed where raw holdings are available)
    - Total Expense Ratio  (TER) -- metadata
    - Management Fees      -- metadata
    - Transaction Costs    -- metadata
    - Load Fees            -- exit load from metadata

Risk
    - Beta
    - Standard Deviation (annualized)
    - Value at Risk (VaR, historical 95% and parametric 95%)

All metrics are computed on daily log-returns; annualization factor 252.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.065   # ~6.5% approx Indian 10-yr G-Sec average
RISK_FREE_DAILY = (1 + RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def compute_log_returns(nav: pd.Series) -> pd.Series:
    """Daily log-returns from a NAV series. Drops first NaN."""
    return np.log(nav / nav.shift(1)).dropna()


def _align_returns(fund_ret: pd.Series, bench_ret: pd.Series) -> pd.DataFrame:
    """Align fund and benchmark daily returns on common dates."""
    df = pd.concat([fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1)
    return df.dropna()


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #

def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return from daily log-returns."""
    if len(returns) == 0:
        return np.nan
    total_log = returns.sum()
    years = len(returns) / TRADING_DAYS
    return float(np.exp(total_log / years) - 1) if years > 0 else np.nan


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan


def sharpe_ratio(returns: pd.Series, rf_daily: float = RISK_FREE_DAILY) -> float:
    """
    Sharpe = (annualized excess return) / (annualized vol)
    Computed on daily excess returns to be distribution-agnostic.
    """
    excess = returns - rf_daily
    if len(excess) < 2 or excess.std(ddof=1) == 0:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf_daily: float = RISK_FREE_DAILY) -> float:
    """
    Sortino penalizes only downside deviation.
    Downside deviation = std of min(ret - target, 0).
    """
    excess = returns - rf_daily
    downside = excess.clip(upper=0)
    # Use population-style RMSE for downside to avoid div-by-zero explosions
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def beta(fund_ret: pd.Series, bench_ret: pd.Series) -> float:
    """Classic OLS beta of fund vs benchmark."""
    df = _align_returns(fund_ret, bench_ret)
    if len(df) < 20:
        return np.nan
    cov = np.cov(df["fund"], df["bench"], ddof=1)[0, 1]
    var_b = np.var(df["bench"], ddof=1)
    return float(cov / var_b) if var_b > 0 else np.nan


def alpha_jensen(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rf_daily: float = RISK_FREE_DAILY,
) -> float:
    """
    Jensen's alpha, annualized:
        alpha = (Rf - Rb*beta) mean  -> scaled to annual
    """
    b = beta(fund_ret, bench_ret)
    if np.isnan(b):
        return np.nan
    df = _align_returns(fund_ret, bench_ret)
    excess_fund = df["fund"] - rf_daily
    excess_bench = df["bench"] - rf_daily
    a_daily = (excess_fund - b * excess_bench).mean()
    return float(a_daily * TRADING_DAYS)


def treynor_ratio(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    rf_daily: float = RISK_FREE_DAILY,
) -> float:
    """Treynor = annualized excess return / beta"""
    b = beta(fund_ret, bench_ret)
    if np.isnan(b) or b == 0:
        return np.nan
    excess = fund_ret - rf_daily
    return float(excess.mean() * TRADING_DAYS / b)


def capture_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> Dict[str, float]:
    """
    Up-Capture  = avg(fund_ret | bench_ret > 0)  / avg(bench_ret | bench_ret > 0)
    Down-Capture= avg(fund_ret | bench_ret < 0)  / avg(bench_ret | bench_ret < 0)
    Combined    = Up / Down (higher is better)
    """
    df = _align_returns(fund_ret, bench_ret)
    up = df[df["bench"] > 0]
    down = df[df["bench"] < 0]

    up_cap = (up["fund"].mean() / up["bench"].mean()) if len(up) > 5 and up["bench"].mean() != 0 else np.nan
    down_cap = (down["fund"].mean() / down["bench"].mean()) if len(down) > 5 and down["bench"].mean() != 0 else np.nan
    combined = up_cap / down_cap if (down_cap and down_cap > 0) else np.nan

    return {"up_capture": float(up_cap), "down_capture": float(down_cap), "capture_combined": float(combined)}


def rolling_returns(nav: pd.Series, window_days: int) -> float:
    """
    CAGR over `window_days`. Uses first/last NAV if available data is shorter.
    """
    if len(nav) < 2:
        return np.nan
    window = min(window_days, len(nav) - 1)
    start_nav = nav.iloc[-(window + 1)]
    end_nav = nav.iloc[-1]
    years = window / TRADING_DAYS
    if start_nav <= 0 or years <= 0:
        return np.nan
    return float((end_nav / start_nav) ** (1 / years) - 1)


# --------------------------------------------------------------------------- #
# Risk metrics
# --------------------------------------------------------------------------- #

def value_at_risk(returns: pd.Series, alpha: float = 0.05, method: str = "historical") -> float:
    """
    One-day VaR at confidence level (1 - alpha). Returned as a *positive* loss number.
        method = 'historical' : empirical quantile
        method = 'parametric' : normal approximation (mu, sigma)
    """
    if len(returns) < 30:
        return np.nan
    if method == "historical":
        q = np.quantile(returns, alpha)
        return float(-q)
    if method == "parametric":
        from scipy.stats import norm
        mu, sigma = returns.mean(), returns.std(ddof=1)
        return float(-(mu + sigma * norm.ppf(alpha)))
    raise ValueError(f"Unknown VaR method: {method}")


def max_drawdown(nav: pd.Series) -> float:
    """
    Max drawdown as a positive fraction (e.g., 0.23 = 23% drawdown).
    """
    if len(nav) < 2:
        return np.nan
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    return float(-dd.min())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class FeatureConfig:
    rolling_windows: tuple = (252, 756, 1260)   # 1Y, 3Y, 5Y
    var_method: str = "historical"


def compute_fund_features(
    universe: pd.DataFrame,
    navs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for downstream ML / optimization.
    One row per fund with all 15 parameters + a few supporting metrics.
    """
    if config is None:
        config = FeatureConfig()

    # Pivot NAVs to wide form keyed by date
    nav_wide = navs.pivot(index="date", columns="scheme_code", values="nav").sort_index()
    bench_wide = benchmarks.pivot(index="date", columns="benchmark", values="close").sort_index()

    # Forward-fill for alignment (exchanges close on different holidays sometimes)
    nav_wide = nav_wide.ffill()
    bench_wide = bench_wide.ffill()

    records = []
    for _, meta in universe.iterrows():
        code = str(meta["scheme_code"])
        bench_key = meta["benchmark"]

        if code not in nav_wide.columns:
            continue
        if bench_key not in bench_wide.columns:
            # fall back to Nifty if benchmark missing
            bench_key = "^NSEI" if "^NSEI" in bench_wide.columns else bench_wide.columns[0]

        nav = nav_wide[code].dropna()
        bench = bench_wide[bench_key].dropna()
        if len(nav) < 60:
            continue

        fund_ret = compute_log_returns(nav)
        bench_ret = compute_log_returns(bench)

        # ---- Performance ----
        ann_ret = annualized_return(fund_ret)
        sharpe = sharpe_ratio(fund_ret)
        sortino = sortino_ratio(fund_ret)
        b = beta(fund_ret, bench_ret)
        alpha = alpha_jensen(fund_ret, bench_ret)
        treynor = treynor_ratio(fund_ret, bench_ret)
        caps = capture_ratio(fund_ret, bench_ret)

        rr_1y = rolling_returns(nav, config.rolling_windows[0])
        rr_3y = rolling_returns(nav, config.rolling_windows[1])
        rr_5y = rolling_returns(nav, config.rolling_windows[2])

        # ---- Risk ----
        std_ann = annualized_volatility(fund_ret)
        var95_hist = value_at_risk(fund_ret, 0.05, "historical")
        var95_param = value_at_risk(fund_ret, 0.05, "parametric")
        mdd = max_drawdown(nav)

        # ---- Cost (mostly metadata passthrough) ----
        # Proxy turnover: high for active equity, low for index / debt
        turnover = {
            "Index": 0.25, "Equity": 0.80, "Thematic": 1.20, "Sectoral": 1.30,
            "International": 0.60, "Debt": 0.40, "Commodity": 0.15,
        }.get(meta["category"], 0.70)

        records.append({
            "scheme_code": code,
            "scheme_name": meta["scheme_name"],
            "category": meta["category"],
            "sub_category": meta["sub_category"],
            # performance
            "alpha": alpha,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "treynor_ratio": treynor,
            "rolling_return_1y": rr_1y,
            "rolling_return_3y": rr_3y,
            "rolling_return_5y": rr_5y,
            "aum_cr": float(meta["aum_cr"]),
            "up_capture": caps["up_capture"],
            "down_capture": caps["down_capture"],
            "capture_ratio": caps["capture_combined"],
            "annualized_return": ann_ret,
            # cost
            "turnover_ratio": turnover,
            "ter": float(meta["ter"]),
            "management_fee": float(meta["management_fee"]),
            "transaction_cost": float(meta["transaction_cost"]),
            "exit_load": float(meta["exit_load"]),
            # risk
            "beta": b,
            "std_dev": std_ann,
            "var_95_historical": var95_hist,
            "var_95_parametric": var95_param,
            "max_drawdown": mdd,
        })

    feature_df = pd.DataFrame(records)

    # Within-category z-score normalization for ranking-friendly scores
    def _zscore_within(grp: pd.DataFrame, col: str) -> pd.Series:
        mu, sd = grp[col].mean(), grp[col].std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=grp.index)
        return (grp[col] - mu) / sd

    zscore_cols = ["alpha", "sharpe_ratio", "sortino_ratio", "rolling_return_3y", "std_dev", "max_drawdown"]
    for col in zscore_cols:
        if col not in feature_df.columns:
            continue
        feature_df[f"{col}_z"] = (
            feature_df.groupby("category", group_keys=False).apply(lambda g: _zscore_within(g, col))
        )

    logger.info("Computed features for %d funds", len(feature_df))
    return feature_df


if __name__ == "__main__":
    from data_loader import load_all, LoaderConfig
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_all(LoaderConfig(offline_mode=True))
    features = compute_fund_features(data["universe"], data["navs"], data["benchmarks"])
    print(features[["scheme_name", "category", "sharpe_ratio", "alpha", "beta", "std_dev", "var_95_historical"]].head(10))
