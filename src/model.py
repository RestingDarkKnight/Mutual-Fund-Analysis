"""
model.py
--------
Supervised ML layer that learns relationships between the 15 parameters and:
    - forward annualized return
    - forward annualized volatility
    - forward max drawdown

Models
------
RandomForestRegressor as the default (no heavy deps).
XGBRegressor if xgboost is installed.

Usage
-----
    trainer = ModelTrainer(features_df, navs_df)
    trainer.build_training_set(forward_window_days=252)
    trainer.fit()
    trainer.evaluate()
    trainer.feature_importance()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    compute_log_returns,
    max_drawdown,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

FEATURE_COLS: List[str] = [
    # Performance
    "alpha", "sharpe_ratio", "sortino_ratio", "treynor_ratio",
    "rolling_return_1y", "rolling_return_3y", "aum_cr",
    "up_capture", "down_capture",
    # Cost
    "turnover_ratio", "ter", "management_fee", "transaction_cost", "exit_load",
    # Risk
    "beta", "std_dev", "var_95_historical", "max_drawdown",
]

TARGETS = ["fwd_return", "fwd_volatility", "fwd_max_drawdown"]


@dataclass
class ModelConfig:
    forward_window_days: int = TRADING_DAYS        # predict 1-year forward
    feature_window_days: int = TRADING_DAYS * 3    # use 3Y history for features
    use_xgboost: bool = False
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = 8
    rf_random_state: int = 42


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class ModelTrainer:
    """Builds a rolling-window training set and fits one model per target."""

    def __init__(
        self,
        universe: pd.DataFrame,
        navs: pd.DataFrame,
        benchmarks: pd.DataFrame,
        config: Optional[ModelConfig] = None,
    ):
        self.universe = universe.set_index("scheme_code")
        self.navs = navs
        self.benchmarks = benchmarks
        self.config = config or ModelConfig()

        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.DataFrame] = None
        self.models: Dict[str, object] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

    # --------------------------------------------------------------------- #
    # Dataset construction
    # --------------------------------------------------------------------- #
    def build_training_set(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each fund and each cut-date (quarterly snapshots), compute features
        on the preceding feature_window and targets on the following forward_window.
        """
        from feature_engineering import (
            alpha_jensen, beta, sharpe_ratio, sortino_ratio, treynor_ratio,
            capture_ratio, value_at_risk,
        )

        nav_wide = self.navs.pivot(index="date", columns="scheme_code", values="nav").sort_index().ffill()
        bench_wide = self.benchmarks.pivot(index="date", columns="benchmark", values="close").sort_index().ffill()

        rows_x, rows_y = [], []
        fw = self.config.forward_window_days
        bw = self.config.feature_window_days

        # quarterly snapshot cut-dates — enough samples without leakage
        cut_dates = nav_wide.index[::63][2:]   # skip first 2 quarters for warmup
        for cut in cut_dates:
            # need bw history before AND fw data after
            start_feat = cut - pd.Timedelta(days=int(bw * 1.5))
            end_fwd = cut + pd.Timedelta(days=int(fw * 1.5))
            if start_feat < nav_wide.index.min() or end_fwd > nav_wide.index.max():
                continue

            for code in nav_wide.columns:
                meta = self.universe.loc[code] if code in self.universe.index else None
                if meta is None:
                    continue
                bench_key = meta["benchmark"]
                if bench_key not in bench_wide.columns:
                    bench_key = "^NSEI"

                hist_nav = nav_wide[code].loc[start_feat:cut].dropna()
                fwd_nav = nav_wide[code].loc[cut:end_fwd].dropna()
                hist_bench = bench_wide[bench_key].loc[start_feat:cut].dropna()
                if len(hist_nav) < 120 or len(fwd_nav) < 60:
                    continue

                hist_ret = compute_log_returns(hist_nav)
                bench_ret = compute_log_returns(hist_bench)
                fwd_ret = compute_log_returns(fwd_nav)

                caps = capture_ratio(hist_ret, bench_ret)

                feat = {
                    "alpha":            alpha_jensen(hist_ret, bench_ret),
                    "sharpe_ratio":     sharpe_ratio(hist_ret),
                    "sortino_ratio":    sortino_ratio(hist_ret),
                    "treynor_ratio":    treynor_ratio(hist_ret, bench_ret),
                    "rolling_return_1y":annualized_return(hist_ret.tail(TRADING_DAYS)),
                    "rolling_return_3y":annualized_return(hist_ret),
                    "aum_cr":           float(meta["aum_cr"]),
                    "up_capture":       caps["up_capture"],
                    "down_capture":     caps["down_capture"],
                    "turnover_ratio":   {"Index":0.25,"Equity":0.80,"Thematic":1.20,"Sectoral":1.30,
                                        "International":0.60,"Debt":0.40,"Commodity":0.15}
                                        .get(meta["category"], 0.70),
                    "ter":              float(meta["ter"]),
                    "management_fee":   float(meta["management_fee"]),
                    "transaction_cost": float(meta["transaction_cost"]),
                    "exit_load":        float(meta["exit_load"]),
                    "beta":             beta(hist_ret, bench_ret),
                    "std_dev":          annualized_volatility(hist_ret),
                    "var_95_historical":value_at_risk(hist_ret, 0.05, "historical"),
                    "max_drawdown":     max_drawdown(hist_nav),
                    "scheme_code":      code,
                    "cut_date":         cut,
                }
                tgt = {
                    "fwd_return":       annualized_return(fwd_ret),
                    "fwd_volatility":   annualized_volatility(fwd_ret),
                    "fwd_max_drawdown": max_drawdown(fwd_nav),
                    "scheme_code":      code,
                    "cut_date":         cut,
                }
                rows_x.append(feat)
                rows_y.append(tgt)

        X = pd.DataFrame(rows_x).dropna(subset=FEATURE_COLS).reset_index(drop=True)
        y = pd.DataFrame(rows_y).reset_index(drop=True)
        # align on (scheme_code, cut_date)
        key = ["scheme_code", "cut_date"]
        y = y.merge(X[key], on=key, how="inner")
        X = X.merge(y[key], on=key, how="inner")
        X = X.sort_values(key).reset_index(drop=True)
        y = y.sort_values(key).reset_index(drop=True)

        self.X, self.y = X, y
        logger.info("Built training set: X=%s, y=%s", X.shape, y.shape)
        return X, y

    # --------------------------------------------------------------------- #
    # Model
    # --------------------------------------------------------------------- #
    def _make_estimator(self):
        if self.config.use_xgboost:
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.config.rf_random_state,
                    n_jobs=-1, verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost not installed, falling back to RandomForest")
        return RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=3,
            random_state=self.config.rf_random_state,
            n_jobs=-1,
        )

    def fit(self) -> None:
        assert self.X is not None and self.y is not None, "Call build_training_set() first"
        X_mat = self.X[FEATURE_COLS].values

        for target in TARGETS:
            y_vec = self.y[target].values
            mask = ~np.isnan(y_vec)
            if mask.sum() < 20:
                logger.warning("Skipping %s: insufficient data", target)
                continue
            model = self._make_estimator()
            model.fit(X_mat[mask], y_vec[mask])
            self.models[target] = model
            logger.info("Trained model for %s on %d samples", target, mask.sum())

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Time-series CV evaluation. Returns metrics dict."""
        assert self.X is not None and self.y is not None
        X_mat = self.X[FEATURE_COLS].values
        tscv = TimeSeriesSplit(n_splits=4)

        out: Dict[str, Dict[str, float]] = {}
        for target in TARGETS:
            y_vec = self.y[target].values
            mask = ~np.isnan(y_vec)
            X_m, y_m = X_mat[mask], y_vec[mask]
            if len(y_m) < 20:
                continue
            r2s, maes = [], []
            for tr, te in tscv.split(X_m):
                if len(tr) < 5 or len(te) < 5:
                    continue
                m = self._make_estimator()
                m.fit(X_m[tr], y_m[tr])
                pred = m.predict(X_m[te])
                r2s.append(r2_score(y_m[te], pred))
                maes.append(mean_absolute_error(y_m[te], pred))
            out[target] = {"r2_mean": float(np.mean(r2s)) if r2s else np.nan,
                           "mae_mean": float(np.mean(maes)) if maes else np.nan,
                           "n_samples": int(len(y_m))}
        self.metrics = out
        logger.info("Evaluation: %s", out)
        return out

    def feature_importance(self) -> pd.DataFrame:
        """Return a feature-importance table per target."""
        rows = []
        for tgt, m in self.models.items():
            imp = getattr(m, "feature_importances_", None)
            if imp is None:
                continue
            for name, val in zip(FEATURE_COLS, imp):
                rows.append({"target": tgt, "feature": name, "importance": float(val)})
        return pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])

    def predict_for_current(self, current_features: pd.DataFrame) -> pd.DataFrame:
        """Score the latest feature snapshot per fund and return predictions."""
        assert self.models, "Call fit() first"
        X_now = current_features[FEATURE_COLS].values
        preds = current_features[["scheme_code"]].copy()
        for tgt, m in self.models.items():
            preds[f"pred_{tgt}"] = m.predict(X_now)
        return preds
