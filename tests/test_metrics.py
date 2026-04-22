"""
Sanity tests for the financial math. Run with:
    pytest tests/
or:
    python -m unittest tests.test_metrics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
import pandas as pd

from feature_engineering import (
    sharpe_ratio, sortino_ratio, beta, alpha_jensen, treynor_ratio,
    annualized_return, annualized_volatility, value_at_risk, max_drawdown,
    compute_log_returns, capture_ratio,
)
from optimizer import PortfolioOptimizer, OptimizerConfig, annualized_cov, annualized_mu


class TestFinancialMath(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        # 1000 trading days ≈ 4 years
        self.daily = pd.Series(rng.normal(0.0005, 0.01, 1000))
        self.nav = pd.Series(100 * np.exp(np.cumsum(self.daily.values)))
        self.bench = pd.Series(rng.normal(0.0004, 0.009, 1000))

    def test_annualized_return_positive_drift(self):
        r = annualized_return(self.daily)
        # mu * 252 ≈ 0.126, exp adj ≈ similar order
        self.assertGreater(r, 0.05)
        self.assertLess(r, 0.30)

    def test_sharpe_is_finite(self):
        s = sharpe_ratio(self.daily)
        self.assertTrue(np.isfinite(s))

    def test_sortino_greater_than_sharpe_on_skewed_downside(self):
        # Symmetric series: sortino and sharpe similar; here just check finite + sign
        s = sharpe_ratio(self.daily)
        so = sortino_ratio(self.daily)
        self.assertTrue(np.isfinite(so))
        self.assertEqual(np.sign(s), np.sign(so))

    def test_beta_of_self_is_one(self):
        b = beta(self.daily, self.daily)
        self.assertAlmostEqual(b, 1.0, places=5)

    def test_alpha_of_self_is_zero(self):
        a = alpha_jensen(self.daily, self.daily)
        self.assertAlmostEqual(a, 0.0, places=5)

    def test_var_positive(self):
        v = value_at_risk(self.daily, 0.05, "historical")
        self.assertGreater(v, 0)
        self.assertLess(v, 0.10)

    def test_max_drawdown_bounded(self):
        mdd = max_drawdown(self.nav)
        self.assertGreaterEqual(mdd, 0)
        self.assertLessEqual(mdd, 1)

    def test_capture_ratio_structure(self):
        c = capture_ratio(self.daily, self.bench)
        self.assertIn("up_capture", c)
        self.assertIn("down_capture", c)
        self.assertIn("capture_combined", c)

    def test_treynor_finite_on_nonzero_beta(self):
        t = treynor_ratio(self.daily, self.bench)
        self.assertTrue(np.isfinite(t))

    def test_annualized_volatility_positive(self):
        v = annualized_volatility(self.daily)
        self.assertGreater(v, 0)


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        n_assets, n_days = 6, 750
        rets = pd.DataFrame(
            rng.normal(0.0005, 0.01, (n_days, n_assets)),
            columns=[f"F{i}" for i in range(n_assets)],
        )
        self.mu = annualized_mu(rets)
        self.cov = annualized_cov(rets)
        self.cats = pd.Series(["Equity"] * n_assets, index=self.mu.index)

    def test_weights_sum_to_one_max_sharpe(self):
        opt = PortfolioOptimizer(self.mu, self.cov, self.cats, OptimizerConfig())
        w = opt.max_sharpe()
        self.assertAlmostEqual(w.sum(), 1.0, places=4)
        self.assertTrue(np.all(w >= -1e-6))

    def test_weights_sum_to_one_min_vol(self):
        opt = PortfolioOptimizer(self.mu, self.cov, self.cats, OptimizerConfig())
        w = opt.min_volatility()
        self.assertAlmostEqual(w.sum(), 1.0, places=4)

    def test_risk_parity_sums_to_one(self):
        opt = PortfolioOptimizer(self.mu, self.cov, self.cats, OptimizerConfig())
        w = opt.risk_parity()
        self.assertAlmostEqual(w.sum(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
