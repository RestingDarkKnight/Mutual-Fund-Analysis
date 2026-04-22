# Mutual Fund Analysis Project:

A production-grade Python system that analyzes Indian mutual funds across **15 quantitative parameters** categorized into 3 types (performance, cost, risk), learns relationships between those parameters and forward performance using ML, and generates **five portfolio recommendations** tailored to user risk appetite and horizon.

Runs end-to-end with a single command. Works online (live AMFI + Yahoo Finance) or fully offline (deterministic synthetic data) so the pipeline never breaks.

---

## Problem Statement

Indian retail investors face **1,400+ open-ended mutual fund schemes** across a dozen categories. Choosing the right mix requires reasoning simultaneously about:

- **Performance** — is alpha real or luck?
- **Cost drag** — will TER + turnover erode forward returns?
- **Risk exposure** — how does this behave in a drawdown?
- **Diversification** — do my holdings actually de-correlate?

This project automates that reasoning. It ingests NAV history, computes the 15 parameters mandated by the spec, learns which parameters predict forward performance, and constructs optimized portfolios.

---

## Workflow Architecture

```
User Input (amount, horizon, risk)
          │
          ▼
┌─────────────────────┐      ┌──────────────────────┐
│   Data Ingestion    │ ───► │  Feature Engineering │
│  (AMFI + yfinance)  │      │    (15 parameters)   │
└─────────────────────┘      └──────────────────────┘
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                    ┌──────────────────┐  ┌─────────────────┐
                    │    ML Layer      │  │   Optimizer     │
                    │  (RF / XGBoost)  │  │ (SLSQP, MV,     │
                    │                  │  │  Max Sharpe,    │
                    │                  │  │  Risk Parity)   │
                    └──────────────────┘  └─────────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │  Monte Carlo     │
                              │  Projection      │
                              └──────────────────┘
                                       │
                                       ▼
                      5 Portfolios + Metrics + Projections
```

---

## The 15 Parameters: (Thanks!! Zero 1 School)

### Performance (7)
| Metric | Definition |
|---|---|
| **Alpha** | Jensen's alpha, annualized: `(R_f − R_b·β)` vs benchmark |
| **Sharpe Ratio** | `(R_p − R_f) / σ_p`, annualized |
| **Rolling Returns** | CAGR over 1Y / 3Y / 5Y windows |
| **Sortino Ratio** | Excess return / downside deviation |
| **Treynor Ratio** | Excess return / β |
| **AUM** | Scheme assets under management |
| **Capture Ratio** | Up-capture / Down-capture |

### Cost (5)
| Metric | Source |
|---|---|
| **Turnover Ratio** | Category proxy (0.25 index → 1.30 sectoral) |
| **Total Expense Ratio** | AMFI scheme disclosure |
| **Management Costs** | AMFI |
| **Transaction Cost** | AMFI |
| **Load Fees** | Exit load from scheme info doc |

### Risk (3)
| Metric | Definition |
|---|---|
| **Beta** | OLS β vs benchmark on daily log-returns |
| **Std Deviation** | Annualized σ of daily returns |
| **Value at Risk (VaR)** | 95% historical + parametric (1-day) |

All metrics are computed on daily log-returns with 252-day annualization and a 6.5% risk-free rate (Indian 10-year Govt.-Securities average).

---

## Selected Funds for MF Portfolio (36 schemes)

| Category | Examples |
|---|---|
| Index | UTI Nifty 50, HDFC Nifty 50, ICICI Next 50 |
| Large Cap | SBI Bluechip, ICICI Bluechip, Axis Bluechip, Mirae Large Cap |
| Mid Cap | Axis Midcap, Kotak Emerging, HDFC Mid-Cap |
| Small Cap | Nippon Small Cap, SBI Small Cap |
| Flexi Cap | Parag Parikh, HDFC Flexi, Kotak Flexicap |
| Multi Cap | Nippon Multi Cap, Mahindra Manulife Multi Cap |
| International | Motilal S&P 500, ICICI US Bluechip |
| Debt (Liquid) | SBI Liquid, HDFC Liquid, ICICI Liquid |
| Debt (Short) | HDFC Short Term, ICICI Short Term |
| Debt (Long) | SBI Magnum Income, ICICI Long Term Bond |
| Commodity | SBI Gold, Nippon Gold, ICICI Silver |
| Thematic | SBI PSU, Invesco PSU, ICICI Infra, Nippon Power & Infra |
| Sectoral | SBI Banking, ICICI Pharma, Nippon Pharma |

The universe CSV (`data/raw/fund_universe.csv`) ships with AMFI scheme codes, TER, AUM, management fees, exit loads, and benchmark mapping. Add your own rows to extend.

---

## Installation & Run (one command)

```bash
git clone <repo-url>
cd mutual-fund-analyzer
./setup.sh --offline         # skip if you have network access to AMFI + Yahoo
```

That's it. The script creates a venv, installs dependencies, ensures data directories exist, and runs the full pipeline with a default profile (₹5 L, 5 years, Medium risk).

### Manual setup (alternative)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --amount 1000000 --horizon 7 --risk Medium
```

---

## Usage

### Interactive

```bash
python main.py
```

```
  INDIAN MUTUAL FUND ANALYZER  ·  Portfolio Recommendation Engine

  Enter your investment profile:

  Investment amount in INR (e.g. 500000): 1000000
  Investment horizon in years (1-30): 7
  Risk appetite [Low/Medium/High] (default Medium): High
  Preference options: balanced, equity-heavy, debt-heavy, gold, psu (or leave blank)
  Preference: equity-heavy
```

### Flags

```bash
python main.py --amount 500000 --horizon 5 --risk Low
python main.py --amount 2000000 --horizon 10 --risk High --preference equity-heavy
python main.py --offline                          # force synthetic data
python main.py --no-cache --lookback-days 1800    # 5-year fresh fetch
python main.py --verbose                          # see INFO-level logs
```

### Notebooks

```bash
jupyter notebook notebooks/
```

Five numbered notebooks walk through every stage:

1. `01_data_collection.ipynb` — fetch universe + NAVs + benchmarks
2. `02_feature_engineering.ipynb` — compute all 15 parameters, correlation heatmap
3. `03_model_building.ipynb` — Random Forest predicting forward return / vol / drawdown
4. `04_portfolio_optimization.ipynb` — efficient frontier, all 4 portfolio types
5. `05_simulation.ipynb` — Monte Carlo terminal-value distributions

---

## Example output

```
════════════════════════════════════════════════════════════════════════════════
  BALANCED
════════════════════════════════════════════════════════════════════════════════
  Expected Return  : +15.88%
  Volatility       : 5.14%
  Sharpe Ratio     : 1.83
  Max Drawdown     : 3.53%
  VaR (95%, 1-day) : 0.46%

  7-Year Projection (Monte Carlo, 4000 paths):
    Worst  (5th pctile)       ₹13.70 L   CAGR +13.2%
    Median (50th pctile)      ₹19.43 L   CAGR +18.1%
    Best   (95th pctile)      ₹27.08 L   CAGR +22.0%

  Allocations:
   Weight          Amount  Category        Fund
  ────────────────────────────────────────────────────────────────────────────
   14.4%         ₹71,935  Equity          SBI Bluechip Fund - Direct Growth
   14.0%         ₹69,802  Equity          Mahindra Manulife Multi Cap Fund
   11.0%         ₹54,831  Equity          SBI Small Cap Fund - Direct Growth
   10.1%         ₹50,573  Equity          Mirae Asset Large Cap Fund
    8.9%         ₹44,318  Index           UTI Nifty 50 Index Fund
    …
```

Five portfolios are always produced:

| Portfolio | Objective | Best for |
|---|---|---|
| Maximum Returns | Highest expected μ | Investors with 10+ year horizons and high risk tolerance |
| Safest | Minimum volatility | Capital preservation, short horizons |
| Balanced | Maximum Sharpe | Most investors — best risk-adjusted return |
| Income Stability | Debt-heavy, low σ | Retirees, emergency corpus |
| Aggressive Growth | Target 75th-percentile return | Long-horizon wealth-building |

---

## Modeling Methodology

**Feature engineering** — every metric is computed from raw NAV and benchmark series (not scraped from aggregators), which means the numbers update automatically as new data arrives. Metrics are also z-scored *within category* so a multi-cap is compared to multi-caps, not to liquid funds.

**ML layer** — rolling quarterly cut-dates build the training set with **no look-ahead leakage**: features are computed on the preceding 3Y window, targets on the next 1Y window. A Random Forest (default) or XGBoost model fits three separate targets: forward return, forward volatility, forward max drawdown. Evaluation uses `TimeSeriesSplit`.

**Optimization** — classical mean-variance with SLSQP. Covariance uses **Ledoit–Wolf shrinkage** to stabilize estimation on short histories. Constraints:
- Long-only (weights ≥ 0)
- Fully invested (weights sum to 1)
- Single-fund cap (default 40%)
- Category caps (Sectoral ≤ 20%, Thematic ≤ 25%, International ≤ 25%, Commodity ≤ 20%)

**Monte Carlo** — 4,000 paths per portfolio, multivariate-normal default, optional historical bootstrap. Terminal-value percentiles (P5 / P50 / P95) give you realistic worst/median/best CAGRs.

---

## Repository structure

```
mutual-fund-analyzer/
├── README.md
├── requirements.txt
├── setup.sh                       # one-click setup
├── main.py                        # entrypoint
│
├── data/
│   ├── raw/
│   │   └── fund_universe.csv      # 36 Indian MF schemes with AMFI codes
│   └── processed/                 # NAV + benchmark parquet caches
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   └── 05_simulation.ipynb
│
├── src/
│   ├── data_loader.py             # AMFI + yfinance + offline fallback
│   ├── feature_engineering.py     # 15 parameters
│   ├── model.py                   # RF / XGBoost + TS-CV
│   ├── optimizer.py               # SLSQP optimization
│   ├── simulation.py              # Monte Carlo
│   └── recommendation.py          # 5-portfolio engine
│
├── app/
│   └── cli_interface.py           # interactive + CLI
│
└── tests/
    └── test_metrics.py            # 13 unit tests for financial math
```

---

## Tests

```bash
python -m unittest tests.test_metrics -v
# 13 tests — all pass
```

Sanity checks:
- β of a series with itself equals 1
- α of a series with itself equals 0
- Optimizer weights always sum to 1 within tolerance
- VaR is a positive loss number, MDD is in [0,1]

---

## Data Sources

| Source | What we use |
|---|---|
| [AMFI India](https://portal.amfiindia.com/) | Daily NAV history (via mfapi.in mirror) |
| [Yahoo Finance](https://finance.yahoo.com/) | `^NSEI`, `^NSMIDCP`, `^NSEBANK`, `^GSPC`, GOLDBEES.NS |
| Manual curation | Fund universe CSV with TER, AUM, management fee, exit load |

When the network is unavailable or AMFI throttles, the loader falls back to a **deterministic category-aware synthetic generator** (per-fund RNG seeded by scheme code) — so the pipeline always completes for demos and CI.

---

## Disclaimer

This system is for **educational and research purposes only**. It is not investment advice. Past performance does not guarantee future returns. Consult a SEBI-registered financial advisor before making investment decisions. The authors and contributors accept no liability for losses arising from use of this software.

---

## 📜 License

MIT — see `LICENSE` for details.
