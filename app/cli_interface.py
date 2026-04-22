"""
cli_interface.py
----------------
Interactive command-line interface for the mutual-fund analyzer.

    python -m app.cli_interface --amount 1000000 --horizon 5 --risk Medium

or fully interactive:

    python -m app.cli_interface
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root:  python app/cli_interface.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_all, LoaderConfig                # noqa: E402
from feature_engineering import compute_fund_features          # noqa: E402
from recommendation import RecommendationEngine, UserInput     # noqa: E402


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #

def _fmt_inr(x: float) -> str:
    """Format INR with lakh/crore abbreviations."""
    if x >= 1e7:  return f"₹{x/1e7:.2f} Cr"
    if x >= 1e5:  return f"₹{x/1e5:.2f} L"
    return f"₹{x:,.0f}"


def _hr(char: str = "─", n: int = 80) -> str:
    return char * n


def print_portfolio(name: str, result, user: UserInput) -> None:
    print()
    print(_hr("═"))
    print(f"  {name.upper()}")
    print(_hr("═"))

    print(f"  Expected Return  : {result.expected_return:+.2%}")
    print(f"  Volatility       : {result.volatility:.2%}")
    print(f"  Sharpe Ratio     : {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown     : {result.max_drawdown:.2%}")
    print(f"  VaR (95%, 1-day) : {result.var_95:.2%}")

    print()
    print(f"  {user.horizon_years:.0f}-Year Projection (Monte Carlo, 4000 paths):")
    s = result.projection["summary"]
    c = result.projection["cagr"]
    print(f"    Worst  (5th pctile)  {_fmt_inr(s['p5']):>14}   CAGR {c['worst_case_p5']:+.2%}")
    print(f"    Median (50th pctile) {_fmt_inr(s['p50']):>14}   CAGR {c['median_p50']:+.2%}")
    print(f"    Best   (95th pctile) {_fmt_inr(s['p95']):>14}   CAGR {c['best_case_p95']:+.2%}")

    print()
    print("  Allocations:")
    print(f"  {'Weight':>7}  {'Amount':>14}  {'Category':<14}  Fund")
    print(f"  {_hr('─', 76)}")
    for _, row in result.allocations.iterrows():
        print(f"  {row['weight']:>6.1%}  {_fmt_inr(row['amount_inr']):>14}  "
              f"{row['category'][:14]:<14}  {row['scheme_name'][:40]}")


# --------------------------------------------------------------------------- #
# Interactive prompts
# --------------------------------------------------------------------------- #

def _prompt_amount() -> float:
    while True:
        raw = input("  Investment amount in INR (e.g. 500000): ").strip().replace(",", "")
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
        print("    Please enter a positive number.")


def _prompt_horizon() -> float:
    while True:
        raw = input("  Investment horizon in years (1-30): ").strip()
        try:
            v = float(raw)
            if 0.5 <= v <= 30:
                return v
        except ValueError:
            pass
        print("    Please enter a number between 0.5 and 30.")


def _prompt_risk() -> str:
    while True:
        raw = input("  Risk appetite [Low/Medium/High] (default Medium): ").strip().capitalize()
        if not raw:
            return "Medium"
        if raw in {"Low", "Medium", "High"}:
            return raw
        print("    Please enter Low, Medium, or High.")


def _prompt_preference() -> str:
    options = ["balanced", "equity-heavy", "debt-heavy", "gold", "psu"]
    print(f"  Preference options: {', '.join(options)} (or leave blank)")
    raw = input("  Preference: ").strip().lower()
    return raw if raw in options else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indian Mutual Fund Recommendation Engine")
    p.add_argument("--amount", type=float, help="Investment amount in INR")
    p.add_argument("--horizon", type=float, help="Horizon in years")
    p.add_argument("--risk", choices=["Low", "Medium", "High"], help="Risk appetite")
    p.add_argument("--preference", type=str, default=None,
                   help="Optional: balanced|equity-heavy|debt-heavy|gold|psu")
    p.add_argument("--offline", action="store_true",
                   help="Force offline (use synthetic data, no network)")
    p.add_argument("--lookback-days", type=int, default=365 * 5)
    p.add_argument("--no-cache", action="store_true", help="Ignore cached data files")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print()
    print(_hr("═"))
    print("  INDIAN MUTUAL FUND ANALYZER  ·  Portfolio Recommendation Engine")
    print(_hr("═"))

    # Interactive prompts where args are missing
    if args.amount is None:
        print("\n  Enter your investment profile:\n")
        amount = _prompt_amount()
        horizon = _prompt_horizon()
        risk = _prompt_risk()
        preference = _prompt_preference()
    else:
        amount = args.amount
        horizon = args.horizon if args.horizon is not None else 5
        risk = args.risk or "Medium"
        preference = args.preference

    print()
    print(f"  Profile   : {_fmt_inr(amount)}  ·  {horizon:g} years  ·  Risk: {risk}"
          f"{'  ·  Pref: '+preference if preference else ''}")
    print()
    print("  [1/3] Loading data ...")
    data = load_all(LoaderConfig(
        offline_mode=args.offline,
        lookback_days=args.lookback_days,
        use_cache=not args.no_cache,
    ))
    print(f"        {len(data['universe'])} funds loaded")

    print("  [2/3] Computing the 15 parameters ...")
    features = compute_fund_features(data["universe"], data["navs"], data["benchmarks"])
    print(f"        Feature matrix: {features.shape}")

    print("  [3/3] Optimizing portfolios ...")
    engine = RecommendationEngine(data["universe"], features, data["navs"])
    user = UserInput(investment_amount=amount, horizon_years=horizon,
                     risk_appetite=risk, preference=preference)
    portfolios = engine.recommend(user)
    print(f"        Generated {len(portfolios)} portfolios")

    for name, result in portfolios.items():
        print_portfolio(name, result, user)

    print()
    print(_hr("═"))
    print("  Disclaimer: For educational / research purposes only. Not investment advice.")
    print("  Past performance does not guarantee future returns. Consult a SEBI-registered advisor.")
    print(_hr("═"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
