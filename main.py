"""
main.py — One-command end-to-end pipeline.

Runs:
    1. Data ingestion (AMFI + yfinance, with offline fallback)
    2. Feature engineering (all 15 parameters)
    3. ML model training + evaluation (forward return / vol / drawdown)
    4. Portfolio optimization
    5. Monte-Carlo projection
    6. Prints 5 recommended portfolios

Usage:
    python main.py                                        # interactive
    python main.py --amount 500000 --horizon 5 --risk Medium
    python main.py --amount 1000000 --horizon 10 --risk High --preference equity-heavy
    python main.py --offline                              # skip network
"""

from app.cli_interface import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
