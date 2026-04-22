"""
data_loader.py
--------------
Fetches NAV history for Indian mutual funds from AMFI, benchmark index data from
yfinance, and loads fund metadata. Has a deterministic synthetic-data fallback so
the pipeline runs end-to-end even when network is unavailable.

Data sources
------------
- AMFI India          : https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx
- Yahoo Finance       : benchmark indices (^NSEI, ^NSMIDCP, ^GSPC, gold ETFs)
- Curated CSV         : fund metadata (TER, AUM, category, management fee, loads)
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FUND_UNIVERSE_CSV = RAW_DIR / "fund_universe.csv"

AMFI_NAV_URL = (
    "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"
    "?mf={amc_code}&frmdt={from_date}&todt={to_date}"
)
# Simpler per-scheme endpoint used by mftool-style clients
MFAPI_URL = "https://api.mfapi.in/mf/{scheme_code}"

# Benchmarks available on yfinance (use sensible proxies when exact index not available)
BENCHMARK_MAP: Dict[str, str] = {
    "^NSEI":           "^NSEI",           # Nifty 50
    "^NSMIDCP":        "NIFTY_MIDCAP_100.NS",  # fallback handled below
    "NIFTY_SMALLCAP":  "^CNXSC",
    "^NSEBANK":        "^NSEBANK",
    "^GSPC":           "^GSPC",
    "^CNXPSU":         "^CNXPSU",
    "^CNXINFRA":       "^CNXINFRA",
    "^CNXPHARMA":      "^CNXPHARMA",
    "CRISIL_LIQUID":   "LIQUIDBEES.NS",   # liquid ETF proxy
    "CRISIL_SHORT_TERM":"LIQUIDBEES.NS",
    "CRISIL_MEDIUM_LONG":"^NSEI",         # no direct long-bond index proxy on yf
    "CRISIL_LONG_TERM": "^NSEI",
    "GOLD":            "GOLDBEES.NS",
    "SILVER":          "SILVERBEES.NS",
}


@dataclass
class LoaderConfig:
    """Runtime configuration for the data loader."""
    lookback_days: int = 365 * 5          # 5 years of history
    use_cache: bool = True
    offline_mode: bool = False            # force synthetic generation
    request_timeout: int = 20
    sleep_between_calls: float = 0.3      # be polite to free APIs


# --------------------------------------------------------------------------- #
# Fund universe
# --------------------------------------------------------------------------- #

def load_fund_universe(path: Path = FUND_UNIVERSE_CSV) -> pd.DataFrame:
    """Load the curated fund metadata CSV (TER, AUM, category, etc.)."""
    if not path.exists():
        raise FileNotFoundError(f"Fund universe CSV not found at {path}")
    df = pd.read_csv(path, dtype={"scheme_code": str})
    df["inception_date"] = pd.to_datetime(df["inception_date"])
    # Guard against duplicate scheme codes from any edits
    df = df.drop_duplicates(subset=["scheme_code"]).reset_index(drop=True)
    logger.info("Loaded fund universe with %d schemes", len(df))
    return df


# --------------------------------------------------------------------------- #
# NAV fetching
# --------------------------------------------------------------------------- #

def _fetch_nav_from_mfapi(scheme_code: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    """
    Fetch full NAV history from mfapi.in (unofficial AMFI mirror).
    Returns a DataFrame with columns [date, nav] or None on failure.
    """
    try:
        url = MFAPI_URL.format(scheme_code=scheme_code)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", [])
        if not records:
            return None
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
        return df[["date", "nav"]]
    except Exception as exc:                                   # broad: any net / parse issue
        logger.warning("mfapi fetch failed for %s: %s", scheme_code, exc)
        return None


def _synthetic_nav_series(
    scheme_code: str,
    category: str,
    lookback_days: int,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """
    Generate a deterministic, category-aware synthetic NAV series.

    Financial intuition:
        - Equity funds: higher drift + vol
        - Debt funds:   low drift, tiny vol
        - Gold:         medium drift, medium vol, low correlation to equity (approx via noise)
        - Thematic:     high vol, fatter tails (use Student-t-like via mixture)
    """
    rng = np.random.default_rng(hash(scheme_code) % (2**32) + seed_offset)

    # Daily drift (mu) and vol (sigma) by category (approx historical Indian MF values)
    profile = {
        "Index":        (0.00050, 0.0100),
        "Equity":       (0.00060, 0.0120),
        "International":(0.00040, 0.0110),
        "Debt":         (0.00025, 0.0015),
        "Commodity":    (0.00035, 0.0090),
        "Thematic":     (0.00055, 0.0160),
        "Sectoral":     (0.00055, 0.0170),
    }
    mu, sigma = profile.get(category, (0.00045, 0.0110))

    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=lookback_days,
    )
    # GBM-style log-returns with mild fat-tail jumps
    rets = rng.normal(mu, sigma, size=len(dates))
    jumps = rng.binomial(1, 0.01, size=len(dates)) * rng.normal(-0.02, 0.01, size=len(dates))
    rets = rets + jumps
    nav = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"date": dates, "nav": nav})


def fetch_nav_history(
    universe: pd.DataFrame,
    config: LoaderConfig = LoaderConfig(),
) -> pd.DataFrame:
    """
    Fetch NAV history for all funds in the universe.

    Returns
    -------
    pd.DataFrame : long-format with columns [scheme_code, date, nav]
    """
    cache_path = PROCESSED_DIR / "nav_history.parquet"
    if config.use_cache and cache_path.exists():
        logger.info("Loading NAV history from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    frames: List[pd.DataFrame] = []
    for _, row in universe.iterrows():
        scheme_code = str(row["scheme_code"]).split("_")[0]  # handle synthetic '_2' suffixes
        nav_df: Optional[pd.DataFrame] = None

        if not config.offline_mode:
            nav_df = _fetch_nav_from_mfapi(scheme_code, timeout=config.request_timeout)
            time.sleep(config.sleep_between_calls)

        if nav_df is None or len(nav_df) < 60:
            logger.info("Using synthetic NAV for %s (%s)", scheme_code, row["scheme_name"])
            nav_df = _synthetic_nav_series(
                scheme_code=scheme_code,
                category=row["category"],
                lookback_days=config.lookback_days,
            )

        # Trim to lookback window
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=config.lookback_days)
        nav_df = nav_df[nav_df["date"] >= cutoff].copy()
        nav_df["scheme_code"] = str(row["scheme_code"])
        frames.append(nav_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["scheme_code", "date", "nav"]].sort_values(["scheme_code", "date"])
    combined.to_parquet(cache_path, index=False)
    logger.info("Saved NAV history cache: %s (%d rows)", cache_path, len(combined))
    return combined


# --------------------------------------------------------------------------- #
# Benchmark fetching
# --------------------------------------------------------------------------- #

def _synthetic_benchmark(ticker: str, lookback_days: int) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=lookback_days)
    mu, sigma = 0.00045, 0.0100
    if "GOLD" in ticker.upper():  mu, sigma = 0.00030, 0.0085
    if "LIQUID" in ticker.upper(): mu, sigma = 0.00020, 0.0010
    if "SILVER" in ticker.upper(): mu, sigma = 0.00025, 0.0150
    rets = rng.normal(mu, sigma, size=len(dates))
    px = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"date": dates, "close": px})


def fetch_benchmarks(
    universe: pd.DataFrame,
    config: LoaderConfig = LoaderConfig(),
) -> pd.DataFrame:
    """
    Fetch benchmark index prices used to compute alpha/beta/capture-ratio.
    Output columns: [benchmark, date, close]
    """
    cache_path = PROCESSED_DIR / "benchmarks.parquet"
    if config.use_cache and cache_path.exists():
        logger.info("Loading benchmarks from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    unique_benchmarks = universe["benchmark"].unique().tolist()
    frames: List[pd.DataFrame] = []

    # yfinance imported lazily so offline mode still works
    yf = None
    if not config.offline_mode:
        try:
            import yfinance as yf  # noqa: F401
        except ImportError:
            logger.warning("yfinance not installed — using synthetic benchmarks")

    start = (datetime.today() - timedelta(days=config.lookback_days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    for bench in unique_benchmarks:
        df: Optional[pd.DataFrame] = None
        yf_ticker = BENCHMARK_MAP.get(bench, bench)

        if yf is not None:
            try:
                hist = yf.download(
                    yf_ticker, start=start, end=end,
                    progress=False, auto_adjust=True, threads=False,
                )
                if hist is not None and len(hist) > 0:
                    close_col = "Close" if "Close" in hist.columns else hist.columns[0]
                    df = pd.DataFrame({
                        "date": hist.index,
                        "close": hist[close_col].values.flatten(),
                    })
            except Exception as exc:
                logger.warning("yfinance fetch failed for %s: %s", yf_ticker, exc)

        if df is None or len(df) < 60:
            df = _synthetic_benchmark(bench, config.lookback_days)

        df["benchmark"] = bench
        frames.append(df[["benchmark", "date", "close"]])

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined = combined.sort_values(["benchmark", "date"]).reset_index(drop=True)
    combined.to_parquet(cache_path, index=False)
    logger.info("Saved benchmark cache: %s (%d rows)", cache_path, len(combined))
    return combined


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #

def load_all(config: Optional[LoaderConfig] = None) -> Dict[str, pd.DataFrame]:
    """Convenience: load universe + NAVs + benchmarks in one call."""
    if config is None:
        config = LoaderConfig()
    universe = load_fund_universe()
    navs = fetch_nav_history(universe, config)
    benchmarks = fetch_benchmarks(universe, config)
    return {"universe": universe, "navs": navs, "benchmarks": benchmarks}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_all(LoaderConfig(offline_mode=True))
    print(f"Universe: {len(data['universe'])} funds")
    print(f"NAVs: {len(data['navs'])} rows, {data['navs']['scheme_code'].nunique()} schemes")
    print(f"Benchmarks: {len(data['benchmarks'])} rows, {data['benchmarks']['benchmark'].nunique()} series")
