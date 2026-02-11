"""Data download from Yahoo Finance.

Downloads maximum available history for all asset tickers and VIX.
Uses centralized constants from src.constants to ensure consistency
across the entire project.

Usage
-----
    python data/download.py                     # download all
    python data/download.py --mode univariate   # SPY only
    python data/download.py --mode multiasset   # cross-asset portfolio
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constants import (
    DOWNLOAD_START,
    DOWNLOAD_END,
    TICKERS_UNIVARIATE,
    TICKERS_MULTIASSET,
    TICKERS_SECTORS,
    TICKER_VIX,
    tickers_for_mode,
)

logger = logging.getLogger(__name__)


def download_prices(
    tickers: list[str],
    output_dir: str | Path = "data",
    force: bool = False,
) -> pd.DataFrame:
    """Download daily close prices for the given tickers.

    Parameters
    ----------
    tickers : list[str]
        Equity / ETF ticker symbols.
    output_dir : str or Path
        Directory to cache the CSV.
    force : bool
        If True, re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        Close prices indexed by date.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "prices.csv"

    if cache_file.exists() and not force:
        logger.info("Loading cached prices from %s", cache_file)
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        missing = set(tickers) - set(df.columns)
        if not missing:
            return df[tickers].dropna()
        logger.info("Cache missing tickers %s; re-downloading.", missing)

    logger.info("Downloading prices for %s (%s to %s) ...", tickers, DOWNLOAD_START, DOWNLOAD_END)
    df = yf.download(
        tickers,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=True,
        threads=True,
        progress=True,
    )

    # Handle multi-level columns from multi-ticker download
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif len(tickers) == 1:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})

    # Forward-fill but do NOT dropna here -- different ticker subsets have
    # different inception dates.  Each runner selects its tickers and drops
    # NaN on that subset, preserving early history for long-lived tickers.
    df = df.ffill()

    df.to_csv(cache_file)
    logger.info("Saved %d rows x %d cols to %s", len(df), len(df.columns), cache_file)

    # Validate date coverage
    date_range_days = (df.index[-1] - df.index[0]).days
    logger.info("Date coverage: %s to %s (%d calendar days, %d trading days)",
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
                date_range_days, len(df))

    # Select requested tickers and drop NaN for THIS subset only
    out = df[tickers] if all(t in df.columns for t in tickers) else df
    return out.dropna()


def download_vix(output_dir: str | Path = "data", force: bool = False) -> pd.DataFrame:
    """Download VIX index data separately.

    Returns
    -------
    pd.DataFrame with 'Close' column for VIX.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    output_dir = Path(output_dir)
    vix_file = output_dir / "vix.csv"

    if vix_file.exists() and not force:
        logger.info("Loading cached VIX from %s", vix_file)
        return pd.read_csv(vix_file, index_col=0, parse_dates=True)

    logger.info("Downloading VIX (%s to %s) ...", DOWNLOAD_START, DOWNLOAD_END)
    vix_df = yf.download(
        TICKER_VIX,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df = vix_df["Close"]
    if isinstance(vix_df, pd.DataFrame) and "Close" in vix_df.columns:
        vix_df = vix_df[["Close"]]
    elif isinstance(vix_df, pd.Series):
        vix_df = pd.DataFrame({"Close": vix_df})
    else:
        vix_df = pd.DataFrame({"Close": vix_df.iloc[:, 0]})

    vix_df = vix_df.dropna()
    vix_df.to_csv(vix_file)
    logger.info("Saved VIX: %d rows to %s", len(vix_df), vix_file)
    return vix_df


def validate_data(prices: pd.DataFrame, vix: pd.DataFrame | None = None) -> dict:
    """Run data integrity checks and print a summary."""
    report = {
        "n_rows": len(prices),
        "n_cols": len(prices.columns),
        "date_start": str(prices.index[0].date()),
        "date_end": str(prices.index[-1].date()),
        "trading_days": len(prices),
        "calendar_days": (prices.index[-1] - prices.index[0]).days,
        "nan_count": int(prices.isna().sum().sum()),
        "inf_count": int(np.isinf(prices.select_dtypes(include=[np.number])).sum().sum()),
        "tickers": list(prices.columns),
    }

    # Check for gaps (weekends/holidays are normal, but multi-day gaps may indicate issues)
    date_diffs = prices.index.to_series().diff().dt.days.dropna()
    report["max_gap_days"] = int(date_diffs.max())
    report["mean_gap_days"] = float(date_diffs.mean())

    if vix is not None:
        report["vix_rows"] = len(vix)
        report["vix_date_start"] = str(vix.index[0].date()) if len(vix) > 0 else None
        report["vix_date_end"] = str(vix.index[-1].date()) if len(vix) > 0 else None

    print("\n" + "=" * 60)
    print("Data Validation Report")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k:<25s}: {v}")
    print("=" * 60 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="KTND-Finance data downloader")
    parser.add_argument("--mode", default="all",
                        choices=["univariate", "multiasset", "sectors", "all"],
                        help="Which ticker set to download.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cache exists.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    if args.mode == "all":
        # Download everything: union of all ticker lists + VIX
        all_tickers = sorted(set(
            TICKERS_UNIVARIATE + TICKERS_MULTIASSET + TICKERS_SECTORS
        ))
    else:
        all_tickers = tickers_for_mode(args.mode)

    prices = download_prices(all_tickers, output_dir=data_dir, force=args.force)
    vix = download_vix(output_dir=data_dir, force=args.force)
    validate_data(prices, vix)


if __name__ == "__main__":
    main()
