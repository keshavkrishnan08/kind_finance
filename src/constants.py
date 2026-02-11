"""Centralized constants for KTND-Finance.

All date ranges, ticker lists, and shared configuration live here so that
every experiment file, baseline, and analysis module references a single
source of truth.  This eliminates the duplicated DATE_RANGES dicts that
previously lived in run_main.py, run_baselines.py, run_robustness.py,
and run_rolling.py.
"""

from __future__ import annotations

# ======================================================================
# Date ranges
# ======================================================================

# Full download window -- go as far back as possible for SPY (1993)
DOWNLOAD_START = "1993-01-01"
DOWNLOAD_END = "2026-12-31"

# Chronological train / val / test splits
# - Train: maximum historical depth for robust estimation
# - Val: 2 years for hyperparameter tuning / early stopping
# - Test: 4+ years covering COVID crash, 2022 bear, 2023-24 rally
DATE_RANGES = {
    "train": ("1994-01-01", "2017-12-31"),
    "val":   ("2018-01-01", "2019-12-31"),
    "test":  ("2020-01-01", "2025-12-31"),
}

# NBER recession periods for regime detection benchmarking
NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-30"),   # dot-com bust
    ("2007-12-01", "2009-06-30"),   # GFC
    ("2020-02-01", "2020-04-30"),   # COVID
]

# Known crisis onset dates for lead-time analysis
CRISIS_DATES = {
    "dot_com":         "2001-03-01",
    "gfc":             "2007-12-01",
    "covid":           "2020-02-20",
    "fed_hike_2022":   "2022-03-16",
}

# ======================================================================
# Ticker lists
# ======================================================================

# Primary benchmark
TICKERS_UNIVARIATE = ["SPY"]

# Broad cross-asset portfolio (all available from at least 2003-2007)
TICKERS_MULTIASSET = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100 (1999)
    "IWM",   # Russell 2000 (2000)
    "EFA",   # MSCI EAFE International Developed (2001)
    "EEM",   # MSCI Emerging Markets (2003)
    "TLT",   # 20+ Year Treasury (2002)
    "IEF",   # 7-10 Year Treasury (2002)
    "LQD",   # Inv. Grade Corporate (2002)
    "HYG",   # High Yield Corporate (2007)
    "GLD",   # Gold (2004)
    "VNQ",   # REITs (2004)
]

# Sector ETFs (all available from 1998-12, except XLC 2018)
TICKERS_SECTORS = [
    "XLF",   # Financials
    "XLK",   # Technology
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLU",   # Utilities
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLP",   # Consumer Staples
]

# VIX (separate because it's an index, not a tradeable ETF)
TICKER_VIX = "^VIX"

# ======================================================================
# Derived helpers
# ======================================================================

def tickers_for_mode(mode: str) -> list[str]:
    """Return the ticker list for a given experiment mode."""
    if mode == "univariate":
        return TICKERS_UNIVARIATE
    elif mode == "multiasset":
        return TICKERS_MULTIASSET
    elif mode == "sectors":
        return TICKERS_SECTORS
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'univariate', 'multiasset', or 'sectors'.")
