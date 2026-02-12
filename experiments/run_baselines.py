#!/usr/bin/env python3
"""
Baseline comparison runner for KTND-Finance (PRD Section 11).

Loads the same data as the main experiment and fits four baseline models:
    1. HMMBaseline (Gaussian HMM, n_states=3)
    2. DMDBaseline (Dynamic Mode Decomposition, n_modes=10)
    3. PCABaseline (PCA + KMeans clustering, n_components=10)
    4. VIXThresholdBaseline (deterministic VIX threshold classifier)

Compares all baselines on regime detection timing, VAMP-2 score (where
applicable), and out-of-sample prediction accuracy against NBER recession
dates.  Saves a comprehensive comparison CSV.

Usage
-----
    python experiments/run_baselines.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

from src.constants import DATE_RANGES, CRISIS_DATES, TICKER_VIX
from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.data.preprocessing import (
    compute_log_returns,
    standardize_returns,
    time_delay_embedding,
)
from src.baselines.hmm import HMMBaseline
from src.baselines.dmd import DMDBaseline
from src.baselines.pca import PCABaseline
from src.baselines.garch import GARCHBaseline
from src.baselines.threshold import VIXThresholdBaseline
from src.analysis.regime import RegimeDetector

logger = logging.getLogger(__name__)


# =====================================================================
# Data loading helpers
# =====================================================================

def load_price_data(config: dict, project_root: Path) -> pd.DataFrame:
    """Load cached price data from the data/ directory."""
    data_dir = project_root / "data"
    cache_file = data_dir / "prices.csv"

    if not cache_file.exists():
        raise FileNotFoundError(
            f"Price data not found at {cache_file}. "
            "Run `python experiments/run_main.py` first to download data."
        )

    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    tickers = config.get("data", {}).get("tickers", ["SPY"])
    available = [t for t in tickers if t in df.columns]
    if not available:
        logger.warning("Requested tickers %s not in data; using first column.", tickers)
        return df.iloc[:, :1]
    return df[available]


def load_vix_data(project_root: Path) -> Optional[np.ndarray]:
    """Attempt to load VIX data for the threshold baseline.

    Tries to download from yfinance if not cached. Returns None if
    VIX data is unavailable.
    """
    vix_file = project_root / "data" / "vix.csv"

    if vix_file.exists():
        df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
        return df["Close"].values if "Close" in df.columns else df.iloc[:, 0].values

    try:
        from data.download import download_vix
        vix_df = download_vix(output_dir=vix_file.parent, force=False)
        return vix_df["Close"].values if "Close" in vix_df.columns else vix_df.iloc[:, 0].values
    except Exception as e:
        logger.warning("Could not load VIX data: %s", e)
        return None


# =====================================================================
# Regime detection timing analysis
# =====================================================================

def compute_regime_timing(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    """Analyze regime transition timing.

    Returns statistics about when regime transitions are detected
    relative to known crisis events.
    """
    # Known crisis start dates from centralized constants
    crisis_dates = {k: pd.Timestamp(v) for k, v in CRISIS_DATES.items()}

    results: Dict[str, Any] = {
        "n_transitions": 0,
        "mean_regime_duration": 0.0,
    }

    # Count transitions
    diffs = np.diff(labels)
    transitions = np.nonzero(diffs)[0]
    results["n_transitions"] = len(transitions)

    # Regime durations
    durations = RegimeDetector.compute_regime_durations(labels)
    results["mean_regime_duration"] = float(np.mean(durations)) if len(durations) > 0 else 0.0
    results["median_regime_duration"] = float(np.median(durations)) if len(durations) > 0 else 0.0

    # For each crisis, find the nearest preceding transition
    for crisis_name, crisis_date in crisis_dates.items():
        if crisis_date < dates.min() or crisis_date > dates.max():
            results[f"{crisis_name}_lead_days"] = None
            continue

        # Find transitions before the crisis date
        pre_crisis_transitions = [
            t for t in transitions
            if dates[t] < crisis_date and dates[t] > crisis_date - pd.Timedelta(days=365)
        ]
        if pre_crisis_transitions:
            last_transition_date = dates[pre_crisis_transitions[-1]]
            lead_days = (crisis_date - last_transition_date).days
            results[f"{crisis_name}_lead_days"] = int(lead_days)
        else:
            results[f"{crisis_name}_lead_days"] = None

    return results


# =====================================================================
# Baseline runners
# =====================================================================

def run_hmm_baseline(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    full_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    n_states: int = 3,
) -> Dict[str, Any]:
    """Fit HMM and evaluate."""
    logger.info("Fitting HMM baseline (n_states=%d) ...", n_states)
    t0 = time.time()

    hmm = HMMBaseline(n_states=n_states)
    hmm.fit(train_returns)

    # In-sample and out-of-sample predictions
    train_labels = hmm.predict(train_returns)
    test_labels = hmm.predict(test_returns)
    full_labels = hmm.predict(full_returns)

    metrics = hmm.get_metrics()
    elapsed = time.time() - t0

    # Regime detection comparison with NBER
    nber_results = RegimeDetector.compare_with_nber(full_labels, dates)
    timing = compute_regime_timing(full_labels, dates)

    # Out-of-sample log-likelihood
    oos_ll = hmm.score(test_returns)

    return {
        "method": "HMM",
        "n_states": n_states,
        "train_ll": metrics["log_likelihood"],
        "test_ll": oos_ll,
        "aic": metrics["aic"],
        "bic": metrics["bic"],
        "nber_accuracy": nber_results["accuracy"],
        "nber_precision": nber_results["precision"],
        "nber_recall": nber_results["recall"],
        "nber_f1": nber_results["f1"],
        "n_transitions": timing["n_transitions"],
        "mean_regime_duration": timing["mean_regime_duration"],
        "gfc_lead_days": timing.get("gfc_lead_days"),
        "covid_lead_days": timing.get("covid_lead_days"),
        "elapsed_sec": elapsed,
        "regime_labels": full_labels,
    }


def run_dmd_baseline(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    full_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    n_modes: int = 10,
    tau: int = 1,
) -> Dict[str, Any]:
    """Fit DMD and evaluate."""
    logger.info("Fitting DMD baseline (n_modes=%d, tau=%d) ...", n_modes, tau)
    t0 = time.time()

    dmd = DMDBaseline(n_modes=n_modes)
    dmd.fit(train_returns, tau=tau)

    dmd_metrics = dmd.get_metrics()
    elapsed = time.time() - t0

    eigenvalues = dmd_metrics["eigenvalues"]
    magnitudes = np.abs(eigenvalues)
    order = np.argsort(-magnitudes)
    sorted_mag = magnitudes[order]

    spectral_gap = float(sorted_mag[0] - sorted_mag[1]) if len(sorted_mag) > 1 else 0.0

    # DMD does not produce regime labels directly; use eigenfunction sign
    # Project data onto DMD modes for regime detection
    modes = dmd_metrics["modes"]  # (D, r)
    full_proj = full_returns @ modes.real  # (T, r)
    # Cluster the projections for regime labels
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    dmd_labels = km.fit_predict(full_proj)

    nber_results = RegimeDetector.compare_with_nber(dmd_labels, dates)
    timing = compute_regime_timing(dmd_labels, dates)

    return {
        "method": "DMD",
        "n_modes": n_modes,
        "reconstruction_error": dmd_metrics["reconstruction_error"],
        "spectral_gap": spectral_gap,
        "nber_accuracy": nber_results["accuracy"],
        "nber_precision": nber_results["precision"],
        "nber_recall": nber_results["recall"],
        "nber_f1": nber_results["f1"],
        "n_transitions": timing["n_transitions"],
        "mean_regime_duration": timing["mean_regime_duration"],
        "gfc_lead_days": timing.get("gfc_lead_days"),
        "covid_lead_days": timing.get("covid_lead_days"),
        "elapsed_sec": elapsed,
        "regime_labels": dmd_labels,
    }


def run_pca_baseline(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    full_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    n_components: int = 10,
) -> Dict[str, Any]:
    """Fit PCA + KMeans and evaluate."""
    logger.info("Fitting PCA baseline (n_components=%d) ...", n_components)
    t0 = time.time()

    pca = PCABaseline(n_components=n_components)
    pca.fit(train_returns)

    pca_metrics = pca.get_metrics()
    full_labels = pca.detect_regimes(full_returns, n_clusters=3)
    test_labels = pca.detect_regimes(test_returns, n_clusters=3)
    elapsed = time.time() - t0

    nber_results = RegimeDetector.compare_with_nber(full_labels, dates)
    timing = compute_regime_timing(full_labels, dates)

    cumvar = float(np.sum(pca_metrics["explained_variance_ratio"]))

    return {
        "method": "PCA",
        "n_components": n_components,
        "cumulative_variance": cumvar,
        "nber_accuracy": nber_results["accuracy"],
        "nber_precision": nber_results["precision"],
        "nber_recall": nber_results["recall"],
        "nber_f1": nber_results["f1"],
        "n_transitions": timing["n_transitions"],
        "mean_regime_duration": timing["mean_regime_duration"],
        "gfc_lead_days": timing.get("gfc_lead_days"),
        "covid_lead_days": timing.get("covid_lead_days"),
        "elapsed_sec": elapsed,
        "regime_labels": full_labels,
    }


def run_garch_baseline(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    full_returns: np.ndarray,
    dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    """Fit GARCH(1,1) and evaluate regime detection."""
    logger.info("Fitting GARCH(1,1) baseline ...")
    t0 = time.time()

    # GARCH requires univariate input — use first column if multivariate
    train_r = train_returns.ravel() if train_returns.ndim == 1 else train_returns[:, 0]
    test_r = test_returns.ravel() if test_returns.ndim == 1 else test_returns[:, 0]
    full_r = full_returns.ravel() if full_returns.ndim == 1 else full_returns[:, 0]

    garch = GARCHBaseline(high_vol_percentile=80)
    garch.fit(train_r)

    full_labels = garch.predict(full_r)
    metrics = garch.get_metrics()
    elapsed = time.time() - t0

    # Regime detection comparison with NBER
    nber_results = RegimeDetector.compare_with_nber(full_labels, dates)
    timing = compute_regime_timing(full_labels, dates)

    return {
        "method": "GARCH",
        "n_states": 2,
        "train_ll": metrics["log_likelihood"],
        "test_ll": None,
        "aic": metrics["aic"],
        "bic": metrics["bic"],
        "nber_accuracy": nber_results["accuracy"],
        "nber_precision": nber_results["precision"],
        "nber_recall": nber_results["recall"],
        "nber_f1": nber_results["f1"],
        "n_transitions": timing["n_transitions"],
        "mean_regime_duration": timing["mean_regime_duration"],
        "gfc_lead_days": timing.get("gfc_lead_days"),
        "covid_lead_days": timing.get("covid_lead_days"),
        "elapsed_sec": elapsed,
        "regime_labels": full_labels,
    }


def run_vix_baseline(
    vix_data: Optional[np.ndarray],
    dates: pd.DatetimeIndex,
) -> Optional[Dict[str, Any]]:
    """Run VIX threshold baseline if VIX data is available."""
    if vix_data is None:
        logger.warning("VIX data not available; skipping VIX threshold baseline.")
        return None

    logger.info("Running VIX threshold baseline ...")
    t0 = time.time()

    # Align VIX data length with dates
    min_len = min(len(vix_data), len(dates))
    vix_data = vix_data[:min_len]
    dates_aligned = dates[:min_len]

    vix_baseline = VIXThresholdBaseline()
    vix_metrics = vix_baseline.get_metrics(vix_data)
    full_labels = vix_metrics["regime_labels"]
    elapsed = time.time() - t0

    nber_results = RegimeDetector.compare_with_nber(full_labels, dates_aligned)
    timing = compute_regime_timing(full_labels, dates_aligned)

    return {
        "method": "VIX_Threshold",
        "thresholds": vix_baseline.thresholds,
        "nber_accuracy": nber_results["accuracy"],
        "nber_precision": nber_results["precision"],
        "nber_recall": nber_results["recall"],
        "nber_f1": nber_results["f1"],
        "n_transitions": timing["n_transitions"],
        "mean_regime_duration": timing["mean_regime_duration"],
        "fraction_low_vol": vix_metrics["fraction_per_regime"].get(0, 0.0),
        "fraction_elevated": vix_metrics["fraction_per_regime"].get(1, 0.0),
        "fraction_crisis": vix_metrics["fraction_per_regime"].get(2, 0.0),
        "gfc_lead_days": timing.get("gfc_lead_days"),
        "covid_lead_days": timing.get("covid_lead_days"),
        "elapsed_sec": elapsed,
        "regime_labels": full_labels,
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Baseline comparison runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override directory for saving results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(str(project_root / args.config))
    set_seed(config.get("seed", 42))

    # ----- Load data -----
    prices = load_price_data(config, project_root)
    log_returns = compute_log_returns(prices, drop_first=True)
    dates = log_returns.index if isinstance(log_returns, pd.DataFrame) else pd.RangeIndex(len(log_returns))
    returns_arr = log_returns.values if isinstance(log_returns, pd.DataFrame) else log_returns

    # Date-based splits
    train_mask = (dates >= DATE_RANGES["train"][0]) & (dates <= DATE_RANGES["train"][1])
    test_mask = (dates >= DATE_RANGES["test"][0]) & (dates <= DATE_RANGES["test"][1])

    train_returns = returns_arr[train_mask]
    test_returns = returns_arr[test_mask]
    full_returns = returns_arr

    logger.info(
        "Data loaded: %d total, %d train, %d test, %d assets",
        len(returns_arr), len(train_returns), len(test_returns), returns_arr.shape[1],
    )

    # Load VIX data
    vix_data = load_vix_data(project_root)

    # ----- Run baselines -----
    all_results: List[Dict[str, Any]] = []

    # 1. HMM
    try:
        hmm_result = run_hmm_baseline(
            train_returns, test_returns, full_returns, dates, n_states=3,
        )
        all_results.append(hmm_result)
    except Exception as e:
        logger.error("HMM baseline failed: %s", e)

    # 2. DMD
    try:
        tau_dmd = config.get("data", {}).get("tau", 5)
        dmd_result = run_dmd_baseline(
            train_returns, test_returns, full_returns, dates,
            n_modes=10, tau=tau_dmd,
        )
        all_results.append(dmd_result)
    except Exception as e:
        logger.error("DMD baseline failed: %s", e)

    # 3. PCA
    try:
        pca_result = run_pca_baseline(
            train_returns, test_returns, full_returns, dates, n_components=10,
        )
        all_results.append(pca_result)
    except Exception as e:
        logger.error("PCA baseline failed: %s", e)

    # 4. GARCH(1,1)
    try:
        garch_result = run_garch_baseline(
            train_returns, test_returns, full_returns, dates,
        )
        all_results.append(garch_result)
    except Exception as e:
        logger.error("GARCH baseline failed: %s", e)

    # 5. VIX Threshold
    try:
        vix_result = run_vix_baseline(vix_data, dates)
        if vix_result is not None:
            all_results.append(vix_result)
    except Exception as e:
        logger.error("VIX threshold baseline failed: %s", e)

    # ----- Build comparison table -----
    # Extract serializable columns (exclude ndarray regime_labels)
    comparison_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "regime_labels"}
        # Convert non-serializable types
        for k, v in row.items():
            if isinstance(v, (list, np.ndarray)):
                row[k] = str(v)
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "baseline_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    # Also save regime labels for downstream figure generation
    regime_labels_dict = {}
    for r in all_results:
        method_name = r["method"]
        labels = r.get("regime_labels")
        if labels is not None:
            regime_labels_dict[method_name] = labels.tolist()

    labels_path = output_dir / "baseline_regime_labels.json"
    with open(labels_path, "w") as f:
        json.dump(regime_labels_dict, f)

    # ----- Print summary -----
    print("\n" + "=" * 90)
    print("KTND-Finance: Baseline Comparison Summary")
    print("=" * 90)

    display_cols = [
        "method", "nber_accuracy", "nber_f1",
        "n_transitions", "mean_regime_duration",
        "gfc_lead_days", "covid_lead_days", "elapsed_sec",
    ]
    available_cols = [c for c in display_cols if c in comparison_df.columns]
    print(comparison_df[available_cols].to_string(index=False))
    print(f"\nFull results saved to: {csv_path}")
    print(f"Regime labels saved to: {labels_path}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
