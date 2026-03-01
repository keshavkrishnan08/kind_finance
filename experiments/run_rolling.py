#!/usr/bin/env python3
"""
Rolling window spectral analysis for KTND-Finance (PRD Section 8.4).

Performs a sliding-window spectral analysis using the trained
NonEquilibriumVAMPNet model.  For each window position the script
extracts eigenvalues, the spectral gap, and entropy production rate,
building a time series of spectral diagnostics that can be compared
against VIX and other market stress indicators.

The key hypothesis tested is whether the spectral gap narrows
*before* VIX spikes, providing a leading indicator of market crises.

Usage
-----
    python experiments/run_rolling.py --config config/default.yaml
    python experiments/run_rolling.py --checkpoint outputs/models/vampnet_univariate.pt --window 500 --stride 5
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
import torch

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

from src.constants import DATE_RANGES, TICKER_VIX, DOWNLOAD_START, DOWNLOAD_END
from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.data.preprocessing import (
    compute_log_returns,
    standardize_returns,
    time_delay_embedding,
)
from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.losses import total_loss
from src.model.koopman import KoopmanAnalyzer

logger = logging.getLogger(__name__)


# =====================================================================
# Data loading (shared with other runners)
# =====================================================================

def load_and_preprocess(
    config: dict, project_root: Path,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load price data and return preprocessed embedded features.

    Returns
    -------
    embedded : np.ndarray of shape (T', D')
    dates : pd.DatetimeIndex of length T'
    returns_raw : np.ndarray of shape (T_ret, D)
    """
    cache_file = project_root / "data" / "prices.csv"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Price data not found at {cache_file}. "
            "Run `python experiments/run_main.py` first."
        )

    prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    tickers = config.get("data", {}).get("tickers", ["SPY"])
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available] if available else prices.iloc[:, :1]

    log_returns = compute_log_returns(prices, drop_first=True)
    train_end_idx = len(log_returns.loc[:DATE_RANGES["train"][1]])
    standardized, _ = standardize_returns(
        log_returns,
        method=config.get("data", {}).get("standardization", "zscore"),
        train_end_idx=train_end_idx,
    )
    std_arr = standardized.values if isinstance(standardized, pd.DataFrame) else standardized
    dates_all = standardized.index if isinstance(standardized, pd.DataFrame) else None

    embedding_dim = config.get("data", {}).get("embedding_dim", 5)
    if embedding_dim >= 2:
        embedded = time_delay_embedding(std_arr, embedding_dim=embedding_dim, delay=1)
        trim = std_arr.shape[0] - embedded.shape[0]
        dates = dates_all[trim:] if dates_all is not None else pd.RangeIndex(embedded.shape[0])
    else:
        embedded = std_arr
        dates = dates_all if dates_all is not None else pd.RangeIndex(embedded.shape[0])

    returns_raw = log_returns.values if isinstance(log_returns, pd.DataFrame) else log_returns
    return embedded, dates, returns_raw


def load_model(
    config: dict,
    input_dim: int,
    checkpoint_path: str,
    device: torch.device,
) -> NonEquilibriumVAMPNet:
    """Instantiate model and load weights from checkpoint."""
    model_cfg = config.get("model", {})
    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 64]),
        output_dim=model_cfg.get("n_modes", 10),
        dropout=model_cfg.get("dropout", 0.1),
        epsilon=model_cfg.get("epsilon", 1e-6),
    ).to(device)

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning(
            "Checkpoint not found at %s; using random weights. "
            "Run run_main.py first.", checkpoint_path,
        )

    model.eval()
    return model


# =====================================================================
# Rolling spectral analysis
# =====================================================================

@torch.no_grad()
def rolling_spectral_analysis(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    dates: pd.DatetimeIndex,
    tau: int,
    window_size: int = 500,
    stride: int = 5,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """Perform rolling-window spectral analysis.

    For each window of ``window_size`` data points (shifted by ``stride``),
    the model is run in inference mode to extract the Koopman matrix,
    eigenvalues, spectral gap, and entropy production.

    Parameters
    ----------
    model : NonEquilibriumVAMPNet
        Trained model in eval mode.
    embedded : np.ndarray (T, D)
        Full preprocessed feature matrix.
    dates : pd.DatetimeIndex
        Corresponding date index.
    tau : int
        Lag time for time-lagged pairs.
    window_size : int
        Number of data points per analysis window.
    stride : int
        Number of points to shift between windows.
    device : torch.device
        Compute device.

    Returns
    -------
    pd.DataFrame
        One row per window, with columns for the window center date,
        eigenvalue magnitudes, spectral gap, entropy production, etc.
    """
    model.eval()
    T = len(embedded)
    n_windows = (T - window_size) // stride + 1

    if n_windows <= 0:
        raise ValueError(
            f"Cannot create any windows: T={T}, window_size={window_size}, stride={stride}"
        )

    logger.info(
        "Rolling analysis: T=%d, window=%d, stride=%d, n_windows=%d",
        T, window_size, stride, n_windows,
    )

    records: List[Dict[str, Any]] = []

    for w in range(n_windows):
        start = w * stride
        end = start + window_size

        if end > T:
            break

        window_data = embedded[start:end]
        window_center_idx = start + window_size // 2

        # Build time-lagged pairs within the window
        if window_size <= tau:
            continue
        x_t = torch.as_tensor(window_data[:-tau], dtype=torch.float32).to(device)
        x_tau = torch.as_tensor(window_data[tau:], dtype=torch.float32).to(device)

        out = model(x_t, x_tau)

        eigenvalues = out["eigenvalues"].cpu().numpy()
        singular_values = out["singular_values"].cpu().numpy()

        # Sort by magnitude
        magnitudes = np.abs(eigenvalues)
        order = np.argsort(-magnitudes)
        sorted_mag = magnitudes[order]

        # Spectral gap — continuous-time: |Re(ln λ₂)|/τ
        spectral_gap = float(
            KoopmanAnalyzer.compute_spectral_gap(out["eigenvalues"], tau=float(tau))
        )

        # Entropy production (spectral): sigma_k = omega_k^2 * A_k / gamma_k
        omega = np.angle(eigenvalues[order]) / tau
        gamma_k = -np.log(np.clip(np.abs(eigenvalues[order]), 1e-15, 1 - 1e-7)) / tau
        gamma_k = np.clip(gamma_k, 1e-6, None)
        # Quick eigenfunction amplitude estimate from chi_t variance
        chi_t = out["chi_t"].cpu().numpy()
        A_k = np.var(chi_t, axis=0)
        entropy_per_mode = omega ** 2 * A_k[:len(omega)] / gamma_k[:len(omega)]
        entropy_total = float(np.sum(np.abs(entropy_per_mode)))

        # Irreversibility
        x_window = torch.as_tensor(window_data, dtype=torch.float32).to(device)
        irrev = model.compute_irreversibility_field(x_window, out).cpu().numpy()
        mean_irrev = float(np.mean(irrev))

        # VAMP-2 score (negative sum of squared singular values; higher = better)
        vamp2_score = float(np.sum(singular_values ** 2))

        # Record
        record = {
            "window_idx": w,
            "window_start": start,
            "window_end": end,
            "center_date": str(dates[window_center_idx]) if window_center_idx < len(dates) else None,
            "spectral_gap": spectral_gap,
            "entropy_production": entropy_total,
            "mean_irreversibility": mean_irrev,
            "vamp2_score": vamp2_score,
            "leading_eigenvalue_mag": float(sorted_mag[0]) if len(sorted_mag) > 0 else None,
            "second_eigenvalue_mag": float(sorted_mag[1]) if len(sorted_mag) > 1 else None,
        }

        # Store top-5 eigenvalue magnitudes
        for k in range(min(5, len(sorted_mag))):
            record[f"eig_mag_{k}"] = float(sorted_mag[k])

        records.append(record)

        if (w + 1) % 100 == 0:
            logger.info("  processed %d / %d windows", w + 1, n_windows)

    logger.info("Rolling analysis complete: %d windows processed.", len(records))
    return pd.DataFrame(records)


# =====================================================================
# Spectral gap vs VIX comparison
# =====================================================================

def compare_spectral_gap_vix(
    rolling_df: pd.DataFrame,
    project_root: Path,
) -> Dict[str, Any]:
    """Compare spectral gap time series against VIX for crisis detection.

    Computes cross-correlation at various lags to measure whether the
    spectral gap leads or lags VIX movements. A negative lead time means
    the spectral gap anticipates VIX.

    Returns
    -------
    dict with correlation statistics and optimal lead time.
    """
    # Load VIX data
    vix_file = project_root / "data" / "vix.csv"
    if not vix_file.exists():
        try:
            from data.download import download_vix
            vix_df_loaded = download_vix(output_dir=vix_file.parent, force=False)
            vix_col = "Close" if "Close" in vix_df_loaded.columns else vix_df_loaded.columns[0]
            vix_df = vix_df_loaded[vix_col]
        except Exception as e:
            logger.warning("Could not load VIX data: %s", e)
            return {"comparison": "skipped", "reason": str(e)}
    else:
        vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
        vix_col = "Close" if "Close" in vix_df.columns else vix_df.columns[0]
        vix_df = vix_df[vix_col]

    # Create date-indexed spectral gap series
    if "center_date" not in rolling_df.columns:
        return {"comparison": "skipped", "reason": "No date column in rolling results"}

    sg_series = rolling_df.set_index(
        pd.to_datetime(rolling_df["center_date"])
    )["spectral_gap"].dropna()

    # Align on common dates
    common_dates = sg_series.index.intersection(vix_df.index)
    if len(common_dates) < 50:
        return {"comparison": "skipped", "reason": f"Only {len(common_dates)} common dates"}

    sg_aligned = sg_series.loc[common_dates].values
    vix_aligned = vix_df.loc[common_dates].values

    # Cross-correlation at various lags
    max_lag = 60  # days
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            sg_lagged = sg_aligned[:-lag]
            vix_lagged = vix_aligned[lag:]
        elif lag < 0:
            sg_lagged = sg_aligned[-lag:]
            vix_lagged = vix_aligned[:lag]
        else:
            sg_lagged = sg_aligned
            vix_lagged = vix_aligned

        if len(sg_lagged) < 30:
            continue

        # Pearson correlation
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(sg_lagged, vix_lagged)
        correlations.append({
            "lag_days": lag,
            "correlation": float(corr),
            "p_value": float(p_value),
        })

    if not correlations:
        return {"comparison": "skipped", "reason": "No valid correlations computed"}

    # Find optimal negative correlation lag (spectral gap narrows -> VIX rises)
    corr_df = pd.DataFrame(correlations)
    # We expect negative correlation (spectral gap shrinks as VIX rises)
    min_corr_idx = corr_df["correlation"].idxmin()
    optimal_lag = corr_df.loc[min_corr_idx, "lag_days"]
    min_correlation = corr_df.loc[min_corr_idx, "correlation"]

    # Concurrent correlation
    concurrent = corr_df[corr_df["lag_days"] == 0]
    concurrent_corr = float(concurrent["correlation"].iloc[0]) if len(concurrent) > 0 else None

    return {
        "comparison": "completed",
        "n_common_dates": len(common_dates),
        "concurrent_correlation": concurrent_corr,
        "optimal_lag_days": int(optimal_lag),
        "optimal_correlation": float(min_correlation),
        "lead_indicator": int(optimal_lag) < 0,
        "lead_time_days": abs(int(optimal_lag)) if int(optimal_lag) < 0 else 0,
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Rolling window spectral analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--mode", type=str, default="univariate",
        choices=["univariate", "multiasset"],
        help="Experiment mode (used to locate the default checkpoint).",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="Rolling window size (overrides config).",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Stride between windows (overrides config).",
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

    # Merge mode config
    mode_config_path = project_root / "config" / f"{args.mode}.yaml"
    if mode_config_path.exists():
        mode_config = load_config(str(mode_config_path))
        config = merge_configs(config, mode_config)

    set_seed(config.get("seed", 42))
    device = get_device()

    tau = int(config.get("data", {}).get("tau", 5))
    rolling_cfg = config.get("rolling", {})
    window_size = args.window if args.window is not None else rolling_cfg.get("window_size", 500)
    stride = args.stride if args.stride is not None else rolling_cfg.get("stride", 5)

    # Load data
    embedded, dates, returns_raw = load_and_preprocess(config, project_root)
    input_dim = embedded.shape[1]
    logger.info("Data loaded: shape=%s, date range=%s to %s",
                embedded.shape, dates[0], dates[-1])

    # Load model
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(project_root / "outputs" / "models" / f"vampnet_{args.mode}.pt")
    model = load_model(config, input_dim, checkpoint_path, device)

    # Run rolling analysis
    t0 = time.time()
    rolling_df = rolling_spectral_analysis(
        model, embedded, dates, tau,
        window_size=window_size,
        stride=stride,
        device=device,
    )
    elapsed = time.time() - t0
    logger.info("Rolling analysis completed in %.1f seconds", elapsed)

    # Save spectral gap timeseries
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "spectral_gap_timeseries.csv"
    rolling_df.to_csv(csv_path, index=False)
    logger.info("Spectral gap timeseries saved to %s", csv_path)

    # Compare with VIX
    vix_comparison = compare_spectral_gap_vix(rolling_df, project_root)

    # Save comparison results
    comparison_path = output_dir / "spectral_gap_vix_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(vix_comparison, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("KTND-Finance: Rolling Spectral Analysis Summary")
    print("=" * 70)
    print(f"  Window size:        {window_size}")
    print(f"  Stride:             {stride}")
    print(f"  N windows:          {len(rolling_df)}")
    print(f"  Elapsed time:       {elapsed:.1f}s")

    if len(rolling_df) > 0:
        print(f"\n  Spectral gap:")
        print(f"    Mean:             {rolling_df['spectral_gap'].mean():.6f}")
        print(f"    Std:              {rolling_df['spectral_gap'].std():.6f}")
        print(f"    Min:              {rolling_df['spectral_gap'].min():.6f}")
        print(f"    Max:              {rolling_df['spectral_gap'].max():.6f}")

        print(f"\n  Entropy production:")
        print(f"    Mean:             {rolling_df['entropy_production'].mean():.6f}")
        print(f"    Std:              {rolling_df['entropy_production'].std():.6f}")

        print(f"\n  Irreversibility:")
        print(f"    Mean:             {rolling_df['mean_irreversibility'].mean():.6f}")

    if vix_comparison.get("comparison") == "completed":
        print(f"\n  Spectral Gap vs VIX:")
        print(f"    Concurrent corr:  {vix_comparison['concurrent_correlation']:.4f}")
        print(f"    Optimal lag:      {vix_comparison['optimal_lag_days']} days")
        print(f"    Optimal corr:     {vix_comparison['optimal_correlation']:.4f}")
        print(f"    Lead indicator:   {vix_comparison['lead_indicator']}")
        if vix_comparison["lead_indicator"]:
            print(f"    Lead time:        {vix_comparison['lead_time_days']} days")
    else:
        print(f"\n  VIX comparison: {vix_comparison.get('reason', 'skipped')}")

    print(f"\n  Results: {csv_path}")
    print(f"  VIX comparison: {comparison_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
