#!/usr/bin/env python3
"""
Figure generation runner for KTND-Finance (PRD Section 12).

Loads all saved results from outputs/results/ and generates the complete
set of main and supplemental figures.  Each figure function handles missing
data gracefully by skipping if the required results are not available.

Main figures (9):
    1. Eigenvalue spectrum on the complex plane
    2. Spectral gap timeseries overlaid with VIX
    3. Eigenfunction heatmap (top K modes over time)
    4. Entropy decomposition by mode
    5. Irreversibility field over time
    6. Regime detection comparison (KTND vs baselines)
    7. Training loss curves
    8. Chapman-Kolmogorov consistency plot
    9. Bootstrap eigenvalue confidence intervals

Supplemental figures (8):
    S1. Full eigenvalue table / bar chart
    S2. Ablation summary heatmap
    S3. Rolling window entropy production
    S4. Cross-correlation: spectral gap vs VIX
    S5. Eigenfunction distributions (train vs test)
    S6. Permutation test null distribution
    S7. Baseline regime timelines
    S8. Singular value spectrum

Usage
-----
    python experiments/run_figures.py
    python experiments/run_figures.py --results-dir outputs/results --figures-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Lazy-load matplotlib to allow running on headless systems
_MPL_LOADED = False


def _ensure_matplotlib():
    """Ensure matplotlib is configured for non-interactive backend."""
    global _MPL_LOADED
    if not _MPL_LOADED:
        import matplotlib
        matplotlib.use("Agg")
        _MPL_LOADED = True


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON file, returning None if it does not exist."""
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV file, returning None if it does not exist."""
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None
    return pd.read_csv(path)


def _load_npy(path: Path) -> Optional[np.ndarray]:
    """Load npy file, returning None if it does not exist."""
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None
    return np.load(path, allow_pickle=True)


def _save_figure(fig, figures_dir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    import matplotlib.pyplot as plt
    for fmt in ["pdf", "png"]:
        out_path = figures_dir / f"{name}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out_path)
    plt.close(fig)


# =====================================================================
# Main Figures
# =====================================================================

def fig1_eigenvalue_spectrum(
    results: dict, figures_dir: Path,
) -> None:
    """Figure 1: Eigenvalue spectrum on the complex plane."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    eig_real = results.get("eigenvalues_real")
    eig_imag = results.get("eigenvalues_imag")
    if eig_real is None or eig_imag is None:
        logger.warning("Skipping Fig 1: eigenvalues not found in results.")
        return

    eig_real = np.array(eig_real)
    eig_imag = np.array(eig_imag)
    magnitudes = np.sqrt(eig_real ** 2 + eig_imag ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.5, label="|$\\lambda$|=1")

    sc = ax.scatter(eig_real, eig_imag, c=magnitudes, cmap="viridis",
                    edgecolors="k", linewidths=0.4, s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label="|$\\lambda$|")

    order = np.argsort(-magnitudes)
    for rank in range(min(5, len(order))):
        idx = order[rank]
        ax.annotate(f"$\\lambda_{{{rank}}}$",
                    (eig_real[idx], eig_imag[idx]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Re($\\lambda$)", fontsize=12)
    ax.set_ylabel("Im($\\lambda$)", fontsize=12)
    ax.set_title("Koopman Eigenvalue Spectrum", fontsize=14)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig1_eigenvalue_spectrum")


def fig2_spectral_gap_vix(
    rolling_df: Optional[pd.DataFrame],
    project_root: Path,
    figures_dir: Path,
) -> None:
    """Figure 2: Spectral gap timeseries overlaid with VIX."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if rolling_df is None or "spectral_gap" not in rolling_df.columns:
        logger.warning("Skipping Fig 2: rolling spectral gap data not found.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Parse dates
    if "center_date" in rolling_df.columns:
        dates = pd.to_datetime(rolling_df["center_date"])
    else:
        dates = np.arange(len(rolling_df))

    ax1.plot(dates, rolling_df["spectral_gap"], color="steelblue", linewidth=1.0,
             label="Spectral Gap", alpha=0.9)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Spectral Gap", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Overlay VIX on secondary axis
    vix_file = project_root / "data" / "vix.csv"
    if vix_file.exists():
        vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
        vix_col = "Close" if "Close" in vix_df.columns else vix_df.columns[0]

        ax2 = ax1.twinx()
        ax2.plot(vix_df.index, vix_df[vix_col], color="firebrick", linewidth=0.8,
                 alpha=0.6, label="VIX")
        ax2.set_ylabel("VIX", fontsize=12, color="firebrick")
        ax2.tick_params(axis="y", labelcolor="firebrick")

        # Add crisis shading
        crisis_periods = [
            ("2007-12-01", "2009-06-30", "GFC"),
            ("2020-02-01", "2020-04-30", "COVID"),
        ]
        for start, end, label in crisis_periods:
            ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                        alpha=0.15, color="gray", label=label)

    ax1.set_title("Spectral Gap vs VIX: Crisis Leading Indicator Analysis", fontsize=14)
    ax1.legend(loc="upper left")
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig2_spectral_gap_vix")


def fig3_eigenfunction_heatmap(
    eigenfunctions: Optional[np.ndarray],
    figures_dir: Path,
    n_modes: int = 5,
) -> None:
    """Figure 3: Eigenfunction heatmap (top K modes over time)."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if eigenfunctions is None:
        logger.warning("Skipping Fig 3: eigenfunctions not found.")
        return

    n_display = min(n_modes, eigenfunctions.shape[1])
    data = eigenfunctions[:, :n_display].T

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-np.percentile(np.abs(data), 95),
                   vmax=np.percentile(np.abs(data), 95))
    plt.colorbar(im, ax=ax, label="Eigenfunction value")
    ax.set_xlabel("Time index", fontsize=12)
    ax.set_ylabel("Mode", fontsize=12)
    ax.set_yticks(range(n_display))
    ax.set_yticklabels([f"$\\psi_{{{k}}}$" for k in range(n_display)])
    ax.set_title("Koopman Eigenfunction Heatmap", fontsize=14)
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig3_eigenfunction_heatmap")


def fig4_entropy_decomposition(
    entropy_df: Optional[pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Figure 4: Entropy production decomposition by mode."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if entropy_df is None:
        logger.warning("Skipping Fig 4: entropy decomposition data not found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    modes = entropy_df["mode"].values
    entropy = entropy_df["entropy_production"].values
    fraction = entropy_df["entropy_fraction"].values

    ax1.bar(modes, entropy, color="coral", edgecolor="darkred", linewidth=0.5)
    ax1.set_xlabel("Mode $k$", fontsize=12)
    ax1.set_ylabel("Entropy Production $\\sigma_k$", fontsize=12)
    ax1.set_title("Per-mode Entropy Production", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(modes, np.cumsum(fraction), color="steelblue",
            edgecolor="navy", linewidth=0.5)
    ax2.set_xlabel("Mode $k$", fontsize=12)
    ax2.set_ylabel("Cumulative Fraction", fontsize=12)
    ax2.set_title("Cumulative Entropy Fraction", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Spectral Entropy Decomposition", fontsize=14, y=1.02)
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig4_entropy_decomposition")


def fig5_irreversibility_field(
    irrev_field: Optional[np.ndarray],
    figures_dir: Path,
) -> None:
    """Figure 5: Irreversibility field over time."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if irrev_field is None:
        logger.warning("Skipping Fig 5: irreversibility field not found.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(len(irrev_field)), irrev_field, alpha=0.4, color="darkorange")
    ax.plot(irrev_field, linewidth=0.5, color="darkorange")
    ax.set_xlabel("Time index", fontsize=12)
    ax.set_ylabel("$I(x)$", fontsize=12)
    ax.set_title("Irreversibility Field: Broken Detailed Balance", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig5_irreversibility_field")


def fig6_regime_comparison(
    baseline_comparison: Optional[pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Figure 6: Regime detection comparison across methods."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if baseline_comparison is None:
        logger.warning("Skipping Fig 6: baseline comparison data not found.")
        return

    metrics_to_plot = ["nber_accuracy", "nber_f1", "nber_precision", "nber_recall"]
    available = [m for m in metrics_to_plot if m in baseline_comparison.columns]
    if not available:
        logger.warning("Skipping Fig 6: no NBER metrics in baseline comparison.")
        return

    methods = baseline_comparison["method"].values
    n_methods = len(methods)
    n_metrics = len(available)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    colors = ["steelblue", "coral", "seagreen", "orchid"]
    for i, metric in enumerate(available):
        values = baseline_comparison[metric].astype(float).values
        bars = ax.bar(x + i * width, values, width, label=metric.replace("nber_", "").title(),
                      color=colors[i % len(colors)], edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Regime Detection: Baseline Comparison", fontsize=14)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig6_regime_comparison")


def fig7_training_curves(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure 7: Training loss curves."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    # Look for metrics.csv in log directories
    logs_dir = results_dir.parent / "logs"
    metrics_file = None
    if logs_dir.exists():
        for run_dir in sorted(logs_dir.iterdir(), reverse=True):
            candidate = run_dir / "metrics.csv"
            if candidate.exists():
                metrics_file = candidate
                break

    # Also check for history.json
    history_file = None
    if logs_dir.exists():
        for run_dir in sorted(logs_dir.iterdir(), reverse=True):
            candidate = run_dir / "history.json"
            if candidate.exists():
                history_file = candidate
                break

    if metrics_file is not None:
        df = pd.read_csv(metrics_file)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if "train_total" in df.columns and "val_total" in df.columns:
            ax1.plot(df["epoch"], df["train_total"], label="Train", color="steelblue")
            ax1.plot(df["epoch"], df["val_total"], label="Validation", color="firebrick")
            ax1.set_xlabel("Epoch", fontsize=12)
            ax1.set_ylabel("Total Loss", fontsize=12)
            ax1.set_title("Total Loss", fontsize=13)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if "train_vamp2" in df.columns and "val_vamp2" in df.columns:
            ax2.plot(df["epoch"], -df["train_vamp2"], label="Train", color="steelblue")
            ax2.plot(df["epoch"], -df["val_vamp2"], label="Validation", color="firebrick")
            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("VAMP-2 Score", fontsize=12)
            ax2.set_title("VAMP-2 Score (higher = better)", fontsize=13)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Training Convergence", fontsize=14, y=1.02)
        fig.tight_layout()
        _save_figure(fig, figures_dir, "fig7_training_curves")

    elif history_file is not None:
        with open(history_file) as f:
            history = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history.get("train_total", [])) + 1)

        if "train_total" in history:
            ax1.plot(epochs, history["train_total"], label="Train", color="steelblue")
        if "val_total" in history:
            ax1.plot(epochs, history["val_total"], label="Val", color="firebrick")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        ax1.set_title("Total Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if "train_vamp2" in history:
            ax2.plot(epochs, [-v for v in history["train_vamp2"]], label="Train", color="steelblue")
        if "val_vamp2" in history:
            ax2.plot(epochs, [-v for v in history["val_vamp2"]], label="Val", color="firebrick")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("VAMP-2 Score")
        ax2.set_title("VAMP-2 Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Training Convergence", fontsize=14, y=1.02)
        fig.tight_layout()
        _save_figure(fig, figures_dir, "fig7_training_curves")
    else:
        logger.warning("Skipping Fig 7: no training metrics found.")


def fig8_chapman_kolmogorov(
    stat_tests: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure 8: Chapman-Kolmogorov consistency plot."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if stat_tests is None:
        logger.warning("Skipping Fig 8: statistical tests not found.")
        return

    ck = stat_tests.get("chapman_kolmogorov", {})
    ck_errors = ck.get("ck_errors", [])
    if not ck_errors:
        logger.warning("Skipping Fig 8: no CK error data.")
        return

    steps = [e["n"] for e in ck_errors]
    errors = [e["error"] for e in ck_errors]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(steps, errors, color="steelblue", edgecolor="navy", linewidth=0.5)
    ax.set_xlabel("$n$ (multiples of $\\tau$)", fontsize=12)
    ax.set_ylabel("Mean |$\\lambda$ error|", fontsize=12)
    ax.set_title(f"Chapman-Kolmogorov Consistency (p={ck.get('p_value', 'N/A'):.4f})",
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig8_chapman_kolmogorov")


def fig9_bootstrap_ci(
    stat_tests: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure 9: Bootstrap eigenvalue confidence intervals."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if stat_tests is None:
        logger.warning("Skipping Fig 9: statistical tests not found.")
        return

    bootstrap = stat_tests.get("bootstrap_eigenvalue_ci", {})
    modes = bootstrap.get("modes", [])
    if not modes:
        logger.warning("Skipping Fig 9: no bootstrap CI data.")
        return

    mode_idx = [m["mode"] for m in modes]
    means = [m["mean_magnitude"] for m in modes]
    ci_lower = [m["ci_lower"] for m in modes]
    ci_upper = [m["ci_upper"] for m in modes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(mode_idx, means,
                yerr=[np.array(means) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(means)],
                fmt="o", capsize=4, color="steelblue", markeredgecolor="navy",
                markersize=6, elinewidth=1.5)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="|$\\lambda$|=1")
    ax.set_xlabel("Mode $k$", fontsize=12)
    ax.set_ylabel("|$\\lambda_k$|", fontsize=12)
    ci_pct = int(bootstrap.get("ci_level", 0.95) * 100)
    ax.set_title(f"Bootstrap Eigenvalue Magnitudes ({ci_pct}% CI)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir, "fig9_bootstrap_ci")


# =====================================================================
# Supplemental Figures
# =====================================================================

def figS1_eigenvalue_bar(
    eigenvalue_df: Optional[pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Figure S1: Full eigenvalue magnitude bar chart."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if eigenvalue_df is None:
        logger.warning("Skipping Fig S1.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(eigenvalue_df["mode"], eigenvalue_df["magnitude"],
            color="steelblue", edgecolor="navy", linewidth=0.3)
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("|$\\lambda_k$|")
    ax1.set_title("Eigenvalue Magnitudes")
    ax1.grid(True, alpha=0.3, axis="y")

    if "decay_rate" in eigenvalue_df.columns:
        ax2.bar(eigenvalue_df["mode"], eigenvalue_df["decay_rate"],
                color="coral", edgecolor="darkred", linewidth=0.3)
        ax2.set_xlabel("Mode")
        ax2.set_ylabel("Decay Rate $\\gamma_k$")
        ax2.set_title("Mode Decay Rates")
        ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Full Eigenvalue Decomposition", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental", "figS1_eigenvalue_bar")


def figS2_ablation_heatmap(
    ablation_df: Optional[pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Figure S2: Ablation summary heatmap."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if ablation_df is None:
        logger.warning("Skipping Fig S2.")
        return

    metrics = [c for c in ablation_df.columns if c.endswith("_mean") and c != "elapsed_sec"]
    if not metrics or "name" not in ablation_df.columns:
        logger.warning("Skipping Fig S2: insufficient ablation data.")
        return

    heatmap_data = ablation_df.set_index("name")[metrics].astype(float)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.5), max(6, len(heatmap_data) * 0.4)))
    im = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_mean", "") for m in metrics], rotation=45, ha="right")
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)
    ax.set_title("Ablation Study: Metric Summary", fontsize=14)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS2_ablation_heatmap")


def figS3_rolling_entropy(
    rolling_df: Optional[pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Figure S3: Rolling window entropy production."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if rolling_df is None or "entropy_production" not in rolling_df.columns:
        logger.warning("Skipping Fig S3.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    if "center_date" in rolling_df.columns:
        x = pd.to_datetime(rolling_df["center_date"])
    else:
        x = np.arange(len(rolling_df))

    ax.fill_between(x, rolling_df["entropy_production"], alpha=0.4, color="coral")
    ax.plot(x, rolling_df["entropy_production"], linewidth=0.5, color="darkred")
    ax.set_xlabel("Date")
    ax.set_ylabel("Entropy Production Rate")
    ax.set_title("Rolling Window Entropy Production", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS3_rolling_entropy")


def figS4_cross_correlation(
    vix_comparison: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure S4: Cross-correlation spectral gap vs VIX."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if vix_comparison is None or vix_comparison.get("comparison") != "completed":
        logger.warning("Skipping Fig S4.")
        return

    # The detailed lag-by-lag data is in spectral_gap_vix_comparison.json
    # but may not contain all lags. Generate a simple indicator plot.
    fig, ax = plt.subplots(figsize=(8, 5))

    optimal_lag = vix_comparison.get("optimal_lag_days", 0)
    concurrent = vix_comparison.get("concurrent_correlation", 0)
    optimal_corr = vix_comparison.get("optimal_correlation", 0)

    ax.bar([0, optimal_lag], [concurrent, optimal_corr],
           color=["steelblue", "coral"], edgecolor="black", width=3)
    ax.set_xlabel("Lag (days)", fontsize=12)
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("Spectral Gap - VIX Cross-Correlation", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (lag, corr) in enumerate(zip([0, optimal_lag], [concurrent, optimal_corr])):
        ax.text(lag, corr + 0.02 * np.sign(corr), f"lag={lag}d\nr={corr:.3f}",
                ha="center", fontsize=9)

    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental", "figS4_cross_correlation")


def figS5_eigenfunction_distributions(
    stat_tests: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure S5: KS test eigenfunction distributions (train vs test)."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if stat_tests is None:
        logger.warning("Skipping Fig S5.")
        return

    ks = stat_tests.get("ks_eigenfunctions", {})
    per_mode = ks.get("per_mode", [])
    if not per_mode:
        logger.warning("Skipping Fig S5: no KS test data.")
        return

    modes = [m["mode"] for m in per_mode]
    ks_stats = [m["ks_statistic"] for m in per_mode]
    p_values = [m["p_value"] for m in per_mode]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["firebrick" if p < 0.05 else "steelblue" for p in p_values]
    ax1.bar(modes, ks_stats, color=colors, edgecolor="black", linewidth=0.3)
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("KS Statistic")
    ax1.set_title("KS Test: Train vs Test Eigenfunctions")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(modes, [-np.log10(max(p, 1e-15)) for p in p_values],
            color=colors, edgecolor="black", linewidth=0.3)
    ax2.axhline(y=-np.log10(0.05), color="red", linestyle="--",
                linewidth=1, label="$\\alpha=0.05$")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("$-\\log_{10}(p)$")
    ax2.set_title("KS Test p-values")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Eigenfunction Distribution Stability", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental", "figS5_eigenfunction_distributions")


def figS6_permutation_null(
    stat_tests: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure S6: Permutation test null distribution for irreversibility."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if stat_tests is None:
        logger.warning("Skipping Fig S6.")
        return

    perm = stat_tests.get("permutation_irreversibility", {})
    observed = perm.get("observed_mean_irreversibility")
    null_mean = perm.get("null_mean")
    null_std = perm.get("null_std")
    p_value = perm.get("p_value")

    if observed is None or null_mean is None:
        logger.warning("Skipping Fig S6: permutation test data incomplete.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Approximate null distribution as Gaussian
    x_null = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 300)
    from scipy.stats import norm
    y_null = norm.pdf(x_null, null_mean, null_std)
    ax.fill_between(x_null, y_null, alpha=0.3, color="steelblue", label="Null distribution")
    ax.plot(x_null, y_null, color="steelblue", linewidth=1)

    ax.axvline(x=observed, color="firebrick", linewidth=2,
               linestyle="--", label=f"Observed (p={p_value:.4f})")
    ax.set_xlabel("Mean Irreversibility $\\langle I(x) \\rangle$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Permutation Test for Time-Irreversibility", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS6_permutation_null")


def figS7_baseline_regimes(
    baseline_labels: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure S7: Baseline regime timelines."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if baseline_labels is None or not baseline_labels:
        logger.warning("Skipping Fig S7.")
        return

    n_methods = len(baseline_labels)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 2.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    colors_map = {0: "seagreen", 1: "goldenrod", 2: "firebrick"}

    for ax, (method, labels) in zip(axes, baseline_labels.items()):
        labels_arr = np.array(labels)
        T = len(labels_arr)
        for regime_id in sorted(set(labels_arr)):
            mask = labels_arr == regime_id
            ax.fill_between(range(T), 0, 1, where=mask,
                            color=colors_map.get(regime_id, "gray"),
                            alpha=0.6, label=f"Regime {regime_id}")
        ax.set_ylabel(method, fontsize=10, rotation=0, labelpad=80, va="center")
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("Time index")
    fig.suptitle("Regime Detection: Method Comparison", fontsize=14, y=1.01)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS7_baseline_regimes")


def figS8_singular_values(
    results: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure S8: Singular value spectrum."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if results is None:
        logger.warning("Skipping Fig S8.")
        return

    svs = results.get("singular_values")
    if svs is None:
        logger.warning("Skipping Fig S8: singular values not found.")
        return

    svs = np.array(svs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(svs)), svs, color="steelblue", edgecolor="navy", linewidth=0.3)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Mode $k$")
    ax1.set_ylabel("$\\sigma_k$")
    ax1.set_title("Singular Values")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.semilogy(range(len(svs)), svs, "o-", color="steelblue", markersize=5)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Mode $k$")
    ax2.set_ylabel("$\\sigma_k$ (log scale)")
    ax2.set_title("Singular Values (Log Scale)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Koopman Singular Value Spectrum", fontsize=14, y=1.02)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS8_singular_values")


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Generate all publication figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing saved results.",
    )
    parser.add_argument(
        "--figures-dir", type=str, default=None,
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "outputs" / "results"
    figures_dir = Path(args.figures_dir) if args.figures_dir else project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "supplemental").mkdir(parents=True, exist_ok=True)

    logger.info("Loading results from: %s", results_dir)
    logger.info("Saving figures to:    %s", figures_dir)

    # ----- Load all available data -----
    analysis_results = _load_json(results_dir / "analysis_results.json")
    eigenvalue_df = _load_csv(results_dir / "eigenvalues.csv")
    entropy_df = _load_csv(results_dir / "entropy_decomposition.csv")
    rolling_df = _load_csv(results_dir / "spectral_gap_timeseries.csv")
    stat_tests = _load_json(results_dir / "statistical_tests.json")
    baseline_comparison = _load_csv(results_dir / "baseline_comparison.csv")
    baseline_labels = _load_json(results_dir / "baseline_regime_labels.json")
    ablation_df = _load_csv(results_dir / "ablation_summary.csv")
    vix_comparison = _load_json(results_dir / "spectral_gap_vix_comparison.json")

    irrev_field = _load_npy(results_dir / "irreversibility_field.npy")
    eigenfunctions = _load_npy(results_dir / "eigenfunctions_right.npy")

    # ----- Generate main figures -----
    logger.info("Generating main figures ...")

    fig1_eigenvalue_spectrum(analysis_results or {}, figures_dir)
    fig2_spectral_gap_vix(rolling_df, project_root, figures_dir)
    fig3_eigenfunction_heatmap(eigenfunctions, figures_dir)
    fig4_entropy_decomposition(entropy_df, figures_dir)
    fig5_irreversibility_field(irrev_field, figures_dir)
    fig6_regime_comparison(baseline_comparison, figures_dir)
    fig7_training_curves(results_dir, figures_dir)
    fig8_chapman_kolmogorov(stat_tests, figures_dir)
    fig9_bootstrap_ci(stat_tests, figures_dir)

    # ----- Generate supplemental figures -----
    logger.info("Generating supplemental figures ...")

    figS1_eigenvalue_bar(eigenvalue_df, figures_dir)
    figS2_ablation_heatmap(ablation_df, figures_dir)
    figS3_rolling_entropy(rolling_df, figures_dir)
    figS4_cross_correlation(vix_comparison, figures_dir)
    figS5_eigenfunction_distributions(stat_tests, figures_dir)
    figS6_permutation_null(stat_tests, figures_dir)
    figS7_baseline_regimes(baseline_labels, figures_dir)
    figS8_singular_values(analysis_results, figures_dir)

    # ----- Summary -----
    generated = list(figures_dir.glob("*.pdf")) + list((figures_dir / "supplemental").glob("*.pdf"))
    print("\n" + "=" * 70)
    print("KTND-Finance: Figure Generation Summary")
    print("=" * 70)
    print(f"  Results dir:  {results_dir}")
    print(f"  Figures dir:  {figures_dir}")
    print(f"  Total figures generated: {len(generated)}")
    for f in sorted(generated):
        relpath = f.relative_to(figures_dir)
        print(f"    {relpath}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
