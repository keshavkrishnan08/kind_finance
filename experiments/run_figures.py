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

    ax1.set_title("Spectral Gap vs VIX: Concurrent Characterization", fontsize=14)
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

    # Look for training_history_*.json in results dir first (preferred)
    history_results = {}
    for mode in ["univariate", "multiasset"]:
        hp = results_dir / f"training_history_{mode}.json"
        if hp.exists():
            with open(hp) as f:
                history_results[mode] = json.load(f)

    if history_results:
        n_modes = len(history_results)
        fig, axes = plt.subplots(n_modes, 2, figsize=(14, 5 * n_modes), squeeze=False)
        for row, (mode, history) in enumerate(history_results.items()):
            epochs = range(1, len(history.get("train_total", [])) + 1)
            ax1, ax2 = axes[row]

            if "train_total" in history:
                ax1.plot(epochs, history["train_total"], label="Train", color="steelblue", alpha=0.8)
            if "val_total" in history:
                ax1.plot(epochs, history["val_total"], label="Val", color="firebrick", alpha=0.8)
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Total Loss")
            ax1.set_title(f"Total Loss ({mode})"); ax1.legend(); ax1.grid(True, alpha=0.3)

            if "train_vamp2" in history:
                ax2.plot(epochs, [-v for v in history["train_vamp2"]], label="Train", color="steelblue", alpha=0.8)
            if "val_vamp2" in history:
                ax2.plot(epochs, [-v for v in history["val_vamp2"]], label="Val", color="firebrick", alpha=0.8)
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("VAMP-2 Score")
            ax2.set_title(f"VAMP-2 Score ({mode})"); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle("Training Convergence", fontsize=14, y=1.02)
        fig.tight_layout()
        _save_figure(fig, figures_dir, "fig7_training_curves")
        return

    # Fallback: look for metrics.csv in log directories
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

    try:
        steps = [e["n"] for e in ck_errors]
        errors = [e["error"] for e in ck_errors]
    except (KeyError, TypeError):
        logger.warning("Skipping Fig 8: malformed CK error data.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(steps, errors, color="steelblue", edgecolor="navy", linewidth=0.5)
    ax.set_xlabel("$n$ (multiples of $\\tau$)", fontsize=12)
    ax.set_ylabel("CK residual $\\|[K(\\tau)]^n - K(n\\tau)\\|_F$", fontsize=12)
    ax.set_title("Chapman-Kolmogorov Consistency Test", fontsize=14)
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

    try:
        mode_idx = [m["mode"] for m in modes]
        means = [m["mean_magnitude"] for m in modes]
        ci_lower = [m["ci_lower"] for m in modes]
        ci_upper = [m["ci_upper"] for m in modes]
    except (KeyError, TypeError):
        logger.warning("Skipping Fig 9: malformed bootstrap CI data.")
        return

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

    if observed is None or null_mean is None or null_std is None or null_std == 0:
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
               linestyle="--", label=f"Observed (p={p_value:.4f})" if p_value is not None else "Observed")
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


def figS9_gallavotti_cohen(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S9: Gallavotti-Cohen symmetry function."""
    gc_data = _load_json(results_dir / "gallavotti_cohen_symmetry.json")
    if gc_data is None:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    try:
        s_vals = np.array(gc_data["s_values"])
        zeta = np.array(gc_data["zeta"])
        slope = gc_data["slope"]
        intercept = gc_data["intercept"]
        r2 = gc_data["r_squared"]
    except (KeyError, TypeError) as e:
        logger.warning("Skipping Fig S9: incomplete GC data (%s)", e)
        return

    # Filter out NaN/Inf values from histogram-based estimate
    mask = np.isfinite(zeta)
    s_vals = s_vals[mask]
    zeta = zeta[mask]

    if len(s_vals) == 0:
        logger.warning("Skipping Fig S9: no valid GC data points.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s_vals, zeta, s=20, color="steelblue", alpha=0.7, label="Data")
    s_range = np.linspace(s_vals.min(), s_vals.max(), 100)
    ax.plot(s_range, s_range, "k--", linewidth=1, label="Ideal: $\\zeta(s) = s$")
    ax.plot(s_range, slope * s_range + intercept, "r-", linewidth=1.5,
            label=f"Fit: slope={slope:.3f}, $R^2$={r2:.3f}")
    ax.set_xlabel("$s$ (entropy production)", fontsize=12)
    ax.set_ylabel("$\\zeta(s) = \\frac{1}{\\tau}\\ln\\frac{P(+s)}{P(-s)}$", fontsize=12)
    ax.set_title("Gallavotti-Cohen Symmetry Function", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS9_gallavotti_cohen")


def figS10_ck_convergence(
    stat_tests: Optional[dict],
    figures_dir: Path,
) -> None:
    """Figure S10: Chapman-Kolmogorov error vs tau (convergence analysis)."""
    if stat_tests is None:
        return

    ck_conv = stat_tests.get("ck_convergence")
    if ck_conv is None or "error" in ck_conv:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    tau_values = ck_conv.get("tau_values")
    ck_errors = ck_conv.get("ck_errors")
    if tau_values is None or ck_errors is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau_values, ck_errors, "o-", color="steelblue", markersize=8, linewidth=2)
    ax.axhline(0.1, color="seagreen", linestyle="--", linewidth=1, alpha=0.7, label="Good (< 0.1)")
    ax.axhline(0.2, color="goldenrod", linestyle="--", linewidth=1, alpha=0.7, label="Approximate (< 0.2)")
    best_tau = ck_conv.get("best_tau")
    if best_tau is not None:
        ax.axvline(best_tau, color="firebrick", linestyle=":", linewidth=1,
                   alpha=0.5, label=f"Best $\\tau$={best_tau}")
    ax.set_xlabel("Lag $\\tau$ (days)", fontsize=12)
    ax.set_ylabel("Mean CK Error", fontsize=12)
    ax.set_title("Chapman-Kolmogorov Consistency vs Lag", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, figures_dir / "supplemental", "figS10_ck_convergence")


def figS11_entropy_convergence(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S11: Spectral entropy vs K (number of modes)."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    for mode in ["univariate", "multiasset"]:
        conv_data = _load_json(results_dir / f"entropy_convergence_{mode}.json")
        if conv_data is None:
            continue

        k_vals = conv_data.get("k_values")
        s_entropy = conv_data.get("spectral_entropies")
        emp_est = conv_data.get("empirical_estimate")
        if k_vals is None or s_entropy is None or emp_est is None:
            continue
        emp_ci_lo = conv_data.get("empirical_ci_lower")
        emp_ci_hi = conv_data.get("empirical_ci_upper")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_vals, s_entropy, "o-", color="steelblue", markersize=8,
                linewidth=2, label="Spectral $\\sum_k \\omega_k^2 A_k / \\gamma_k$")
        ax.axhline(emp_est, color="firebrick", linestyle="--", linewidth=1.5,
                   label=f"Empirical (KDE): {emp_est:.2f}")
        if emp_ci_lo is not None and emp_ci_hi is not None:
            ax.axhspan(emp_ci_lo, emp_ci_hi, alpha=0.1, color="firebrick")
        ax.set_xlabel("Number of modes $K$", fontsize=12)
        ax.set_ylabel("Entropy production (bits/day)", fontsize=12)
        ax.set_title(f"Entropy Convergence ({mode})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        _save_figure(fig, figures_dir / "supplemental", f"figS11_entropy_convergence_{mode}")


def figS12_cv_results(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S12: Walk-forward CV fold-by-fold metrics."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    for mode in ["univariate", "multiasset"]:
        cv_data = _load_json(results_dir / f"cv_results_{mode}.json")
        if cv_data is None:
            continue

        folds = cv_data.get("folds", [])
        if not folds:
            continue

        metrics = ["vamp2", "spectral_gap", "db_violation"]
        available = [m for m in metrics if m in folds[0]]
        if not available:
            continue

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available):
            vals = [f[metric] for f in folds if metric in f]
            fold_ids = list(range(1, len(vals) + 1))
            ax.bar(fold_ids, vals, color="steelblue", edgecolor="navy", linewidth=0.5)
            mean_val = float(np.mean(vals))
            ax.axhline(mean_val, color="firebrick", linestyle="--", linewidth=1,
                       label=f"Mean: {mean_val:.4f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel(metric)
            ax.set_title(metric.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Walk-Forward CV ({mode}, {len(folds)} folds)", fontsize=14, y=1.02)
        fig.tight_layout()

        _save_figure(fig, figures_dir / "supplemental", f"figS12_cv_results_{mode}")


def figS13_multiseed(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S13: Multi-seed result distributions."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    # Try unified multi_seed_summary.json (notebook format)
    ms_all = _load_json(results_dir / "multi_seed_summary.json")

    for mode in ["univariate", "multiasset"]:
        # Try per-mode file first, then unified file
        ms_data = _load_json(results_dir / f"multiseed_summary_{mode}.json")

        if ms_data is None and ms_all is not None and mode in ms_all:
            ms_data = ms_all[mode]

        if ms_data is None:
            continue

        # Extract mean/std metrics (notebook format: key_mean / key_std)
        metric_names = []
        means = []
        stds = []
        for key in ms_data:
            if key.endswith("_mean") and key != "n_seeds":
                base = key[:-5]  # strip "_mean"
                std_key = f"{base}_std"
                if std_key in ms_data:
                    metric_names.append(base.replace("_", " ").title()[:20])
                    means.append(float(ms_data[key]))
                    stds.append(float(ms_data[std_key]))

        # Also handle {"metrics": {...}} format
        if not metric_names:
            metrics = ms_data.get("metrics", {})
            for m, vals in metrics.items():
                if isinstance(vals, dict) and "mean" in vals:
                    metric_names.append(m[:20])
                    means.append(vals["mean"])
                    stds.append(vals.get("std", 0))

        if not metric_names:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metric_names))
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue",
               edgecolor="navy", linewidth=0.5, ecolor="firebrick")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Value")
        ax.set_title(f"Multi-Seed Results ({mode}, mean $\\pm$ std)", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        _save_figure(fig, figures_dir / "supplemental", f"figS13_multiseed_{mode}")


def figS14_crisis_prediction(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S14: Out-of-sample crisis prediction AUROC comparison."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    pred = _load_json(results_dir / "crisis_prediction.json")
    if pred is None or pred.get("prediction") != "completed":
        logger.info("figS14: skipped (no crisis prediction results)")
        return

    auroc_spectral = pred.get("auroc_spectral")
    if auroc_spectral is None:
        logger.info("figS14: skipped (no auroc_spectral in results)")
        return
    auroc_vix = pred.get("auroc_vix_baseline")

    labels = ["KTND Spectral"]
    values = [auroc_spectral]
    colors = ["steelblue"]

    if auroc_vix is not None:
        labels.append("VIX Level")
        values.append(auroc_vix)
        colors.append("coral")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5,
                  width=0.5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_ylabel("AUROC")
    horizon = pred.get('horizon_days', '?')
    n_oos = pred.get('n_oos_windows', '?')
    ax.set_title(f"Out-of-Sample Crisis Prediction\n"
                 f"(expanding window, {horizon}-day horizon, "
                 f"n={n_oos})", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental", "figS14_crisis_prediction")


def figS15_per_crisis_timing(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S15: Per-crisis spectral gap early warning."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    data = _load_json(results_dir / "per_crisis_analysis.json")
    if data is None or data.get("analysis") != "completed":
        logger.info("Skipping figS15: no per-crisis analysis data")
        return

    crises = data["crises"]
    if not crises:
        return

    labels = [c["onset"] for c in crises]
    leads = [c["lead_days"] for c in crises]
    declines = [c["decline_pct"] for c in crises]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Lead time bar chart
    bars1 = ax1.bar(labels, leads, color="steelblue", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Lead Time (days)")
    ax1.set_title("Spectral Gap Early Warning Lead Time")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, leads):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val}d", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Decline magnitude bar chart
    bars2 = ax2.bar(labels, declines, color="coral", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Decline (%)")
    ax2.set_title("Spectral Gap Decline Before Crisis")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, declines):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental", "figS15_per_crisis_timing")


def figS16_multi_horizon(
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Figure S16: Multi-horizon crisis prediction AUROC."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    data = _load_json(results_dir / "multi_horizon_prediction.json")
    if data is None or "horizons" not in data:
        logger.info("Skipping figS16: no multi-horizon data")
        return

    horizons_data = data["horizons"]
    if not horizons_data:
        return

    # JSON keys are always strings; filter to entries with valid AUROC
    horizons = sorted(
        int(k) for k, v in horizons_data.items()
        if isinstance(v, dict) and "auroc_spectral" in v
    )
    if not horizons:
        return

    auroc_spec = [horizons_data[str(h)]["auroc_spectral"] for h in horizons]
    auroc_vix = [horizons_data[str(h)].get("auroc_vix") for h in horizons]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(horizons))

    ax.plot(x, auroc_spec, "o-", color="steelblue", linewidth=2, markersize=8,
            label="KTND Spectral", zorder=3)
    if any(v is not None for v in auroc_vix):
        vix_vals = [v if v is not None else 0.5 for v in auroc_vix]
        ax.plot(x, vix_vals, "s--", color="coral", linewidth=2, markersize=8,
                label="VIX Level", zorder=3)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("AUROC")
    ax.set_title("Crisis Prediction: Robustness Across Horizons")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental", "figS16_multi_horizon")


def fig0_pipeline_schematic(
    figures_dir: Path,
) -> None:
    """Figure 0: KTND pipeline schematic for the methods section."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.8, 2.2)
    ax.axis("off")

    # Box styles
    box_kw = dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="navy", linewidth=1.5)
    phys_kw = dict(boxstyle="round,pad=0.3", facecolor="#e8f4fd",
                   edgecolor="steelblue", linewidth=1.5)
    out_kw = dict(boxstyle="round,pad=0.3", facecolor="#fdf0e8",
                  edgecolor="coral", linewidth=1.5)

    # Pipeline stages
    stages = [
        (0.5, 1.0, "Returns\n$r_t$", box_kw),
        (2.3, 1.0, "Delay\nEmbed", box_kw),
        (4.1, 1.0, "Dual-Lobe\nVAMPNet", phys_kw),
        (5.9, 1.0, "Koopman\n$\\mathbf{K}$", phys_kw),
        (7.7, 1.0, "Eigenvalues\n$\\lambda_k$", phys_kw),
    ]

    # Output branches
    outputs = [
        (9.5, 1.8, "$\\Delta$  Spectral gap", out_kw),
        (9.5, 1.0, "$\\dot{S}_k$  Entropy", out_kw),
        (9.5, 0.2, "$I(\\mathbf{x})$  Irreversibility", out_kw),
    ]

    for x, y, txt, kw in stages:
        ax.text(x, y, txt, ha="center", va="center", fontsize=9,
                bbox=kw, fontfamily="serif")

    for x, y, txt, kw in outputs:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8,
                bbox=kw, fontfamily="serif")

    # Arrows between stages
    arrow_kw = dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                    color="navy", linewidth=1.5)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.55
        x2 = stages[i + 1][0] - 0.55
        ax.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                    arrowprops=arrow_kw)

    # Arrows from eigenvalues to outputs
    for _, oy, _, _ in outputs:
        ax.annotate("", xy=(8.8, oy), xytext=(8.3, 1.0),
                    arrowprops=dict(arrowstyle="->", color="coral",
                                   linewidth=1.2))

    # Annotations below
    ax.text(0.5, -0.3, "Daily\nlog-returns", ha="center", fontsize=7,
            color="gray", style="italic")
    ax.text(2.3, -0.3, "Takens\n$m{=}5, \\delta{=}1$", ha="center",
            fontsize=7, color="gray", style="italic")
    ax.text(4.1, -0.3, "$f_{\\theta_1}, g_{\\theta_2}$\nindependent", ha="center",
            fontsize=7, color="gray", style="italic")
    ax.text(5.9, -0.3, "SVD +\neigendecomp.", ha="center",
            fontsize=7, color="gray", style="italic")
    ax.text(7.7, -0.3, "Complex $\\Rightarrow$\nbroken DB", ha="center",
            fontsize=7, color="gray", style="italic")

    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig0_pipeline_schematic")


def figS17_perturbative_correction(
    figures_dir: Path,
) -> None:
    """Figure S17: Perturbative correction analysis for per-mode EP."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tau = 5.0
    modes_data = [
        (0,  0.400,  0.165, 0.033),
        (1,  0.400, -0.165, 0.022),
        (2,  0.425,  0.0,   0.0),
        (3, -0.210,  0.315, 0.223),
        (4, -0.210, -0.315, 0.024),
        (5, -0.311,  0.175, 0.145),
        (6, -0.311, -0.175, 0.173),
        (7,  0.031,  0.331, 0.036),
        (8,  0.031, -0.331, 0.079),
        (9,  0.183,  0.169, 0.000),
        (10, 0.183, -0.169, 0.031),
        (11,-0.158,  0.0,   0.183),
        (12, 0.095,  0.0,   0.0),
        (13,-0.013,  0.033, 0.005),
        (14,-0.013, -0.033, 0.010),
    ]

    ks, omega_taus, corrections, sk_perts, sk_corrs = [], [], [], [], []
    for k, re_v, im_v, sk_p in modes_data:
        lam = complex(re_v, im_v)
        wt = abs(np.angle(lam))
        wt_sq = wt ** 2
        sin2 = np.sin(wt) ** 2
        corr = sin2 / wt_sq if wt_sq > 1e-10 else 1.0
        ks.append(k)
        omega_taus.append(wt)
        corrections.append(corr)
        sk_perts.append(sk_p)
        sk_corrs.append(sk_p * corr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Correction factor
    ax = axes[0]
    colors = ["#2166ac" if wt < 0.5 else "#d6604d" if wt > 1.0
              else "#f4a582" for wt in omega_taus]
    ax.bar(ks, corrections, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Mode $k$", fontsize=11)
    ax.set_ylabel("Correction factor\n"
                  "$\\sin^2(\\omega_k\\tau)\\,/\\,(\\omega_k\\tau)^2$",
                  fontsize=10)
    ax.set_title("(a) Small-angle correction by mode", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(ks)
    legend_elements = [
        Patch(facecolor="#2166ac", edgecolor="k",
              label="$|\\omega_k\\tau| < 0.5$"),
        Patch(facecolor="#f4a582", edgecolor="k",
              label="$0.5 \\leq |\\omega_k\\tau| < 1$"),
        Patch(facecolor="#d6604d", edgecolor="k",
              label="$|\\omega_k\\tau| \\geq 1$"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    # Panel (b): Perturbative vs corrected
    ax2 = axes[1]
    x = np.arange(len(ks))
    w = 0.35
    ax2.bar(x - w / 2, sk_perts, w, color="#d6604d", edgecolor="k",
            linewidth=0.5, label="Perturbative $\\dot{S}_k$", alpha=0.8)
    ax2.bar(x + w / 2, sk_corrs, w, color="#2166ac", edgecolor="k",
            linewidth=0.5, label="Corrected $\\dot{S}_k$", alpha=0.8)
    ax2.set_xlabel("Mode $k$", fontsize=11)
    ax2.set_ylabel("$\\dot{S}_k$ (nats/day)", fontsize=11)
    ax2.set_title("(b) Per-mode entropy production", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ks)
    ax2.legend(fontsize=9, loc="upper right")
    tot_p = sum(sk_perts)
    tot_c = sum(sk_corrs)
    ax2.text(0.98, 0.65, f"Total (pert.): {tot_p:.2f}\n"
             f"Total (corr.): {tot_c:.2f}\n"
             f"$k$-NN ref: 0.31",
             transform=ax2.transAxes, fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    _save_figure(fig, figures_dir / "supplemental",
                 "figS17_perturbative_correction")


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
    figS9_gallavotti_cohen(results_dir, figures_dir)
    figS10_ck_convergence(stat_tests, figures_dir)
    figS11_entropy_convergence(results_dir, figures_dir)
    figS12_cv_results(results_dir, figures_dir)
    figS13_multiseed(results_dir, figures_dir)
    figS14_crisis_prediction(results_dir, figures_dir)
    figS15_per_crisis_timing(results_dir, figures_dir)
    figS16_multi_horizon(results_dir, figures_dir)

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
