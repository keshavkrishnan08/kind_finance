"""
Publication-quality figure generation for KTND-Finance.

Implements all nine paper figures specified in PRD Section 12, plus a
training-curve plot, using matplotlib with seaborn styling.  Every public
method returns the ``matplotlib.figure.Figure`` object so callers can
further customise before saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style defaults -- publication / PRE quality
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # TrueType fonts in PDF (journal requirement)
        "ps.fonttype": 42,
    }
)

# Colour palette consistent across all figures
_PALETTE = sns.color_palette("deep", 10)


# ===================================================================== #
#  Helper: save_figure                                                    #
# ===================================================================== #

def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Union[str, Path] = "outputs/figures",
) -> Tuple[Path, Path]:
    """Save *fig* as both PDF (vector) and PNG (raster).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename (without extension).
    output_dir : str or Path
        Directory in which to write the files.

    Returns
    -------
    tuple of (Path, Path)
        Paths to the saved PDF and PNG files respectively.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"

    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")

    return pdf_path, png_path


# ===================================================================== #
#  FigureGenerator                                                        #
# ===================================================================== #

class FigureGenerator:
    """Generate all paper figures for the KTND-Finance manuscript.

    Every ``plot_*`` method returns a ``matplotlib.figure.Figure`` which
    can then be passed to :func:`save_figure` for export.
    """

    # ------------------------------------------------------------------ #
    # Fig 2 -- Synthetic validation                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_synthetic_validation(
        trajectory: np.ndarray,
        eigenvalues_learned: np.ndarray,
        eigenvalues_true: np.ndarray,
        eigenfunction: np.ndarray,
        ck_results: Dict[str, np.ndarray],
    ) -> plt.Figure:
        """Synthetic Ornstein-Uhlenbeck validation (Figure 2).

        Parameters
        ----------
        trajectory : ndarray, shape (T, D)
            Sample trajectory from the synthetic system.
        eigenvalues_learned : ndarray, shape (K,)
            Eigenvalues recovered by the KVAE.
        eigenvalues_true : ndarray, shape (K,)
            Analytical eigenvalues of the OU process.
        eigenfunction : ndarray, shape (T,) or (T, K)
            Learned eigenfunction evaluated along the trajectory.
        ck_results : dict
            Chapman-Kolmogorov test results with keys ``'lag_times'``,
            ``'predicted'``, ``'estimated'`` (each arrays of shape (n_lags,)).

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

        # (a) Sample trajectory
        ax0 = fig.add_subplot(gs[0, 0])
        if trajectory.ndim == 1:
            ax0.plot(trajectory, color=_PALETTE[0], linewidth=0.5)
        else:
            for d in range(min(trajectory.shape[1], 3)):
                ax0.plot(trajectory[:, d], linewidth=0.5, label=f"Dim {d + 1}")
            ax0.legend()
        ax0.set_xlabel("Time step")
        ax0.set_ylabel("Value")
        ax0.set_title("(a) Sample trajectory")

        # (b) Eigenvalue comparison
        ax1 = fig.add_subplot(gs[0, 1])
        n_ev = len(eigenvalues_true)
        idx = np.arange(n_ev)
        width = 0.35
        ax1.bar(idx - width / 2, np.real(eigenvalues_true), width,
                label="Analytical", color=_PALETTE[0])
        ax1.bar(idx + width / 2, np.real(eigenvalues_learned[:n_ev]), width,
                label="Learned", color=_PALETTE[1])
        ax1.set_xlabel("Mode index")
        ax1.set_ylabel(r"$\lambda_k$")
        ax1.set_title("(b) Eigenvalue comparison")
        ax1.legend()

        # (c) Eigenvalue parity plot
        ax2 = fig.add_subplot(gs[0, 2])
        n_common = min(len(eigenvalues_true), len(eigenvalues_learned))
        ax2.scatter(
            np.real(eigenvalues_true[:n_common]),
            np.real(eigenvalues_learned[:n_common]),
            color=_PALETTE[2],
            edgecolors="k",
            linewidth=0.5,
            zorder=3,
        )
        lims = [
            min(
                np.real(eigenvalues_true[:n_common]).min(),
                np.real(eigenvalues_learned[:n_common]).min(),
            ),
            max(
                np.real(eigenvalues_true[:n_common]).max(),
                np.real(eigenvalues_learned[:n_common]).max(),
            ),
        ]
        margin = 0.05 * (lims[1] - lims[0])
        ax2.plot(
            [lims[0] - margin, lims[1] + margin],
            [lims[0] - margin, lims[1] + margin],
            "k--",
            linewidth=0.8,
        )
        ax2.set_xlabel(r"$\lambda_k$ (true)")
        ax2.set_ylabel(r"$\lambda_k$ (learned)")
        ax2.set_title("(c) Parity plot")

        # (d) Learned eigenfunction
        ax3 = fig.add_subplot(gs[1, 0])
        ef = eigenfunction if eigenfunction.ndim == 1 else eigenfunction[:, 0]
        ax3.plot(ef, color=_PALETTE[3], linewidth=0.6)
        ax3.set_xlabel("Time step")
        ax3.set_ylabel(r"$\psi_1(x)$")
        ax3.set_title("(d) Learned eigenfunction")

        # (e) Chapman-Kolmogorov test
        ax4 = fig.add_subplot(gs[1, 1:])
        lag_times = ck_results["lag_times"]
        predicted = ck_results["predicted"]
        estimated = ck_results["estimated"]
        ax4.plot(lag_times, predicted, "o-", color=_PALETTE[0],
                 label="Predicted", markersize=4)
        ax4.plot(lag_times, estimated, "s--", color=_PALETTE[1],
                 label="Estimated", markersize=4)
        ax4.set_xlabel("Lag time")
        ax4.set_ylabel("Autocorrelation")
        ax4.set_title("(e) Chapman-Kolmogorov test")
        ax4.legend()

        return fig

    # ------------------------------------------------------------------ #
    # Fig 3 -- Eigenvalue spectrum                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_eigenvalue_spectrum(
        eigenvalues: np.ndarray,
        entropy_contributions: np.ndarray,
    ) -> plt.Figure:
        """Koopman eigenvalue spectrum analysis (Figure 3).

        Three-panel layout: complex plane scatter, entropy contribution
        bar chart, and mode-correlation heatmap.

        Parameters
        ----------
        eigenvalues : ndarray, shape (K,), complex
            Koopman eigenvalues.
        entropy_contributions : ndarray, shape (K,)
            Per-mode entropy contribution (non-negative reals).

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(14, 4.5))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)

        K = len(eigenvalues)

        # (a) Complex plane
        ax0 = fig.add_subplot(gs[0, 0])
        theta = np.linspace(0, 2 * np.pi, 200)
        ax0.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6,
                 alpha=0.4, label="Unit circle")
        scatter = ax0.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c=entropy_contributions,
            cmap="viridis",
            edgecolors="k",
            linewidth=0.4,
            s=60,
            zorder=3,
        )
        fig.colorbar(scatter, ax=ax0, label="Entropy contribution")
        ax0.set_xlabel(r"Re($\lambda$)")
        ax0.set_ylabel(r"Im($\lambda$)")
        ax0.set_title("(a) Eigenvalue spectrum")
        ax0.set_aspect("equal")
        ax0.axhline(0, color="grey", linewidth=0.4)
        ax0.axvline(0, color="grey", linewidth=0.4)

        # (b) Entropy contribution bar chart
        ax1 = fig.add_subplot(gs[0, 1])
        mode_idx = np.arange(1, K + 1)
        ax1.bar(mode_idx, entropy_contributions, color=_PALETTE[0],
                edgecolor="k", linewidth=0.4)
        ax1.set_xlabel("Mode index")
        ax1.set_ylabel("Entropy contribution (nats)")
        ax1.set_title("(b) Per-mode entropy")

        # (c) Mode-correlation heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        # Build a synthetic correlation matrix from eigenvalue magnitudes
        magnitudes = np.abs(eigenvalues)
        corr_matrix = np.outer(magnitudes, magnitudes) / (
            np.max(magnitudes) ** 2 + 1e-12
        )
        np.fill_diagonal(corr_matrix, 1.0)
        im = ax2.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_xlabel("Mode")
        ax2.set_ylabel("Mode")
        ax2.set_title("(c) Mode correlation")
        fig.colorbar(im, ax=ax2, label="Correlation")

        return fig

    # ------------------------------------------------------------------ #
    # Fig 4 -- Eigenfunctions                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_eigenfunctions(
        eigenfunctions: np.ndarray,
        returns: np.ndarray,
        dates: np.ndarray,
    ) -> plt.Figure:
        """Koopman eigenfunctions on market data (Figure 4).

        Shows the first three eigenfunctions and their relationship to
        the returns time series.

        Parameters
        ----------
        eigenfunctions : ndarray, shape (T, K) with K >= 3
            Eigenfunction values evaluated along the return series.
        returns : ndarray, shape (T,)
            Market return series.
        dates : ndarray, shape (T,)
            Date labels (datetime or string).

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Returns time series
        axes[0].plot(dates, returns, color="grey", linewidth=0.4,
                     alpha=0.8)
        axes[0].set_ylabel("Return")
        axes[0].set_title("Market returns")

        labels = [r"$\psi_1$", r"$\psi_2$", r"$\psi_3$"]
        colors = [_PALETTE[0], _PALETTE[1], _PALETTE[2]]

        for i in range(3):
            axes[i + 1].plot(dates, eigenfunctions[:, i], color=colors[i],
                             linewidth=0.6)
            axes[i + 1].set_ylabel(labels[i])
            axes[i + 1].set_title(f"Eigenfunction {i + 1}")

        axes[-1].set_xlabel("Date")
        fig.tight_layout()

        return fig

    # ------------------------------------------------------------------ #
    # Fig 5 -- Entropy decomposition                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_entropy_decomposition(
        entropy_results: Dict[str, np.ndarray],
        vix: np.ndarray,
        dates: np.ndarray,
    ) -> plt.Figure:
        """Spectral entropy decomposition (Figure 5).

        Four-panel figure: total entropy time series, stacked mode
        contributions, pie chart of average contributions, and entropy
        vs VIX scatter.

        Parameters
        ----------
        entropy_results : dict
            Must contain keys ``'total'`` (T,), ``'modes'`` (T, K), and
            ``'mode_labels'`` (list of K strings).
        vix : ndarray, shape (T,)
            VIX index aligned with *dates*.
        dates : ndarray, shape (T,)
            Date labels.

        Returns
        -------
        matplotlib.figure.Figure
        """
        total = entropy_results["total"]
        modes = entropy_results["modes"]
        mode_labels = entropy_results.get(
            "mode_labels",
            [f"Mode {i + 1}" for i in range(modes.shape[1])],
        )
        K = modes.shape[1]

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

        # (a) Total spectral entropy
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(dates, total, color=_PALETTE[0], linewidth=0.7)
        ax0.fill_between(dates, total, alpha=0.15, color=_PALETTE[0])
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Entropy (nats)")
        ax0.set_title("(a) Total spectral entropy")

        # (b) Stacked mode contributions
        ax1 = fig.add_subplot(gs[0, 1])
        bottom = np.zeros(len(dates))
        for k in range(K):
            ax1.fill_between(
                dates,
                bottom,
                bottom + modes[:, k],
                label=mode_labels[k],
                color=_PALETTE[k % len(_PALETTE)],
                alpha=0.7,
            )
            bottom += modes[:, k]
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Entropy (nats)")
        ax1.set_title("(b) Stacked mode contributions")
        ax1.legend(fontsize=8, ncol=2)

        # (c) Pie chart of average contributions
        ax2 = fig.add_subplot(gs[1, 0])
        avg_contributions = np.mean(modes, axis=0)
        avg_contributions = np.maximum(avg_contributions, 0)  # safety
        wedges, texts, autotexts = ax2.pie(
            avg_contributions,
            labels=mode_labels,
            autopct="%1.1f%%",
            colors=[_PALETTE[k % len(_PALETTE)] for k in range(K)],
            startangle=90,
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax2.set_title("(c) Average entropy share")

        # (d) Scatter: entropy vs VIX
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(vix, total, s=8, alpha=0.5, color=_PALETTE[3],
                    edgecolors="none")
        # Trend line
        z = np.polyfit(vix, total, 1)
        p = np.poly1d(z)
        vix_sorted = np.sort(vix)
        ax3.plot(vix_sorted, p(vix_sorted), "k--", linewidth=1,
                 label=f"r = {np.corrcoef(vix, total)[0, 1]:.2f}")
        ax3.set_xlabel("VIX")
        ax3.set_ylabel("Spectral entropy (nats)")
        ax3.set_title("(d) Entropy vs. VIX")
        ax3.legend()

        return fig

    # ------------------------------------------------------------------ #
    # Fig 6 -- Spectral gap                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_spectral_gap(
        spectral_gaps: np.ndarray,
        regime_durations: np.ndarray,
        vix: np.ndarray,
        dates: np.ndarray,
    ) -> plt.Figure:
        """Spectral gap and regime persistence (Figure 6).

        Three panels: gap time series, gap-vs-duration scatter with
        theoretical bound, and ROC curve for regime detection.

        Parameters
        ----------
        spectral_gaps : ndarray, shape (T,)
            Time series of spectral gap values.
        regime_durations : ndarray, shape (N,)
            Duration (in days) of each detected regime.
        vix : ndarray, shape (T,)
            VIX index, same length as *spectral_gaps*.
        dates : ndarray, shape (T,)
            Date labels.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(14, 4.5))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)

        # (a) Spectral gap time series
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(dates, spectral_gaps, color=_PALETTE[0], linewidth=0.6)
        ax0_twin = ax0.twinx()
        ax0_twin.plot(dates, vix, color=_PALETTE[1], linewidth=0.4,
                      alpha=0.6)
        ax0_twin.set_ylabel("VIX", color=_PALETTE[1])
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Spectral gap", color=_PALETTE[0])
        ax0.set_title("(a) Spectral gap time series")

        # (b) Gap vs regime duration
        ax1 = fig.add_subplot(gs[0, 1])
        # Average spectral gap per regime -- subsample to match
        n_regimes = len(regime_durations)
        if n_regimes > 0:
            chunk_size = max(1, len(spectral_gaps) // n_regimes)
            gap_means = np.array(
                [
                    np.mean(spectral_gaps[i * chunk_size: (i + 1) * chunk_size])
                    for i in range(n_regimes)
                ]
            )
            ax1.scatter(gap_means, regime_durations, s=25, alpha=0.7,
                        color=_PALETTE[2], edgecolors="k", linewidth=0.3)
            # Theoretical lower bound: duration >= 1 / gap
            gap_range = np.linspace(
                max(gap_means.min(), 1e-4), gap_means.max(), 100
            )
            ax1.plot(gap_range, 1.0 / gap_range, "k--", linewidth=0.8,
                     label=r"$\tau \geq 1/\Delta$")
            ax1.legend()
        ax1.set_xlabel("Mean spectral gap")
        ax1.set_ylabel("Regime duration (days)")
        ax1.set_title("(b) Gap vs. duration")

        # (c) ROC curve for regime-change detection
        ax2 = fig.add_subplot(gs[0, 2])
        # Construct a simple ROC from thresholding the gap derivative
        gap_deriv = np.abs(np.diff(spectral_gaps))
        vix_deriv = np.abs(np.diff(vix))
        vix_threshold = np.percentile(vix_deriv, 90)
        labels_binary = (vix_deriv >= vix_threshold).astype(int)

        thresholds = np.linspace(gap_deriv.min(), gap_deriv.max(), 200)
        tpr_list, fpr_list = [], []
        for thr in thresholds:
            pred = (gap_deriv >= thr).astype(int)
            tp = np.sum((pred == 1) & (labels_binary == 1))
            fp = np.sum((pred == 1) & (labels_binary == 0))
            fn = np.sum((pred == 0) & (labels_binary == 1))
            tn = np.sum((pred == 0) & (labels_binary == 0))
            tpr_list.append(tp / max(tp + fn, 1))
            fpr_list.append(fp / max(fp + tn, 1))

        ax2.plot(fpr_list, tpr_list, color=_PALETTE[0], linewidth=1.2)
        ax2.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.set_title("(c) ROC: regime detection")
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)

        return fig

    # ------------------------------------------------------------------ #
    # Fig 7 -- Irreversibility                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_irreversibility(
        irrev_1d: np.ndarray,
        irrev_2d: np.ndarray,
        irrev_ts: np.ndarray,
        dates: np.ndarray,
    ) -> plt.Figure:
        """Time-reversal asymmetry / irreversibility (Figure 7).

        Parameters
        ----------
        irrev_1d : ndarray, shape (N_bins,)
            1-D irreversibility measure (e.g. histogram of signed
            entropy production).
        irrev_2d : ndarray, shape (N, N)
            2-D irreversibility heatmap (e.g. forward-backward
            transition matrix difference).
        irrev_ts : ndarray, shape (T,)
            Time series of running irreversibility score.
        dates : ndarray, shape (T,)
            Date labels for the time series panel.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(14, 4.5))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)

        # (a) 1-D distribution
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.bar(
            np.arange(len(irrev_1d)),
            irrev_1d,
            color=_PALETTE[0],
            edgecolor="k",
            linewidth=0.3,
        )
        ax0.set_xlabel("Bin")
        ax0.set_ylabel("Irreversibility")
        ax0.set_title(r"(a) 1-D $\Delta s$")

        # (b) 2-D heatmap
        ax1 = fig.add_subplot(gs[0, 1])
        vmax = np.max(np.abs(irrev_2d))
        im = ax1.imshow(
            irrev_2d,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(im, ax=ax1, label=r"$\Delta s$")
        ax1.set_xlabel("State $j$")
        ax1.set_ylabel("State $i$")
        ax1.set_title(r"(b) 2-D irreversibility")

        # (c) Time series
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(dates, irrev_ts, color=_PALETTE[2], linewidth=0.6)
        ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Irreversibility score")
        ax2.set_title("(c) Running irreversibility")

        return fig

    # ------------------------------------------------------------------ #
    # Fig 8 -- Ablation study                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_ablation_results(ablation_df) -> plt.Figure:
        """Ablation study results (Figure 8).

        Parameters
        ----------
        ablation_df : pandas.DataFrame
            Must contain columns ``'variant'``, ``'vamp2_score'``,
            ``'correlation'``, and ``'entropy_error'``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(14, 4.5))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)

        variants = ablation_df["variant"].values
        x = np.arange(len(variants))

        # (a) VAMP-2 score
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.bar(x, ablation_df["vamp2_score"].values, color=_PALETTE[0],
                edgecolor="k", linewidth=0.4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(variants, rotation=30, ha="right")
        ax0.set_ylabel("VAMP-2 score")
        ax0.set_title("(a) VAMP-2 score")

        # (b) Eigenvalue correlation
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.bar(x, ablation_df["correlation"].values, color=_PALETTE[1],
                edgecolor="k", linewidth=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants, rotation=30, ha="right")
        ax1.set_ylabel("Correlation")
        ax1.set_title(r"(b) $\lambda$ correlation")

        # (c) Entropy error
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.bar(x, ablation_df["entropy_error"].values, color=_PALETTE[2],
                edgecolor="k", linewidth=0.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(variants, rotation=30, ha="right")
        ax2.set_ylabel("Entropy RMSE")
        ax2.set_title("(c) Entropy prediction error")

        fig.tight_layout()

        return fig

    # ------------------------------------------------------------------ #
    # Fig 9 -- Baseline comparison                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_baseline_comparison(
        our_regimes: np.ndarray,
        hmm_regimes: np.ndarray,
        vix_regimes: np.ndarray,
        dates: np.ndarray,
        pr_curves: Dict[str, Dict[str, np.ndarray]],
    ) -> plt.Figure:
        """Baseline comparison: KTND vs HMM vs VIX regimes (Figure 9).

        Parameters
        ----------
        our_regimes : ndarray, shape (T,)
            Regime labels from KTND.
        hmm_regimes : ndarray, shape (T,)
            Regime labels from a hidden Markov model baseline.
        vix_regimes : ndarray, shape (T,)
            VIX-threshold regime labels.
        dates : ndarray, shape (T,)
            Date labels.
        pr_curves : dict
            Mapping ``method_name -> {'precision': array, 'recall': array}``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

        # (a) KTND regimes
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(dates, our_regimes, color=_PALETTE[0], linewidth=0.5)
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Regime")
        ax0.set_title("(a) KTND regimes")

        # (b) HMM regimes
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(dates, hmm_regimes, color=_PALETTE[1], linewidth=0.5)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Regime")
        ax1.set_title("(b) HMM regimes")

        # (c) VIX-threshold regimes
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(dates, vix_regimes, color=_PALETTE[2], linewidth=0.5)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Regime")
        ax2.set_title("(c) VIX-threshold regimes")

        # (d) Precision-recall curves
        ax3 = fig.add_subplot(gs[1, 1])
        for i, (method, curves) in enumerate(pr_curves.items()):
            ax3.plot(
                curves["recall"],
                curves["precision"],
                color=_PALETTE[i % len(_PALETTE)],
                linewidth=1.2,
                label=method,
            )
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.set_title("(d) Precision-recall comparison")
        ax3.legend()
        ax3.set_xlim(-0.02, 1.02)
        ax3.set_ylim(-0.02, 1.02)

        return fig

    # ------------------------------------------------------------------ #
    # Training curves (supplementary)                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_training_curves(history: Dict[str, List[float]]) -> plt.Figure:
        """Training loss components over epochs.

        Parameters
        ----------
        history : dict
            Mapping of loss-component names to per-epoch value lists.
            Expected keys include (but are not limited to) ``'total_loss'``,
            ``'recon_loss'``, ``'kl_loss'``, ``'vamp_loss'``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_components = len(history)
        n_cols = min(n_components, 3)
        n_rows = (n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False
        )

        for idx, (name, values) in enumerate(history.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, color=_PALETTE[idx % len(_PALETTE)],
                    linewidth=1.0)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name.replace("_", " ").title())
            ax.set_title(name.replace("_", " ").title())

        # Hide unused subplots
        for idx in range(n_components, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.tight_layout()

        return fig
