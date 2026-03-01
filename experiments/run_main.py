#!/usr/bin/env python3
"""
Main experiment runner for KTND-Finance.

Trains a NonEquilibriumVAMPNet on financial time-series data, extracts
Koopman eigenvalues, eigenfunctions, entropy decomposition, and the
irreversibility field. Produces all core results per PRD Sections 7.2,
8.2, and 8.3.

Usage
-----
    python experiments/run_main.py --config config/default.yaml --mode univariate
    python experiments/run_main.py --config config/multiasset.yaml --mode multiasset
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Allow imports from the project root when running as a standalone script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constants import (
    DATE_RANGES,
    TICKERS_UNIVARIATE,
    TICKERS_MULTIASSET,
    DOWNLOAD_START,
    DOWNLOAD_END,
    tickers_for_mode,
)
from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.utils.logging import ExperimentLogger
from src.data.preprocessing import (
    compute_log_returns,
    standardize_returns,
    time_delay_embedding,
    validate_no_leakage,
)
from src.data.loader import TimeLaggedDataset
from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.losses import total_loss
from src.model.koopman import KoopmanAnalyzer
from src.analysis.spectral import SpectralAnalyzer
from src.analysis.nonequilibrium import (
    detailed_balance_violation,
    fluctuation_theorem_ratio,
    eigenvalue_complex_plane_statistics,
)
from src.model.entropy import (
    estimate_empirical_entropy_production_with_ci,
    estimate_per_sample_entropy_production,
)
from src.analysis.regime import RegimeDetector
from src.constants import NBER_RECESSIONS

logger = logging.getLogger(__name__)


# =====================================================================
# Data acquisition
# =====================================================================

def download_data(tickers: list[str], data_dir: Path) -> pd.DataFrame:
    """Download daily close prices via yfinance if not cached on disk.

    Uses centralized date range from src.constants for consistency.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to retrieve.
    data_dir : Path
        Directory for caching the raw CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per ticker (close prices).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "prices.csv"

    if cache_file.exists():
        logger.info("Loading cached price data from %s", cache_file)
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        missing = set(tickers) - set(df.columns)
        if not missing:
            return df[tickers].dropna()
        logger.warning("Cached data missing tickers %s; re-downloading.", missing)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for data download.  Install via: pip install yfinance"
        )

    logger.info(
        "Downloading price data for %s (%s to %s) ...",
        tickers, DOWNLOAD_START, DOWNLOAD_END,
    )
    df = yf.download(
        tickers,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=True,
        progress=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif len(tickers) == 1:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})

    # Forward-fill but do NOT dropna on the full cache -- different modes
    # have different ticker subsets with different inception dates.
    df = df.ffill()
    df.to_csv(cache_file)
    logger.info("Price data saved to %s  (%d rows, %d assets)", cache_file, len(df), len(tickers))

    # Select requested tickers and drop NaN rows for THIS subset only
    out = df[tickers] if all(t in df.columns for t in tickers) else df
    out = out.dropna()
    return out


# =====================================================================
# Preprocessing
# =====================================================================

def preprocess(
    prices: pd.DataFrame,
    config: dict,
    train_end_date: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, dict]:
    """Full preprocessing pipeline: log returns -> standardize -> time-delay embed.

    Returns
    -------
    embedded : np.ndarray, shape (T', D')
        Preprocessed and embedded feature matrix.
    returns_raw : np.ndarray, shape (T_ret, D)
        Raw log returns (before embedding) for downstream analysis.
    dates : pd.DatetimeIndex
        Aligned date index for the embedded features.
    stats : dict
        Standardization statistics (for downstream inverse transforms).
    """
    log_returns = compute_log_returns(prices, drop_first=True)
    train_end_idx = len(log_returns.loc[:train_end_date])

    standardized, stats = standardize_returns(
        log_returns,
        method=config.get("data", {}).get("standardization", "zscore"),
        train_end_idx=train_end_idx,
    )

    std_arr = standardized.values if isinstance(standardized, pd.DataFrame) else standardized
    dates_all = standardized.index if isinstance(standardized, pd.DataFrame) else None

    # Time-delay embedding (Takens)
    embedding_dim = config.get("data", {}).get("embedding_dim", 5)
    delay = config.get("data", {}).get("tau", 5)

    if embedding_dim >= 2:
        embedded = time_delay_embedding(std_arr, embedding_dim=embedding_dim, delay=1)
        # Trim dates to match embedded length
        trim = std_arr.shape[0] - embedded.shape[0]
        if dates_all is not None:
            dates = dates_all[trim:]
        else:
            dates = pd.RangeIndex(embedded.shape[0])
    else:
        embedded = std_arr
        dates = dates_all if dates_all is not None else pd.RangeIndex(embedded.shape[0])

    return embedded, std_arr, dates, stats


# =====================================================================
# Dataset / DataLoader creation
# =====================================================================

def create_dataloaders(
    embedded: np.ndarray,
    dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Split into train/val/test and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    split_info : dict
        Index boundaries and counts for each split.
    """
    tau = config.get("data", {}).get("tau", 5)
    batch_size = config.get("training", {}).get("batch_size", 512)

    # Resolve date-based splits
    split_info = {}
    masks = {}
    for name, (start, end) in DATE_RANGES.items():
        mask = (dates >= start) & (dates <= end)
        masks[name] = mask
        split_info[name] = int(mask.sum())

    def _make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        data_slice = embedded[mask]
        ds = TimeLaggedDataset(data_slice, lag=tau, preprocess=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = _make_loader(masks["train"], shuffle=True)
    val_loader = _make_loader(masks["val"], shuffle=False)
    test_loader = _make_loader(masks["test"], shuffle=False)

    logger.info(
        "Data splits -- train: %d, val: %d, test: %d",
        split_info["train"], split_info["val"], split_info["test"],
    )
    return train_loader, val_loader, test_loader, split_info


# =====================================================================
# Model construction
# =====================================================================

def build_model(config: dict, input_dim: int, device: torch.device) -> NonEquilibriumVAMPNet:
    """Instantiate and move the model to the target device."""
    model_cfg = config.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [128, 128, 64])
    output_dim = model_cfg.get("n_modes", 10)
    dropout = model_cfg.get("dropout", 0.1)
    epsilon = model_cfg.get("epsilon", 1e-6)

    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout,
        epsilon=epsilon,
    )
    model.to(device)
    logger.info(
        "Model built: input_dim=%d, hidden=%s, output=%d, params=%d",
        input_dim, hidden_dims, output_dim,
        sum(p.numel() for p in model.parameters()),
    )
    return model


# =====================================================================
# Training loop
# =====================================================================

def train_one_epoch(
    model: NonEquilibriumVAMPNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_cfg: dict,
    tau: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch. Returns averaged loss dict."""
    model.train()
    accum = {}
    n_batches = 0

    for x_t, x_tau in loader:
        x_t = x_t.to(device)
        x_tau = x_tau.to(device)

        optimizer.zero_grad()
        out = model(x_t, x_tau)

        loss, loss_dict = total_loss(
            out,
            tau=tau,
            w_vamp2=loss_cfg.get("w_vamp2", 1.0),
            w_orthogonality=loss_cfg.get("beta_orthogonality", 0.01),
            w_entropy=loss_cfg.get("alpha_entropy", 0.1),
            w_spectral=loss_cfg.get("spectral_penalty_weight",
                       loss_cfg.get("gamma_regularization", 0.1)),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


@torch.no_grad()
def evaluate(
    model: NonEquilibriumVAMPNet,
    loader: DataLoader,
    loss_cfg: dict,
    tau: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on a validation / test loader. Returns averaged metrics."""
    model.eval()
    accum = {}
    n_batches = 0

    for x_t, x_tau in loader:
        x_t = x_t.to(device)
        x_tau = x_tau.to(device)

        out = model(x_t, x_tau)
        _, loss_dict = total_loss(
            out,
            tau=tau,
            w_vamp2=loss_cfg.get("w_vamp2", 1.0),
            w_orthogonality=loss_cfg.get("beta_orthogonality", 0.01),
            w_entropy=loss_cfg.get("alpha_entropy", 0.1),
            w_spectral=loss_cfg.get("spectral_penalty_weight",
                       loss_cfg.get("gamma_regularization", 0.1)),
        )

        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


def train(
    model: NonEquilibriumVAMPNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    exp_logger: ExperimentLogger,
) -> dict:
    """Full training loop with early stopping on validation VAMP-2 score.

    Returns
    -------
    history : dict
        Training history with per-epoch loss components.
    """
    train_cfg = config.get("training", {})
    loss_cfg = config.get("losses", {})
    tau = float(config.get("data", {}).get("tau", 5))

    n_epochs = train_cfg.get("n_epochs", 500)
    patience = train_cfg.get("patience", 50)
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_vamp2 = float("inf")  # We minimize negative VAMP-2
    epochs_without_improvement = 0
    best_state = None

    history: dict[str, list[float]] = {
        "train_total": [], "train_vamp2": [],
        "val_total": [], "val_vamp2": [],
        "lr": [],
    }

    logger.info("Starting training for %d epochs (patience=%d)", n_epochs, patience)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_cfg, tau, device,
        )
        val_metrics = evaluate(model, val_loader, loss_cfg, tau, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_total"].append(train_metrics.get("total", 0.0))
        history["train_vamp2"].append(train_metrics.get("vamp2", 0.0))
        history["val_total"].append(val_metrics.get("total", 0.0))
        history["val_vamp2"].append(val_metrics.get("vamp2", 0.0))
        history["lr"].append(current_lr)

        exp_logger.log_epoch(epoch, train_metrics, val_metrics)

        # Early stopping on validation VAMP-2 (we minimize the negative score)
        val_vamp2 = val_metrics.get("vamp2", float("inf"))
        if val_vamp2 < best_val_vamp2:
            best_val_vamp2 = val_vamp2
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d  train_vamp2=%.6f  val_vamp2=%.6f  "
                "lr=%.2e  elapsed=%.1fs  patience=%d/%d",
                epoch, n_epochs,
                train_metrics.get("vamp2", 0.0),
                val_vamp2, current_lr, elapsed,
                epochs_without_improvement, patience,
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model (val_vamp2=%.6f)", best_val_vamp2)

    return history


# =====================================================================
# Post-training analysis
# =====================================================================

@torch.no_grad()
def post_training_analysis(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    dates: pd.DatetimeIndex,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Extract eigenvalues, eigenfunctions, entropy decomposition, irreversibility.

    Returns
    -------
    results : dict
        All analysis results as serializable values.
    """
    model.eval()
    tau = int(config.get("data", {}).get("tau", 5))

    # Collect full-dataset forward pass
    x_t_np = embedded[:-tau]
    x_tau_np = embedded[tau:]
    x_t = torch.as_tensor(x_t_np, dtype=torch.float32).to(device)
    x_tau = torch.as_tensor(x_tau_np, dtype=torch.float32).to(device)

    out = model(x_t, x_tau)

    # --- Eigenvalues ---
    eigenvalues = out["eigenvalues"].cpu().numpy()
    singular_values = out["singular_values"].cpu().numpy()
    koopman_matrix = out["koopman_matrix"].cpu().numpy()

    # Sort by magnitude
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues_sorted = eigenvalues[order]

    eig_df = pd.DataFrame({
        "mode": np.arange(len(eigenvalues_sorted)),
        "eigenvalue_real": eigenvalues_sorted.real,
        "eigenvalue_imag": eigenvalues_sorted.imag,
        "magnitude": np.abs(eigenvalues_sorted),
        "phase": np.angle(eigenvalues_sorted),
        "decay_rate": -np.log(np.clip(np.abs(eigenvalues_sorted), 1e-15, None)) / tau,
        "frequency": np.angle(eigenvalues_sorted) / tau,
    })
    eig_path = output_dir / "eigenvalues.csv"
    eig_df.to_csv(eig_path, index=False)
    logger.info("Eigenvalues saved to %s", eig_path)

    # --- Eigenfunctions ---
    x_all = torch.as_tensor(embedded, dtype=torch.float32).to(device)
    u, v = model.compute_eigenfunctions(x_all, out)
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()

    # --- Irreversibility field (prefer eigendecomposition, fall back to SVD) ---
    irrev_field_eig = model.compute_irreversibility_field_eig(x_all, out)
    if irrev_field_eig is not None:
        irrev_field = irrev_field_eig.cpu().numpy()
        irrev_method = "eigendecomposition"
    else:
        irrev_field = model.compute_irreversibility_field(x_all, out).cpu().numpy()
        irrev_method = "svd_fallback"
    logger.info("Irreversibility field computed via %s", irrev_method)

    # --- Spectral gap (continuous-time: |Re(ln lambda_2)| / tau) ---
    eig_tensor = torch.as_tensor(eigenvalues_sorted)
    spectral_gap = float(KoopmanAnalyzer.compute_spectral_gap(eig_tensor, tau=tau))
    magnitudes = np.abs(eigenvalues_sorted)

    # --- Entropy decomposition ---
    # Spectral entropy production: sigma_k = omega_k^2 * A_k / gamma_k
    # omega_k = arg(lambda_k) / tau   (oscillation frequency)
    # gamma_k = -ln|lambda_k| / tau   (decay rate)
    # A_k = <u_k * v_k> (bilinear product of right/left eigenfunctions)
    omega = np.angle(eigenvalues_sorted) / tau
    gamma_k = -np.log(np.clip(np.abs(eigenvalues_sorted), 1e-15, 1 - 1e-7)) / tau
    gamma_k = np.clip(gamma_k, 1e-6, None)  # avoid division by zero
    A_k_full = np.mean(u_np * v_np, axis=0)  # bilinear amplitude per mode
    # Reorder A_k to match eigenvalue ordering
    if len(A_k_full) == len(order):
        A_k = np.abs(A_k_full[order])
    else:
        A_k = np.abs(A_k_full[:len(omega)])
    entropy_per_mode = omega ** 2 * A_k / gamma_k

    entropy_total = float(np.sum(np.abs(entropy_per_mode)))

    # --- Empirical entropy production with bootstrap CI ---
    returns_tensor = torch.as_tensor(
        embedded, dtype=torch.float32,
    )
    entropy_ci = estimate_empirical_entropy_production_with_ci(
        returns_tensor, tau=tau, n_bootstrap=200, block_length=50,
    )
    logger.info(
        "Empirical entropy production: %.6f [%.6f, %.6f] (95%% CI)",
        entropy_ci["point_estimate"], entropy_ci["ci_lower"], entropy_ci["ci_upper"],
    )

    entropy_df = pd.DataFrame({
        "mode": np.arange(len(entropy_per_mode)),
        "frequency_omega": omega,
        "decay_rate_gamma": gamma_k,
        "amplitude_A_k": A_k,
        "entropy_production": np.abs(entropy_per_mode),
        "entropy_fraction": np.abs(entropy_per_mode) / max(entropy_total, 1e-15),
    })
    entropy_path = output_dir / "entropy_decomposition.csv"
    entropy_df.to_csv(entropy_path, index=False)
    logger.info("Entropy decomposition saved to %s", entropy_path)

    # --- Relaxation times ---
    decay_rates = -np.log(np.clip(magnitudes, 1e-15, None)) / tau
    with np.errstate(divide="ignore"):
        relaxation_times = np.where(decay_rates > 1e-15, 1.0 / decay_rates, np.inf)

    # --- Non-equilibrium diagnostics (zero-GPU, post-hoc) ---
    neq_results = {}

    # Detailed balance violation
    db_violation = detailed_balance_violation(koopman_matrix)
    neq_results["detailed_balance_violation"] = db_violation["violation"]
    neq_results["detailed_balance_ratio"] = db_violation["ratio"]
    logger.info("Detailed balance violation: %.6f", db_violation["violation"])

    # Eigenvalue complex plane statistics
    eig_stats = eigenvalue_complex_plane_statistics(eigenvalues_sorted)
    neq_results["n_complex_modes"] = eig_stats["n_complex_modes"]
    neq_results["complex_fraction"] = eig_stats["complex_fraction"]
    neq_results["spectral_radius"] = eig_stats["spectral_radius"]

    # Fluctuation theorem: requires per-sample entropy production (N values)
    ep_samples = estimate_per_sample_entropy_production(
        returns_tensor, tau=tau, n_samples=5000,
    )
    ft_result = fluctuation_theorem_ratio(ep_samples)
    neq_results["fluctuation_theorem_ratio"] = ft_result["mean_exp_neg_sigma"]
    neq_results["ft_log_deviation"] = ft_result["log_deviation"]

    # --- KTND regime detection vs NBER ---
    # Use Gaussian HMM on top eigenfunctions to detect regimes.
    # HMM captures temporal transition structure and uses multiple
    # eigenfunctions (amplitude + transitions), unlike sign-thresholding.
    ktnd_regime_results = {}
    try:
        efunc_for_regime = u_np  # right eigenfunctions, shape (T, K)
        regime_dates = dates[:len(efunc_for_regime)] if dates is not None else None

        if regime_dates is not None and len(efunc_for_regime) > 0:
            # BIC model selection: compare 2-4 state HMMs
            bic_result = RegimeDetector.select_n_regimes_bic(
                efunc_for_regime, max_regimes=4, n_features=5,
            )
            best_n = bic_result.get("best_n", 2)
            logger.info("BIC model selection: best_n=%d, BIC=%s",
                        best_n, bic_result.get("bic_values", {}))

            # Use BIC-selected n_regimes for detection
            regime_labels = RegimeDetector.detect_from_eigenfunctions(
                efunc_for_regime, n_regimes=2, method="hmm",
            )
            nber_comparison = RegimeDetector.compare_with_nber(
                regime_labels, regime_dates,
                nber_recessions=NBER_RECESSIONS,
                train_end=DATE_RANGES["train"][1],
            )
            ktnd_regime_results = {
                "ktnd_nber_accuracy": nber_comparison["accuracy"],
                "ktnd_nber_precision": nber_comparison["precision"],
                "ktnd_nber_recall": nber_comparison["recall"],
                "ktnd_nber_f1": nber_comparison["f1"],
                "ktnd_naive_accuracy": nber_comparison["naive_accuracy"],
                "ktnd_recession_label": int(nber_comparison["recession_label"]),
                "ktnd_detection_method": "hmm",
                "ktnd_bic_best_n": best_n,
                "ktnd_bic_values": bic_result.get("bic_values", {}),
            }

            # Regime durations
            durations = RegimeDetector.compute_regime_durations(regime_labels)
            ktnd_regime_results["ktnd_mean_regime_duration"] = float(np.mean(durations))
            ktnd_regime_results["ktnd_n_regimes_detected"] = len(set(regime_labels))

            logger.info(
                "KTND regime detection: acc=%.3f, F1=%.3f (naive=%.3f), BIC_best=%d",
                nber_comparison["accuracy"],
                nber_comparison["f1"],
                nber_comparison["naive_accuracy"],
                best_n,
            )

            # Save regime labels
            regime_df = pd.DataFrame({
                "date": regime_dates,
                "regime_label": regime_labels[:len(regime_dates)],
            })
            regime_df.to_csv(output_dir / "ktnd_regime_labels.csv", index=False)
    except Exception as e:
        logger.warning("KTND regime detection failed: %s", e)
        ktnd_regime_results = {"ktnd_regime_error": str(e)}

    # --- Collect all results ---
    results = {
        "eigenvalues_real": eigenvalues_sorted.real.tolist(),
        "eigenvalues_imag": eigenvalues_sorted.imag.tolist(),
        "singular_values": singular_values.tolist(),
        "spectral_gap": spectral_gap,
        "entropy_total": entropy_total,
        "entropy_empirical": entropy_ci["point_estimate"],
        "entropy_ci_lower": entropy_ci["ci_lower"],
        "entropy_ci_upper": entropy_ci["ci_upper"],
        "entropy_std_error": entropy_ci["std_error"],
        "mean_irreversibility": float(np.mean(irrev_field)),
        "max_irreversibility": float(np.max(irrev_field)),
        "irrev_method": irrev_method,
        "relaxation_times": relaxation_times.tolist(),
        "n_modes": len(eigenvalues_sorted),
        **neq_results,
        **ktnd_regime_results,
    }

    # Save irreversibility field
    np.save(output_dir / "irreversibility_field.npy", irrev_field)
    np.save(output_dir / "eigenfunctions_right.npy", u_np)
    np.save(output_dir / "eigenfunctions_left.npy", v_np)
    np.save(output_dir / "koopman_matrix.npy", koopman_matrix)

    # Save all results as JSON (both generic and mode-specific)
    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Also save a mode-specific copy so multiasset doesn't overwrite univariate
    mode = config.get("_mode", "unknown")
    mode_results_path = output_dir / f"analysis_results_{mode}.json"
    with open(mode_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Analysis results saved to %s and %s", results_path, mode_results_path)

    return results


# =====================================================================
# CLI entry point
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Main experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode", type=str, default="univariate",
        choices=["univariate", "multiasset"],
        help="Experiment mode: univariate (SPY only) or multiasset (11 ETFs).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the random seed from the config file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override directory for saving outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----- Load configuration -----
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    config = load_config(str(config_path))

    # Merge mode-specific config if available
    mode_config_path = project_root / "config" / f"{args.mode}.yaml"
    if mode_config_path.exists():
        mode_config = load_config(str(mode_config_path))
        config = merge_configs(config, mode_config)

    # Seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)
    device = get_device()

    # ----- Output directories -----
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ----- Experiment logger -----
    exp_logger = ExperimentLogger(
        log_dir=str(output_dir / "logs"),
        run_name=f"main_{args.mode}",
    )
    exp_logger.log_config(config)

    # ----- Data acquisition -----
    tickers = config.get("data", {}).get("tickers", None)
    if tickers is None:
        tickers = tickers_for_mode(args.mode)
    data_dir = project_root / "data"
    prices = download_data(tickers, data_dir)

    # ----- Preprocessing -----
    train_end_date = DATE_RANGES["train"][1]
    embedded, returns_raw, dates, stats = preprocess(prices, config, train_end_date)
    input_dim = embedded.shape[1]
    logger.info(
        "Preprocessed data: embedded shape=%s, input_dim=%d",
        embedded.shape, input_dim,
    )

    # ----- Data leakage validation -----
    log_returns = compute_log_returns(prices, drop_first=True)
    train_end_idx = len(log_returns.loc[:train_end_date])
    leakage_report = validate_no_leakage(
        dates, DATE_RANGES, stats, train_end_idx, returns=log_returns,
    )
    if leakage_report["passed"]:
        logger.info("Data leakage validation PASSED")
    else:
        logger.warning("Data leakage validation FAILED: %s", leakage_report["checks"])
    leakage_path = results_dir / "leakage_validation.json"
    with open(leakage_path, "w") as f:
        json.dump(leakage_report, f, indent=2, default=str)

    # ----- DataLoaders -----
    train_loader, val_loader, test_loader, split_info = create_dataloaders(
        embedded, dates, config,
    )

    # ----- Build model -----
    model = build_model(config, input_dim, device)

    # ----- Train -----
    history = train(model, train_loader, val_loader, config, device, exp_logger)
    exp_logger.save_history(history)

    # Also save history to results dir for figure generation
    history_path = results_dir / f"training_history_{args.mode}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Training history saved to %s", history_path)

    # ----- Save checkpoint -----
    ckpt_path = models_dir / f"vampnet_{args.mode}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "input_dim": input_dim,
        "mode": args.mode,
        "seed": seed,
    }, ckpt_path)
    logger.info("Model checkpoint saved to %s", ckpt_path)

    # ----- Post-training analysis -----
    config["_mode"] = args.mode  # pass mode for mode-specific result files
    results = post_training_analysis(
        model, embedded, dates, config, device, results_dir,
    )

    # ----- Test set evaluation -----
    loss_cfg = config.get("losses", {})
    tau = float(config.get("data", {}).get("tau", 5))
    test_metrics = evaluate(model, test_loader, loss_cfg, tau, device)
    for k, v in test_metrics.items():
        exp_logger.log_result(f"test_{k}", v)

    # Add VAMP-2 score to results JSON (needed for multi-seed aggregation)
    results["vamp2_score"] = test_metrics.get("vamp2", None)
    results["test_total_loss"] = test_metrics.get("total", None)
    # Re-save results with VAMP-2 included
    results_path = results_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    mode_results_path = results_dir / f"analysis_results_{args.mode}.json"
    with open(mode_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ----- Print summary -----
    print("\n" + "=" * 70)
    print("KTND-Finance: Main Experiment Summary")
    print("=" * 70)
    print(f"  Mode:               {args.mode}")
    print(f"  Tickers:            {tickers}")
    print(f"  Input dim:          {input_dim}")
    print(f"  N modes:            {results['n_modes']}")
    print(f"  Spectral gap:       {results['spectral_gap']:.6f}")
    print(f"  Entropy total:      {results['entropy_total']:.6f}")
    print(f"  Entropy empirical:  {results['entropy_empirical']:.6f} "
          f"[{results['entropy_ci_lower']:.6f}, {results['entropy_ci_upper']:.6f}] 95% CI")
    print(f"  Mean irreversibility: {results['mean_irreversibility']:.6f}")
    print(f"  Irrev method:       {results['irrev_method']}")
    print(f"  DB violation:       {results.get('detailed_balance_violation', 'N/A')}")
    print(f"  Complex modes:      {results.get('n_complex_modes', 'N/A')}/{results['n_modes']}")
    print(f"  FT ratio:           {results.get('fluctuation_theorem_ratio', 'N/A')}")
    print(f"  Test VAMP-2:        {test_metrics.get('vamp2', 'N/A')}")
    print(f"  Test total loss:    {test_metrics.get('total', 'N/A')}")
    print(f"  Leading relaxation: {results['relaxation_times'][0]:.2f} days")
    # KTND regime detection metrics
    if "ktnd_nber_accuracy" in results:
        print(f"  KTND NBER accuracy: {results['ktnd_nber_accuracy']:.3f}")
        print(f"  KTND NBER F1:       {results['ktnd_nber_f1']:.3f}")
        print(f"  KTND naive acc:     {results['ktnd_naive_accuracy']:.3f}")
    elif "ktnd_regime_error" in results:
        print(f"  KTND regime:        FAILED ({results['ktnd_regime_error']})")
    print(f"  Checkpoint:         {ckpt_path}")
    print(f"  Results dir:        {results_dir}")
    print("=" * 70 + "\n")

    exp_logger.close()


if __name__ == "__main__":
    main()
