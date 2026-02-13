#!/usr/bin/env python3
"""
Walk-forward cross-validation for KTND-Finance.

Implements expanding-window time-series cross-validation to validate that
the model generalizes across temporal regimes rather than overfitting to
a single train/val/test split.

For each fold k:
    Train on [start, split_k], validate on [split_k, split_k + window],
    extract out-of-sample VAMP-2, spectral gap, and eigenvalue statistics.

Reports mean +/- std across folds, providing error bars on key metrics.

Usage
-----
    python experiments/run_cv.py --config config/default.yaml --mode univariate
    python experiments/run_cv.py --config config/default.yaml --mode multiasset --n-folds 4
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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

from src.constants import tickers_for_mode
from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.data.preprocessing import (
    compute_log_returns,
    standardize_returns,
    time_delay_embedding,
)
from src.data.loader import TimeLaggedDataset
from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.losses import total_loss
from src.model.koopman import KoopmanAnalyzer

logger = logging.getLogger(__name__)


# =====================================================================
# Walk-forward fold generation
# =====================================================================

def generate_walk_forward_folds(
    dates: pd.DatetimeIndex,
    n_folds: int = 5,
    min_train_years: int = 10,
    val_years: int = 2,
) -> List[Dict[str, Any]]:
    """Generate expanding-window walk-forward CV folds.

    Each fold uses an expanding training window and a fixed-length
    validation window.  The folds advance chronologically so that
    every fold's test data comes strictly after its training data.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full date index of the dataset.
    n_folds : int
        Number of CV folds.
    min_train_years : int
        Minimum training window in years (first fold).
    val_years : int
        Validation/test window in years per fold.

    Returns
    -------
    folds : list of dict
        Each dict has 'train_mask', 'val_mask', 'train_end', 'val_end'.
    """
    start_date = dates.min()
    end_date = dates.max()

    # Total span available for val windows after min training
    min_train_end = start_date + pd.DateOffset(years=min_train_years)
    val_span = (end_date - min_train_end).days
    fold_stride_days = max(val_span // n_folds, 365)

    folds = []
    for k in range(n_folds):
        train_end = min_train_end + pd.DateOffset(days=k * fold_stride_days)
        val_end = train_end + pd.DateOffset(years=val_years)

        if train_end >= end_date:
            break
        if val_end > end_date:
            val_end = end_date

        train_mask = dates <= train_end
        val_mask = (dates > train_end) & (dates <= val_end)

        if val_mask.sum() < 50:
            continue

        folds.append({
            "fold": k,
            "train_end": str(train_end.date()),
            "val_end": str(val_end.date()),
            "train_mask": train_mask,
            "val_mask": val_mask,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
        })

    return folds


# =====================================================================
# Train one fold
# =====================================================================

def train_fold(
    embedded: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Train model on one CV fold and extract out-of-sample metrics."""
    set_seed(seed)

    tau = int(config.get("data", {}).get("tau", 5))
    batch_size = config.get("training", {}).get("batch_size", 512)
    loss_cfg = config.get("losses", {})
    train_cfg = config.get("training", {})

    # Create data loaders
    train_data = embedded[train_mask]
    val_data = embedded[val_mask]

    train_ds = TimeLaggedDataset(train_data, lag=tau, preprocess=False)
    val_ds = TimeLaggedDataset(val_data, lag=tau, preprocess=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Build model
    model_cfg = config.get("model", {})
    input_dim = embedded.shape[1]
    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 64]),
        output_dim=model_cfg.get("n_modes", 10),
        dropout=model_cfg.get("dropout", 0.1),
        epsilon=model_cfg.get("epsilon", 1e-6),
    ).to(device)

    # Training loop (abbreviated — fewer epochs for CV efficiency)
    n_epochs = min(train_cfg.get("n_epochs", 800), 400)  # cap at 400 for CV
    patience = min(train_cfg.get("patience", 80), 50)
    lr = train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_vamp2 = float("inf")
    stale = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        for x_t, x_tau in train_loader:
            x_t, x_tau = x_t.to(device), x_tau.to(device)
            out = model(x_t, x_tau)
            loss, _ = total_loss(
                out, tau=float(tau),
                w_vamp2=loss_cfg.get("w_vamp2", 1.0),
                w_orthogonality=loss_cfg.get("beta_orthogonality", 0.01),
                w_entropy=loss_cfg.get("alpha_entropy", 0.1),
                w_spectral=loss_cfg.get("spectral_penalty_weight",
                           loss_cfg.get("gamma_regularization", 0.1)),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_accum = {}
        n_batches = 0
        with torch.no_grad():
            for x_t, x_tau in val_loader:
                x_t, x_tau = x_t.to(device), x_tau.to(device)
                out = model(x_t, x_tau)
                _, loss_dict = total_loss(
                    out, tau=float(tau),
                    w_vamp2=loss_cfg.get("w_vamp2", 1.0),
                    w_orthogonality=loss_cfg.get("beta_orthogonality", 0.01),
                    w_entropy=loss_cfg.get("alpha_entropy", 0.1),
                    w_spectral=loss_cfg.get("spectral_penalty_weight",
                               loss_cfg.get("gamma_regularization", 0.1)),
                )
                for k, v in loss_dict.items():
                    val_accum[k] = val_accum.get(k, 0.0) + v.item()
                n_batches += 1

        val_vamp2 = val_accum.get("vamp2", float("inf")) / max(n_batches, 1)
        if val_vamp2 < best_val_vamp2:
            best_val_vamp2 = val_vamp2
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract out-of-sample metrics on validation set
    model.eval()
    with torch.no_grad():
        val_x_t = torch.as_tensor(val_data[:-tau], dtype=torch.float32).to(device)
        val_x_tau = torch.as_tensor(val_data[tau:], dtype=torch.float32).to(device)
        out_val = model(val_x_t, val_x_tau)

    eigenvalues = out_val["eigenvalues"].cpu().numpy()
    K = out_val["koopman_matrix"].cpu().numpy()

    # VAMP-2 score (sum of squared singular values)
    singular_values = out_val["singular_values"].cpu().numpy()
    vamp2 = float(np.sum(singular_values ** 2))

    # Spectral gap
    spectral_gap = float(KoopmanAnalyzer.compute_spectral_gap(eigenvalues, tau))

    # Complex eigenvalue fraction
    imag_parts = np.abs(eigenvalues.imag)
    n_complex = int(np.sum(imag_parts > 0.01))
    n_modes = len(eigenvalues)

    # Detailed balance violation
    K_np = K
    db_violation = float(np.linalg.norm(K_np - K_np.T, 'fro') / max(np.linalg.norm(K_np, 'fro'), 1e-15))

    return {
        "vamp2": vamp2,
        "spectral_gap": spectral_gap,
        "n_complex_modes": n_complex,
        "complex_fraction": n_complex / max(n_modes, 1),
        "db_violation": db_violation,
        "best_val_vamp2": float(best_val_vamp2),
        "epochs_trained": epoch,
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Walk-forward cross-validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--mode", type=str, default="univariate",
                        choices=["univariate", "multiasset"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(str(project_root / args.config))

    mode_config_path = project_root / "config" / f"{args.mode}.yaml"
    if mode_config_path.exists():
        mode_config = load_config(str(mode_config_path))
        config = merge_configs(config, mode_config)

    set_seed(args.seed)
    device = get_device()

    # Load data
    cache_file = project_root / "data" / "prices.csv"
    if not cache_file.exists():
        raise FileNotFoundError(f"Price data not found at {cache_file}. Run run_main.py first.")

    prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    tickers = config.get("data", {}).get("tickers", ["SPY"])
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available] if available else prices.iloc[:, :1]

    # Preprocess
    log_returns = compute_log_returns(prices, drop_first=True)
    train_end_idx = len(log_returns)  # use all data for standardization
    standardized, stats = standardize_returns(
        log_returns,
        method=config.get("data", {}).get("standardization", "zscore"),
        train_end_idx=train_end_idx,
    )
    std_arr = standardized.values if isinstance(standardized, pd.DataFrame) else standardized
    dates = standardized.index if isinstance(standardized, pd.DataFrame) else pd.RangeIndex(len(std_arr))

    embedding_dim = config.get("data", {}).get("embedding_dim", 5)
    if embedding_dim >= 2:
        embedded = time_delay_embedding(std_arr, embedding_dim=embedding_dim, delay=1)
        trim = std_arr.shape[0] - embedded.shape[0]
        dates = dates[trim:]
    else:
        embedded = std_arr

    # Generate folds
    folds = generate_walk_forward_folds(dates, n_folds=args.n_folds)
    logger.info("Generated %d walk-forward folds", len(folds))

    # Run each fold
    fold_results = []
    for fold_info in folds:
        k = fold_info["fold"]
        logger.info(
            "Fold %d: train=[...%s] (%d pts), val=[...%s] (%d pts)",
            k, fold_info["train_end"], fold_info["n_train"],
            fold_info["val_end"], fold_info["n_val"],
        )
        t0 = time.time()

        metrics = train_fold(
            embedded,
            fold_info["train_mask"].values if hasattr(fold_info["train_mask"], 'values') else fold_info["train_mask"],
            fold_info["val_mask"].values if hasattr(fold_info["val_mask"], 'values') else fold_info["val_mask"],
            config, device, seed=args.seed + k,
        )
        elapsed = time.time() - t0

        metrics["fold"] = k
        metrics["train_end"] = fold_info["train_end"]
        metrics["val_end"] = fold_info["val_end"]
        metrics["n_train"] = fold_info["n_train"]
        metrics["n_val"] = fold_info["n_val"]
        metrics["elapsed_sec"] = elapsed

        fold_results.append(metrics)
        logger.info(
            "  Fold %d: VAMP-2=%.4f, SG=%.4f, DB=%.3f, complex=%d (%d epochs, %.0fs)",
            k, metrics["vamp2"], metrics["spectral_gap"],
            metrics["db_violation"], metrics["n_complex_modes"],
            metrics["epochs_trained"], elapsed,
        )

    # Aggregate
    agg_metrics = ["vamp2", "spectral_gap", "db_violation", "complex_fraction", "n_complex_modes"]
    summary = {"mode": args.mode, "n_folds": len(fold_results), "folds": fold_results}

    for metric in agg_metrics:
        vals = [f[metric] for f in fold_results if metric in f]
        if vals:
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_path = output_dir / f"cv_results_{args.mode}.json"
    with open(cv_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("KTND-Finance: Walk-Forward Cross-Validation")
    print("=" * 70)
    print(f"  Mode:    {args.mode}")
    print(f"  Folds:   {len(fold_results)}")
    for metric in agg_metrics:
        mk, sk = f"{metric}_mean", f"{metric}_std"
        if mk in summary:
            print(f"  {metric:25s}  {summary[mk]:.4f} +/- {summary[sk]:.4f}")
    print(f"\n  Results saved to: {cv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
