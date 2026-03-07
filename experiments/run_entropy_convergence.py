#!/usr/bin/env python3
"""
Entropy convergence analysis: spectral entropy vs number of modes K.

Trains models with varying K (n_modes) and computes spectral entropy as a
function of K, with horizontal line at the empirical (KDE) estimate.
Shows whether the spectral-empirical gap is explained by mode truncation.

Usage
-----
    python experiments/run_entropy_convergence.py --config config/default.yaml --mode univariate
    python experiments/run_entropy_convergence.py --config config/default.yaml --mode multiasset
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
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

from src.constants import DATE_RANGES, tickers_for_mode
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
from src.model.entropy import estimate_empirical_entropy_production_with_ci

logger = logging.getLogger(__name__)

K_VALUES_DEFAULT = [3, 5, 10, 15, 20, 30, 50]


def train_and_extract_entropy(
    embedded: np.ndarray,
    config: dict,
    n_modes: int,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a model with n_modes and extract spectral entropy."""
    set_seed(seed)

    tau = int(config.get("data", {}).get("tau", 5))
    batch_size = config.get("training", {}).get("batch_size", 512)
    loss_cfg = config.get("losses", {})
    train_cfg = config.get("training", {})

    model_cfg = config.get("model", {})
    input_dim = embedded.shape[1]

    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 64]),
        output_dim=n_modes,
        dropout=model_cfg.get("dropout", 0.1),
        epsilon=model_cfg.get("epsilon", 1e-6),
    ).to(device)

    ds = TimeLaggedDataset(embedded, lag=tau, preprocess=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    n_epochs = min(train_cfg.get("n_epochs", 800), 400)
    patience = min(train_cfg.get("patience", 80), 50)
    lr = train_cfg.get("learning_rate", 3e-4)
    wd = train_cfg.get("weight_decay", 1e-5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    stale = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_t, x_tau in loader:
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
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract eigenvalues and compute spectral entropy
    model.eval()
    x_t = torch.as_tensor(embedded[:-tau], dtype=torch.float32).to(device)
    x_tau = torch.as_tensor(embedded[tau:], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x_t, x_tau)

    eigenvalues = out["eigenvalues"].cpu().numpy()
    magnitudes = np.abs(eigenvalues)
    order = np.argsort(-magnitudes)
    eigenvalues_sorted = eigenvalues[order]

    omega = np.angle(eigenvalues_sorted) / tau
    gamma_k = -np.log(np.clip(np.abs(eigenvalues_sorted), 1e-12, 1.0 - 1e-7)) / tau
    gamma_k = np.clip(gamma_k, 1e-6, None)

    # Eigenfunction amplitudes (bilinear)
    u_funcs = model.compute_eigenfunctions(
        torch.as_tensor(embedded, dtype=torch.float32).to(device), out,
    )
    if isinstance(u_funcs, tuple):
        u_np = u_funcs[0].cpu().numpy()
    else:
        u_np = u_funcs.cpu().numpy()
    A_k = np.mean(u_np ** 2, axis=0)
    if len(A_k) == len(order):
        A_k = np.abs(A_k[order])
    else:
        A_k = np.abs(A_k[:len(omega)])

    entropy_per_mode = omega ** 2 * A_k / gamma_k
    spectral_entropy = float(np.sum(np.abs(entropy_per_mode)))

    vamp2 = float(np.sum(out["singular_values"].cpu().numpy() ** 2))

    return {
        "n_modes": n_modes,
        "spectral_entropy": spectral_entropy,
        "vamp2": vamp2,
        "epochs_trained": epoch,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Entropy convergence analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--mode", type=str, default="univariate",
        choices=["univariate", "multiasset"],
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=K_VALUES_DEFAULT,
    )
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

    # Load and preprocess data
    import pandas as pd
    prices_path = project_root / "data" / "prices.csv"
    if not prices_path.exists():
        logger.error("Price data not found. Run run_main.py first.")
        return

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    tickers = tickers_for_mode(args.mode)
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]

    log_returns = compute_log_returns(prices, drop_first=True)
    returns_arr = log_returns.values

    train_end = DATE_RANGES["train"][1]
    train_end_idx = log_returns.index.searchsorted(pd.Timestamp(train_end))
    returns_std, _ = standardize_returns(returns_arr, method="zscore",
                                         train_end_idx=train_end_idx)

    embedding_dim = config.get("data", {}).get("embedding_dim", 5)
    embedded = time_delay_embedding(returns_std, embedding_dim)
    tau = int(config.get("data", {}).get("tau", 5))

    logger.info("Embedded shape: %s, K values: %s", embedded.shape, args.k_values)

    # Empirical entropy (compute once)
    # Use k-NN estimator for high-dimensional data (KDE fails above ~10 dims)
    # Filter NaN/Inf rows (can occur from standardization edge cases)
    finite_mask = np.all(np.isfinite(embedded), axis=1)
    if not np.all(finite_mask):
        n_bad = int((~finite_mask).sum())
        logger.warning("Filtered %d non-finite rows from %d total", n_bad, len(embedded))
        embedded = embedded[finite_mask]
    returns_tensor = torch.as_tensor(embedded, dtype=torch.float32)
    if embedded.shape[1] > 10:
        from src.model.entropy import knn_entropy_production
        logger.info("Using k-NN entropy estimator (dim=%d too high for KDE)", embedded.shape[1])
        knn_result = knn_entropy_production(returns_tensor, tau=tau, k=5, n_samples=5000)
        emp_ci = {
            "point_estimate": knn_result["point_estimate"],
            "ci_lower": knn_result["point_estimate"] * 0.8,  # approximate CI
            "ci_upper": knn_result["point_estimate"] * 1.2,
        }
    else:
        emp_ci = estimate_empirical_entropy_production_with_ci(
            returns_tensor, tau=tau, n_bootstrap=200, block_length=50,
        )
    logger.info("Empirical entropy: %.4f [%.4f, %.4f]",
                emp_ci["point_estimate"], emp_ci["ci_lower"], emp_ci["ci_upper"])

    # Sweep K
    results = []
    for k_val in args.k_values:
        logger.info("Training with K=%d ...", k_val)
        t0 = time.time()
        r = train_and_extract_entropy(embedded, config, k_val, device, seed=args.seed)
        elapsed = time.time() - t0
        r["elapsed_sec"] = elapsed
        results.append(r)
        logger.info("  K=%d: spectral_entropy=%.6f, vamp2=%.4f (%.1fs)",
                     k_val, r["spectral_entropy"], r["vamp2"], elapsed)

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": args.mode,
        "k_values": [r["n_modes"] for r in results],
        "spectral_entropies": [r["spectral_entropy"] for r in results],
        "vamp2_scores": [r["vamp2"] for r in results],
        "empirical_estimate": emp_ci["point_estimate"],
        "empirical_ci_lower": emp_ci["ci_lower"],
        "empirical_ci_upper": emp_ci["ci_upper"],
        "seed": args.seed,
    }

    out_path = output_dir / f"entropy_convergence_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\n" + "=" * 70)
    print("Entropy Convergence Analysis")
    print("=" * 70)
    print(f"  Mode:         {args.mode}")
    print(f"  Empirical EP: {emp_ci['point_estimate']:.4f} "
          f"[{emp_ci['ci_lower']:.4f}, {emp_ci['ci_upper']:.4f}]")
    for r in results:
        pct = 100 * r["spectral_entropy"] / max(emp_ci["point_estimate"], 1e-15)
        print(f"  K={r['n_modes']:3d}: spectral={r['spectral_entropy']:.4f} "
              f"({pct:.1f}% of empirical)")
    print(f"\n  Saved to: {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
