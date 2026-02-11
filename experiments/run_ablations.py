#!/usr/bin/env python3
"""
Ablation study runner for KTND-Finance (PRD Section 9.3).

Iterates over all ablation configurations in config/ablation/, runs each
variant for N_SEEDS independent trials, and aggregates results into a
summary table.  Sweep configs (with a ``sweep.parameter`` / ``sweep.values``
block) are expanded into one trial per value.  Non-sweep configs are run
as-is.

Parallelism across seeds is handled via joblib.

Usage
-----
    python experiments/run_ablations.py --config config/default.yaml
    python experiments/run_ablations.py --config config/default.yaml --n-seeds 5 --n-jobs 4
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

from src.constants import DATE_RANGES
from src.utils.config import load_config, merge_configs, set_nested, save_config
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

N_SEEDS_DEFAULT = 10


# =====================================================================
# Single-seed trial
# =====================================================================

def run_single_trial(
    config: dict,
    seed: int,
    device_str: str,
) -> Dict[str, float]:
    """Train a single VAMPNet with the given config and seed.

    Returns a flat dict of scalar metrics.
    """
    set_seed(seed)
    device = torch.device(device_str)
    project_root = Path(__file__).resolve().parent.parent

    # -- Data -----------------------------------------------------------
    tickers = config.get("data", {}).get("tickers", ["SPY"])
    data_dir = project_root / "data"
    cache_file = data_dir / "prices.csv"

    if cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        avail = [t for t in tickers if t in prices.columns]
        if avail:
            prices = prices[avail]
        else:
            prices = prices.iloc[:, :1]
    else:
        # Fallback: generate synthetic data for ablation testing
        logger.warning("No cached price data. Use run_main.py first to download data.")
        return {"error": 1.0}

    log_returns = compute_log_returns(prices, drop_first=True)
    train_end_idx = len(log_returns.loc[:DATE_RANGES["train"][1]])
    standardized, _ = standardize_returns(
        log_returns,
        method=config.get("data", {}).get("standardization", "zscore"),
        train_end_idx=train_end_idx,
    )
    std_arr = standardized.values if isinstance(standardized, pd.DataFrame) else standardized
    dates = standardized.index if isinstance(standardized, pd.DataFrame) else None

    embedding_dim = config.get("data", {}).get("embedding_dim", 5)
    if embedding_dim >= 2:
        embedded = time_delay_embedding(std_arr, embedding_dim=embedding_dim, delay=1)
        trim = std_arr.shape[0] - embedded.shape[0]
        if dates is not None:
            dates = dates[trim:]
    else:
        embedded = std_arr

    tau = config.get("data", {}).get("tau", 5)
    batch_size = config.get("training", {}).get("batch_size", 512)

    # Splits
    train_mask = (dates >= DATE_RANGES["train"][0]) & (dates <= DATE_RANGES["train"][1])
    val_mask = (dates >= DATE_RANGES["val"][0]) & (dates <= DATE_RANGES["val"][1])
    test_mask = (dates >= DATE_RANGES["test"][0]) & (dates <= DATE_RANGES["test"][1])

    def _make_loader(mask, shuffle):
        ds = TimeLaggedDataset(embedded[mask], lag=tau, preprocess=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = _make_loader(train_mask, shuffle=True)
    val_loader = _make_loader(val_mask, shuffle=False)
    test_loader = _make_loader(test_mask, shuffle=False)

    # -- Model ----------------------------------------------------------
    input_dim = embedded.shape[1]
    model_cfg = config.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [128, 128, 64])
    output_dim = model_cfg.get("n_modes", 10)
    dropout = model_cfg.get("dropout", 0.1)

    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)

    # -- Training -------------------------------------------------------
    train_cfg = config.get("training", {})
    loss_cfg = config.get("losses", {})
    n_epochs = train_cfg.get("n_epochs", 500)
    patience = train_cfg.get("patience", 50)
    lr = train_cfg.get("learning_rate", 1e-3)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_vamp2 = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        for x_t, x_tau in train_loader:
            x_t, x_tau = x_t.to(device), x_tau.to(device)
            optimizer.zero_grad()
            out = model(x_t, x_tau)
            loss, _ = total_loss(
                out, tau=float(tau),
                w_vamp2=loss_cfg.get("w_vamp2", 1.0),
                w_orthogonality=loss_cfg.get("beta_orthogonality", 0.01),
                w_spectral=loss_cfg.get("gamma_regularization", 0.1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_vamp2_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x_t, x_tau in val_loader:
                x_t, x_tau = x_t.to(device), x_tau.to(device)
                out = model(x_t, x_tau)
                _, ld = total_loss(out, tau=float(tau))
                val_vamp2_sum += ld["vamp2"].item()
                val_n += 1
        val_vamp2 = val_vamp2_sum / max(val_n, 1)

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

    # -- Evaluate metrics -----------------------------------------------
    model.eval()
    metrics: Dict[str, float] = {}

    # Test VAMP-2
    test_vamp2_sum, test_n = 0.0, 0
    with torch.no_grad():
        for x_t, x_tau in test_loader:
            x_t, x_tau = x_t.to(device), x_tau.to(device)
            out = model(x_t, x_tau)
            _, ld = total_loss(out, tau=float(tau))
            test_vamp2_sum += ld["vamp2"].item()
            test_n += 1
    metrics["vamp2"] = test_vamp2_sum / max(test_n, 1)

    # Full forward for spectral quantities
    with torch.no_grad():
        x_t_full = torch.as_tensor(embedded[:-tau], dtype=torch.float32).to(device)
        x_tau_full = torch.as_tensor(embedded[tau:], dtype=torch.float32).to(device)
        out = model(x_t_full, x_tau_full)

    eigenvalues = out["eigenvalues"].cpu().numpy()
    magnitudes = np.abs(eigenvalues)
    order = np.argsort(-magnitudes)
    sorted_mag = magnitudes[order]

    metrics["spectral_gap"] = float(sorted_mag[0] - sorted_mag[1]) if len(sorted_mag) > 1 else 0.0

    # Eigenvalue CV (coefficient of variation)
    metrics["eigenvalue_cv"] = float(np.std(sorted_mag) / max(np.mean(sorted_mag), 1e-15))

    # Entropy total
    omega = np.angle(eigenvalues[order]) / tau
    x_all = torch.as_tensor(embedded, dtype=torch.float32).to(device)
    with torch.no_grad():
        u, v = model.compute_eigenfunctions(x_all, out)
    A_k = np.mean(u.cpu().numpy() ** 2, axis=0)
    entropy_per_mode = omega ** 2 * A_k[:len(omega)]
    metrics["entropy_total"] = float(np.sum(np.abs(entropy_per_mode)))

    return metrics


# =====================================================================
# Ablation orchestration
# =====================================================================

def enumerate_ablation_configs(
    base_config: dict,
    ablation_dir: Path,
) -> List[Tuple[str, dict]]:
    """Parse all ablation YAML files and expand sweeps.

    Returns a list of (name, config) pairs.
    """
    configs: List[Tuple[str, dict]] = []

    for yaml_file in sorted(ablation_dir.glob("*.yaml")):
        ablation_cfg = load_config(str(yaml_file))
        name_base = yaml_file.stem

        # Check for sweep specification
        sweep = ablation_cfg.pop("sweep", None)
        # Remove the 'defaults' key (handled by merge)
        ablation_cfg.pop("defaults", None)

        if sweep is not None:
            parameter = sweep["parameter"]
            values = sweep["values"]
            for val in values:
                merged = merge_configs(base_config, ablation_cfg)
                set_nested(merged, parameter, val)
                val_label = str(val).replace(" ", "").replace(",", "_")
                configs.append((f"{name_base}_{parameter}={val_label}", merged))
        else:
            merged = merge_configs(base_config, ablation_cfg)
            configs.append((name_base, merged))

    return configs


def run_ablation_seeds(
    name: str,
    config: dict,
    n_seeds: int,
    n_jobs: int,
    device_str: str,
) -> Dict[str, Any]:
    """Run one ablation config across multiple seeds, return aggregated stats."""
    logger.info("Running ablation '%s' with %d seeds ...", name, n_seeds)

    try:
        from joblib import Parallel, delayed
        results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_single_trial)(config, seed, device_str)
            for seed in range(n_seeds)
        )
    except ImportError:
        logger.warning("joblib not available; falling back to sequential execution.")
        results_list = [
            run_single_trial(config, seed, device_str)
            for seed in range(n_seeds)
        ]

    # Filter out error results
    valid = [r for r in results_list if "error" not in r]
    if not valid:
        return {"name": name, "n_valid": 0}

    # Aggregate
    all_keys = set()
    for r in valid:
        all_keys.update(r.keys())

    aggregated: Dict[str, Any] = {"name": name, "n_valid": len(valid)}
    for key in sorted(all_keys):
        values = [r[key] for r in valid if key in r]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Ablation study runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the base configuration file.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=N_SEEDS_DEFAULT,
        help="Number of independent seeds per ablation variant.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel jobs for seed runs (via joblib).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override directory for saving ablation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    base_config = load_config(str(config_path))

    ablation_dir = project_root / "config" / "ablation"
    if not ablation_dir.exists():
        logger.error("Ablation config directory not found: %s", ablation_dir)
        sys.exit(1)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Enumerate all ablation variants
    all_ablations = enumerate_ablation_configs(base_config, ablation_dir)
    logger.info("Found %d ablation variants across %d config files",
                len(all_ablations), len(list(ablation_dir.glob("*.yaml"))))

    # Also run the base config as reference
    all_ablations.insert(0, ("baseline", copy.deepcopy(base_config)))

    # Run all ablations
    all_results: List[Dict[str, Any]] = []
    for i, (name, cfg) in enumerate(all_ablations):
        logger.info("[%d/%d] Ablation: %s", i + 1, len(all_ablations), name)
        t0 = time.time()
        try:
            result = run_ablation_seeds(
                name, cfg, args.n_seeds, args.n_jobs, device_str,
            )
            elapsed = time.time() - t0
            result["elapsed_sec"] = elapsed
            all_results.append(result)
            logger.info(
                "  -> %s  n_valid=%d  elapsed=%.1fs",
                name, result.get("n_valid", 0), elapsed,
            )
        except Exception as e:
            logger.error("  -> FAILED: %s\n%s", e, traceback.format_exc())
            all_results.append({"name": name, "n_valid": 0, "error": str(e)})

    # Build summary table
    summary_df = pd.DataFrame(all_results)

    # Reorder columns for readability
    priority_cols = ["name", "n_valid"]
    metric_cols = [
        "vamp2_mean", "vamp2_std",
        "spectral_gap_mean", "spectral_gap_std",
        "entropy_total_mean", "entropy_total_std",
        "eigenvalue_cv_mean", "eigenvalue_cv_std",
    ]
    existing_priority = [c for c in priority_cols if c in summary_df.columns]
    existing_metrics = [c for c in metric_cols if c in summary_df.columns]
    remaining = [c for c in summary_df.columns if c not in existing_priority + existing_metrics]
    summary_df = summary_df[existing_priority + existing_metrics + remaining]

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("KTND-Finance: Ablation Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {csv_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
