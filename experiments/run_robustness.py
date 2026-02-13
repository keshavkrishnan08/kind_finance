#!/usr/bin/env python3
"""
Robustness and statistical validation runner for KTND-Finance (PRD Section 10).

Performs the full battery of statistical tests required to validate the
Koopman-Thermodynamic model:

    1. Chapman-Kolmogorov consistency test
    2. Bootstrap confidence intervals for eigenvalues
    3. Permutation test for irreversibility (with Cohen's d effect size)
    4. Ljung-Box test on residuals
    5. Granger causality: spectral gap -> VIX
    6. KS test on eigenfunctions (train vs test distribution shift)
    7. Time-reversal asymmetry (model-free detailed balance violation)

All results are saved to a structured JSON file.

Usage
-----
    python experiments/run_robustness.py --config config/default.yaml
    python experiments/run_robustness.py --checkpoint outputs/models/vampnet_univariate.pt
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ---------------------------------------------------------------------------

from src.constants import DATE_RANGES
from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.data.preprocessing import (
    compute_log_returns,
    standardize_returns,
    time_delay_embedding,
)
from src.data.augmentation import block_bootstrap, random_time_reversal, iaaft_surrogate
from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.losses import total_loss
from src.model.koopman import KoopmanAnalyzer

logger = logging.getLogger(__name__)


# =====================================================================
# Helper: load model and data
# =====================================================================

def load_model_and_data(
    config: dict,
    checkpoint_path: Optional[str],
    project_root: Path,
    device: torch.device,
) -> Tuple[NonEquilibriumVAMPNet, np.ndarray, pd.DatetimeIndex, dict]:
    """Load trained model and preprocessed data.

    Returns
    -------
    model : NonEquilibriumVAMPNet
    embedded : np.ndarray
    dates : pd.DatetimeIndex
    config : dict
    """
    # -- Load price data --
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

    # -- Preprocess --
    log_returns = compute_log_returns(prices, drop_first=True)
    train_end_idx = len(log_returns.loc[:DATE_RANGES["train"][1]])
    standardized, stats = standardize_returns(
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

    # -- Load model --
    input_dim = embedded.shape[1]
    model_cfg = config.get("model", {})

    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 64]),
        output_dim=model_cfg.get("n_modes", 10),
        dropout=model_cfg.get("dropout", 0.1),
        epsilon=model_cfg.get("epsilon", 1e-6),
    ).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded model checkpoint from %s", checkpoint_path)
    else:
        logger.warning(
            "No checkpoint found at '%s'; using randomly initialized model. "
            "Results will not be meaningful. Run run_main.py first.",
            checkpoint_path,
        )

    model.eval()
    return model, embedded, dates, config


# =====================================================================
# Test 1: Chapman-Kolmogorov consistency
# =====================================================================

def chapman_kolmogorov_test(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    tau: int,
    device: torch.device,
    n_steps: int = 3,
) -> Dict[str, Any]:
    """Chapman-Kolmogorov test for Markov consistency.

    Checks whether K(n*tau) approximately equals K(tau)^n by comparing
    eigenvalues estimated at multiples of the lag time.  Uses relative
    error and top-k eigenvalue matching for robustness.

    Returns
    -------
    dict with 'ck_errors', 'ck_p_value', and diagnostic info.
    """
    logger.info("Running Chapman-Kolmogorov test (n_steps=%d) ...", n_steps)
    model.eval()

    # Estimate K at the base lag tau
    x_t = torch.as_tensor(embedded[:-tau], dtype=torch.float32).to(device)
    x_tau = torch.as_tensor(embedded[tau:], dtype=torch.float32).to(device)

    with torch.no_grad():
        out_base = model(x_t, x_tau)
    K_base = out_base["koopman_matrix"].cpu().numpy()
    eigs_base = np.linalg.eigvals(K_base)
    eigs_base_sorted = eigs_base[np.argsort(-np.abs(eigs_base))]

    # Only compare top-k eigenvalues (rest dominated by noise)
    n_modes = len(eigs_base_sorted)
    k_compare = max(3, n_modes // 2)

    ck_errors = []
    for n in range(2, n_steps + 1):
        n_tau = n * tau
        if n_tau >= len(embedded):
            break

        # Predicted: eigenvalues of K(tau)^n = eigenvalues(K(tau))^n
        eigs_predicted = eigs_base_sorted[:k_compare] ** n

        # Estimated: eigenvalues from K(n*tau) directly
        x_t_n = torch.as_tensor(embedded[:-n_tau], dtype=torch.float32).to(device)
        x_tau_n = torch.as_tensor(embedded[n_tau:], dtype=torch.float32).to(device)

        with torch.no_grad():
            out_n = model(x_t_n, x_tau_n)
        K_n = out_n["koopman_matrix"].cpu().numpy()
        eigs_n = np.linalg.eigvals(K_n)
        eigs_n_sorted = eigs_n[np.argsort(-np.abs(eigs_n))][:k_compare]

        # Relative error on magnitudes (avoids issues when values are near 0)
        mags_pred = np.abs(eigs_predicted)
        mags_obs = np.abs(eigs_n_sorted)
        denom = np.maximum(mags_pred, 1e-8)
        rel_error = float(np.mean(np.abs(mags_pred - mags_obs) / denom))
        ck_errors.append({"n": n, "error": rel_error})

    errors = [e["error"] for e in ck_errors]
    mean_error = float(np.mean(errors)) if errors else 0.0

    # Block-bootstrap null: small blocks to destroy Markov structure
    n_bootstrap = 200
    block_size = min(10, len(embedded) // 20)
    block_size = max(block_size, 2)
    rng = np.random.default_rng(seed=42)
    boot_mean_errors = []

    for _b in range(n_bootstrap):
        n_blocks = int(np.ceil(len(embedded) / block_size))
        starts = rng.integers(0, len(embedded) - block_size, size=n_blocks)
        boot_idx = np.concatenate(
            [np.arange(s, s + block_size) for s in starts]
        )[:len(embedded)]
        emb_boot = embedded[boot_idx]

        boot_errors = []
        for n in range(2, n_steps + 1):
            n_tau = n * tau
            if n_tau >= len(emb_boot):
                break
            x_t_b = torch.as_tensor(emb_boot[:-tau], dtype=torch.float32).to(device)
            x_tau_b = torch.as_tensor(emb_boot[tau:], dtype=torch.float32).to(device)
            with torch.no_grad():
                out_b = model(x_t_b, x_tau_b)
            K_b = out_b["koopman_matrix"].cpu().numpy()
            eigs_b = np.linalg.eigvals(K_b)
            eigs_b_sorted = eigs_b[np.argsort(-np.abs(eigs_b))][:k_compare]

            eigs_pred_b = eigs_b_sorted ** n
            x_t_bn = torch.as_tensor(emb_boot[:-n_tau], dtype=torch.float32).to(device)
            x_tau_bn = torch.as_tensor(emb_boot[n_tau:], dtype=torch.float32).to(device)
            with torch.no_grad():
                out_bn = model(x_t_bn, x_tau_bn)
            K_bn = out_bn["koopman_matrix"].cpu().numpy()
            eigs_bn = np.linalg.eigvals(K_bn)
            eigs_bn_sorted = eigs_bn[np.argsort(-np.abs(eigs_bn))][:k_compare]

            mags_pred_b = np.abs(eigs_pred_b)
            mags_obs_b = np.abs(eigs_bn_sorted)
            denom_b = np.maximum(mags_pred_b, 1e-8)
            err_b = float(np.mean(np.abs(mags_pred_b - mags_obs_b) / denom_b))
            boot_errors.append(err_b)

        if boot_errors:
            boot_mean_errors.append(float(np.mean(boot_errors)))

    # p-value: fraction of bootstrap replicates with error <= observed
    # (bootstrap destroys Markov property, so errors should be larger)
    boot_arr = np.array(boot_mean_errors)
    p_value = float(np.mean(boot_arr <= mean_error)) if len(boot_arr) > 0 else 1.0

    return {
        "test": "Chapman-Kolmogorov",
        "ck_errors": ck_errors,
        "mean_error": mean_error,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "n_steps_tested": len(ck_errors),
        "k_eigenvalues_compared": k_compare,
        "passed": mean_error < float(np.median(boot_arr)) if len(boot_arr) > 0 else True,
    }


# =====================================================================
# Test 2: Bootstrap eigenvalue CIs
# =====================================================================

def bootstrap_eigenvalue_cis(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    tau: int,
    device: torch.device,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals for Koopman eigenvalue magnitudes.

    Uses block bootstrap to resample time series while preserving
    short-range autocorrelation.
    """
    logger.info("Running bootstrap eigenvalue CIs (n_bootstrap=%d) ...", n_bootstrap)
    model.eval()

    block_size = max(2, int(len(embedded) ** (1 / 3)))
    rng = np.random.default_rng(42)

    all_magnitudes: List[np.ndarray] = []

    for b in range(n_bootstrap):
        # Block bootstrap resample
        resampled = block_bootstrap(embedded, block_size=block_size, n_samples=1, rng=rng)[0]

        x_t = torch.as_tensor(resampled[:-tau], dtype=torch.float32).to(device)
        x_tau = torch.as_tensor(resampled[tau:], dtype=torch.float32).to(device)

        with torch.no_grad():
            out = model(x_t, x_tau)

        eigs = out["eigenvalues"].cpu().numpy()
        mags = np.sort(np.abs(eigs))[::-1]
        all_magnitudes.append(mags)

    # Stack and compute quantiles
    all_mags = np.array(all_magnitudes)  # (n_bootstrap, n_modes)
    alpha = (1 - ci_level) / 2

    ci_lower = np.percentile(all_mags, 100 * alpha, axis=0)
    ci_upper = np.percentile(all_mags, 100 * (1 - alpha), axis=0)
    mean_mags = np.mean(all_mags, axis=0)
    std_mags = np.std(all_mags, axis=0)

    results_per_mode = []
    for k in range(all_mags.shape[1]):
        results_per_mode.append({
            "mode": k,
            "mean_magnitude": float(mean_mags[k]),
            "std_magnitude": float(std_mags[k]),
            "ci_lower": float(ci_lower[k]),
            "ci_upper": float(ci_upper[k]),
        })

    return {
        "test": "Bootstrap_Eigenvalue_CI",
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "block_size": block_size,
        "modes": results_per_mode,
    }


# =====================================================================
# Test 3: Permutation test for irreversibility
# =====================================================================

def permutation_test_irreversibility(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    tau: int,
    device: torch.device,
    n_permutations: int = 1000,
) -> Dict[str, Any]:
    """Test whether the observed irreversibility is significantly different
    from a time-reversible null model.

    Uses IAAFT surrogates (Schreiber & Schmitz, 2000) as the null model.
    IAAFT preserves the power spectrum and amplitude distribution of each
    channel but destroys all nonlinear temporal structure, including
    time-reversal asymmetry.  This is the gold-standard surrogate method
    for testing irreversibility in nonlinear dynamics.
    """
    logger.info("Running permutation test for irreversibility (n=%d, IAAFT surrogates) ...", n_permutations)
    model.eval()

    # Observed irreversibility
    x_all = torch.as_tensor(embedded, dtype=torch.float32).to(device)
    x_t = torch.as_tensor(embedded[:-tau], dtype=torch.float32).to(device)
    x_tau = torch.as_tensor(embedded[tau:], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x_t, x_tau)
        irrev_observed = model.compute_irreversibility_field(x_all, out).cpu().numpy()

    mean_irrev_observed = float(np.mean(irrev_observed))

    # Null distribution: IAAFT surrogates (destroys ALL temporal asymmetry)
    rng = np.random.default_rng(42)
    null_means: List[float] = []

    for p in range(n_permutations):
        surrogate = iaaft_surrogate(embedded, n_samples=1, max_iter=100, rng=rng)[0]

        x_t_s = torch.as_tensor(surrogate[:-tau], dtype=torch.float32).to(device)
        x_tau_s = torch.as_tensor(surrogate[tau:], dtype=torch.float32).to(device)
        x_all_s = torch.as_tensor(surrogate, dtype=torch.float32).to(device)

        with torch.no_grad():
            out_s = model(x_t_s, x_tau_s)
            irrev_s = model.compute_irreversibility_field(x_all_s, out_s).cpu().numpy()

        null_means.append(float(np.mean(irrev_s)))

    null_arr = np.array(null_means)
    p_value = float(np.mean(null_arr >= mean_irrev_observed))

    # Cohen's d effect size: (observed - null_mean) / null_std
    null_std = float(np.std(null_arr))
    cohens_d = float((mean_irrev_observed - np.mean(null_arr)) / max(null_std, 1e-15))

    # Effect size interpretation (Cohen, 1988)
    if abs(cohens_d) >= 0.8:
        effect_interpretation = "large"
    elif abs(cohens_d) >= 0.5:
        effect_interpretation = "medium"
    elif abs(cohens_d) >= 0.2:
        effect_interpretation = "small"
    else:
        effect_interpretation = "negligible"

    return {
        "test": "Permutation_Irreversibility",
        "surrogate_method": "IAAFT",
        "observed_mean_irreversibility": mean_irrev_observed,
        "null_mean": float(np.mean(null_arr)),
        "null_std": null_std,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size": effect_interpretation,
        "n_permutations": n_permutations,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


# =====================================================================
# Test 4: Ljung-Box on residuals
# =====================================================================

def ljung_box_test(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    tau: int,
    device: torch.device,
    n_lags: int = 20,
) -> Dict[str, Any]:
    """Ljung-Box test for residual autocorrelation.

    Tests whether the residuals from the Koopman-predicted one-step-ahead
    outputs are uncorrelated (white noise), indicating the model captures
    the relevant temporal dynamics.
    """
    logger.info("Running Ljung-Box test (n_lags=%d) ...", n_lags)
    model.eval()

    x_t = torch.as_tensor(embedded[:-tau], dtype=torch.float32).to(device)
    x_tau = torch.as_tensor(embedded[tau:], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x_t, x_tau)

    # Residuals: difference between predicted and actual chi_tau
    chi_t = out["chi_t"].cpu().numpy()
    chi_tau = out["chi_tau"].cpu().numpy()
    K = out["koopman_matrix"].cpu().numpy()

    # Predicted chi_tau from chi_t via Koopman
    chi_tau_pred = chi_t @ K.T
    residuals = chi_tau - chi_tau_pred

    # Run Ljung-Box on each output dimension
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        logger.warning("statsmodels not available; skipping Ljung-Box test.")
        return {"test": "Ljung-Box", "skipped": True, "reason": "statsmodels not installed"}

    lb_results = []
    for d in range(residuals.shape[1]):
        lb = acorr_ljungbox(residuals[:, d], lags=n_lags, return_df=True)
        min_p = float(lb["lb_pvalue"].min())
        max_stat = float(lb["lb_stat"].max())
        lb_results.append({
            "dimension": d,
            "min_p_value": min_p,
            "max_stat": max_stat,
            "white_noise": min_p > 0.05,
        })

    overall_min_p = min(r["min_p_value"] for r in lb_results)

    return {
        "test": "Ljung-Box",
        "n_lags": n_lags,
        "per_dimension": lb_results,
        "overall_min_p_value": overall_min_p,
        "all_white_noise": all(r["white_noise"] for r in lb_results),
    }


# =====================================================================
# Test 5: Granger causality (spectral gap -> VIX)
# =====================================================================

def _run_adf_test(series: np.ndarray, name: str) -> Dict[str, Any]:
    """Augmented Dickey-Fuller stationarity test on a single series."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series, autolag="AIC")
    return {
        "series": name,
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags_used": int(result[2]),
        "stationary_5pct": float(result[1]) < 0.05,
    }


def _granger_one_direction(
    y: np.ndarray, x: np.ndarray, max_lag: int, direction: str,
) -> Dict[str, Any]:
    """Run Granger test in one direction with Bonferroni correction."""
    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.column_stack([y, x])
    results_by_lag: List[Dict[str, Any]] = []

    gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    for lag in range(1, max_lag + 1):
        f_stat = gc_results[lag][0]["ssr_ftest"][0]
        p_value = gc_results[lag][0]["ssr_ftest"][1]
        results_by_lag.append({
            "lag": lag,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
        })

    # Bonferroni correction across lags
    min_p = min(r["p_value"] for r in results_by_lag) if results_by_lag else 1.0
    bonferroni_p = min(min_p * max_lag, 1.0)
    best_lag = min(results_by_lag, key=lambda r: r["p_value"])["lag"] if results_by_lag else None

    for r in results_by_lag:
        r["bonferroni_p"] = min(r["p_value"] * max_lag, 1.0)
        r["significant_bonferroni"] = r["bonferroni_p"] < 0.05

    return {
        "direction": direction,
        "results_by_lag": results_by_lag,
        "min_p_value": min_p,
        "bonferroni_min_p": bonferroni_p,
        "best_lag": best_lag,
        "significant_bonferroni": bonferroni_p < 0.05,
    }


def granger_causality_test(
    spectral_gap_ts: Optional[np.ndarray],
    vix_ts: Optional[np.ndarray],
    max_lag: int = 20,
) -> Dict[str, Any]:
    """Bidirectional Granger causality test with ADF pre-check and Bonferroni.

    Tests both directions (spectral_gap -> VIX and VIX -> spectral_gap)
    with proper stationarity verification and multiple-comparison correction.
    """
    logger.info("Running Granger causality test (bidirectional, with ADF check) ...")

    if spectral_gap_ts is None or vix_ts is None:
        logger.warning("Spectral gap or VIX data not available; skipping Granger test.")
        return {
            "test": "Granger_Causality",
            "skipped": True,
            "reason": "Data not available",
        }

    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    except ImportError:
        return {"test": "Granger_Causality", "skipped": True, "reason": "statsmodels not installed"}

    # Align lengths and remove NaN/Inf
    min_len = min(len(spectral_gap_ts), len(vix_ts))
    sg = spectral_gap_ts[:min_len].copy()
    vix = vix_ts[:min_len].copy()
    valid = np.isfinite(sg) & np.isfinite(vix)
    sg = sg[valid]
    vix = vix[valid]

    if len(sg) < max_lag + 10:
        return {"test": "Granger_Causality", "skipped": True, "reason": "Insufficient data"}

    # ADF stationarity pre-check
    adf_sg = _run_adf_test(sg, "spectral_gap")
    adf_vix = _run_adf_test(vix, "vix")
    logger.info("ADF spectral_gap: stat=%.3f, p=%.4f, stationary=%s",
                adf_sg["adf_statistic"], adf_sg["p_value"], adf_sg["stationary_5pct"])
    logger.info("ADF VIX: stat=%.3f, p=%.4f, stationary=%s",
                adf_vix["adf_statistic"], adf_vix["p_value"], adf_vix["stationary_5pct"])

    # Difference non-stationary series
    sg_use = np.diff(sg) if not adf_sg["stationary_5pct"] else sg
    vix_use = np.diff(vix) if not adf_vix["stationary_5pct"] else vix
    # Align after differencing
    min_use = min(len(sg_use), len(vix_use))
    sg_use = sg_use[:min_use]
    vix_use = vix_use[:min_use]

    if len(sg_use) < max_lag + 10:
        return {"test": "Granger_Causality", "skipped": True, "reason": "Insufficient data after differencing"}

    # Bidirectional Granger test with Bonferroni correction
    try:
        fwd = _granger_one_direction(vix_use, sg_use, max_lag, "spectral_gap -> VIX")
        rev = _granger_one_direction(sg_use, vix_use, max_lag, "VIX -> spectral_gap")
    except Exception as e:
        logger.warning("Granger causality test failed: %s", e)
        return {"test": "Granger_Causality", "skipped": True, "reason": str(e)}

    logger.info("Granger (sg->VIX): best_lag=%s, bonferroni_p=%.4f, sig=%s",
                fwd["best_lag"], fwd["bonferroni_min_p"], fwd["significant_bonferroni"])
    logger.info("Granger (VIX->sg): best_lag=%s, bonferroni_p=%.4f, sig=%s",
                rev["best_lag"], rev["bonferroni_min_p"], rev["significant_bonferroni"])

    return {
        "test": "Granger_Causality",
        "adf_checks": {"spectral_gap": adf_sg, "vix": adf_vix},
        "differenced": {
            "spectral_gap": not adf_sg["stationary_5pct"],
            "vix": not adf_vix["stationary_5pct"],
        },
        "forward": fwd,
        "reverse": rev,
        "max_lag": max_lag,
        "any_significant": fwd["significant_bonferroni"] or rev["significant_bonferroni"],
    }


# =====================================================================
# Test 6: KS test on eigenfunctions (train vs test)
# =====================================================================

def ks_test_eigenfunctions(
    model: NonEquilibriumVAMPNet,
    embedded: np.ndarray,
    dates: pd.DatetimeIndex,
    tau: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Kolmogorov-Smirnov test comparing eigenfunction distributions between
    train and test periods.

    A significant result indicates distributional shift in the learned
    representations, which may compromise out-of-sample validity.
    """
    logger.info("Running KS test on eigenfunctions (train vs test) ...")
    model.eval()

    train_mask = (dates >= DATE_RANGES["train"][0]) & (dates <= DATE_RANGES["train"][1])
    test_mask = (dates >= DATE_RANGES["test"][0]) & (dates <= DATE_RANGES["test"][1])

    train_data = embedded[train_mask]
    test_data = embedded[test_mask]

    # Get eigenfunctions for both splits using the full-data Koopman
    x_t_full = torch.as_tensor(embedded[:-tau], dtype=torch.float32).to(device)
    x_tau_full = torch.as_tensor(embedded[tau:], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x_t_full, x_tau_full)

    # Evaluate eigenfunctions on train and test data
    x_train = torch.as_tensor(train_data, dtype=torch.float32).to(device)
    x_test = torch.as_tensor(test_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        u_train, _ = model.compute_eigenfunctions(x_train, out)
        u_test, _ = model.compute_eigenfunctions(x_test, out)

    u_train_np = u_train.cpu().numpy()
    u_test_np = u_test.cpu().numpy()

    # KS test per eigenfunction mode
    ks_results = []
    for k in range(u_train_np.shape[1]):
        stat, pval = scipy_stats.ks_2samp(u_train_np[:, k], u_test_np[:, k])
        ks_results.append({
            "mode": k,
            "ks_statistic": float(stat),
            "p_value": float(pval),
            "significant": float(pval) < 0.05,
        })

    n_significant = sum(r["significant"] for r in ks_results)

    # Bonferroni correction
    bonferroni_alpha = 0.05 / len(ks_results) if ks_results else 0.05
    n_significant_bonferroni = sum(
        r["p_value"] < bonferroni_alpha for r in ks_results
    )

    return {
        "test": "KS_Eigenfunctions",
        "n_modes": len(ks_results),
        "per_mode": ks_results,
        "n_significant_uncorrected": n_significant,
        "n_significant_bonferroni": n_significant_bonferroni,
        "bonferroni_alpha": bonferroni_alpha,
    }


# =====================================================================
# Test 7: Time-reversal asymmetry statistic (model-free)
# =====================================================================

def time_reversal_asymmetry_test(
    embedded: np.ndarray,
    tau: int,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """Model-free time-reversal asymmetry test.

    Computes the third-order time-reversal asymmetry statistic:
        A(tau) = <x(t+tau)^2 * x(t) - x(t)^2 * x(t+tau)>

    For a time-reversible process A(tau) = 0.  Significance is assessed
    via block bootstrap under the null that the asymmetry is zero.

    This provides a model-independent confirmation that the data itself
    (not just the KTND model) violates detailed balance.

    Reference: Weiss (1975); Lawrance (1991) time reversibility tests.
    """
    logger.info("Running time-reversal asymmetry test (tau=%d, bootstrap=%d) ...", tau, n_bootstrap)

    T, D = embedded.shape
    x_t = embedded[:-tau]
    x_tau = embedded[tau:]

    # Asymmetry: <x(t+tau)^2 * x(t) - x(t)^2 * x(t+tau)> per dimension
    asym_per_dim = np.mean(x_tau**2 * x_t - x_t**2 * x_tau, axis=0)
    # Aggregate: mean absolute asymmetry across dimensions
    observed_asym = float(np.mean(np.abs(asym_per_dim)))

    # Block bootstrap for CI
    rng = np.random.default_rng(42)
    block_length = max(50, tau * 5)
    n_eff = T - tau
    n_blocks = max(n_eff // block_length, 2)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Sample block start indices with replacement
        starts = rng.integers(0, n_eff - block_length, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n_eff]

        x_t_b = embedded[indices]
        x_tau_b = embedded[indices + tau] if np.max(indices) + tau < T else embedded[np.minimum(indices + tau, T - 1)]

        asym_b = np.mean(x_tau_b**2 * x_t_b - x_t_b**2 * x_tau_b, axis=0)
        bootstrap_stats.append(float(np.mean(np.abs(asym_b))))

    bootstrap_arr = np.array(bootstrap_stats)
    ci_lower = float(np.percentile(bootstrap_arr, 2.5))
    ci_upper = float(np.percentile(bootstrap_arr, 97.5))
    std_err = float(np.std(bootstrap_arr))

    # p-value: fraction of bootstrap samples with |asym| <= 0
    # (since null is A(tau) = 0 for reversible processes)
    # Use: how extreme is 0 relative to the bootstrap distribution?
    p_value = float(2.0 * min(
        np.mean(bootstrap_arr <= 0),
        np.mean(bootstrap_arr >= 0)
    ))
    p_value = min(p_value, 1.0)

    # Also compute per-dimension z-scores
    per_dim_results = []
    for d in range(D):
        dim_asym = float(asym_per_dim[d])
        # Quick bootstrap std for this dimension
        dim_boot = []
        for _ in range(200):
            idx = rng.integers(0, n_eff, size=n_eff)
            a = np.mean(x_tau[idx, d]**2 * x_t[idx, d] - x_t[idx, d]**2 * x_tau[idx, d])
            dim_boot.append(float(a))
        dim_std = float(np.std(dim_boot))
        z = dim_asym / max(dim_std, 1e-15)
        per_dim_results.append({
            "dimension": d,
            "asymmetry": dim_asym,
            "bootstrap_std": dim_std,
            "z_score": float(z),
            "significant_at_005": abs(z) > 1.96,
        })

    n_sig_dims = sum(r["significant_at_005"] for r in per_dim_results)

    return {
        "test": "Time_Reversal_Asymmetry",
        "observed_asymmetry": observed_asym,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_err,
        "p_value_zero_in_ci": p_value,
        "ci_excludes_zero": ci_lower > 0,
        "n_bootstrap": n_bootstrap,
        "per_dimension": per_dim_results,
        "n_significant_dimensions": n_sig_dims,
        "n_dimensions": D,
        "tau": tau,
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Robustness and statistical validation",
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
        "--n-bootstrap", type=int, default=200,
        help="Number of bootstrap resamples for eigenvalue CIs.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for the irreversibility test.",
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

    # Locate checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(project_root / "outputs" / "models" / f"vampnet_{args.mode}.pt")

    # Load model and data
    model, embedded, dates, config = load_model_and_data(
        config, checkpoint_path, project_root, device,
    )

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run all tests ----
    all_tests: Dict[str, Any] = {
        "metadata": {
            "config": str(project_root / args.config),
            "checkpoint": checkpoint_path,
            "mode": args.mode,
            "tau": tau,
            "n_samples": len(embedded),
            "input_dim": embedded.shape[1],
        }
    }

    # Test 1: Chapman-Kolmogorov
    try:
        ck_result = chapman_kolmogorov_test(model, embedded, tau, device, n_steps=5)
        all_tests["chapman_kolmogorov"] = ck_result
        logger.info("CK test: mean_error=%.6f, p=%.4f, passed=%s",
                     ck_result["mean_error"], ck_result["p_value"], ck_result["passed"])
    except Exception as e:
        logger.error("Chapman-Kolmogorov test failed: %s", e)
        all_tests["chapman_kolmogorov"] = {"error": str(e)}

    # Test 2: Bootstrap eigenvalue CIs
    try:
        bootstrap_result = bootstrap_eigenvalue_cis(
            model, embedded, tau, device, n_bootstrap=args.n_bootstrap,
        )
        all_tests["bootstrap_eigenvalue_ci"] = bootstrap_result
        logger.info("Bootstrap CIs computed for %d modes", len(bootstrap_result["modes"]))
    except Exception as e:
        logger.error("Bootstrap eigenvalue CIs failed: %s", e)
        all_tests["bootstrap_eigenvalue_ci"] = {"error": str(e)}

    # Test 3: Permutation test for irreversibility
    try:
        perm_result = permutation_test_irreversibility(
            model, embedded, tau, device, n_permutations=args.n_permutations,
        )
        all_tests["permutation_irreversibility"] = perm_result
        logger.info("Permutation test: observed=%.6f, null_mean=%.6f, p=%.4f, Cohen's d=%.2f (%s)",
                     perm_result["observed_mean_irreversibility"],
                     perm_result["null_mean"], perm_result["p_value"],
                     perm_result["cohens_d"], perm_result["effect_size"])
    except Exception as e:
        logger.error("Permutation irreversibility test failed: %s", e)
        all_tests["permutation_irreversibility"] = {"error": str(e)}

    # Test 4: Ljung-Box
    try:
        lb_result = ljung_box_test(model, embedded, tau, device)
        all_tests["ljung_box"] = lb_result
        if not lb_result.get("skipped", False):
            logger.info("Ljung-Box: overall_min_p=%.4f, all_white_noise=%s",
                         lb_result["overall_min_p_value"], lb_result["all_white_noise"])
    except Exception as e:
        logger.error("Ljung-Box test failed: %s", e)
        all_tests["ljung_box"] = {"error": str(e)}

    # Test 5: Granger causality
    try:
        # Load spectral gap timeseries WITH dates from rolling analysis
        sg_file = output_dir / "spectral_gap_timeseries.csv"
        spectral_gap_ts = None
        vix_ts = None

        if sg_file.exists():
            sg_df = pd.read_csv(sg_file)
            if "spectral_gap" in sg_df.columns and "center_date" in sg_df.columns:
                sg_dates = pd.to_datetime(sg_df["center_date"])
                sg_series = pd.Series(
                    sg_df["spectral_gap"].values, index=sg_dates
                ).sort_index()
                sg_series = sg_series[~sg_series.index.duplicated(keep="first")]

                # Load VIX with date index and align on common dates
                vix_file = project_root / "data" / "vix.csv"
                if vix_file.exists():
                    vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
                    vix_col = "Close" if "Close" in vix_df.columns else vix_df.columns[0]
                    vix_series = vix_df[vix_col].sort_index()
                    vix_series = vix_series[~vix_series.index.duplicated(keep="first")]

                    # Date-align: only use dates present in both series
                    common_dates = sg_series.index.intersection(vix_series.index)
                    if len(common_dates) > 50:
                        spectral_gap_ts = sg_series.loc[common_dates].values
                        vix_ts = vix_series.loc[common_dates].values
                        logger.info("Granger: %d common dates between spectral gap and VIX",
                                    len(common_dates))
                    else:
                        logger.warning("Granger: only %d common dates, need >50", len(common_dates))

        gc_result = granger_causality_test(spectral_gap_ts, vix_ts)
        all_tests["granger_causality"] = gc_result
        if not gc_result.get("skipped", False):
            fwd_p = gc_result["forward"]["bonferroni_min_p"]
            rev_p = gc_result["reverse"]["bonferroni_min_p"]
            logger.info("Granger causality (sg->VIX): bonferroni_p=%.4f; (VIX->sg): bonferroni_p=%.4f",
                         fwd_p, rev_p)
    except Exception as e:
        logger.error("Granger causality test failed: %s", e)
        all_tests["granger_causality"] = {"error": str(e)}

    # Test 6: KS test on eigenfunctions
    try:
        ks_result = ks_test_eigenfunctions(model, embedded, dates, tau, device)
        all_tests["ks_eigenfunctions"] = ks_result
        logger.info("KS test: %d/%d modes significant (uncorrected), %d (Bonferroni)",
                     ks_result["n_significant_uncorrected"],
                     ks_result["n_modes"],
                     ks_result["n_significant_bonferroni"])
    except Exception as e:
        logger.error("KS test failed: %s", e)
        all_tests["ks_eigenfunctions"] = {"error": str(e)}

    # Test 7: Time-reversal asymmetry (model-free)
    try:
        tra_result = time_reversal_asymmetry_test(embedded, tau, n_bootstrap=args.n_bootstrap)
        all_tests["time_reversal_asymmetry"] = tra_result
        logger.info("Time-reversal asymmetry: observed=%.6f, CI=[%.6f, %.6f], %d/%d dims significant",
                     tra_result["observed_asymmetry"], tra_result["ci_lower"], tra_result["ci_upper"],
                     tra_result["n_significant_dimensions"], tra_result["n_dimensions"])
    except Exception as e:
        logger.error("Time-reversal asymmetry test failed: %s", e)
        all_tests["time_reversal_asymmetry"] = {"error": str(e)}

    # ---- Save all results ----
    suffix = f"_{args.mode}" if args.mode != "univariate" else ""
    results_path = output_dir / f"statistical_tests{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_tests, f, indent=2, default=str)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("KTND-Finance: Statistical Validation Summary")
    print("=" * 70)

    for test_name, result in all_tests.items():
        if test_name == "metadata":
            continue
        if isinstance(result, dict):
            if "error" in result:
                status = f"FAILED ({result['error'][:60]})"
            elif result.get("skipped", False):
                status = f"SKIPPED ({result.get('reason', 'unknown')})"
            elif "passed" in result:
                status = "PASSED" if result["passed"] else "FAILED"
            elif "p_value" in result:
                p = result["p_value"]
                d_str = f", d={result['cohens_d']:.2f} ({result['effect_size']})" if "cohens_d" in result else ""
                status = f"p={p:.4f} ({'sig' if p < 0.05 else 'n.s.'}){d_str}"
            elif "any_significant" in result:
                status = "SIGNIFICANT" if result["any_significant"] else "NOT SIGNIFICANT"
            else:
                status = "COMPLETED"
            print(f"  {test_name:<35s}  {status}")

    print(f"\nResults saved to: {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
