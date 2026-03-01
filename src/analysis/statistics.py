"""Statistical tests for Koopman spectral analysis of financial time series.

Implements the validation battery described in PRD Sections 10.2--10.4:
bootstrap confidence intervals for eigenvalues, permutation tests for
irreversibility, Granger causality, Ljung--Box residual diagnostics,
Kolmogorov--Smirnov eigenfunction stability, and time-series
cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Collection of statistical validation methods for KTND-Finance models."""

    # ------------------------------------------------------------------
    # 10.2  Bootstrap confidence intervals for eigenvalues
    # ------------------------------------------------------------------

    @staticmethod
    def bootstrap_eigenvalue_ci(
        model: torch.nn.Module,
        data: torch.Tensor,
        tau: float,
        n_bootstrap: int = 500,
        block_size: int = 20,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """Block-bootstrap confidence intervals for eigenvalue magnitudes and angles.

        Non-overlapping blocks of length ``block_size`` are resampled with
        replacement.  For each replicate the Koopman matrix is re-estimated
        and its eigenvalues extracted.  Percentile confidence intervals are
        then reported for both ``|lambda_k|`` and ``arg(lambda_k)``.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model with ``transform``.
        data : torch.Tensor
            Trajectory ``(T, d)``.
        tau : float
            Lag time.
        n_bootstrap : int
            Number of bootstrap replicates.
        block_size : int
            Block length for non-overlapping block bootstrap.
        confidence_level : float
            Confidence level (default 0.95).

        Returns
        -------
        dict
            ``magnitude_ci`` -- ``(K, 2)`` array with (lower, upper) per mode.
            ``angle_ci``     -- ``(K, 2)`` array with (lower, upper) per mode.
            ``magnitude_samples`` -- ``(n_bootstrap, K)`` bootstrap samples.
            ``angle_samples``     -- ``(n_bootstrap, K)`` bootstrap samples.
        """
        from .chapman_kolmogorov import model_koopman_matrix

        tau_int = int(tau)

        # Get reference eigenvalues to establish mode ordering
        K_ref = model_koopman_matrix(model, data, tau_int)
        ref_evals = _safe_eigvals(K_ref)
        ref_order = np.argsort(-np.abs(ref_evals))
        ref_evals = ref_evals[ref_order]
        n_modes = len(ref_evals)

        # Transform full trajectory once
        model.eval()
        with torch.no_grad():
            features = model.transform(data).cpu().numpy()

        T = features.shape[0]
        n_blocks = T // block_size
        if n_blocks < 2:
            block_size = max(1, T // 4)
            n_blocks = T // block_size

        rng = np.random.default_rng(seed=0)
        mag_samples = np.empty((n_bootstrap, n_modes))
        ang_samples = np.empty((n_bootstrap, n_modes))

        for b in range(n_bootstrap):
            chosen = rng.integers(0, n_blocks, size=n_blocks)
            indices = np.concatenate(
                [np.arange(i * block_size, (i + 1) * block_size) for i in chosen]
            )
            surrogate = features[indices]

            K_b = _koopman_from_features(surrogate, tau_int)
            evals_b = _safe_eigvals(K_b)

            # Match modes to reference by nearest-neighbour in complex plane
            evals_b_sorted = _match_eigenvalues(ref_evals, evals_b)

            mag_samples[b] = np.abs(evals_b_sorted)
            ang_samples[b] = np.angle(evals_b_sorted)

        alpha = 1.0 - confidence_level
        mag_ci = np.column_stack(
            [
                np.percentile(mag_samples, 100 * alpha / 2, axis=0),
                np.percentile(mag_samples, 100 * (1 - alpha / 2), axis=0),
            ]
        )
        ang_ci = np.column_stack(
            [
                np.percentile(ang_samples, 100 * alpha / 2, axis=0),
                np.percentile(ang_samples, 100 * (1 - alpha / 2), axis=0),
            ]
        )

        logger.info(
            "Bootstrap eigenvalue CI (%d replicates): leading |lambda| in [%.4f, %.4f]",
            n_bootstrap,
            mag_ci[0, 0],
            mag_ci[0, 1],
        )

        return {
            "magnitude_ci": mag_ci,
            "angle_ci": ang_ci,
            "magnitude_samples": mag_samples,
            "angle_samples": ang_samples,
        }

    # ------------------------------------------------------------------
    # 10.3  Permutation test for irreversibility
    # ------------------------------------------------------------------

    @staticmethod
    def permutation_test_irreversibility(
        data: torch.Tensor,
        tau: float,
        model_factory: Callable[[], torch.nn.Module],
        n_permutations: int = 1000,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Permutation test for time-reversal asymmetry (PRD Section 10.3).

        The null hypothesis is that the process is time-reversible.  Under
        this null, random permutations of the time series destroy any
        temporal asymmetry.  We compare the entropy production (or VAMP-E
        irreversibility proxy) of the original data against the null
        distribution obtained from permuted surrogates.

        Parameters
        ----------
        data : torch.Tensor
            Trajectory ``(T, d)``.
        tau : float
            Lag time.
        model_factory : callable
            Zero-argument callable returning a fresh, untrained model.
        n_permutations : int
            Number of random permutations.
        n_epochs : int
            Training epochs for each surrogate model.
        learning_rate : float
            Optimizer learning rate.

        Returns
        -------
        dict
            ``observed``  -- irreversibility statistic on original data.
            ``null_dist``  -- array of null statistics.
            ``p_value``    -- one-sided p-value.
            ``z_score``    -- standard z-score of observed vs null.
        """
        tau_int = int(tau)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Observed statistic ----
        observed = _train_and_compute_irreversibility(
            data, tau_int, model_factory, n_epochs, learning_rate, device
        )

        # ---- Null distribution via time-index permutation ----
        rng = np.random.default_rng(seed=42)
        null_stats = np.empty(n_permutations)

        for p in range(n_permutations):
            perm_idx = rng.permutation(data.shape[0])
            data_perm = data[perm_idx]
            null_stats[p] = _train_and_compute_irreversibility(
                data_perm, tau_int, model_factory, n_epochs, learning_rate, device
            )

        p_value = float(np.mean(null_stats >= observed))
        null_mean = float(np.mean(null_stats))
        null_std = float(np.std(null_stats))
        z_score = (observed - null_mean) / max(null_std, 1e-15)

        logger.info(
            "Permutation test: observed=%.4f, null_mean=%.4f +/- %.4f, "
            "p=%.4f, z=%.2f",
            observed,
            null_mean,
            null_std,
            p_value,
            z_score,
        )

        return {
            "observed": observed,
            "null_dist": null_stats,
            "p_value": p_value,
            "z_score": z_score,
        }

    # ------------------------------------------------------------------
    # Granger causality
    # ------------------------------------------------------------------

    @staticmethod
    def granger_causality(
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = 10,
    ) -> Dict[str, Any]:
        """Granger causality test: does *x* Granger-cause *y*?

        Uses ``statsmodels.tsa.stattools.grangercausalitytests``.

        Parameters
        ----------
        x : np.ndarray
            Potential causal series, shape ``(T,)``.
        y : np.ndarray
            Potential effect series, shape ``(T,)``.
        max_lag : int
            Maximum lag order to test.

        Returns
        -------
        dict
            ``f_stats``   -- dict mapping lag -> F-statistic.
            ``p_values``  -- dict mapping lag -> p-value.
            ``best_lag``  -- lag with smallest p-value.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        # statsmodels expects a 2-column array [effect, cause]
        joint = np.column_stack([y, x])
        gc_results = grangercausalitytests(joint, maxlag=max_lag, verbose=False)

        f_stats: Dict[int, float] = {}
        p_values: Dict[int, float] = {}

        for lag in range(1, max_lag + 1):
            # gc_results[lag] is a tuple: (test_dict, ols_results)
            test_dict = gc_results[lag][0]
            # Use the standard F-test ('ssr_ftest')
            f_stat, p_val, _, _ = test_dict["ssr_ftest"]
            f_stats[lag] = float(f_stat)
            p_values[lag] = float(p_val)

        best_lag = min(p_values, key=p_values.get)  # type: ignore[arg-type]

        logger.info(
            "Granger causality: best_lag=%d, F=%.3f, p=%.4f",
            best_lag,
            f_stats[best_lag],
            p_values[best_lag],
        )

        return {
            "f_stats": f_stats,
            "p_values": p_values,
            "best_lag": best_lag,
        }

    # ------------------------------------------------------------------
    # Ljung--Box residual test
    # ------------------------------------------------------------------

    @staticmethod
    def ljung_box_residuals(
        residuals: np.ndarray,
        n_lags: int = 20,
    ) -> Dict[str, Any]:
        """Ljung--Box test for residual autocorrelation.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals, shape ``(T,)`` or ``(T, d)``.
        n_lags : int
            Number of lags for the test.

        Returns
        -------
        dict
            ``q_stat``  -- Ljung--Box Q statistic (or array for multivariate).
            ``p_value`` -- associated p-value(s).
            ``reject_5pct`` -- boolean, True if any p-value < 0.05.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        q_stats = []
        p_values = []

        for col_idx in range(residuals.shape[1]):
            lb_result = acorr_ljungbox(residuals[:, col_idx], lags=n_lags, return_df=True)
            # Take the last lag row as the summary statistic
            q_stats.append(float(lb_result["lb_stat"].iloc[-1]))
            p_values.append(float(lb_result["lb_pvalue"].iloc[-1]))

        q_stat = np.array(q_stats)
        p_value = np.array(p_values)
        reject_5pct = bool(np.any(p_value < 0.05))

        logger.info(
            "Ljung-Box (n_lags=%d): min_p=%.4f, reject_H0=%s",
            n_lags,
            float(np.min(p_value)),
            reject_5pct,
        )

        return {
            "q_stat": q_stat.squeeze(),
            "p_value": p_value.squeeze(),
            "reject_5pct": reject_5pct,
        }

    # ------------------------------------------------------------------
    # KS test for eigenfunction stability
    # ------------------------------------------------------------------

    @staticmethod
    def ks_test_eigenfunctions(
        train_eigfuncs: np.ndarray,
        test_eigfuncs: np.ndarray,
    ) -> Dict[str, Any]:
        """Two-sample Kolmogorov--Smirnov test per eigenfunction mode.

        Tests whether the marginal distribution of each eigenfunction is
        the same on training and test data.  Significant deviation signals
        distribution shift or overfitting.

        Parameters
        ----------
        train_eigfuncs : np.ndarray
            Training eigenfunction values, shape ``(N_train, K)``.
        test_eigfuncs : np.ndarray
            Test eigenfunction values, shape ``(N_test, K)``.

        Returns
        -------
        dict
            ``d_stats``   -- KS D_n statistic per mode, shape ``(K,)``.
            ``p_values``  -- p-value per mode, shape ``(K,)``.
            ``reject_5pct`` -- boolean array, True where p < 0.05.
        """
        n_modes = train_eigfuncs.shape[1]
        d_stats = np.empty(n_modes)
        p_values = np.empty(n_modes)

        for k in range(n_modes):
            ks_result = sp_stats.ks_2samp(train_eigfuncs[:, k], test_eigfuncs[:, k])
            d_stats[k] = ks_result.statistic
            p_values[k] = ks_result.pvalue

        reject_5pct = p_values < 0.05

        logger.info(
            "KS test eigenfunctions: %d/%d modes reject H0 at 5%%",
            int(reject_5pct.sum()),
            n_modes,
        )

        return {
            "d_stats": d_stats,
            "p_values": p_values,
            "reject_5pct": reject_5pct,
        }

    # ------------------------------------------------------------------
    # 10.4  Time-series cross-validation
    # ------------------------------------------------------------------

    @staticmethod
    def time_series_cv(
        data: torch.Tensor,
        tau: float,
        model_factory: Callable[[], torch.nn.Module],
        n_folds: int = 5,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        """Expanding-window time-series cross-validation (PRD Section 10.4).

        Uses an expanding training window with a fixed-size validation
        segment that always lies strictly in the future of the training
        data, preserving temporal ordering.

        Parameters
        ----------
        data : torch.Tensor
            Full trajectory ``(T, d)``.
        tau : float
            Lag time.
        model_factory : callable
            Zero-argument callable returning a fresh model.
        n_folds : int
            Number of CV folds.
        n_epochs : int
            Training epochs per fold.
        learning_rate : float
            Optimizer learning rate.
        validation_fraction : float
            Fraction of the trajectory reserved for the final validation
            window.  The validation window slides; this parameter sets its
            size relative to the fold segment.

        Returns
        -------
        dict
            ``fold_losses``     -- validation loss per fold.
            ``mean_loss``       -- mean validation loss across folds.
            ``std_loss``        -- standard deviation of validation losses.
            ``eigenvalue_stability`` -- max eigenvalue magnitude deviation
                                       across folds.
        """
        from .chapman_kolmogorov import model_koopman_matrix

        tau_int = int(tau)
        T = data.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Split into n_folds expanding-window segments
        # Each fold k uses data[0 : split_k] for training and
        # data[split_k : split_k + val_size] for validation.
        val_size = max(int(T * validation_fraction / n_folds), 2 * tau_int + 1)
        usable = T - val_size
        fold_boundaries = np.linspace(
            2 * tau_int + val_size, usable, n_folds, dtype=int
        )

        fold_losses: List[float] = []
        fold_eigenvalues: List[np.ndarray] = []

        for fold_idx, train_end in enumerate(fold_boundaries):
            val_start = int(train_end)
            val_end = min(val_start + val_size, T)

            train_data = data[:train_end].to(device)
            val_data = data[val_start:val_end].to(device)

            if val_data.shape[0] <= tau_int:
                logger.warning("Fold %d: validation too short, skipping.", fold_idx)
                continue

            # Train
            model = model_factory().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            x_t_train = train_data[:-tau_int]
            x_t_tau_train = train_data[tau_int:]

            model.train()
            for _epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = model(x_t_train, x_t_tau_train)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                x_t_val = val_data[:-tau_int]
                x_t_tau_val = val_data[tau_int:]
                val_loss = model(x_t_val, x_t_tau_val).item()

            fold_losses.append(val_loss)

            # Eigenvalue snapshot for stability assessment
            K_fold = model_koopman_matrix(model, val_data.cpu(), tau_int)
            evals_fold = _safe_eigvals(K_fold)
            evals_fold = evals_fold[np.argsort(-np.abs(evals_fold))]
            fold_eigenvalues.append(evals_fold)

            logger.info(
                "CV fold %d/%d: train_end=%d, val_loss=%.4f",
                fold_idx + 1,
                n_folds,
                train_end,
                val_loss,
            )

        fold_losses_arr = np.array(fold_losses)
        mean_loss = float(fold_losses_arr.mean())
        std_loss = float(fold_losses_arr.std())

        # Eigenvalue stability: max deviation in leading magnitude across folds
        if len(fold_eigenvalues) >= 2:
            leading_mags = np.array([np.abs(ev[0]) for ev in fold_eigenvalues])
            eigenvalue_stability = float(leading_mags.max() - leading_mags.min())
        else:
            eigenvalue_stability = 0.0

        logger.info(
            "Time-series CV: mean_loss=%.4f +/- %.4f, eigenvalue_stability=%.4f",
            mean_loss,
            std_loss,
            eigenvalue_stability,
        )

        return {
            "fold_losses": fold_losses_arr,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "eigenvalue_stability": eigenvalue_stability,
        }


# ======================================================================
# Module-level helpers
# ======================================================================


def _safe_eigvals(K: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of K, sanitizing NaN/Inf inputs to avoid MKL crashes.

    ``np.linalg.eigvals`` calls MKL's SGEBAL which can segfault on matrices
    containing non-finite values.  This wrapper replaces non-finite entries
    with zeros before the call.
    """
    if not np.isfinite(K).all():
        K = np.where(np.isfinite(K), K, 0.0)
    return np.linalg.eigvals(K)


def _koopman_from_features(features: np.ndarray, lag: int) -> np.ndarray:
    """Estimate Koopman matrix from pre-computed features."""
    chi_t = features[:-lag]
    chi_lag = features[lag:]
    n = chi_t.shape[0]
    mean_t = chi_t.mean(axis=0, keepdims=True)
    mean_lag = chi_lag.mean(axis=0, keepdims=True)
    chi_t_c = chi_t - mean_t
    chi_lag_c = chi_lag - mean_lag
    C_00 = (chi_t_c.T @ chi_t_c) / n
    C_01 = (chi_t_c.T @ chi_lag_c) / n
    reg = 1e-6 * np.eye(C_00.shape[0])
    K = np.linalg.solve(C_00 + reg, C_01)
    # Sanitize: replace NaN/Inf (can occur from degenerate resamples)
    if not np.isfinite(K).all():
        K = np.where(np.isfinite(K), K, 0.0)
    return K


def _match_eigenvalues(
    reference: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Match *candidates* to *reference* eigenvalues by nearest distance.

    Uses a greedy nearest-neighbour assignment in the complex plane to
    establish a consistent mode ordering across bootstrap replicates.

    Parameters
    ----------
    reference : np.ndarray
        Reference complex eigenvalues (sorted).
    candidates : np.ndarray
        Candidate complex eigenvalues (arbitrary order).

    Returns
    -------
    np.ndarray
        Re-ordered candidates matching reference ordering.
    """
    n = len(reference)
    m = len(candidates)
    matched = np.full(n, np.nan + 0j)
    used = set()

    for i in range(n):
        best_dist = np.inf
        best_j = -1
        for j in range(m):
            if j in used:
                continue
            dist = abs(reference[i] - candidates[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0:
            matched[i] = candidates[best_j]
            used.add(best_j)

    return matched


def _train_and_compute_irreversibility(
    data: torch.Tensor,
    tau_int: int,
    model_factory: Callable[[], torch.nn.Module],
    n_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    """Train model and compute an irreversibility proxy.

    The irreversibility is measured as the asymmetry between the forward
    and backward Koopman matrices:  ``||K_fwd - K_bwd^T||_F``.

    Parameters
    ----------
    data : torch.Tensor
        Trajectory.
    tau_int : int
        Integer lag.
    model_factory : callable
        Returns a fresh model.
    n_epochs : int
        Training epochs.
    learning_rate : float
        LR.
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Irreversibility statistic.
    """
    data_dev = data.to(device)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_t = data_dev[:-tau_int]
    x_t_tau = data_dev[tau_int:]

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = model(x_t, x_t_tau)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        features = model.transform(data_dev.cpu()).cpu().numpy()

    # Forward Koopman
    K_fwd = _koopman_from_features(features, tau_int)

    # Backward Koopman (time-reversed trajectory)
    features_rev = features[::-1].copy()
    K_bwd = _koopman_from_features(features_rev, tau_int)

    # Irreversibility = ||K_fwd - K_bwd^T||_F
    irreversibility = float(np.linalg.norm(K_fwd - K_bwd.T, "fro"))
    return irreversibility
