#!/usr/bin/env python3
"""
Targeted experiments to address PRE reviewer risk factors.

1. Non-perturbative Frobenius EP — no expansion needed
2. Gyrator Frobenius validation — analytical benchmark
3. Multiasset bootstrap at 5 modes — proper UQ
4. Drawdown-based prediction — larger N than NBER
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# Ensure src is importable
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# 1. Non-perturbative Frobenius EP from saved Koopman eigenvalues
# =====================================================================
def experiment_frobenius_ep():
    """
    Compute the Frobenius EP estimate: 2*||K_A||_F^2 / tau.
    This requires NO perturbative expansion.
    Uses eigenvalues saved in analysis_results_{mode}.json.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Non-perturbative Frobenius EP")
    logger.info("=" * 60)

    tau = 5.0
    results = {}

    for mode_name in ["univariate", "multiasset"]:
        analysis_file = RESULTS_DIR / f"analysis_results_{mode_name}.json"
        if not analysis_file.exists():
            logger.warning(f"  Skipping {mode_name}: no analysis results")
            continue

        with open(analysis_file) as f:
            analysis = json.load(f)

        # Eigenvalues stored as separate real/imag arrays
        eig_real = analysis.get("eigenvalues_real", [])
        eig_imag = analysis.get("eigenvalues_imag", [])

        if not eig_real:
            logger.warning(f"  Skipping {mode_name}: no eigenvalues")
            continue

        eigenvalues = np.array([
            complex(r, im) for r, im in zip(eig_real, eig_imag)
        ])
        K = len(eigenvalues)

        # Frobenius EP: 2/tau * sum_k |lambda_k|^2 * sin^2(arg(lambda_k))
        # This is ||K_A||_F^2 in the whitened basis
        frobenius_ka_sq = sum(
            abs(lam)**2 * np.sin(np.angle(lam))**2
            for lam in eigenvalues
        )
        ep_frobenius = 2.0 * frobenius_ka_sq / tau

        # Also compute perturbative per-mode total for comparison
        ep_perturbative = 0.0
        ep_sin2_corrected = 0.0
        for lam in eigenvalues:
            mag = abs(lam)
            omega_tau = np.angle(lam)
            omega = omega_tau / tau
            gamma = -np.log(mag) / tau if mag > 1e-10 else 1.0

            # Perturbative: omega^2 * A_k / gamma
            if abs(gamma) > 1e-10:
                sk_pert = omega**2 * mag / gamma
                # Sin²-corrected
                sk_sin2 = np.sin(omega_tau)**2 / tau**2 * mag / gamma
            else:
                sk_pert = 0.0
                sk_sin2 = 0.0

            ep_perturbative += sk_pert
            ep_sin2_corrected += sk_sin2

        # k-NN reference from analysis results
        knn_ref = analysis.get("entropy_knn", None)

        # DB violation ratio from analysis
        ka_ks_ratio = analysis.get("detailed_balance_ratio", None)

        results[mode_name] = {
            "n_modes": K,
            "ep_frobenius": round(ep_frobenius, 4),
            "ep_perturbative": round(ep_perturbative, 4),
            "ep_sin2_corrected": round(ep_sin2_corrected, 4),
            "ep_knn": round(knn_ref, 4) if knn_ref else None,
            "ka_ks_ratio": round(ka_ks_ratio, 4) if ka_ks_ratio else None,
        }

        logger.info(f"\n  {mode_name} ({K} modes):")
        logger.info(f"    Frobenius EP (non-pert.):  {ep_frobenius:.4f} nats/day")
        logger.info(f"    Sin²-corrected total:      {ep_sin2_corrected:.4f} nats/day")
        logger.info(f"    Perturbative total:         {ep_perturbative:.4f} nats/day")
        logger.info(f"    k-NN reference:             {knn_ref}")

    return results


# =====================================================================
# 2. Gyrator Frobenius validation
# =====================================================================
def experiment_gyrator_frobenius():
    """
    Run the Brownian gyrator at multiple T2 values and compare
    Frobenius EP vs analytical EP. Shows Frobenius tracks truth
    monotonically (unlike perturbative which saturates).
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: Gyrator Frobenius validation")
    logger.info("=" * 60)

    from scipy.linalg import sqrtm, inv

    T1, k_spring, kappa = 1.0, 1.0, 0.5
    T2_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    dt, n_steps, tau = 0.01, 50000, 5
    rng = np.random.default_rng(42)

    calibration_points = []

    for T2 in T2_values:
        # Analytical EP
        num = kappa**2 * (T1 - T2)**2
        denom = (k_spring**2 - kappa**2) * T1 * T2 + k_spring * (T1**2 + T2**2) / 2
        ep_analytical = num / denom

        # Simulate gyrator
        A = np.array([[k_spring, kappa], [kappa, k_spring]])
        x = np.zeros(2)
        trajectory = np.zeros((n_steps, 2))
        sqrt_2dt = np.sqrt(2 * dt)

        for i in range(n_steps):
            noise = rng.normal(size=2) * sqrt_2dt * np.sqrt([T1, T2])
            x = x - A @ x * dt + noise
            trajectory[i] = x

        # Build time-lagged pairs and compute Koopman matrix
        X = trajectory[:-tau]
        Y = trajectory[tau:]
        n = len(X)

        # Center
        X_c = X - X.mean(axis=0)
        Y_c = Y - Y.mean(axis=0)

        # Covariance matrices
        eps = 1e-6
        C00 = X_c.T @ X_c / n + eps * np.eye(2)
        C0t = X_c.T @ Y_c / n
        Ctt = Y_c.T @ Y_c / n + eps * np.eye(2)

        # Whitened Koopman matrix
        C00_inv_sqrt = inv(sqrtm(C00)).real
        Ctt_inv_sqrt = inv(sqrtm(Ctt)).real
        K_matrix = C00_inv_sqrt @ C0t @ Ctt_inv_sqrt

        # Frobenius EP: 2 * ||K_A||_F^2 / tau
        K_A = (K_matrix - K_matrix.T) / 2
        ka_frob_sq = np.sum(K_A**2)
        ep_frobenius = 2.0 * ka_frob_sq / tau

        # Perturbative EP: 2/tau * Tr(K_A^T K_S^{-1} K_A)
        K_S = (K_matrix + K_matrix.T) / 2
        try:
            K_S_inv = inv(K_S)
            ep_perturbative = 2.0 / tau * np.trace(K_A.T @ K_S_inv @ K_A)
        except Exception:
            ep_perturbative = float('nan')

        # Eigenvalue-based Frobenius
        eigvals = np.linalg.eigvals(K_matrix)
        ep_eig_frob = sum(
            2.0 * abs(lam)**2 * np.sin(np.angle(lam))**2 / tau
            for lam in eigvals
        )

        point = {
            "T2": T2,
            "ep_analytical": round(ep_analytical, 6),
            "ep_frobenius": round(ep_frobenius, 6),
            "ep_perturbative": round(ep_perturbative, 6),
            "ep_eig_frobenius": round(ep_eig_frob, 6),
        }
        calibration_points.append(point)
        logger.info(f"  T2={T2}: analytical={ep_analytical:.4f}, "
                    f"Frobenius={ep_frobenius:.4f}, pert={ep_perturbative:.4f}")

    # Check monotonicity of Frobenius EP
    frob_eps = [p["ep_frobenius"] for p in calibration_points]
    analytical_eps = [p["ep_analytical"] for p in calibration_points]

    # Correlation
    corr = np.corrcoef(analytical_eps, frob_eps)[0, 1]

    # Monotonicity: wherever analytical increases, does Frobenius also increase?
    mono_checks = []
    for i in range(len(analytical_eps) - 1):
        if analytical_eps[i + 1] > analytical_eps[i]:
            mono_checks.append(frob_eps[i + 1] > frob_eps[i])
    is_monotonic = all(mono_checks) if mono_checks else True

    result = {
        "calibration_points": calibration_points,
        "frobenius_analytical_correlation": round(corr, 4),
        "frobenius_monotonic": is_monotonic,
    }

    logger.info(f"\n  Frobenius-analytical correlation: {corr:.4f}")
    logger.info(f"  Frobenius monotonic with truth: {is_monotonic}")

    return result


# =====================================================================
# 3. Multiasset bootstrap at 5 modes
# =====================================================================
def experiment_multiasset_bootstrap_5modes():
    """
    Run block bootstrap on multiasset with n_modes=5 to get proper CIs.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: Multiasset bootstrap (5 modes)")
    logger.info("=" * 60)

    try:
        import torch
        from src.utils.config import load_config, merge_configs
        from src.model.vampnet import NonEquilibriumVAMPNet
        from src.model.losses import total_loss
        from src.data.preprocessing import (
            compute_log_returns,
            standardize_returns,
            time_delay_embedding,
        )
        import yfinance as yf
    except ImportError as e:
        logger.warning(f"  Cannot import required modules: {e}")
        return None

    # Load config
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))
    mode_config = load_config(str(PROJECT_ROOT / "config" / "multiasset.yaml"))
    config = merge_configs(config, mode_config)

    # Load data
    data_cfg = config.get("data", {})
    tickers = data_cfg.get("tickers", ["SPY"])
    start_date = data_cfg.get("start_date", "2007-04-01")
    end_date = data_cfg.get("end_date", "2026-03-01")

    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if "Close" in prices.columns.get_level_values(0) if isinstance(prices.columns, pd.MultiIndex) else "Close" in prices.columns:
            if isinstance(prices.columns, pd.MultiIndex):
                close = prices["Close"]
            else:
                close = prices[["Close"]]
                close.columns = tickers
        else:
            close = prices
    except Exception as e:
        logger.warning(f"  Data loading failed: {e}")
        return None

    # Compute log returns
    returns = np.log(close / close.shift(1)).dropna()

    # Remove NaN rows
    returns = returns.dropna()
    returns_arr = returns.values

    if len(returns_arr) < 500:
        logger.warning(f"  Insufficient data: {len(returns_arr)}")
        return None

    logger.info(f"  Loaded {len(returns_arr)} returns for {len(tickers)} tickers")

    # Standardize using training set stats
    n_train = min(2700, len(returns_arr))
    mean = returns_arr[:n_train].mean(axis=0)
    std = returns_arr[:n_train].std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (returns_arr - mean) / std

    # Delay embedding
    embedding_dim = data_cfg.get("embedding_dim", 2)  # multiasset uses 2
    tau = config.get("model", {}).get("tau", 5)
    n_modes = 5  # KEY: cap at 5 modes

    d = standardized.shape[1]
    input_dim = d * embedding_dim
    logger.info(f"  Embedding: {d} assets × {embedding_dim} lags = {input_dim} dims")

    # Build embedded data
    embedded = []
    for i in range(embedding_dim - 1, len(standardized)):
        window = standardized[i - embedding_dim + 1:i + 1].flatten()
        embedded.append(window)
    embedded = np.array(embedded)

    if len(embedded) < 200:
        logger.warning(f"  Insufficient embedded data: {len(embedded)}")
        return None

    # Build time-lagged pairs
    X_pairs = embedded[:-tau]
    Y_pairs = embedded[tau:]

    # Train a quick model with 5 modes
    model_cfg = config.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [128, 128, 64])
    device = "cpu"

    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=n_modes,
        dropout=0.1,
        epsilon=1e-3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    n_train_pairs = min(2700, len(X_pairs))
    X_train = torch.tensor(X_pairs[:n_train_pairs], dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_pairs[:n_train_pairs], dtype=torch.float32, device=device)

    logger.info(f"  Training 5-mode model (input_dim={input_dim}, "
                f"n_samples={n_train_pairs})...")

    model.train()
    batch_size = 512
    for epoch in range(200):
        perm = torch.randperm(n_train_pairs)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train_pairs - batch_size, batch_size):
            idx = perm[i:i + batch_size]
            x_b = X_train[idx]
            y_b = Y_train[idx]
            output = model(x_b, y_b)
            loss, _ = total_loss(output, tau=float(tau))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 50 == 0:
            logger.info(f"    Epoch {epoch+1}/200, loss={epoch_loss/max(n_batches,1):.4f}")

    model.eval()

    # Block bootstrap
    logger.info("  Running block bootstrap (200 replicates, block_size=20)...")
    n_bootstrap = 200
    block_size = 20
    n_samples = len(X_pairs)

    boot_eigenvalues = []
    with torch.no_grad():
        for b in range(n_bootstrap):
            # Block bootstrap indices
            n_blocks = n_samples // block_size + 1
            blocks = np.random.randint(0, n_samples - block_size, size=n_blocks)
            indices = np.concatenate([
                np.arange(start, min(start + block_size, n_samples))
                for start in blocks
            ])[:n_samples]

            x_boot = torch.tensor(X_pairs[indices], dtype=torch.float32, device=device)
            y_boot = torch.tensor(Y_pairs[indices], dtype=torch.float32, device=device)

            # Process in batches to avoid memory issues
            all_outputs = []
            bs = 2048
            for j in range(0, len(x_boot), bs):
                out = model(x_boot[j:j+bs], y_boot[j:j+bs])
                all_outputs.append(out)

            # Use the last batch's Koopman matrix (it's a global parameter)
            K_matrix = all_outputs[-1].get("koopman_matrix")
            if K_matrix is None:
                continue

            K_np = K_matrix.detach().cpu().numpy()
            eigvals = np.linalg.eigvals(K_np)
            mags = np.sort(np.abs(eigvals))[::-1]
            boot_eigenvalues.append(mags.tolist())

    boot_eigenvalues = np.array(boot_eigenvalues)
    logger.info(f"  Successful bootstrap replicates: {len(boot_eigenvalues)}")

    if len(boot_eigenvalues) < 50:
        logger.warning("  Too few bootstrap replicates")
        return None

    # Compute CIs
    ci_level = 0.95
    alpha = (1 - ci_level) / 2
    modes_ci = []
    for k in range(min(n_modes, boot_eigenvalues.shape[1])):
        vals = boot_eigenvalues[:, k]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 10:
            continue
        ci_lower = np.percentile(vals, 100 * alpha)
        ci_upper = np.percentile(vals, 100 * (1 - alpha))
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ci_width = ci_upper - ci_lower
        modes_ci.append({
            "mode": k,
            "mean": round(float(mean_val), 4),
            "std": round(float(std_val), 4),
            "ci_lower": round(float(ci_lower), 4),
            "ci_upper": round(float(ci_upper), 4),
            "ci_width": round(float(ci_width), 4),
        })
        logger.info(f"    Mode {k}: {mean_val:.4f} ± {std_val:.4f} "
                    f"[{ci_lower:.4f}, {ci_upper:.4f}]")

    # Check if CIs are non-degenerate
    non_degenerate = all(m["ci_width"] > 0.01 for m in modes_ci)
    logger.info(f"  Non-degenerate CIs: {non_degenerate}")

    return {
        "n_modes": n_modes,
        "n_bootstrap": n_bootstrap,
        "n_successful": len(boot_eigenvalues),
        "modes_ci": modes_ci,
        "non_degenerate": non_degenerate,
    }


# =====================================================================
# 4. Drawdown-based prediction (larger N)
# =====================================================================
def experiment_drawdown_prediction():
    """
    Predict market drawdowns > 15% instead of just NBER recessions.
    More events = better statistical power.
    Uses rolling spectral_gap_timeseries.csv from run_rolling.py.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 4: Drawdown-based crisis prediction")
    logger.info("=" * 60)

    # Load rolling spectral features
    rolling_file = RESULTS_DIR / "spectral_gap_timeseries.csv"
    if not rolling_file.exists():
        logger.warning("  No spectral_gap_timeseries.csv found")
        return None

    rolling_df = pd.read_csv(rolling_file, parse_dates=["center_date"])
    logger.info(f"  Loaded {len(rolling_df)} rolling windows")

    # Load SPY prices for drawdown calculation
    try:
        import yfinance as yf
        spy = yf.download("SPY", start="1993-01-01", end="2026-03-01",
                          progress=False)
        spy_close = spy["Close"].squeeze()
    except Exception as e:
        logger.warning(f"  Cannot load SPY data: {e}")
        return None

    # Compute rolling max drawdown
    # A point is "in drawdown > 15%" if current price is > 15% below
    # the rolling 252-day high
    rolling_high = spy_close.rolling(252, min_periods=1).max()
    drawdown = (spy_close - rolling_high) / rolling_high
    in_drawdown = (drawdown < -0.15).astype(int)

    # Build features from rolling windows
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    dates, features, labels = [], [], []
    horizon = 60  # 60-day forward prediction

    feature_cols = ["spectral_gap", "entropy_production", "mean_irreversibility", "vamp2_score"]

    for _, row in rolling_df.iterrows():
        date = row.get("center_date")
        if pd.isna(date):
            continue

        date = pd.Timestamp(date)

        # Check if drawdown occurs within horizon
        future_start = date + pd.Timedelta(days=1)
        future_end = date + pd.Timedelta(days=horizon)
        try:
            future_dd = in_drawdown.loc[future_start:future_end]
        except Exception:
            continue
        if len(future_dd) == 0:
            continue

        label = int(future_dd.any())

        # Features
        feat = []
        for col in feature_cols:
            v = row.get(col, 0.0)
            if pd.notna(v):
                feat.append(float(v))
            else:
                feat.append(0.0)

        if all(f == 0.0 for f in feat):
            continue

        dates.append(date)
        features.append(feat)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    logger.info(f"  Total windows with labels: {len(labels)}")
    logger.info(f"  Drawdown events (>15%): {labels.sum()} ({labels.mean():.1%})")

    if labels.sum() < 5 or len(labels) < 100:
        logger.warning("  Too few events for meaningful prediction")
        return None

    # Expanding-window prediction
    min_train = max(200, len(labels) // 3)
    y_pred, y_true = [], []

    for i in range(min_train, len(labels)):
        X_train = features[:i]
        y_train = labels[:i]

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(features[i:i+1])[0, 1]
        y_pred.append(prob)
        y_true.append(labels[i])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if len(y_true) < 50 or y_true.sum() == 0:
        logger.warning("  Insufficient OOS predictions")
        return None

    auroc = roc_auc_score(y_true, y_pred)

    # VIX baseline
    auroc_vix = None
    try:
        vix = yf.download("^VIX", start="1993-01-01", end="2026-03-01",
                          progress=False)
        vix_close = vix["Close"].squeeze()

        vix_preds = []
        vix_labels = []
        for i in range(min_train, len(dates)):
            date = dates[i]
            try:
                vix_val = vix_close.asof(date)
                if np.isnan(vix_val):
                    continue
                vix_preds.append(float(vix_val))
                vix_labels.append(labels[i])
            except Exception:
                continue

        if len(vix_preds) > 50 and sum(vix_labels) > 0:
            auroc_vix = roc_auc_score(vix_labels, vix_preds)
    except Exception:
        pass

    logger.info(f"\n  AUROC (spectral, drawdown >15%): {auroc:.4f}")
    if auroc_vix is not None:
        logger.info(f"  AUROC (VIX baseline):             {auroc_vix:.4f}")
    else:
        logger.info("  AUROC (VIX baseline):             N/A")
    logger.info(f"  OOS predictions: {len(y_true)}")
    logger.info(f"  OOS positive rate: {y_true.mean():.1%}")
    logger.info(f"  N drawdown events (OOS): {int(y_true.sum())}")

    return {
        "auroc_spectral": round(auroc, 4),
        "auroc_vix": round(auroc_vix, 4) if auroc_vix else None,
        "n_oos": len(y_true),
        "n_drawdown_events": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
        "horizon_days": horizon,
        "drawdown_threshold": -0.15,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    t0 = time.time()
    all_results = {}

    # 1. Frobenius EP
    all_results["frobenius_ep"] = experiment_frobenius_ep()

    # 2. Gyrator validation
    all_results["gyrator_frobenius"] = experiment_gyrator_frobenius()

    # 3. Multiasset bootstrap
    all_results["multiasset_bootstrap_5modes"] = experiment_multiasset_bootstrap_5modes()

    # 4. Drawdown prediction
    all_results["drawdown_prediction"] = experiment_drawdown_prediction()

    # Save
    out_file = RESULTS_DIR / "acceptance_fixes.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All experiments complete in {elapsed:.1f}s")
    logger.info(f"Results saved to {out_file}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
