#!/usr/bin/env python3
"""
Last-minute high-impact experiments for PRE submission.

1. Bootstrap AUROC confidence intervals (recession prediction)
2. Univariate Frobenius EP from saved model's Koopman matrix
3. Gyrator Frobenius calibration with tau=1 (fixes 1000x underestimate)
4. Combined spectral+VIX drawdown prediction
"""
import json
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# 1. Bootstrap AUROC CIs for recession prediction
# =====================================================================
def experiment_auroc_bootstrap():
    """
    Bootstrap the out-of-sample AUROC to get 95% CIs.
    Critical: a referee needs to know if 0.78 is significantly > 0.56.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Bootstrap AUROC confidence intervals")
    logger.info("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Load rolling features
    rolling_file = RESULTS_DIR / "spectral_gap_timeseries.csv"
    if not rolling_file.exists():
        logger.warning("  No rolling data")
        return None

    rolling_df = pd.read_csv(rolling_file, parse_dates=["center_date"])

    # Load VIX
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start="1993-01-01", end="2026-03-01",
                          progress=False)
        vix_close = vix["Close"].squeeze()
    except Exception as e:
        logger.warning(f"  Cannot load VIX: {e}")
        return None

    # NBER recession STARTS (same as run_rolling.py)
    from src.constants import NBER_RECESSIONS
    recession_starts = [pd.Timestamp(s) for s, _e in NBER_RECESSIONS]

    # Build features and labels
    feature_cols = ["spectral_gap", "entropy_production",
                    "mean_irreversibility", "vamp2_score"]
    horizon = 60
    dates, features, labels, vix_vals = [], [], [], []

    for _, row in rolling_df.iterrows():
        date = pd.Timestamp(row["center_date"])
        if pd.isna(date):
            continue

        # Label positive if a recession START is within horizon days
        label = 0
        for rs in recession_starts:
            if 0 <= (rs - date).days <= horizon:
                label = 1
                break

        feat = [float(row.get(c, 0.0) or 0.0) for c in feature_cols]
        if all(f == 0.0 for f in feat):
            continue

        # VIX value
        try:
            v = float(vix_close.asof(date))
            if np.isnan(v):
                continue
        except Exception:
            continue

        dates.append(date)
        features.append(feat)
        labels.append(label)
        vix_vals.append(v)

    features = np.array(features)
    labels = np.array(labels)
    vix_vals = np.array(vix_vals)

    logger.info(f"  Windows: {len(labels)}, positive: {labels.sum()}")

    # Expanding-window predictions
    min_train = max(200, len(labels) // 3)
    y_pred_spec, y_pred_vix, y_true = [], [], []

    for i in range(min_train, len(labels)):
        X_tr = features[:i]
        y_tr = labels[:i]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue

        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(features[i:i+1])[0, 1]
        y_pred_spec.append(prob)
        y_pred_vix.append(vix_vals[i])
        y_true.append(labels[i])

    y_pred_spec = np.array(y_pred_spec)
    y_pred_vix = np.array(y_pred_vix)
    y_true = np.array(y_true)

    auroc_spec = roc_auc_score(y_true, y_pred_spec)
    auroc_vix = roc_auc_score(y_true, y_pred_vix)

    logger.info(f"  Point AUROC: spectral={auroc_spec:.4f}, VIX={auroc_vix:.4f}")

    # Bootstrap
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_spec, boot_vix, boot_diff = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        a_s = roc_auc_score(yt, y_pred_spec[idx])
        a_v = roc_auc_score(yt, y_pred_vix[idx])
        boot_spec.append(a_s)
        boot_vix.append(a_v)
        boot_diff.append(a_s - a_v)

    boot_spec = np.array(boot_spec)
    boot_vix = np.array(boot_vix)
    boot_diff = np.array(boot_diff)

    ci_spec = np.percentile(boot_spec, [2.5, 97.5])
    ci_vix = np.percentile(boot_vix, [2.5, 97.5])
    ci_diff = np.percentile(boot_diff, [2.5, 97.5])
    p_value = np.mean(boot_diff <= 0)  # one-sided

    logger.info(f"  Spectral AUROC: {auroc_spec:.3f} [{ci_spec[0]:.3f}, {ci_spec[1]:.3f}]")
    logger.info(f"  VIX AUROC:      {auroc_vix:.3f} [{ci_vix[0]:.3f}, {ci_vix[1]:.3f}]")
    logger.info(f"  Difference:     {auroc_spec - auroc_vix:.3f} [{ci_diff[0]:.3f}, {ci_diff[1]:.3f}]")
    logger.info(f"  P(spectral > VIX): {1 - p_value:.4f}")

    return {
        "auroc_spectral": round(auroc_spec, 4),
        "auroc_spectral_ci": [round(ci_spec[0], 4), round(ci_spec[1], 4)],
        "auroc_vix": round(auroc_vix, 4),
        "auroc_vix_ci": [round(ci_vix[0], 4), round(ci_vix[1], 4)],
        "auroc_diff": round(float(auroc_spec - auroc_vix), 4),
        "auroc_diff_ci": [round(ci_diff[0], 4), round(ci_diff[1], 4)],
        "p_spectral_gt_vix": round(float(1 - p_value), 4),
        "n_bootstrap": n_boot,
        "n_oos": len(y_true),
    }


# =====================================================================
# 2. Univariate Frobenius EP from saved model
# =====================================================================
def experiment_univariate_frobenius():
    """
    Load saved univariate model and compute Frobenius EP from
    the actual Koopman matrix (not eigenvalues).
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: Univariate Frobenius EP from saved model")
    logger.info("=" * 60)

    import torch
    from src.model.vampnet import NonEquilibriumVAMPNet
    from src.utils.config import load_config, merge_configs
    from src.data.preprocessing import compute_log_returns, standardize_returns

    model_path = PROJECT_ROOT / "outputs" / "models" / "vampnet_univariate.pt"
    if not model_path.exists():
        logger.warning("  No saved univariate model")
        return None

    # Load config
    config = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))

    # Load data for forward pass
    try:
        import yfinance as yf
        spy = yf.download("SPY", start="1993-01-01", end="2026-03-01",
                          progress=False)
        close = spy[["Close"]]
    except Exception as e:
        logger.warning(f"  Cannot load data: {e}")
        return None

    returns = np.log(close / close.shift(1)).dropna().values
    n_train = min(2700, len(returns))
    mean = returns[:n_train].mean(axis=0)
    std = returns[:n_train].std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (returns - mean) / std

    # Delay embedding
    embedding_dim = config.get("data", {}).get("embedding_dim", 5)
    tau = config.get("model", {}).get("tau", 5)
    n_modes = 5

    d = standardized.shape[1]
    input_dim = d * embedding_dim

    embedded = []
    for i in range(embedding_dim - 1, len(standardized)):
        window = standardized[i - embedding_dim + 1:i + 1].flatten()
        embedded.append(window)
    embedded = np.array(embedded)

    X_pairs = embedded[:-tau]
    Y_pairs = embedded[tau:]

    # Load model — univariate uses [64,64,32], not the multiasset default
    hidden_dims = [64, 64, 32]
    model = NonEquilibriumVAMPNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=n_modes,
        dropout=0.1,
        epsilon=1e-3,
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Forward pass to get Koopman matrix
    with torch.no_grad():
        x = torch.tensor(X_pairs[:n_train], dtype=torch.float32)
        y = torch.tensor(Y_pairs[:n_train], dtype=torch.float32)

        # Process in batches
        batch_size = 2048
        K_matrix = None
        for i in range(0, len(x), batch_size):
            out = model(x[i:i+batch_size], y[i:i+batch_size])
            K_matrix = out.get("koopman_matrix")

    if K_matrix is None:
        logger.warning("  No Koopman matrix in output")
        return None

    K_np = K_matrix.detach().cpu().numpy()
    logger.info(f"  Koopman matrix shape: {K_np.shape}")

    # Frobenius EP from actual matrix
    K_A = (K_np - K_np.T) / 2
    K_S = (K_np + K_np.T) / 2
    ka_frob_sq = np.sum(K_A**2)
    ks_frob_sq = np.sum(K_S**2)
    ep_frobenius = 2.0 * ka_frob_sq / tau

    # From eigenvalues for comparison
    eigvals = np.linalg.eigvals(K_np)
    ep_eig = sum(2.0 * abs(l)**2 * np.sin(np.angle(l))**2 / tau
                 for l in eigvals)

    ratio = np.sqrt(ka_frob_sq / ks_frob_sq) if ks_frob_sq > 0 else 0

    logger.info(f"  ||K_A||_F / ||K_S||_F = {ratio:.4f}")
    logger.info(f"  Frobenius EP (matrix):      {ep_frobenius:.4f} nats/day")
    logger.info(f"  Frobenius EP (eigenvalues):  {ep_eig:.4f} nats/day")
    logger.info(f"  k-NN reference:              0.26 nats/day")

    return {
        "ep_frobenius_matrix": round(ep_frobenius, 4),
        "ep_frobenius_eigenvalues": round(ep_eig, 4),
        "ka_ks_ratio": round(ratio, 4),
        "ep_knn": 0.26,
        "matrix_shape": list(K_np.shape),
    }


# =====================================================================
# 3. Gyrator Frobenius with tau=1 (better calibration)
# =====================================================================
def experiment_gyrator_tau1():
    """
    Repeat gyrator Frobenius at tau=1 instead of tau=5.
    Smaller tau means less damping → more asymmetry preserved → better match.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: Gyrator Frobenius at tau=1")
    logger.info("=" * 60)

    from scipy.linalg import sqrtm, inv

    T1, k_spring, kappa = 1.0, 1.0, 0.5
    T2_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    dt, n_steps = 0.01, 100000  # longer for better statistics
    rng = np.random.default_rng(42)

    results_by_tau = {}

    for tau in [1, 2, 5]:
        points = []
        for T2 in T2_values:
            # Analytical EP
            num = kappa**2 * (T1 - T2)**2
            denom = ((k_spring**2 - kappa**2) * T1 * T2 +
                     k_spring * (T1**2 + T2**2) / 2)
            ep_analytical = num / denom

            # Simulate
            A = np.array([[k_spring, kappa], [kappa, k_spring]])
            x = np.zeros(2)
            trajectory = np.zeros((n_steps, 2))
            sqrt_2dt = np.sqrt(2 * dt)

            for i in range(n_steps):
                noise = rng.normal(size=2) * sqrt_2dt * np.sqrt([T1, T2])
                x = x - A @ x * dt + noise
                trajectory[i] = x

            # Time-lagged Koopman
            X = trajectory[:-tau]
            Y = trajectory[tau:]
            n = len(X)

            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)

            eps = 1e-6
            C00 = X_c.T @ X_c / n + eps * np.eye(2)
            C0t = X_c.T @ Y_c / n
            Ctt = Y_c.T @ Y_c / n + eps * np.eye(2)

            C00_inv_sqrt = inv(sqrtm(C00)).real
            Ctt_inv_sqrt = inv(sqrtm(Ctt)).real
            K_matrix = C00_inv_sqrt @ C0t @ Ctt_inv_sqrt

            K_A = (K_matrix - K_matrix.T) / 2
            ka_frob_sq = np.sum(K_A**2)
            ep_frobenius = 2.0 * ka_frob_sq / tau

            points.append({
                "T2": T2,
                "ep_analytical": round(ep_analytical, 6),
                "ep_frobenius": round(ep_frobenius, 6),
            })

        frob_eps = [p["ep_frobenius"] for p in points]
        anal_eps = [p["ep_analytical"] for p in points]
        corr = np.corrcoef(anal_eps, frob_eps)[0, 1] if np.std(frob_eps) > 0 else 0

        # Ratio at T2=3 (the benchmark)
        idx_T2_3 = next(i for i, p in enumerate(points) if p["T2"] == 3.0)
        ratio_T2_3 = (frob_eps[idx_T2_3] / anal_eps[idx_T2_3]
                       if anal_eps[idx_T2_3] > 0 else 0)

        results_by_tau[str(tau)] = {
            "points": points,
            "correlation": round(corr, 4),
            "ratio_at_T2_3": round(ratio_T2_3, 4),
        }

        logger.info(f"\n  tau={tau}: correlation={corr:.4f}, "
                    f"ratio@T2=3: {ratio_T2_3:.4f}")
        for p in points:
            logger.info(f"    T2={p['T2']}: analytical={p['ep_analytical']:.4f}, "
                        f"Frobenius={p['ep_frobenius']:.6f}")

    return results_by_tau


# =====================================================================
# 4. Combined spectral+VIX for drawdowns
# =====================================================================
def experiment_combined_drawdown():
    """
    Test if spectral features + VIX combined beat VIX alone for drawdowns.
    If so, kills the 'VIX beats you' critique.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 4: Combined spectral+VIX drawdown prediction")
    logger.info("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rolling_file = RESULTS_DIR / "spectral_gap_timeseries.csv"
    if not rolling_file.exists():
        logger.warning("  No rolling data")
        return None

    rolling_df = pd.read_csv(rolling_file, parse_dates=["center_date"])

    try:
        import yfinance as yf
        spy = yf.download("SPY", start="1993-01-01", end="2026-03-01",
                          progress=False)
        spy_close = spy["Close"].squeeze()
        vix = yf.download("^VIX", start="1993-01-01", end="2026-03-01",
                          progress=False)
        vix_close = vix["Close"].squeeze()
    except Exception as e:
        logger.warning(f"  Cannot load data: {e}")
        return None

    # Drawdown labels
    rolling_high = spy_close.rolling(252, min_periods=1).max()
    drawdown = (spy_close - rolling_high) / rolling_high
    in_drawdown = (drawdown < -0.15).astype(int)

    feature_cols = ["spectral_gap", "entropy_production",
                    "mean_irreversibility", "vamp2_score"]
    horizon = 60

    dates, feat_spec, feat_vix, feat_combined, labels = [], [], [], [], []

    for _, row in rolling_df.iterrows():
        date = pd.Timestamp(row["center_date"])
        if pd.isna(date):
            continue

        future_start = date + pd.Timedelta(days=1)
        future_end = date + pd.Timedelta(days=horizon)
        try:
            future_dd = in_drawdown.loc[future_start:future_end]
        except Exception:
            continue
        if len(future_dd) == 0:
            continue

        label = int(future_dd.any())

        feat = [float(row.get(c, 0.0) or 0.0) for c in feature_cols]
        if all(f == 0.0 for f in feat):
            continue

        try:
            v = float(vix_close.asof(date))
            if np.isnan(v):
                continue
        except Exception:
            continue

        dates.append(date)
        feat_spec.append(feat)
        feat_vix.append([v])
        feat_combined.append(feat + [v])
        labels.append(label)

    feat_spec = np.array(feat_spec)
    feat_vix = np.array(feat_vix)
    feat_combined = np.array(feat_combined)
    labels = np.array(labels)

    logger.info(f"  Windows: {len(labels)}, drawdown events: {labels.sum()}")

    if labels.sum() < 5:
        logger.warning("  Too few events")
        return None

    # Expanding-window for all three feature sets
    min_train = max(200, len(labels) // 3)
    preds = {"spectral": [], "vix": [], "combined": [], "vix_raw": []}
    y_true = []

    for i in range(min_train, len(labels)):
        y_tr = labels[:i]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue

        # Spectral only
        clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                                 random_state=42)
        clf.fit(feat_spec[:i], y_tr)
        preds["spectral"].append(clf.predict_proba(feat_spec[i:i+1])[0, 1])

        # VIX logistic
        clf2 = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=42)
        clf2.fit(feat_vix[:i], y_tr)
        preds["vix"].append(clf2.predict_proba(feat_vix[i:i+1])[0, 1])

        # Combined
        clf3 = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=42)
        clf3.fit(feat_combined[:i], y_tr)
        preds["combined"].append(clf3.predict_proba(feat_combined[i:i+1])[0, 1])

        # VIX raw score
        preds["vix_raw"].append(feat_vix[i, 0])
        y_true.append(labels[i])

    y_true = np.array(y_true)
    aurocs = {}
    for name, pred in preds.items():
        pred = np.array(pred)
        if len(pred) == len(y_true) and y_true.sum() > 0:
            aurocs[name] = round(roc_auc_score(y_true, pred), 4)

    logger.info(f"\n  Drawdown AUROC (spectral only):   {aurocs.get('spectral', 'N/A')}")
    logger.info(f"  Drawdown AUROC (VIX logistic):    {aurocs.get('vix', 'N/A')}")
    logger.info(f"  Drawdown AUROC (VIX raw score):   {aurocs.get('vix_raw', 'N/A')}")
    logger.info(f"  Drawdown AUROC (spectral + VIX):  {aurocs.get('combined', 'N/A')}")

    improvement = None
    if "combined" in aurocs and "vix_raw" in aurocs:
        improvement = round(aurocs["combined"] - aurocs["vix_raw"], 4)
        logger.info(f"  Combined improvement over VIX raw: {improvement:+.4f}")

    return {
        "aurocs": aurocs,
        "combined_improvement_over_vix": improvement,
        "n_oos": len(y_true),
        "n_drawdown_events": int(y_true.sum()),
    }


# =====================================================================
def main():
    t0 = time.time()
    results = {}

    results["auroc_bootstrap"] = experiment_auroc_bootstrap()
    results["univariate_frobenius"] = experiment_univariate_frobenius()
    results["gyrator_tau_sweep"] = experiment_gyrator_tau1()
    results["combined_drawdown"] = experiment_combined_drawdown()

    out_file = RESULTS_DIR / "last_minute.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All experiments complete in {elapsed:.1f}s")
    logger.info(f"Results saved to {out_file}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
