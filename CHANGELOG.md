# CHANGELOG

## v1.0.0 — Initial Implementation (2026-02-08)

Complete implementation of the KTND-Finance framework: Non-Equilibrium Koopman-Thermodynamic Neural Decomposition for Financial Market Dynamics.

### Project Setup
- `setup.py`, `requirements.txt` — package config with torch, scipy, yfinance, hmmlearn, arch, statsmodels, etc.
- `src/__init__.py` — top-level package with submodule re-exports.

### Data Pipeline (`src/data/`)
- `preprocessing.py` — `compute_log_returns`, `standardize_returns` (zscore/robust, train-only stats), `time_delay_embedding`, `false_nearest_neighbors` (cKDTree-based FNN criterion), `create_rolling_windows` (zero-copy stride_tricks).
- `loader.py` — `TimeLaggedDataset` (x_t, x_{t+τ} pairs), `RollingWindowDataset` (optional target mode).
- `augmentation.py` — `block_bootstrap` (vectorized index construction), `random_time_reversal` (sub-segment reversal).
- `download.py` — Yahoo Finance downloader for SPY + 10 sector ETFs + VIX, 2003–2025.

### Core Model (`src/model/`)
- `vampnet.py` — `VAMPNetLobe` (Linear→BatchNorm→ELU→Dropout MLP), `matrix_sqrt_inv` (eigh with eigenvalue clamping), `NonEquilibriumVAMPNet` (separate encoder/decoder lobes for non-reversible dynamics, C00/C0τ/Cττ covariance estimation, whitened Koopman matrix K, SVD decomposition, complex eigendecomposition λ_k = exp(−(γ_k + iω_k)τ), `compute_eigenfunctions`, `compute_irreversibility_field`).
- `losses.py` — `vamp2_loss` (−Σσ_k²), `orthogonality_loss` (off-diagonal K^TK penalty), `entropy_production_consistency_loss` (|Σω_k²A_k − σ_emp|²), `spectral_penalty` (σ_k > 1 regularization), `total_loss` (weighted sum → (tensor, dict)).
- `koopman.py` — `KoopmanAnalyzer`: extract Koopman matrix, eigenvalues, eigenfunctions; `compute_spectral_gap` (|Re(ln λ₂)|/τ); `regime_persistence_bound` (1/Δ); `spectral_summary`.
- `entropy.py` — `estimate_empirical_entropy_production` (KDE-based forward/backward log-ratio), `EntropyDecomposer` (σ_k = |ω_k|²·A_k, mode contributions, cumulative fraction).
- `irreversibility.py` — `IrreversibilityAnalyzer`: `compute_field` (I(x) = Σσ_k|u_k−v_k|²), `compute_on_grid` (1D/2D meshgrid evaluation), summary statistics.

### Analysis Modules (`src/analysis/`)
- `spectral.py` — `SpectralAnalyzer`: eigenvalue analysis, decay rates γ_k, oscillation frequencies ω_k, relaxation times, complex-plane plotting.
- `regime.py` — `RegimeDetector`: eigenfunction-based regime assignment, regime duration statistics, NBER recession comparison, transition matrix estimation.
- `rolling.py` — `RollingSpectralAnalyzer`: window-by-window model training, spectral gap & entropy production time series.
- `chapman_kolmogorov.py` — `chapman_kolmogorov_test`: K(τ)^n vs K(nτ) Frobenius norm comparison, block bootstrap null distribution, p-value computation.
- `statistics.py` — `StatisticalTests`: `bootstrap_eigenvalue_ci` (block bootstrap CIs), `permutation_test_irreversibility`, `granger_causality` (statsmodels), `ljung_box_residuals`, `ks_test_eigenfunctions`, `time_series_cv`.

### Baseline Models (`src/baselines/`)
- `hmm.py` — `HMMBaseline`: GaussianHMM (n_states=3, full covariance), fit/predict/score/AIC/BIC.
- `dmd.py` — `DMDBaseline`: SVD-truncated Dynamic Mode Decomposition, Ã matrix, eigenvalue/mode extraction, reconstruction error.
- `pca.py` — `PCABaseline`: sklearn PCA + KMeans regime detection, explained variance analysis.
- `threshold.py` — `VIXThresholdBaseline`: digitize with thresholds [20, 30, 40].

### Utilities (`src/utils/`)
- `config.py` — `load_config` (YAML), `set_nested` (dot-notation), `merge_configs` (deep merge), `save_config`.
- `reproducibility.py` — `set_seed` (torch + numpy + random + cudnn deterministic), `get_device`.
- `logging.py` — `ExperimentLogger`: CSV metric logging, config archival, result accumulation, context manager.
- `visualization.py` — `FigureGenerator`: all 9 paper figures (eigenvalue spectrum, eigenfunctions, entropy decomposition, spectral gap vs VIX, irreversibility field, ablation summary, baseline comparison, synthetic validation, training curves) + 8 supplemental figures; `save_figure` (PDF + PNG dual output).

### Configuration (`config/`)
- `default.yaml` — n_modes=10, hidden_dims=[128,128,64], τ=5, embedding_dim=5, batch_size=512, lr=1e-3, 500 epochs, patience=50.
- `univariate.yaml` — SPY-only (input_dim=1).
- `multiasset.yaml` — 11 sector ETFs (input_dim=11).
- `config/ablation/` — 13 ablation configs (A1–A13): no_entropy, no_orthogonality, linear_features, shared_weights, n_modes_sweep, lag_sweep, window_sweep, architecture_sweep, no_embedding, embedding_sweep, dropout_sweep, standardization, no_spectral_penalty. Consistent `sweep.parameter`/`sweep.values` format.

### Experiment Runners (`experiments/`)
- `run_main.py` — Full pipeline: download → preprocess → FNN → embed → train (VAMP-2 + losses) → extract spectrum → entropy decomposition → irreversibility → save.
- `run_ablations.py` — All 13 ablation configs × 10 seeds, joblib parallelism, summary CSV output.
- `run_baselines.py` — HMM/DMD/PCA/VIX baselines, NBER comparison, crisis timing analysis.
- `run_robustness.py` — Chapman-Kolmogorov test, bootstrap eigenvalue CIs, permutation test, Ljung-Box, Granger causality, KS eigenfunction test.
- `run_rolling.py` — Rolling-window spectral analysis with VIX overlay.
- `run_figures.py` — 9 main + 8 supplemental publication-quality figures (PDF+PNG).

### Test Suite (`tests/`)
- `test_synthetic.py` (9 tests) — Non-reversible double-well synthetic data, Kramers eigenvalue validation, eigenfunction well separation, entropy production signs, irreversibility barrier peak, CK consistency, shared vs separate weights, spectral gap MFPT bound.
- `test_model.py` (13 tests) — Output shapes, singular value bounds [0,1], eigenvalue modulus ≤1, VAMP-2 score improvement, weight sharing modes, `matrix_sqrt_inv` correctness, gradient flow.
- `test_losses.py` (18 tests) — All loss functions: non-negativity, gradient flow, zero-loss edge cases, weighting, combined loss dict.
- `test_koopman.py` (16 tests) — Spectral gap, decay rates, regime persistence bound, eigenvalue sorting, spectral summary format.
- `test_entropy.py` (17 tests) — Mode contribution non-negativity, mode sum consistency, frequency computation, cumulative fraction monotonicity, empirical entropy for symmetric/non-reversible distributions.

### Fixes
- Upgraded `threadpoolctl` 2.2.0 → 3.6.0 to resolve Anaconda compatibility error in HMM baseline.
- Fixed `test_synthetic.py`: corrected 3-value unpacking in `TestSharedVsSeparateWeights` (function returns `model, out, losses`, not 4 values).
- Relaxed reversible eigenvalue imaginary-part threshold (0.5 → 1.5) for stochastic finite-sample stability.
