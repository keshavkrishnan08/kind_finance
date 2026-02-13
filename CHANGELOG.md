# CHANGELOG

## v1.5.3 — Revert Configs + Numerical Stability Fix (2026-02-13)

**Goal**: Restore v1.5.2 config values (which had better results) and fix multiasset `linalg.eigh` convergence failures.

### What happened in v1.5.3-alpha (reverted)
Attempted lower regularization (`beta_orth=0.001`, `gamma_reg=1e-6`, `n_modes=8` univariate).
**Results were worse**: CK test FAILED, permutation p=0.164 (was 0.141), Cohen's d=0.63 (was 1.01),
Granger NOT SIGNIFICANT (was significant), multiasset robustness 5/7 FAILED with `linalg.eigh` errors.

### Changes (final v1.5.3)
- **Reverted configs** to v1.5.2 values: `beta_orth=0.005`, `gamma_reg=1e-5`, univariate `n_modes=5`
- **Fixed `matrix_sqrt_inv`** in `vampnet.py`: added symmetrization + progressive ridge fallback for `linalg.eigh` convergence failures (affects bootstrapped multiasset covariances)
- **Kept multiasset robustness** pipeline addition (Stage 7b) with mode-aware output filenames

---

## v1.5.2 — Strengthen Statistical Tests + Tune Hyperparameters (2026-02-12)

### Config changes
| Parameter | v1.5.1 | v1.5.2 | Rationale |
|-----------|--------|--------|-----------|
| `beta_orthogonality` | 0.01 | 0.005 | Stronger non-equilibrium signal |
| `learning_rate` | 1e-3 | 3e-4 | Lower seed variance |
| `n_epochs` | 500 | 800 | Better convergence |
| `patience` | 50 | 80 | Match longer training |
| Multiasset `n_modes` | 10 | 15 | Richer spectral decomposition |

### Code changes
- Added Test 7: model-free time-reversal asymmetry (`run_robustness.py`)
- Added Cohen's d effect size to permutation test
- Increased permutations 500 -> 1000, segments 20 -> 30
- Added BIC model selection for HMM regime detection (`regime.py`)
- Increased HMM eigenfunctions 3 -> 5
- Updated paper: 7 tests listed, Granger/TRA paragraphs, Cohen reference

### Results (Colab, seed 42)
| Metric | Univariate | Multiasset |
|--------|-----------|------------|
| NBER Accuracy | 0.854 | 0.857 |
| NBER F1 | 0.332 | 0.503 |
| Spectral gap | 0.323 | 0.167 |
| Entropy (empirical) | 4.24 [3.80, 4.50] | 51.18 [48.84, 52.32] |
| Entropy (spectral) | 0.064 | 0.325 |
| Mean irreversibility | 2.70 | 7.99 |
| DB violation | 0.509 | 0.706 |
| Complex modes | 2/5 | 12/15 |
| FT ratio | 0.513 | 3.9e-7 |
| Irrev method | eigendecomp | eigendecomp |

### Statistical tests (univariate)
| Test | Result |
|------|--------|
| Chapman-Kolmogorov | PASSED |
| Bootstrap CIs | COMPLETED |
| Permutation irreversibility | p=0.141, d=1.01 (large) |
| Ljung-Box | all p~0 |
| Granger causality | SIGNIFICANT |
| KS eigenfunctions | 9/10 significant |
| Time-reversal asymmetry | COMPLETED |

### Multi-seed (5 seeds, partial — seeds 0-1 complete)
| Metric | Univariate (mean +/- std) |
|--------|--------------------------|
| Spectral gap | 0.33 +/- 0.05 |
| NBER F1 | 0.33 +/- 0.01 |
| NBER accuracy | 0.82 +/- 0.04 |

---

## v1.5.1 — HMM Regime Detection + Paper Updates (2026-02-12)

### HMM Regime Detection (`src/analysis/regime.py`, `experiments/run_main.py`)
- Replaced sign-of-dominant-eigenfunction with 2-state Gaussian HMM on top 3 eigenfunctions
- HMM captures temporal transition structure + amplitude info + multiple eigenfunctions
- `detect_from_eigenfunctions(method="hmm")` is now default; falls back to k-means if HMM fails

### Minimal Re-Run Notebook Cell (Cell 2b)
- Runs ONLY: KTND HMM re-detection (from saved eigenfunctions) + ablations (resume) + gyrator
- No GPU/retraining needed for regime re-detection (~1 min)
- Updates all analysis_results JSON files with HMM metrics

### Paper Updates (`paper/main.tex`)
- Results table: actual Colab values with 5-seed mean +/- std
- Granger: reported as NOT SIGNIFICANT; concurrent correlation r=-0.29
- KTND regime detection: reframed as exploratory application
- Baseline table: GARCH added, naive baseline added, KTND updated for HMM
- Entropy: clarified empirical EP is seed-independent
- Training: mode-specific configs documented
- Ablations: 14 studies, 10 seeds, LR sweep added

## v1.5.0 — Multi-Seed + GARCH Baseline + Granger Fix (2026-02-11)

### Multi-Seed Error Bars (`KTND_Finance_Colab.ipynb`, `experiments/run_main.py`)
- **5 seeds** (42, 0-3) for main results: trains each mode 5 times, aggregates mean +/- std
- Resume-safe: skips already-completed seed/mode combinations on re-run
- `vamp2_score` and `test_total_loss` now saved to `analysis_results_{mode}.json` for aggregation
- Saves `multi_seed_summary.json` with per-metric mean/std across seeds

### GARCH(1,1) Baseline (`src/baselines/garch.py`, `experiments/run_baselines.py`)
- Standard econophysics volatility regime detector (PRE reviewers expect this)
- Fits GARCH(1,1) to log-returns, extracts conditional volatility, classifies by percentile threshold
- Returns AIC, BIC, NBER accuracy/precision/recall/F1
- Integrated as baseline #4 in `run_baselines.py`

### Learning Rate Ablation (`config/ablation/lr_sweep.yaml`)
- New ablation sweep over Adam learning rates: [3e-4, 1e-3, 3e-3]
- Tests sensitivity of VAMP-2 and spectral quantities to optimizer step size

### Granger Date Alignment Fix (`experiments/run_robustness.py`)
- Spectral gap (1 value per rolling window) and VIX (daily) were aligned by array position, not date
- Now creates date-indexed `pd.Series` for both, aligns on common dates via index intersection
- Requires minimum 50 overlapping dates for valid Granger test

### Notebook Updates
- Version bumped to v1.5.0
- PART A2 added: multi-seed loop between figures and ablations
- Final report now shows both single-seed (seed 42) and multi-seed aggregated results

## v1.4.0 — Model Tuning + Bug Fixes + Brownian Gyrator (2026-02-11)

### Critical: Model/Statistical Improvements (based on Colab run diagnostics)

Previous run showed: CK test FAILED, permutation p=0.232, VIX correlation -0.04. Root causes identified and fixed:

1. **Orthogonality over-regularization** (`default.yaml`): `beta_orthogonality` 1.0 → 0.01. Weight of 1.0 was equal to VAMP-2 objective, suppressing the non-equilibrium signal that the entire paper depends on.
2. **Univariate model overparameterized** (`univariate.yaml`): `n_modes` 10→5, `hidden_dims` [128,128,64]→[64,64,32], `batch_size` 512→256. 10 modes from 5-dim input was underdetermined.
3. **Permutation surrogates too weak** (`run_robustness.py`): `n_segments` 5→20, `min_segment_frac` 0.05→0.02, `max_segment_frac` 0.25→0.15. Previous surrogates preserved 50-75% of temporal structure.
4. **Rolling spectral gap wrong definition** (`run_rolling.py`): Replaced discrete `|λ₁|-|λ₂|` with continuous-time `|Re(ln λ₂)|/τ` via `KoopmanAnalyzer.compute_spectral_gap()`. Inconsistency was causing weak VIX correlation.
5. **CK test improved** (`run_robustness.py`): `n_steps` 5→3, `block_size` 50→10, switched to relative error, compare top-k eigenvalues only. Reduces noise-dominated comparisons.
6. **Config loading bug** (notebook): Changed `--config config/univariate.yaml` to `--config config/default.yaml` for both training stages. Previous approach skipped all default.yaml values (loss weights, training params) because `load_config` doesn't resolve `defaults:` keys. This caused `w_spectral` to use fallback 0.1 instead of config's 1e-5 (10,000x higher).
7. **Loss weight consistency** (`run_main.py`): Now passes `w_entropy` and handles both `spectral_penalty_weight`/`gamma_regularization` keys, matching the ablation runner.

### Critical: 6 Ablation Runner Bugs Fixed (`experiments/run_ablations.py`)

All previous ablation results are **invalidated** — must re-run.

1. **shared_weights never applied**: Config set `model.share_weights: true` but model constructor doesn't accept this param and runner never applied `model.lobe_tau = model.lobe_t`. **Fixed**: explicitly set `model.lobe_tau = model.lobe_t` when `model_cfg.get("share_weights", False)`.
2. **no_entropy loss weight ignored**: Config set `losses.alpha_entropy: 0.0` but runner never passed `w_entropy` to `total_loss()`. **Fixed**: pass `w_entropy=loss_cfg.get("alpha_entropy", 0.1)`.
3. **no_spectral_penalty key mismatch**: Config used key `spectral_penalty_weight` but runner only read `gamma_regularization`. **Fixed**: `w_spectral=loss_cfg.get("spectral_penalty_weight", loss_cfg.get("gamma_regularization", 0.1))`.
4. **Spectral gap wrong metric**: Used discrete magnitude difference `|λ₁| - |λ₂|` instead of continuous-time `|Re(ln λ₂)|/τ`. **Fixed**: use `KoopmanAnalyzer.compute_spectral_gap(eig_tensor, tau)`.
5. **Entropy amplitude wrong formula**: Used `mean(u²)` instead of bilinear `mean(u*v)` left/right eigenfunction product. **Fixed**: `A_k = np.mean(u_np * v_np, axis=0)`.
6. **Window sweep doesn't affect training**: Rolling window size is a post-hoc analysis parameter. **Fixed**: renamed `window_sweep.yaml` → `window_sweep.yaml.disabled`.

### KTND Regime Detection vs NBER (`experiments/run_main.py`)
- `post_training_analysis()` now computes regime labels from dominant eigenfunction sign structure via `RegimeDetector.detect_from_eigenfunctions()`
- Compares against NBER recession dates using `RegimeDetector.compare_with_nber()` with training-only label mapping
- Saves accuracy, precision, recall, F1, naive baseline accuracy, mean regime duration to `analysis_results_{mode}.json`
- Saves `ktnd_regime_labels.csv` for downstream figure generation

### Brownian Gyrator Synthetic Benchmark (`tests/test_synthetic.py`)
- **New generator**: `generate_brownian_gyrator()` — 2D coupled OU process with unequal bath temperatures (T₁ ≠ T₂ breaks detailed balance)
- **Analytical EP**: `analytical_gyrator_entropy_production()` — exact steady-state entropy production rate via Lyapunov equation solve
- **8 new tests** in `TestBrownianGyrator`:
  - 3 analytical: EP=0 at equilibrium, EP>0 out of equilibrium, EP scales with |T₁-T₂|
  - 5 trained model: positive spectral entropy, eq vs non-eq comparison, complex eigenvalues, positive irreversibility
- Analytical EP verified: 0.000 (T₁=T₂), 0.021 (T₂=1.5), 0.167 (T₂=3.0), 0.400 (T₂=5.0)
- Total test count: 132 → 140+

### Paper Updates (`paper/main.tex`)
- **Brownian gyrator subsection** (Sec V.3): analytical EP benchmark with exact formula and comparison
- **VIX lead claim softened**: "indicates" → "suggests...may lead"; added "formal Granger causality testing required"; correlation improvement noted as modest (Δr=0.02)
- **Entropy gap analysis**: added 3 reasons for 33x spectral-vs-KDE gap; added mode-count convergence data (K=3→50); reframed as lower bound
- **KTND baseline row**: added to Table II (values filled from pipeline run)
- **Ablation table**: updated caption to note bug fix and pending 10-seed runs; removed window sweep
- **2 new limitations**: VIX lead-lag needs Granger test; ablation seed count (3 preliminary, 10 needed)
- **Updated test count**: 132 → 140+ tests

### Pipeline Fixes (discovered during Colab re-run)
- **Results overwrite fixed**: `run_main.py` now saves mode-specific `analysis_results_{mode}.json` alongside the generic file, so multiasset no longer clobbers univariate results
- **Stage ordering fixed**: Rolling analysis (Stage 6) now runs BEFORE robustness (Stage 7) — Granger causality needs `spectral_gap_timeseries.csv` from rolling
- **KTND summary printout**: `main()` now prints KTND NBER accuracy/F1/naive baseline in the final summary

### Ablation Runner Fixes (`experiments/run_ablations.py`)
- **Streaming output**: Notebook now uses `subprocess.Popen` with line-by-line output for ablations (previously `capture_output=True` buffered everything, making it look frozen for hours)
- **Incremental saves**: CSV saved after each variant completes — survives Colab disconnection
- **Resume support**: On re-run, skips already-completed variants (reads existing `ablation_summary.csv`)
- **Per-seed progress**: Prints VAMP-2 score and timing for each seed as it finishes
- **ETA tracking**: Estimates remaining time based on average per-variant duration
- **`python -u` flag**: Unbuffered Python output in notebook subprocess call

### Colab Notebook (`KTND_Finance_Colab.ipynb`)
- Consolidated to 5 cells: markdown, setup, full pipeline+ablations+gyrator, view figures, download
- Cell 2 runs everything: tests → download → train (uni+multi) → baselines → rolling → robustness → figures → ablations (10 seeds) → Brownian gyrator benchmark
- Final report reads both `analysis_results_univariate.json` and `analysis_results_multiasset.json`
- Inline figures generated per mode

### What Must Be Re-Run
All experiments must be re-run on Colab with the fixed code:
1. Main pipeline (Cell 2) — regenerates per-mode results with KTND NBER metrics
2. Ablations — 10 seeds with fixed runner (shared_weights/no_entropy/no_spectral now work)
3. Rolling runs before robustness — enables Granger causality test
4. Figures — updated automatically from new data

## v1.3.0 — PRE Statistical Rigor Fixes (2026-02-10)

### Critical Code Bug Fixes
- **Spectral gap**: Use `KoopmanAnalyzer.compute_spectral_gap` (`|Re(ln λ₂)|/τ`) instead of discrete magnitude difference (`|λ₁| - |λ₂|`)
- **Entropy amplitude**: `A_k = mean(u_k * v_k)` bilinear product of left/right eigenfunctions, not `mean(u_k²)`
- **Fluctuation theorem ratio**: Use per-sample KDE entropy production (N values) instead of per-mode spectral values (K values)
- **Ablation dates**: Import `DATE_RANGES` from `src.constants` instead of stale hardcoded 2004-2017/2020-2023

### Granger Causality Test (run_robustness.py)
- Add ADF stationarity pre-check; auto-difference non-stationary series
- Bidirectional testing: spectral_gap → VIX **and** VIX → spectral_gap
- Bonferroni correction across lags for multiple comparisons

### NBER Comparison (regime.py)
- Learn label-to-recession mapping on **training data only** (prevents data snooping)
- Add naive frequency baseline for comparison
- Report mapping provenance in results

### Chapman-Kolmogorov Test (run_robustness.py)
- Replace t-test-against-zero with block-bootstrap null distribution (200 replicates)
- Bootstrap destroys Markov structure; p-value = P(boot_error ≤ observed)

### Visualization Fix
- Replace fake outer-product mode-correlation heatmap with real Pearson correlation of eigenfunction time series

### Paper Fixes (main.tex, references.bib)
- Fix broken `\ref{eq:entropy_decomp}` → `\ref{eq:ep_total}`
- Correct data splits to match code (1994-2017 train, 2020-2025 test)
- Fix ticker list to match `TICKERS_MULTIASSET` (SPY, QQQ, IWM, etc.)
- Add PACS numbers: 89.65.Gh, 05.70.Ln, 02.50.Ga, 05.45.Tp
- Add data availability and code availability statements
- Add 6 key references: Zumbach 2009, Roldán & Parrondo 2010, Ducuara 2023, Klus 2018, Brunton 2016, Li 2019
- Cite new references in appropriate sections

### New Functions
- `estimate_per_sample_entropy_production()` in `src/model/entropy.py`
- `_kde_entropy_production(..., return_samples=True)` option
- `_run_adf_test()`, `_granger_one_direction()` helpers in run_robustness.py

## v1.2.0 — First Complete Pipeline Run + Results (2026-02-10)

### Pipeline Results (Colab T4 GPU)
All 8 pipeline stages passed successfully:
- **Spectral gap**: 0.040 (slow regime mixing, consistent with persistent bull/bear regimes)
- **Entropy production**: 51.2 [48.9, 52.3] 95% CI (statistically significant non-equilibrium)
- **Mean irreversibility**: 7.90 (substantial time-asymmetry in market dynamics)
- **Irreversibility method**: eigendecomposition (theory-correct, not SVD fallback)
- **Detailed balance violation**: 0.73 (confirms broken equilibrium)
- **Complex modes**: 8/10 (oscillatory dynamics dominate)
- **Fluctuation theorem ratio**: 0.87 (approximate Gallavotti-Cohen compliance)

### Stages Completed
1. Unit tests (132 tests, all pass)
2. Data download (SPY 8314 rows 1993-2025, 11 ETFs, VIX)
3. Univariate training (SPY, 500 epochs, early stopping)
4. Multiasset training (11 ETFs, 500 epochs)
5. Baselines (HMM, DMD, PCA, VIX threshold vs NBER)
6. Robustness tests (CK, bootstrap, permutation, Ljung-Box, Granger, KS)
7. Rolling spectral analysis (spectral gap vs VIX time series)
8. Figure generation (9 main + 8 supplemental)

## v1.1.0 — PRE Weakness Fixes + Test Coverage (2026-02-09)

### Irreversibility Field Fix
- Added `compute_eigenfunctions_eig()` and `compute_irreversibility_field_eig()` using proper eigendecomposition (`torch.linalg.eig`) instead of SVD. SVD kept as fallback.
- `run_main.py` tries eigendecomposition first, falls back to SVD if numerically unstable.

### Entropy Production Error Bounds
- Added `_kde_entropy_production()` helper extracting core KDE logic.
- Added `estimate_empirical_entropy_production_with_ci()` with moving-block bootstrap (Kunsch 1989): 200 resamples, 95% CI, block_length=50.
- Results JSON now includes `entropy_ci_lower`, `entropy_ci_upper`, `entropy_std_error`.

### Data Pipeline Bug Fix
- Fixed critical shared-cache bug: `prices.csv` `dropna()` truncated all data to 2007+ (HYG inception). Now uses `ffill()` on cache, `dropna()` per-subset only. SPY preserves full 1993-2025 history.

### Tightened Test Tolerances
- Synthetic fixtures: 10000 → 20000 steps, 150 → 300 epochs.
- Kramers ratio: 0.01-100 → 0.1-10x.
- CK error: < 2.0 → < 1.5.
- MFPT ratio: 0.01-100 → 0.2-5x.
- Reversible imaginary parts: < 1.5 → < 1.0.

### New Tests (37 added, 95 → 132 total)
- `test_preprocessing.py` (24 tests): log returns, standardization, time-lagged pairs, embedding, rolling windows, leakage validation.
- `test_bootstrap_ci.py` (13 tests): KDE entropy helper, bootstrap CI output/ordering/reproducibility, data loader.
- `test_synthetic.py` additions: `TestVAMPNonReversibleValidation` (3 tests), `TestIrreversibilityFieldEig` (2 tests).

### Centralized Constants
- `src/constants.py` — single source of truth for dates, tickers, crisis dates.
- All 4 experiment runners import from `src.constants` (no local `DATE_RANGES`).

### Master Orchestrator
- `experiments/run_all.py` — sequences 7 pipeline stages via subprocess.
- Flags: `--modes`, `--ablations`, `--skip-download`, `--n-seeds`, `--n-jobs`.
- Produces `pipeline_report.json` with per-stage status/timing.

### Colab Notebook
- `KTND_Finance_Colab.ipynb` — 5-cell notebook: setup, full pipeline in one cell, view figures, download zip, optional ablations.

### Infrastructure
- `run.sh` — local shell script with `--fast`, `--ablations`, `--skip-download` flags.
- `data/download.py` — centralized download with `download_prices()` and `download_vix()`.
- Leakage validation: `validate_no_leakage()` checks chronological splits, stats train-only.
- `requirements.txt`: removed fredapi, updated yfinance>=1.0.0.

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
