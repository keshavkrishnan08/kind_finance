# Spectral Decomposition of Entropy Production from Non-Self-Adjoint Koopman Operators

Code and data for the KTND (Koopman-Thermodynamic Neural Decomposition) framework. This method learns the spectral structure of non-equilibrium dynamics directly from time-series data, without access to equations of motion.

**Paper:** *Spectral Decomposition of Entropy Production from Non-Self-Adjoint Koopman Operators* — submitted to Physical Review E.

## What this does

A dual-lobe variational neural network learns the left and right eigenfunctions of a non-self-adjoint Koopman operator from raw time series. Complex eigenvalues encode rotational probability currents (broken detailed balance), and the antisymmetric part of the learned operator yields entropy production estimates without requiring a force field, diffusion matrix, or Fokker-Planck equation.

Key outputs:
- **Complex Koopman eigenvalues** — oscillation frequencies and decay rates of dynamical modes
- **Per-mode entropy production** — spectral decomposition of irreversibility (only complex modes contribute)
- **Irreversibility field I(x)** — pointwise map of where detailed balance breaks in state space
- **Frobenius entropy production** — non-perturbative estimator from the antisymmetric Koopman component
- **Spectral gap** — predicts regime transitions out of sample

## Repository structure

```
src/
  model/          # VAMPnet architecture, Koopman analysis, losses
  analysis/       # Spectral, regime detection, rolling, Chapman-Kolmogorov
  data/           # Data loading, preprocessing, IAAFT surrogates
  baselines/      # HMM, GARCH, DMD, PCA, LSTM-AE, threshold
  utils/          # Config, visualization, reproducibility
experiments/      # All experiment scripts (training, CV, ablations, figures)
tests/            # 139 unit tests (pytest)
config/           # YAML configs for univariate and multiasset runs
data/             # Market data (SPY + 10 cross-asset ETFs, 1993-2026)
outputs/
  figures/        # All publication figures
  results/        # JSON/CSV results from all experiments
```

## Quick start

```bash
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python experiments/run_all.py
```

This trains both univariate (SPY, K=5 modes) and multiasset (11 ETFs, K=15 modes) models, runs surrogate testing, cross-validation, entropy calibration, and generates all figures.

### Run individual experiments

```bash
# Train and analyze
python experiments/run_main.py --config config/default.yaml --mode univariate
python experiments/run_main.py --config config/default.yaml --mode multiasset

# Walk-forward cross-validation (5 folds)
python experiments/run_cv.py

# IAAFT surrogate testing + robustness
python experiments/run_robustness.py

# Brownian gyrator entropy calibration
python experiments/run_entropy_calibration.py

# Ablation study (10 seeds)
python experiments/run_ablations.py

# Generate figures
python experiments/run_figures.py
python experiments/gen_appendix_figures.py
```

### Run tests

```bash
pytest tests/ -v
```

## Key results

| Observable | Univariate (SPY) | Multiasset (11 ETFs) |
|---|---|---|
| Complex modes | 2/5 | 12/15 |
| Spectral gap | 0.196 | 0.168 |
| Frobenius EP | 0.10 nats/day | 0.40 nats/day |
| k-NN EP | 0.26 nats/day | 0.31 nats/day |
| IAAFT Cohen's d | 31.0 | 7.1 |
| Crisis AUROC | 0.78 | — |
| Gyrator calibration | r = 0.94 | — |

## Configuration

Default hyperparameters in `config/default.yaml`. Key settings:

- `n_modes`: 5 (univariate), 15 (multiasset)
- `hidden_dims`: [64, 64, 32] / [128, 128, 64]
- `beta_orthogonality`: 0.005
- `learning_rate`: 3e-4
- `n_epochs`: 800, `patience`: 80
- `tau`: 5 trading days

## Data

Market data is included in `data/`. To refresh from Yahoo Finance:

```bash
python data/download.py
```

## Citation

```bibtex
@article{krishnan2026spectral,
  title={Spectral Decomposition of Entropy Production from Non-Self-Adjoint Koopman Operators},
  author={Krishnan, Keshav},
  journal={Physical Review E},
  year={2026},
  note={Submitted}
}
```

## License

MIT
