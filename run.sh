#!/usr/bin/env bash
#
# KTND-Finance: Full Experiment Pipeline
#
# Runs the complete pipeline from data download through figure generation.
# All results land in outputs/ (or --output-dir if specified).
#
# Usage:
#   ./run.sh                          # Full pipeline (univariate + multiasset)
#   ./run.sh --fast                   # Univariate only, skip ablations
#   ./run.sh --ablations              # Include ablation sweep (~hours)
#   ./run.sh --output-dir /path/to    # Custom output directory
#
# Prerequisites:
#   conda activate base  (or your env with pytorch, scipy, yfinance, etc.)
#   pip install yfinance>=1.0 hmmlearn statsmodels arch
#
# Data source:
#   Yahoo Finance via yfinance API (free, no key needed).
#   Tickers: SPY (1993+), 11 cross-asset ETFs (2007+), VIX (1993+).
#   ~8300 trading days for univariate, ~4700 for multiasset.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────
MODES="univariate multiasset"
ABLATIONS=""
FAST=""
OUTPUT_DIR=""
N_SEEDS=10
N_JOBS=4
SKIP_DOWNLOAD=""

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            FAST=1
            MODES="univariate"
            shift ;;
        --ablations)
            ABLATIONS="--ablations"
            shift ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2 ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2 ;;
        --n-jobs)
            N_JOBS="$2"
            shift 2 ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift ;;
        --help|-h)
            echo "Usage: ./run.sh [--fast] [--ablations] [--output-dir DIR] [--n-seeds N] [--n-jobs N] [--skip-download]"
            echo ""
            echo "Options:"
            echo "  --fast           Univariate only, skip ablations (quick ~30 min)"
            echo "  --ablations      Include ablation sweep (adds ~2-8 hours)"
            echo "  --output-dir     Custom output directory (default: outputs/)"
            echo "  --n-seeds N      Seeds per ablation variant (default: 10)"
            echo "  --n-jobs N       Parallel jobs for ablations (default: 4)"
            echo "  --skip-download  Skip data download (use cached data)"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# ── Environment check ────────────────────────────────────────────────
echo "========================================"
echo "  KTND-Finance Experiment Pipeline"
echo "========================================"
echo "  Python:    $(python --version 2>&1)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  yfinance:  $(python -c 'import yfinance; print(yfinance.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  Modes:     $MODES"
echo "  Ablations: ${ABLATIONS:-no}"
echo "  Output:    ${OUTPUT_DIR:-outputs/}"
echo "========================================"

# Check critical imports
python -c "
import torch, scipy, numpy, pandas, yfinance
from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.losses import total_loss
print('All imports OK')
" || { echo "ERROR: Missing dependencies. Install via: pip install torch scipy numpy pandas yfinance hmmlearn statsmodels arch"; exit 1; }

# ── Build orchestrator command ───────────────────────────────────────
CMD="python experiments/run_all.py --modes $MODES"

if [[ -n "$ABLATIONS" ]]; then
    CMD="$CMD --ablations --n-seeds $N_SEEDS --n-jobs $N_JOBS"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [[ -n "$SKIP_DOWNLOAD" ]]; then
    CMD="$CMD --skip-download"
fi

# ── Run tests first ─────────────────────────────────────────────────
echo ""
echo ">> Step 0: Running test suite..."
python -m pytest tests/ -q --tb=short 2>&1 | tail -5
PYTEST_EXIT=${PIPESTATUS[0]}
if [[ $PYTEST_EXIT -ne 0 ]]; then
    echo "WARNING: Some tests failed (exit code $PYTEST_EXIT). Continuing anyway..."
fi

# ── Run pipeline ────────────────────────────────────────────────────
echo ""
echo ">> Running: $CMD"
echo ""
START_TIME=$(date +%s)

$CMD

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REMAIN=$(( ELAPSED % 60 ))

echo ""
echo "========================================"
echo "  Pipeline Complete"
echo "  Total time: ${MINUTES}m ${SECONDS_REMAIN}s"
echo "  Report:     ${OUTPUT_DIR:-outputs}/pipeline_report.json"
echo "========================================"
