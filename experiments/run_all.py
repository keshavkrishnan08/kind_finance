#!/usr/bin/env python3
"""
Master orchestrator for the full KTND-Finance experiment pipeline.

Runs all 15 stages matching the Colab notebook. Each stage is isolated
in a subprocess so a failure in one stage never kills the pipeline.

Usage
-----
    # Full pipeline from scratch
    python experiments/run_all.py

    # Resume from stage 7 (skip stages 1-6)
    python experiments/run_all.py --resume-from 7

    # Univariate only
    python experiments/run_all.py --modes univariate

    # Include ablations (expensive)
    python experiments/run_all.py --ablations --n-seeds 10

    # Custom output directory
    python experiments/run_all.py --output-dir outputs/run_2026_03_07

Stage map
---------
     1  Quick tests
     2  Download data
     3  Train univariate
     4  Train multiasset
     5  Baselines
     6  Rolling spectral analysis
     7  Robustness univariate
     8  Robustness multiasset
     9  Walk-forward CV (both modes)
    10  Entropy calibration
    11  Entropy convergence (both modes)
    12  Figures
    13  Multi-seed (5 seeds)
    14  Ablations (optional, --ablations flag)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fix Intel OpenMP (MKL) / LLVM OpenMP (PyTorch) conflict → SIGSEGV on macOS/Anaconda
# KMP_DUPLICATE_LIB_OK: suppresses Intel OpenMP abort on duplicate detection
# MKL_THREADING_LAYER=GNU: forces MKL to use GNU OpenMP instead of Intel OpenMP
# OMP_NUM_THREADS=1: single-threaded BLAS avoids the thread-pool conflict entirely
_SUBPROCESS_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "MKL_THREADING_LAYER": "GNU",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


# =====================================================================
# Helpers
# =====================================================================

def run_stage(
    name: str,
    cmd: list[str],
    report: dict,
    check_files: list[Path] | None = None,
    cwd: Path = PROJECT_ROOT,
) -> bool:
    """Run a pipeline stage, record result, never raise."""
    print(f"\n{'='*70}")
    print(f"  STAGE: {name}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max per stage
            env=_SUBPROCESS_ENV,
        )
        elapsed = time.time() - t0

        # Print last 40 lines of stdout
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-40:]:
                print(f"  {line}")

        if result.returncode != 0:
            print(f"\n  === STDERR (last 20 lines) ===")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"  ! {line}")
            report["stages"][name] = {
                "status": "failed",
                "returncode": result.returncode,
                "elapsed_seconds": round(elapsed, 1),
            }
            print(f"\n  >> {name}: FAILED (exit {result.returncode}, {elapsed/60:.1f} min)")
            return False

        # Verify expected output files
        if check_files:
            missing = [f for f in check_files if not f.exists()]
            if missing:
                for f in missing:
                    print(f"    MISSING: {f.name}")
                report["stages"][name] = {
                    "status": "incomplete",
                    "missing_files": [str(f) for f in missing],
                    "elapsed_seconds": round(elapsed, 1),
                }
                print(f"  >> {name}: INCOMPLETE ({elapsed/60:.1f} min)")
                return False
            for f in check_files:
                sz = f.stat().st_size
                print(f"  OK: {f.name} ({sz:,} bytes)")

        report["stages"][name] = {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"  >> {name}: OK ({elapsed/60:.1f} min)")
        return True

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        report["stages"][name] = {
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"\n  >> {name}: TIMEOUT ({elapsed/60:.1f} min)")
        return False
    except Exception as e:
        elapsed = time.time() - t0
        report["stages"][name] = {
            "status": "error",
            "message": str(e),
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"\n  >> {name}: ERROR ({e})")
        return False


def _safe_json_load(path: Path) -> dict | None:
    """Load JSON file, return None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def should_skip(stage_num: int, resume_from: int, name: str,
                check_files: list[Path] | None = None) -> bool:
    """Check if stage should be skipped. Returns True if skipped."""
    if stage_num >= resume_from:
        return False
    if check_files:
        missing = [f for f in check_files if not f.exists()]
        if missing:
            print(f"  STAGE {stage_num} ({name}): Cannot skip -- "
                  f"missing: {[f.name for f in missing]}")
            return False
    print(f"  STAGE {stage_num} ({name}): SKIPPED (resume_from={resume_from})")
    return True


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KTND-Finance: Full experiment pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modes", nargs="+", default=["univariate", "multiasset"],
                   choices=["univariate", "multiasset"])
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--resume-from", type=int, default=1,
                   help="Stage number to resume from (1=fresh).")
    p.add_argument("--ablations", action="store_true",
                   help="Run ablation sweep after main pipeline.")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Seeds per ablation variant.")
    p.add_argument("--multi-seeds", type=int, default=5,
                   help="Number of seeds for multi-seed analysis.")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--skip-tests", action="store_true")
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()
    py = sys.executable

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    data_dir = PROJECT_ROOT / "data"

    for d in [output_dir, results_dir, models_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "pipeline": "KTND-Finance",
        "modes": args.modes,
        "stages": {},
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results: dict[str, bool] = {}
    t_start = time.time()
    rf = args.resume_from

    if rf > 1:
        print(f"RESUMING FROM STAGE {rf} (skipping stages 1-{rf-1})")

    # ==================================================================
    # STAGE 1: Quick tests
    # ==================================================================
    if args.skip_tests:
        print("  STAGE 1 (Quick tests): SKIPPED (--skip-tests)")
        results["tests"] = True
    elif should_skip(1, rf, "Quick tests"):
        results["tests"] = True
    else:
        results["tests"] = run_stage(
            "Quick tests",
            [py, "-m", "pytest", "tests/", "-q", "--tb=short",
             "-k", "not test_synthetic"],
            report,
        )

    # ==================================================================
    # STAGE 2: Download data
    # ==================================================================
    data_files = [data_dir / "prices.csv", data_dir / "vix.csv"]
    if should_skip(2, rf, "Download data", data_files):
        results["download"] = True
    else:
        results["download"] = run_stage(
            "Download data",
            [py, "data/download.py", "--mode", "all"],
            report,
            check_files=data_files,
        )

    # ==================================================================
    # STAGE 3: Train univariate
    # ==================================================================
    if "univariate" in args.modes:
        uni_files = [
            results_dir / "analysis_results_univariate.json",
            models_dir / "vampnet_univariate.pt",
            results_dir / "training_history_univariate.json",
        ]
        if should_skip(3, rf, "Train univariate", uni_files):
            results["train_uni"] = True
        else:
            results["train_uni"] = run_stage(
                "Train univariate (SPY)",
                [py, "experiments/run_main.py",
                 "--config", args.config, "--mode", "univariate",
                 "--seed", "42", "--output-dir", str(output_dir)],
                report,
                check_files=uni_files,
            )

    # ==================================================================
    # STAGE 4: Train multiasset
    # ==================================================================
    if "multiasset" in args.modes:
        multi_files = [
            results_dir / "analysis_results_multiasset.json",
            models_dir / "vampnet_multiasset.pt",
            results_dir / "training_history_multiasset.json",
        ]
        if should_skip(4, rf, "Train multiasset", multi_files):
            results["train_multi"] = True
        else:
            results["train_multi"] = run_stage(
                "Train multiasset (11 ETFs)",
                [py, "experiments/run_main.py",
                 "--config", args.config, "--mode", "multiasset",
                 "--seed", "42", "--output-dir", str(output_dir)],
                report,
                check_files=multi_files,
            )

    # ==================================================================
    # STAGE 5: Baselines
    # ==================================================================
    baseline_file = results_dir / "baseline_comparison.csv"
    if should_skip(5, rf, "Baselines", [baseline_file]):
        results["baselines"] = True
    else:
        results["baselines"] = run_stage(
            "Baselines (HMM, DMD, PCA, GARCH, VIX, LSTM-AE)",
            [py, "experiments/run_baselines.py",
             "--config", args.config, "--output-dir", str(results_dir)],
            report,
            check_files=[baseline_file],
        )

    # ==================================================================
    # STAGE 6: Rolling spectral analysis (requires univariate model)
    # ==================================================================
    rolling_file = results_dir / "spectral_gap_timeseries.csv"
    uni_ckpt = models_dir / "vampnet_univariate.pt"
    if should_skip(6, rf, "Rolling", [rolling_file]):
        results["rolling"] = True
    elif not uni_ckpt.exists():
        print("  STAGE 6 (Rolling): SKIPPED (no univariate checkpoint)")
        results["rolling"] = False
    else:
        results["rolling"] = run_stage(
            "Rolling spectral analysis",
            [py, "experiments/run_rolling.py",
             "--config", args.config, "--mode", "univariate",
             "--checkpoint", str(uni_ckpt),
             "--output-dir", str(results_dir)],
            report,
            check_files=[rolling_file],
        )

    # ==================================================================
    # STAGE 7: Robustness univariate
    # ==================================================================
    if "univariate" in args.modes:
        stat_file = results_dir / "statistical_tests.json"
        if should_skip(7, rf, "Robustness univariate", [stat_file]):
            results["robustness_uni"] = True
        else:
            results["robustness_uni"] = run_stage(
                "Robustness (univariate, IAAFT, 200 perms)",
                [py, "experiments/run_robustness.py",
                 "--config", args.config, "--mode", "univariate",
                 "--checkpoint", str(models_dir / "vampnet_univariate.pt"),
                 "--output-dir", str(results_dir),
                 "--n-permutations", "200"],
                report,
                check_files=[stat_file],
            )

    # ==================================================================
    # STAGE 8: Robustness multiasset
    # ==================================================================
    if "multiasset" in args.modes:
        stat_multi = results_dir / "statistical_tests_multiasset.json"
        if should_skip(8, rf, "Robustness multiasset", [stat_multi]):
            results["robustness_multi"] = True
        else:
            results["robustness_multi"] = run_stage(
                "Robustness (multiasset, IAAFT, 200 perms + PCA)",
                [py, "experiments/run_robustness.py",
                 "--config", args.config, "--mode", "multiasset",
                 "--checkpoint", str(models_dir / "vampnet_multiasset.pt"),
                 "--output-dir", str(results_dir),
                 "--n-permutations", "200"],
                report,
                check_files=[stat_multi],
            )

    # ==================================================================
    # STAGE 9: Walk-forward CV
    # ==================================================================
    for mode in args.modes:
        cv_file = results_dir / f"cv_results_{mode}.json"
        if should_skip(9, rf, f"CV {mode}", [cv_file]):
            results[f"cv_{mode}"] = True
        else:
            results[f"cv_{mode}"] = run_stage(
                f"Walk-forward CV ({mode})",
                [py, "experiments/run_cv.py",
                 "--config", args.config, "--mode", mode,
                 "--n-folds", "5", "--output-dir", str(results_dir)],
                report,
                check_files=[cv_file],
            )

    # ==================================================================
    # STAGE 10: Entropy calibration (Brownian gyrator)
    # ==================================================================
    ecal_file = results_dir / "entropy_calibration.json"
    if should_skip(10, rf, "Entropy calibration", [ecal_file]):
        results["entropy_cal"] = True
    else:
        results["entropy_cal"] = run_stage(
            "Entropy calibration (Brownian gyrator)",
            [py, "experiments/run_entropy_calibration.py",
             "--output-dir", str(results_dir), "--n-steps", "50000"],
            report,
            check_files=[ecal_file],
        )

    # ==================================================================
    # STAGE 11: Entropy convergence vs K
    # ==================================================================
    for mode in args.modes:
        ec_file = results_dir / f"entropy_convergence_{mode}.json"
        if should_skip(11, rf, f"Entropy convergence {mode}", [ec_file]):
            results[f"entropy_conv_{mode}"] = True
        else:
            results[f"entropy_conv_{mode}"] = run_stage(
                f"Entropy convergence ({mode}, K=3..50)",
                [py, "experiments/run_entropy_convergence.py",
                 "--config", args.config, "--mode", mode,
                 "--k-values", "3", "5", "10", "15", "20", "30", "50",
                 "--output-dir", str(results_dir)],
                report,
                check_files=[ec_file],
            )

    # ==================================================================
    # STAGE 12: Figures
    # ==================================================================
    if should_skip(12, rf, "Figures"):
        results["figures"] = True
    else:
        results["figures"] = run_stage(
            "Generate figures",
            [py, "experiments/run_figures.py",
             "--results-dir", str(results_dir),
             "--figures-dir", str(figures_dir)],
            report,
        )

    # ==================================================================
    # STAGE 13: Multi-seed
    # ==================================================================
    ms_file = results_dir / "multi_seed_summary.json"
    if should_skip(13, rf, "Multi-seed", [ms_file]):
        results["multi_seed"] = True
    else:
        extra_seeds = list(range(args.multi_seeds - 1))  # [0, 1, 2, 3]
        multi_seed_data: dict = {}

        # Collect seed-42 results
        for mode in args.modes:
            ap = results_dir / f"analysis_results_{mode}.json"
            data = _safe_json_load(ap) if ap.exists() else None
            if data is not None:
                multi_seed_data.setdefault(mode, {})[42] = data

        # Train extra seeds
        for seed in extra_seeds:
            seed_dir = output_dir / f"seed_{seed}"
            seed_results = seed_dir / "results"
            seed_results.mkdir(parents=True, exist_ok=True)
            (seed_dir / "models").mkdir(parents=True, exist_ok=True)

            for mode in args.modes:
                seed_ap = seed_results / f"analysis_results_{mode}.json"
                if seed_ap.exists():
                    data = _safe_json_load(seed_ap)
                    if data is not None:
                        print(f"  Seed {seed} {mode}: CACHED")
                        multi_seed_data.setdefault(mode, {})[seed] = data
                        continue

                print(f"  Seed {seed} {mode}: TRAINING...", flush=True)
                run_stage(
                    f"Seed {seed} {mode}",
                    [py, "experiments/run_main.py",
                     "--config", args.config, "--mode", mode,
                     "--seed", str(seed), "--output-dir", str(seed_dir)],
                    report,
                )
                data = _safe_json_load(seed_ap) if seed_ap.exists() else None
                if data is not None:
                    multi_seed_data.setdefault(mode, {})[seed] = data

        # Compute summaries
        metrics = [
            "vamp2_score", "spectral_gap", "entropy_empirical", "entropy_total",
            "mean_irreversibility", "detailed_balance_violation",
            "fluctuation_theorem_ratio", "n_complex_modes", "complex_fraction",
            "ktnd_nber_accuracy", "ktnd_nber_f1",
        ]
        multi_seed_summary: dict = {}
        for mode in args.modes:
            if mode not in multi_seed_data:
                continue
            seed_data = multi_seed_data[mode]
            seeds_present = sorted(seed_data.keys(), key=lambda x: int(x))
            summary: dict = {"n_seeds": len(seeds_present), "seeds": seeds_present}
            for metric in metrics:
                vals = []
                for s in seeds_present:
                    v = seed_data[s].get(metric) if isinstance(seed_data[s], dict) else None
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            pass
                if vals:
                    summary[f"{metric}_mean"] = float(np.nanmean(vals))
                    summary[f"{metric}_std"] = (
                        float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
                    )
            multi_seed_summary[mode] = summary

        with open(ms_file, "w") as f:
            json.dump(multi_seed_summary, f, indent=2, default=str, allow_nan=True)

        results["multi_seed"] = ms_file.exists()

    # ==================================================================
    # STAGE 14: Ablations (optional)
    # ==================================================================
    if args.ablations:
        abl_file = results_dir / "ablation_summary.csv"
        if should_skip(14, rf, "Ablations", [abl_file]):
            results["ablations"] = True
        else:
            results["ablations"] = run_stage(
                f"Ablations ({args.n_seeds} seeds)",
                [py, "-u", "experiments/run_ablations.py",
                 "--config", args.config,
                 "--n-seeds", str(args.n_seeds),
                 "--n-jobs", "1",
                 "--output-dir", str(results_dir)],
                report,
                check_files=[abl_file],
            )

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    total_min = (time.time() - t_start) / 60
    report["total_elapsed_seconds"] = round(total_min * 60, 1)
    report["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    n_ok = sum(1 for v in results.values() if v)
    n_total = len(results)

    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print(f"\n{'='*70}")
    print(f"  KTND-Finance Pipeline Report")
    print(f"{'='*70}")
    print(f"  Stages passed: {n_ok}/{n_total} ({total_min:.1f} min)")
    for name, ok in results.items():
        print(f"    {'OK' if ok else 'FAIL':6s}  {name}")

    # Print key results (wrapped so report always completes)
    try:
        for mode in args.modes:
            ap = results_dir / f"analysis_results_{mode}.json"
            r = _safe_json_load(ap) if ap.exists() else None
            if r is None:
                continue
            label = "Univariate (SPY)" if mode == "univariate" else "Multiasset (11 ETFs)"
            print(f"\n  === {label} ===")
            print(f"    Spectral gap: {r.get('spectral_gap', 'N/A')}")
            print(f"    Entropy spec: {r.get('entropy_total', 'N/A')}")
            print(f"    DB violation: {r.get('detailed_balance_violation', 'N/A')}")
            nber_f1 = r.get("ktnd_nber_f1")
            if isinstance(nber_f1, (int, float)):
                print(f"    NBER F1:      {nber_f1:.3f}")

        # Statistical tests
        for mode, suffix in [("univariate", ""), ("multiasset", "_multiasset")]:
            sp = results_dir / f"statistical_tests{suffix}.json"
            st = _safe_json_load(sp) if sp.exists() else None
            if st is None:
                continue
            perm = st.get("permutation_irreversibility", {})
            if "p_value" in perm:
                p = perm["p_value"]
                d = perm.get("cohens_d", "?")
                p_str = f"{p:.4f}" if isinstance(p, (int, float)) else str(p)
                d_str = f"{d:.2f}" if isinstance(d, (int, float)) else str(d)
                print(f"    Permutation ({mode}): p={p_str}, d={d_str}")

        # CV
        for mode in args.modes:
            cvp = results_dir / f"cv_results_{mode}.json"
            cv = _safe_json_load(cvp) if cvp.exists() else None
            if cv is not None:
                v = cv.get("vamp2_mean", "?")
                s = cv.get("vamp2_std", "?")
                print(f"    CV ({mode}): vamp2={v} +/- {s}")

        # Crisis prediction
        cp = results_dir / "crisis_prediction.json"
        pred = _safe_json_load(cp) if cp.exists() else None
        if pred is not None:
            auroc = pred.get("auroc_spectral")
            vix = pred.get("auroc_vix_baseline")
            if isinstance(auroc, (int, float)):
                vix_str = f"{vix:.4f}" if isinstance(vix, (int, float)) else "N/A"
                print(f"    Crisis AUROC: {auroc:.4f} vs VIX {vix_str}")
    except Exception as e:
        print(f"\n  (Report printing error: {e})")

    print(f"\n  Report: {report_path}")
    print(f"  Results: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
