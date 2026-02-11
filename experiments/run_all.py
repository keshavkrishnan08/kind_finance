#!/usr/bin/env python3
"""
Master orchestrator for the full KTND-Finance experiment pipeline.

Sequences all experiment stages via subprocess calls so each runner
stays independent (separate process, clean imports).  Produces a final
pipeline report summarising which stages succeeded.

Usage
-----
    # Full pipeline (univariate + multiasset)
    python experiments/run_all.py

    # Univariate only, skip data download
    python experiments/run_all.py --modes univariate --skip-download

    # Include ablation sweep (expensive ~1400 trials)
    python experiments/run_all.py --ablations --n-seeds 5 --n-jobs 4

    # Custom output directory
    python experiments/run_all.py --output-dir outputs/run_2026_02_09
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_stage(
    name: str,
    cmd: list[str],
    report: dict,
    cwd: Path = PROJECT_ROOT,
) -> bool:
    """Run a pipeline stage and record the result.

    Returns True if the stage succeeded, False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"  STAGE: {name}")
    print(f"  CMD:   {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            capture_output=False,
        )
        elapsed = time.time() - t0
        report["stages"][name] = {
            "status": "success",
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"\n  >> {name}: SUCCESS ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        report["stages"][name] = {
            "status": "failed",
            "returncode": e.returncode,
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"\n  >> {name}: FAILED (returncode={e.returncode}, {elapsed:.1f}s)")
        return False
    except FileNotFoundError:
        report["stages"][name] = {
            "status": "error",
            "message": "Command not found",
        }
        print(f"\n  >> {name}: ERROR (command not found)")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KTND-Finance: Full experiment pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["univariate", "multiasset"],
        choices=["univariate", "multiasset"],
        help="Experiment modes to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Base configuration file.",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run ablation sweep (expensive; ~1400 trials with 10 seeds).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds per ablation variant.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for ablations.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download (use cached data).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "pipeline": "KTND-Finance",
        "modes": args.modes,
        "stages": {},
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t_pipeline = time.time()
    all_ok = True

    # ---- Stage 1: Download data ----
    if not args.skip_download:
        ok = run_stage(
            "download_data",
            [python, "data/download.py", "--mode", "all"],
            report,
        )
        if not ok:
            all_ok = False

    # ---- Stage 2: Train main model (per mode) ----
    for mode in args.modes:
        mode_config = f"config/{mode}.yaml"
        mode_config_path = PROJECT_ROOT / mode_config
        config_flag = str(mode_config) if mode_config_path.exists() else args.config

        cmd = [
            python, "experiments/run_main.py",
            "--config", config_flag,
            "--mode", mode,
        ]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])

        ok = run_stage(f"train_{mode}", cmd, report)
        if not ok:
            all_ok = False

    # ---- Stage 3: Baselines ----
    if not args.skip_baselines:
        cmd = [python, "experiments/run_baselines.py", "--config", args.config]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        ok = run_stage("baselines", cmd, report)
        if not ok:
            all_ok = False

    # ---- Stage 4: Robustness / statistical tests (per mode) ----
    for mode in args.modes:
        cmd = [
            python, "experiments/run_robustness.py",
            "--config", args.config,
            "--mode", mode,
        ]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        ok = run_stage(f"robustness_{mode}", cmd, report)
        if not ok:
            all_ok = False

    # ---- Stage 5: Rolling spectral analysis (per mode) ----
    for mode in args.modes:
        cmd = [
            python, "experiments/run_rolling.py",
            "--config", args.config,
            "--mode", mode,
        ]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        ok = run_stage(f"rolling_{mode}", cmd, report)
        if not ok:
            all_ok = False

    # ---- Stage 6: Ablation sweep (optional) ----
    if args.ablations:
        cmd = [
            python, "experiments/run_ablations.py",
            "--config", args.config,
            "--n-seeds", str(args.n_seeds),
            "--n-jobs", str(args.n_jobs),
        ]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        ok = run_stage("ablations", cmd, report)
        if not ok:
            all_ok = False

    # ---- Stage 7: Generate figures ----
    cmd = [python, "experiments/run_figures.py"]
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    if results_dir.exists():
        cmd.extend(["--results-dir", str(results_dir)])
    cmd.extend(["--figures-dir", str(figures_dir)])
    ok = run_stage("figures", cmd, report)
    if not ok:
        all_ok = False

    # ---- Final report ----
    elapsed_total = time.time() - t_pipeline
    report["total_elapsed_seconds"] = round(elapsed_total, 1)
    report["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["all_passed"] = all_ok

    n_success = sum(1 for s in report["stages"].values() if s["status"] == "success")
    n_total = len(report["stages"])

    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("  KTND-Finance Pipeline Report")
    print(f"{'='*70}")
    print(f"  Stages passed:  {n_success}/{n_total}")
    print(f"  Total time:     {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    for name, info in report["stages"].items():
        status = info["status"].upper()
        t = info.get("elapsed_seconds", "N/A")
        print(f"    {name:30s}  {status:8s}  {t}s")
    print(f"  Report saved:   {report_path}")
    print(f"{'='*70}\n")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
