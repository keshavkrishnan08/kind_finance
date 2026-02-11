"""
Experiment logging for KTND-Finance.

Provides structured logging of training runs, including per-epoch metrics
(via CSV), configuration snapshots (via JSON), named results, and full
training history serialisation.  All artefacts are written under a
timestamped run directory so that concurrent experiments never collide.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ExperimentLogger:
    """Structured logger for a single experiment run.

    On construction a unique run directory is created under *log_dir* using
    the current UTC timestamp.  All subsequent outputs (CSV metrics, JSON
    config, result summaries, training histories) are written into this
    directory.

    Parameters
    ----------
    log_dir : str or Path, optional
        Root directory for all experiment logs.  Defaults to
        ``outputs/logs``.
    run_name : str or None, optional
        Human-readable name for the run.  If ``None``, a name is derived
        from the current timestamp.

    Attributes
    ----------
    run_dir : Path
        The directory created for this particular run.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "outputs/logs",
        run_name: Optional[str] = None,
    ) -> None:
        self.log_dir = Path(log_dir)

        # Build a unique run directory name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if run_name is not None:
            dir_name = f"{timestamp}_{run_name}"
        else:
            dir_name = timestamp

        self.run_dir = self.log_dir / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # --- Standard-library logger ---
        self._logger = logging.getLogger(f"ktnd_finance.{dir_name}")
        self._logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers when ExperimentLogger is instantiated
        # multiple times in the same process (e.g. notebooks).
        if not self._logger.handlers:
            # File handler: every message
            fh = logging.FileHandler(
                self.run_dir / "experiment.log", encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(fh)

            # Console handler: INFO and above
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(
                logging.Formatter("%(levelname)-8s | %(message)s")
            )
            self._logger.addHandler(ch)

        # --- CSV metrics file ---
        self._metrics_path = self.run_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

        # --- Results accumulator ---
        self._results: Dict[str, Any] = {}

        self._logger.info("Experiment run initialised: %s", self.run_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Append one row of training / validation metrics to the CSV log.

        Column names are derived from the metric dictionaries on the first
        call.  Train metrics are prefixed with ``train_`` and validation
        metrics with ``val_``.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed or 1-indexed -- the logger is
            agnostic).
        train_metrics : dict
            Mapping of metric names to scalar training values.
        val_metrics : dict
            Mapping of metric names to scalar validation values.
        """
        row: Dict[str, Any] = {"epoch": epoch}
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value

        # Lazy-initialise the CSV writer on the first call so that columns
        # are inferred from actual data.
        if not self._csv_header_written:
            self._csv_file = open(
                self._metrics_path, "w", newline="", encoding="utf-8"
            )
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
            self._csv_header_written = True

        self._csv_writer.writerow(row)  # type: ignore[union-attr]
        self._csv_file.flush()  # type: ignore[union-attr]

        # Human-readable summary
        train_str = ", ".join(f"{k}={v:.6g}" for k, v in train_metrics.items())
        val_str = ", ".join(f"{k}={v:.6g}" for k, v in val_metrics.items())
        self._logger.info(
            "Epoch %d  |  train: %s  |  val: %s", epoch, train_str, val_str
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        """Save the experiment configuration as a JSON file.

        Parameters
        ----------
        config : dict
            Arbitrary (JSON-serialisable) configuration dictionary.
        """
        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        self._logger.info("Configuration saved to %s", config_path)

    def log_result(self, name: str, value: Any) -> None:
        """Log a named scalar or summary result.

        Results are accumulated in memory and flushed to
        ``results.json`` each time this method is called so that partial
        results survive unexpected termination.

        Parameters
        ----------
        name : str
            Identifier for the result (e.g. ``'test_rmse'``).
        value : Any
            The result value.  Must be JSON-serialisable.
        """
        self._results[name] = value
        self._logger.info("Result  %s = %s", name, value)

        # Persist incrementally
        results_path = self.run_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, default=str)

    def save_history(
        self,
        history: Dict[str, List[float]],
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Save a full training history dictionary as JSON.

        Parameters
        ----------
        history : dict
            Mapping of metric names to lists of per-epoch values, e.g.
            ``{'train_loss': [0.5, 0.3, ...], 'val_loss': [0.6, 0.4, ...]}``.
        path : str or Path or None, optional
            Destination path.  If ``None`` the history is written to
            ``<run_dir>/history.json``.
        """
        if path is None:
            path = self.run_dir / "history.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        self._logger.info("Training history saved to %s", path)

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close open file handles."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; do not raise during garbage collection.
        try:
            self.close()
        except Exception:
            pass
