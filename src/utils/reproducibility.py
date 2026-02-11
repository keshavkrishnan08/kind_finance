"""
Deterministic execution utilities for KTND-Finance experiments.

Ensures reproducible results across runs by seeding all relevant random number
generators and selecting the appropriate compute device.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all random number generators for reproducible execution.

    Sets seeds for Python's ``random`` module, NumPy, and PyTorch (both CPU
    and CUDA).  Also configures cuDNN for deterministic behaviour at the
    cost of some performance.

    Parameters
    ----------
    seed : int, optional
        The random seed to use across all generators.  Defaults to 42.

    Notes
    -----
    Setting ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` disables cuDNN auto-tuner
    heuristics, which may reduce GPU throughput but guarantees bitwise
    reproducibility for supported operations.

    The environment variable ``PYTHONHASHSEED`` is also set so that hash
    randomisation is disabled for the current process (though it only takes
    effect for child processes started after the call).
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns ``torch.device('cuda')`` when a CUDA-capable GPU is detected,
    otherwise ``torch.device('cpu')``.

    Returns
    -------
    torch.device
        The selected compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
