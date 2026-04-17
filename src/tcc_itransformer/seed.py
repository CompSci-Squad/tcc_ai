"""Deterministic seed management — single source of truth for all RNG seeds."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds and enable deterministic mode.

    Args:
        seed: Integer seed for reproducibility across all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
