"""Effective sample size computation for overlapping windows."""

from __future__ import annotations

import math

import numpy as np


def extract_non_overlapping_indices(
    n_windows: int,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Return indices of non-overlapping windows from an overlapping window set.

    When windows are created with ``stride < window_size``, adjacent windows
    share data points.  This function selects a subset of window indices such
    that no two selected windows overlap.

    Args:
        n_windows: Total number of overlapping windows.
        window_size: Length of each window.
        stride: Step between consecutive window start positions.

    Returns:
        1-D integer array of selected window indices.
    """
    if stride <= 0:
        msg = f"stride must be positive, got {stride}"
        raise ValueError(msg)

    step = max(1, math.ceil(window_size / stride))
    return np.arange(0, n_windows, step)


def compute_effective_n(n_total: int, window_size: int) -> int:
    """Compute the effective (independent) sample count for overlapping windows.

    Args:
        n_total: Total number of time-series observations.
        window_size: Length of each window.

    Returns:
        Effective number of independent samples (floor division).
    """
    if window_size <= 0:
        msg = f"window_size must be positive, got {window_size}"
        raise ValueError(msg)
    return n_total // window_size
