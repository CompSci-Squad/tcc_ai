"""Unit tests for effective sample size and non-overlapping index extraction."""

from __future__ import annotations

import numpy as np
import pytest

from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)


class TestComputeEffectiveN:
    def test_basic(self) -> None:
        assert compute_effective_n(120, 12) == 10

    def test_floor_division(self) -> None:
        assert compute_effective_n(125, 12) == 10

    def test_small(self) -> None:
        assert compute_effective_n(5, 6) == 0

    def test_equal(self) -> None:
        assert compute_effective_n(12, 12) == 1

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size must be positive"):
            compute_effective_n(100, 0)


class TestExtractNonOverlappingIndices:
    def test_stride_1(self) -> None:
        idx = extract_non_overlapping_indices(n_windows=100, window_size=12, stride=1)
        # step = ceil(12/1) = 12
        expected = np.arange(0, 100, 12)
        np.testing.assert_array_equal(idx, expected)

    def test_stride_equals_window(self) -> None:
        idx = extract_non_overlapping_indices(n_windows=10, window_size=6, stride=6)
        # step = ceil(6/6) = 1 → all indices
        expected = np.arange(0, 10, 1)
        np.testing.assert_array_equal(idx, expected)

    def test_invalid_stride(self) -> None:
        with pytest.raises(ValueError, match="stride must be positive"):
            extract_non_overlapping_indices(n_windows=10, window_size=6, stride=0)

    def test_no_overlap(self) -> None:
        idx = extract_non_overlapping_indices(n_windows=50, window_size=6, stride=1)
        # Consecutive selected indices differ by >= window_size
        diffs = np.diff(idx)
        assert np.all(diffs >= 6)
