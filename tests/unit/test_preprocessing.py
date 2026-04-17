"""Unit tests for data preprocessing, scaling, windowing, and Dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tcc_itransformer.data.dataset import FREDMDWindowDataset
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    scale_splits,
    split_by_date,
)

# ---------------------------------------------------------------------------
# Inline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def panel_with_nans() -> pd.DataFrame:
    """Panel where one column has >10% NaN."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=100, freq="MS")
    df = pd.DataFrame(
        {
            "good_1": rng.standard_normal(100),
            "good_2": rng.standard_normal(100),
            "bad": rng.standard_normal(100),
        },
        index=dates,
    )
    # Inject 15% NaN into "bad"
    nan_idx = rng.choice(100, size=15, replace=False)
    df.iloc[nan_idx, df.columns.get_loc("bad")] = np.nan
    return df


@pytest.fixture()
def time_series_df() -> pd.DataFrame:
    """Deterministic panel for split and scaling tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")  # 10 years
    return pd.DataFrame(
        rng.standard_normal((120, 5)),
        index=dates,
        columns=[f"s{i}" for i in range(5)],
    )


# ---------------------------------------------------------------------------
# Tests: drop_high_nan_series
# ---------------------------------------------------------------------------


class TestDropHighNanSeries:
    def test_drop_high_nan_series(self, panel_with_nans: pd.DataFrame) -> None:
        cleaned, dropped = drop_high_nan_series(panel_with_nans, threshold=0.1)
        assert "bad" in dropped
        assert "good_1" not in dropped
        assert "good_2" not in dropped
        assert "bad" not in cleaned.columns

    def test_no_drop_when_threshold_high(self, panel_with_nans: pd.DataFrame) -> None:
        cleaned, dropped = drop_high_nan_series(panel_with_nans, threshold=0.99)
        assert len(dropped) == 0
        assert cleaned.shape[1] == panel_with_nans.shape[1]


# ---------------------------------------------------------------------------
# Tests: forward_fill_nans
# ---------------------------------------------------------------------------


class TestForwardFillNans:
    def test_forward_fill_nans_no_remaining(self) -> None:
        dates = pd.date_range("2000-01-01", periods=10, freq="MS")
        df = pd.DataFrame(
            {"A": [np.nan, 1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0]},
            index=dates,
        )
        filled = forward_fill_nans(df)
        assert filled.isna().sum().sum() == 0
        # First value should be backfilled from the second row
        assert filled.iloc[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Tests: split_by_date
# ---------------------------------------------------------------------------


class TestSplitByDate:
    def test_split_by_date_boundaries(self, time_series_df: pd.DataFrame) -> None:
        train, val, test = split_by_date(
            time_series_df,
            train_end="2018-12-01",
            val_end="2021-12-01",
        )
        # Train: 2015-01 through 2018-12 = 48 months
        assert len(train) == 48
        assert train.index.max() == pd.Timestamp("2018-12-01")

        # Val: 2019-01 through 2021-12 = 36 months
        assert len(val) == 36
        assert val.index.min() > pd.Timestamp("2018-12-01")
        assert val.index.max() == pd.Timestamp("2021-12-01")

        # Test: 2022-01 through 2024-12 = 36 months
        assert len(test) == 36
        assert test.index.min() > pd.Timestamp("2021-12-01")

        # No leakage: total == original
        assert len(train) + len(val) + len(test) == len(time_series_df)


# ---------------------------------------------------------------------------
# Tests: scaling
# ---------------------------------------------------------------------------


class TestScaling:
    def test_scaler_fit_train_only(self, time_series_df: pd.DataFrame) -> None:
        """Scaler fitted on train — val/test means should NOT be zero."""
        train, val, test = split_by_date(
            time_series_df, train_end="2018-12-01", val_end="2021-12-01"
        )
        scaler = fit_scaler(train)
        train_s, val_s, test_s = scale_splits(train, val, test, scaler)

        # Train mean should be ~0 (within floating-point tolerance)
        np.testing.assert_allclose(train_s.mean(axis=0), 0.0, atol=1e-10)

        # Val and test means should generally NOT be zero
        # (different distribution window than train)
        val_means = np.abs(val_s.mean(axis=0))
        test_means = np.abs(test_s.mean(axis=0))
        # At least some columns should have non-zero mean
        assert np.any(val_means > 0.01), "Val means unexpectedly close to zero"
        assert np.any(test_means > 0.01), "Test means unexpectedly close to zero"


# ---------------------------------------------------------------------------
# Tests: create_windows
# ---------------------------------------------------------------------------


class TestCreateWindows:
    def test_create_windows_shape(self) -> None:
        data = np.random.default_rng(42).standard_normal((100, 5))
        windows = create_windows(data, window_size=12, stride=1)
        expected_n = (100 - 12) // 1 + 1  # 89
        assert windows.shape == (expected_n, 12, 5)

    def test_create_windows_non_overlapping(self) -> None:
        data = np.random.default_rng(42).standard_normal((48, 3))
        windows = create_windows(data, window_size=12, stride=12)
        assert windows.shape == (4, 12, 3)
        # Windows should be contiguous, non-overlapping chunks
        np.testing.assert_array_equal(windows[0], data[0:12])
        np.testing.assert_array_equal(windows[1], data[12:24])

    def test_create_windows_stride(self) -> None:
        data = np.random.default_rng(42).standard_normal((24, 2))
        windows = create_windows(data, window_size=6, stride=3)
        expected_n = (24 - 6) // 3 + 1  # 7
        assert windows.shape == (expected_n, 6, 2)

    def test_create_windows_too_short_raises(self) -> None:
        data = np.random.default_rng(42).standard_normal((5, 2))
        with pytest.raises(ValueError, match="Not enough timesteps"):
            create_windows(data, window_size=12)


# ---------------------------------------------------------------------------
# Tests: FREDMDWindowDataset
# ---------------------------------------------------------------------------


class TestFREDMDWindowDataset:
    def test_dataset_getitem_shape(self) -> None:
        windows = np.random.default_rng(42).standard_normal((30, 12, 20)).astype(np.float32)
        ds = FREDMDWindowDataset(windows)
        x, idx = ds[0]
        assert x.shape == (12, 20)
        assert x.dtype == torch.float32
        assert idx == 0

    def test_dataset_length(self) -> None:
        windows = np.random.default_rng(42).standard_normal((50, 6, 10)).astype(np.float32)
        ds = FREDMDWindowDataset(windows)
        assert len(ds) == 50

    def test_dataset_last_item(self) -> None:
        windows = np.random.default_rng(42).standard_normal((20, 12, 5)).astype(np.float32)
        ds = FREDMDWindowDataset(windows)
        x, idx = ds[19]
        assert idx == 19
        assert x.shape == (12, 5)
