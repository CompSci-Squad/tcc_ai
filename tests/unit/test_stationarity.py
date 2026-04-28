"""Tests for ADF+KPSS joint stationarity validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcc_itransformer.data.stationarity import (
    check_series_stationarity,
    validate_panel_stationarity,
)


def test_white_noise_is_stationary():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    res = check_series_stationarity(x, name="wn")
    assert res.is_stationary is True
    assert res.adf_pvalue < 0.05
    assert res.kpss_pvalue >= 0.05


def test_random_walk_is_not_stationary():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(300))
    res = check_series_stationarity(x, name="rw")
    assert res.is_stationary is False


def test_validate_panel_returns_one_row_per_series():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", periods=200, freq="MS")
    df = pd.DataFrame(
        {
            "stationary": rng.standard_normal(200),
            "trending": np.cumsum(rng.standard_normal(200)),
        },
        index=dates,
    )
    out = validate_panel_stationarity(df)
    assert set(out.index) == {"stationary", "trending"}
    assert "is_stationary" in out.columns


def test_short_series_returns_non_stationary_safely():
    res = check_series_stationarity(np.arange(5), name="short")
    assert res.is_stationary is False
