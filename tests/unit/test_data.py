"""Unit tests for FRED-MD data loading and tcode transformations."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcc_itransformer.data.fred_md import (
    apply_tcode,
    load_fred_md,
    remove_outliers,
    transform_panel,
)

# ---------------------------------------------------------------------------
# Inline fixtures
# ---------------------------------------------------------------------------

FRED_MD_CSV = """\
sasdate,RPI,W875RX1,INDPRO
Transform:,5,5,2
1990-01-01,100.5,200.3,50.0
1990-02-01,101.2,201.1,51.5
1990-03-01,102.0,202.5,53.0
1990-04-01,103.1,204.0,54.2
1990-05-01,104.0,205.8,55.1
1990-06-01,105.2,207.1,56.3
"""


@pytest.fixture()
def fred_csv_file(tmp_path: Path) -> Path:
    """Write a minimal FRED-MD CSV to a temporary directory."""
    csv_path = tmp_path / "fred_md.csv"
    csv_path.write_text(FRED_MD_CSV)
    return csv_path


# ---------------------------------------------------------------------------
# Tests: load_fred_md
# ---------------------------------------------------------------------------


class TestLoadFredMD:
    def test_load_fred_md_shape(self, fred_csv_file: Path) -> None:
        data, tcodes = load_fred_md(fred_csv_file)
        # 6 data rows, 3 series
        assert data.shape == (6, 3)
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_tcode_extraction_from_row2(self, fred_csv_file: Path) -> None:
        _, tcodes = load_fred_md(fred_csv_file)
        assert tcodes["RPI"] == 5
        assert tcodes["W875RX1"] == 5
        assert tcodes["INDPRO"] == 2

    def test_alternative_date_format(self, tmp_path: Path) -> None:
        csv_text = """\
sasdate,A,B
Transform:,1,2
1/1/1990,10.0,20.0
2/1/1990,11.0,21.0
3/1/1990,12.0,22.0
"""
        csv_path = tmp_path / "alt_date.csv"
        csv_path.write_text(csv_text)
        data, _ = load_fred_md(csv_path)
        assert len(data) == 3
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fred_md(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Tests: remove_outliers
# ---------------------------------------------------------------------------


class TestRemoveOutliers:
    def test_outlier_removal_spike(self) -> None:
        """A single extreme spike should be replaced with NaN."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(100)
        values[50] = 999.0  # extreme spike
        s = pd.Series(values)

        cleaned = remove_outliers(s, iqr_multiplier=10.0)
        assert pd.isna(cleaned.iloc[50]), "Spike should be NaN"

    def test_outlier_removal_preserves_normal(self) -> None:
        """Normal values within IQR bounds should be preserved."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(200)
        s = pd.Series(values)

        cleaned = remove_outliers(s, iqr_multiplier=10.0)
        # With 10*IQR, very few (if any) standard-normal values should be clipped
        assert cleaned.notna().sum() >= 195


# ---------------------------------------------------------------------------
# Tests: apply_tcode
# ---------------------------------------------------------------------------


class TestApplyTcode:
    def test_tcode_1_noop(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = apply_tcode(s, 1)
        pd.testing.assert_series_equal(result, s.astype(np.float64))

    def test_tcode_2_diff(self) -> None:
        s = pd.Series([10.0, 13.0, 17.0, 22.0])
        result = apply_tcode(s, 2)
        expected = s.diff()
        pd.testing.assert_series_equal(result, expected)

    def test_tcode_5_log_diff(self) -> None:
        s = pd.Series([100.0, 110.0, 121.0, 133.1])
        result = apply_tcode(s, 5)
        expected = np.log(s).diff()
        pd.testing.assert_series_equal(result, expected)

    def test_tcode_7_percent_change_diff(self) -> None:
        s = pd.Series([100.0, 105.0, 110.25, 115.0])
        result = apply_tcode(s, 7)
        expected = s.pct_change().diff()
        pd.testing.assert_series_equal(result, expected)

    def test_tcode_3_second_diff(self) -> None:
        s = pd.Series([1.0, 3.0, 6.0, 10.0, 15.0])
        result = apply_tcode(s, 3)
        expected = s.diff().diff()
        pd.testing.assert_series_equal(result, expected)

    def test_tcode_4_log(self) -> None:
        s = pd.Series([1.0, np.e, np.e**2])
        result = apply_tcode(s, 4)
        expected = pd.Series([0.0, 1.0, 2.0])
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_tcode_6_second_log_diff(self) -> None:
        s = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
        result = apply_tcode(s, 6)
        expected = np.log(s).diff().diff()
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_tcode_raises(self) -> None:
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="Invalid tcode"):
            apply_tcode(s, 0)
        with pytest.raises(ValueError, match="Invalid tcode"):
            apply_tcode(s, 8)


# ---------------------------------------------------------------------------
# Tests: transform_panel
# ---------------------------------------------------------------------------


class TestTransformPanel:
    def test_transform_panel_drops_nan_rows(self) -> None:
        """After differencing, leading NaN rows should be removed."""
        dates = pd.date_range("2000-01-01", periods=10, freq="MS")
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "A": rng.uniform(100, 200, 10),
                "B": rng.uniform(50, 100, 10),
            },
            index=dates,
        )
        tcodes = pd.Series({"A": 2, "B": 5})  # diff and log-diff both lose row 0

        result = transform_panel(data, tcodes)
        # First row should be dropped (NaN from differencing)
        assert result.shape[0] < data.shape[0]
        # No all-NaN rows at the start
        assert result.iloc[0].notna().all()
