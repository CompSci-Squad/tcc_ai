"""FRED-MD dataset loading with tcode stationarity transformations."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TCODE_NAMES: dict[int, str] = {
    1: "level",
    2: "first_difference",
    3: "second_difference",
    4: "log",
    5: "first_difference_log",
    6: "second_difference_log",
    7: "first_difference_pct_change",
}


def load_fred_md(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a FRED-MD CSV file and extract transformation codes.

    The FRED-MD CSV layout is:
        Row 1: column headers (sasdate, series_1, series_2, ...)
        Row 2: transformation codes prefixed by "Transform:" in the date column
        Row 3+: actual monthly data

    Args:
        path: Path to the FRED-MD CSV file.

    Returns:
        A tuple of (data_df, tcodes_series) where:
            - data_df: DataFrame with DatetimeIndex and float columns
            - tcodes_series: Series mapping column name -> integer tcode

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the file format is unexpected.
    """
    path = Path(path)
    if not path.exists():
        msg = f"FRED-MD CSV not found: {path}"
        raise FileNotFoundError(msg)

    # Verify SHA-256 integrity if hash file exists
    hash_path = path.with_suffix(".sha256")
    if hash_path.exists():
        if not verify_sha256(path, hash_path):
            msg = f"SHA-256 verification failed for {path}"
            raise ValueError(msg)
        logger.info("SHA-256 verified for %s", path.name)

    raw = pd.read_csv(path, header=0)

    # --- Extract tcodes from row 0 of the body (row 2 of the file) ---
    tcode_row = raw.iloc[0]
    date_col = raw.columns[0]  # "sasdate" or similar
    series_cols = [c for c in raw.columns if c != date_col]

    tcodes = pd.Series(
        {col: int(float(tcode_row[col])) for col in series_cols},
        dtype=int,
        name="tcode",
    )

    # --- Parse the actual data (row 1 onward in the body, i.e. row 3+ of file) ---
    data = raw.iloc[1:].copy()
    data[date_col] = pd.to_datetime(data[date_col], format="mixed", dayfirst=False)
    data = data.set_index(date_col)
    data.index.name = None

    # Convert all columns to float
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.astype(np.float64)
    logger.info("Loaded FRED-MD: %d observations × %d series", len(data), len(series_cols))
    return data, tcodes


def verify_sha256(csv_path: Path, hash_path: Path) -> bool:
    """Verify SHA-256 integrity of a FRED-MD CSV against a stored hash.

    Args:
        csv_path: Path to the CSV file.
        hash_path: Path to the file containing the expected hex digest.

    Returns:
        True if the computed hash matches, False otherwise.
    """
    csv_path = Path(csv_path)
    hash_path = Path(hash_path)

    expected_hash = hash_path.read_text().strip().lower()

    sha256 = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            sha256.update(chunk)

    computed = sha256.hexdigest().lower()
    match = computed == expected_hash
    if not match:
        logger.warning(
            "SHA-256 mismatch for %s: expected %s, got %s",
            csv_path.name,
            expected_hash,
            computed,
        )
    return match


def remove_outliers(series: pd.Series, iqr_multiplier: float = 10.0) -> pd.Series:
    """Replace outliers with NaN using the IQR fence method.

    A value is considered an outlier if its absolute deviation from the median
    exceeds ``iqr_multiplier * IQR``.

    Args:
        series: A numeric pandas Series.
        iqr_multiplier: Multiplier for the IQR threshold.

    Returns:
        Series with outliers replaced by NaN.
    """
    s = series.copy()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    median = s.median()

    if iqr == 0:
        return s

    mask = (s - median).abs() > iqr_multiplier * iqr
    s[mask] = np.nan
    return s


def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """Apply a FRED-MD transformation code to a single series.

    Transformation codes follow the FRED-MD specification:
        1: level (no transformation)
        2: Δx (first difference)
        3: Δ²x (second difference)
        4: log(x)
        5: Δlog(x)
        6: Δ²log(x)
        7: Δ(x_t / x_{t-1} - 1) (first difference of percent change)

    Args:
        series: Numeric pandas Series (time series for one variable).
        tcode: Integer transformation code (1-7).

    Returns:
        Transformed Series (may contain leading NaNs from differencing).

    Raises:
        ValueError: If tcode is not in 1-7.
    """
    if tcode not in range(1, 8):
        msg = f"Invalid tcode {tcode}; must be 1-7"
        raise ValueError(msg)

    s = series.copy().astype(np.float64)

    if tcode == 1:
        return s
    if tcode == 2:
        return s.diff()
    if tcode == 3:
        return s.diff().diff()
    if tcode == 4:
        return np.log(s)
    if tcode == 5:
        return np.log(s).diff()
    if tcode == 6:
        return np.log(s).diff().diff()
    # tcode == 7
    pct = s.pct_change()
    return pct.diff()


def transform_panel(
    data: pd.DataFrame,
    tcodes: pd.Series,
    iqr_multiplier: float = 10.0,
) -> pd.DataFrame:
    """Apply outlier removal and tcode transformation to all series.

    Steps for each column:
        1. Remove outliers via IQR fence
        2. Apply the corresponding tcode transformation

    Leading NaN rows introduced by differencing are dropped from the result.

    Args:
        data: DataFrame with DatetimeIndex, one column per economic series.
        tcodes: Series mapping column name to integer tcode (1-7).
        iqr_multiplier: Multiplier for outlier detection.

    Returns:
        Transformed DataFrame with leading NaN rows removed.
    """
    result = pd.DataFrame(index=data.index)

    for col in data.columns:
        if col not in tcodes.index:
            logger.warning("Column %s has no tcode — skipping", col)
            continue
        cleaned = remove_outliers(data[col], iqr_multiplier=iqr_multiplier)
        result[col] = apply_tcode(cleaned, int(tcodes[col]))

    # Drop leading rows that are all-NaN (from differencing)
    first_valid = result.apply(lambda c: c.first_valid_index()).max()
    if first_valid is not None:
        result = result.loc[first_valid:]

    logger.info(
        "Panel transformed: %d observations × %d series after dropping leading NaNs",
        len(result),
        len(result.columns),
    )
    return result
