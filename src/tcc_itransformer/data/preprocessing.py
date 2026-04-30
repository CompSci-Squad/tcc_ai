"""Data preprocessing: NaN handling, splitting, scaling, and windowing."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed (no full load).

    Used to verify the ETL-v2 panel that landed inside the SageMaker
    container matches the one referenced in the ExperimentConfig.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_etl_v2_panel(
    panel_path: str | Path,
    mask_path: str | Path | None = None,
    *,
    date_column: str = "date",
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load the ETL-v2 transformed-balanced panel + optional imputation mask.

    The ETL writes wide parquet files with a ``date`` column followed by one
    column per series. Both panel and mask share the same column set. Panel
    values are already stationary-transformed (per FRED-MD tcodes), outliers
    removed, and EM-PCA imputed; the mask is Boolean (True = imputed).

    Args:
        panel_path: Local path to ``fred_md_transformed_balanced_*.parquet``.
        mask_path: Optional path to ``fred_md_mask_balanced_*.parquet``.
        date_column: Column to set as DatetimeIndex.
        expected_sha256: When provided, raise ``ValueError`` if the panel
            file's hash does not match. Q5 Tier 3 lineage assertion.

    Returns:
        Tuple of (panel_df, mask_df). ``mask_df`` is None when mask_path is None.
    """
    if expected_sha256:
        actual = file_sha256(panel_path)
        if actual != expected_sha256:
            raise ValueError(
                f"data_sha256 mismatch for {panel_path}: "
                f"expected {expected_sha256}, got {actual}. "
                "Refusing to train on a different dataset than was registered."
            )
    panel = pd.read_parquet(panel_path)
    panel[date_column] = pd.to_datetime(panel[date_column])
    panel = panel.set_index(date_column).sort_index()

    if mask_path is None:
        return panel, None

    mask = pd.read_parquet(mask_path)
    mask[date_column] = pd.to_datetime(mask[date_column])
    mask = mask.set_index(date_column).sort_index()
    # Align mask columns to panel; missing -> assume not imputed.
    mask = mask.reindex(columns=panel.columns, fill_value=False).astype(bool)
    return panel, mask


def drop_high_nan_series(
    df: pd.DataFrame,
    threshold: float = 0.1,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop series (columns) that exceed a NaN fraction threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowable fraction of NaN values per column.

    Returns:
        A tuple of (cleaned_df, dropped_names) where dropped_names lists the
        columns that were removed.
    """
    nan_frac = df.isna().mean()
    to_drop = nan_frac[nan_frac > threshold].index.tolist()

    if to_drop:
        logger.info("Dropping %d series with >%.0f%% NaN: %s", len(to_drop), threshold * 100, to_drop)

    cleaned = df.drop(columns=to_drop)
    return cleaned, to_drop


def forward_fill_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaN values, then backfill any remaining at the start.

    Args:
        df: DataFrame that may contain NaN values.

    Returns:
        DataFrame with no NaN values.
    """
    filled = df.ffill().bfill()
    remaining = filled.isna().sum().sum()
    if remaining > 0:
        logger.warning("%d NaN values remain after forward/back-fill", remaining)
    return filled


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train / validation / test sets.

    The splits are inclusive of the boundary dates:
        - train: index <= train_end
        - val:   train_end < index <= val_end
        - test:  index > val_end

    Args:
        df: DataFrame with a DatetimeIndex.
        train_end: Last date (inclusive) for the training set.
        val_end: Last date (inclusive) for the validation set.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df.loc[df.index <= train_end_ts]
    val = df.loc[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test = df.loc[df.index > val_end_ts]

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training set only.

    Args:
        train: Training DataFrame (used to compute mean and std).

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(train.values)
    return scaler


def scale_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform train / val / test splits using a pre-fitted scaler.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        scaler: A StandardScaler already fitted on training data.

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled) as numpy arrays.
    """
    return (
        scaler.transform(train.values),
        scaler.transform(val.values),
        scaler.transform(test.values),
    )


def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Create rolling windows from a 2-D array of time series.

    Args:
        data: Array of shape (n_timesteps, n_features).
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.

    Returns:
        Array of shape (n_windows, window_size, n_features).

    Raises:
        ValueError: If data has fewer rows than window_size.
    """
    n_timesteps, n_features = data.shape
    if n_timesteps < window_size:
        msg = (
            f"Not enough timesteps ({n_timesteps}) for window_size={window_size}"
        )
        raise ValueError(msg)

    n_windows = (n_timesteps - window_size) // stride + 1
    windows = np.empty((n_windows, window_size, n_features), dtype=data.dtype)

    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start : start + window_size]

    return windows
