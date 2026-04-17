"""Data preprocessing: NaN handling, splitting, scaling, and windowing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


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
