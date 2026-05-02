"""Loaders for external pseudo-label series (NBER USREC, etc.)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_USREC_PATH = Path("data/snapshots/nber_usrec.csv")


def load_usrec(path: str | Path = DEFAULT_USREC_PATH) -> pd.Series:
    """Load NBER USREC recession indicator from FRED CSV snapshot.

    Returns:
        Series of 0/1 indexed by month-start dates, name='USREC'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"USREC snapshot not found at {path}. "
            "Run `tcc data pull-nber` first."
        )
    df = pd.read_csv(path)
    # FRED CSV format: DATE, USREC (or 'observation_date,USREC' on newer exports)
    date_col = next(
        (c for c in df.columns if c.lower() in {"date", "observation_date"}),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).rename(columns=str.upper)
    series = df["USREC"].astype(int)
    series.index = series.index.to_period("M").to_timestamp()
    series.name = "USREC"
    logger.info("Loaded USREC: %d months (%s..%s)", len(series), series.index.min(), series.index.max())
    return series
