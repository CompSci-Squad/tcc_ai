"""Loaders for external pseudo-label series (NBER USREC, Chauvet-Piger, Sahm, CFNAI, OECD CLI)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_USREC_PATH = Path("data/snapshots/nber_usrec.csv")

# FRED series IDs for the multi-label validation panel (Phase C1)
_FRED_SERIES: dict[str, str] = {
    "chauvet_piger": "RECPROUSM156N",   # Chauvet-Piger MSI recession prob [0,1]
    "sahm": "SAHMREALTIME",              # Sahm rule real-time indicator (u-rate diff)
    "cfnai_ma3": "CFNAIMA3",             # CFNAI 3-month MA (activity index)
    "oecd_cli": "USALOLITOAASTSAM",      # OECD CLI amplitude-adjusted SA (US)
}


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


def _pull_fred_series(series_id: str, api_key: str | None = None) -> pd.Series:
    """Pull a monthly FRED series via fredapi. Returns a float Series indexed by month-start."""
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "fredapi is required. Install with: uv sync --extra labels"
        ) from exc
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY environment variable not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set it via: export FRED_API_KEY=your_key"
        )
    fred = Fred(api_key=key)
    raw: pd.Series = fred.get_series(series_id, observation_start="1959-01-01")
    raw = raw.dropna()
    raw.index = pd.DatetimeIndex(raw.index).to_period("M").to_timestamp()
    raw.name = series_id
    logger.info("Pulled %s from FRED: %d obs (%s..%s)", series_id, len(raw), raw.index.min(), raw.index.max())
    return raw


def load_chauvet_piger(api_key: str | None = None) -> pd.Series:
    """Load Chauvet-Piger recession probability (RECPROUSM156N).

    Threshold ≥0.5 → recession month (binary label).
    Returns float Series [0, 1] for direct AUC computation plus binarisation.
    """
    raw = _pull_fred_series(_FRED_SERIES["chauvet_piger"], api_key)
    raw = raw.clip(0.0, 1.0)
    raw.name = "chauvet_piger"
    return raw


def load_sahm(api_key: str | None = None) -> pd.Series:
    """Load Sahm Rule real-time indicator (SAHMREALTIME).

    Sahm ≥0.5 pp → recession signal (binary label).
    Returns raw float indicator; caller thresholds at 0.5.
    """
    raw = _pull_fred_series(_FRED_SERIES["sahm"], api_key)
    raw.name = "sahm"
    return raw


def load_cfnai_ma3(api_key: str | None = None) -> pd.Series:
    """Load CFNAI 3-month moving average (CFNAIMA3).

    CFNAI-MA3 ≤ -0.70 → increasing probability of recession (binary label).
    Returns raw float; caller thresholds at -0.70.
    """
    raw = _pull_fred_series(_FRED_SERIES["cfnai_ma3"], api_key)
    raw.name = "cfnai_ma3"
    return raw


def load_oecd_cli(api_key: str | None = None) -> pd.Series:
    """Load OECD CLI amplitude-adjusted SA for USA (USALOLITOAASTSAM).

    Uses simplified peak-trough detection: below 100 and declining → contraction.
    Returns binary 0/1 Series (0=expansion, 1=contraction).
    """
    from scipy.signal import argrelextrema

    raw = _pull_fred_series(_FRED_SERIES["oecd_cli"], api_key)
    arr = raw.to_numpy()

    # Find local peaks and troughs (1-month neighbourhood)
    peaks = set(int(i) for i in argrelextrema(arr, np.greater, order=3)[0])
    troughs = set(int(i) for i in argrelextrema(arr, np.less, order=3)[0])

    # Build contraction mask: from trough to next peak = expansion; peak to trough = contraction
    label = np.zeros(len(arr), dtype=int)
    turn_points = sorted(
        [(i, "peak") for i in peaks] + [(i, "trough") for i in troughs],
        key=lambda x: x[0],
    )
    in_contraction = arr[0] < 100.0
    prev_idx = 0
    for idx, kind in turn_points:
        label[prev_idx:idx] = int(in_contraction)
        in_contraction = kind == "peak"  # after a peak → enters contraction
        prev_idx = idx
    label[prev_idx:] = int(in_contraction)

    result = pd.Series(label, index=raw.index, name="oecd_cli")
    logger.info(
        "OECD CLI contraction months: %d / %d", int(result.sum()), len(result)
    )
    return result


def load_all_external_labels(api_key: str | None = None) -> dict[str, pd.Series]:
    """Pull all four external label series in one call.

    Returns:
        Dict mapping label name → Series (float or binary 0/1).
        Thresholds:
          - chauvet_piger: float [0,1], threshold 0.5
          - sahm:          float, threshold 0.5
          - cfnai_ma3:     float, threshold -0.70
          - oecd_cli:      binary 0/1 (contraction detection already applied)
    """
    return {
        "chauvet_piger": load_chauvet_piger(api_key),
        "sahm": load_sahm(api_key),
        "cfnai_ma3": load_cfnai_ma3(api_key),
        "oecd_cli": load_oecd_cli(api_key),
    }
