"""Stationarity validation via ADF and KPSS in parallel.

Per pre_projeto_tcc.md §4.2: "validadas por testes ADF e KPSS aplicados
em paralelo" — the tcode-transformed panel must be confirmed stationary
before windowing and encoding.

Decision rule (Kwiatkowski et al., 1992):
    - ADF rejects H0 of unit root  AND  KPSS fails to reject H0 of stationarity
      => series is stationary.
    - Otherwise => flagged for review.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationarityResult:
    """Joint ADF+KPSS verdict for a single series."""

    series: str
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    is_stationary: bool


def _safe_adf(x: np.ndarray) -> tuple[float, float]:
    try:
        stat, pvalue, *_ = adfuller(x, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception as exc:  # pragma: no cover - statsmodels edge cases
        logger.warning("ADF failed: %s", exc)
        return float("nan"), float("nan")


def _safe_kpss(x: np.ndarray) -> tuple[float, float]:
    try:
        with warnings.catch_warnings():
            # KPSS warns when p-value is at boundary; this is informational.
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = kpss(x, regression="c", nlags="auto")
        return float(stat), float(pvalue)
    except Exception as exc:  # pragma: no cover
        logger.warning("KPSS failed: %s", exc)
        return float("nan"), float("nan")


def check_series_stationarity(
    x: np.ndarray | pd.Series,
    *,
    name: str = "series",
    alpha: float = 0.05,
) -> StationarityResult:
    """Run ADF and KPSS on one series and return joint verdict.

    Args:
        x: 1-D series (NaNs are dropped).
        name: Series identifier for logging.
        alpha: Significance level.

    Returns:
        StationarityResult with both p-values and joint verdict.
    """
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    if arr.size < 12:
        return StationarityResult(
            series=name,
            adf_stat=float("nan"),
            adf_pvalue=float("nan"),
            kpss_stat=float("nan"),
            kpss_pvalue=float("nan"),
            is_stationary=False,
        )

    adf_stat, adf_p = _safe_adf(arr)
    kpss_stat, kpss_p = _safe_kpss(arr)

    adf_rejects_unit_root = (not np.isnan(adf_p)) and adf_p < alpha
    kpss_keeps_stationary = (not np.isnan(kpss_p)) and kpss_p >= alpha

    return StationarityResult(
        series=name,
        adf_stat=adf_stat,
        adf_pvalue=adf_p,
        kpss_stat=kpss_stat,
        kpss_pvalue=kpss_p,
        is_stationary=adf_rejects_unit_root and kpss_keeps_stationary,
    )


def validate_panel_stationarity(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run joint ADF+KPSS on every column of a panel.

    Args:
        df: Panel with one column per series (index = time).
        alpha: Significance level for both tests.

    Returns:
        DataFrame indexed by series name with columns:
            adf_stat, adf_pvalue, kpss_stat, kpss_pvalue, is_stationary.
    """
    rows = [
        check_series_stationarity(df[col], name=col, alpha=alpha).__dict__
        for col in df.columns
    ]
    out = pd.DataFrame(rows).set_index("series")
    n_pass = int(out["is_stationary"].sum())
    logger.info(
        "Stationarity: %d/%d series pass joint ADF+KPSS at alpha=%.2f",
        n_pass,
        len(out),
        alpha,
    )
    return out
