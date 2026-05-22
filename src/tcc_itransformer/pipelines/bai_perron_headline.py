"""Phase C3: Bai-Perron break agreement on 4 headline macro series.

Runs ruptures PELT (multi-break) and statsmodels Zivot-Andrews (unit-root
structural change) on INDPRO, PAYEMS, UNRATE, and T10Y3M.  Compares detected
break dates to cluster-label transitions with ±tolerance-month tolerance.

Requires: ruptures (in main deps), statsmodels (in main deps), fredapi (--extra labels).

Output: results/diagnostics/bai_perron_headline.csv
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series for headline analysis
_HEADLINE_SERIES: dict[str, str] = {
    "INDPRO": "INDPRO",       # Industrial Production Index (level)
    "PAYEMS": "PAYEMS",       # Total Nonfarm Payrolls (level)
    "UNRATE": "UNRATE",       # Unemployment Rate (level)
    "T10Y3M": "T10Y3M",       # 10-Year minus 3-Month Treasury Spread
}


def _pull_fred_series(series_id: str, api_key: str | None = None) -> pd.Series:
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError("Install fredapi: uv sync --extra labels") from exc
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    fred = Fred(api_key=key)
    raw: pd.Series = fred.get_series(series_id, observation_start="1959-01-01")
    raw = raw.dropna()
    raw.index = pd.DatetimeIndex(raw.index).to_period("M").to_timestamp()
    raw.name = series_id
    return raw


def _ruptures_breaks(
    series: np.ndarray,
    penalty: float = 10.0,
    max_breaks: int = 8,
) -> list[int]:
    """Detect structural break indices via PELT with L2 cost."""
    import ruptures as rpt

    arr = series.reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=12).fit(arr)
    bkps = algo.predict(pen=penalty)
    # ruptures includes T as the last breakpoint; remove it
    bkps = [b for b in bkps if b < len(arr)]
    # Cap at max_breaks (take the largest penalty-filtered ones)
    if len(bkps) > max_breaks:
        # re-run with Dynp to get exactly max_breaks
        algo2 = rpt.Dynp(model="l2", min_size=12).fit(arr)
        bkps = [b for b in algo2.predict(n_bkps=max_breaks) if b < len(arr)]
    return sorted(bkps)


def _zivot_andrews_break(series: np.ndarray) -> int | None:
    """Run statsmodels Zivot-Andrews for a single structural break.

    Returns the break index (0-based) or None if the test fails.
    """
    try:
        from statsmodels.tsa.stattools import zivot_andrews
        stat, pvalue, cvt, bpidx, baselag = zivot_andrews(series, trim=0.15)
        if pvalue < 0.10:  # reject null of unit root with no break
            return int(bpidx)
        return None
    except Exception as exc:
        logger.debug("Zivot-Andrews failed: %s", exc)
        return None


def _cluster_transition_indices(labels: np.ndarray) -> list[int]:
    """Return 0-based indices where the cluster label changes."""
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def _match_breaks(
    cluster_changes: list[int],
    break_indices: list[int],
    tolerance: int,
) -> dict[str, float]:
    """Compute precision/recall/F1 between two sets of break dates.

    A cluster change is a TP if any breakpoint is within `tolerance` steps.
    A breakpoint is a TP if any cluster change is within `tolerance` steps.
    """
    if not break_indices:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_cluster_changes": len(cluster_changes),
                "n_breakpoints": 0, "n_matched": 0}
    if not cluster_changes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_cluster_changes": 0,
                "n_breakpoints": len(break_indices), "n_matched": 0}

    tp_pred = sum(
        1 for c in cluster_changes
        if any(abs(c - b) <= tolerance for b in break_indices)
    )
    tp_ref = sum(
        1 for b in break_indices
        if any(abs(b - c) <= tolerance for c in cluster_changes)
    )
    precision = tp_pred / len(cluster_changes)
    recall = tp_ref / len(break_indices)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_cluster_changes": len(cluster_changes),
        "n_breakpoints": len(break_indices),
        "n_matched": tp_pred,
    }


def run_bai_perron_headline(
    clustering_parquet: Path,
    output: Path,
    penalty: float = 10.0,
    tolerance: int = 3,
    fred_api_key: str | None = None,
) -> pd.DataFrame:
    """Run C3 Bai-Perron headline analysis.

    For each of INDPRO, PAYEMS, UNRATE, T10Y3M:
    1. Pull series from FRED.
    2. Align to the test-set date range.
    3. Detect structural breaks via ruptures PELT.
    4. Run Zivot-Andrews for single-break detection.
    5. Compare break dates to cluster-label transitions with ±tolerance months.
    6. Report precision / recall / F1.

    Args:
        clustering_parquet: Parquet with 'date' (month-start) and 'label'.
        output: Output CSV path.
        penalty: PELT penalty parameter.
        tolerance: Month tolerance for break-date matching.
        fred_api_key: FRED API key (falls back to FRED_API_KEY env var).

    Returns:
        DataFrame with one row per headline series.
    """
    # ------------------------------------------------------------------ #
    # Load clustering result
    # ------------------------------------------------------------------ #
    clust_df = pd.read_parquet(clustering_parquet)
    clust_df["date"] = pd.to_datetime(clust_df["date"])
    clust_df = clust_df.sort_values("date").reset_index(drop=True)
    labels = clust_df["label"].to_numpy()
    dates = pd.DatetimeIndex(clust_df["date"])
    cluster_changes = _cluster_transition_indices(labels)
    logger.info("Cluster transitions (n=%d): %s", len(cluster_changes), cluster_changes)

    date_start = dates.min()
    date_end = dates.max()

    rows = []
    for series_name, fred_id in _HEADLINE_SERIES.items():
        logger.info("Pulling %s from FRED …", fred_id)
        try:
            raw = _pull_fred_series(fred_id, api_key=fred_api_key)
        except Exception as exc:
            logger.error("Failed to pull %s: %s", fred_id, exc)
            rows.append({"series": series_name, "error": str(exc)})
            continue

        # Align to test period
        mask = (raw.index >= date_start) & (raw.index <= date_end)
        aligned = raw[mask].copy()
        if len(aligned) < 24:
            logger.warning("%s has only %d obs in test period — skipping", series_name, len(aligned))
            rows.append({"series": series_name, "error": "too_few_obs"})
            continue

        arr = aligned.to_numpy(dtype=float)

        # Ruptures PELT multi-break
        try:
            bkps_pelt = _ruptures_breaks(arr, penalty=penalty)
        except Exception as exc:
            logger.warning("PELT failed for %s: %s", series_name, exc)
            bkps_pelt = []

        # Zivot-Andrews single-break
        za_bkp = _zivot_andrews_break(arr)

        # Adjust cluster-change indices to the aligned date range offset
        # (cluster dates start at date_start; offset = 0 for test-period parquets)
        aligned_start_idx = dates.searchsorted(date_start)
        adjusted_changes = [c - aligned_start_idx for c in cluster_changes
                            if 0 <= (c - aligned_start_idx) < len(arr)]

        pelt_match = _match_breaks(adjusted_changes, bkps_pelt, tolerance)
        za_match = _match_breaks(
            adjusted_changes,
            [za_bkp] if za_bkp is not None else [],
            tolerance,
        )

        rows.append({
            "series": series_name,
            "n_obs": len(aligned),
            "pelt_n_breaks": len(bkps_pelt),
            "pelt_break_dates": str([str(aligned.index[b].date()) for b in bkps_pelt]),
            "pelt_precision": pelt_match["precision"],
            "pelt_recall": pelt_match["recall"],
            "pelt_f1": pelt_match["f1"],
            "pelt_n_matched": pelt_match["n_matched"],
            "za_break_date": str(aligned.index[za_bkp].date()) if za_bkp is not None else "none",
            "za_f1": za_match["f1"],
            "n_cluster_changes": pelt_match["n_cluster_changes"],
            "penalty": penalty,
            "tolerance_months": tolerance,
        })
        logger.info(
            "%s | PELT breaks=%d F1=%.3f | ZA break=%s F1=%.3f",
            series_name, len(bkps_pelt), pelt_match["f1"],
            rows[-1]["za_break_date"], za_match["f1"],
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Bai-Perron headline analysis written to %s", output)
    return df
