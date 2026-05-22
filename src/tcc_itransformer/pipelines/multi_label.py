"""Phase C1: Multi-label validation panel (Chauvet-Piger, Sahm, CFNAI-MA3, OECD CLI).

Requires the `labels` optional extra:
    uv sync --extra labels

Expected output: results/diagnostics/multi_label_panel.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_multi_label(
    clustering_parquet: Path,
    usrec_csv: Path,
    output: Path,
    fred_api_key: str | None = None,
    val_parquet: Path | None = None,
    assignment: dict | None = None,
) -> pd.DataFrame:
    """Run the multi-label validation panel for a given clustering result.

    Args:
        clustering_parquet: Parquet with columns 'date' (month-start) and 'label'.
            Should contain TEST-set labels only.
        usrec_csv: NBER USREC snapshot CSV (for val-set assignment).
        output: Path to write the CSV report.
        fred_api_key: FRED API key (falls back to FRED_API_KEY env var).
        val_parquet: Optional path to a parquet with VAL-set labels ('date', 'label').
            When provided, ``fit_nber_assignment`` is called on VAL labels to derive
            the frozen assignment — eliminating the test-set selection leakage of the
            old behaviour.  If None and ``assignment`` is also None, falls back to
            using the labels in ``clustering_parquet`` (legacy biased behaviour).
        assignment: Optional pre-computed frozen assignment dict ``{cluster_id: 0|1}``.
            When provided, ``val_parquet`` and ``fit_nber_assignment`` are skipped.
            Takes precedence over ``val_parquet``.

    Returns:
        DataFrame with one row per external label.
    """
    from tcc_itransformer.data.external_labels import load_all_external_labels, load_usrec
    from tcc_itransformer.evaluation.regime_validation import (
        fit_nber_assignment,
        multi_label_overlap,
    )

    # ------------------------------------------------------------------ #
    # Load clustering result (TEST labels)
    # ------------------------------------------------------------------ #
    clust_df = pd.read_parquet(clustering_parquet)
    if "date" not in clust_df.columns or "label" not in clust_df.columns:
        raise ValueError(
            f"Expected 'date' and 'label' columns in {clustering_parquet}, "
            f"got {clust_df.columns.tolist()}"
        )
    clust_df["date"] = pd.to_datetime(clust_df["date"])
    clust_df = clust_df.sort_values("date").reset_index(drop=True)

    labels = clust_df["label"].to_numpy()
    dates = pd.DatetimeIndex(clust_df["date"])

    # ------------------------------------------------------------------ #
    # Derive frozen assignment — using VAL labels when available (locked protocol)
    # ------------------------------------------------------------------ #
    usrec = load_usrec(usrec_csv)

    if assignment is not None:
        # Pre-computed assignment passed in directly — locked protocol path.
        logger.info("Frozen assignment (pre-computed, locked protocol): %s", assignment)
    elif val_parquet is not None and val_parquet.exists():
        # VAL parquet provided — derive assignment from VAL (no test leakage).
        val_df = pd.read_parquet(val_parquet)
        val_df["date"] = pd.to_datetime(val_df["date"])
        val_df = val_df.sort_values("date").reset_index(drop=True)
        val_labels = val_df["label"].to_numpy()
        val_dates = pd.DatetimeIndex(val_df["date"])
        assignment = fit_nber_assignment(val_labels, val_dates, usrec)
        logger.info("Frozen assignment (from VAL, locked): %s", assignment)
    else:
        # Legacy: use the clustering_parquet labels (TEST) — biased, kept for backward compat.
        assignment = fit_nber_assignment(labels, dates, usrec)
        logger.warning(
            "Frozen assignment derived from TEST labels (legacy biased behaviour). "
            "Pass val_parquet= for the locked protocol."
        )

    # ------------------------------------------------------------------ #
    # Pull external label series from FRED
    # ------------------------------------------------------------------ #
    logger.info("Pulling external label series from FRED …")
    label_series = load_all_external_labels(api_key=fred_api_key)

    # ------------------------------------------------------------------ #
    # Compute per-label overlap metrics
    # ------------------------------------------------------------------ #
    result_df = multi_label_overlap(labels, dates, label_series, assignment)

    # ------------------------------------------------------------------ #
    # Annotate with metadata
    # ------------------------------------------------------------------ #
    result_df.insert(0, "clustering_file", clustering_parquet.name)

    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    logger.info("Multi-label panel written to %s:\n%s", output, result_df.to_string())

    return result_df
