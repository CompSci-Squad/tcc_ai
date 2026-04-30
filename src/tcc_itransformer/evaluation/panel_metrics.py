"""Locked 7-metric evaluation panel.

Implements the LOCKED panel from ``docs/pre_analysis_plan.md`` §2 in a single
helper so every results-producing script (``run_clustering_ablation.py``,
``run_baselines.py``, ``run_hdphmm_baseline.py``, ``run_single.py``) emits
the same keys in the same order.

Canonical column order (TEST split):

    1. dbcv                       — primary; HDBSCAN validity_index, NaN for
                                    KMeans/HMM hard assignments
    2. n_clusters_test            — count of unique non-noise labels on TEST
    3. noise_fraction_test        — share of TEST points labelled -1
    4. nber_f1                    — Hungarian-on-VAL frozen → applied on TEST
    5. nber_f1_legacy_maxF1       — biased post-hoc max-F1 (kept as bias witness)
    6. bai_perron_f1              — agreement of cluster transitions with
                                    Bai-Perron breaks on PC1 of the TEST signal
    7. crisis_window_coverage     — fraction of canonical crises (dotcom, GFC,
                                    COVID) whose dominant cluster matches the
                                    NBER-recession-mapped one

Use :func:`compute_panel_metrics` from any caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_itransformer.data.external_labels import load_usrec
from tcc_itransformer.evaluation.regime_validation import (
    CANONICAL_CRISIS_WINDOWS,
    bai_perron_alignment,
    crisis_window_coverage,
    fit_nber_assignment,
    nber_overlap,
    nber_overlap_frozen,
)

logger = logging.getLogger(__name__)

PANEL_COLUMNS: tuple[str, ...] = (
    "dbcv",
    "n_clusters_test",
    "noise_fraction_test",
    "nber_f1",
    "nber_f1_legacy_maxF1",
    "bai_perron_f1",
    "crisis_window_coverage",
)


def _empty_panel() -> dict[str, float]:
    return {k: float("nan") for k in PANEL_COLUMNS}


def _safe_dbcv(
    Y_test: np.ndarray,
    test_labels: np.ndarray,
    *,
    is_density_clusterer: bool,
) -> float:
    """DBCV on the TEST 2-D embedding.

    Uses ``hdbscan.validity.validity_index``, which implements the DBCV
    score from Moulavi et al. 2014. Defined only when ``is_density_clusterer``
    is True and there are ≥2 non-noise clusters with ≥2 points each.
    """
    if not is_density_clusterer:
        return float("nan")
    labels = np.asarray(test_labels)
    unique = [c for c in set(labels.tolist()) if c != -1]
    if len(unique) < 2:
        return float("nan")
    # validity_index requires at least 2 points per cluster
    if any(int(np.sum(labels == c)) < 2 for c in unique):
        return float("nan")
    try:
        from hdbscan.validity import validity_index

        X = np.ascontiguousarray(Y_test, dtype=np.float64)
        return float(validity_index(X, labels.astype(np.intp)))
    except Exception as exc:  # pragma: no cover
        logger.warning("validity_index failed: %s", exc)
        return float("nan")


def _bai_perron(test_labels: np.ndarray, signal_2d_or_1d: np.ndarray) -> float:
    """Bai-Perron F1 on PC1 of the supplied TEST signal."""
    arr = np.asarray(signal_2d_or_1d, dtype=float)
    if arr.ndim == 1:
        pc1 = arr
    elif arr.shape[1] == 1:
        pc1 = arr.ravel()
    else:
        from sklearn.decomposition import PCA as _PCA

        pc1 = _PCA(n_components=1).fit_transform(arr).ravel()
    if len(pc1) < 10:
        return float("nan")
    try:
        bp = bai_perron_alignment(test_labels, pc1, penalty=10.0, tolerance=2)
        return float(bp["f1"])
    except Exception as exc:  # pragma: no cover
        logger.warning("bai_perron_alignment failed: %s", exc)
        return float("nan")


def _crisis_coverage_frac(
    test_labels: np.ndarray,
    test_dates: pd.DatetimeIndex,
    nber_assignment: dict[int, int] | None,
) -> float:
    """Fraction of canonical crises whose dominant cluster maps to recession.

    A canonical crisis (dotcom / GFC / COVID) "covered" iff:
      * the window has ≥1 month inside ``test_dates``, AND
      * the dominant cluster in that window is mapped to recession (=1) by
        the frozen NBER assignment fit on VAL.

    If no NBER assignment is available, falls back to "any non-noise dominant
    cluster" — i.e. just the share of crises that ARE inside TEST and
    captured by some cluster.
    """
    cov = crisis_window_coverage(test_labels, test_dates)
    in_test = [name for name, info in cov.items() if info.get("dominant_cluster", -1) != -1]
    if not in_test:
        return float("nan")
    if nber_assignment is None:
        return float(len(in_test) / len(cov))
    n_match = 0
    for name in in_test:
        dom = int(cov[name]["dominant_cluster"])
        if int(nber_assignment.get(dom, 0)) == 1:
            n_match += 1
    return float(n_match / len(cov))


def compute_panel_metrics(
    *,
    val_labels: np.ndarray | None,
    val_dates: pd.DatetimeIndex | None,
    test_labels: np.ndarray,
    test_dates: pd.DatetimeIndex,
    Y_test: np.ndarray,
    test_signal: np.ndarray | None = None,
    usrec_csv: str | Path | None = None,
    is_density_clusterer: bool = False,
) -> dict[str, float]:
    """Compute the locked 7-metric panel for a single (model, clusterer) cell.

    Args:
        val_labels: VAL cluster labels (used only for the Hungarian frozen
            mapping). If None, NBER F1 columns are NaN.
        val_dates: VAL DatetimeIndex matching ``val_labels``.
        test_labels: TEST cluster labels.
        test_dates: TEST DatetimeIndex matching ``test_labels``.
        Y_test: TEST 2-D (or low-dim) embedding used for DBCV.
        test_signal: 1-D or n-D TEST signal for Bai-Perron PC1. Defaults to
            ``Y_test`` when omitted.
        usrec_csv: Path to ``USREC.csv`` snapshot. If None, NBER cols are NaN.
        is_density_clusterer: True for HDBSCAN (DBCV defined); False for
            KMeans / HMM (DBCV is NaN by spec).

    Returns:
        Dict with exactly the 7 PANEL_COLUMNS keys, plus auxiliary ones
        (``nber_assignment``, ``nber_precision``, ``nber_recall``,
        ``crisis_n_canonical_in_test``).
    """
    out: dict[str, float] = _empty_panel()
    test_labels = np.asarray(test_labels)
    test_dates = pd.DatetimeIndex(test_dates)

    out["n_clusters_test"] = float(
        len(set(test_labels.tolist())) - (1 if -1 in test_labels else 0)
    )
    out["noise_fraction_test"] = float(np.mean(test_labels == -1))
    out["dbcv"] = _safe_dbcv(Y_test, test_labels, is_density_clusterer=is_density_clusterer)

    nber_assignment: dict[int, int] | None = None
    if usrec_csv is not None and Path(usrec_csv).exists():
        usrec = load_usrec(Path(usrec_csv))
        if val_labels is not None and val_dates is not None and len(val_labels) > 0:
            try:
                nber_assignment = fit_nber_assignment(
                    np.asarray(val_labels),
                    pd.DatetimeIndex(val_dates),
                    usrec,
                    lead=0,
                    lag=2,
                )
                res = nber_overlap_frozen(
                    test_labels, test_dates, usrec, nber_assignment, lead=0, lag=2,
                )
                out["nber_f1"] = float(res.f1)
                out["nber_precision"] = float(res.precision)
                out["nber_recall"] = float(res.recall)
                out["nber_assignment"] = str(nber_assignment)
            except Exception as exc:  # pragma: no cover
                logger.warning("NBER frozen overlap failed: %s", exc)
        try:
            legacy = nber_overlap(test_labels, test_dates, usrec, lead=0, lag=2)
            out["nber_f1_legacy_maxF1"] = float(legacy.f1)
        except Exception as exc:  # pragma: no cover
            logger.warning("nber_overlap (legacy) failed: %s", exc)

    out["bai_perron_f1"] = _bai_perron(
        test_labels, test_signal if test_signal is not None else Y_test,
    )

    out["crisis_window_coverage"] = _crisis_coverage_frac(
        test_labels, test_dates, nber_assignment,
    )
    cov = crisis_window_coverage(test_labels, test_dates)
    out["crisis_n_canonical_in_test"] = float(
        sum(1 for info in cov.values() if info.get("dominant_cluster", -1) != -1)
    )
    return out


__all__ = ["PANEL_COLUMNS", "compute_panel_metrics"]
