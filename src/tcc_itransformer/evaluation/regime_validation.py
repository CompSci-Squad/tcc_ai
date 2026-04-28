"""Regime validation: external pseudo-labels + econometric interpretability.

Implements pre_projeto_tcc.md §4.4 validation layers 2 and 3:

Layer 2 (external pseudo-labels):
    - nber_overlap: hit-rate of clusters vs NBER USREC recession indicator
      with configurable lead/lag tolerance.
    - bai_perron_alignment: agreement between cluster transitions and
      structural breakpoints detected by Bai-Perron (Bai & Perron 2003).
    - crisis_window_coverage: fraction of canonical crisis months
      (dot-com 2001, GFC 2007-09, COVID 2020) captured by minority clusters.

Layer 3 (econometric interpretability):
    - regime_conditional_moments: per-regime mean / std / correlations.
    - transition_matrix: empirical P[i -> j] between consecutive regimes.
    - regime_durations: mean / median run-length per regime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canonical crisis windows (pre_projeto §4.4)
# --------------------------------------------------------------------------- #
CANONICAL_CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "dotcom": ("2001-03-01", "2001-11-30"),
    "gfc": ("2007-12-01", "2009-06-30"),
    "covid": ("2020-02-01", "2020-04-30"),
}


# =========================================================================== #
# Layer 2 — external pseudo-labels
# =========================================================================== #
@dataclass(frozen=True)
class NBEROverlapResult:
    precision: float
    recall: float
    f1: float
    matched_cluster: int
    n_recession_months: int
    n_predicted_months: int


def nber_overlap(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    usrec: pd.Series,
    *,
    lead: int = 0,
    lag: int = 2,
) -> NBEROverlapResult:
    """Best-cluster overlap with NBER USREC indicator.

    Selects the single cluster whose F1 against USREC is maximal,
    allowing a [lead, lag] month tolerance window. This handles the
    publication delay of NBER recession dating.

    Args:
        labels: Cluster labels for each window timestamp.
        dates: DatetimeIndex aligned with labels (same length).
        usrec: 0/1 NBER recession series indexed by month.
        lead: Months by which a cluster may lead a recession (>=0).
        lag: Months by which a cluster may lag a recession (>=0).

    Returns:
        NBEROverlapResult for the cluster with maximal F1.
    """
    if len(labels) != len(dates):
        raise ValueError("labels and dates must have same length")

    s_labels = pd.Series(labels, index=pd.DatetimeIndex(dates))
    rec = usrec.reindex(s_labels.index).fillna(0).astype(int)

    # Tolerance: expand recession to a window [t-lead, t+lag]
    rec_expanded = rec.copy()
    for k in range(1, lead + 1):
        rec_expanded = rec_expanded | rec.shift(-k, fill_value=0)
    for k in range(1, lag + 1):
        rec_expanded = rec_expanded | rec.shift(k, fill_value=0)

    n_rec = int(rec.sum())
    best = NBEROverlapResult(0.0, 0.0, 0.0, -1, n_rec, 0)

    for c in sorted(set(labels)):
        if c == -1:  # noise — skip
            continue
        pred = (s_labels == c).astype(int)
        n_pred = int(pred.sum())
        if n_pred == 0:
            continue
        tp = int(((pred == 1) & (rec_expanded == 1)).sum())
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_rec if n_rec else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        if f1 > best.f1:
            best = NBEROverlapResult(
                precision=precision,
                recall=recall,
                f1=f1,
                matched_cluster=int(c),
                n_recession_months=n_rec,
                n_predicted_months=n_pred,
            )
    return best


def bai_perron_alignment(
    labels: np.ndarray,
    series: np.ndarray,
    *,
    n_breaks: int | None = None,
    penalty: float = 10.0,
    tolerance: int = 2,
) -> dict[str, float]:
    """Align cluster transitions with Bai-Perron structural breaks.

    Uses the `ruptures` library (Pelt + l2 cost) as a Bai-Perron-style
    multiple-breakpoint detector. Two transitions are considered aligned
    if they fall within `tolerance` time steps.

    Args:
        labels: Cluster labels per timestep.
        series: 1-D reference series (e.g. first principal component of
            the panel) on which to detect structural breaks.
        n_breaks: If provided, force this number of breaks (uses Dynp);
            otherwise use Pelt with `penalty`.
        penalty: PELT penalty (only used when n_breaks is None).
        tolerance: Number of timesteps allowed between cluster change and
            breakpoint to count as a hit.

    Returns:
        Dict with precision, recall, f1, n_cluster_changes, n_breakpoints.
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ruptures is required. Install with: pip install ruptures>=1.1.9"
        ) from exc

    series = np.asarray(series, dtype=float).reshape(-1, 1)
    if n_breaks is not None:
        algo = rpt.Dynp(model="l2").fit(series)
        bkps = algo.predict(n_bkps=int(n_breaks))
    else:
        algo = rpt.Pelt(model="l2").fit(series)
        bkps = algo.predict(pen=penalty)
    # ruptures returns endpoints; drop the trailing T
    bkps = [b for b in bkps if b < len(series)]

    cluster_changes = [
        i for i in range(1, len(labels)) if labels[i] != labels[i - 1]
    ]

    matched_changes = sum(
        1
        for c in cluster_changes
        if any(abs(c - b) <= tolerance for b in bkps)
    )
    matched_breaks = sum(
        1
        for b in bkps
        if any(abs(c - b) <= tolerance for c in cluster_changes)
    )
    precision = matched_changes / len(cluster_changes) if cluster_changes else 0.0
    recall = matched_breaks / len(bkps) if bkps else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_cluster_changes": len(cluster_changes),
        "n_breakpoints": len(bkps),
    }


def crisis_window_coverage(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    windows: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-crisis distribution of cluster assignments.

    For each canonical crisis window, returns the fraction of months
    covered by each cluster — used to inspect whether the partition
    isolates well-known stress periods.

    Returns:
        {crisis_name: {cluster_id: fraction, ..., 'dominant_cluster': int}}
    """
    windows = windows or CANONICAL_CRISIS_WINDOWS
    s_labels = pd.Series(labels, index=pd.DatetimeIndex(dates))
    out: dict[str, dict[str, float]] = {}

    for name, (start, end) in windows.items():
        mask = (s_labels.index >= pd.Timestamp(start)) & (
            s_labels.index <= pd.Timestamp(end)
        )
        sub = s_labels[mask]
        if len(sub) == 0:
            out[name] = {"coverage": 0.0, "dominant_cluster": -1}
            continue
        counts = sub.value_counts(normalize=True)
        out[name] = {
            **{f"cluster_{int(k)}": float(v) for k, v in counts.items()},
            "dominant_cluster": int(counts.idxmax()),
            "n_months": int(len(sub)),
        }
    return out


# =========================================================================== #
# Layer 3 — econometric interpretability
# =========================================================================== #
def regime_conditional_moments(
    panel: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Mean and std of each variable conditional on regime.

    Args:
        panel: DataFrame (n_timesteps, n_features) aligned with labels.
        labels: Cluster labels.

    Returns:
        DataFrame indexed by (regime, statistic) with one column per feature.
    """
    if len(panel) != len(labels):
        raise ValueError("panel and labels must have same length")
    df = panel.copy()
    df["__regime__"] = labels
    grouped = df.groupby("__regime__")
    mean = grouped.mean()
    std = grouped.std()
    mean.index = pd.MultiIndex.from_product([mean.index, ["mean"]])
    std.index = pd.MultiIndex.from_product([std.index, ["std"]])
    out = pd.concat([mean, std]).sort_index()
    out.index.names = ["regime", "statistic"]
    return out


def transition_matrix(labels: np.ndarray) -> pd.DataFrame:
    """Empirical Markov transition matrix P[i -> j] between consecutive labels.

    Noise points (-1) are excluded from rows but counted in columns when a
    transition lands on noise. Returns row-normalized probabilities.
    """
    labels = np.asarray(labels)
    states = sorted({int(s) for s in labels if s != -1})
    if not states:
        return pd.DataFrame()

    idx = {s: i for i, s in enumerate(states)}
    M = np.zeros((len(states), len(states)), dtype=float)
    for prev, nxt in zip(labels[:-1], labels[1:]):
        if prev == -1 or nxt == -1:
            continue
        M[idx[int(prev)], idx[int(nxt)]] += 1.0

    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = M / row_sums
    return pd.DataFrame(P, index=states, columns=states)


def regime_durations(labels: np.ndarray) -> pd.DataFrame:
    """Run-length statistics per regime.

    Returns:
        DataFrame indexed by regime with columns:
            n_runs, mean_duration, median_duration, max_duration, total_months.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return pd.DataFrame()

    runs: dict[int, list[int]] = {}
    cur_label = int(labels[0])
    cur_len = 1
    for x in labels[1:]:
        x = int(x)
        if x == cur_label:
            cur_len += 1
        else:
            runs.setdefault(cur_label, []).append(cur_len)
            cur_label = x
            cur_len = 1
    runs.setdefault(cur_label, []).append(cur_len)

    rows = []
    for regime, lens in sorted(runs.items()):
        arr = np.asarray(lens)
        rows.append(
            {
                "regime": regime,
                "n_runs": int(arr.size),
                "mean_duration": float(arr.mean()),
                "median_duration": float(np.median(arr)),
                "max_duration": int(arr.max()),
                "total_months": int(arr.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("regime")
