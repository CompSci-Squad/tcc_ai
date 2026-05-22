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


def _expand_recession(rec: pd.Series, lead: int, lag: int) -> pd.Series:
    """Tolerance window: recession at t marks [t-lead, t+lag] as positive."""
    rec_expanded = rec.copy()
    for k in range(1, lead + 1):
        rec_expanded = rec_expanded | rec.shift(-k, fill_value=0)
    for k in range(1, lag + 1):
        rec_expanded = rec_expanded | rec.shift(k, fill_value=0)
    return rec_expanded


def fit_nber_assignment(
    val_labels: np.ndarray,
    val_dates: pd.DatetimeIndex,
    usrec: pd.Series,
    *,
    lead: int = 0,
    lag: int = 2,
) -> dict[int, int]:
    """Fit a frozen cluster→regime mapping on the VALIDATION split.

    Avoids the post-hoc max-F1 selection bias of :func:`nber_overlap`. Each
    cluster is mapped to 1 (recession) or 0 (expansion) based on the
    majority of its VAL months that fall inside the tolerance-expanded
    NBER window. The mapping is returned and must be applied verbatim to
    the TEST split via :func:`nber_overlap_frozen`.

    Noise points (label ``-1``) are mapped to 0 and excluded from F1.

    Returns:
        Dict ``{cluster_id: 0 or 1}``.
    """
    if len(val_labels) != len(val_dates):
        raise ValueError("val_labels and val_dates must have same length")
    s_labels = pd.Series(val_labels, index=pd.DatetimeIndex(val_dates))
    # Expand on the full monthly USREC series FIRST, then reindex to window dates.
    # Expanding after reindex loses interior recession months that don't land
    # on a window boundary (e.g., 3-month COVID recession missed by W=6 stride).
    rec_full_expanded = _expand_recession(usrec.astype(int), lead, lag)
    rec_expanded = rec_full_expanded.reindex(s_labels.index).fillna(0).astype(int)

    # Base rate of recession across all dates in this split.
    base_rate = float(rec_expanded.mean()) if len(rec_expanded) > 0 else 0.0

    shares: dict[int, float] = {}
    assignment: dict[int, int] = {}
    for c in sorted(set(val_labels)):
        if c == -1:
            assignment[-1] = 0
            continue
        mask = (s_labels == c)
        if int(mask.sum()) == 0:
            assignment[int(c)] = 0
            shares[int(c)] = 0.0
            continue
        share_rec = float(rec_expanded[mask].mean())
        shares[int(c)] = share_rec

    # Assign as recession the cluster(s) whose share is both:
    # (a) highest relative to base rate (enrichment >= 2x) AND
    # (b) at least 0.05 absolute — avoids assigning recession to noise.
    # If no cluster passes (b) but one passes (a), use the best cluster only.
    _ENRICH_FACTOR = 2.0
    _ABS_MIN = 0.05
    enriched = {
        c: s for c, s in shares.items()
        if s >= max(_ABS_MIN, base_rate * _ENRICH_FACTOR)
    }
    if not enriched:
        # Fall back to the single cluster with max recession share if above base_rate.
        best_c = max(shares, key=shares.__getitem__) if shares else None
        if best_c is not None and shares[best_c] > base_rate:
            enriched = {best_c: shares[best_c]}
    for c in sorted(set(val_labels)):
        if c == -1:
            continue
        assignment[int(c)] = 1 if int(c) in enriched else 0
    return assignment


def nber_overlap_frozen(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    usrec: pd.Series,
    assignment: dict[int, int],
    *,
    lead: int = 0,
    lag: int = 2,
) -> NBEROverlapResult:
    """Apply a frozen cluster→regime mapping (from val) to TEST and score.

    The prediction is the union of all clusters whose ``assignment``
    value is 1. Clusters absent from ``assignment`` (new on test) are
    treated as 0. ``matched_cluster`` is set to the *first* cluster
    flagged as recession, or -1 when none.
    """
    if len(labels) != len(dates):
        raise ValueError("labels and dates must have same length")
    s_labels = pd.Series(labels, index=pd.DatetimeIndex(dates))
    rec_full_expanded = _expand_recession(usrec.astype(int), lead, lag)
    rec_expanded = rec_full_expanded.reindex(s_labels.index).fillna(0).astype(int)
    rec = usrec.reindex(s_labels.index).fillna(0).astype(int)

    pred = s_labels.map(lambda c: int(assignment.get(int(c), 0))).astype(int)
    n_pred = int(pred.sum())
    n_rec = int(rec.sum())

    if n_pred == 0 or n_rec == 0:
        return NBEROverlapResult(0.0, 0.0, 0.0, -1, n_rec, n_pred)

    tp = int(((pred == 1) & (rec_expanded == 1)).sum())
    precision = tp / n_pred
    recall = tp / n_rec
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    rec_clusters = sorted(c for c, v in assignment.items() if v == 1 and c != -1)
    matched = rec_clusters[0] if rec_clusters else -1
    return NBEROverlapResult(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        matched_cluster=int(matched),
        n_recession_months=n_rec,
        n_predicted_months=n_pred,
    )


def nber_overlap(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    usrec: pd.Series,
    *,
    lead: int = 0,
    lag: int = 2,
) -> NBEROverlapResult:
    """Best-cluster overlap with NBER USREC indicator (LEGACY).

    .. deprecated::
        This metric has post-hoc cluster-selection bias: the best F1 over
        all clusters is reported, inflating the score. Use
        :func:`fit_nber_assignment` on VAL plus :func:`nber_overlap_frozen`
        on TEST instead. Kept for backward compatibility with notebooks
        and the unit-test fixtures only.
    """
    if len(labels) != len(dates):
        raise ValueError("labels and dates must have same length")

    s_labels = pd.Series(labels, index=pd.DatetimeIndex(dates))
    rec_full_expanded = _expand_recession(usrec.astype(int), lead, lag)
    rec_expanded = rec_full_expanded.reindex(s_labels.index).fillna(0).astype(int)
    rec = usrec.reindex(s_labels.index).fillna(0).astype(int)

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


# =========================================================================== #
# Phase C1 — Multi-label validation panel
# =========================================================================== #

def _binarise_label(series: pd.Series, threshold: float, above: bool = True) -> pd.Series:
    """Convert a continuous indicator to binary using a threshold.

    Args:
        series: Float indicator aligned to month-start DatetimeIndex.
        threshold: Decision boundary.
        above: If True, values >= threshold → 1 (recession). If False, values <= threshold → 1.
    """
    if above:
        return (series >= threshold).astype(int)
    return (series <= threshold).astype(int)


def multi_label_overlap(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    label_series_dict: dict[str, pd.Series],
    assignment: dict[int, int],
    *,
    thresholds: dict[str, tuple[float, bool]] | None = None,
) -> pd.DataFrame:
    """Compute overlap F1/precision/recall between cluster predictions and multiple external labels.

    For each external label series, the function:
    1. Binarises the label series using the provided threshold (or default).
    2. Applies the frozen cluster→recession assignment to get binary predictions.
    3. Computes precision, recall, F1, and AUC-ROC (when the label is float).

    Args:
        labels: Array of cluster labels for the evaluation period.
        dates: DatetimeIndex aligned with ``labels`` (month-start).
        label_series_dict: Mapping of label_name → raw series (float or binary).
            Supported names and their defaults:
              - ``chauvet_piger``: threshold=0.5, above=True
              - ``sahm``:          threshold=0.5, above=True
              - ``cfnai_ma3``:     threshold=-0.70, above=False
              - ``oecd_cli``:      already binary, threshold=0.5, above=True
        assignment: Frozen cluster→{0,1} mapping from validation set.
        thresholds: Optional override for (threshold, above) per label name.

    Returns:
        DataFrame with one row per label and columns:
            label, precision, recall, f1, auc_roc, n_ref_months, n_pred_months, overlap_months.
    """
    _DEFAULT_THRESHOLDS: dict[str, tuple[float, bool]] = {
        "chauvet_piger": (0.5, True),
        "sahm": (0.5, True),
        "cfnai_ma3": (-0.70, False),
        "oecd_cli": (0.5, True),
    }
    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    s_labels = pd.Series(labels, index=pd.DatetimeIndex(dates))
    pred_proba = s_labels.map(lambda c: float(assignment.get(int(c), 0))).astype(float)

    rows = []
    for name, raw_series in label_series_dict.items():
        threshold, above = th.get(name, (0.5, True))

        ref_bin = _binarise_label(raw_series, threshold, above)
        ref_aligned = ref_bin.reindex(s_labels.index).fillna(0).astype(int)
        pred_bin = (pred_proba >= 0.5).astype(int)

        n_ref = int(ref_aligned.sum())
        n_pred = int(pred_bin.sum())
        overlap = int(((pred_bin == 1) & (ref_aligned == 1)).sum())

        precision = overlap / n_pred if n_pred > 0 else 0.0
        recall = overlap / n_ref if n_ref > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # AUC-ROC: balanced accuracy = (TPR + TNR) / 2 of the binary cluster prediction.
        # We use the cluster's binary prediction (pred_bin) as the score, not the raw
        # indicator series. Using the raw series predicts its own threshold (trivially
        # perfect or sign-flipped), which is meaningless as a cluster quality measure.
        auc = float("nan")
        if n_ref > 0 and n_ref < len(ref_aligned):
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(ref_aligned.values, pred_bin.values))
            except Exception:
                pass

        rows.append({
            "label": name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auc_roc": round(auc, 4) if not (auc != auc) else float("nan"),
            "n_ref_months": n_ref,
            "n_pred_months": n_pred,
            "overlap_months": overlap,
            "threshold": threshold,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Majority-vote composite: month is "recession" if ≥2/4 labels agree
        # (or ≥1/N when fewer labels available)
        all_refs = []
        for name, raw_series in label_series_dict.items():
            threshold, above = th.get(name, (0.5, True))
            ref_bin = _binarise_label(raw_series, threshold, above)
            all_refs.append(ref_bin.reindex(s_labels.index).fillna(0).astype(int))
        if all_refs:
            vote_stack = pd.concat(all_refs, axis=1)
            majority_threshold = max(1, len(all_refs) // 2)
            composite_ref = (vote_stack.sum(axis=1) >= majority_threshold).astype(int)
            n_ref_c = int(composite_ref.sum())
            n_pred_c = int(pred_bin.sum())
            overlap_c = int(((pred_bin == 1) & (composite_ref == 1)).sum())
            prec_c = overlap_c / n_pred_c if n_pred_c > 0 else 0.0
            rec_c = overlap_c / n_ref_c if n_ref_c > 0 else 0.0
            f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
            composite_row = {
                "label": "composite_majority_vote",
                "precision": round(prec_c, 4),
                "recall": round(rec_c, 4),
                "f1": round(f1_c, 4),
                "auc_roc": float("nan"),
                "n_ref_months": n_ref_c,
                "n_pred_months": n_pred_c,
                "overlap_months": overlap_c,
                "threshold": float("nan"),
            }
            df = pd.concat([df, pd.DataFrame([composite_row])], ignore_index=True)

    return df
