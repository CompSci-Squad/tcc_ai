"""Regime explainability — Module 4 of pre_projeto_tcc.md §4.3.

Given a fitted clustering and the original (scaled) panel, produces a
structured payload for each window:

    {
        "regime": int,
        "membership": float,        # HDBSCAN soft prob OR normalized distance
        "top_features": [           # k features with largest |z| vs cluster mean
            {"name": str, "z_score": float, "value": float, "regime_mean": float},
            ...
        ],
    }

This payload is the deliverable of pre_projeto §4.3 Module 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureExplanation:
    name: str
    z_score: float
    value: float
    regime_mean: float


@dataclass(frozen=True)
class RegimeExplanation:
    timestamp: pd.Timestamp | None
    regime: int
    membership: float
    top_features: list[FeatureExplanation] = field(default_factory=list)


def _regime_profiles(
    panel: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-regime mean and std (excluding noise -1)."""
    df = panel.copy()
    df["__regime__"] = labels
    df = df[df["__regime__"] != -1]
    grouped = df.groupby("__regime__")
    return grouped.mean(), grouped.std().replace(0.0, 1e-12)


def explain_assignment(
    panel: pd.DataFrame,
    labels: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    top_k: int = 5,
    membership_source: Literal["soft", "distance", "auto"] = "auto",
) -> list[RegimeExplanation]:
    """Build per-timestep regime explanations.

    Args:
        panel: DataFrame (T, F) of (preferably scaled) features aligned with
            labels and embeddings.
        labels: Cluster labels per row of panel.
        probabilities: HDBSCAN soft membership (T,). Required when
            membership_source='soft'.
        centroids: K-Means centroids (k, d) for distance-based membership.
        embeddings: Low-dim representations (T, d) for distance computation.
        top_k: Number of features to return per timestep.
        membership_source: 'soft' uses HDBSCAN probs; 'distance' uses
            normalized distance to centroid; 'auto' picks soft if probs
            available else distance.

    Returns:
        List of RegimeExplanation with length len(panel).
    """
    if len(panel) != len(labels):
        raise ValueError("panel and labels must have same length")

    means, stds = _regime_profiles(panel, labels)

    # Resolve membership strategy
    use_soft = membership_source == "soft" or (
        membership_source == "auto" and probabilities is not None
    )
    use_distance = (not use_soft) and (
        centroids is not None and embeddings is not None
    )

    if use_distance:
        # Normalize distances to [0, 1] within each row's regime distance
        if embeddings.shape[0] != len(panel):
            raise ValueError("embeddings rows must match panel rows")
        dists_to_own = np.full(len(panel), np.nan, dtype=float)
        for i, lab in enumerate(labels):
            if lab == -1 or lab >= centroids.shape[0]:
                continue
            d = np.linalg.norm(embeddings[i] - centroids[lab])
            dists_to_own[i] = d
        # Normalize: 1 = at centroid, 0 = furthest
        d_max = np.nanmax(dists_to_own) if np.isfinite(np.nanmax(dists_to_own)) else 1.0
        membership_arr = 1.0 - (dists_to_own / d_max if d_max > 0 else dists_to_own)
    elif use_soft:
        if probabilities is None or len(probabilities) != len(panel):
            raise ValueError("probabilities required and must match panel length")
        membership_arr = np.asarray(probabilities, dtype=float)
    else:
        membership_arr = np.full(len(panel), float("nan"))

    timestamps = panel.index if isinstance(panel.index, pd.DatetimeIndex) else [None] * len(panel)
    feature_names = list(panel.columns)
    out: list[RegimeExplanation] = []

    for i in range(len(panel)):
        lab = int(labels[i])
        if lab == -1 or lab not in means.index:
            out.append(
                RegimeExplanation(
                    timestamp=timestamps[i] if timestamps[i] is not None else None,
                    regime=lab,
                    membership=float(membership_arr[i]) if not np.isnan(membership_arr[i]) else 0.0,
                    top_features=[],
                )
            )
            continue

        x = panel.iloc[i].to_numpy(dtype=float)
        mu = means.loc[lab].to_numpy(dtype=float)
        sd = stds.loc[lab].to_numpy(dtype=float)
        z = (x - mu) / sd
        idx = np.argsort(-np.abs(z))[:top_k]
        feats = [
            FeatureExplanation(
                name=feature_names[j],
                z_score=float(z[j]),
                value=float(x[j]),
                regime_mean=float(mu[j]),
            )
            for j in idx
        ]
        out.append(
            RegimeExplanation(
                timestamp=timestamps[i] if timestamps[i] is not None else None,
                regime=lab,
                membership=float(membership_arr[i]) if not np.isnan(membership_arr[i]) else 0.0,
                top_features=feats,
            )
        )
    return out


def explanations_to_frame(explanations: list[RegimeExplanation]) -> pd.DataFrame:
    """Flatten explanations to a long-format DataFrame for export/MLflow."""
    rows = []
    for e in explanations:
        if not e.top_features:
            rows.append(
                {
                    "timestamp": e.timestamp,
                    "regime": e.regime,
                    "membership": e.membership,
                    "rank": -1,
                    "feature": None,
                    "z_score": None,
                    "value": None,
                    "regime_mean": None,
                }
            )
            continue
        for rank, f in enumerate(e.top_features):
            rows.append(
                {
                    "timestamp": e.timestamp,
                    "regime": e.regime,
                    "membership": e.membership,
                    "rank": rank,
                    "feature": f.name,
                    "z_score": f.z_score,
                    "value": f.value,
                    "regime_mean": f.regime_mean,
                }
            )
    return pd.DataFrame(rows)
