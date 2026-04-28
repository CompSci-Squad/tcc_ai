"""Density-based clustering with HDBSCAN + DBCV optimization.

Pre_projeto §4.3 Module 3 (principal): HDBSCAN (Campello, Moulavi, Sander 2013)
identifies clusters of arbitrary shape and density, with a noise label for
points that don't belong to any high-density region — appropriate for
macro-financial regimes whose duration and volatility differ.

Pre_projeto §4.4: hyperparameters (min_cluster_size, min_samples) are
optimized via DBCV (Moulavi et al. 2014, Density-Based Clustering Validation),
the principal internal metric for density-based partitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # pragma: no cover
    import hdbscan
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "hdbscan is required. Install with: pip install hdbscan>=0.8.33"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HDBSCANResult:
    """Result of fitting HDBSCAN on a low-dim embedding."""

    labels: np.ndarray  # shape (n_samples,), -1 = noise
    probabilities: np.ndarray  # shape (n_samples,), soft membership
    dbcv: float
    min_cluster_size: int
    min_samples: int
    n_clusters: int
    noise_fraction: float
    clusterer: "hdbscan.HDBSCAN"


def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> HDBSCANResult:
    """Fit HDBSCAN with a single hyperparameter combination.

    Args:
        X: Low-dim embedding (n_samples, n_components).
        min_cluster_size: Minimum size of a cluster.
        min_samples: Conservativeness of clustering; defaults to
            min_cluster_size.
        metric: Distance metric.
        cluster_selection_method: 'eom' (default) or 'leaf'.

    Returns:
        HDBSCANResult with labels, soft probabilities, DBCV score.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(X)
    probs = clusterer.probabilities_

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_fraction = float(np.mean(labels == -1))

    # DBCV: hdbscan exposes relative_validity_ as a fast DBCV approximation.
    dbcv = float(getattr(clusterer, "relative_validity_", float("nan")))

    return HDBSCANResult(
        labels=labels,
        probabilities=probs,
        dbcv=dbcv,
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples) if min_samples is not None else int(min_cluster_size),
        n_clusters=n_clusters,
        noise_fraction=noise_fraction,
        clusterer=clusterer,
    )


def optimize_hdbscan_dbcv(
    X: np.ndarray,
    *,
    min_cluster_sizes: Sequence[int] = (5, 8, 10, 15, 20, 30),
    min_samples_grid: Sequence[int | None] = (None, 1, 5, 10),
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    min_clusters: int = 2,
    max_noise_fraction: float = 0.5,
) -> tuple[HDBSCANResult, list[dict]]:
    """Grid-search (min_cluster_size, min_samples) maximizing DBCV.

    Per pre_projeto §4.4: DBCV is the principal internal validation metric
    for density-based clustering. Configurations producing fewer than
    `min_clusters` clusters or more than `max_noise_fraction` noise points
    are excluded from selection.

    Args:
        X: Low-dim embedding (output of UMAP).
        min_cluster_sizes: Candidate values for min_cluster_size.
        min_samples_grid: Candidate values for min_samples (None = use mcs).
        metric: Distance metric.
        cluster_selection_method: 'eom' or 'leaf'.
        min_clusters: Reject configs with fewer clusters than this.
        max_noise_fraction: Reject configs with more noise than this.

    Returns:
        Tuple of (best HDBSCANResult, full grid log as list of dicts).
    """
    grid_log: list[dict] = []
    best: HDBSCANResult | None = None

    for mcs in min_cluster_sizes:
        for ms in min_samples_grid:
            try:
                res = fit_hdbscan(
                    X,
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric=metric,
                    cluster_selection_method=cluster_selection_method,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("HDBSCAN failed mcs=%d ms=%s: %s", mcs, ms, exc)
                continue

            valid = (
                res.n_clusters >= min_clusters
                and res.noise_fraction <= max_noise_fraction
                and not np.isnan(res.dbcv)
            )
            grid_log.append(
                {
                    "min_cluster_size": mcs,
                    "min_samples": ms if ms is not None else mcs,
                    "n_clusters": res.n_clusters,
                    "noise_fraction": res.noise_fraction,
                    "dbcv": res.dbcv,
                    "valid": valid,
                }
            )
            if valid and (best is None or res.dbcv > best.dbcv):
                best = res

    if best is None:
        # Fallback: pick the config with largest n_clusters among any successful run.
        if not grid_log:
            raise RuntimeError("HDBSCAN grid search produced no results")
        logger.warning(
            "DBCV optimization found no config matching constraints; "
            "falling back to default min_cluster_size=10."
        )
        best = fit_hdbscan(X, min_cluster_size=10, metric=metric)

    logger.info(
        "HDBSCAN best: mcs=%d ms=%d k=%d noise=%.2f DBCV=%.4f",
        best.min_cluster_size,
        best.min_samples,
        best.n_clusters,
        best.noise_fraction,
        best.dbcv,
    )
    return best, grid_log
