"""Non-linear dimensionality reduction via UMAP (pre_projeto §4.3 Module 2).

UMAP (McInnes, Healy, Melville 2018) preserves local non-linear structure
of the iTransformer embedding space, projecting the latent vectors onto a
low-dimensional manifold suitable for density-based clustering with HDBSCAN.

This module is the principal dim-reduction step of the pipeline; the PCA
in `baseline_clustering.py` (renamed from `clustering.py`) remains only as
a baseline comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - import-time guard
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "umap-learn is required for dim_reduction. "
        "Install with: pip install umap-learn>=0.5.5"
    ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UMAPConfig:
    """UMAP hyperparameters (defaults chosen for monthly macro panels)."""

    n_components: int = 5
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "euclidean"
    random_state: int = 42


def fit_umap(
    embeddings_train: np.ndarray,
    cfg: UMAPConfig | None = None,
) -> "umap.UMAP":
    """Fit UMAP on TRAIN embeddings only (no leakage).

    Args:
        embeddings_train: Encoder output for non-overlapping train windows,
            shape (n_train_windows, latent_dim).
        cfg: UMAP hyperparameters; defaults to UMAPConfig().

    Returns:
        Fitted umap.UMAP reducer.
    """
    cfg = cfg or UMAPConfig()
    n_samples = embeddings_train.shape[0]
    # n_neighbors must be < n_samples
    n_neighbors = min(cfg.n_neighbors, max(2, n_samples - 1))
    n_components = min(cfg.n_components, max(1, n_samples - 2))

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=cfg.random_state,
    )
    reducer.fit(embeddings_train)
    logger.info(
        "UMAP fitted: n_samples=%d -> n_components=%d, n_neighbors=%d",
        n_samples,
        n_components,
        n_neighbors,
    )
    return reducer


def apply_umap(embeddings: np.ndarray, reducer: "umap.UMAP") -> np.ndarray:
    """Project embeddings with a fitted UMAP reducer."""
    return reducer.transform(embeddings)
