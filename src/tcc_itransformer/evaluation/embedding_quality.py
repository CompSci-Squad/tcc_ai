"""Embedding quality diagnostics: collapse detection, effective rank, isotropy."""

from __future__ import annotations

import numpy as np


def reconstruction_mse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Mean squared error between original and reconstructed windows."""
    return float(np.mean((x_true - x_pred) ** 2))


def check_embedding_collapse(
    embeddings: np.ndarray, threshold: float = 1e-4
) -> dict:
    """Check for embedding collapse.

    Returns:
        Dictionary with per-dimension variance, collapsed dimensions, and
        a boolean flag indicating whether any dimension collapsed.
    """
    per_dim_variance = np.var(embeddings, axis=0)
    collapsed_dims = [int(i) for i, v in enumerate(per_dim_variance) if v < threshold]
    return {
        "per_dim_variance": per_dim_variance,
        "collapsed_dims": collapsed_dims,
        "n_collapsed": len(collapsed_dims),
        "is_collapsed": len(collapsed_dims) > 0,
    }


def compute_effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank via entropy of singular value distribution.

    eff_rank = exp(-sum(p_i * log(p_i))) where p_i = sigma_i / sum(sigma).
    """
    _, sigma, _ = np.linalg.svd(embeddings, full_matrices=False)
    sigma = sigma[sigma > 0]
    p = sigma / sigma.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def compute_isotropy(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity. Near 0 = good isotropy.

    Normalizes embeddings, computes pairwise cosines, returns the mean of the
    upper triangle.  For efficiency with large arrays, samples at most 1000
    pairs.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    n = normed.shape[0]
    n_pairs = n * (n - 1) // 2

    if n_pairs <= 1000:
        # Full pairwise cosine via upper triangle
        sim_matrix = normed @ normed.T
        triu_indices = np.triu_indices(n, k=1)
        cosines = sim_matrix[triu_indices]
    else:
        # Sample at most 1000 pairs
        rng = np.random.default_rng(42)
        cosines = np.empty(1000)
        idx_i = rng.integers(0, n, size=1000)
        idx_j = rng.integers(0, n, size=1000)
        # Ensure i != j
        mask = idx_i == idx_j
        idx_j[mask] = (idx_j[mask] + 1) % n
        for p in range(1000):
            cosines[p] = np.dot(normed[idx_i[p]], normed[idx_j[p]])

    return float(np.mean(cosines))
