"""Clustering pipeline: adaptive PCA, K-Means, intrinsic metrics, stability."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def fit_adaptive_pca(
    embeddings_train: np.ndarray,
    latent_dim: int,
    variance_threshold: float = 0.9,
    n_max: int = 5,
) -> tuple[PCA, int]:
    """Fit PCA with adaptive component selection.

    n_components = min(latent_dim - 1, components_for_threshold_variance, n_max).
    Fit on train non-overlapping embeddings only.

    Returns:
        Tuple of (fitted PCA object, actual number of components used).
    """
    # First fit a full PCA to determine variance-based component count
    # Guard: n_components must be <= min(n_samples, n_features)
    n_samples = embeddings_train.shape[0]
    max_possible = min(latent_dim - 1, n_max, embeddings_train.shape[1], n_samples - 1)
    if max_possible < 1:
        max_possible = 1

    pca_full = PCA(n_components=max_possible)
    pca_full.fit(embeddings_train)

    # Find number of components for variance threshold
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_var = int(np.searchsorted(cumvar, variance_threshold) + 1)

    n_components = min(latent_dim - 1, n_for_var, n_max, n_samples - 1)
    n_components = max(1, n_components)

    pca = PCA(n_components=n_components)
    pca.fit(embeddings_train)
    return pca, n_components


def apply_pca(embeddings: np.ndarray, pca: PCA) -> np.ndarray:
    """Transform embeddings with fitted PCA. No refit."""
    return pca.transform(embeddings)


def fit_kmeans(
    embeddings_pca: np.ndarray, k: int, random_state: int = 42
) -> KMeans:
    """Fit KMeans with n_init=20 for stability."""
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    km.fit(embeddings_pca)
    return km


def compute_clustering_metrics(
    embeddings_pca: np.ndarray, labels: np.ndarray
) -> dict:
    """Compute intrinsic clustering metrics.

    Returns:
        Dictionary with silhouette, davies_bouldin, and calinski_harabasz scores.
    """
    return {
        "silhouette": float(silhouette_score(embeddings_pca, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings_pca, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings_pca, labels)),
    }


def select_k(
    embeddings_pca: np.ndarray,
    k_range: list[int] | None = None,
) -> dict:
    """Select best K by argmax silhouette on validation set.

    Default k_range = [3, 4, 5].

    Returns:
        Dictionary with best_k and per-k silhouette scores.
    """
    if k_range is None:
        k_range = [3, 4, 5]

    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        pred = km.fit_predict(embeddings_pca)
        scores[k] = float(silhouette_score(embeddings_pca, pred))

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return {"best_k": best_k, "scores": scores}


def select_k_combined(
    embeddings_pca: np.ndarray,
    k_range: list[int] | None = None,
    *,
    random_state: int = 42,
) -> dict:
    """Select best K combining Silhouette (KMeans) with BIC (GMM).

    Pre_projeto §4.4 requires the K-Means baseline to use Silhouette **and**
    BIC computed on a Gaussian Mixture Model fit in parallel — preventing
    cluster-count selection driven by a single criterion.

    Strategy:
        - For each k in k_range, fit KMeans and a GMM (full covariance).
        - Compute silhouette(KMeans labels) and BIC(GMM).
        - Normalize each criterion to [0, 1] (higher is better) and average.
        - best_k = argmax of the combined score.

    Returns:
        Dict with keys: best_k, silhouette (per-k), bic (per-k),
        combined (per-k), and the chosen criterion.
    """
    if k_range is None:
        k_range = [3, 4, 5]

    sil: dict[int, float] = {}
    bic: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(embeddings_pca)
        sil[k] = float(silhouette_score(embeddings_pca, labels))

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            random_state=random_state,
        )
        gmm.fit(embeddings_pca)
        bic[k] = float(gmm.bic(embeddings_pca))

    sil_arr = np.array([sil[k] for k in k_range])
    bic_arr = np.array([bic[k] for k in k_range])

    def _norm(v: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
        if lower_is_better:
            v = -v
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    sil_n = _norm(sil_arr, lower_is_better=False)
    bic_n = _norm(bic_arr, lower_is_better=True)
    combined_arr = 0.5 * sil_n + 0.5 * bic_n
    combined = {k: float(combined_arr[i]) for i, k in enumerate(k_range)}

    best_k = max(combined, key=combined.get)  # type: ignore[arg-type]
    return {
        "best_k": int(best_k),
        "silhouette": sil,
        "bic": bic,
        "combined": combined,
        "criterion": "silhouette+bic_gmm",
    }


def clustering_stability(
    embeddings_pca: np.ndarray,
    k: int,
    n_runs: int = 10,
    random_state: int = 42,
) -> float:
    """Clustering stability via mean Adjusted Rand Index across runs.

    Runs KMeans n_runs times with different random states, computes ARI
    between all pairs of labelings, and returns the mean.
    """
    labelings = []
    for i in range(n_runs):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state + i)
        labelings.append(km.fit_predict(embeddings_pca))

    aris = [
        adjusted_rand_score(labelings[i], labelings[j])
        for i, j in combinations(range(n_runs), 2)
    ]
    return float(np.mean(aris))


def compute_regime_transitions(labels: np.ndarray) -> int:
    """Count the number of regime transitions between consecutive labels.

    A transition occurs when ``labels[i] != labels[i-1]``.
    """
    if len(labels) < 2:
        return 0
    return int(np.sum(labels[1:] != labels[:-1]))
