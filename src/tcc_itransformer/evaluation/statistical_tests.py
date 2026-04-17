"""Statistical tests: Kruskal-Wallis, Mann-Whitney, permutation, block bootstrap."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests


def kruskal_wallis_per_dim(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict:
    """Kruskal-Wallis test per embedding dimension across K clusters.

    BH-FDR correction across d dimensions.

    Returns:
        Dictionary with H statistics, raw/corrected p-values, rejection mask,
        count of significant dimensions, and η²_H effect sizes per dimension.
    """
    n_samples, n_dims = embeddings.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    h_stats = np.zeros(n_dims)
    p_values_raw = np.zeros(n_dims)

    for d in range(n_dims):
        groups = [embeddings[labels == lbl, d] for lbl in unique_labels]
        # Filter out groups with < 1 observation
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            h_stats[d] = 0.0
            p_values_raw[d] = 1.0
        else:
            h_stat, p_val = kruskal(*groups)
            h_stats[d] = h_stat
            p_values_raw[d] = p_val

    # BH-FDR correction
    rejected, p_corrected, _, _ = multipletests(p_values_raw, method="fdr_bh")

    # Effect size: η²_H = (H - k + 1) / (n - k)
    denominator = n_samples - k
    if denominator > 0:
        effect_sizes = np.clip((h_stats - k + 1) / denominator, 0.0, 1.0)
    else:
        effect_sizes = np.zeros(n_dims)

    return {
        "h_stats": h_stats,
        "p_values_raw": p_values_raw,
        "p_values_corrected": p_corrected,
        "rejected": rejected,
        "n_significant": int(np.sum(rejected)),
        "effect_sizes": effect_sizes,
    }


def pairwise_mann_whitney(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict:
    """Pairwise Mann-Whitney U between cluster pairs, per dimension.

    BH-FDR corrected across all (pairs × dims) comparisons.
    Effect size: rank-biserial r = 1 - (2*U) / (n1*n2).

    Returns:
        Dictionary with pair list, U statistics, raw/corrected p-values,
        and rank-biserial effect sizes per (pair, dim).
    """
    unique_labels = np.unique(labels)
    pairs = list(combinations(unique_labels, 2))
    n_pairs = len(pairs)
    n_dims = embeddings.shape[1]

    u_stats = np.zeros((n_pairs, n_dims))
    p_values_raw = np.zeros((n_pairs, n_dims))
    effect_sizes = np.zeros((n_pairs, n_dims))

    for pi, (la, lb) in enumerate(pairs):
        group_a = embeddings[labels == la]
        group_b = embeddings[labels == lb]
        n1, n2 = len(group_a), len(group_b)
        for d in range(n_dims):
            u_stat, p_val = mannwhitneyu(
                group_a[:, d], group_b[:, d], alternative="two-sided"
            )
            u_stats[pi, d] = u_stat
            p_values_raw[pi, d] = p_val
            if n1 * n2 > 0:
                effect_sizes[pi, d] = 1.0 - (2.0 * u_stat) / (n1 * n2)
            else:
                effect_sizes[pi, d] = 0.0

    # BH correction across all comparisons flattened
    p_flat = p_values_raw.ravel()
    _, p_corrected_flat, _, _ = multipletests(p_flat, method="fdr_bh")
    p_corrected = p_corrected_flat.reshape(n_pairs, n_dims)

    return {
        "pairs": pairs,
        "u_stats": u_stats,
        "p_values_raw": p_values_raw,
        "p_values_corrected": p_corrected,
        "effect_sizes": effect_sizes,
    }


def permutation_test_silhouette(
    embeddings_a: np.ndarray,
    labels_a: np.ndarray,
    embeddings_b: np.ndarray,
    labels_b: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict:
    """Permutation test: is silhouette(A) significantly > silhouette(B)?

    Pools both embedding sets, permutes assignment, computes Δsilhouette.

    Returns:
        Dictionary with observed difference, p-value, and null distribution.
    """
    sil_a = silhouette_score(embeddings_a, labels_a)
    sil_b = silhouette_score(embeddings_b, labels_b)
    observed_diff = float(sil_a - sil_b)

    # Pool embeddings and labels
    pooled_emb = np.vstack([embeddings_a, embeddings_b])
    pooled_labels = np.concatenate([labels_a, labels_b])
    n_a = len(embeddings_a)

    rng = np.random.default_rng(random_state)
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm = rng.permutation(len(pooled_emb))
        emb_perm_a = pooled_emb[perm[:n_a]]
        emb_perm_b = pooled_emb[perm[n_a:]]
        labels_perm_a = pooled_labels[perm[:n_a]]
        labels_perm_b = pooled_labels[perm[n_a:]]

        try:
            s_a = silhouette_score(emb_perm_a, labels_perm_a)
            s_b = silhouette_score(emb_perm_b, labels_perm_b)
            null_distribution[i] = s_a - s_b
        except ValueError:
            # Can happen if permutation produces single-cluster split
            null_distribution[i] = 0.0

    p_value = float(np.mean(null_distribution >= observed_diff))
    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "null_distribution": null_distribution,
    }


def moving_block_bootstrap(
    statistic_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    block_length: int,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Moving block bootstrap for temporally correlated data.

    Uses the percentile method for confidence intervals.

    Returns:
        Dictionary with point estimate, CI bounds, and bootstrap distribution.
    """
    n = len(data)
    rng = np.random.default_rng(random_state)
    estimate = float(statistic_fn(data))

    n_blocks = math.ceil(n / block_length)
    max_start = n - block_length

    bootstrap_dist = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        blocks = [data[s : s + block_length] for s in starts]
        sample = np.concatenate(blocks)[:n]
        bootstrap_dist[i] = statistic_fn(sample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_dist, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_dist, 100 * (1 - alpha / 2)))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bootstrap_dist,
    }
