"""Statistical tests: Kruskal-Wallis, Mann-Whitney, permutation, block bootstrap."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from itertools import combinations

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, norm
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

MIN_BOOTSTRAP_N_EFF = 20


def _bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute BCa bootstrap confidence interval for a statistic.

    Falls back to percentile method if BCa computation fails.
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    theta_hat = statistic_fn(data)

    # Bootstrap distribution
    boot_dist = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_dist[i] = statistic_fn(data[idx])

    # BCa: bias correction
    z0 = norm.ppf(np.mean(boot_dist < theta_hat))

    if not np.isfinite(z0):
        # Fall back to percentile
        alpha = 1.0 - confidence_level
        return (
            float(np.percentile(boot_dist, 100 * alpha / 2)),
            float(np.percentile(boot_dist, 100 * (1 - alpha / 2))),
        )

    # BCa: acceleration via jackknife
    jack_vals = np.zeros(n)
    for i in range(n):
        jack_sample = np.delete(data, i, axis=0)
        jack_vals[i] = statistic_fn(jack_sample)
    jack_mean = jack_vals.mean()
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)

    if den == 0:
        # Fall back to percentile
        alpha = 1.0 - confidence_level
        return (
            float(np.percentile(boot_dist, 100 * alpha / 2)),
            float(np.percentile(boot_dist, 100 * (1 - alpha / 2))),
        )

    a = num / den

    alpha = 1.0 - confidence_level
    z_alpha_lower = norm.ppf(alpha / 2)
    z_alpha_upper = norm.ppf(1 - alpha / 2)

    # BCa adjusted percentiles
    def _bca_percentile(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        adjusted = z0 + numerator / (1 - a * numerator)
        return float(norm.cdf(adjusted) * 100)

    p_lower = _bca_percentile(z_alpha_lower)
    p_upper = _bca_percentile(z_alpha_upper)

    if not (np.isfinite(p_lower) and np.isfinite(p_upper)):
        alpha = 1.0 - confidence_level
        p_lower = 100 * alpha / 2
        p_upper = 100 * (1 - alpha / 2)

    p_lower = np.clip(p_lower, 0.5, 99.5)
    p_upper = np.clip(p_upper, 0.5, 99.5)

    return (
        float(np.percentile(boot_dist, p_lower)),
        float(np.percentile(boot_dist, p_upper)),
    )


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

    # Bootstrap CIs for effect sizes per dimension
    effect_size_cis = np.zeros((n_dims, 2))
    for d in range(n_dims):
        # Stack dim values and labels so bootstrap resamples rows together
        dim_data_with_labels = np.column_stack([embeddings[:, d], labels])

        def _eta_h_for_dim(paired: np.ndarray) -> float:
            vals, lbls = paired[:, 0], paired[:, 1]
            groups = [vals[lbls == lbl] for lbl in unique_labels]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                return 0.0
            h, _ = kruskal(*groups)
            denom = len(vals) - len(unique_labels)
            if denom <= 0:
                return 0.0
            return float(np.clip((h - len(unique_labels) + 1) / denom, 0.0, 1.0))

        if n_samples >= MIN_BOOTSTRAP_N_EFF:
            ci = _bootstrap_ci(dim_data_with_labels, _eta_h_for_dim, n_bootstrap=2000, random_state=42 + d)
            effect_size_cis[d] = ci
        else:
            effect_size_cis[d] = [np.nan, np.nan]

    return {
        "h_stats": h_stats,
        "p_values_raw": p_values_raw,
        "p_values_corrected": p_corrected,
        "rejected": rejected,
        "n_significant": int(np.sum(rejected)),
        "effect_sizes": effect_sizes,
        "effect_size_cis": effect_size_cis,
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

    # Bootstrap CIs for rank-biserial per (pair, dim)
    effect_size_cis = np.full((n_pairs, n_dims, 2), np.nan)
    for pi, (la, lb) in enumerate(pairs):
        group_a = embeddings[labels == la]
        group_b = embeddings[labels == lb]
        n1, n2 = len(group_a), len(group_b)
        if n1 + n2 >= MIN_BOOTSTRAP_N_EFF:
            for d in range(n_dims):
                combined = np.concatenate([group_a[:, d], group_b[:, d]])
                _n1 = n1

                def _rb(data: np.ndarray, _n1: int = _n1) -> float:
                    a_d = data[:_n1]
                    b_d = data[_n1:]
                    if len(a_d) == 0 or len(b_d) == 0:
                        return 0.0
                    u, _ = mannwhitneyu(a_d, b_d, alternative="two-sided")
                    return float(1.0 - (2.0 * u) / (len(a_d) * len(b_d)))

                ci = _bootstrap_ci(combined, _rb, n_bootstrap=2000, random_state=42 + pi * n_dims + d)
                effect_size_cis[pi, d] = ci

    return {
        "pairs": pairs,
        "u_stats": u_stats,
        "p_values_raw": p_values_raw,
        "p_values_corrected": p_corrected,
        "effect_sizes": effect_sizes,
        "effect_size_cis": effect_size_cis,
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

    # Bootstrap CI for Δ_silhouette
    all_diffs = np.concatenate([[observed_diff], null_distribution])
    ci_lower = float(np.percentile(null_distribution, 2.5))
    ci_upper = float(np.percentile(null_distribution, 97.5))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
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

    Uses BCa (bias-corrected and accelerated) method for confidence intervals.
    Falls back to percentile method if BCa computation fails.

    Returns:
        Dictionary with point estimate, CI bounds, and bootstrap distribution.
    """
    n = len(data)
    if n < MIN_BOOTSTRAP_N_EFF:
        logger.warning(
            "n_eff=%d < %d: bootstrap CIs may be unreliable",
            n, MIN_BOOTSTRAP_N_EFF,
        )

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

    # BCa bias correction
    z0 = norm.ppf(np.mean(bootstrap_dist < estimate))

    # BCa acceleration via jackknife (block-delete)
    jack_vals = np.zeros(n)
    for i in range(n):
        jack_sample = np.delete(data, i, axis=0)
        jack_vals[i] = statistic_fn(jack_sample)
    jack_mean = jack_vals.mean()
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)

    alpha = 1.0 - confidence_level

    if den == 0 or not np.isfinite(z0):
        # Fallback to percentile
        ci_lower = float(np.percentile(bootstrap_dist, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_dist, 100 * (1 - alpha / 2)))
    else:
        a = num / den
        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)

        def _adj(z_a: float) -> float:
            num_ = z0 + z_a
            return float(norm.cdf(z0 + num_ / (1 - a * num_)) * 100)

        p_lo = np.clip(_adj(z_lo), 0.5, 99.5)
        p_hi = np.clip(_adj(z_hi), 0.5, 99.5)
        ci_lower = float(np.percentile(bootstrap_dist, p_lo))
        ci_upper = float(np.percentile(bootstrap_dist, p_hi))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bootstrap_dist,
    }
