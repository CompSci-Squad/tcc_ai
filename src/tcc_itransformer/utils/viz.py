"""Visualization utilities: scree plots, cluster scatter, regime timelines, bootstrap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")


def plot_scree(pca: PCA, save_path: Path | None = None) -> Figure:
    """Scree plot of PCA explained variance ratio."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    n = len(pca.explained_variance_ratio_)
    components = np.arange(1, n + 1)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    ax.bar(components, pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.plot(components, cumvar, "o-", color="red", label="Cumulative")
    ax.axhline(y=0.9, linestyle="--", color="gray", alpha=0.7, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.set_xticks(components)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_cluster_scatter(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """2D scatter plot of embeddings colored by cluster labels."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f"Cluster {lbl}",
            alpha=0.7,
            s=30,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title or "Cluster Scatter Plot")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_regime_timeline(
    dates: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """Timeline showing regime (cluster) assignments over time."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    colors = [color_map[lbl] for lbl in labels]
    ax.scatter(dates, labels, c=colors, s=15, alpha=0.8)

    for lbl in unique_labels:
        mask = labels == lbl
        ax.fill_between(
            dates,
            lbl - 0.3,
            lbl + 0.3,
            where=mask,
            alpha=0.15,
            color=color_map[lbl],
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_title(title or "Regime Timeline")
    ax.set_yticks(unique_labels)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_bootstrap_distribution(
    bootstrap_dist: np.ndarray,
    ci: tuple[float, float],
    observed: float,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """Histogram of bootstrap distribution with CI and observed value."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.hist(bootstrap_dist, bins=50, alpha=0.6, color="steelblue", edgecolor="white")
    ax.axvline(observed, color="red", linewidth=2, label=f"Observed = {observed:.4f}")
    ax.axvline(ci[0], color="orange", linestyle="--", linewidth=1.5, label=f"CI lower = {ci[0]:.4f}")
    ax.axvline(ci[1], color="orange", linestyle="--", linewidth=1.5, label=f"CI upper = {ci[1]:.4f}")
    ax.set_xlabel("Statistic")
    ax.set_ylabel("Frequency")
    ax.set_title(title or "Bootstrap Distribution")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
