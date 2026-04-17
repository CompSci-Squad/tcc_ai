"""Visualization utilities — publication-ready plots for thesis figures.

Style: ≥11pt body text, ≥14pt titles, Okabe-Ito colorblind-safe palette,
PDF+PNG export, consistent figure sizes per plot type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

_RC = {
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
}
sns.set_theme(style="whitegrid", rc=_RC)


def _save(fig: Figure, save_path: Path | None) -> None:
    """Save figure as PNG and PDF."""
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Existing plots (refactored)
# ---------------------------------------------------------------------------

def plot_scree(pca: PCA, save_path: Path | None = None) -> Figure:
    """Scree plot of PCA explained variance ratio."""
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(pca.explained_variance_ratio_)
    components = np.arange(1, n + 1)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    ax.bar(components, pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.plot(components, cumvar, "o-", color=OKABE_ITO[1], label="Cumulative")
    ax.axhline(y=0.9, linestyle="--", color="gray", alpha=0.7, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.set_xticks(components)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_cluster_scatter(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """2D scatter plot of embeddings colored by cluster labels."""
    fig, ax = plt.subplots(figsize=(7, 5))
    unique_labels = np.unique(labels)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f"Cluster {lbl}",
            alpha=0.7,
            s=30,
            color=OKABE_ITO[i % len(OKABE_ITO)],
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title or "Cluster Scatter Plot")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_regime_timeline(
    dates: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """Timeline showing regime (cluster) assignments over time."""
    fig, ax = plt.subplots(figsize=(10, 3))
    unique_labels = np.unique(labels)
    color_map = {lbl: OKABE_ITO[i % len(OKABE_ITO)] for i, lbl in enumerate(unique_labels)}

    colors = [color_map[lbl] for lbl in labels]
    ax.scatter(dates, labels, c=colors, s=15, alpha=0.8)

    for lbl in unique_labels:
        mask = labels == lbl
        ax.fill_between(
            dates, lbl - 0.3, lbl + 0.3,
            where=mask, alpha=0.15, color=color_map[lbl],
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_title(title or "Regime Timeline")
    ax.set_yticks(unique_labels)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_bootstrap_distribution(
    bootstrap_dist: np.ndarray,
    ci: tuple[float, float],
    observed: float,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """Histogram of bootstrap distribution with CI and observed value."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bootstrap_dist, bins=50, alpha=0.6, color=OKABE_ITO[4], edgecolor="white")
    ax.axvline(observed, color=OKABE_ITO[5], linewidth=2, label=f"Observed = {observed:.4f}")
    ax.axvline(ci[0], color=OKABE_ITO[0], linestyle="--", linewidth=1.5, label=f"CI lower = {ci[0]:.4f}")
    ax.axvline(ci[1], color=OKABE_ITO[0], linestyle="--", linewidth=1.5, label=f"CI upper = {ci[1]:.4f}")
    ax.set_xlabel("Statistic")
    ax.set_ylabel("Frequency")
    ax.set_title(title or "Bootstrap Distribution")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# New visualization functions
# ---------------------------------------------------------------------------

def plot_missing_data_heatmap(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> Figure:
    """Heatmap of missing data patterns (rows=time, cols=series)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df.isnull().astype(int).T,
        cbar=False, cmap=["white", OKABE_ITO[5]],
        ax=ax, yticklabels=True,
    )
    ax.set_title("Missing Data Pattern")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Series")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dist_histograms_grid(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    n_cols: int = 4,
    save_path: Path | None = None,
) -> Figure:
    """Grid of histograms for selected columns (top by kurtosis if not specified)."""
    if columns is None:
        kurt = df.kurtosis().sort_values(ascending=False)
        columns = kurt.head(12).index.tolist()
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(columns):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.hist(df[col].dropna(), bins=30, alpha=0.7, color=OKABE_ITO[i % len(OKABE_ITO)])
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=8)
    for i in range(len(columns), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].set_visible(False)
    fig.suptitle("Distribution Histograms", fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_stationarity_summary(
    adf_pvalues: np.ndarray,
    kpss_pvalues: np.ndarray,
    series_names: Sequence[str],
    save_path: Path | None = None,
) -> Figure:
    """Scatter of ADF vs KPSS p-values with quadrant annotations."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(adf_pvalues, kpss_pvalues, alpha=0.6, s=20, color=OKABE_ITO[4])
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.05, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ADF p-value (< 0.05 → stationary)")
    ax.set_ylabel("KPSS p-value (> 0.05 → stationary)")
    ax.set_title("Stationarity Summary")
    ax.text(0.01, 0.95, "Stationary", transform=ax.transAxes, fontsize=9, color="green")
    ax.text(0.7, 0.05, "Non-stationary", transform=ax.transAxes, fontsize=9, color="red")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_correlation_heatmaps(
    df: pd.DataFrame,
    method: str = "spearman",
    save_path: Path | None = None,
) -> Figure:
    """Clustered correlation heatmap."""
    corr = df.corr(method=method)
    g = sns.clustermap(
        corr, figsize=(10, 10), cmap="RdBu_r", center=0,
        linewidths=0, dendrogram_ratio=0.1,
    )
    g.fig.suptitle(f"{method.title()} Correlation (hierarchical clustering)", y=1.02, fontsize=14)
    if save_path is not None:
        _save(g.fig, save_path)
    return g.fig


def plot_window_statistics(
    window_means: np.ndarray,
    window_stds: np.ndarray,
    save_path: Path | None = None,
) -> Figure:
    """Line plots of per-window mean and std over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    x = np.arange(len(window_means))
    ax1.plot(x, window_means, color=OKABE_ITO[4])
    ax1.set_ylabel("Window Mean")
    ax1.set_title("Per-Window Statistics")
    ax2.plot(x, window_stds, color=OKABE_ITO[5])
    ax2.set_ylabel("Window Std")
    ax2.set_xlabel("Window Index")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dim_variance_heatmap(
    embeddings: np.ndarray,
    save_path: Path | None = None,
) -> Figure:
    """Heatmap of per-dimension variance across samples."""
    var = np.var(embeddings, axis=0)
    fig, ax = plt.subplots(figsize=(max(6, len(var) * 0.5), 2))
    sns.heatmap(
        var.reshape(1, -1), annot=True, fmt=".3f", cmap="YlOrRd",
        ax=ax, xticklabels=[f"D{i}" for i in range(len(var))],
        yticklabels=False,
    )
    ax.set_title("Per-Dimension Variance")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_silhouette_vs_k(
    k_range: Sequence[int],
    silhouette_scores: Sequence[float],
    best_k: int | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Silhouette score vs K with optional best-K marker."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_range, silhouette_scores, "o-", color=OKABE_ITO[4])
    if best_k is not None:
        idx = list(k_range).index(best_k)
        ax.plot(best_k, silhouette_scores[idx], "s", markersize=12,
                color=OKABE_ITO[5], label=f"Best K={best_k}")
        ax.legend()
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette vs K")
    ax.set_xticks(list(k_range))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_regime_timeline_nber(
    dates: np.ndarray,
    labels: np.ndarray,
    nber_recessions: Sequence[tuple] | None = None,
    title: str = "",
    save_path: Path | None = None,
) -> Figure:
    """Regime timeline with optional NBER recession shading."""
    fig, ax = plt.subplots(figsize=(12, 3))
    unique_labels = np.unique(labels)
    color_map = {lbl: OKABE_ITO[i % len(OKABE_ITO)] for i, lbl in enumerate(unique_labels)}

    colors = [color_map[lbl] for lbl in labels]
    ax.scatter(dates, labels, c=colors, s=20, alpha=0.8)

    if nber_recessions is not None:
        for start, end in nber_recessions:
            ax.axvspan(start, end, alpha=0.15, color="gray", label="NBER Recession")

    # Deduplicate legend
    handles, leg_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_title(title or "Regime Timeline with NBER Recessions")
    ax.set_yticks(unique_labels)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_statistical_results_table(
    results: dict,
    title: str = "Statistical Test Results",
    save_path: Path | None = None,
) -> Figure:
    """Render dict of statistical results as a matplotlib table figure."""
    rows = []
    for key, val in results.items():
        if isinstance(val, float):
            rows.append([key, f"{val:.4f}"])
        elif isinstance(val, np.ndarray) and val.ndim == 0:
            rows.append([key, f"{val.item():.4f}"])
        else:
            rows.append([key, str(val)])

    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(rows))))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    ax.set_title(title, fontsize=14, pad=20)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pairwise_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str = "Pairwise Comparison",
    cmap: str = "RdBu_r",
    save_path: Path | None = None,
) -> Figure:
    """Symmetric heatmap for pairwise comparisons (e.g. MW p-values)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=ax,
        square=True, linewidths=0.5,
    )
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_baseline_comparison_bar(
    baseline_names: Sequence[str],
    silhouette_scores: Sequence[float],
    p_values: Sequence[float] | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Bar chart comparing silhouette scores across baselines + model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(baseline_names))
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(baseline_names))]
    bars = ax.bar(x, silhouette_scores, color=colors, alpha=0.8)

    if p_values is not None:
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    marker, ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(baseline_names, rotation=30, ha="right")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Model vs Baselines")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
