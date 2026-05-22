"""Phase C4: Cluster stability bootstrap (Ben-Hur 2002 Jaccard stability).

For each clustering pipeline in the B1 ablation directory:
  1. Load the reference labels from the pre-computed parquet.
  2. Load the iTransformer Z_test embeddings (185 × 7) from emb_dir.
  3. Run N bootstrap resamples (default 80% of test-period windows).
  4. On each resample: re-apply the same DR (PCA/UMAP) + clustering step.
  5. Compute per-cluster mean Jaccard (Ben-Hur 2002) and overall ARI.
  6. Flag clusters with mean Jaccard < 0.6 as unstable.

Output: results/diagnostics/cluster_stability.csv

Input embeddings must come from ``emb_dir`` (path to Z_{train,val,test}.parquet).
The bootstrap operates on Z_test — the same split the reference clusters were
derived from. No AWS or SageMaker calls are needed.

Requires: scikit-learn, hdbscan (all in main deps).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)

_JACCARD_UNSTABLE_THRESHOLD = 0.6
_PIPELINE_FILES = (
    "pca_kmeans.parquet",
    "pca_hdbscan.parquet",
    "mlp_ae_hdbscan.parquet",
)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two boolean cluster membership arrays."""
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return intersection / union if union > 0 else 0.0


def _best_jaccard_for_cluster(
    ref_labels: np.ndarray,
    ref_cluster: int,
    boot_labels: np.ndarray,
    boot_unique: list[int],
) -> float:
    """Best Jaccard between a reference cluster and any bootstrap cluster."""
    ref_mask = ref_labels == ref_cluster
    if ref_mask.sum() == 0:
        return 0.0
    return max(
        _jaccard(ref_mask, boot_labels == bc)
        for bc in boot_unique
        if bc != -1
    ) if boot_unique else 0.0


def _fit_pca_hdbscan(
    embeddings: np.ndarray,
    n_components: int,
    min_cluster_size: int,
    min_samples: int | None,
    seed: int,
) -> np.ndarray:
    """Fit PCA + HDBSCAN on a sub-sample, return labels."""
    import hdbscan as _hdbscan

    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embeddings)
    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced)


def _fit_pca_kmeans(
    embeddings: np.ndarray,
    n_components: int,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """Fit PCA + KMeans on a sub-sample, return labels (no noise = -1)."""
    from sklearn.cluster import KMeans

    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embeddings)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    return km.fit_predict(reduced)


def _load_reference(parquet_path: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load reference labels and dates from pre-computed parquet."""
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df["label"].to_numpy(), pd.DatetimeIndex(df["date"])


def _load_z_embeddings(emb_dir: Path, split: str = "test") -> np.ndarray:
    """Load iTransformer Z embeddings (N × d) for the given split.

    The clustering ablation uses these 7-dim latent vectors, not windowed panel.
    """
    path = emb_dir / f"Z_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Z_{split}.parquet not found at {path}. "
            "Run SageMaker training and export embeddings first."
        )
    df = pd.read_parquet(path)
    return df.filter(like="z_").to_numpy(dtype=np.float64)


def run_cluster_stability(
    ablation_dir: Path,
    emb_dir: Path,
    output: Path,
    n_bootstrap: int = 100,
    resample_frac: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Ben-Hur 2002 cluster stability bootstrap.

    Args:
        ablation_dir: Directory containing {pipeline}.parquet files from B1 ablation.
        emb_dir: Directory with Z_test.parquet (iTransformer latent embeddings, N×7).
        output: Output CSV path.
        n_bootstrap: Number of bootstrap resamples.
        resample_frac: Fraction of windows per resample.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with one row per (pipeline, cluster) + overall ARI row.
    """
    rng = np.random.default_rng(seed)

    # Load Z_test embeddings — shared across all pipeline bootstraps
    embeddings = _load_z_embeddings(emb_dir, split="test")
    n_windows = len(embeddings)
    logger.info("Loaded Z_test embeddings: %s", embeddings.shape)

    # Determine which pipelines to evaluate
    pipelines = []
    for fname in _PIPELINE_FILES:
        p = ablation_dir / fname
        if p.exists():
            pipelines.append(p)
        else:
            logger.info("Pipeline parquet not found (skipping): %s", p)

    if not pipelines:
        raise FileNotFoundError(
            f"No supported parquet files found in {ablation_dir}. "
            "Run `make baselines` or `make ablation` first."
        )

    all_rows: list[dict] = []

    for parquet_path in pipelines:
        pipeline_name = parquet_path.stem  # e.g. "pca_hdbscan"
        logger.info("Running stability bootstrap for: %s", pipeline_name)

        ref_labels, _ref_dates = _load_reference(parquet_path)

        # Read clustering params from the summary CSV in the same directory
        summary_csv = ablation_dir / "summary.csv"
        is_kmeans = "kmeans" in pipeline_name
        min_cluster_size = 5
        min_samples = None
        n_clusters_k = 4  # default for kmeans
        if summary_csv.exists():
            try:
                summary_df = pd.read_csv(summary_csv)
                row_df = summary_df[summary_df["cell"] == pipeline_name]
                if not row_df.empty:
                    if is_kmeans:
                        fk = row_df["fit_k"].iloc[0]
                        if not pd.isna(fk):
                            n_clusters_k = int(fk)
                    else:
                        mcs = row_df["fit_min_cluster_size"].iloc[0]
                        ms = row_df["fit_min_samples"].iloc[0]
                        if not pd.isna(mcs):
                            min_cluster_size = int(mcs)
                        if not pd.isna(ms) and ms > 0:
                            min_samples = int(ms)
            except Exception as exc:
                logger.warning("Could not read params from summary.csv: %s", exc)

        # PCA(2D) matching the original pipeline: Z (7D) → PCA → 2D → cluster
        n_components = min(2, embeddings.shape[1])

        # Reference cluster ids (excluding noise)
        ref_clusters = sorted(c for c in np.unique(ref_labels) if c != -1)
        logger.info("Reference clusters: %s (n=%d windows)", ref_clusters, n_windows)

        # Bootstrap loop
        jaccard_per_cluster: dict[int, list[float]] = {c: [] for c in ref_clusters}
        ari_list: list[float] = []

        for b in range(n_bootstrap):
            n_resample = max(int(n_windows * resample_frac), 10)
            boot_idx = rng.choice(n_windows, size=n_resample, replace=False)
            boot_idx_sorted = np.sort(boot_idx)

            boot_embeddings = embeddings[boot_idx_sorted]

            try:
                boot_seed = int(rng.integers(0, 2**31))
                if is_kmeans:
                    boot_labels = _fit_pca_kmeans(
                        boot_embeddings,
                        n_components=n_components,
                        n_clusters=n_clusters_k,
                        seed=boot_seed,
                    )
                else:
                    boot_labels = _fit_pca_hdbscan(
                        boot_embeddings,
                        n_components=n_components,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        seed=boot_seed,
                    )
            except Exception as exc:
                logger.debug("Bootstrap %d failed: %s", b, exc)
                continue

            boot_unique = sorted(np.unique(boot_labels).tolist())

            # Per-cluster Jaccard (Ben-Hur 2002, cluster-wise, Hennig variant)
            for c in ref_clusters:
                ref_mask_boot = ref_labels[boot_idx_sorted] == c
                if ref_mask_boot.sum() == 0:
                    continue
                j = max(
                    (_jaccard(ref_mask_boot, boot_labels == bc) for bc in boot_unique if bc != -1),
                    default=0.0,
                )
                jaccard_per_cluster[c].append(j)

            # Overall ARI: compare reference partition on boot indices to bootstrap labels
            ref_boot_labels = ref_labels[boot_idx_sorted]
            if len(set(boot_labels)) > 1 and len(set(ref_boot_labels)) > 1:
                ari = adjusted_rand_score(ref_boot_labels, boot_labels)
                ari_list.append(ari)

            if (b + 1) % 25 == 0:
                logger.info("  %s: %d/%d resamples done", pipeline_name, b + 1, n_bootstrap)

        # Aggregate per-cluster Jaccard results
        for c in ref_clusters:
            jacs = jaccard_per_cluster[c]
            mean_j = float(np.mean(jacs)) if jacs else 0.0
            std_j = float(np.std(jacs)) if jacs else 0.0
            n_valid = len(jacs)
            stable = mean_j >= _JACCARD_UNSTABLE_THRESHOLD
            n_months = int((ref_labels == c).sum())

            all_rows.append({
                "pipeline": pipeline_name,
                "cluster": c,
                "n_months": n_months,
                "metric": "jaccard",
                "n_valid_resamples": n_valid,
                "mean_value": round(mean_j, 4),
                "std_value": round(std_j, 4),
                "stable": stable,
                "jaccard_threshold": _JACCARD_UNSTABLE_THRESHOLD,
                "n_bootstrap": n_bootstrap,
                "resample_frac": resample_frac,
            })

        # Aggregate overall ARI row (one per pipeline)
        mean_ari = float(np.mean(ari_list)) if ari_list else float("nan")
        std_ari = float(np.std(ari_list)) if ari_list else float("nan")
        all_rows.append({
            "pipeline": pipeline_name,
            "cluster": "all",
            "n_months": n_windows,
            "metric": "ari",
            "n_valid_resamples": len(ari_list),
            "mean_value": round(mean_ari, 4),
            "std_value": round(std_ari, 4),
            "stable": mean_ari >= 0.5 if not np.isnan(mean_ari) else False,
            "jaccard_threshold": _JACCARD_UNSTABLE_THRESHOLD,
            "n_bootstrap": n_bootstrap,
            "resample_frac": resample_frac,
        })

        n_stable = sum(
            1 for r in all_rows
            if r["pipeline"] == pipeline_name and r["metric"] == "jaccard" and r["stable"]
        )
        logger.info(
            "%s: %d/%d clusters stable (mean_jaccard >= %.2f) | mean ARI=%.3f",
            pipeline_name, n_stable, len(ref_clusters), _JACCARD_UNSTABLE_THRESHOLD, mean_ari,
        )

    df = pd.DataFrame(all_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Cluster stability report written to %s\n%s", output, df.to_string())
    return df

