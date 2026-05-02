"""A3 HDBSCAN+DR parameter sweep over UMAP/t-SNE x (min_cluster_size, min_samples)."""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from tcc_itransformer.evaluation.density_clustering import fit_hdbscan

logger = logging.getLogger(__name__)


def _load_full_z(emb_dir: Path) -> np.ndarray:
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(emb_dir / f"Z_{split}.parquet")
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)
    z_cols = [c for c in full.columns if c.startswith("z_")]
    return full[z_cols].to_numpy()


def _silhouette_clean(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    if n_clusters < 2:
        return float("nan")
    keep = labels != -1
    if keep.sum() <= n_clusters:
        return float("nan")
    return float(silhouette_score(X[keep], labels[keep]))


def run_hdbscan_sweep(
    embeddings_dir: Path,
    output_csv: Path = Path("results/clustering_ablation/param_sweep.csv"),
    *,
    seed: int = 42,
    mcs_grid: tuple[int, ...] = (5, 10, 15, 20),
    ms_grid: tuple[int, ...] = (1, 3, 5),
    umap_nn_grid: tuple[int, ...] = (5, 15, 30),
    tsne_perp_grid: tuple[int, ...] = (5, 15, 30, 50),
) -> pd.DataFrame:
    import umap
    from sklearn.manifold import TSNE

    Z = _load_full_z(embeddings_dir)
    logger.info("Loaded Z: shape=%s", Z.shape)

    rows: list[dict] = []
    for nn in umap_nn_grid:
        red = umap.UMAP(
            n_components=2, n_neighbors=nn, min_dist=0.0, random_state=seed,
        )
        X2 = red.fit_transform(Z)
        for mcs, ms in product(mcs_grid, ms_grid):
            res = fit_hdbscan(X2, min_cluster_size=mcs, min_samples=ms)
            sil = _silhouette_clean(X2, res.labels, res.n_clusters)
            rows.append({
                "dr": "umap", "dr_param": nn, "mcs": mcs, "ms": ms,
                "n_clusters": res.n_clusters,
                "noise_fraction": res.noise_fraction,
                "dbcv": res.dbcv, "silhouette": sil,
            })
            logger.info(
                "umap nn=%d mcs=%d ms=%d -> k=%d noise=%.2f dbcv=%.3f sil=%.3f",
                nn, mcs, ms, res.n_clusters, res.noise_fraction, res.dbcv, sil,
            )

    for perp in tsne_perp_grid:
        if perp >= Z.shape[0]:
            continue
        red = TSNE(
            n_components=2, perplexity=perp, random_state=seed,
            init="pca", learning_rate="auto",
        )
        X2 = red.fit_transform(Z)
        for mcs, ms in product(mcs_grid, ms_grid):
            res = fit_hdbscan(X2, min_cluster_size=mcs, min_samples=ms)
            sil = _silhouette_clean(X2, res.labels, res.n_clusters)
            rows.append({
                "dr": "tsne", "dr_param": perp, "mcs": mcs, "ms": ms,
                "n_clusters": res.n_clusters,
                "noise_fraction": res.noise_fraction,
                "dbcv": res.dbcv, "silhouette": sil,
            })
            logger.info(
                "tsne perp=%d mcs=%d ms=%d -> k=%d noise=%.2f dbcv=%.3f sil=%.3f",
                perp, mcs, ms, res.n_clusters, res.noise_fraction, res.dbcv, sil,
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows -> %s", len(df), output_csv)

    for dr_name in df["dr"].unique():
        sub = df[(df["dr"] == dr_name) & df["dbcv"].notna()].nlargest(5, "dbcv")
        print(f"\nTop 5 by DBCV ({dr_name}):")
        print(sub.to_string(index=False))
    return df
