"""A3 HDBSCAN+DR parameter sweep.

For a given embeddings dir, project Z_train+Z_val+Z_test with UMAP (varying
n_neighbors) and t-SNE (varying perplexity), then run HDBSCAN over a grid of
(min_cluster_size, min_samples). Reports DBCV, silhouette, n_clusters, noise.

Used to diagnose why certain (DR x clusterer) cells fail in the main ablation.

Output: <output_csv> with one row per (dr, dr_param, mcs, ms).
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--output-csv", default="results/clustering_ablation/param_sweep.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mcs-grid", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--ms-grid", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--umap-nn-grid", type=int, nargs="+", default=[5, 15, 30])
    p.add_argument("--tsne-perp-grid", type=int, nargs="+", default=[5, 15, 30, 50])
    args = p.parse_args()

    Z = _load_full_z(Path(args.embeddings_dir))
    logger.info("Loaded Z: shape=%s", Z.shape)

    rows: list[dict] = []
    for nn in args.umap_nn_grid:
        red = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.0,
                        random_state=args.seed)
        X2 = red.fit_transform(Z)
        for mcs, ms in product(args.mcs_grid, args.ms_grid):
            res = fit_hdbscan(X2, min_cluster_size=mcs, min_samples=ms)
            sil = (silhouette_score(X2[res.labels != -1], res.labels[res.labels != -1])
                   if res.n_clusters >= 2 and (res.labels != -1).sum() > res.n_clusters
                   else float("nan"))
            rows.append({"dr": "umap", "dr_param": nn, "mcs": mcs, "ms": ms,
                         "n_clusters": res.n_clusters, "noise_fraction": res.noise_fraction,
                         "dbcv": res.dbcv, "silhouette": sil})
            logger.info("umap nn=%d mcs=%d ms=%d -> k=%d noise=%.2f dbcv=%.3f sil=%.3f",
                        nn, mcs, ms, res.n_clusters, res.noise_fraction, res.dbcv, sil)

    for perp in args.tsne_perp_grid:
        if perp >= Z.shape[0]:
            continue
        red = TSNE(n_components=2, perplexity=perp, random_state=args.seed,
                   init="pca", learning_rate="auto")
        X2 = red.fit_transform(Z)
        for mcs, ms in product(args.mcs_grid, args.ms_grid):
            res = fit_hdbscan(X2, min_cluster_size=mcs, min_samples=ms)
            sil = (silhouette_score(X2[res.labels != -1], res.labels[res.labels != -1])
                   if res.n_clusters >= 2 and (res.labels != -1).sum() > res.n_clusters
                   else float("nan"))
            rows.append({"dr": "tsne", "dr_param": perp, "mcs": mcs, "ms": ms,
                         "n_clusters": res.n_clusters, "noise_fraction": res.noise_fraction,
                         "dbcv": res.dbcv, "silhouette": sil})
            logger.info("tsne perp=%d mcs=%d ms=%d -> k=%d noise=%.2f dbcv=%.3f sil=%.3f",
                        perp, mcs, ms, res.n_clusters, res.noise_fraction, res.dbcv, sil)

    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows -> %s", len(df), out)

    # Summary: top 5 by DBCV per dr.
    for dr_name in df["dr"].unique():
        sub = df[(df["dr"] == dr_name) & df["dbcv"].notna()].nlargest(5, "dbcv")
        print(f"\nTop 5 by DBCV ({dr_name}):")
        print(sub.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
