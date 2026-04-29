"""Run the clustering ablation grid against cached embeddings.

Reads ``Z_{train,val,test}.parquet`` produced by the AE training job (either
local ``aux_dir`` or the SageMaker ``output.tar.gz`` extracted on disk) and
sweeps the 2x2 grid:

    {UMAP, t-SNE} x {KMeans, HDBSCAN}

Per cell we save:
    - ``clustering/{dr}_{cl}.parquet``  — date, label, prob, x_2d, y_2d
    - one row in ``clustering/summary.csv`` — silhouette / DBCV / NBER F1 / etc.

Usage:
    python scripts/run_clustering_ablation.py \\
        --embeddings-dir results/runs/<job>/aux/embeddings \\
        --usrec-csv data/raw/USREC.csv \\
        --output-dir  results/runs/<job>/clustering \\
        --n-clusters 4

Designed to run on a laptop or the cheapest CPU box -- no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_itransformer.evaluation.clustering import (
    compute_clustering_metrics,
    fit_kmeans,
)
from tcc_itransformer.evaluation.density_clustering import optimize_hdbscan_dbcv
from tcc_itransformer.evaluation.dim_reduction import (
    UMAPConfig,
    apply_umap,
    fit_umap,
)
from tcc_itransformer.evaluation.regime_validation import nber_overlap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Split:
    name: str
    Z: np.ndarray
    dates: pd.DatetimeIndex


def _load_split(emb_dir: Path, name: str) -> _Split:
    df = pd.read_parquet(emb_dir / f"Z_{name}.parquet")
    dates = pd.DatetimeIndex(df["date"])
    Z = df.filter(like="z_").to_numpy(dtype=np.float32)
    return _Split(name=name, Z=Z, dates=dates)


def _reduce(
    method: str, X_train: np.ndarray, X_test: np.ndarray, *, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project train + test into 2-D using UMAP or t-SNE."""
    if method == "umap":
        reducer = fit_umap(
            X_train, UMAPConfig(n_components=2, random_state=seed),
        )
        return apply_umap(X_train, reducer), apply_umap(X_test, reducer)
    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE has no .transform(): fit on stacked, then split.
        n_train = len(X_train)
        Y = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(np.vstack([X_train, X_test]))
        return Y[:n_train], Y[n_train:]
    raise ValueError(f"unknown dim-reduction method: {method!r}")


def _cluster(
    method: str, Y_train: np.ndarray, Y_test: np.ndarray, *,
    n_clusters: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Cluster on train and predict test labels.

    Returns:
        (test_labels, test_probs, fit_diagnostics)
    """
    if method == "kmeans":
        km = fit_kmeans(Y_train, n_clusters=n_clusters, random_state=seed)
        labels = km.predict(Y_test)
        probs = np.ones_like(labels, dtype=np.float32)
        return labels, probs, {"k": float(n_clusters)}

    if method == "hdbscan":
        best, _grid = optimize_hdbscan_dbcv(Y_train)
        try:
            import hdbscan as _hdbscan
            labels, probs = _hdbscan.approximate_predict(best.clusterer, Y_test)
        except Exception:  # pragma: no cover -- fall back to refit
            from tcc_itransformer.evaluation.density_clustering import fit_hdbscan
            refit = fit_hdbscan(
                Y_test,
                min_cluster_size=best.min_cluster_size,
                min_samples=best.min_samples,
            )
            labels = refit.labels
            probs = refit.probabilities
        diag = {
            "train_dbcv": float(best.dbcv),
            "train_n_clusters": float(best.n_clusters),
            "min_cluster_size": float(best.min_cluster_size),
            "min_samples": float(best.min_samples or -1),
        }
        return np.asarray(labels), np.asarray(probs, dtype=np.float32), diag

    raise ValueError(f"unknown clusterer: {method!r}")


def _maybe_nber_overlap(
    labels: np.ndarray, dates: pd.DatetimeIndex, usrec_csv: Path | None,
) -> dict[str, float]:
    if usrec_csv is None or not usrec_csv.exists():
        return {}
    raw = pd.read_csv(usrec_csv, parse_dates=[0])
    raw = raw.set_index(raw.columns[0]).iloc[:, 0].astype(int)
    res = nber_overlap(labels, dates, raw, lead=0, lag=2)
    return {
        "nber_f1": float(res.f1),
        "nber_precision": float(res.precision),
        "nber_recall": float(res.recall),
        "nber_matched_cluster": float(res.matched_cluster),
    }


def run_ablation(
    embeddings_dir: Path,
    output_dir: Path,
    *,
    n_clusters: int,
    seed: int,
    usrec_csv: Path | None,
    methods_dr: tuple[str, ...] = ("umap", "tsne"),
    methods_cl: tuple[str, ...] = ("kmeans", "hdbscan"),
) -> pd.DataFrame:
    """Execute the full ablation grid; persist per-cell parquets + summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_split(embeddings_dir, "train")
    test = _load_split(embeddings_dir, "test")

    rows: list[dict[str, Any]] = []
    for dr in methods_dr:
        Y_tr, Y_te = _reduce(dr, train.Z, test.Z, seed=seed)
        for cl in methods_cl:
            cell = f"{dr}_{cl}"
            try:
                labels, probs, diag = _cluster(
                    cl, Y_tr, Y_te, n_clusters=n_clusters, seed=seed,
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("Cell %s failed: %s", cell, exc)
                rows.append({"cell": cell, "dr": dr, "clusterer": cl, "error": str(exc)})
                continue

            # Per-cell parquet: date + label + prob + 2D coords (test set only).
            cell_df = pd.DataFrame(
                {
                    "date": test.dates,
                    "label": labels,
                    "probability": probs,
                    "x_2d": Y_te[:, 0],
                    "y_2d": Y_te[:, 1],
                }
            )
            cell_df.to_parquet(output_dir / f"{cell}.parquet", index=False)

            # Summary metrics on TEST set.
            try:
                test_metrics = compute_clustering_metrics(Y_te, labels)
            except Exception:  # pragma: no cover -- e.g. all-noise HDBSCAN
                test_metrics = {}
            row: dict[str, Any] = {
                "cell": cell,
                "dr": dr,
                "clusterer": cl,
                "seed": seed,
                "n_clusters_test": int(
                    len(set(labels)) - (1 if -1 in labels else 0)
                ),
                "noise_fraction": float(np.mean(labels == -1)),
            }
            row.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
            row.update({f"fit_{k}": v for k, v in diag.items()})
            row.update(_maybe_nber_overlap(labels, test.dates, usrec_csv))
            rows.append(row)
            logger.info(
                "%s -> n_clusters=%d  silhouette=%.3f  noise=%.2f",
                cell, row["n_clusters_test"],
                row.get("test_silhouette", float("nan")),
                row["noise_fraction"],
            )

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info("Wrote %d cells -> %s", len(summary), summary_csv)

    # Manifest with config used.
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "embeddings_dir": str(embeddings_dir),
                "n_clusters": n_clusters,
                "seed": seed,
                "methods_dr": list(methods_dr),
                "methods_cl": list(methods_cl),
                "n_train": int(len(train.Z)),
                "n_test": int(len(test.Z)),
                "embedding_dim": int(train.Z.shape[1]),
            },
            indent=2,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--usrec-csv", type=Path, default=None)
    p.add_argument(
        "--dr-methods", nargs="+", default=["umap", "tsne"],
        choices=["umap", "tsne"],
    )
    p.add_argument(
        "--cl-methods", nargs="+", default=["kmeans", "hdbscan"],
        choices=["kmeans", "hdbscan"],
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    run_ablation(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        seed=args.seed,
        usrec_csv=args.usrec_csv,
        methods_dr=tuple(args.dr_methods),
        methods_cl=tuple(args.cl_methods),
    )


if __name__ == "__main__":
    main()
