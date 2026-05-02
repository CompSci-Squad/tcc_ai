"""Clustering ablation: {PCA, UMAP, t-SNE} x {KMeans, HDBSCAN} on cached embeddings.

Reads ``Z_{train,val,test}.parquet`` produced by the AE training job. For each
cell, persists a per-cell parquet (date, label, prob, 2D coords) and one row
in ``summary.csv`` with the locked 7-metric panel + silhouette permutation
p-value + bootstrap CI.

Designed for laptop/CPU execution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from tcc_itransformer.evaluation.clustering import compute_clustering_metrics, fit_kmeans
from tcc_itransformer.evaluation.density_clustering import optimize_hdbscan_dbcv
from tcc_itransformer.evaluation.dim_reduction import UMAPConfig, apply_umap, fit_umap
from tcc_itransformer.evaluation.panel_metrics import PANEL_COLUMNS, compute_panel_metrics

logger = logging.getLogger(__name__)


def _silhouette_perm_and_ci(
    Y: np.ndarray, labels: np.ndarray, *, n_perm: int, n_boot: int, seed: int,
) -> tuple[float, float, float, float]:
    mask = labels != -1
    if mask.sum() < 4 or len(set(labels[mask])) < 2:
        nan = float("nan")
        return nan, nan, nan, nan
    Y_m, L_m = Y[mask], labels[mask]
    sil_obs = float(silhouette_score(Y_m, L_m))
    rng = np.random.default_rng(seed)
    perm_geq = 0
    for _ in range(n_perm):
        L_perm = rng.permutation(L_m)
        if len(set(L_perm)) < 2:
            continue
        if silhouette_score(Y_m, L_perm) >= sil_obs:
            perm_geq += 1
    p_perm = (perm_geq + 1) / (n_perm + 1)

    boots = np.empty(n_boot, dtype=np.float64)
    n = len(L_m)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        L_b = L_m[idx]
        if len(set(L_b)) < 2:
            boots[i] = np.nan
            continue
        boots[i] = silhouette_score(Y_m[idx], L_b)
    boots = boots[~np.isnan(boots)]
    if boots.size < 10:
        return sil_obs, float(p_perm), float("nan"), float("nan")
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return sil_obs, float(p_perm), float(ci_low), float(ci_high)


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
    method: str, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    *, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "umap":
        reducer = fit_umap(X_train, UMAPConfig(n_components=2, random_state=seed))
        return (
            apply_umap(X_train, reducer),
            apply_umap(X_val, reducer),
            apply_umap(X_test, reducer),
        )
    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE has no .transform(): fit on stacked, then split.
        n_train, n_val = len(X_train), len(X_val)
        Y = TSNE(
            n_components=2, random_state=seed,
            init="pca", learning_rate="auto",
        ).fit_transform(np.vstack([X_train, X_val, X_test]))
        return Y[:n_train], Y[n_train : n_train + n_val], Y[n_train + n_val :]
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=seed)
        pca.fit(X_train)
        return pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)
    raise ValueError(f"unknown dim-reduction method: {method!r}")


def _cluster(
    method: str, Y_train: np.ndarray, Y_val: np.ndarray, Y_test: np.ndarray,
    *, n_clusters: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if method == "kmeans":
        km = fit_kmeans(Y_train, k=n_clusters, random_state=seed)
        val_labels = km.predict(Y_val)
        test_labels = km.predict(Y_test)
        probs = np.ones_like(test_labels, dtype=np.float32)
        return val_labels, test_labels, probs, {"k": float(n_clusters)}

    if method == "hdbscan":
        best, _ = optimize_hdbscan_dbcv(Y_train)

        def _predict(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            try:
                import hdbscan as _hdbscan
                lbl, prb = _hdbscan.approximate_predict(best.clusterer, Y)
            except Exception:  # pragma: no cover
                from tcc_itransformer.evaluation.density_clustering import fit_hdbscan
                refit = fit_hdbscan(
                    Y, min_cluster_size=best.min_cluster_size,
                    min_samples=best.min_samples,
                )
                lbl, prb = refit.labels, refit.probabilities
            return np.asarray(lbl), np.asarray(prb, dtype=np.float32)

        val_labels, _ = _predict(Y_val)
        test_labels, test_probs = _predict(Y_test)
        diag = {
            "train_dbcv": float(best.dbcv),
            "train_n_clusters": float(best.n_clusters),
            "min_cluster_size": float(best.min_cluster_size),
            "min_samples": float(best.min_samples or -1),
        }
        return val_labels, test_labels, test_probs, diag

    raise ValueError(f"unknown clusterer: {method!r}")


def run_ablation(
    embeddings_dir: Path, output_dir: Path,
    *,
    n_clusters: int = 4, seed: int = 42,
    usrec_csv: Path | None = None,
    methods_dr: tuple[str, ...] = ("pca", "umap", "tsne"),
    methods_cl: tuple[str, ...] = ("kmeans", "hdbscan"),
    n_perm: int = 1000, n_boot: int = 1000,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _load_split(embeddings_dir, "train")
    val = _load_split(embeddings_dir, "val")
    test = _load_split(embeddings_dir, "test")

    rows: list[dict[str, Any]] = []
    for dr in methods_dr:
        Y_tr, Y_va, Y_te = _reduce(dr, train.Z, val.Z, test.Z, seed=seed)
        for cl in methods_cl:
            cell = f"{dr}_{cl}"
            try:
                val_labels, labels, probs, diag = _cluster(
                    cl, Y_tr, Y_va, Y_te, n_clusters=n_clusters, seed=seed,
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("Cell %s failed: %s", cell, exc)
                rows.append({
                    "cell": cell, "dr": dr, "clusterer": cl, "error": str(exc),
                })
                continue

            cell_df = pd.DataFrame({
                "date": test.dates,
                "label": labels,
                "probability": probs,
                "x_2d": Y_te[:, 0],
                "y_2d": Y_te[:, 1],
            })
            cell_df.to_parquet(output_dir / f"{cell}.parquet", index=False)

            try:
                test_metrics = compute_clustering_metrics(Y_te, labels)
            except Exception:  # pragma: no cover
                test_metrics = {}

            panel = compute_panel_metrics(
                val_labels=val_labels, val_dates=val.dates,
                test_labels=labels, test_dates=test.dates,
                Y_test=Y_te, test_signal=Y_te,
                usrec_csv=usrec_csv,
                is_density_clusterer=(cl == "hdbscan"),
            )

            row: dict[str, Any] = {
                "cell": cell, "dr": dr, "clusterer": cl, "seed": seed,
                **{k: panel.get(k, float("nan")) for k in PANEL_COLUMNS},
                "nber_precision": panel.get("nber_precision", float("nan")),
                "nber_recall": panel.get("nber_recall", float("nan")),
                "nber_assignment": panel.get("nber_assignment", ""),
                "crisis_n_canonical_in_test": panel.get(
                    "crisis_n_canonical_in_test", float("nan"),
                ),
                "dbcv_applicable": (cl == "hdbscan"),
            }
            row.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
            row.update({f"fit_{k}": v for k, v in diag.items()})

            sil_obs, p_perm, ci_lo, ci_hi = _silhouette_perm_and_ci(
                Y_te, labels, n_perm=n_perm, n_boot=n_boot, seed=seed,
            )
            row["silhouette_perm_p"] = p_perm
            row["silhouette_ci_low"] = ci_lo
            row["silhouette_ci_high"] = ci_hi
            rows.append(row)
            logger.info(
                "%s -> n_clusters=%d  dbcv=%.3f  silhouette=%.3f  noise=%.2f  "
                "nber_f1=%.3f  bp=%.3f  crisis=%.2f",
                cell, int(row["n_clusters_test"]), row["dbcv"],
                row.get("test_silhouette", float("nan")),
                row["noise_fraction_test"], row["nber_f1"],
                row["bai_perron_f1"], row["crisis_window_coverage"],
            )

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info("Wrote %d cells -> %s", len(summary), summary_csv)

    (output_dir / "manifest.json").write_text(json.dumps(
        {
            "embeddings_dir": str(embeddings_dir),
            "n_clusters": n_clusters, "seed": seed,
            "methods_dr": list(methods_dr),
            "methods_cl": list(methods_cl),
            "n_train": int(len(train.Z)),
            "n_test": int(len(test.Z)),
            "embedding_dim": int(train.Z.shape[1]),
        },
        indent=2,
    ))
    return summary
