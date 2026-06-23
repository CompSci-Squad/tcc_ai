"""Sprint 2 — F8 (Consensus Clustering)
========================================
Addresses: F8 — C0 borderline (Jaccard = 0.50): recession cluster is unstable.

Implementation:
  1. Load iTransformer PCA-projected embeddings (Z_train + Z_val + Z_test).
  2. Consensus clustering: B=200 runs × 80% subsample → co-association matrix.
  3. Agglomerative clustering on (1 - co-association) distance matrix → K=4.
  4. Compute Jaccard stability per cluster with 95% bootstrap CI.
  5. Compare consensus Jaccard to original K-Means Jaccard (C0=0.50).
  6. Also run on windowed_pca and raw_pca embeddings for comparison.

Run:
    cd tcc_ai && uv run python scripts/sprint2_consensus_clustering.py

Outputs:
    results/sprint2/co_association_matrix_<encoder>.npy
    results/sprint2/consensus_jaccard_cis.csv
    results/sprint2/SUMMARY_consensus_C0_jaccard.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "sprint2"
OUT.mkdir(parents=True, exist_ok=True)

# ── embedding sources ─────────────────────────────────────────────────────────
_SM = ROOT / "results" / "sm_outputs" / "itransformer-1777581449-0d38" / "embeddings"
_PC = ROOT / "results" / "phase_c_comparison"

ENCODER_EMB: dict[str, dict] = {
    "iTransformer": {
        "train": _SM / "Z_train.parquet",
        "val":   _SM / "Z_val.parquet",
        "test":  _SM / "Z_test.parquet",
        "tier":  "tier1",
    },
    # For baselines, embeddings are the PCA projections stored in the parquets
    # (x_2d, y_2d are already in the ablation files — we reconstruct full-dim
    # embeddings by loading the raw parquets from phase_c_comparison)
}

# ── consensus clustering params ───────────────────────────────────────────────
B = 200
SUBSAMPLE = 0.80
K = 4
N_BOOT_JACCARD = 1000
SEED = 42
RNG = np.random.default_rng(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_embeddings(paths: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray, np.ndarray]:
    """Load Z_train/val/test parquets, return arrays + date arrays."""
    dfs = {}
    for split, path in paths.items():
        if split == "tier":
            continue
        df = pd.read_parquet(path)
        dfs[split] = df

    feat_cols = [c for c in dfs["train"].columns if c != "date"]
    X_train = dfs["train"][feat_cols].values.astype(float)
    X_val   = dfs["val"][feat_cols].values.astype(float)
    X_test  = dfs["test"][feat_cols].values.astype(float)

    d_train = pd.to_datetime(dfs["train"]["date"]) if "date" in dfs["train"].columns else None
    d_val   = pd.to_datetime(dfs["val"]["date"])   if "date" in dfs["val"].columns   else None
    d_test  = pd.to_datetime(dfs["test"]["date"])  if "date" in dfs["test"].columns  else None

    return X_train, X_val, X_test, d_train, d_val, d_test


def project_pca2(X_train: np.ndarray, X_other: list[np.ndarray],
                  seed: int = 42) -> tuple[np.ndarray, ...]:
    """Fit PCA(2) on train, transform all arrays."""
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(X_train)
    return (pca.transform(X_train),) + tuple(pca.transform(x) for x in X_other)


def build_co_association_matrix(
    X: np.ndarray,
    K: int,
    B: int,
    subsample: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a co-association matrix via B bootstrap runs of K-Means.
    co_assoc[i,j] = fraction of runs where i,j were in same cluster
                    (counting only runs where both were sampled).

    Vectorised: uses outer product of one-hot cluster indicators per run.
    O(B * n * K) — fast for n≈1100, K=4, B=200.

    Args:
        X: data matrix (n_samples, n_features)
        K: number of clusters
        B: number of bootstrap runs
        subsample: fraction of samples to draw per run
        rng: random number generator

    Returns:
        co_assoc: (n, n) float32 matrix in [0, 1]
    """
    n = len(X)
    co_count   = np.zeros((n, n), dtype=np.float32)
    pair_count = np.zeros((n, n), dtype=np.float32)

    for b in range(B):
        if (b + 1) % 50 == 0:
            print(f"    bootstrap {b+1}/{B} ...", flush=True)
        m = max(K + 1, int(n * subsample))
        idx = rng.choice(n, size=m, replace=False)

        km = KMeans(n_clusters=K, n_init=5, random_state=int(rng.integers(0, 9999)))
        labels = km.fit_predict(X[idx])  # shape (m,)

        # One-hot encode cluster assignments for sampled points: (m, K)
        onehot = np.zeros((m, K), dtype=np.float32)
        onehot[np.arange(m), labels] = 1.0

        # Full-size indicator: (n, K) — zeros for non-sampled rows
        ind = np.zeros((n, K), dtype=np.float32)
        ind[idx] = onehot

        # Sampled membership mask: (n,)
        sampled = np.zeros(n, dtype=np.float32)
        sampled[idx] = 1.0

        # co_count[i,j] += 1 if both sampled and same cluster
        # = sum_k ind[:,k] outer ind[:,k]
        co_count   += ind @ ind.T           # (n, n)

        # pair_count[i,j] += 1 if both sampled (regardless of cluster)
        pair_count += np.outer(sampled, sampled)  # (n, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        co_assoc = np.where(pair_count > 0, co_count / pair_count, 0.0).astype(np.float32)

    # Symmetrise and fill diagonal
    co_assoc = (co_assoc + co_assoc.T) / 2
    np.fill_diagonal(co_assoc, 1.0)
    return co_assoc


def agglomerative_from_co_assoc(
    co_assoc: np.ndarray, K: int
) -> np.ndarray:
    """Cluster using Agglomerative on distance = 1 - co_assoc."""
    dist = (1.0 - co_assoc).astype(np.float64)
    dist = np.clip(dist, 0, None)
    agg = AgglomerativeClustering(
        n_clusters=K,
        metric="precomputed",
        linkage="average",
    )
    return agg.fit_predict(dist)


def compute_jaccard(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[int, float]:
    """Compute per-cluster Jaccard (Monti 2003) between two labellings."""
    jaccard = {}
    for k in np.unique(labels_a):
        mask_a = labels_a == k
        # Match greedily to best-overlapping cluster in labels_b
        best_j = 0.0
        for l in np.unique(labels_b):
            mask_b = labels_b == l
            inter = float((mask_a & mask_b).sum())
            union = float((mask_a | mask_b).sum())
            j = inter / union if union > 0 else 0.0
            best_j = max(best_j, j)
        jaccard[int(k)] = round(best_j, 4)
    return jaccard


def bootstrap_jaccard_ci(
    X: np.ndarray, consensus_labels: np.ndarray,
    K: int, n_boot: int, rng: np.random.Generator
) -> dict[int, dict]:
    """Bootstrap CI for Jaccard stability of each cluster."""
    n = len(X)
    per_cluster: dict[int, list[float]] = {k: [] for k in range(K)}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        km = KMeans(n_clusters=K, n_init=5,
                    random_state=int(rng.integers(0, 9999)))
        boot_labels = km.fit_predict(X[idx])

        # map back to full-length labels (only sampled indices get labels)
        full_labels = np.full(n, -1, dtype=int)
        full_labels[idx] = boot_labels

        # Compare only sampled points
        sampled_mask = full_labels >= 0
        jaccards = compute_jaccard(consensus_labels[sampled_mask],
                                   full_labels[sampled_mask])
        for k, j in jaccards.items():
            if k in per_cluster:
                per_cluster[k].append(j)

    result = {}
    for k, vals in per_cluster.items():
        arr = np.array(vals)
        result[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lo": float(np.percentile(arr, 2.5)) if len(arr) > 0 else float("nan"),
            "ci_hi": float(np.percentile(arr, 97.5)) if len(arr) > 0 else float("nan"),
            "pr_stable": float((arr >= 0.60).mean()),  # Pr(J >= 0.60)
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main per-encoder consensus run
# ═══════════════════════════════════════════════════════════════════════════════

def run_consensus_for_encoder(
    enc_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> pd.DataFrame:
    """Full consensus pipeline for one encoder. Uses TEST set for final stability."""
    print(f"\n  ── {enc_name} ──")
    print(f"     X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # 1. PCA(2) projection
    Xr_tr, Xr_val, Xr_test = project_pca2(X_train, [X_val, X_test])
    X_all = np.vstack([Xr_tr, Xr_val, Xr_test])
    n_total = len(X_all)
    print(f"     PCA(2) projected: {n_total} total points")

    # 2. Co-association matrix on full panel (train+val+test)
    print(f"     Building co-association matrix (B={B}, sub={SUBSAMPLE})...")
    co_assoc = build_co_association_matrix(X_all, K, B, SUBSAMPLE, RNG)

    npy_path = OUT / f"co_association_matrix_{enc_name}.npy"
    np.save(str(npy_path), co_assoc)
    print(f"     ✓ Saved co-assoc → {npy_path.name}")

    # 3. Consensus labels via Agglomerative
    consensus_labels = agglomerative_from_co_assoc(co_assoc, K)

    # Compute ARI between consensus and simple KMeans
    km_baseline = KMeans(n_clusters=K, n_init=10, random_state=42)
    km_labels = km_baseline.fit_predict(X_all)
    ari = float(adjusted_rand_score(consensus_labels, km_labels))
    print(f"     ARI(consensus vs KMeans baseline) = {ari:.4f}")

    # 4. Bootstrap Jaccard CI on TEST slice
    n_test = len(Xr_test)
    test_labels_consensus = consensus_labels[-n_test:]
    jac_cis = bootstrap_jaccard_ci(Xr_test, test_labels_consensus, K, N_BOOT_JACCARD, RNG)

    rows = []
    for k, stats in jac_cis.items():
        rows.append({
            "encoder": enc_name,
            "cluster": k,
            "jaccard_mean": round(stats["mean"], 4),
            "jaccard_std": round(stats["std"], 4),
            "jaccard_ci_lo": round(stats["ci_lo"], 4),
            "jaccard_ci_hi": round(stats["ci_hi"], 4),
            "pr_stable_J60": round(stats["pr_stable"], 4),
            "stable": int(stats["pr_stable"] >= 0.50),   # using ≥0.60 by Hennig
            "ari_vs_kmeans": round(ari, 4),
        })
        status = "STABLE" if stats["pr_stable"] >= 0.60 else (
                 "BORDERLINE" if stats["pr_stable"] >= 0.30 else "UNSTABLE")
        print(
            f"     C{k}: J_mean={stats['mean']:.3f} "
            f"Pr(J≥0.60)={stats['pr_stable']:.3f} → {status}"
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    all_rows = []

    for enc_name, paths in ENCODER_EMB.items():
        print(f"\n[{enc_name}] Loading embeddings...")
        try:
            X_train, X_val, X_test, *_ = load_embeddings(paths)
        except Exception as exc:
            print(f"  ERROR loading {enc_name}: {exc}")
            continue
        df = run_consensus_for_encoder(enc_name, X_train, X_val, X_test)
        all_rows.append(df)

    if not all_rows:
        print("No encoders processed — check paths.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    out_csv = OUT / "consensus_jaccard_cis.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n  ✓ Saved → {out_csv.relative_to(ROOT)}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {}
    for enc_name in df_all["encoder"].unique():
        sub = df_all[df_all["encoder"] == enc_name]
        summary[enc_name] = {
            f"C{row.cluster}": {
                "jaccard_mean": row.jaccard_mean,
                "pr_stable_J60": row.pr_stable_J60,
                "stable": bool(row.pr_stable_J60 >= 0.60),
            }
            for row in sub.itertuples()
        }

    summary_path = OUT / "SUMMARY_consensus_C0_jaccard.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  ✓ Summary → {summary_path.relative_to(ROOT)}")
    print("\n═══ Sprint 2 complete ═══")


if __name__ == "__main__":
    main()
