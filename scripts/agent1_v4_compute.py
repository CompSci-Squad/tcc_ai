"""
Agent-1 v4 Audit Compute — answers 5 pending reviewer queries.

Run:
    cd tcc_ai && uv run python scripts/agent1_v4_compute.py

Outputs (all JSON/CSV) are written to:
    artifacts/agent1_response_v4_supporting/

Queries:
    F1 — Seed sensitivity with CANONICAL iTransformer pipeline (Z available).
    F2 — VAL F1 raw (no tolerance) for all 13 encoders → Tabela 5 Gap column.
    F7 — raw_pca tolerance F1 (TEST) for Tabela 3 row completion.
    F4 — B=1000 bootstrap stability for Tabela 6.
    F5 — K-sensitivity table for supplementary material.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── repo root & path setup ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # tcc_ai/
sys.path.insert(0, str(ROOT / "src"))

from tcc_itransformer.evaluation.regime_validation import (
    fit_nber_assignment,
    nber_overlap_frozen,
)

OUT = ROOT.parent / "artifacts" / "agent1_response_v4_supporting"
OUT.mkdir(parents=True, exist_ok=True)

NBER_CSV = ROOT / "data" / "snapshots" / "nber_usrec.csv"
CANONICAL_EMB = ROOT / "results" / "sm_outputs" / "itransformer-1777581449-0d38" / "embeddings"
K4_B1_TEST = ROOT / "results" / "clustering_ablation" / "W6_d7_K4_b1" / "pca_kmeans.parquet"
K4_B1_VAL = ROOT / "results" / "clustering_ablation" / "W6_d7_K4_b1" / "val_pca_kmeans.parquet"
CANONICAL_TABLE = ROOT / "outputs" / "tables" / "phase_c_canonical_pca_kmeans.csv"

VAL_START = pd.Timestamp("2000-06-01")
VAL_END = pd.Timestamp("2009-12-01")
TEST_START = pd.Timestamp("2010-06-01")
TEST_END = pd.Timestamp("2026-01-01")

SEEDS = [0, 1, 7, 42, 123]
LEAD, LAG = 0, 2

# ── load NBER ─────────────────────────────────────────────────────────────────
nber_raw = pd.read_csv(NBER_CSV, parse_dates=["observation_date"])
nber_raw = nber_raw.rename(columns={"observation_date": "date"})
nber_series = nber_raw.set_index("date")["USREC"].astype(int)

# ── helper: run canonical clustering pipeline on Z ────────────────────────────
def cluster_on_z(
    z_train: pd.DataFrame,
    z_val: pd.DataFrame,
    z_test: pd.DataFrame,
    K: int,
    seed: int,
    n_pca_components: int = 2,  # canonical: PCA(n_components=2) per clustering_ablation.py
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Canonical clustering pipeline from tcc_itransformer/pipelines/clustering_ablation.py.

    Pipeline (matching clustering_ablation.py exactly):
      1. PCA(n_components=2, random_state=seed) fit on Z_train
      2. Transform Z_train, Z_val, Z_test separately
      3. KMeans(K, random_state=seed) fit on PCA-projected Z_train
      4. km.predict(Y_val), km.predict(Y_test) for labels

    Returns (val_labels, test_labels, silhouette_on_test, n_comp).
    """
    feat_cols = [c for c in z_train.columns if c != "date"]
    X_train = z_train[feat_cols].values
    X_val   = z_val[feat_cols].values
    X_test  = z_test[feat_cols].values

    n_comp = min(n_pca_components, X_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(X_train)
    Xr_train = pca.transform(X_train)
    Xr_val   = pca.transform(X_val)
    Xr_test  = pca.transform(X_test)

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(Xr_train)  # fit on train only
    val_labels  = km.predict(Xr_val)
    test_labels = km.predict(Xr_test)

    sil = float(silhouette_score(Xr_test, test_labels)) if len(set(test_labels)) > 1 else float("nan")
    return val_labels, test_labels, sil, n_comp


def compute_f1_frozen(
    val_labels: np.ndarray,
    val_dates: pd.DatetimeIndex,
    test_labels: np.ndarray,
    test_dates: pd.DatetimeIndex,
    *,
    lag: int = LAG,
    raw: bool = False,
) -> dict:
    """Fit assignment on VAL (with lag), apply frozen to TEST.
    
    raw=True: also return F1 with lag=0 (no tolerance) on TEST.
    """
    assignment = fit_nber_assignment(val_labels, val_dates, nber_series, lead=LEAD, lag=lag)
    result_tol = nber_overlap_frozen(test_labels, test_dates, nber_series, assignment, lead=LEAD, lag=lag)
    out = {
        "assignment": assignment,
        "f1_tolerance": result_tol.f1,
        "precision_tolerance": result_tol.precision,
        "recall_tolerance": result_tol.recall,
        "n_recession_months": result_tol.n_recession_months,
        "n_predicted_months": result_tol.n_predicted_months,
    }
    if raw:
        result_raw = nber_overlap_frozen(test_labels, test_dates, nber_series, assignment, lead=0, lag=0)
        out.update({
            "f1_raw": result_raw.f1,
            "precision_raw": result_raw.precision,
            "recall_raw": result_raw.recall,
        })
    return out


def compute_val_f1_raw(
    val_labels: np.ndarray,
    val_dates: pd.DatetimeIndex,
) -> dict:
    """Compute F1 on VAL against raw NBER (no tolerance).

    No frozen assignment needed since we evaluate directly against raw NBER.
    Recession cluster = cluster with highest raw NBER overlap on VAL itself (in-sample,
    which is what the reviewer wants for the gap analysis).
    
    Also returns the lag=2 tolerance F1 on VAL for reference.
    """
    rec_val_raw = nber_series.reindex(pd.DatetimeIndex(val_dates)).fillna(0).astype(int)
    labels_s = pd.Series(val_labels, index=pd.DatetimeIndex(val_dates))

    # Identify recession cluster by raw VAL overlap (no tolerance)
    best_f1, best_c = 0.0, -1
    assignment_raw: dict[int, int] = {}
    for c in sorted(set(val_labels)):
        if c == -1:
            assignment_raw[c] = 0
            continue
        pred_c = (labels_s == c).astype(int)
        n_pred = int(pred_c.sum())
        n_rec  = int(rec_val_raw.sum())
        if n_pred == 0 or n_rec == 0:
            assignment_raw[c] = 0
            continue
        tp = int(((pred_c == 1) & (rec_val_raw == 1)).sum())
        p  = tp / n_pred
        r  = tp / n_rec
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_c  = int(c)
        assignment_raw[c] = 0
    if best_c >= 0:
        assignment_raw[best_c] = 1

    pred = labels_s.map(lambda c: assignment_raw.get(int(c), 0)).astype(int)
    n_pred = int(pred.sum())
    n_rec  = int(rec_val_raw.sum())
    if n_pred == 0 or n_rec == 0:
        return {"val_f1_raw": 0.0, "val_precision_raw": 0.0, "val_recall_raw": 0.0, "val_n_pred": n_pred, "val_n_rec": n_rec}
    tp = int(((pred == 1) & (rec_val_raw == 1)).sum())
    p  = tp / n_pred
    r  = tp / n_rec
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"val_f1_raw": f1, "val_precision_raw": p, "val_recall_raw": r, "val_n_pred": n_pred, "val_n_rec": n_rec, "val_tp": tp}


# ═══════════════════════════════════════════════════════════════════════════════
# F1 — SEED SENSITIVITY with CANONICAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("F1: Seed sensitivity — canonical Z embeddings")
print("=" * 60)

z_train = pd.read_parquet(CANONICAL_EMB / "Z_train.parquet")
z_val   = pd.read_parquet(CANONICAL_EMB / "Z_val.parquet")
z_test  = pd.read_parquet(CANONICAL_EMB / "Z_test.parquet")

val_dates  = pd.DatetimeIndex(z_val["date"])
test_dates = pd.DatetimeIndex(z_test["date"])

seed_rows = []
for seed in SEEDS:
    val_labels, test_labels, sil, n_comp = cluster_on_z(z_train, z_val, z_test, K=4, seed=seed)
    metrics = compute_f1_frozen(val_labels, val_dates, test_labels, test_dates, raw=True)
    assignment_str = str(metrics["assignment"])
    rec_cluster = [c for c, v in metrics["assignment"].items() if v == 1]
    row = {
        "seed": seed,
        "n_pca_components": n_comp,
        "n_predicted_test": metrics["n_predicted_months"],
        "recession_cluster": rec_cluster,
        "f1_tolerance": round(metrics["f1_tolerance"], 4),
        "precision_tolerance": round(metrics["precision_tolerance"], 4),
        "recall_tolerance": round(metrics["recall_tolerance"], 4),
        "f1_raw": round(metrics["f1_raw"], 4),
        "precision_raw": round(metrics["precision_raw"], 4),
        "recall_raw": round(metrics["recall_raw"], 4),
        "silhouette_test": round(sil, 4),
    }
    seed_rows.append(row)
    tag = "← CANONICAL" if seed == 42 else ""
    print(f"  seed={seed:3d}: F1_tol={row['f1_tolerance']:.4f}  F1_raw={row['f1_raw']:.4f}  "
          f"n_pred={row['n_predicted_test']}  n_comp={n_comp}  sil={sil:.3f}  {tag}")

df_seeds = pd.DataFrame(seed_rows)
df_seeds.to_csv(OUT / "f1_seed_sensitivity.csv", index=False)
print(f"\n  → {OUT / 'f1_seed_sensitivity.csv'}")

# ═══════════════════════════════════════════════════════════════════════════════
# F2 — VAL F1 RAW for all 13 encoders
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("F2: VAL F1 raw (no tolerance) — all 13 encoders")
print("=" * 60)

# Load canonical phase_c table for reference (tolerance values)
df_canonical = pd.read_csv(CANONICAL_TABLE)

ENCODERS: dict[str, dict] = {
    "iTransformer": {
        "val": str(K4_B1_VAL),
        "test": str(K4_B1_TEST),
    },
    "linear_ae":    {
        "val": str(ROOT / "results/phase_c_comparison/linear_ae/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_c_comparison/linear_ae/pca_kmeans.parquet"),
    },
    "mlp_ae":       {
        "val": str(ROOT / "results/phase_c_comparison/mlp_ae/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_c_comparison/mlp_ae/pca_kmeans.parquet"),
    },
    "svd":          {
        "val": str(ROOT / "results/phase_c_comparison/svd/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_c_comparison/svd/pca_kmeans.parquet"),
    },
    "windowed_pca": {
        "val": str(ROOT / "results/phase_c_comparison/windowed_pca/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_c_comparison/windowed_pca/pca_kmeans.parquet"),
    },
    "raw_pca":      {
        "val": str(ROOT / "results/phase_c_comparison/raw_pca/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_c_comparison/raw_pca/pca_kmeans.parquet"),
    },
    "bocpd":        {
        "val": str(ROOT / "results/phase_e/bocpd/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/bocpd/ablation/pca_kmeans.parquet"),
    },
    "hamilton_hmm": {
        "val": str(ROOT / "results/phase_e/hamilton_hmm/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/hamilton_hmm/ablation/pca_kmeans.parquet"),
    },
    "moment":       {
        "val": str(ROOT / "results/phase_e/moment/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/moment/ablation/pca_kmeans.parquet"),
    },
    "patchtst":     {
        "val": str(ROOT / "results/phase_e/patchtst/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/patchtst/ablation/pca_kmeans.parquet"),
    },
    "tfc":          {
        "val": str(ROOT / "results/phase_e/tfc/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/tfc/ablation/pca_kmeans.parquet"),
    },
    "timesnet":     {
        "val": str(ROOT / "results/phase_e/timesnet/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/timesnet/ablation/pca_kmeans.parquet"),
    },
    "ts2vec":       {
        "val": str(ROOT / "results/phase_e/ts2vec/ablation/val_pca_kmeans.parquet"),
        "test": str(ROOT / "results/phase_e/ts2vec/ablation/pca_kmeans.parquet"),
    },
}

f2_rows = []
for enc, paths in ENCODERS.items():
    val_path = Path(paths["val"])
    if not val_path.exists():
        print(f"  {enc}: MISSING {val_path}")
        f2_rows.append({"encoder": enc, "error": f"missing {val_path.name}"})
        continue
    df_v = pd.read_parquet(val_path)
    v_labels = df_v["label"].values
    v_dates  = pd.DatetimeIndex(df_v["date"])
    res = compute_val_f1_raw(v_labels, v_dates)

    # Get canonical VAL tolerance F1 from phase_c table if available
    canon_row = df_canonical[df_canonical["encoder"] == enc] if "encoder" in df_canonical.columns else pd.DataFrame()
    if len(canon_row) == 0 and "model" in df_canonical.columns:
        canon_row = df_canonical[df_canonical["model"] == enc]
    val_f1_tol = float(canon_row["val_nber_f1_locked"].values[0]) if len(canon_row) > 0 and "val_nber_f1_locked" in df_canonical.columns else float("nan")

    row = {
        "encoder": enc,
        "val_f1_raw": round(res["val_f1_raw"], 4),
        "val_precision_raw": round(res["val_precision_raw"], 4),
        "val_recall_raw": round(res["val_recall_raw"], 4),
        "val_n_rec": res["val_n_rec"],
        "val_n_pred": res["val_n_pred"],
        "val_tp": res.get("val_tp", 0),
        "val_f1_tolerance_canonical": round(val_f1_tol, 4) if not np.isnan(val_f1_tol) else "n/a",
    }
    f2_rows.append(row)
    print(f"  {enc:15s}: VAL_F1_raw={row['val_f1_raw']:.4f}  n_rec={res['val_n_rec']}  n_pred={res['val_n_pred']}  tp={res.get('val_tp',0)}")

df_f2 = pd.DataFrame(f2_rows)
df_f2.to_csv(OUT / "f2_val_f1_raw.csv", index=False)
print(f"\n  → {OUT / 'f2_val_f1_raw.csv'}")

# ═══════════════════════════════════════════════════════════════════════════════
# F7 — raw_pca tolerance F1 (TEST) for Tabela 3
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("F7: raw_pca tolerance F1 on TEST")
print("=" * 60)

# Use phase_c_comparison raw_pca (189 rows) — slightly longer TEST window (to 2026-02)
rp_test_path = ROOT / "results/phase_c_comparison/raw_pca/pca_kmeans.parquet"
rp_val_path  = ROOT / "results/phase_c_comparison/raw_pca/val_pca_kmeans.parquet"

df_rp_test = pd.read_parquet(rp_test_path)
df_rp_val  = pd.read_parquet(rp_val_path)

rp_test_labels = df_rp_test["label"].values
rp_test_dates  = pd.DatetimeIndex(df_rp_test["date"])
rp_val_labels  = df_rp_val["label"].values
rp_val_dates   = pd.DatetimeIndex(df_rp_val["date"])

# Frozen assignment from VAL
rp_assignment = fit_nber_assignment(rp_val_labels, rp_val_dates, nber_series, lead=LEAD, lag=LAG)

# Tolerance F1 on TEST (lag=2)
rp_tol = nber_overlap_frozen(rp_test_labels, rp_test_dates, nber_series, rp_assignment, lead=LEAD, lag=LAG)
# Raw F1 on TEST (lag=0)
rp_raw = nber_overlap_frozen(rp_test_labels, rp_test_dates, nber_series, rp_assignment, lead=0, lag=0)

print(f"  VAL assignment: {rp_assignment}")
print(f"  TEST n_rows: {len(df_rp_test)}  dates: {rp_test_dates.min()} → {rp_test_dates.max()}")
print(f"  TEST F1 tolerance (lag=2): {rp_tol.f1:.4f}  P={rp_tol.precision:.4f}  R={rp_tol.recall:.4f}  n_rec={rp_tol.n_recession_months}  n_pred={rp_tol.n_predicted_months}")
print(f"  TEST F1 raw (lag=0):       {rp_raw.f1:.4f}  P={rp_raw.precision:.4f}  R={rp_raw.recall:.4f}")

# Get canonical value from table
canon_rp = df_canonical[df_canonical.get("encoder", df_canonical.columns[0]) == "raw_pca"] if "encoder" in df_canonical.columns else pd.DataFrame()
if len(canon_rp) == 0 and "model" in df_canonical.columns:
    canon_rp = df_canonical[df_canonical["model"] == "raw_pca"]
nber_f1_locked_canonical = float(canon_rp["nber_f1_locked"].values[0]) if len(canon_rp) > 0 and "nber_f1_locked" in df_canonical.columns else float("nan")
print(f"  Canonical table nber_f1_locked: {nber_f1_locked_canonical:.4f}")

f7_result = {
    "encoder": "raw_pca",
    "test_n_rows": len(df_rp_test),
    "test_date_min": str(rp_test_dates.min().date()),
    "test_date_max": str(rp_test_dates.max().date()),
    "val_assignment": rp_assignment,
    "f1_tolerance_lag2": round(rp_tol.f1, 4),
    "precision_tolerance": round(rp_tol.precision, 4),
    "recall_tolerance": round(rp_tol.recall, 4),
    "n_recession_months_test": rp_tol.n_recession_months,
    "n_predicted_months": rp_tol.n_predicted_months,
    "f1_raw_lag0": round(rp_raw.f1, 4),
    "precision_raw": round(rp_raw.precision, 4),
    "recall_raw": round(rp_raw.recall, 4),
    "nber_f1_locked_from_canonical_table": round(nber_f1_locked_canonical, 4) if not np.isnan(nber_f1_locked_canonical) else "n/a",
}
with open(OUT / "f7_raw_pca_tolerance_f1.json", "w") as fh:
    json.dump(f7_result, fh, indent=2)
print(f"\n  → {OUT / 'f7_raw_pca_tolerance_f1.json'}")

# ═══════════════════════════════════════════════════════════════════════════════
# F4 — B=1000 BOOTSTRAP STABILITY
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("F4: Bootstrap stability — B=1000")
print("=" * 60)

from tcc_itransformer.pipelines.cluster_stability import run_cluster_stability

B = 1000
RESAMPLE_FRAC = 0.80
ABLATION_DIR = ROOT / "results" / "clustering_ablation" / "W6_d7_K4_b1"
EMB_DIR = CANONICAL_EMB  # Z_test.parquet is in this dir

t0 = time.time()
df_stability = run_cluster_stability(
    ablation_dir=ABLATION_DIR,
    emb_dir=EMB_DIR,
    output=OUT / "f4_bootstrap_stability_b1000.csv",
    n_bootstrap=B,
    resample_frac=RESAMPLE_FRAC,
    seed=42,
)
elapsed = time.time() - t0
print(f"  Completed B={B} in {elapsed:.1f}s")

# Save raw stability output
df_stability.to_csv(OUT / "f4_bootstrap_stability_b1000.csv", index=False)

# Extract pca_kmeans cluster rows for summary
km_stab = df_stability[
    (df_stability["pipeline"] == "pca_kmeans") &
    (df_stability["metric"] == "jaccard")
].copy()

summary_rows = []
for c in sorted(km_stab["cluster"].unique()):
    if c == "all":
        continue
    row_c = km_stab[km_stab["cluster"] == c]
    if len(row_c) == 0:
        continue
    # values are pre-aggregated (mean_value, std_value, etc.)
    summary_rows.append(row_c.iloc[0].to_dict())
    print(f"  C{c}: {row_c.iloc[0].to_dict()}")

# ARI summary
ari_row = df_stability[
    (df_stability["pipeline"] == "pca_kmeans") &
    (df_stability["metric"] == "ari") &
    (df_stability["cluster"] == "all")
]
ari_summary: dict = {}
if not ari_row.empty:
    r = ari_row.iloc[0]
    ari_summary = {
        "ari_mean": round(float(r.get("mean_value", r.get("value", float("nan")))), 4),
        "ari_std": round(float(r.get("std_value", float("nan"))), 4),
        "n_bootstrap": B,
        "resample_frac": RESAMPLE_FRAC,
    }
    print(f"  ARI: mean={ari_summary['ari_mean']:.3f}  std={ari_summary['ari_std']:.4f}")

df_boot_summary = pd.DataFrame(summary_rows)
df_boot_summary.to_csv(OUT / "f4_bootstrap_summary_b1000.csv", index=False)
with open(OUT / "f4_bootstrap_ari_b1000.json", "w") as fh:
    json.dump(ari_summary, fh, indent=2)
print(f"\n  → {OUT / 'f4_bootstrap_stability_b1000.csv'}")
print(f"  → {OUT / 'f4_bootstrap_summary_b1000.csv'}")
print(f"  → {OUT / 'f4_bootstrap_ari_b1000.json'}")

# ═══════════════════════════════════════════════════════════════════════════════
# F5 — K SENSITIVITY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("F5: K-sensitivity — K=3,4,5 with canonical Z")
print("=" * 60)

k_rows = []
for K_val in [3, 4, 5]:
    val_labels_k, test_labels_k, sil_k, n_comp_k = cluster_on_z(
        z_train, z_val, z_test, K=K_val, seed=42
    )
    metrics_k = compute_f1_frozen(val_labels_k, val_dates, test_labels_k, test_dates, raw=True)
    val_metrics_k = compute_val_f1_raw(val_labels_k, val_dates)

    row_k = {
        "K": K_val,
        "n_pca_components": n_comp_k,
        "silhouette_test": round(sil_k, 4),
        "test_f1_tolerance": round(metrics_k["f1_tolerance"], 4),
        "test_f1_raw": round(metrics_k["f1_raw"], 4),
        "test_precision_tolerance": round(metrics_k["precision_tolerance"], 4),
        "test_recall_tolerance": round(metrics_k["recall_tolerance"], 4),
        "test_n_predicted": metrics_k["n_predicted_months"],
        "val_f1_raw": round(val_metrics_k["val_f1_raw"], 4),
        "assignment": str(metrics_k["assignment"]),
        "recession_clusters": [c for c, v in metrics_k["assignment"].items() if v == 1],
    }
    k_rows.append(row_k)
    print(f"  K={K_val}: sil={sil_k:.4f}  F1_tol={row_k['test_f1_tolerance']:.4f}  F1_raw={row_k['test_f1_raw']:.4f}  n_pred={row_k['test_n_predicted']}  n_comp={n_comp_k}")

df_k = pd.DataFrame(k_rows)
df_k.to_csv(OUT / "f5_k_sensitivity.csv", index=False)
print(f"\n  → {OUT / 'f5_k_sensitivity.csv'}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY INDEX
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

summary = {
    "F1_seed_sensitivity": {
        "description": "KMeans seeds {0,1,7,42,123} on canonical iT Z (n_train=485,n_test=185)",
        "canonical_seed42_f1_tolerance": df_seeds.loc[df_seeds["seed"] == 42, "f1_tolerance"].values[0],
        "canonical_seed42_f1_raw": df_seeds.loc[df_seeds["seed"] == 42, "f1_raw"].values[0],
        "all_seeds": df_seeds[["seed","f1_tolerance","f1_raw","n_predicted_test","silhouette_test"]].to_dict(orient="records"),
    },
    "F2_val_f1_raw": {
        "description": "VAL F1 (raw NBER, no tolerance) for all 13 encoders",
        "results": f2_rows,
    },
    "F7_raw_pca_tolerance_f1": f7_result,
    "F4_bootstrap_b1000": {
        "description": f"Jaccard/ARI from B={B} resamples (frac={RESAMPLE_FRAC}) on canonical K4_b1 Z_test, refitting PCA+KMeans per boot",
        "jaccard_per_cluster": summary_rows,
        "ari": ari_summary,
    },
    "F5_k_sensitivity": {
        "description": "K=3,4,5 on canonical iT Z (same PCA, seed=42)",
        "results": k_rows,
    },
}

with open(OUT / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2, default=str)

print(f"\nAll outputs written to: {OUT}")
print()
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
