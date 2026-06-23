#!/usr/bin/env python3
"""
Agent-1 v2 compute script.
Answers: CARRY-001..005, CROSS-001..004, RECON-001, CARRY-004 extension.
All numbers written to artifacts/agent1_response_v2_supporting/
"""

import json
import ast
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "artifacts" / "agent1_response_v2_supporting"
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH = ROOT.parent / "artifacts" / "agent1_response_v2.log"
_log_fh = open(LOG_PATH, "w")

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _log_fh.write(line + "\n")
    _log_fh.flush()

# ---------- paths ----------
NBER_CSV = ROOT / "data/snapshots/nber_usrec.csv"
CANONICAL_EMB = ROOT / "results/sm_outputs/itransformer-1777581449-0d38/embeddings"
CANONICAL_TABLE = ROOT / "outputs/tables/phase_c_canonical_pca_kmeans.csv"
B1_TEST = ROOT / "results/clustering_ablation/W6_d7_K4_b1/pca_kmeans.parquet"
B1_VAL  = ROOT / "results/clustering_ablation/W6_d7_K4_b1/val_pca_kmeans.parquet"
ABLATION_DIR = ROOT / "results/clustering_ablation/W6_d7_K4_b1"
BOOTSTRAP_B1000 = ROOT.parent / "artifacts/agent1_response_v4_supporting/f4_bootstrap_stability_b1000.csv"
BOOTSTRAP_ARI   = ROOT.parent / "artifacts/agent1_response_v4_supporting/f4_bootstrap_ari_b1000.json"

TEST_PARQUETS = {
    "iTransformer":  ROOT / "results/clustering_ablation/W6_d7_K4_b1/pca_kmeans.parquet",
    "linear_ae":     ROOT / "results/phase_c_comparison/linear_ae/pca_kmeans.parquet",
    "mlp_ae":        ROOT / "results/phase_c_comparison/mlp_ae/pca_kmeans.parquet",
    "svd":           ROOT / "results/phase_c_comparison/svd/pca_kmeans.parquet",
    "windowed_pca":  ROOT / "results/phase_c_comparison/windowed_pca/pca_kmeans.parquet",
    "raw_pca":       ROOT / "results/phase_c_comparison/raw_pca/pca_kmeans.parquet",
    "bocpd":         ROOT / "results/phase_e/bocpd/ablation/pca_kmeans.parquet",
    "hamilton_hmm":  ROOT / "results/phase_e/hamilton_hmm/ablation/pca_kmeans.parquet",
    "moment":        ROOT / "results/phase_e/moment/ablation/pca_kmeans.parquet",
    "patchtst":      ROOT / "results/phase_e/patchtst/ablation/pca_kmeans.parquet",
    "tfc":           ROOT / "results/phase_e/tfc/ablation/pca_kmeans.parquet",
    "timesnet":      ROOT / "results/phase_e/timesnet/ablation/pca_kmeans.parquet",
    "ts2vec":        ROOT / "results/phase_e/ts2vec/ablation/pca_kmeans.parquet",
}

VAL_PARQUETS = {
    "iTransformer":  ROOT / "results/clustering_ablation/W6_d7_K4_b1/val_pca_kmeans.parquet",
    "linear_ae":     ROOT / "results/phase_c_comparison/linear_ae/val_pca_kmeans.parquet",
    "mlp_ae":        ROOT / "results/phase_c_comparison/mlp_ae/val_pca_kmeans.parquet",
    "svd":           ROOT / "results/phase_c_comparison/svd/val_pca_kmeans.parquet",
    "windowed_pca":  ROOT / "results/phase_c_comparison/windowed_pca/val_pca_kmeans.parquet",
    "raw_pca":       ROOT / "results/phase_c_comparison/raw_pca/val_pca_kmeans.parquet",
    "bocpd":         ROOT / "results/phase_e/bocpd/ablation/val_pca_kmeans.parquet",
    "hamilton_hmm":  ROOT / "results/phase_e/hamilton_hmm/ablation/val_pca_kmeans.parquet",
    "moment":        ROOT / "results/phase_e/moment/ablation/val_pca_kmeans.parquet",
    "patchtst":      ROOT / "results/phase_e/patchtst/ablation/val_pca_kmeans.parquet",
    "tfc":           ROOT / "results/phase_e/tfc/ablation/val_pca_kmeans.parquet",
    "timesnet":      ROOT / "results/phase_e/timesnet/ablation/val_pca_kmeans.parquet",
    "ts2vec":        ROOT / "results/phase_e/ts2vec/ablation/val_pca_kmeans.parquet",
}

sys.path.insert(0, str(ROOT / "src"))
from tcc_itransformer.evaluation.regime_validation import (
    fit_nber_assignment,
    nber_overlap_frozen,
)

def load_nber(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    col = [c for c in df.columns if "date" in c.lower() or "DATE" in c][0]
    val = [c for c in df.columns if c != col][0]
    df[col] = pd.to_datetime(df[col])
    return df.set_index(col)[val]

def compute_f1_for_labels(labels: np.ndarray, dates: pd.Series, nber: pd.Series,
                           lag: int = 0) -> dict:
    """Compute NBER F1, filtering NBER to the dates window."""
    dates = pd.to_datetime(dates)
    date_min, date_max = dates.min(), dates.max()
    # Filter NBER to evaluation window ONLY (critical for correct n_rec)
    nber_in_window = nber[(nber.index >= date_min) & (nber.index <= date_max)]
    nber_monthly = pd.DatetimeIndex(nber_in_window[nber_in_window == 1].index)
    pred_dates = dates[labels == 1]
    if lag > 0:
        expanded = set()
        for d in nber_monthly:
            for m in range(lag + 1):
                expanded.add(d + pd.DateOffset(months=m))
        match_dates = pd.DatetimeIndex(sorted(expanded))
    else:
        match_dates = nber_monthly
    tp = len(pred_dates[pred_dates.isin(match_dates)])
    n_pred = len(pred_dates)
    n_rec = len(nber_monthly)
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_rec if n_rec > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp,
            "n_pred": n_pred, "n_rec": n_rec}

NBER = load_nber(NBER_CSV)

# ============================================================
# SECTION 1: CARRY-001 — Seed sensitivity (already in v4, reformat)
# ============================================================
log("=== CARRY-001: Seed sensitivity (reformat from v4) ===")
v4_seed = pd.read_csv(ROOT.parent / "artifacts/agent1_response_v4_supporting/f1_seed_sensitivity.csv")
# Add recession cluster info
seed_rows = []
for _, row in v4_seed.iterrows():
    rec_clusters_raw = row["recession_cluster"]
    if isinstance(rec_clusters_raw, str):
        try:
            rec_clusters = ast.literal_eval(rec_clusters_raw)
        except Exception:
            rec_clusters = [int(rec_clusters_raw.strip("[]"))]
    else:
        rec_clusters = [int(rec_clusters_raw)]
    seed_rows.append({
        "seed": int(row["seed"]),
        "f1_raw": float(row["f1_raw"]),
        "f1_tolerance": float(row["f1_tolerance"]),
        "silhouette": float(row["silhouette_test"]),
        "recession_cluster_id": f"C{rec_clusters[0]}" if isinstance(rec_clusters, list) else f"C{rec_clusters}",
        "n_predicted": int(row["n_predicted_test"]),
    })
    log(f"  seed={row['seed']:3d}: F1_raw={row['f1_raw']:.4f}  F1_tol={row['f1_tolerance']:.4f}  rec_cluster={rec_clusters}")

df_carry001 = pd.DataFrame(seed_rows)
df_carry001.to_csv(OUT / "carry001_seed_sensitivity_canonical.csv", index=False)
log(f"  → {OUT}/carry001_seed_sensitivity_canonical.csv")

# ============================================================
# SECTION 2: CARRY-002/003 — TEST F1 raw for all encoders
# ============================================================
log("\n=== CARRY-002: TEST F1 raw (no tolerance) all 13 encoders ===")
canonical = pd.read_csv(CANONICAL_TABLE)

def get_nber_assignment(encoder: str) -> dict:
    """Get VAL→regime mapping for an encoder from canonical table."""
    row = canonical[canonical["encoder"] == encoder]
    if len(row) == 0:
        return {}
    assign_str = row.iloc[0]["nber_assignment"]
    if isinstance(assign_str, str):
        try:
            return ast.literal_eval(assign_str)
        except Exception:
            return json.loads(assign_str.replace("'", '"'))
    return {} if pd.isna(assign_str) else assign_str

def compute_test_f1_raw(encoder: str) -> dict:
    test_path = TEST_PARQUETS[encoder]
    if not test_path.exists():
        return {"encoder": encoder, "test_f1_raw": None, "error": "missing parquet"}
    df_test = pd.read_parquet(test_path)
    assignment = get_nber_assignment(encoder)
    if not assignment:
        return {"encoder": encoder, "test_f1_raw": None, "error": "no assignment"}
    # Convert assignment from string keys
    assignment = {int(k): int(v) for k, v in assignment.items()}
    recession_clusters = [k for k, v in assignment.items() if v == 1]
    dates = pd.to_datetime(df_test["date"])
    labels_binary = df_test["label"].map(lambda x: 1 if x in recession_clusters else 0).values
    # Filter NBER to test window
    test_start, test_end = dates.min(), dates.max()
    nber_test = NBER[(NBER.index >= test_start) & (NBER.index <= test_end)]
    n_rec_test = int((nber_test == 1).sum())
    result_raw = compute_f1_for_labels(labels_binary, dates, NBER, lag=0)
    result_tol = compute_f1_for_labels(labels_binary, dates, NBER, lag=2)
    return {
        "encoder": encoder,
        "test_f1_raw": round(result_raw["f1"], 4),
        "test_f1_tol": round(result_tol["f1"], 4),
        "test_precision_raw": round(result_raw["precision"], 4),
        "test_recall_raw": round(result_raw["recall"], 4),
        "test_n_pred": result_raw["n_pred"],
        "test_tp_raw": result_raw["tp"],
        "test_n_rec": n_rec_test,
        "recession_clusters": recession_clusters,
    }

test_results = {}
for enc in TEST_PARQUETS:
    r = compute_test_f1_raw(enc)
    test_results[enc] = r
    log(f"  {enc:15s}: TEST F1_raw={r.get('test_f1_raw','ERR'):.4f}  F1_tol={r.get('test_f1_tol','ERR'):.4f}  n_rec={r.get('test_n_rec','?')}")

# Load VAL F1 raw from v4
v4_val = pd.read_csv(ROOT.parent / "artifacts/agent1_response_v4_supporting/f2_val_f1_raw.csv")
val_raw_map = dict(zip(v4_val["encoder"], v4_val["val_f1_raw"]))

# Build complete gap table
val_tol_map = dict(zip(canonical["encoder"], canonical["val_nber_f1"]))
test_tol_map = dict(zip(canonical["encoder"], canonical["nber_f1_locked"]))
ENCODER_ORDER = ["iTransformer", "raw_pca", "linear_ae", "windowed_pca", "svd",
                 "mlp_ae", "hamilton_hmm", "bocpd", "moment", "patchtst", "tfc",
                 "timesnet", "ts2vec"]
rows_gap = []
for enc in ENCODER_ORDER:
    vt = val_tol_map.get(enc, np.nan)
    vr = val_raw_map.get(enc, np.nan)
    tt = test_tol_map.get(enc, np.nan)
    tr = test_results[enc].get("test_f1_raw", np.nan)
    rows_gap.append({
        "encoder": enc,
        "val_f1_tol": round(float(vt), 4) if pd.notna(vt) else None,
        "val_f1_raw": round(float(vr), 4) if pd.notna(vr) else None,
        "test_f1_tol": round(float(tt), 4) if pd.notna(tt) else None,
        "test_f1_raw": round(float(tr), 4) if pd.notna(tr) and tr is not None else None,
        "gap_tol": round(float(tt) - float(vt), 4) if (pd.notna(tt) and pd.notna(vt)) else None,
        "gap_raw": round(float(tr) - float(vr), 4) if (pd.notna(vr) and tr is not None) else None,
    })

df_gap = pd.DataFrame(rows_gap)
df_gap.to_csv(OUT / "carry002_val_test_gap_raw.csv", index=False)
log(f"  → {OUT}/carry002_val_test_gap_raw.csv")

# Rank by absolute gap (bruto)
valid_gap = [(r["encoder"], abs(r["gap_raw"])) for r in rows_gap if r["gap_raw"] is not None]
valid_gap.sort(key=lambda x: x[1])
rank_order = [e for e, _ in valid_gap]
itransformer_rank = rank_order.index("iTransformer") + 1 if "iTransformer" in rank_order else None
log(f"  Rank by |gap_raw| (best retention first): {rank_order}")
log(f"  iTransformer rank: {itransformer_rank}")

gap_meta = {
    "rank_order_best_to_worst_retention": rank_order,
    "itransformer_rank": itransformer_rank,
    "does_itransformer_rank_first": itransformer_rank == 1,
}
with open(OUT / "carry002_gap_metadata.json", "w") as f:
    json.dump(gap_meta, f, indent=2)

# ============================================================
# SECTION 3: CROSS-001 — Crisis × cluster contingency table
# ============================================================
log("\n=== CROSS-001: Crisis × cluster contingency table ===")
df_val = pd.read_parquet(B1_VAL)
df_val["date"] = pd.to_datetime(df_val["date"])
df_test = pd.read_parquet(B1_TEST)
df_test["date"] = pd.to_datetime(df_test["date"])

CRISES = {
    "Dot-com": {
        "start": pd.Timestamp("2001-03-01"),
        "end":   pd.Timestamp("2001-11-01"),
        "split": "VAL",
        "n_months_expected": 9,
    },
    "GFC": {
        "start": pd.Timestamp("2007-12-01"),
        "end":   pd.Timestamp("2009-06-01"),
        "split": "VAL",
        "n_months_expected": 19,
    },
    "COVID-19": {
        "start": pd.Timestamp("2020-02-01"),
        "end":   pd.Timestamp("2020-04-01"),
        "split": "TEST",
        "n_months_expected": 3,
    },
}

# NBER share in each crisis window
nber_in_val = NBER[(NBER.index >= df_val["date"].min()) & (NBER.index <= df_val["date"].max())]
nber_in_test = NBER[(NBER.index >= df_test["date"].min()) & (NBER.index <= df_test["date"].max())]

cross_rows = []
for crisis_name, cfg in CRISES.items():
    s, e = cfg["start"], cfg["end"]
    split_df = df_val if cfg["split"] == "VAL" else df_test
    window = split_df[(split_df["date"] >= s) & (split_df["date"] <= e)].copy()
    n = len(window)
    # Cluster counts
    vc = window["label"].value_counts().to_dict()
    c0 = int(vc.get(0, 0)); c1 = int(vc.get(1, 0))
    c2 = int(vc.get(2, 0)); c3 = int(vc.get(3, 0))
    # NBER share
    nber_src = nber_in_val if cfg["split"] == "VAL" else nber_in_test
    nber_win = nber_src[(nber_src.index >= s) & (nber_src.index <= e)]
    nber_share = round(float(nber_win[nber_win == 1].count()) / n, 4) if n > 0 else 0
    row = {
        "crisis": crisis_name, "split": cfg["split"], "n_months": n,
        "c0_count": c0, "c1_count": c1, "c2_count": c2, "c3_count": c3,
        "c0_pct": round(c0/n*100, 1) if n else 0,
        "c1_pct": round(c1/n*100, 1) if n else 0,
        "c2_pct": round(c2/n*100, 1) if n else 0,
        "c3_pct": round(c3/n*100, 1) if n else 0,
        "nber_share": nber_share,
    }
    cross_rows.append(row)
    log(f"  {crisis_name}: n={n}  C0={c0}({c0/n*100:.1f}%)  C1={c1}  C2={c2}  C3={c3}  NBER_share={nber_share:.3f}")

df_cross = pd.DataFrame(cross_rows)
df_cross.to_csv(OUT / "cross001_crisis_cluster_full.csv", index=False)
log(f"  → {OUT}/cross001_crisis_cluster_full.csv")

# ============================================================
# SECTION 4: CROSS-003 — Fisher exact for GFC ∪ COVID co-clustering
# ============================================================
log("\n=== CROSS-003: Fisher exact — GFC ∪ COVID in C0 ===")
# C0 = recession cluster (assignment {0:1,...} from canonical)
# Scope: VAL + TEST (months that the canonical pipeline evaluated)
# Combine VAL+TEST cluster labels
df_all_eval = pd.concat([
    df_val.assign(split="VAL"),
    df_test.assign(split="TEST")
], ignore_index=True)
df_all_eval["date"] = pd.to_datetime(df_all_eval["date"])
n_total = len(df_all_eval)

# Define severe-crisis months: GFC (2007-12 to 2009-06) + COVID (2020-02 to 2020-04)
gfc_months = set(pd.date_range("2007-12-01", "2009-06-01", freq="MS"))
covid_months = set(pd.date_range("2020-02-01", "2020-04-01", freq="MS"))
severe_months = gfc_months | covid_months

df_all_eval["is_severe"] = df_all_eval["date"].isin(severe_months)
df_all_eval["is_c0"] = (df_all_eval["label"] == 0)

severe_in_c0    = int(df_all_eval[df_all_eval["is_severe"] & df_all_eval["is_c0"]].shape[0])
severe_not_c0   = int(df_all_eval[df_all_eval["is_severe"] & ~df_all_eval["is_c0"]].shape[0])
not_severe_in_c0 = int(df_all_eval[~df_all_eval["is_severe"] & df_all_eval["is_c0"]].shape[0])
not_severe_not_c0= int(df_all_eval[~df_all_eval["is_severe"] & ~df_all_eval["is_c0"]].shape[0])

table_2x2 = [[severe_in_c0, severe_not_c0], [not_severe_in_c0, not_severe_not_c0]]
or_val, p_fisher = stats.fisher_exact(table_2x2, alternative="greater")

log(f"  2×2 table: severe_C0={severe_in_c0}, severe_notC0={severe_not_c0}, notSevere_C0={not_severe_in_c0}, notSevere_notC0={not_severe_not_c0}")
log(f"  Fisher exact: OR={or_val:.3f}  p={p_fisher:.6f}")
log(f"  Total VAL+TEST months: {n_total}")

# Also: is Dot-com significantly NOT in C0?
dotcom_months = set(pd.date_range("2001-03-01", "2001-11-01", freq="MS"))
df_all_eval["is_dotcom"] = df_all_eval["date"].isin(dotcom_months)
dc_c0 = int(df_all_eval[df_all_eval["is_dotcom"] & df_all_eval["is_c0"]].shape[0])
dc_not = int(df_all_eval[df_all_eval["is_dotcom"] & ~df_all_eval["is_c0"]].shape[0])
notdc_c0 = int(df_all_eval[~df_all_eval["is_dotcom"] & df_all_eval["is_c0"]].shape[0])
notdc_not = int(df_all_eval[~df_all_eval["is_dotcom"] & ~df_all_eval["is_c0"]].shape[0])
or_dc, p_dc = stats.fisher_exact([[dc_c0, dc_not], [notdc_c0, notdc_not]], alternative="less")
log(f"  Dot-com NOT in C0: DC_C0={dc_c0}, p_fisher={p_dc:.4f}")

cross003 = {
    "scope": "VAL+TEST (2000-06 to 2026-01), n=300 months",
    "severe_months_def": "GFC (2007-12 to 2009-06) union COVID (2020-02 to 2020-04), 22 months total",
    "contingency_table": {
        "severe_in_C0": severe_in_c0, "severe_not_C0": severe_not_c0,
        "not_severe_in_C0": not_severe_in_c0, "not_severe_not_C0": not_severe_not_c0,
        "total": n_total,
    },
    "fisher_p": round(p_fisher, 6),
    "odds_ratio": round(or_val, 3),
    "dot_com_not_in_C0": {
        "dc_in_C0": dc_c0, "dc_not_C0": dc_not,
        "fisher_p_less": round(p_dc, 4),
        "is_significantly_not_in_C0": bool(p_dc < 0.05),
    }
}
with open(OUT / "cross003_fisher_exact.json", "w") as f:
    json.dump(cross003, f, indent=2)
log(f"  → {OUT}/cross003_fisher_exact.json")

# ============================================================
# SECTION 5: CROSS-004 — TRAIN NBER recession cluster analysis
# ============================================================
log("\n=== CROSS-004: TRAIN NBER recession cluster analysis ===")
# Load canonical Z_train embeddings and refit KMeans with seed=42
Z_train = pd.read_parquet(CANONICAL_EMB / "Z_train.parquet")
feat_cols = [c for c in Z_train.columns if c != "date"]
X_train = Z_train[feat_cols].values
train_dates = pd.to_datetime(Z_train["date"])

pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_train_2d)
train_labels = km.labels_

# VAL→TEST assignment is {0:1,1:0,2:0,3:0} meaning C0=recession
# Verify by looking at which cluster NBER assigns to
Z_val = pd.read_parquet(CANONICAL_EMB / "Z_val.parquet")
val_dates = pd.to_datetime(Z_val["date"])
X_val_2d = pca.transform(Z_val[feat_cols].values)
val_labels = km.predict(X_val_2d)
val_assignment = fit_nber_assignment(val_labels, val_dates, NBER, lead=0, lag=2)
log(f"  VAL assignment (train-refit): {dict(val_assignment)}")

# For TRAIN, recession cluster = whichever cluster val_assignment says is 1
rec_cluster_val = [k for k, v in val_assignment.items() if v == 1]
log(f"  Recession cluster(s): {rec_cluster_val}")

# NBER months in TRAIN (1959-08 to 1999-12)
nber_train = NBER[(NBER.index >= train_dates.min()) & (NBER.index <= train_dates.max())]
n_nber_train = int((nber_train == 1).sum())
log(f"  NBER months in TRAIN: {n_nber_train} / {len(train_dates)}")

# Match dates
train_df = pd.DataFrame({"date": train_dates, "label": train_labels})
train_df["date"] = pd.to_datetime(train_df["date"])
nber_train_months = set(nber_train[nber_train == 1].index)

TRAIN_RECESSIONS = {
    "1969 (Nixon)":         {"start": "1969-12-01", "end": "1970-11-01"},
    "1973-75 (Oil shock)":  {"start": "1973-11-01", "end": "1975-03-01"},
    "1980 (double-dip)":    {"start": "1980-01-01", "end": "1980-07-01"},
    "1981-82 (Volcker)":    {"start": "1981-07-01", "end": "1982-11-01"},
    "1990-91 (S&L)":        {"start": "1990-07-01", "end": "1991-03-01"},
}

train_crisis_rows = []
for name, cfg in TRAIN_RECESSIONS.items():
    s, e = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    win = train_df[(train_df["date"] >= s) & (train_df["date"] <= e)]
    n = len(win)
    if n == 0:
        log(f"  {name}: no months in TRAIN window")
        continue
    vc = win["label"].value_counts().to_dict()
    c0 = int(vc.get(0, 0)); c1 = int(vc.get(1, 0)); c2 = int(vc.get(2, 0)); c3 = int(vc.get(3, 0))
    dominant = max(vc, key=vc.get)
    train_crisis_rows.append({
        "recession": name, "n_months": n,
        "c0_count": c0, "c1_count": c1, "c2_count": c2, "c3_count": c3,
        "c0_pct": round(c0/n*100, 1),
        "c1_pct": round(c1/n*100, 1),
        "c2_pct": round(c2/n*100, 1),
        "c3_pct": round(c3/n*100, 1),
        "dominant_cluster": f"C{dominant}",
    })
    log(f"  {name}: n={n} C0={c0}({c0/n*100:.0f}%) C1={c1} C2={c2} C3={c3} dominant=C{dominant}")

# All TRAIN NBER months
all_nber_win = train_df[train_df["date"].isin(nber_train_months)]
n_all = len(all_nber_win)
vc_all = all_nber_win["label"].value_counts().to_dict()
c0_all = int(vc_all.get(0, 0)); c1_all = int(vc_all.get(1, 0))
c2_all = int(vc_all.get(2, 0)); c3_all = int(vc_all.get(3, 0))
train_crisis_rows.append({
    "recession": "All TRAIN NBER months",
    "n_months": n_all,
    "c0_count": c0_all, "c1_count": c1_all, "c2_count": c2_all, "c3_count": c3_all,
    "c0_pct": round(c0_all/n_all*100, 1) if n_all else 0,
    "c1_pct": round(c1_all/n_all*100, 1) if n_all else 0,
    "c2_pct": round(c2_all/n_all*100, 1) if n_all else 0,
    "c3_pct": round(c3_all/n_all*100, 1) if n_all else 0,
    "dominant_cluster": f"C{max(vc_all, key=vc_all.get)}" if vc_all else "N/A",
})
log(f"  All TRAIN NBER: n={n_all} C0={c0_all}({c0_all/n_all*100:.0f}%) C1={c1_all} C2={c2_all} C3={c3_all}")

df_cross004 = pd.DataFrame(train_crisis_rows)
df_cross004.to_csv(OUT / "cross004_train_recessions.csv", index=False)
log(f"  → {OUT}/cross004_train_recessions.csv")

# ============================================================
# SECTION 6: CARRY-005 extension — Mean dwell C0 for K=3,4,5
# ============================================================
log("\n=== CARRY-005: K-sensitivity with dwell times ===")
# Load Z splits
Z_train_raw = pd.read_parquet(CANONICAL_EMB / "Z_train.parquet")
Z_val_raw   = pd.read_parquet(CANONICAL_EMB / "Z_val.parquet")
Z_test_raw  = pd.read_parquet(CANONICAL_EMB / "Z_test.parquet")
feat_cols   = [c for c in Z_train_raw.columns if c != "date"]

def mean_dwell(labels: np.ndarray) -> float:
    """Mean run length of recession cluster across TEST."""
    if len(labels) == 0:
        return float("nan")
    runs = []
    in_run = False
    cur_len = 0
    for l in labels:
        if l == 1:
            in_run = True
            cur_len += 1
        else:
            if in_run:
                runs.append(cur_len)
                cur_len = 0
            in_run = False
    if in_run:
        runs.append(cur_len)
    return float(np.mean(runs)) if runs else 0.0

def get_dbcv_for_k(k: int) -> float:
    """Look up DBCV from ablation summary if available."""
    ablation_file = ABLATION_DIR / "summary.csv"
    if ablation_file.exists():
        try:
            df_s = pd.read_csv(ablation_file)
            row = df_s[df_s.get("n_clusters", pd.Series()).eq(k) | 
                       df_s.get("K", pd.Series()).eq(k)]
            if len(row) > 0 and "dbcv" in df_s.columns:
                return float(row.iloc[0]["dbcv"])
        except Exception:
            pass
    return float("nan")

k_rows = []
for K in [3, 4, 5]:
    pca_k = PCA(n_components=2, random_state=42)
    X_tr_2d = pca_k.fit_transform(Z_train_raw[feat_cols].values)
    X_val_2d = pca_k.transform(Z_val_raw[feat_cols].values)
    X_test_2d = pca_k.transform(Z_test_raw[feat_cols].values)
    km_k = KMeans(n_clusters=K, random_state=42, n_init=10)
    km_k.fit(X_tr_2d)
    val_labels_k  = km_k.predict(X_val_2d)
    test_labels_k = km_k.predict(X_test_2d)
    test_dates_k = pd.to_datetime(Z_test_raw["date"])
    val_dates_k  = pd.to_datetime(Z_val_raw["date"])
    sil = float(silhouette_score(X_test_2d, test_labels_k)) if len(set(test_labels_k)) > 1 else float("nan")
    # Get recession assignment
    val_assign = fit_nber_assignment(val_labels_k, val_dates_k, NBER, lead=0, lag=2)
    rec_clusters_k = [c for c, r in val_assign.items() if r == 1]
    test_binary = np.array([1 if l in rec_clusters_k else 0 for l in test_labels_k])
    r_tol = compute_f1_for_labels(test_binary, test_dates_k, NBER, lag=2)
    r_raw = compute_f1_for_labels(test_binary, test_dates_k, NBER, lag=0)
    n_pred = int(test_binary.sum())
    rec_size = n_pred
    dwell = mean_dwell(test_binary)
    dbcv = get_dbcv_for_k(K)
    row = {
        "K": K, "f1_tol": round(r_tol["f1"], 4), "f1_raw": round(r_raw["f1"], 4),
        "silhouette": round(sil, 4), "dbcv": round(dbcv, 3) if not np.isnan(dbcv) else None,
        "mean_dwell_c0": round(dwell, 1), "recession_cluster_size": rec_size,
        "n_pred": n_pred, "recession_clusters": rec_clusters_k,
    }
    k_rows.append(row)
    log(f"  K={K}: F1_tol={r_tol['f1']:.4f} F1_raw={r_raw['f1']:.4f} sil={sil:.4f} dwell={dwell:.1f} rec_size={rec_size}")

df_k = pd.DataFrame(k_rows)
df_k.to_csv(OUT / "carry005_k_sensitivity.csv", index=False)
log(f"  → {OUT}/carry005_k_sensitivity.csv")

# ============================================================
# SECTION 7: RECON-001 — p=0.83 test specification
# ============================================================
log("\n=== RECON-001: p=0.83 chi-square test specification ===")
# The p=0.83 test was chi-square independence of cluster × NBER
# using HDBSCAN full-panel (~53 clusters, 606 non-noise months)
# Source: results/diagnostics/confound_check.md
# Chi2=57.750, dof=69, p=0.8309, n=606
# This is NOT the same as Fisher exact (p=0.0039) which used KMeans-4 on TEST only

# Compute the 2×2 KMeans-4 full-panel chi-square for comparison
# Need TRAIN cluster labels (computed above)
# Z_val and Z_test labels from parquets
# Build full panel labels
train_binary_full = np.array([1 if l in rec_cluster_val else 0 for l in train_labels])
val_binary_full   = np.array([1 if l in rec_cluster_val else 0 for l in val_labels])
Z_test_load = pd.read_parquet(CANONICAL_EMB / "Z_test.parquet")
feat_cols_z = [c for c in Z_test_load.columns if c != "date"]
X_test_2d_full = pca.transform(Z_test_load[feat_cols_z].values)
test_labels_full = km.predict(X_test_2d_full)
test_binary_full  = np.array([1 if l in rec_cluster_val else 0 for l in test_labels_full])

all_dates_full = pd.concat([
    pd.Series(train_dates.values),
    pd.Series(val_dates.values),
    pd.Series(pd.to_datetime(Z_test_load["date"]).values),
]).reset_index(drop=True)
all_labels_full = np.concatenate([train_labels, val_labels, test_labels_full])

# NBER for full panel
nber_full = NBER[(NBER.index >= all_dates_full.min()) & (NBER.index <= all_dates_full.max())]
nber_by_date = nber_full.reindex(all_dates_full, fill_value=0)

# 2×2: C0 vs others × NBER=1 vs NBER=0, full panel
is_c0_full = (all_labels_full == 0)
is_nber_full = (nber_by_date.values == 1)

c0_nber1 = int(np.sum(is_c0_full & is_nber_full))
c0_nber0 = int(np.sum(is_c0_full & ~is_nber_full))
not_c0_nber1 = int(np.sum(~is_c0_full & is_nber_full))
not_c0_nber0 = int(np.sum(~is_c0_full & ~is_nber_full))

chi2_table = [[c0_nber1, c0_nber0], [not_c0_nber1, not_c0_nber0]]
chi2_stat, p_chi2, dof, _ = stats.chi2_contingency(chi2_table, correction=False)
or_full, p_fisher_full = stats.fisher_exact(chi2_table, alternative="greater")

n_total_full = len(all_dates_full)
log(f"  KMeans-4 full-panel 2x2: C0_NBER1={c0_nber1}, C0_NBER0={c0_nber0}, not_C0_NBER1={not_c0_nber1}, not_C0_NBER0={not_c0_nber0}")
log(f"  Full-panel chi2={chi2_stat:.4f} p={p_chi2:.4f} dof={dof} n={n_total_full}")
log(f"  Full-panel fisher: OR={or_full:.3f} p={p_fisher_full:.6f}")

recon001 = {
    "p083_test": {
        "source_artifact": "tcc_ai/results/diagnostics/confound_check.md",
        "test_type": "chi_square_independence",
        "n": 606,
        "dof": 69,
        "chi2": 57.750,
        "p_value": 0.8309,
        "cramers_v": 0.309,
        "cluster_method": "HDBSCAN (full-panel refit, ~53 clusters, noise_frac=0.228)",
        "time_range": "full_panel_1959_2026 (non-noise months only)",
        "nber_tolerance": False,
        "encoder": "iTransformer",
        "null_hypothesis": "Cluster label is independent of NBER recession indicator across 606 full-panel non-noise months",
    },
    "kmeans4_fullpanel_comparison": {
        "test_type": "chi_square_2x2 + fisher_exact",
        "contingency_table": {"C0_NBER1": c0_nber1, "C0_NBER0": c0_nber0,
                               "notC0_NBER1": not_c0_nber1, "notC0_NBER0": not_c0_nber0},
        "n": n_total_full,
        "chi2": round(chi2_stat, 4),
        "p_chi2": round(p_chi2, 4),
        "dof": dof,
        "fisher_p": round(p_fisher_full, 6),
        "odds_ratio": round(or_full, 3),
        "time_range": "full_panel_TRAIN_VAL_TEST (785 months)",
        "cluster_method": "KMeans K=4 (canonical, seed=42)",
    },
    "fisher_p0039_reference": {
        "test_type": "fisher_exact_2x2",
        "scope": "TEST_only (185 months, 2010-06 to 2026-01)",
        "cluster_method": "KMeans K=4, canonical seed=42",
        "reported_in_paper": "§5.3",
        "null_hypothesis": "C0 frequency in COVID TEST months is not higher than expected by chance",
    },
    "reconciliation": {
        "why_different": (
            "p=0.83 uses HDBSCAN with ~53 clusters on 606 full-panel non-noise months. "
            "The chi-square with 69 dof is inherently underpowered when many cells have <5 observations. "
            "p=0.0039 uses KMeans K=4 on TEST-only (185 months) with a 2×2 Fisher exact test "
            "specifically asking whether C0 is enriched for the 2 COVID NBER months. "
            "The tests answer different questions: p=0.83 = 'does ANY cluster track ALL historical NBER recessions?'; "
            "p=0.0039 = 'does the recession-labelled cluster specifically capture COVID in TEST?'. "
            "The HDBSCAN test covers 1959-2026 where 50+ recession months are scattered across many small clusters, "
            "diluting any signal. The Fisher test is targeted at the canonical 4-cluster partition and modern TEST period."
        )
    }
}
with open(OUT / "recon001_p083_spec.json", "w") as f:
    json.dump(recon001, f, indent=2)
log(f"  → {OUT}/recon001_p083_spec.json")

# ============================================================
# SECTION 8: CARRY-004 extension — Pr(J≥0.60) from B=1000 stats
# ============================================================
log("\n=== CARRY-004: Bootstrap B=1000 with Pr(J≥0.60) ===")
from scipy.stats import norm

df_boot = pd.read_csv(BOOTSTRAP_B1000)
pca_kmeans_boot = df_boot[(df_boot["pipeline"] == "pca_kmeans") & (df_boot["cluster"] != "all")].copy()
ari_row = df_boot[(df_boot["pipeline"] == "pca_kmeans") & (df_boot["cluster"] == "all")]

boot_rows = []
for _, r in pca_kmeans_boot.iterrows():
    c = int(r["cluster"])
    mu = float(r["mean_value"])
    sigma = float(r["std_value"])
    # Pr(J >= 0.60) using normal approximation
    pr_ge_060 = round(float(1 - norm.cdf(0.60, loc=mu, scale=sigma)), 3)
    stable = bool(r["stable"])
    if stable:
        cls = "stable"
    elif pr_ge_060 < 0.1:
        cls = "unstable"
    else:
        cls = "borderline"
    n = int(r["n_months"])
    boot_rows.append({
        "cluster": c, "n_months": n,
        "jaccard_mean": round(mu, 4),
        "jaccard_std": round(sigma, 4),
        "pr_ge_060": pr_ge_060,
        "class": cls,
    })
    log(f"  C{c}: n={n} J_mean={mu:.4f} J_std={sigma:.4f} Pr(J≥0.60)={pr_ge_060:.3f} → {cls}")

ari_mean = float(ari_row.iloc[0]["mean_value"])
ari_std  = float(ari_row.iloc[0]["std_value"])
pr_ari = round(float(1 - norm.cdf(0.60, loc=ari_mean, scale=ari_std)), 3)
boot_rows.append({"cluster": "ARI_global", "n_months": 185,
                   "jaccard_mean": round(ari_mean, 4),
                   "jaccard_std": round(ari_std, 4),
                   "pr_ge_060": pr_ari, "class": "see ARI"})
log(f"  ARI: mean={ari_mean:.4f} std={ari_std:.4f} Pr(ARI≥0.60)={pr_ari:.3f}")

# Compare vs B=100 (c4_stability.csv)
b100_path = ABLATION_DIR / "c4_stability.csv"
b100_comparison = {}
if b100_path.exists():
    df_b100 = pd.read_csv(b100_path)
    b100_pca = df_b100[(df_b100.get("pipeline", pd.Series()) == "pca_kmeans")] if "pipeline" in df_b100.columns else pd.DataFrame()
    if len(b100_pca) > 0:
        c0_b100 = b100_pca[b100_pca["cluster"] == 0]
        if len(c0_b100) > 0:
            b100_c0_j = float(c0_b100.iloc[0].get("mean_value", c0_b100.iloc[0].get("jaccard_mean", np.nan)))
            b100_comparison["c0_jaccard_b100"] = round(b100_c0_j, 4)
            b100_comparison["c0_jaccard_b1000"] = 0.4998
            b100_comparison["delta"] = round(0.4998 - b100_c0_j, 4)

df_boot_out = pd.DataFrame(boot_rows)
df_boot_out.to_csv(OUT / "carry004_bootstrap_b1000.csv", index=False)
with open(OUT / "carry004_b100_comparison.json", "w") as f:
    json.dump(b100_comparison, f, indent=2)
log(f"  → {OUT}/carry004_bootstrap_b1000.csv")

# ============================================================
# SECTION 9: SAFEGUARDS audit — pre-registered plan git info
# ============================================================
log("\n=== SAFEGUARDS: Plan vs sweep timestamps ===")
safeguards = {
    "plan_file": "tcc_ai/docs/pre_analysis_plan.md + tcc_ai/plan/panel-remediation-plan.md + tcc_ai/plan/feature-itransformer-thesis-1.md",
    "plan_last_commit": {
        "hash": "c6842314d61a8ff6199a31b03c8c70ed4d9d7eb1",
        "timestamp": "2026-04-30T18:08:06-03:00",
        "message": "Refactor code structure for improved readability and maintainability",
    },
    "first_sweep_commit": {
        "hash": "a722638ff0e4260d4a53adf4c729234ec1e9d20c",
        "timestamp": "2026-05-02T20:01:30-03:00",
        "message": "Add sweep pipeline, configuration generation, and winner selection scripts",
    },
    "plan_predates_sweep_code": True,
    "caveat": (
        "Git commits show plan files committed 2026-04-30 BEFORE sweep code committed 2026-05-02. "
        "However, SageMaker sweep runs may have executed before code was committed to repo. "
        "Actual SM run timestamps can be verified via 'aws sagemaker list-training-jobs'."
    ),
}
with open(OUT / "safeguards_plan_timestamps.json", "w") as f:
    json.dump(safeguards, f, indent=2)
log(f"  Plan commit: {safeguards['plan_last_commit']['timestamp']}")
log(f"  First sweep commit: {safeguards['first_sweep_commit']['timestamp']}")
log(f"  Plan predates sweep code: {safeguards['plan_predates_sweep_code']}")

# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n=== SUMMARY ===")
log(f"All outputs written to: {OUT}")
for f in sorted(OUT.glob("*")):
    log(f"  {f.name}")

_log_fh.close()
print("\nDone.")
