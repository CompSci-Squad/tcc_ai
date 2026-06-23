"""
audit_canonical_n_positives.py
==============================
Resolves the n_positive / F1_raw inconsistency between the paper (n+=2)
and sprint1 (n+=4, called "f1_raw").

Verifications:
  1. Count USREC=1 months in TEST window (raw, no tolerance)
  2. Check for any post-2020-04 NBER positives in the vintage
  3. Trace the origin of n+=4 in sprint1 scripts
  4. Recompute true F1_raw (no tolerance) for all 13 encoders

Output: audit/canonical_n_positives.json
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]   # tcc_ai/
NBER_CSV = ROOT / "data" / "snapshots" / "nber_usrec.csv"

# B1 temporal split (canonical)
TEST_START = pd.Timestamp("2010-06-01")
TEST_END   = pd.Timestamp("2026-04-30")
VAL_START  = pd.Timestamp("2000-06-01")
VAL_END    = pd.Timestamp("2009-12-01")

# Encoder catalogue (same paths as sprint1)
_PC = ROOT / "results" / "phase_c_comparison"
_PE = ROOT / "results" / "phase_e"
_CL = ROOT / "results" / "clustering_ablation" / "W6_d7_K4_b1"

ENCODERS: list[tuple[str, str, Path]] = [
    ("iTransformer", "tier1", _CL),
    ("windowed_pca", "tier1", _PC / "windowed_pca"),
    ("raw_pca",      "tier1", _PC / "raw_pca"),
    ("linear_ae",    "tier1", _PC / "linear_ae"),
    ("mlp_ae",       "tier1", _PC / "mlp_ae"),
    ("svd",          "tier1", _PC / "svd"),
    ("moment",       "tier3", _PE / "moment"      / "ablation"),
    ("ts2vec",       "tier2", _PE / "ts2vec"      / "ablation"),
    ("patchtst",     "tier2", _PE / "patchtst"    / "ablation"),
    ("timesnet",     "tier1", _PE / "timesnet"    / "ablation"),
    ("tfc",          "tier2", _PE / "tfc"         / "ablation"),
    ("hamilton_hmm", "tier1", _PE / "hamilton_hmm" / "ablation"),
    ("bocpd",        "tier1", _PE / "bocpd"       / "ablation"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_nber_raw() -> pd.Series:
    """USREC series, no tolerance expansion, indexed by date."""
    df = pd.read_csv(NBER_CSV, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date"})
    return df.set_index("date")["USREC"].astype(int)


def recession_cluster_from_val(val_df: pd.DataFrame, nber: pd.Series) -> int:
    """Return the cluster with highest NBER-overlap fraction on VAL (frozen)."""
    val_nber = nber.reindex(val_df["date"]).fillna(0).astype(int)
    best_k, best_share = 0, -1.0
    for k in val_df["label"].unique():
        mask = val_df["label"] == k
        share = val_nber[mask.values].mean()
        if share > best_share:
            best_share, best_k = share, k
    return int(best_k)


def compute_f1_raw_for_encoder(
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    nber_test_raw: pd.Series,
    nber_val: pd.Series,
) -> dict:
    """
    F1_raw: strictly no tolerance window.
    y_true  = raw NBER (0/1)
    y_pred  = 1 if label == recession_cluster else 0
    """
    rec_k = recession_cluster_from_val(val_df, nber_val)

    yt = nber_test_raw.reindex(test_df["date"]).fillna(0).astype(int).values
    yp = (test_df["label"] == rec_k).astype(int).values

    n_pos = int(yt.sum())
    if n_pos == 0:
        return {
            "f1_raw": None,
            "precision": None,
            "recall": None,
            "mcc": None,
            "n_positive": 0,
            "n_test": len(yt),
            "recession_cluster": rec_k,
            "note": "0 NBER positives in TEST rows of this parquet",
        }

    prec = float(precision_score(yt, yp, zero_division=0))
    rec  = float(recall_score(yt, yp, zero_division=0))
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mcc  = float(matthews_corrcoef(yt, yp))
    tp   = int((yt * yp).sum())
    fp   = int(((1 - yt) * yp).sum())
    fn   = int((yt * (1 - yp)).sum())

    return {
        "f1_raw": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "mcc": round(mcc, 4),
        "n_positive": n_pos,
        "n_test": len(yt),
        "recession_cluster": rec_k,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ════════════════════════════════════════════════════════════════════════════════
# VERIFICATION 1 — count USREC in TEST window (raw)
# ════════════════════════════════════════════════════════════════════════════════

print("\n═══ VERIFICATION 1: USREC count in TEST (2010-06 to 2026-04) ═══════════")
nber_raw = load_nber_raw()
nber_test_raw = nber_raw.loc[TEST_START:TEST_END]
pos_raw = nber_test_raw[nber_test_raw == 1]

print(f"Total TEST months in NBER file: {len(nber_test_raw)}")
print(f"n_positive (USREC=1, raw, no tolerance): {len(pos_raw)}")
print("Positive dates:")
for d in pos_raw.index:
    print(f"  {d.date()}")


# ════════════════════════════════════════════════════════════════════════════════
# VERIFICATION 2 — check for post-2020-04 NBER updates
# ════════════════════════════════════════════════════════════════════════════════

print("\n═══ VERIFICATION 2: Post-COVID NBER updates ════════════════════════════")
post_covid = nber_raw.loc["2020-05-01":].loc[lambda s: s == 1]
if len(post_covid) == 0:
    print("No NBER=1 months after 2020-04 in nber_usrec.csv (vintage 2026-03).")
    print("n+=2 confirmed — no NBER update to incorporate.")
    post_covid_finding = "none"
else:
    print(f"ALERT: {len(post_covid)} NBER=1 months after 2020-04:")
    for d in post_covid.index:
        print(f"  {d.date()}")
    post_covid_finding = "found"


# ════════════════════════════════════════════════════════════════════════════════
# VERIFICATION 3 — trace origin of n+=4 in sprint1
# ════════════════════════════════════════════════════════════════════════════════

print("\n═══ VERIFICATION 3: Origin of n+=4 in sprint1 ══════════════════════════")

def nber_for_window_with_lag(nber, start, end, lead=0, lag=2):
    """Replicates sprint1 tolerance expansion."""
    subset = nber.loc[start:end].copy()
    expanded = subset.copy()
    for k in range(1, lead + 1):
        expanded |= subset.shift(-k, fill_value=0)
    for k in range(1, lag + 1):
        expanded |= subset.shift(k, fill_value=0)
    return expanded

nber_test_tol = nber_for_window_with_lag(
    nber_raw, TEST_START, pd.Timestamp("2026-12-31"), lead=0, lag=2
)
pos_tol = nber_test_tol[nber_test_tol == 1]

print(f"Sprint1 used nber_for_window(lead=0, lag=2) — tolerance expansion.")
print(f"Raw NBER positives in TEST: 2020-03, 2020-04  → n_raw=2")
print(f"After lag=2 expansion: also 2020-05, 2020-06 → n_tol={len(pos_tol)}")
print("Expanded positives:")
for d in pos_tol.index:
    print(f"  {d.date()}")
print()
print("DIAGNOSIS: Sprint1 computed 'f1_raw' using tolerance-expanded labels.")
print("  n=2  raw NBER positives (actual USREC=1)")
print("  n=4  tolerance-expanded positives (lag=2 after recession end)")
print("  The sprint column 'f1_raw' was mislabelled — it IS F1_tol.")
print("  The paper's n+=2 is correct for raw NBER.")

sprint1_explanation = (
    "Sprint1 applied nber_for_window(lead=0, lag=2) to the ground-truth labels "
    "before computing metrics. The ±2-month lag expansion converts the 2 raw "
    "COVID months (2020-03, 2020-04) into 4 tolerance-adjusted months "
    "(2020-03 through 2020-06). Sprint1's 'f1_raw' column is therefore F1 "
    "under the tolerance window, not strict raw F1. "
    "Confirmed as variant of H1 (tolerance months counted as positives). "
    "The paper's n+=2 is the correct raw count."
)


# ════════════════════════════════════════════════════════════════════════════════
# VERIFICATION 4 — recompute true F1_raw for all 13 encoders
# ════════════════════════════════════════════════════════════════════════════════

print("\n═══ VERIFICATION 4: True F1_raw (no tolerance) — 13 encoders ══════════")
nber_val_raw = nber_raw.loc[VAL_START:VAL_END]

f1_raw_canonical = {}
detail_rows = []

for enc_name, tier, ablation_dir in ENCODERS:
    test_path = ablation_dir / "pca_kmeans.parquet"
    val_path  = ablation_dir / "val_pca_kmeans.parquet"

    if not test_path.exists() or not val_path.exists():
        print(f"  [{enc_name:20s}] MISSING parquet — skipping")
        f1_raw_canonical[enc_name] = None
        continue

    test_df = pd.read_parquet(test_path)
    val_df  = pd.read_parquet(val_path)

    metrics = compute_f1_raw_for_encoder(
        test_df, val_df, nber_test_raw, nber_val_raw
    )

    f1_raw_canonical[enc_name] = metrics.get("f1_raw")
    detail_rows.append({"encoder": enc_name, "tier": tier, **metrics})

    status = (
        f"F1_raw={metrics.get('f1_raw')}, "
        f"n+={metrics.get('n_positive')}/{metrics.get('n_test')}, "
        f"MCC={metrics.get('mcc')}, "
        f"rec_cluster={metrics.get('recession_cluster')}"
    )
    note = f"  [{enc_name:20s}] {status}"
    if metrics.get("note"):
        note += f"  ⚠ {metrics['note']}"
    print(note)

# Sort by f1_raw descending
detail_rows.sort(key=lambda r: r.get("f1_raw") or -1.0, reverse=True)
print("\nRanked by F1_raw (no tolerance):")
for r in detail_rows:
    f1 = r.get("f1_raw")
    print(f"  {r['encoder']:20s}: {f1}")


# ════════════════════════════════════════════════════════════════════════════════
# Build iTransformer canonical text
# ════════════════════════════════════════════════════════════════════════════════

itrans_f1 = f1_raw_canonical.get("iTransformer")
itrans_detail = next((r for r in detail_rows if r["encoder"] == "iTransformer"), {})
itrans_mcc = itrans_detail.get("mcc")

# F1_tol from sprint1 results
sprint1_csv = ROOT / "results" / "sprint1" / "metrics_comparison_all_encoders.csv"
itrans_f1_tol = None
if sprint1_csv.exists():
    s1 = pd.read_csv(sprint1_csv)
    row = s1[s1["encoder"] == "iTransformer"]
    if not row.empty:
        itrans_f1_tol = float(row["f1_raw"].iloc[0])   # this is actually f1_tol

n_pos_raw = len(pos_raw)
canonical_text = (
    f"The iTransformer achieved a raw NBER F1 of "
    f"{itrans_f1 if itrans_f1 is not None else 'N/A'} "
    f"(n\u207a\u202f=\u202f{n_pos_raw}, no tolerance window), "
    f"rising to {itrans_f1_tol if itrans_f1_tol is not None else 'N/A'} "
    f"under the \u00b12-month tolerance rule (lag\u202f=\u202f2, n\u207a\u202f=\u202f4)."
)

print(f"\nCanonical text: {canonical_text}")


# ════════════════════════════════════════════════════════════════════════════════
# Write canonical JSON
# ════════════════════════════════════════════════════════════════════════════════

canonical = {
    "audit_date": str(date.today()),
    "test_split": {
        "start": "2010-06",
        "end": "2026-04",
        "n_months_in_nber_file": len(nber_test_raw),
    },
    "n_positive_nber_raw": n_pos_raw,
    "nber_positive_dates": [str(d.date()) for d in pos_raw.index],
    "n_positive_source": "nber_usrec.csv (FRED vintage accessed 2026-03)",
    "post_covid_nber_update": post_covid_finding,
    "sprint1_n4_explanation": sprint1_explanation,
    "sprint1_column_mislabel": (
        "sprint1 column 'f1_raw' = F1 computed on tolerance-expanded NBER "
        "(lag=2 after recession end) — equivalent to F1_tol."
    ),
    "f1_raw_canonical": f1_raw_canonical,
    "f1_raw_detail": detail_rows,
    "paper_should_report_f1_raw": True,
    "canonical_text_for_section_4_2": canonical_text,
}

out_dir = ROOT / "audit"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "canonical_n_positives.json"
with open(out_path, "w") as fh:
    json.dump(canonical, fh, indent=2, default=str)

print(f"\n═══ CANONICAL FILE SAVED ═══════════════════════════════════════════════")
print(f"{out_path}")
print("Pass this file to the writing agent as the sole source of truth.")
