"""Sprint 1 — F1 (MCC / PR-AUC / Brier) + F7 (Stress Score Ablation)
====================================================================
Addresses two methodological fragilities from the v11 paper critique:

  F1 — n=2 recession positives in TEST: F1 is discretized to 4 values.
       Fix: add MCC, PR-AUC (continuous centroid-distance score), Brier
       + bootstrap 95% CI for all 13 encoders.

  F7 — Stress-score weights not pre-registered: potential HARKing.
       Fix: ablation over 5 weight configurations; report whether the same
       cluster is identified as the recession cluster across all configs.

Run:
    cd tcc_ai && uv run python scripts/sprint1_metrics_and_stress.py

Outputs:
    results/sprint1/metrics_comparison_all_encoders.csv
    results/sprint1/ablation_stress_score_5configs.csv
    results/sprint1/SUMMARY_sprint1.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ── repo paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # tcc_ai/
sys.path.insert(0, str(ROOT / "src"))

from tcc_itransformer.evaluation.regime_validation import fit_nber_assignment

OUT = ROOT / "results" / "sprint1"
OUT.mkdir(parents=True, exist_ok=True)

NBER_CSV = ROOT / "data" / "snapshots" / "nber_usrec.csv"
FRED_PARQUET = ROOT / "data" / "raw" / "fred_md_transformed_balanced_2026_04.parquet"

# ── temporal windows (B1 split) ───────────────────────────────────────────────
VAL_START = pd.Timestamp("2000-06-01")
VAL_END   = pd.Timestamp("2009-12-01")
TEST_START = pd.Timestamp("2010-06-01")
LEAD, LAG = 0, 2
N_BOOT = 2000
RNG = np.random.default_rng(42)

# ── encoder catalogue ─────────────────────────────────────────────────────────
# Each entry: (display_name, tier, ablation_dir, val_ablation_dir)
# ablation_dir  = folder containing pca_kmeans.parquet (TEST labels)
# val_ablation_dir = same folder for val_pca_kmeans.parquet
_PC = ROOT / "results" / "phase_c_comparison"
_PE = ROOT / "results" / "phase_e"
_CL = ROOT / "results" / "clustering_ablation" / "W6_d7_K4_b1"

ENCODERS: list[tuple[str, str, Path, Path]] = [
    # (name, tier, test_dir, val_dir)
    ("iTransformer",  "tier1", _CL,                  _CL),
    ("windowed_pca",  "tier1", _PC / "windowed_pca",  _PC / "windowed_pca"),
    ("raw_pca",       "tier1", _PC / "raw_pca",        _PC / "raw_pca"),
    ("linear_ae",     "tier1", _PC / "linear_ae",      _PC / "linear_ae"),
    ("mlp_ae",        "tier1", _PC / "mlp_ae",          _PC / "mlp_ae"),
    ("svd",           "tier1", _PC / "svd",             _PC / "svd"),
    ("moment",        "tier3", _PE / "moment"     / "ablation", _PE / "moment"     / "ablation"),
    ("ts2vec",        "tier2", _PE / "ts2vec"     / "ablation", _PE / "ts2vec"     / "ablation"),
    ("patchtst",      "tier2", _PE / "patchtst"   / "ablation", _PE / "patchtst"   / "ablation"),
    ("timesnet",      "tier1", _PE / "timesnet"   / "ablation", _PE / "timesnet"   / "ablation"),
    ("tfc",           "tier2", _PE / "tfc"        / "ablation", _PE / "tfc"        / "ablation"),
    ("hamilton_hmm",  "tier1", _PE / "hamilton_hmm"/ "ablation",_PE / "hamilton_hmm"/ "ablation"),
    ("bocpd",         "tier1", _PE / "bocpd"      / "ablation", _PE / "bocpd"      / "ablation"),
]

# ── stress score component definitions ────────────────────────────────────────
# After FRED-MD tcode transformations:
#   UNRATE  : Δ(unemployment rate) — higher = worse
#   INDPRO  : log-diff(industrial production) — negative = worse
#   PAYEMS  : log-diff(nonfarm payrolls) — negative = worse
#   "S&P 500": log-diff(S&P 500) — negative = worse
#
# stress_score = w1*UNRATE + w2*(-INDPRO) + w3*(-PAYEMS) + w4*(-SP500)

STRESS_CONFIGS: dict[str, list[float]] = {
    "equal":           [0.25, 0.25, 0.25, 0.25],   # current (implicit)
    "labor_only":      [0.33, 0.33, 0.33, 0.00],   # no financial markets
    "market_heavy":    [0.25, 0.25, 0.00, 0.50],   # double weight on equities
    "unrate_only":     [1.00, 0.00, 0.00, 0.00],   # only unemployment
    "nber_weighted":   [0.20, 0.30, 0.30, 0.20],   # NBER-component inspired
}
STRESS_COLS = ["UNRATE", "INDPRO", "PAYEMS", "S&P 500"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_nber() -> pd.Series:
    """Return monthly NBER USREC series indexed by date."""
    df = pd.read_csv(NBER_CSV, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date"})
    return df.set_index("date")["USREC"].astype(int)


def nber_for_window(nber: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                    lead: int = 0, lag: int = 2) -> pd.Series:
    """Slice + tolerance-expand NBER for a given window."""
    subset = nber.loc[start:end].copy()
    expanded = subset.copy()
    for k in range(1, lead + 1):
        expanded |= subset.shift(-k, fill_value=0)
    for k in range(1, lag + 1):
        expanded |= subset.shift(k, fill_value=0)
    return expanded


def load_labels(ablation_dir: Path, split: str = "test") -> pd.DataFrame | None:
    """Load pca_kmeans parquet for TEST or VAL split."""
    fname = "pca_kmeans.parquet" if split == "test" else "val_pca_kmeans.parquet"
    path = ablation_dir / fname
    if not path.exists():
        return None
    return pd.read_parquet(path)


def recession_cluster_from_nber(
    val_df: pd.DataFrame, nber: pd.Series
) -> int:
    """Return the cluster with highest NBER-overlap fraction on VAL."""
    val_nber = nber.reindex(val_df["date"]).fillna(0).astype(int)
    best_k, best_share = 0, -1.0
    for k in val_df["label"].unique():
        mask = val_df["label"] == k
        share = val_nber[mask.values].mean()
        if share > best_share:
            best_share, best_k = share, k
    return int(best_k)


def compute_centroid_distances(
    val_df: pd.DataFrame, test_df: pd.DataFrame, rec_cluster: int
) -> np.ndarray:
    """
    Compute Euclidean distance from each TEST point to the VAL recession
    cluster centroid in (x_2d, y_2d) PCA space.
    Returns: ndarray of shape (n_test,)
    """
    rec_val = val_df[val_df["label"] == rec_cluster]
    cx = rec_val["x_2d"].mean()
    cy = rec_val["y_2d"].mean()
    dx = test_df["x_2d"].values - cx
    dy = test_df["y_2d"].values - cy
    return np.sqrt(dx ** 2 + dy ** 2)


def continuous_score(distances: np.ndarray) -> np.ndarray:
    """score = 1 / (1 + dist) — higher = closer to recession centroid."""
    return 1.0 / (1.0 + distances)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% CI for MCC, PR-AUC, Brier."""
    n = len(y_true)
    mccs, aucs, briers = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys, yp = y_true[idx], y_score[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            mccs.append(matthews_corrcoef(yt, yp))
            aucs.append(average_precision_score(yt, ys))
            briers.append(brier_score_loss(yt, ys))
        except Exception:
            pass
    def _ci(vals: list) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        a = np.array(vals)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"mcc": _ci(mccs), "pr_auc": _ci(aucs), "brier": _ci(briers)}


# ═══════════════════════════════════════════════════════════════════════════════
# F1 — MCC / PR-AUC / Brier for all encoders
# ═══════════════════════════════════════════════════════════════════════════════

def run_f1_metrics(nber: pd.Series) -> pd.DataFrame:
    print("\n── F1: Computing MCC / PR-AUC / Brier for 13 encoders ──────────────")
    nber_test = nber_for_window(nber, TEST_START, pd.Timestamp("2026-12-31"),
                                 lead=LEAD, lag=LAG)
    rows = []

    for name, tier, test_dir, val_dir in ENCODERS:
        test_df = load_labels(test_dir, "test")
        val_df  = load_labels(val_dir,  "val")
        if test_df is None or val_df is None:
            print(f"  [{name}] SKIP — parquet not found")
            continue

        # Align NBER to test dates
        yt = nber_test.reindex(test_df["date"]).fillna(0).astype(int).values
        if yt.sum() == 0:
            print(f"  [{name}] SKIP — 0 NBER positives in TEST window")
            continue

        # NBER-frozen recession cluster
        nber_val = nber_for_window(nber, VAL_START, VAL_END, lead=LEAD, lag=LAG)
        nber_val_aligned = nber_val.reindex(val_df["date"]).fillna(0).astype(int)
        rec_k = recession_cluster_from_nber(val_df, nber_val_aligned)

        # Binary labels
        yp = (test_df["label"] == rec_k).astype(int).values

        # Continuous score via centroid distance
        if "x_2d" in test_df.columns and "x_2d" in val_df.columns:
            dists = compute_centroid_distances(val_df, test_df, rec_k)
            ys = continuous_score(dists)
        else:
            # fallback: hard binary as float
            ys = yp.astype(float)

        # Point estimates
        n_pos = int(yt.sum())
        mcc   = float(matthews_corrcoef(yt, yp))
        brier = float(brier_score_loss(yt, ys))
        try:
            pr_auc = float(average_precision_score(yt, ys))
        except Exception:
            pr_auc = float("nan")
        prec   = float(precision_score(yt, yp, zero_division=0))
        rec_v  = float(recall_score(yt, yp, zero_division=0))
        f1_raw = 2 * prec * rec_v / (prec + rec_v) if (prec + rec_v) > 0 else 0.0

        # Bootstrap CI
        ci = bootstrap_ci(yt, ys, yp, N_BOOT, RNG)

        rows.append({
            "encoder": name,
            "tier": tier,
            "n_positives_test": n_pos,
            "recession_cluster": rec_k,
            "f1_raw": round(f1_raw, 4),
            "mcc": round(mcc, 4),
            "mcc_ci_lo": round(ci["mcc"][0], 4),
            "mcc_ci_hi": round(ci["mcc"][1], 4),
            "pr_auc": round(pr_auc, 4),
            "pr_auc_ci_lo": round(ci["pr_auc"][0], 4),
            "pr_auc_ci_hi": round(ci["pr_auc"][1], 4),
            "brier": round(brier, 4),
            "brier_ci_lo": round(ci["brier"][0], 4),
            "brier_ci_hi": round(ci["brier"][1], 4),
        })
        print(
            f"  [{name:14s}] tier={tier} n+={n_pos} "
            f"F1={f1_raw:.4f} MCC={mcc:+.4f} "
            f"PR-AUC={pr_auc:.4f} Brier={brier:.4f}"
        )

    df = pd.DataFrame(rows)
    # Sort by PR-AUC descending
    df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    out_path = OUT / "metrics_comparison_all_encoders.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved → {out_path.relative_to(ROOT)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# F7 — Stress-score weight ablation
# ═══════════════════════════════════════════════════════════════════════════════

def load_fred_md() -> pd.DataFrame:
    df = pd.read_parquet(FRED_PARQUET)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_stress_score(
    fred: pd.DataFrame,
    dates: pd.Series,
    weights: list[float],
) -> pd.Series:
    """
    Compute monthly stress score for given dates.
    stress = w1*ΔUNRATE + w2*(-ΔINDPRO) + w3*(-ΔPAYEMS) + w4*(-ΔSP500)
    Higher = more stressed.
    """
    fred_idx = fred.set_index("date")
    aligned = fred_idx.reindex(dates)
    w1, w2, w3, w4 = weights
    score = (
        w1 * aligned["UNRATE"]
        + w2 * (-aligned["INDPRO"])
        + w3 * (-aligned["PAYEMS"])
        + w4 * (-aligned["S&P 500"])
    ).fillna(0)
    return score.values  # ndarray aligned to dates


def recession_cluster_from_stress(
    val_df: pd.DataFrame, stress_scores: np.ndarray
) -> int:
    """Return cluster with highest mean stress score on VAL."""
    best_k, best_mean = 0, -np.inf
    for k in val_df["label"].unique():
        mask = (val_df["label"] == k).values
        mean_stress = float(stress_scores[mask].mean())
        if mean_stress > best_mean:
            best_mean, best_k = mean_stress, k
    return int(best_k)


def f1_for_cluster(
    test_df: pd.DataFrame, rec_k: int, nber_test: pd.Series,
    lead: int = 0, lag: int = 2
) -> tuple[float, float]:
    """Compute F1_raw and F1_tol for a given recession cluster assignment."""
    yt_raw = nber_test.reindex(test_df["date"]).fillna(0).astype(int).values
    # tolerance
    nber_tol = nber_test.copy()
    for k_shift in range(1, lag + 1):
        nber_tol = nber_tol | nber_test.shift(k_shift, fill_value=0)
    yt_tol = nber_tol.reindex(test_df["date"]).fillna(0).astype(int).values

    yp = (test_df["label"] == rec_k).astype(int).values

    def _f1(yt: np.ndarray, yp_: np.ndarray) -> float:
        p = precision_score(yt, yp_, zero_division=0)
        r = recall_score(yt, yp_, zero_division=0)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return _f1(yt_raw, yp), _f1(yt_tol, yp)


def run_f7_stress_ablation(nber: pd.Series) -> pd.DataFrame:
    print("\n── F7: Stress score weight ablation (5 configs × 13 encoders) ──────")
    fred = load_fred_md()
    nber_test = nber.loc[TEST_START:]

    rows = []
    for name, tier, test_dir, val_dir in ENCODERS:
        test_df = load_labels(test_dir, "test")
        val_df  = load_labels(val_dir,  "val")
        if test_df is None or val_df is None:
            continue

        # Reference: NBER-based frozen mapping
        nber_val = nber.loc[VAL_START:VAL_END]
        nber_val_aligned = nber_val.reindex(val_df["date"]).fillna(0).astype(int)
        rec_k_nber = recession_cluster_from_nber(val_df, nber_val_aligned)
        f1r_nber, f1t_nber = f1_for_cluster(test_df, rec_k_nber, nber_test, LEAD, LAG)

        for cfg_name, weights in STRESS_CONFIGS.items():
            # Compute stress score for VAL months
            stress_val = compute_stress_score(fred, val_df["date"], weights)
            rec_k_stress = recession_cluster_from_stress(val_df, stress_val)

            # F1 on TEST using the stress-derived mapping
            f1r, f1t = f1_for_cluster(test_df, rec_k_stress, nber_test, LEAD, LAG)

            rows.append({
                "encoder": name,
                "tier": tier,
                "stress_config": cfg_name,
                "weights": str(weights),
                "cluster_identified": rec_k_stress,
                "cluster_identified_nber": rec_k_nber,
                "agrees_with_nber": int(rec_k_stress == rec_k_nber),
                "f1_raw_test": round(f1r, 4),
                "f1_tol_test": round(f1t, 4),
                "f1_raw_nber_ref": round(f1r_nber, 4),
                "f1_tol_nber_ref": round(f1t_nber, 4),
            })

        # Summary per encoder
        df_enc = pd.DataFrame([r for r in rows if r["encoder"] == name])
        agree_pct = df_enc["agrees_with_nber"].mean() * 100
        print(
            f"  [{name:14s}] NBER_cluster=C{rec_k_nber} "
            f"stress-agreement={agree_pct:.0f}%  "
            f"F1_tol_nber={f1t_nber:.4f}"
        )

    df = pd.DataFrame(rows)
    out_path = OUT / "ablation_stress_score_5configs.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved → {out_path.relative_to(ROOT)}")

    # Summary stats
    agg = df.groupby("encoder")["agrees_with_nber"].mean().reset_index()
    agg.columns = ["encoder", "stress_nber_agreement_rate"]
    print("\n  Agreement between stress-score and NBER cluster identification:")
    print(agg.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    nber = load_nber()
    print(f"NBER loaded: {len(nber)} months, "
          f"{nber.sum()} recession months total")

    df_f1  = run_f1_metrics(nber)
    df_f7  = run_f7_stress_ablation(nber)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    # F1 top encoder by PR-AUC
    top_prauc = df_f1.iloc[0]
    # F7 overall agreement rate
    f7_agree = df_f7.groupby("encoder")["agrees_with_nber"].mean()
    f7_itrans = f7_agree.get("iTransformer", float("nan"))

    summary = {
        "sprint": 1,
        "f1_findings": {
            "top_encoder_pr_auc": str(top_prauc["encoder"]),
            "top_pr_auc_value": float(top_prauc["pr_auc"]),
            "iTransformer_mcc": float(df_f1.loc[df_f1.encoder == "iTransformer", "mcc"].values[0])
                if "iTransformer" in df_f1.encoder.values else None,
            "iTransformer_pr_auc": float(df_f1.loc[df_f1.encoder == "iTransformer", "pr_auc"].values[0])
                if "iTransformer" in df_f1.encoder.values else None,
            "n_encoders_computed": len(df_f1),
        },
        "f7_findings": {
            "iTransformer_stress_nber_agreement": float(f7_itrans)
                if not np.isnan(f7_itrans) else None,
            "global_mean_agreement": float(f7_agree.mean()),
            "config_breakdown": df_f7.groupby("stress_config")["agrees_with_nber"]
                .mean().round(3).to_dict(),
        },
        "outputs": [
            "results/sprint1/metrics_comparison_all_encoders.csv",
            "results/sprint1/ablation_stress_score_5configs.csv",
        ],
    }

    summary_path = OUT / "SUMMARY_sprint1.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  ✓ Summary → {summary_path.relative_to(ROOT)}")
    print("\n═══ Sprint 1 complete ═══")


if __name__ == "__main__":
    main()
