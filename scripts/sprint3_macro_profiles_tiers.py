"""Sprint 3 — F6 (Macro Profiles) + F9 (Tier Stratification)
=============================================================
Addresses two methodological fragilities:

  F6 — Geometric-economic dissociation unexplained: what do the high-DBCV
       clusters of SVD/MLP-AE actually capture if not recessions?
       Fix: compute mean z-score of each FRED-MD series per cluster × encoder.
       Top-5 discriminative series per cluster explain the dissociation.

  F9 — Heterogeneous benchmark (zero-shot vs fine-tuned vs from-scratch):
       Fix: stratify the benchmark into 3 formal tiers and report separately.

Run:
    cd tcc_ai && uv run python scripts/sprint3_macro_profiles_tiers.py

Outputs:
    results/sprint3/macro_profiles_all_encoders.csv   (122 series × 13 enc × 4 clusters)
    results/sprint3/top5_discriminative_series.csv    (per encoder × cluster)
    results/sprint3/tier_stratified_comparison.csv    (Tier 1/2/3 breakdown)
    results/sprint3/INTERPRETACAO_economica_clusters.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "sprint3"
OUT.mkdir(parents=True, exist_ok=True)

FRED_PARQUET = ROOT / "data" / "raw" / "fred_md_transformed_balanced_2026_04.parquet"
PHASE_C_CSV  = ROOT / "results" / "phase_c_comparison" / "phase_c_comparison.csv"

# ── temporal windows ──────────────────────────────────────────────────────────
TEST_START = pd.Timestamp("2010-06-01")
VAL_START  = pd.Timestamp("2000-06-01")
VAL_END    = pd.Timestamp("2009-12-01")

# ── Tier definitions ──────────────────────────────────────────────────────────
TIER_MAP: dict[str, str] = {
    # Tier 1 — trained from scratch on FRED-MD only
    "iTransformer":  "Tier 1 — From Scratch",
    "windowed_pca":  "Tier 1 — From Scratch",
    "raw_pca":       "Tier 1 — From Scratch",
    "linear_ae":     "Tier 1 — From Scratch",
    "mlp_ae":        "Tier 1 — From Scratch",
    "svd":           "Tier 1 — From Scratch",
    "hamilton_hmm":  "Tier 1 — From Scratch",
    "bocpd":         "Tier 1 — From Scratch",
    "timesnet":      "Tier 1 — From Scratch",
    # Tier 2 — pre-trained, fine-tuned on FRED-MD
    "ts2vec":        "Tier 2 — Domain-Adapted",
    "patchtst":      "Tier 2 — Domain-Adapted",
    "tfc":           "Tier 2 — Domain-Adapted",
    # Tier 3 — zero-shot, no FRED-MD adaptation
    "moment":        "Tier 3 — Zero-Shot",
}

# ── encoder ablation parquet locations ───────────────────────────────────────
_PC = ROOT / "results" / "phase_c_comparison"
_PE = ROOT / "results" / "phase_e"
_CL = ROOT / "results" / "clustering_ablation" / "W6_d7_K4_b1"

ENCODER_ABLATION: dict[str, Path] = {
    "iTransformer": _CL,
    "windowed_pca": _PC / "windowed_pca",
    "raw_pca":      _PC / "raw_pca",
    "linear_ae":    _PC / "linear_ae",
    "mlp_ae":       _PC / "mlp_ae",
    "svd":          _PC / "svd",
    "moment":       _PE / "moment"      / "ablation",
    "ts2vec":       _PE / "ts2vec"      / "ablation",
    "patchtst":     _PE / "patchtst"    / "ablation",
    "timesnet":     _PE / "timesnet"    / "ablation",
    "tfc":          _PE / "tfc"         / "ablation",
    "hamilton_hmm": _PE / "hamilton_hmm"/ "ablation",
    "bocpd":        _PE / "bocpd"       / "ablation",
}

# Key FRED-MD series for economic interpretation
KEY_SERIES_GROUPS: dict[str, list[str]] = {
    "real_activity": ["INDPRO", "IPFINAL", "IPCONGD", "CMRMTSPLx", "PAYEMS",
                      "MANEMP", "SRVPRD", "USWTRADE"],
    "labor":         ["UNRATE", "CLAIMSx", "CES0600000007", "UEMP15T26"],
    "credit":        ["TOTALSL", "BUSLOANS", "NONREVSL", "CONSUMER"],
    "financial":     ["S&P 500", "S&P PE ratio", "VIXCLSx", "BAAFFM", "TB3SMFFM"],
    "prices":        ["CPIAUCSL", "PPIFGS", "PCEPI", "CUSR0000SAC"],
    "money":         ["M1SL", "M2SL", "FEDFUNDS", "TB3MS", "GS10", "GS1"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# F6 — Macro Profiles per Cluster
# ═══════════════════════════════════════════════════════════════════════════════

def load_fred_test_window(fred: pd.DataFrame) -> pd.DataFrame:
    """Return FRED-MD rows from TEST window, z-score normalized per series."""
    fred = fred.copy()
    fred["date"] = pd.to_datetime(fred["date"])
    test = fred[fred["date"] >= TEST_START].copy()
    series_cols = [c for c in test.columns if c != "date"]
    # z-score normalize each series independently (over TEST window)
    for col in series_cols:
        s = test[col]
        mu, sigma = s.mean(), s.std()
        if sigma > 1e-8:
            test[col] = (s - mu) / sigma
        else:
            test[col] = 0.0
    return test


def compute_cluster_profiles(
    fred_test: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute mean z-score per (cluster, FRED-MD series).

    Returns DataFrame with columns:
        cluster, series, mean_zscore, std_zscore, n_months
    """
    series_cols = [c for c in fred_test.columns if c != "date"]
    merged = fred_test.merge(labels_df[["date", "label"]], on="date", how="inner")

    rows = []
    for k in sorted(merged["label"].unique()):
        grp = merged[merged["label"] == k]
        n   = len(grp)
        for col in series_cols:
            vals = grp[col].dropna()
            rows.append({
                "cluster": int(k),
                "series": col,
                "mean_zscore": float(vals.mean()) if len(vals) > 0 else float("nan"),
                "std_zscore":  float(vals.std())  if len(vals) > 1 else float("nan"),
                "n_months": n,
            })
    return pd.DataFrame(rows)


def top5_discriminative(profiles_df: pd.DataFrame, encoder: str) -> pd.DataFrame:
    """
    For each cluster, return top-5 FRED-MD series with highest |mean_zscore|.
    These are the series most characteristic of each regime.
    """
    rows = []
    for k in sorted(profiles_df["cluster"].unique()):
        sub = profiles_df[profiles_df["cluster"] == k].copy()
        sub["abs_z"] = sub["mean_zscore"].abs()
        top5 = sub.nlargest(5, "abs_z")[["series", "mean_zscore", "abs_z"]]
        for rank, row in enumerate(top5.itertuples(), start=1):
            rows.append({
                "encoder": encoder,
                "cluster": int(k),
                "rank": rank,
                "series": row.series,
                "mean_zscore": round(row.mean_zscore, 4),
                "abs_z": round(row.abs_z, 4),
            })
    return pd.DataFrame(rows)


def run_f6_macro_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n── F6: Macro profiles per cluster (all 13 encoders) ────────────────")
    fred_raw = pd.read_parquet(FRED_PARQUET).sort_values("date").reset_index(drop=True)
    fred_test = load_fred_test_window(fred_raw)

    all_profiles: list[pd.DataFrame] = []
    all_top5:     list[pd.DataFrame] = []

    for enc_name, ablation_dir in ENCODER_ABLATION.items():
        test_path = ablation_dir / "pca_kmeans.parquet"
        if not test_path.exists():
            print(f"  [{enc_name}] SKIP — pca_kmeans.parquet not found")
            continue

        labels_df = pd.read_parquet(test_path)[["date", "label"]]
        labels_df["date"] = pd.to_datetime(labels_df["date"])

        profiles = compute_cluster_profiles(fred_test, labels_df)
        profiles.insert(0, "encoder", enc_name)
        all_profiles.append(profiles)

        top5 = top5_discriminative(profiles, enc_name)
        all_top5.append(top5)

        # Print summary: which series are top discriminators for each cluster?
        n_clusters = profiles["cluster"].nunique()
        print(f"\n  [{enc_name}] {n_clusters} clusters in TEST")
        for k in sorted(profiles["cluster"].unique()):
            sub = profiles[profiles["cluster"] == k]
            sub_abs = sub.assign(abs_z=sub["mean_zscore"].abs())
            top3 = sub_abs.nlargest(3, "abs_z")[["series", "mean_zscore"]]
            series_str = ", ".join(
                f"{r.series}({r.mean_zscore:+.2f})" for r in top3.itertuples()
            )
            print(f"    C{k}: {series_str}")

    df_profiles = pd.concat(all_profiles, ignore_index=True)
    df_top5     = pd.concat(all_top5, ignore_index=True)

    out_profiles = OUT / "macro_profiles_all_encoders.csv"
    out_top5     = OUT / "top5_discriminative_series.csv"
    df_profiles.to_csv(out_profiles, index=False)
    df_top5.to_csv(out_top5, index=False)
    print(f"\n  ✓ Saved profiles → {out_profiles.relative_to(ROOT)}")
    print(f"  ✓ Saved top5     → {out_top5.relative_to(ROOT)}")

    return df_profiles, df_top5


# ═══════════════════════════════════════════════════════════════════════════════
# F9 — Tier Stratification
# ═══════════════════════════════════════════════════════════════════════════════

def run_f9_tier_stratification() -> pd.DataFrame:
    print("\n── F9: Tier-stratified benchmark ────────────────────────────────────")

    if not PHASE_C_CSV.exists():
        print(f"  ERROR: {PHASE_C_CSV} not found")
        return pd.DataFrame()

    df = pd.read_csv(PHASE_C_CSV)
    df["tier"] = df["encoder"].map(TIER_MAP).fillna("Unknown")

    # Focus on pca_kmeans cell (canonical comparison cell)
    pca_km = df[df["cell"] == "pca_kmeans"].copy()

    # Metrics of interest
    metric_cols = [
        c for c in pca_km.columns
        if c not in {"encoder", "cell", "clusterer", "tier"}
    ]

    # Per-tier summary: mean ± std for key metrics
    key_metrics = [
        "nber_f1", "test_silhouette", "dbcv",
        "c1_sahm_f1", "c3_indpro_f1", "c4_ari",
    ]
    key_metrics = [m for m in key_metrics if m in pca_km.columns]

    tier_summary = (
        pca_km.groupby("tier")[key_metrics]
        .agg(["mean", "std", "max"])
        .round(4)
    )
    tier_summary.columns = ["_".join(c).strip() for c in tier_summary.columns]
    tier_summary = tier_summary.reset_index()

    print("\n  Tier summary (pca_kmeans cell):")
    print(tier_summary.to_string(index=False))

    # Full stratified table
    pca_km_sorted = pca_km.sort_values(
        ["tier", "nber_f1"], ascending=[True, False]
    ).reset_index(drop=True)
    pca_km_sorted = pca_km_sorted[["tier", "encoder"] + key_metrics + [
        c for c in metric_cols if c not in key_metrics and c not in {"tier", "encoder"}
    ]]

    out_csv = OUT / "tier_stratified_comparison.csv"
    pca_km_sorted.to_csv(out_csv, index=False)
    print(f"\n  ✓ Saved → {out_csv.relative_to(ROOT)}")

    # Tier-level winner per metric
    print("\n  Best encoder per tier × metric:")
    for tier in pca_km_sorted["tier"].unique():
        sub = pca_km_sorted[pca_km_sorted["tier"] == tier]
        row = sub.nlargest(1, "nber_f1").iloc[0]
        print(
            f"  {tier}: best={row['encoder']} "
            f"nber_f1={row.get('nber_f1', 'N/A'):.4f} "
            f"silhouette={row.get('test_silhouette', float('nan')):.4f}"
        )

    return pca_km_sorted


# ═══════════════════════════════════════════════════════════════════════════════
# Interpretation JSON
# ═══════════════════════════════════════════════════════════════════════════════

def build_interpretation_json(
    df_top5: pd.DataFrame, df_tiers: pd.DataFrame
) -> dict:
    """
    Build a structured economic interpretation of clusters per encoder.
    Maps discriminative series to economic concepts.
    """
    series_concept: dict[str, str] = {
        "UNRATE":      "unemployment (labor)",
        "PAYEMS":      "nonfarm payrolls (labor)",
        "MANEMP":      "manufacturing employment (labor)",
        "INDPRO":      "industrial production (real activity)",
        "IPFINAL":     "final products production (real activity)",
        "CMRMTSPLx":   "manufacturing sales (real activity)",
        "S&P 500":     "equity prices (financial)",
        "VIXCLSx":     "equity volatility (financial)",
        "BAAFFM":      "Baa-FF credit spread (financial)",
        "TB3SMFFM":    "3m T-bill spread (financial)",
        "FEDFUNDS":    "federal funds rate (monetary)",
        "GS10":        "10y Treasury yield (monetary)",
        "M2SL":        "M2 money supply (monetary)",
        "CPIAUCSL":    "CPI inflation (prices)",
        "CLAIMSx":     "initial jobless claims (labor)",
    }

    interpretation: dict = {}
    for enc in df_top5["encoder"].unique():
        enc_data = {}
        for k in sorted(df_top5[df_top5["encoder"] == enc]["cluster"].unique()):
            sub = df_top5[(df_top5["encoder"] == enc) & (df_top5["cluster"] == k)]
            top = sub.nlargest(3, "abs_z")
            series_list = [
                {
                    "series": r.series,
                    "mean_zscore": r.mean_zscore,
                    "concept": series_concept.get(r.series, "other"),
                    "direction": "stressed/elevated" if r.mean_zscore > 0 else "relaxed/depressed",
                }
                for r in top.itertuples()
            ]
            def _extract_category(concept: str) -> str:
                parts = concept.split("(")
                return parts[1].rstrip(")") if len(parts) > 1 else concept

            dominant_concept = max(
                set(_extract_category(s["concept"]) for s in series_list),
                key=lambda c: sum(1 for s in series_list if c in s["concept"])
            ) if series_list else "unknown"
            enc_data[f"C{k}"] = {
                "dominant_theme": dominant_concept,
                "top3_series": series_list,
            }
        interpretation[enc] = enc_data

    # Add tier context
    if not df_tiers.empty and "tier" in df_tiers.columns:
        for enc in interpretation:
            sub = df_tiers[df_tiers["encoder"] == enc]
            if not sub.empty:
                row = sub.iloc[0]
                interpretation[enc]["tier"] = row.get("tier", "Unknown")
                interpretation[enc]["nber_f1_pca_kmeans"] = float(
                    row.get("nber_f1", float("nan"))
                )
                interpretation[enc]["test_silhouette_pca_kmeans"] = float(
                    row.get("test_silhouette", float("nan"))
                )

    return interpretation


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df_profiles, df_top5 = run_f6_macro_profiles()
    df_tiers              = run_f9_tier_stratification()

    interp = build_interpretation_json(df_top5, df_tiers)
    interp_path = OUT / "INTERPRETACAO_economica_clusters.json"
    with open(interp_path, "w", encoding="utf-8") as fh:
        json.dump(interp, fh, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Interpretation JSON → {interp_path.relative_to(ROOT)}")

    # Print executive summary: dissociation explanation
    print("\n── Executive Summary: Geometric-Economic Dissociation ──────────────")
    print(
        "  Encoders with high DBCV but low NBER F1 (SVD, MLP-AE) likely capture\n"
        "  regime-level structure NOT aligned with NBER recession dates.\n"
        "  The top discriminative series per cluster reveal what structure they DO capture:\n"
    )
    for enc in ["svd", "mlp_ae", "iTransformer"]:
        if enc not in interp:
            continue
        print(f"  [{enc}]")
        for ck, data in interp[enc].items():
            if not ck.startswith("C"):
                continue
            theme = data.get("dominant_theme", "?")
            top3  = data.get("top3_series", [])
            series_str = " | ".join(
                f"{s['series']}({s['mean_zscore']:+.2f})" for s in top3
            )
            print(f"    {ck}: theme='{theme}' → {series_str}")

    print("\n═══ Sprint 3 complete ═══")


if __name__ == "__main__":
    main()
