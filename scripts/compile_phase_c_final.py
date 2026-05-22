#!/usr/bin/env python3
"""Compile final locked Phase C table from pre-computed locked C1/C3 CSVs.

Reads:
  - val_{cell}.parquet (VAL labels for assignment computation)
  - c1_multi_label_{cell}_locked.csv (locked C1 metrics)
  - c3_pelt_{cell}_locked.csv OR c3_bai_perron_{cell}.csv (C3 metrics)
  - c4_stability.csv (C4 ARI)
  - summary.csv (DBCV, silhouette, etc.)
Does NOT make FRED API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tcc_itransformer.data.external_labels import load_usrec
from tcc_itransformer.evaluation.regime_validation import (
    fit_nber_assignment,
    nber_overlap_frozen,
)
from tcc_itransformer.pipelines.clustering_ablation import _load_split

ROOT = Path(__file__).resolve().parent.parent
USREC_CSV = ROOT / "data/snapshots/nber_usrec.csv"
OUT_TABLES = ROOT / "outputs/tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

PHASE_E_NAMES = {"moment", "ts2vec", "patchtst", "timesnet", "tfc", "hamilton_hmm", "bocpd"}
BASELINE_NAMES = {"windowed_pca", "raw_pca", "linear_ae", "mlp_ae", "svd"}
ALL_CELLS = ["pca_kmeans", "pca_hdbscan", "umap_kmeans", "umap_hdbscan", "tsne_kmeans", "tsne_hdbscan"]

C1_INDICATOR_MAP = {
    "sahm": "c1_sahm_f1", "cfnai_ma3": "c1_cfnai_f1",
    "chauvet_piger": "c1_chauvet_f1", "oecd_cli": "c1_oecd_f1",
}
C3_SERIES_MAP = {"INDPRO": "c3_indpro_f1", "PAYEMS": "c3_payems_f1", "UNRATE": "c3_unrate_f1"}


def _emb_ablation(enc: str) -> tuple[Path, Path]:
    if enc == "iTransformer":
        return (ROOT / "results/sm_outputs/itransformer-1777581449-0d38/embeddings",
                ROOT / "results/clustering_ablation/W6_d7_K4_b1")
    if enc in PHASE_E_NAMES:
        return ROOT / "results/phase_e" / enc / "embeddings", ROOT / "results/phase_e" / enc / "ablation"
    return ROOT / "results/phase_c_comparison" / enc / "emb", ROOT / "results/phase_c_comparison" / enc


def _read_c1(adir: Path, cell: str) -> dict:
    for name in [f"c1_multi_label_{cell}_locked.csv", f"c1_multi_label_{cell}.csv"]:
        p = adir / name
        if p.exists():
            df = pd.read_csv(p)
            lc = "label" if "label" in df.columns else "indicator"
            out = {}
            for _, row in df.iterrows():
                k = C1_INDICATOR_MAP.get(row[lc])
                if k:
                    out[k] = round(float(row["f1"]), 4) if pd.notna(row["f1"]) else float("nan")
                    auc_k = k.replace("_f1", "_auc")
                    out[auc_k] = round(float(row["auc_roc"]), 4) if "auc_roc" in row and pd.notna(row["auc_roc"]) else float("nan")
            return out
    return {}


def _read_c3(adir: Path, cell: str) -> dict:
    for name in [f"c3_pelt_{cell}_locked.csv", f"c3_bai_perron_{cell}.csv", f"c3_pelt_{cell}.csv"]:
        p = adir / name
        if p.exists():
            df = pd.read_csv(p)
            pelt_col = "pelt_f1" if "pelt_f1" in df.columns else ("bai_perron_f1" if "bai_perron_f1" in df.columns else None)
            if pelt_col is None:
                continue
            out = {}
            for _, row in df.iterrows():
                k = C3_SERIES_MAP.get(row.get("series", ""))
                if k:
                    out[k] = round(float(row[pelt_col]), 4) if pd.notna(row[pelt_col]) else float("nan")
            if out:
                return out
    return {}


def main() -> None:
    usrec = load_usrec(USREC_CSV)
    ENCODERS = (
        ["iTransformer"]
        + sorted(PHASE_E_NAMES)
        + sorted(BASELINE_NAMES)
    )

    all_rows = []
    for enc in ENCODERS:
        emb_dir, adir = _emb_ablation(enc)

        # Load summary CSV for DBCV/silhouette
        summary = pd.read_csv(adir / "summary.csv") if (adir / "summary.csv").exists() else pd.DataFrame()

        # Compute cell scores for best-cell selection
        cell_scores: dict[str, float] = {}
        cell_data: dict[str, dict] = {}

        for cell in ALL_CELLS:
            val_p = adir / f"val_{cell}.parquet"
            test_p = adir / f"{cell}.parquet"
            if not val_p.exists() or not test_p.exists():
                continue

            val_df = pd.read_parquet(val_p)
            val_labels = val_df["label"].to_numpy()
            val_dates = pd.DatetimeIndex(val_df["date"])

            test_df = pd.read_parquet(test_p)
            test_labels = test_df["label"].to_numpy()
            test_dates = pd.DatetimeIndex(test_df["date"])

            assignment = fit_nber_assignment(val_labels, val_dates, usrec, lead=0, lag=2)

            # VAL NBER F1
            val_res = nber_overlap_frozen(val_labels, val_dates, usrec, assignment)
            val_nber_f1 = float(val_res.f1)

            # TEST NBER F1 (locked)
            test_res = nber_overlap_frozen(test_labels, test_dates, usrec, assignment)

            # Load C1/C3 from cached files
            c1 = _read_c1(adir, cell)
            c3 = _read_c3(adir, cell)

            # C4 ARI from c4_stability.csv
            c4_ari = float("nan")
            c4_p = adir / "c4_stability.csv"
            if c4_p.exists():
                c4_df = pd.read_csv(c4_p)
                c4_row = c4_df[(c4_df["pipeline"] == cell) & (c4_df.get("cluster", c4_df.get("cluster", pd.Series(["?"]))) == "all") & (c4_df["metric"] == "ari")]
                if not c4_row.empty:
                    c4_ari = round(float(c4_row.iloc[0]["mean_value"]), 4)

            # DBCV / silhouette from summary
            dbcv = sil = float("nan")
            if not summary.empty and "cell" in summary.columns:
                sm_row = summary[summary["cell"] == cell]
                if not sm_row.empty:
                    r = sm_row.iloc[0]
                    dbcv = float(r.get("dbcv", float("nan"))) if pd.notna(r.get("dbcv", float("nan"))) else float("nan")
                    sil = float(r.get("test_silhouette", float("nan"))) if pd.notna(r.get("test_silhouette", float("nan"))) else float("nan")

            # Best-cell score: mean(val_nber_f1, val_sahm [if avail], val_cfnai [if avail])
            cell_scores[cell] = val_nber_f1

            cell_data[cell] = {
                "val_labels": val_labels, "val_dates": val_dates,
                "assignment": assignment, "val_nber_f1": val_nber_f1,
                "test_nber_f1": float(test_res.f1),
                "test_prec": float(test_res.precision),
                "test_rec": float(test_res.recall),
                "c1": c1, "c3": c3, "c4_ari": c4_ari,
                "dbcv": dbcv, "sil": sil,
            }

        if not cell_data:
            continue

        best_cell = max(cell_data, key=lambda c: cell_scores.get(c, float("nan")))

        for cell, d in cell_data.items():
            row = {
                "encoder": enc,
                "cell": cell,
                "is_best_cell_on_val": (cell == best_cell),
                "val_nber_f1": round(d["val_nber_f1"], 4),
                "nber_assignment": str(d["assignment"]),
                "dbcv": round(d["dbcv"], 4) if not np.isnan(d["dbcv"]) else float("nan"),
                "test_silhouette": round(d["sil"], 4) if not np.isnan(d["sil"]) else float("nan"),
                "nber_f1_locked": round(d["test_nber_f1"], 4),
                "nber_precision_locked": round(d["test_prec"], 4),
                "nber_recall_locked": round(d["test_rec"], 4),
                **d["c1"],
                **d["c3"],
                "c4_ari": d["c4_ari"],
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    enc_order = {"iTransformer": 0, "windowed_pca": 1, "raw_pca": 2, "linear_ae": 3,
                 "mlp_ae": 4, "svd": 5, "bocpd": 6, "hamilton_hmm": 7, "moment": 8,
                 "patchtst": 9, "tfc": 10, "timesnet": 11, "ts2vec": 12}
    cell_order = {"pca_kmeans": 0, "pca_hdbscan": 1, "umap_kmeans": 2,
                  "umap_hdbscan": 3, "tsne_kmeans": 4, "tsne_hdbscan": 5}
    df["_eo"] = df["encoder"].map(enc_order).fillna(99)
    df["_co"] = df["cell"].map(cell_order).fillna(9)
    df = df.sort_values(["_eo", "_co"]).drop(columns=["_eo", "_co"]).reset_index(drop=True)

    float_cols = df.select_dtypes(include=[float]).columns
    df[float_cols] = df[float_cols].round(4)

    df.to_csv(OUT_TABLES / "phase_c_locked_all_cells.csv", index=False)
    df[df["is_best_cell_on_val"]].to_csv(OUT_TABLES / "phase_c_locked_best_cell.csv", index=False)
    df[df["cell"] == "pca_kmeans"].to_csv(OUT_TABLES / "phase_c_canonical_pca_kmeans.csv", index=False)

    print(f"Rows: {len(df)} | Encoders: {df['encoder'].nunique()}")
    print(df[["encoder", "cell", "is_best_cell_on_val", "val_nber_f1",
              "nber_f1_locked", "c1_sahm_f1", "c1_cfnai_f1",
              "c3_indpro_f1", "c3_payems_f1", "c4_ari"]].to_string(index=False))


if __name__ == "__main__":
    main()
