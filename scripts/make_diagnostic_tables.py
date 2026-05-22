#!/usr/bin/env python3
"""Generate diagnostic tables for Agent 2 (writing agent).

Produces:
  outputs/tables/bootstrap_stability.csv    — Pr(J≥0.60) via normal CDF approx
  outputs/tables/nber_positive_count.json   — VAL/TEST recession month counts
  outputs/tables/b1_distribution_shift.csv  — KS test TRAIN vs TEST per FRED-MD series
  outputs/tables/c1_arithmetic_audit.csv    — C1 F1/AUC values from all cached CSVs
  outputs/tables/falsification_summary.csv  — DBCV scores + gate status
  outputs/tables/mlflow_stage_results.csv   — Stage-1/Stage-2 MLflow run summaries
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/tables"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Bootstrap stability with Pr(J≥0.60) ───────────────────────────────────
def make_bootstrap_stability() -> None:
    c4_csv = ROOT / "results/clustering_ablation/W6_d7_K4_b1/c4_stability.csv"
    if not c4_csv.exists():
        logger.warning("c4_stability.csv not found at %s", c4_csv)
        return

    df = pd.read_csv(c4_csv)
    # Keep per-cluster Jaccard rows (not 'all' ARI rows)
    jaccard = df[df["metric"] == "jaccard"].copy()
    jaccard["pr_jaccard_ge_060"] = jaccard.apply(
        lambda row: float(stats.norm.sf(0.60, loc=row["mean_value"], scale=row["std_value"]))
        if row["std_value"] > 0 else (1.0 if row["mean_value"] >= 0.60 else 0.0),
        axis=1,
    ).round(4)

    # Also keep ARI rows for completeness
    ari_rows = df[df["metric"] == "ari"].copy()
    ari_rows["pr_jaccard_ge_060"] = float("nan")

    combined = pd.concat([jaccard, ari_rows], ignore_index=True)
    combined = combined.sort_values(["pipeline", "cluster"])

    out = OUT / "bootstrap_stability.csv"
    combined.to_csv(out, index=False)
    logger.info("bootstrap_stability.csv: %d rows → %s", len(combined), out)

    # Summary: pca_kmeans only
    pk = jaccard[jaccard["pipeline"] == "pca_kmeans"].copy()
    print("\n=== Bootstrap stability (pca_kmeans, iTransformer B1) ===")
    print(pk[["cluster", "n_months", "mean_value", "std_value", "pr_jaccard_ge_060", "stable"]].to_string(index=False))


# ── 2. NBER positive count ────────────────────────────────────────────────────
def make_nber_positive_count() -> None:
    usrec_csv = ROOT / "data/snapshots/nber_usrec.csv"
    if not usrec_csv.exists():
        logger.warning("nber_usrec.csv not found")
        return

    usrec = pd.read_csv(usrec_csv, parse_dates=["observation_date"])
    usrec = usrec.rename(columns={"observation_date": "date"})
    # Split boundaries (B1)
    val_start, val_end = pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")
    test_start, test_end = pd.Timestamp("2010-01-01"), pd.Timestamp("2026-04-30")
    train_end = pd.Timestamp("1999-12-31")

    val_mask = (usrec["date"] >= val_start) & (usrec["date"] <= val_end)
    test_mask = (usrec["date"] >= test_start) & (usrec["date"] <= test_end)
    train_mask = usrec["date"] <= train_end

    val_rec = usrec[val_mask & (usrec["USREC"] == 1)]
    test_rec = usrec[test_mask & (usrec["USREC"] == 1)]
    train_rec = usrec[train_mask & (usrec["USREC"] == 1)]

    result = {
        "split_B1": {
            "train": {"start": "1965-01", "end": "1999-12", "n_total": int(train_mask.sum()), "n_recession": int(len(train_rec))},
            "val": {"start": "2000-01", "end": "2009-12", "n_total": int(val_mask.sum()), "n_recession": int(len(val_rec)),
                    "recession_months": sorted(val_rec["date"].dt.strftime("%Y-%m").tolist())},
            "test": {"start": "2010-01", "end": "2026-04", "n_total": int(test_mask.sum()), "n_recession": int(len(test_rec)),
                     "recession_months": sorted(test_rec["date"].dt.strftime("%Y-%m").tolist())},
        },
        "note": (
            "TEST has only 2 recession months (2020-03, 2020-04 — COVID-19). "
            "With lag_tolerance=±2, nber_overlap_frozen can achieve recall=1.0 if the "
            "recession cluster covers the Feb-Jun 2020 window. "
            "F1 is dominated by precision; high F1 requires cluster not over-extending into expansion."
        ),
    }

    out = OUT / "nber_positive_count.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("nber_positive_count.json → %s", out)

    print("\n=== NBER recession counts (B1 split) ===")
    for split, data in result["split_B1"].items():
        print(f"  {split}: {data.get('n_recession', '?')}/{data.get('n_total', '?')} recession months")
    print(f"  TEST months: {result['split_B1']['test']['recession_months']}")


# ── 3. Distribution shift: TRAIN vs TEST KS test ─────────────────────────────
def make_distribution_shift() -> None:
    panel_path = (
        ROOT / "data/raw/fred_md_transformed_balanced_2026_04.parquet"
    )
    if not panel_path.exists():
        logger.warning("Panel parquet not found at %s", panel_path)
        return

    df = pd.read_parquet(panel_path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    train_mask = df[date_col] <= pd.Timestamp("1999-12-31")
    test_mask = df[date_col] >= pd.Timestamp("2010-01-01")

    train_df = df[train_mask].drop(columns=[date_col])
    test_df = df[test_mask].drop(columns=[date_col])

    rows = []
    for col in train_df.columns:
        a = train_df[col].dropna().values
        b = test_df[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        stat, pval = stats.ks_2samp(a, b, method="auto")
        rows.append({
            "series": col,
            "ks_stat": round(stat, 4),
            "p_value": round(pval, 4),
            "train_mean": round(float(np.mean(a)), 4),
            "test_mean": round(float(np.mean(b)), 4),
            "mean_shift": round(float(np.mean(b) - np.mean(a)), 4),
            "train_std": round(float(np.std(a)), 4),
            "test_std": round(float(np.std(b)), 4),
            "significant_p05": pval < 0.05,
        })

    shift_df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)

    out = OUT / "b1_distribution_shift.csv"
    shift_df.to_csv(out, index=False)
    logger.info("b1_distribution_shift.csv: %d series → %s", len(shift_df), out)

    n_sig = (shift_df["p_value"] < 0.05).sum()
    print(f"\n=== Distribution shift (TRAIN 1965-1999 vs TEST 2010-2026): {n_sig}/{len(shift_df)} series shifted (p<0.05) ===")
    print(shift_df.head(10)[["series", "ks_stat", "p_value", "mean_shift"]].to_string(index=False))


# ── 4. C1 arithmetic audit: all cached C1 CSVs ───────────────────────────────
def make_c1_arithmetic_audit() -> None:
    """Collect all cached C1 multi-label CSVs and produce a merged audit table."""
    ablation_dirs = {
        "iTransformer": ROOT / "results/clustering_ablation/W6_d7_K4_b1",
    }
    # Phase E encoders
    for enc_dir in (ROOT / "results/phase_e").iterdir() if (ROOT / "results/phase_e").exists() else []:
        if enc_dir.is_dir():
            ablation_dirs[enc_dir.name] = enc_dir / "ablation"
    # Baseline encoders
    for enc_dir in (ROOT / "results/phase_c_comparison").iterdir() if (ROOT / "results/phase_c_comparison").exists() else []:
        if enc_dir.is_dir() and enc_dir.name not in {"phase_c_comparison.csv"}:
            ablation_dirs[enc_dir.name] = enc_dir

    cells = ["pca_kmeans", "pca_hdbscan", "umap_kmeans", "umap_hdbscan", "tsne_kmeans", "tsne_hdbscan"]
    rows = []
    for enc, adir in ablation_dirs.items():
        for cell in cells:
            # Try locked CSV first, then original
            for suffix in [f"c1_multi_label_{cell}_locked.csv", f"c1_multi_label_{cell}.csv"]:
                csv_path = adir / suffix
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        label_col = "label" if "label" in df.columns else "indicator"
                        for _, row in df.iterrows():
                            rows.append({
                                "encoder": enc,
                                "cell": cell,
                                "source": suffix,
                                "indicator": row.get(label_col, "?"),
                                "f1": round(float(row["f1"]), 4) if pd.notna(row.get("f1")) else float("nan"),
                                "auc_roc": round(float(row["auc_roc"]), 4) if "auc_roc" in row and pd.notna(row.get("auc_roc")) else float("nan"),
                                "precision": round(float(row["precision"]), 4) if pd.notna(row.get("precision")) else float("nan"),
                                "recall": round(float(row["recall"]), 4) if pd.notna(row.get("recall")) else float("nan"),
                            })
                    except Exception as exc:
                        logger.debug("Failed to read %s: %s", csv_path, exc)
                    break  # Use first found

    if not rows:
        logger.warning("No C1 CSVs found for audit")
        return

    audit_df = pd.DataFrame(rows)
    out = OUT / "c1_arithmetic_audit.csv"
    audit_df.to_csv(out, index=False)
    logger.info("c1_arithmetic_audit.csv: %d rows → %s", len(audit_df), out)

    print("\n=== C1 audit (iTransformer pca_kmeans) ===")
    itr_pk = audit_df[(audit_df["encoder"] == "iTransformer") & (audit_df["cell"] == "pca_kmeans")]
    if not itr_pk.empty:
        print(itr_pk[["indicator", "f1", "auc_roc", "precision", "recall"]].to_string(index=False))


# ── 5. Falsification summary ──────────────────────────────────────────────────
def make_falsification_summary() -> None:
    fals_csv = ROOT / "results/falsification.csv"
    if not fals_csv.exists():
        logger.warning("falsification.csv not found")
        return

    df = pd.read_csv(fals_csv)
    out = OUT / "falsification_summary.csv"
    df.to_csv(out, index=False)
    logger.info("falsification_summary.csv: %d rows → %s", len(df), out)

    print("\n=== Falsification gate ===")
    print(df.to_string(index=False))


# ── 6. MLflow stage summaries ─────────────────────────────────────────────────
def make_mlflow_summary() -> None:
    mlruns = ROOT / "mlruns"
    if not mlruns.exists():
        logger.warning("mlruns/ not found")
        return

    rows = []
    for exp_dir in mlruns.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta_file = run_dir / "meta.yaml"
            if not meta_file.exists():
                continue
            try:
                import yaml  # noqa: PLC0415
                meta = yaml.safe_load(meta_file.read_text())
                run_name = meta.get("run_name", run_dir.name)
                status = meta.get("status", "?")
                start_time = meta.get("start_time", 0)

                # Read key metrics
                metrics_dir = run_dir / "metrics"
                params_dir = run_dir / "params"
                metric_vals: dict[str, float] = {}
                param_vals: dict[str, str] = {}

                if metrics_dir.exists():
                    for mf in metrics_dir.iterdir():
                        try:
                            lines = mf.read_text().strip().split("\n")
                            last_line = lines[-1].split()
                            if len(last_line) >= 2:
                                metric_vals[mf.name] = float(last_line[1])
                        except Exception:
                            pass

                if params_dir.exists():
                    for pf in params_dir.iterdir():
                        try:
                            param_vals[pf.name] = pf.read_text().strip()
                        except Exception:
                            pass

                rows.append({
                    "exp_id": exp_dir.name,
                    "run_id": run_dir.name[:8],
                    "run_name": run_name,
                    "status": status,
                    "val_loss": round(metric_vals.get("val_loss", float("nan")), 4),
                    "val_nber_f1": round(metric_vals.get("val_nber_f1", float("nan")), 4),
                    "lr": param_vals.get("lr", "?"),
                    "dropout": param_vals.get("dropout", "?"),
                    "d_lat": param_vals.get("d_lat", "?"),
                    "window_size": param_vals.get("window_size", "?"),
                    "n_clusters": param_vals.get("n_clusters", "?"),
                })
            except Exception as exc:
                logger.debug("Failed to read run %s: %s", run_dir, exc)

    if not rows:
        logger.warning("No MLflow runs found")
        return

    mlflow_df = pd.DataFrame(rows).sort_values(["exp_id", "val_loss"])
    out = OUT / "mlflow_stage_results.csv"
    mlflow_df.to_csv(out, index=False)
    logger.info("mlflow_stage_results.csv: %d runs → %s", len(mlflow_df), out)

    # Print stage-1 winner
    finished = mlflow_df[mlflow_df["status"].isin(["FINISHED", "3"])]
    if not finished.empty:
        best = finished.nsmallest(3, "val_loss")
        print("\n=== Top-3 MLflow runs (lowest val_loss) ===")
        print(best[["run_name", "val_loss", "val_nber_f1", "lr", "dropout", "d_lat", "window_size"]].to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    make_bootstrap_stability()
    make_nber_positive_count()
    make_distribution_shift()
    make_c1_arithmetic_audit()
    make_falsification_summary()
    make_mlflow_summary()
    print(f"\nAll diagnostic tables written to: {OUT}")


if __name__ == "__main__":
    main()
