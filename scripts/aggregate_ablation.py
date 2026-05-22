"""Aggregate all encoder × downstream × metric results into a single canonical table.

Reads:
  - results/phase_c_comparison/phase_c_comparison.csv  (42 rows, all Phase C metrics)
  - results/clustering_ablation/W6_d7_K4_b1/summary.csv (iTransformer pca_kmeans extras)
  - results/baselines/baselines_panel_baseline_W6_d7_K4.csv (baselines panel)

Writes:
  - results/encoders_panel.csv  — canonical 42-row master table (Phase E ready)

Usage:
    uv run python scripts/aggregate_ablation.py
    uv run python scripts/aggregate_ablation.py --output results/encoders_panel.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Canonical column order for the thesis table (locked 7-panel + Phase C metrics).
CANONICAL_COLUMNS = [
    "encoder",
    "cell",
    "clusterer",
    "n_clusters_test",
    "noise_fraction_test",
    "nber_f1",
    "nber_f1_legacy_maxF1",
    "bai_perron_f1",
    "crisis_window_coverage",
    "test_silhouette",
    "dbcv",
    "c1_sahm_f1",
    "c1_cfnai_f1",
    "c1_chauvet_f1",
    "c1_oecd_f1",
    "c3_indpro_f1",
    "c3_payems_f1",
    "c3_unrate_f1",
    "c4_ari",
]


def _load_phase_c(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded phase_c_comparison: %d rows × %d cols", len(df), len(df.columns))
    return df


def _enrich_from_ablation(df: pd.DataFrame, ablation_path: Path) -> pd.DataFrame:
    """Patch in any extra columns from the iTransformer B1 ablation summary."""
    if not ablation_path.exists():
        logger.warning("Ablation summary not found at %s — skipping enrichment", ablation_path)
        return df

    abl = pd.read_csv(ablation_path)
    # The ablation summary may have crisis_window_coverage or other fields
    # that aren't in phase_c_comparison for the iTransformer+pca_kmeans row.
    for col in abl.columns:
        if col in ("encoder", "cell", "clusterer"):
            continue
        if col not in df.columns:
            continue
        # Find the matching row in df.
        mask = (
            (df["encoder"] == "itransformer")
            & (df["clusterer"] == "kmeans")
            & (df["cell"].str.contains("pca", na=False))
        )
        if "cell" in abl.columns:
            for _, row in abl.iterrows():
                row_mask = mask & (df["cell"] == row.get("cell", ""))
                if row_mask.any() and pd.notna(row.get(col)):
                    df.loc[row_mask, col] = row[col]
        elif mask.any():
            val = abl[col].dropna()
            if not val.empty:
                df.loc[mask, col] = val.iloc[0]

    logger.info("Enriched from ablation summary (%s)", ablation_path.name)
    return df


def aggregate(
    phase_c_csv: Path,
    ablation_summary: Path,
    output: Path,
) -> pd.DataFrame:
    df = _load_phase_c(phase_c_csv)
    df = _enrich_from_ablation(df, ablation_summary)

    # Reorder to canonical columns; add any missing as NaN.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    extra_cols = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    df = df[CANONICAL_COLUMNS + extra_cols]

    # Sort: encoder asc, nber_f1 desc.
    df = df.sort_values(["encoder", "nber_f1"], ascending=[True, False]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote %d rows × %d cols to %s", len(df), len(df.columns), output)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-c-csv",
        type=Path,
        default=ROOT / "results/phase_c_comparison/phase_c_comparison.csv",
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=ROOT / "results/clustering_ablation/W6_d7_K4_b1/summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/encoders_panel.csv",
    )
    args = parser.parse_args()

    if not args.phase_c_csv.exists():
        logger.error("phase_c_comparison.csv not found at %s", args.phase_c_csv)
        logger.error("Run 'uv run python scripts/run_phase_c_comparison.py' first.")
        raise SystemExit(1)

    df = aggregate(args.phase_c_csv, args.ablation_summary, args.output)
    print(df[["encoder", "clusterer", "nber_f1", "bai_perron_f1", "c4_ari"]].to_string())


if __name__ == "__main__":
    main()
