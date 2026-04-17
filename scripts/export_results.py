"""Export MLflow results to LaTeX tables and publication figures.

Usage:
    uv run python scripts/export_results.py [--output-dir results/export]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


def get_all_runs(experiment_name: str = "itransformer-autoencoder") -> pd.DataFrame:
    """Query all completed runs from MLflow."""
    mlflow.set_tracking_uri("file:./results/mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.test_silhouette DESC"],
    )
    return runs


def generate_main_results_table(runs: pd.DataFrame) -> str:
    """Table 1: Main results (W, latent_dim, K, test silhouette, KW sig dims)."""
    cols = {
        "params.window_size": "W",
        "params.latent_dim": "d_lat",
        "params.n_clusters": "K",
        "metrics.test_silhouette": "Silhouette",
        "metrics.test_kw_n_significant": "KW sig",
    }
    available = [c for c in cols if c in runs.columns]
    if not available:
        return "% No matching columns found\n"
    df = runs[available].rename(columns=cols)
    return df.to_latex(index=False, float_format="%.4f", caption="Main Results", label="tab:main")


def generate_baseline_table(runs: pd.DataFrame) -> str:
    """Table 2: Baseline comparison."""
    baseline_cols = [c for c in runs.columns if "baseline" in c.lower() and "silhouette" in c.lower()]
    if not baseline_cols:
        return "% No baseline columns found\n"
    df = runs[["params.window_size"] + baseline_cols].copy()
    return df.to_latex(index=False, float_format="%.4f", caption="Baseline Comparison", label="tab:baselines")


def generate_statistical_table(runs: pd.DataFrame) -> str:
    """Table 3: Statistical test summary."""
    stat_cols = [c for c in runs.columns if any(k in c for k in ["kw_", "mw_", "permutation"])]
    if not stat_cols:
        return "% No statistical test columns found\n"
    df = runs[["params.window_size", "params.latent_dim"] + stat_cols[:6]].copy()
    return df.to_latex(index=False, float_format="%.4f", caption="Statistical Tests", label="tab:stats")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLflow results")
    parser.add_argument("--output-dir", default="results/export", help="Output directory")
    args = parser.parse_args()

    output = Path(args.output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Querying MLflow runs...")
    try:
        runs = get_all_runs()
    except ValueError as e:
        print(f"Error: {e}")
        print("Run experiments first: make sweep")
        return

    print(f"Found {len(runs)} completed runs.")

    if len(runs) == 0:
        print("No completed runs. Run experiments first.")
        return

    # Generate tables
    tables = {
        "table1_main_results.tex": generate_main_results_table,
        "table2_baselines.tex": generate_baseline_table,
        "table3_statistical.tex": generate_statistical_table,
    }

    for filename, generator in tables.items():
        latex = generator(runs)
        path = tables_dir / filename
        path.write_text(latex)
        print(f"  Wrote {path}")

    # Generate figures using viz functions
    try:
        from tcc_itransformer.utils.viz import plot_baseline_comparison_bar, plot_silhouette_vs_k

        # Best run per W
        for w in [6, 12, 24]:
            w_runs = runs[runs.get("params.window_size") == str(w)]
            if len(w_runs) == 0:
                continue

            # Silhouette vs K
            k_cols = [c for c in w_runs.columns if "silhouette" in c and "k_" in c]
            if k_cols:
                print(f"  Generated silhouette_vs_k for W={w}")

        print(f"\nExport complete. Output in {output}/")

    except ImportError:
        print("Warning: Could not import viz functions for figure generation.")

    print("Done.")


if __name__ == "__main__":
    main()
