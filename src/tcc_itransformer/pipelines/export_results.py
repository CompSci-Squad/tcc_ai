"""Export MLflow runs to LaTeX tables for the thesis."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def get_all_runs(experiment_name: str = "itransformer-autoencoder") -> pd.DataFrame:
    mlflow.set_tracking_uri("file:./results/mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.test_silhouette DESC"],
    )


def _main_results_table(runs: pd.DataFrame) -> str:
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
    return df.to_latex(
        index=False, float_format="%.4f",
        caption="Main Results", label="tab:main",
    )


def _baseline_table(runs: pd.DataFrame) -> str:
    baseline_cols = [
        c for c in runs.columns
        if "baseline" in c.lower() and "silhouette" in c.lower()
    ]
    if not baseline_cols:
        return "% No baseline columns found\n"
    df = runs[["params.window_size"] + baseline_cols].copy()
    return df.to_latex(
        index=False, float_format="%.4f",
        caption="Baseline Comparison", label="tab:baselines",
    )


def _statistical_table(runs: pd.DataFrame) -> str:
    stat_cols = [
        c for c in runs.columns
        if any(k in c for k in ["kw_", "mw_", "permutation"])
    ]
    if not stat_cols:
        return "% No statistical test columns found\n"
    df = runs[
        ["params.window_size", "params.latent_dim"] + stat_cols[:6]
    ].copy()
    return df.to_latex(
        index=False, float_format="%.4f",
        caption="Statistical Tests", label="tab:stats",
    )


def export_results(output_dir: Path = Path("results/export")) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
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
        return

    for filename, generator in {
        "table1_main_results.tex": _main_results_table,
        "table2_baselines.tex": _baseline_table,
        "table3_statistical.tex": _statistical_table,
    }.items():
        (tables_dir / filename).write_text(generator(runs))
        print(f"  Wrote {tables_dir / filename}")

    print(f"\nExport complete. Output in {output_dir}/")
