"""Run all 4 baselines for comparison against iTransformer.

For each (W, d_latent, K) from sweep configs, load the same preprocessed data,
run all baselines, and log results to MLflow under experiment "baselines".

Usage:
    python scripts/run_baselines.py --config-dir configs/sweep
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.data.fred_md import load_fred_md, transform_panel
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    scale_splits,
    split_by_date,
)
from tcc_itransformer.evaluation.baselines import run_all_baselines
from tcc_itransformer.seed import set_global_seed
from tcc_itransformer.tracking.mlflow_utils import (
    log_evaluation_metrics,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline comparisons.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/sweep",
        help="Directory containing sweep YAML configs.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args = parse_args()
    config_dir = Path(args.config_dir)

    if not config_dir.exists():
        logger.error("Config directory %s does not exist.", config_dir)
        return

    configs = [
        ExperimentConfig.from_yaml(p)
        for p in sorted(config_dir.glob("*.yaml"))
    ]

    if not configs:
        logger.error("No YAML configs found in %s", config_dir)
        return

    ref_config = configs[0]
    set_global_seed(ref_config.seed)

    # --- Shared data loading ---
    data, tcodes = load_fred_md(ref_config.data_path)
    transformed = transform_panel(data, tcodes)
    cleaned, _dropped = drop_high_nan_series(transformed)
    filled = forward_fill_nans(cleaned)
    train_df, val_df, _test_df = split_by_date(filled, ref_config.train_end, ref_config.val_end)

    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, _test_scaled = scale_splits(train_df, val_df, _test_df, scaler)

    tracking_uri = f"file:./{ref_config.results_dir}/mlruns"
    experiment_id = setup_mlflow(tracking_uri, "baselines")

    for config in configs:
        w = config.window_size
        d = config.latent_dim
        k = config.n_clusters

        train_windows = create_windows(train_scaled, w)
        val_windows = create_windows(val_scaled, w)

        run_name = f"baseline_W{w}_d{d}_K{k}"
        logger.info("Running baselines: %s", run_name)

        baseline_results = run_all_baselines(
            train_windows=train_windows,
            eval_windows=val_windows,
            n_components=d,
            k=k,
            random_state=config.seed,
        )

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_params({
                "window_size": w,
                "latent_dim": d,
                "n_clusters": k,
                "seed": config.seed,
            })

            metrics: dict[str, float] = {}
            for baseline_name, result in baseline_results.items():
                metrics[f"{baseline_name}_silhouette"] = result["silhouette"]

            log_evaluation_metrics(metrics)

        logger.info(
            "  %s",
            "  ".join(f"{name}={res['silhouette']:.4f}" for name, res in baseline_results.items()),
        )

    logger.info("All baselines complete.")


if __name__ == "__main__":
    main()
