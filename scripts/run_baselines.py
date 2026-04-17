"""Run all 4 baselines for comparison against iTransformer.

For each (W, d_latent, K) from sweep configs, load the same preprocessed data,
run all baselines on the TEST set with non-overlapping windows, adaptive PCA,
and log results to MLflow under experiment "baselines".

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
from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    compute_clustering_metrics,
    fit_adaptive_pca,
    fit_kmeans,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.statistical_tests import (
    permutation_test_silhouette,
)
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
    train_df, val_df, test_df = split_by_date(filled, ref_config.train_end, ref_config.val_end)

    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(train_df, val_df, test_df, scaler)

    tracking_uri = f"file:./{ref_config.results_dir}/mlruns"
    experiment_id = setup_mlflow(tracking_uri, "baselines")

    for config in configs:
        w = config.window_size
        d = config.latent_dim
        k = config.n_clusters

        train_windows = create_windows(train_scaled, w)
        test_windows = create_windows(test_scaled, w)

        # Non-overlapping indices for fair evaluation
        test_no_idx = extract_non_overlapping_indices(n_windows=len(test_windows), window_size=w)
        test_windows_no = test_windows[test_no_idx]
        n_eff_test = compute_effective_n(len(test_windows), w)

        run_name = f"baseline_W{w}_d{d}_K{k}"
        logger.info("Running baselines: %s (n_eff_test=%d)", run_name, n_eff_test)

        # Use adaptive PCA (same as model pipeline) for fair comparison
        baseline_results = run_all_baselines(
            train_windows=train_windows,
            eval_windows=test_windows_no,
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

            if w == 24:
                mlflow.set_tag("analysis_type", "exploratory")
                mlflow.set_tag("power_warning", "W=24: n_eff too low for inference")

            metrics: dict[str, float] = {
                "n_eff_test": float(n_eff_test),
            }
            for baseline_name, result in baseline_results.items():
                metrics[f"{baseline_name}_silhouette"] = result["silhouette"]
                # Also log full clustering metrics for each baseline
                bmetrics = compute_clustering_metrics(result["embeddings"], result["labels"])
                for mname, mval in bmetrics.items():
                    metrics[f"{baseline_name}_{mname}"] = mval

            # Pairwise permutation tests between baselines
            baseline_names = list(baseline_results.keys())
            for i, name_a in enumerate(baseline_names):
                for name_b in baseline_names[i + 1:]:
                    res_a = baseline_results[name_a]
                    res_b = baseline_results[name_b]
                    if len(res_a["embeddings"]) >= 3 and len(res_b["embeddings"]) >= 3:
                        perm = permutation_test_silhouette(
                            res_a["embeddings"], res_a["labels"],
                            res_b["embeddings"], res_b["labels"],
                            n_permutations=5000, random_state=config.seed,
                        )
                        metrics[f"perm_{name_a}_vs_{name_b}_delta"] = perm["observed_diff"]
                        metrics[f"perm_{name_a}_vs_{name_b}_p"] = perm["p_value"]

            log_evaluation_metrics(metrics)

        logger.info(
            "  %s",
            "  ".join(f"{name}={res['silhouette']:.4f}" for name, res in baseline_results.items()),
        )

    logger.info("All baselines complete.")


if __name__ == "__main__":
    main()
