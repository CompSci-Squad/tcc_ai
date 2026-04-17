"""Run full hyperparameter sweep.

Grid: W ∈ {6, 12, 24} × d_latent ∈ {6, 7, 8, 9} = 12 training runs
Each training run → evaluate K ∈ {3, 4, 5} = 3 eval combos
Total: 12 × 3 = 36 MLflow runs

Usage:
    python scripts/run_sweep.py --config-dir configs/sweep
    python scripts/run_sweep.py --config-dir configs/sweep --dry-run
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.data.dataset import FREDMDWindowDataset
from tcc_itransformer.data.fred_md import load_fred_md, transform_panel
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    scale_splits,
    split_by_date,
)
from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    compute_clustering_metrics,
    fit_adaptive_pca,
    fit_kmeans,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
)
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.seed import set_global_seed
from tcc_itransformer.tracking.mlflow_utils import (
    log_config,
    log_epoch_metrics,
    log_evaluation_metrics,
    setup_mlflow,
)
from tcc_itransformer.training.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/sweep",
        help="Directory containing sweep YAML configs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing.",
    )
    return parser.parse_args()


def group_configs(config_dir: Path) -> dict[tuple[int, int], list[ExperimentConfig]]:
    """Group configs by (window_size, latent_dim) — same model, different K.

    Returns:
        Mapping from (W, d) to list of configs (one per K).
    """
    groups: dict[tuple[int, int], list[ExperimentConfig]] = defaultdict(list)

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        config = ExperimentConfig.from_yaml(yaml_path)
        key = (config.window_size, config.latent_dim)
        groups[key].append(config)

    return dict(groups)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args = parse_args()
    config_dir = Path(args.config_dir)

    if not config_dir.exists():
        logger.error("Config directory %s does not exist. Run generate_sweep_configs.py first.", config_dir)
        return

    groups = group_configs(config_dir)
    logger.info(
        "Sweep plan: %d model groups, %d total runs",
        len(groups),
        sum(len(v) for v in groups.values()),
    )

    if args.dry_run:
        for (w, d), configs in sorted(groups.items()):
            k_values = [c.n_clusters for c in configs]
            logger.info("  W=%d d=%d → K=%s", w, d, k_values)
        logger.info("DRY RUN complete. No experiments executed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use first config for shared data params (all share same data path/splits)
    ref_config = next(iter(next(iter(groups.values()))))

    # --- Data loading (shared across all runs) ---
    set_global_seed(ref_config.seed)
    data, tcodes = load_fred_md(ref_config.data_path)
    transformed = transform_panel(data, tcodes)
    cleaned, _dropped = drop_high_nan_series(transformed)
    filled = forward_fill_nans(cleaned)
    train_df, val_df, test_df = split_by_date(filled, ref_config.train_end, ref_config.val_end)

    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(train_df, val_df, test_df, scaler)
    n_series = train_scaled.shape[1]

    for (w, d), configs in sorted(groups.items()):
        logger.info("=== Training model W=%d d=%d ===", w, d)

        # Use first config for model training (they differ only in n_clusters)
        train_config = configs[0]
        set_global_seed(train_config.seed)

        # Windows for this window size
        tw = create_windows(train_scaled, w)
        vw = create_windows(val_scaled, w)
        _testw = create_windows(test_scaled, w)

        train_ds = FREDMDWindowDataset(tw)
        val_ds = FREDMDWindowDataset(vw)

        train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False)

        # Train model once per (W, d)
        model = iTransformerAE.from_config(train_config, n_series)
        trainer = Trainer(model, train_config, train_loader, val_loader, device)
        history = trainer.train()
        trainer.checkpoint.load_best(model)

        # Extract embeddings
        train_emb = trainer.extract_embeddings(train_loader)
        val_emb = trainer.extract_embeddings(val_loader)

        # Embedding quality
        collapse_info = check_embedding_collapse(train_emb)
        eff_rank = compute_effective_rank(train_emb)
        isotropy = compute_isotropy(train_emb)

        # PCA
        non_overlap_idx = extract_non_overlapping_indices(n_windows=len(train_emb), window_size=w)
        train_emb_no = train_emb[non_overlap_idx]
        pca, n_pca = fit_adaptive_pca(
            train_emb_no,
            train_config.latent_dim,
            variance_threshold=train_config.pca_variance_threshold,
            n_max=train_config.n_pca_max,
        )
        train_pca = apply_pca(train_emb_no, pca)

        val_no_idx = extract_non_overlapping_indices(n_windows=len(val_emb), window_size=w)
        val_emb_no = val_emb[val_no_idx]
        val_pca = apply_pca(val_emb_no, pca)

        # Evaluate each K
        for config in configs:
            k = config.n_clusters
            tracking_uri = f"file:./{config.results_dir}/mlruns"
            experiment_id = setup_mlflow(tracking_uri, config.experiment_name)
            run_name = f"W{w}_d{d}_K{k}"

            km = fit_kmeans(train_pca, k, random_state=config.seed)
            val_labels = km.predict(val_pca)
            cluster_metrics = compute_clustering_metrics(val_pca, val_labels)

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                log_config(config)

                for epoch, (tl, vl) in enumerate(
                    zip(history["train_losses"], history["val_losses"]),
                ):
                    log_epoch_metrics(epoch, tl, vl)

                eval_metrics: dict[str, float] = {
                    "n_collapsed_dims": float(collapse_info["n_collapsed"]),
                    "effective_rank": eff_rank,
                    "isotropy": isotropy,
                    "n_pca_components": float(n_pca),
                    "best_epoch": float(history["best_epoch"]),
                    "stopped_epoch": float(history["stopped_epoch"]),
                    "final_train_loss": history["train_losses"][-1],
                    "final_val_loss": history["val_losses"][-1],
                    "best_val_loss": float(min(history["val_losses"])),
                }

                for metric_name, metric_val in cluster_metrics.items():
                    eval_metrics[metric_name] = metric_val

                log_evaluation_metrics(eval_metrics)

            logger.info("Logged run %s: silhouette=%.4f", run_name, cluster_metrics["silhouette"])

    logger.info("Sweep complete.")


if __name__ == "__main__":
    main()
