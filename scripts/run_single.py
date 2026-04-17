"""Run a single experiment configuration end-to-end.

Usage:
    python scripts/run_single.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
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
    parser = argparse.ArgumentParser(description="Run single iTransformer experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def run_experiment(config: ExperimentConfig) -> tuple[dict, dict]:
    """Execute the full experiment pipeline for one configuration.

    Args:
        config: The experiment configuration.

    Returns:
        Tuple of (evaluation_metrics, training_history).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- 1. Seed ---
    set_global_seed(config.seed)

    # --- 2. Data loading & preprocessing ---
    data, tcodes = load_fred_md(config.data_path)
    transformed = transform_panel(data, tcodes)
    cleaned, _dropped = drop_high_nan_series(transformed)
    filled = forward_fill_nans(cleaned)
    train_df, val_df, test_df = split_by_date(filled, config.train_end, config.val_end)

    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(
        train_df, val_df, test_df, scaler,
    )

    n_series = train_scaled.shape[1]
    train_windows = create_windows(train_scaled, config.window_size)
    val_windows = create_windows(val_scaled, config.window_size)
    test_windows = create_windows(test_scaled, config.window_size)

    # --- 3. DataLoaders ---
    train_ds = FREDMDWindowDataset(train_windows)
    val_ds = FREDMDWindowDataset(val_windows)
    test_ds = FREDMDWindowDataset(test_windows)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # --- 4. Model ---
    model = iTransformerAE.from_config(config, n_series)

    # --- 5. Train ---
    trainer = Trainer(model, config, train_loader, val_loader, device)
    history = trainer.train()

    # --- 6. Load best checkpoint ---
    trainer.checkpoint.load_best(model)

    # --- 7. Extract embeddings ---
    train_emb = trainer.extract_embeddings(train_loader)
    val_emb = trainer.extract_embeddings(val_loader)
    test_emb = trainer.extract_embeddings(test_loader)

    # --- 8. Embedding quality ---
    collapse_info = check_embedding_collapse(train_emb)
    eff_rank = compute_effective_rank(train_emb)
    isotropy = compute_isotropy(train_emb)

    metrics: dict[str, float] = {
        "n_collapsed_dims": float(collapse_info["n_collapsed"]),
        "effective_rank": eff_rank,
        "isotropy": isotropy,
        "best_epoch": float(history["best_epoch"]),
        "stopped_epoch": float(history["stopped_epoch"]),
        "final_train_loss": history["train_losses"][-1],
        "final_val_loss": history["val_losses"][-1],
        "best_val_loss": float(min(history["val_losses"])),
    }

    # --- 9. PCA on non-overlapping train embeddings ---
    non_overlap_idx = extract_non_overlapping_indices(
        n_windows=len(train_emb),
        window_size=config.window_size,
    )
    train_emb_no = train_emb[non_overlap_idx]
    pca, n_pca = fit_adaptive_pca(
        train_emb_no,
        config.latent_dim,
        variance_threshold=config.pca_variance_threshold,
        n_max=config.n_pca_max,
    )
    metrics["n_pca_components"] = float(n_pca)

    train_pca = apply_pca(train_emb_no, pca)

    # --- 10. K-Means for each K ---
    val_emb_no_idx = extract_non_overlapping_indices(
        n_windows=len(val_emb),
        window_size=config.window_size,
    )
    val_emb_no = val_emb[val_emb_no_idx]
    val_pca = apply_pca(val_emb_no, pca)

    for k in [3, 4, 5]:
        km = fit_kmeans(train_pca, k, random_state=config.seed)
        val_labels = km.predict(val_pca)
        cluster_metrics = compute_clustering_metrics(val_pca, val_labels)

        for metric_name, metric_val in cluster_metrics.items():
            metrics[f"K{k}_{metric_name}"] = metric_val

    return metrics, history


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    tracking_uri = f"file:./{config.results_dir}/mlruns"
    experiment_id = setup_mlflow(tracking_uri, config.experiment_name)

    run_name = f"W{config.window_size}_d{config.latent_dim}_K{config.n_clusters}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        log_config(config)
        metrics, history = run_experiment(config)

        # Log per-epoch metrics
        for epoch, (tl, vl) in enumerate(
            zip(history["train_losses"], history["val_losses"]),
        ):
            log_epoch_metrics(epoch, tl, vl)

        # Log evaluation metrics
        log_evaluation_metrics(metrics)

    logger.info("Experiment complete. Run: %s", run_name)


if __name__ == "__main__":
    main()
