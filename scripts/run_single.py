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
from tcc_itransformer.evaluation.baselines import run_all_baselines
from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    clustering_stability,
    compute_clustering_metrics,
    compute_regime_transitions,
    fit_adaptive_pca,
    fit_kmeans,
    select_k,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
    reconstruction_mse,
)
from tcc_itransformer.evaluation.statistical_tests import (
    kruskal_wallis_per_dim,
    moving_block_bootstrap,
    pairwise_mann_whitney,
    permutation_test_silhouette,
)
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.model.losses import reconstruction_loss
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


def _compute_naive_baseline_mse(
    model: iTransformerAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute naive baseline MSE: predict train mean for every test window."""
    # Compute train mean
    total = None
    count = 0
    for batch in train_loader:
        x = batch[0]
        if total is None:
            total = torch.zeros_like(x[0])
        total += x.sum(dim=0)
        count += x.shape[0]
    train_mean = total / count  # (W, N)

    # MSE on test set using train mean as prediction
    mse_sum = 0.0
    n = 0
    for batch in test_loader:
        x = batch[0]
        pred = train_mean.unsqueeze(0).expand_as(x)
        mse_sum += float(((x - pred) ** 2).mean(dim=(1, 2)).sum())
        n += x.shape[0]
    return mse_sum / max(n, 1)


def _compute_model_test_mse(
    model: iTransformerAE,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute model reconstruction MSE on test set."""
    model.eval()
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            x_hat, _ = model(x)
            mse_sum += float(((x - x_hat) ** 2).mean(dim=(1, 2)).sum())
            n += x.shape[0]
    return mse_sum / max(n, 1)


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

    # --- 7. Extract embeddings (all splits) ---
    train_emb = trainer.extract_embeddings(train_loader)
    val_emb = trainer.extract_embeddings(val_loader)
    test_emb = trainer.extract_embeddings(test_loader)

    # --- 8. Embedding quality ---
    collapse_info = check_embedding_collapse(train_emb)
    eff_rank = compute_effective_rank(train_emb)
    isotropy = compute_isotropy(train_emb)

    # --- 9. Reconstruction MSE on test set ---
    model_test_mse = _compute_model_test_mse(model, test_loader, device)
    naive_baseline_mse = _compute_naive_baseline_mse(model, train_loader, test_loader, device)

    # --- 10. Effective sample size ---
    n_eff_train = compute_effective_n(len(train_emb), config.window_size)
    n_eff_test = compute_effective_n(len(test_emb), config.window_size)

    metrics: dict[str, float] = {
        "n_collapsed_dims": float(collapse_info["n_collapsed"]),
        "effective_rank": eff_rank,
        "isotropy": isotropy,
        "model_test_mse": model_test_mse,
        "naive_baseline_mse": naive_baseline_mse,
        "n_eff_train": float(n_eff_train),
        "n_eff_test": float(n_eff_test),
        "best_epoch": float(history["best_epoch"]),
        "stopped_epoch": float(history["stopped_epoch"]),
        "final_train_loss": history["train_losses"][-1],
        "final_val_loss": history["val_losses"][-1],
        "best_val_loss": float(min(history["val_losses"])),
    }

    # --- 11. PCA on non-overlapping train embeddings ---
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
    pca_var_explained = float(np.sum(pca.explained_variance_ratio_))
    metrics["n_pca_components"] = float(n_pca)
    metrics["pca_variance_explained"] = pca_var_explained

    train_pca = apply_pca(train_emb_no, pca)

    # --- 12. TEST set evaluation (non-overlapping) ---
    test_no_idx = extract_non_overlapping_indices(
        n_windows=len(test_emb),
        window_size=config.window_size,
    )
    test_emb_no = test_emb[test_no_idx]
    test_pca = apply_pca(test_emb_no, pca)

    # Also val for K selection
    val_no_idx = extract_non_overlapping_indices(
        n_windows=len(val_emb),
        window_size=config.window_size,
    )
    val_emb_no = val_emb[val_no_idx]
    val_pca = apply_pca(val_emb_no, pca)

    # --- 13. K selection on train, evaluate on test ---
    k_selection = select_k(train_pca, k_range=[3, 4, 5])
    best_k = k_selection["best_k"]
    metrics["best_k"] = float(best_k)
    for k_val, sil_val in k_selection["scores"].items():
        metrics[f"train_silhouette_K{k_val}"] = sil_val

    # Fit KMeans on train, predict on test
    km = fit_kmeans(train_pca, config.n_clusters, random_state=config.seed)
    test_labels = km.predict(test_pca)
    test_cluster_metrics = compute_clustering_metrics(test_pca, test_labels)
    for metric_name, metric_val in test_cluster_metrics.items():
        metrics[f"test_{metric_name}"] = metric_val

    # Val for comparison logging
    val_labels = km.predict(val_pca)
    val_cluster_metrics = compute_clustering_metrics(val_pca, val_labels)
    for metric_name, metric_val in val_cluster_metrics.items():
        metrics[f"val_{metric_name}"] = metric_val

    # --- 14. Regime transitions ---
    metrics["test_regime_transitions"] = float(compute_regime_transitions(test_labels))

    # --- 15. Clustering stability ---
    stability = clustering_stability(train_pca, config.n_clusters, n_runs=5, random_state=config.seed)
    metrics["clustering_stability_ari"] = stability

    # --- 16. Statistical tests on test set ---
    is_exploratory = config.window_size == 24
    if is_exploratory:
        mlflow.set_tag("analysis_type", "exploratory")
        mlflow.set_tag("power_warning", "W=24: n_eff too low for inference")

    # Kruskal-Wallis per dimension
    if len(test_pca) >= 3 and len(np.unique(test_labels)) >= 2:
        kw_results = kruskal_wallis_per_dim(test_pca, test_labels)
        metrics["kw_n_significant"] = float(kw_results["n_significant"])
        metrics["kw_mean_effect_size"] = float(np.mean(kw_results["effect_sizes"]))

        # Pairwise Mann-Whitney
        mw_results = pairwise_mann_whitney(test_pca, test_labels)
        metrics["mw_mean_effect_size"] = float(np.mean(np.abs(mw_results["effect_sizes"])))

    # --- 17. Baselines + permutation test ---
    baseline_results = run_all_baselines(
        train_windows=train_windows,
        eval_windows=test_windows[test_no_idx] if test_windows.ndim == 3 else test_windows,
        n_components=n_pca,
        k=config.n_clusters,
        random_state=config.seed,
    )

    for bname, bresult in baseline_results.items():
        metrics[f"baseline_{bname}_silhouette"] = bresult["silhouette"]

    # Permutation test: iTransformer vs Raw PCA (primary test)
    if "raw_pca" in baseline_results and len(test_pca) >= 3:
        b1_emb = baseline_results["raw_pca"]["embeddings"]
        b1_labels = baseline_results["raw_pca"]["labels"]
        perm_result = permutation_test_silhouette(
            test_pca, test_labels, b1_emb, b1_labels,
            n_permutations=10000, random_state=config.seed,
        )
        metrics["perm_delta_silhouette"] = perm_result["observed_diff"]
        metrics["perm_p_value"] = perm_result["p_value"]
        metrics["perm_ci_lower"] = perm_result["ci_lower"]
        metrics["perm_ci_upper"] = perm_result["ci_upper"]

    # --- 18. Block bootstrap CI for silhouette (if viable) ---
    if n_eff_train >= 20 and len(train_pca) >= 3:
        def _sil_fn(data: np.ndarray) -> float:
            _km = fit_kmeans(data, config.n_clusters, random_state=config.seed)
            _labels = _km.predict(data)
            if len(np.unique(_labels)) < 2:
                return 0.0
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(data, _labels))

        boot_result = moving_block_bootstrap(
            _sil_fn, train_pca,
            block_length=max(1, config.window_size // 2),
            n_bootstrap=5000, random_state=config.seed,
        )
        metrics["bootstrap_silhouette_ci_lower"] = boot_result["ci_lower"]
        metrics["bootstrap_silhouette_ci_upper"] = boot_result["ci_upper"]

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
