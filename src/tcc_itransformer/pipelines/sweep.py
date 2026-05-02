"""W x d x K sweep pipeline.

Trains the AE once per (W, d) group (K is post-hoc clustering only), then
evaluates each K on shared embeddings. Same data preprocessing as `single.py`.
"""

from __future__ import annotations

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
from tcc_itransformer.evaluation.baselines import run_all_baselines
from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    clustering_stability,
    compute_clustering_metrics,
    compute_regime_transitions,
    fit_adaptive_pca,
    fit_kmeans,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
)
from tcc_itransformer.evaluation.statistical_tests import (
    kruskal_wallis_per_dim,
    moving_block_bootstrap,
    pairwise_mann_whitney,
    permutation_test_silhouette,
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


def group_configs(config_dir: Path) -> dict[tuple[int, int], list[ExperimentConfig]]:
    """Group YAMLs in `config_dir` by (window_size, latent_dim)."""
    groups: dict[tuple[int, int], list[ExperimentConfig]] = defaultdict(list)
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        cfg = ExperimentConfig.from_yaml(yaml_path)
        groups[(cfg.window_size, cfg.latent_dim)].append(cfg)
    return dict(groups)


def run_sweep(config_dir: Path, dry_run: bool = False) -> None:
    """Train one model per (W, d) and evaluate each K against MLflow."""
    if not config_dir.exists():
        logger.error("Config directory %s does not exist.", config_dir)
        return

    groups = group_configs(config_dir)
    logger.info(
        "Sweep plan: %d model groups, %d total runs",
        len(groups), sum(len(v) for v in groups.values()),
    )
    if dry_run:
        for (w, d), configs in sorted(groups.items()):
            ks = [c.n_clusters for c in configs]
            logger.info("  W=%d d=%d -> K=%s", w, d, ks)
        logger.info("DRY RUN complete. No experiments executed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_config = next(iter(next(iter(groups.values()))))

    set_global_seed(ref_config.seed)
    data, tcodes = load_fred_md(ref_config.data_path)
    transformed = transform_panel(data, tcodes)
    cleaned, _ = drop_high_nan_series(transformed)
    filled = forward_fill_nans(cleaned)
    train_df, val_df, test_df = split_by_date(
        filled, ref_config.train_end, ref_config.val_end,
    )
    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(
        train_df, val_df, test_df, scaler,
    )
    n_series = train_scaled.shape[1]

    for (w, d), configs in sorted(groups.items()):
        logger.info("=== Training model W=%d d=%d ===", w, d)
        train_config = configs[0]
        set_global_seed(train_config.seed)

        tw = create_windows(train_scaled, w)
        vw = create_windows(val_scaled, w)
        testw = create_windows(test_scaled, w)
        train_loader = DataLoader(
            FREDMDWindowDataset(tw), batch_size=train_config.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            FREDMDWindowDataset(vw), batch_size=train_config.batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            FREDMDWindowDataset(testw), batch_size=train_config.batch_size, shuffle=False,
        )

        model = iTransformerAE.from_config(train_config, n_series)
        trainer = Trainer(model, train_config, train_loader, val_loader, device)
        history = trainer.train()
        trainer.checkpoint.load_best(model)

        train_emb = trainer.extract_embeddings(train_loader)
        val_emb = trainer.extract_embeddings(val_loader)
        test_emb = trainer.extract_embeddings(test_loader)

        collapse_info = check_embedding_collapse(train_emb)
        eff_rank = compute_effective_rank(train_emb)
        isotropy = compute_isotropy(train_emb)
        n_eff_train = compute_effective_n(len(train_emb), w)
        n_eff_test = compute_effective_n(len(test_emb), w)

        non_overlap_idx = extract_non_overlapping_indices(
            n_windows=len(train_emb), window_size=w,
        )
        train_emb_no = train_emb[non_overlap_idx]
        pca, n_pca = fit_adaptive_pca(
            train_emb_no, train_config.latent_dim,
            variance_threshold=train_config.pca_variance_threshold,
            n_max=train_config.n_pca_max,
        )
        pca_var_explained = float(np.sum(pca.explained_variance_ratio_))
        train_pca = apply_pca(train_emb_no, pca)

        test_no_idx = extract_non_overlapping_indices(
            n_windows=len(test_emb), window_size=w,
        )
        test_pca = apply_pca(test_emb[test_no_idx], pca)

        val_no_idx = extract_non_overlapping_indices(
            n_windows=len(val_emb), window_size=w,
        )
        val_pca = apply_pca(val_emb[val_no_idx], pca)

        for config in configs:
            k = config.n_clusters
            tracking_uri = f"file:./{config.results_dir}/mlruns"
            experiment_id = setup_mlflow(tracking_uri, config.experiment_name)
            run_name = f"W{w}_d{d}_K{k}"

            km = fit_kmeans(train_pca, k, random_state=config.seed)
            test_labels = km.predict(test_pca)
            test_cluster_metrics = compute_clustering_metrics(test_pca, test_labels)
            val_labels = km.predict(val_pca)
            val_cluster_metrics = compute_clustering_metrics(val_pca, val_labels)

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                log_config(config)
                if w == 24:
                    mlflow.set_tag("analysis_type", "exploratory")
                    mlflow.set_tag(
                        "power_warning", "W=24: n_eff too low for inference",
                    )

                for epoch, (tl, vl) in enumerate(
                    zip(history["train_losses"], history["val_losses"]),
                ):
                    log_epoch_metrics(epoch, tl, vl)

                eval_metrics: dict[str, float] = {
                    "n_collapsed_dims": float(collapse_info["n_collapsed"]),
                    "effective_rank": eff_rank,
                    "isotropy": isotropy,
                    "n_pca_components": float(n_pca),
                    "pca_variance_explained": pca_var_explained,
                    "n_eff_train": float(n_eff_train),
                    "n_eff_test": float(n_eff_test),
                    "best_epoch": float(history["best_epoch"]),
                    "stopped_epoch": float(history["stopped_epoch"]),
                    "final_train_loss": history["train_losses"][-1],
                    "final_val_loss": history["val_losses"][-1],
                    "best_val_loss": float(min(history["val_losses"])),
                    "clustering_stability_ari": clustering_stability(
                        train_pca, k, n_runs=5, random_state=config.seed,
                    ),
                    "test_regime_transitions": float(
                        compute_regime_transitions(test_labels),
                    ),
                }
                for name, val in test_cluster_metrics.items():
                    eval_metrics[f"test_{name}"] = val
                for name, val in val_cluster_metrics.items():
                    eval_metrics[f"val_{name}"] = val

                if len(test_pca) >= 3 and len(np.unique(test_labels)) >= 2:
                    kw = kruskal_wallis_per_dim(test_pca, test_labels)
                    eval_metrics["kw_n_significant"] = float(kw["n_significant"])
                    eval_metrics["kw_mean_effect_size"] = float(
                        np.mean(kw["effect_sizes"]),
                    )
                    mw = pairwise_mann_whitney(test_pca, test_labels)
                    eval_metrics["mw_mean_effect_size"] = float(
                        np.mean(np.abs(mw["effect_sizes"])),
                    )

                baseline_results = run_all_baselines(
                    train_windows=tw,
                    eval_windows=testw[test_no_idx] if len(testw.shape) >= 2 else testw,
                    n_components=n_pca, k=k, random_state=config.seed,
                )
                for bname, bresult in baseline_results.items():
                    eval_metrics[f"baseline_{bname}_silhouette"] = bresult["silhouette"]

                if "raw_pca" in baseline_results and len(test_pca) >= 3:
                    perm = permutation_test_silhouette(
                        test_pca, test_labels,
                        baseline_results["raw_pca"]["embeddings"],
                        baseline_results["raw_pca"]["labels"],
                        n_permutations=10000, random_state=config.seed,
                    )
                    eval_metrics["perm_delta_silhouette"] = perm["observed_diff"]
                    eval_metrics["perm_p_value"] = perm["p_value"]

                if n_eff_test >= 20 and len(test_pca) >= 3:
                    from sklearn.metrics import silhouette_score

                    def _sil_stat(data: np.ndarray) -> float:
                        _km = fit_kmeans(data, k, random_state=config.seed)
                        _lbl = _km.predict(data)
                        if len(np.unique(_lbl)) < 2:
                            return float("nan")
                        return float(silhouette_score(data, _lbl))

                    boot = moving_block_bootstrap(
                        statistic_fn=_sil_stat, data=test_pca,
                        block_length=max(1, w // 2),
                        n_bootstrap=2000, random_state=config.seed,
                    )
                    eval_metrics["bootstrap_silhouette_ci_lower"] = boot["ci_lower"]
                    eval_metrics["bootstrap_silhouette_ci_upper"] = boot["ci_upper"]

                log_evaluation_metrics(eval_metrics)

            logger.info(
                "Logged run %s: test_silhouette=%.4f",
                run_name, test_cluster_metrics["silhouette"],
            )

    logger.info("Sweep complete.")
