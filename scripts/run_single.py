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
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.data.dataset import FREDMDWindowDataset
from tcc_itransformer.data.external_labels import load_usrec
from tcc_itransformer.data.fred_md import load_fred_md, transform_panel
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    load_etl_v2_panel,
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
    select_k_combined,
)
from tcc_itransformer.evaluation.density_clustering import optimize_hdbscan_dbcv
from tcc_itransformer.evaluation.dim_reduction import UMAPConfig, apply_umap, fit_umap
from tcc_itransformer.evaluation.explain import (
    explain_assignment,
    explanations_to_frame,
)
from tcc_itransformer.evaluation.regime_validation import (
    bai_perron_alignment,
    crisis_window_coverage,
    fit_nber_assignment,
    nber_overlap,
    nber_overlap_frozen,
    regime_conditional_moments,
    regime_durations,
    transition_matrix,
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


class _SkipClustering(Exception):
    """Sentinel raised to short-circuit the UMAP+HDBSCAN block when disabled."""


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


def run_experiment(config: ExperimentConfig) -> tuple[dict, dict, dict]:
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
    if config.data_format == "etl_v2_parquet":
        # Already-transformed-and-imputed parquet from tcc_etl v2.
        # Skip stationarity/dropna/ffill — ETL did them.
        panel_df, mask_df = load_etl_v2_panel(
            config.data_path,
            config.mask_path,
            expected_sha256=config.data_sha256,
        )
        train_df, val_df, test_df = split_by_date(
            panel_df, config.train_end, config.val_end,
        )
        if mask_df is not None:
            train_mask_df, val_mask_df, test_mask_df = split_by_date(
                mask_df, config.train_end, config.val_end,
            )
        else:
            train_mask_df = val_mask_df = test_mask_df = None
        logger.info("Loaded ETL-v2 panel: %d series, mask=%s", panel_df.shape[1], mask_df is not None)
    else:
        data, tcodes = load_fred_md(config.data_path)
        transformed = transform_panel(data, tcodes)
        cleaned, _dropped = drop_high_nan_series(transformed)
        filled = forward_fill_nans(cleaned)
        train_df, val_df, test_df = split_by_date(filled, config.train_end, config.val_end)
        train_mask_df = val_mask_df = test_mask_df = None

    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(
        train_df, val_df, test_df, scaler,
    )

    n_series = train_scaled.shape[1]
    train_windows = create_windows(train_scaled, config.window_size)
    val_windows = create_windows(val_scaled, config.window_size)
    test_windows = create_windows(test_scaled, config.window_size)

    # Mask windows (D7): same shape as data windows, Boolean.
    train_mask_w = (
        create_windows(train_mask_df.to_numpy(dtype=bool), config.window_size)
        if train_mask_df is not None else None
    )
    val_mask_w = (
        create_windows(val_mask_df.to_numpy(dtype=bool), config.window_size)
        if val_mask_df is not None else None
    )
    test_mask_w = (
        create_windows(test_mask_df.to_numpy(dtype=bool), config.window_size)
        if test_mask_df is not None else None
    )

    # --- 3. DataLoaders ---
    # D7 policy (pre_analysis_plan addendum 2026-04-29):
    #   train/val: keep all windows, surface mask, use masked MSE in trainer (D7.c).
    #   test: drop windows whose target row has any imputed cell (D7.a).
    use_masked_loss = config.loss_mask_imputed and train_mask_w is not None
    train_ds = FREDMDWindowDataset(
        train_windows, train_mask_w, drop_imputed=False, return_mask=use_masked_loss,
    )
    val_ds = FREDMDWindowDataset(
        val_windows, val_mask_w, drop_imputed=False, return_mask=use_masked_loss,
    )
    test_ds = FREDMDWindowDataset(
        test_windows, test_mask_w,
        drop_imputed=config.eval_drop_imputed_target,
        min_observed_fraction=config.eval_min_observed_fraction,
        return_mask=False,
    )

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

    # Window-end dates for each split (stride=1 in create_windows).
    # If D7 mask filter dropped windows, follow the kept_indices.
    W = config.window_size
    train_all_dates = pd.DatetimeIndex(train_df.index[W - 1 : W - 1 + len(train_windows)])
    val_all_dates = pd.DatetimeIndex(val_df.index[W - 1 : W - 1 + len(val_windows)])
    test_all_dates = pd.DatetimeIndex(test_df.index[W - 1 : W - 1 + len(test_windows)])
    train_emb_dates = train_all_dates[train_ds.kept_indices]
    val_emb_dates = val_all_dates[val_ds.kept_indices]
    test_emb_dates = test_all_dates[test_ds.kept_indices]

    # Initialise artifacts dict early so embeddings are persisted even if the
    # downstream UMAP+HDBSCAN pipeline raises.
    principal_artifacts: dict[str, object] = {
        "embeddings_train": (train_emb, train_emb_dates),
        "embeddings_val": (val_emb, val_emb_dates),
        "embeddings_test": (test_emb, test_emb_dates),
    }

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
    # When D7.a drops every test window (e.g. the panel tail is fully imputed
    # for at least one series in every potential target row), test_emb is
    # empty. We still want the run to complete: train+val embeddings are
    # exported, and the AE-only metrics block above is intact. Skip
    # everything that requires a non-empty test set.
    test_split_empty = len(test_emb) == 0
    if test_split_empty:
        logger.warning(
            "TEST split is empty after D7.a target-row filter; skipping test-side "
            "PCA/cluster/HDBSCAN evaluation. Train and val embeddings are still exported."
        )
        metrics["test_split_empty"] = 1.0
        test_emb_no = np.zeros((0, train_emb.shape[1]), dtype=train_emb.dtype)
        test_pca = np.zeros((0, n_pca), dtype=train_pca.dtype)
    else:
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
    # Skipped when config.run_clustering=False (e.g. SageMaker AE-only sweeps).
    # Embeddings + reconstruction metrics are still persisted for downstream
    # clustering ablation via scripts/run_clustering_ablation.py.
    if config.run_clustering:
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

        # --- 14b. Combined K selection (Silhouette + BIC/GMM) — pre_projeto §4.4 ---
        if len(train_pca) >= 10:
            k_combined = select_k_combined(train_pca, k_range=[3, 4, 5], random_state=config.seed)
            metrics["best_k_combined"] = float(k_combined["best_k"])
            for k_val, v in k_combined["combined"].items():
                metrics[f"k_combined_score_K{k_val}"] = float(v)
    else:
        logger.info("Skipping KMeans/K-selection block (run_clustering=False).")

    # --- 14c. Principal pipeline: UMAP -> HDBSCAN -> validation -> explain ---
    # Pre_projeto §4.3 Modules 2-4 + §4.4 validation layers 2-3.
    # Skipped when config.run_clustering=False (e.g. SageMaker AE-only sweeps).
    # Embeddings + reconstruction metrics are still persisted for downstream
    # clustering ablation via scripts/run_clustering_ablation.py.
    try:
        if not config.run_clustering:
            logger.info("Skipping UMAP+HDBSCAN block (run_clustering=False).")
            raise _SkipClustering
        umap_cfg = UMAPConfig(
            n_components=config.umap_n_components,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            random_state=config.seed,
        )
        umap_reducer = fit_umap(train_emb_no, umap_cfg)
        train_umap = apply_umap(train_emb_no, umap_reducer)
        val_umap = apply_umap(val_emb_no, umap_reducer)
        test_umap = apply_umap(test_emb_no, umap_reducer)

        # HDBSCAN with DBCV optimization on TRAIN (no leakage).
        hdb_best, hdb_log = optimize_hdbscan_dbcv(
            train_umap,
            min_cluster_sizes=tuple(config.hdbscan_min_cluster_sizes),
            min_samples_grid=tuple(config.hdbscan_min_samples_grid),
            max_noise_fraction=config.hdbscan_max_noise_fraction,
        )
        metrics["hdbscan_train_dbcv"] = hdb_best.dbcv
        metrics["hdbscan_train_n_clusters"] = float(hdb_best.n_clusters)
        metrics["hdbscan_train_noise_fraction"] = hdb_best.noise_fraction
        metrics["hdbscan_min_cluster_size"] = float(hdb_best.min_cluster_size)
        metrics["hdbscan_min_samples"] = float(hdb_best.min_samples)

        # Apply best clusterer to TEST via approximate_predict (HDBSCAN soft API).
        try:
            import hdbscan as _hdbscan  # local import; already a hard dep
            hdb_test_labels, hdb_test_probs = _hdbscan.approximate_predict(
                hdb_best.clusterer, test_umap,
            )
            hdb_val_labels, _hdb_val_probs = _hdbscan.approximate_predict(
                hdb_best.clusterer, val_umap,
            )
        except Exception:  # pragma: no cover — fall back to refitting on test
            from tcc_itransformer.evaluation.density_clustering import fit_hdbscan
            _refit = fit_hdbscan(
                test_umap,
                min_cluster_size=hdb_best.min_cluster_size,
                min_samples=hdb_best.min_samples,
            )
            hdb_test_labels = _refit.labels
            hdb_test_probs = _refit.probabilities
            _refit_val = fit_hdbscan(
                val_umap,
                min_cluster_size=hdb_best.min_cluster_size,
                min_samples=hdb_best.min_samples,
            )
            hdb_val_labels = _refit_val.labels

        n_test_clusters = int(len(set(hdb_test_labels)) - (1 if -1 in hdb_test_labels else 0))
        metrics["hdbscan_test_n_clusters"] = float(n_test_clusters)
        metrics["hdbscan_test_noise_fraction"] = float(np.mean(hdb_test_labels == -1))

        # Recover dates for non-overlapping TEST windows (use last timestep of each window).
        stride_test = config.window_size  # extract_non_overlapping_indices uses stride=W
        test_dates_all = test_df.index
        test_window_dates = pd.DatetimeIndex(
            [
                test_dates_all[i * stride_test + config.window_size - 1]
                for i in range(len(test_emb_no))
                if i * stride_test + config.window_size - 1 < len(test_dates_all)
            ]
        )
        # Trim labels/probs to match available dates.
        m = len(test_window_dates)
        hdb_test_labels = np.asarray(hdb_test_labels[:m])
        hdb_test_probs = np.asarray(hdb_test_probs[:m])

        # ---- Layer 2: NBER overlap (Hungarian on VAL, frozen on TEST) ----
        # Q5 Tier 1 fix: legacy nber_overlap picked the cluster with maximum
        # F1 on TEST, a textbook post-hoc selection bias. We now fit the
        # cluster→regime mapping on VAL only and apply it verbatim to TEST.
        try:
            usrec = load_usrec(config.nber_usrec_path)
            # Recover dates for non-overlapping VAL windows (mirror of TEST).
            val_dates_all = val_df.index
            val_window_dates = pd.DatetimeIndex(
                [
                    val_dates_all[i * stride_test + config.window_size - 1]
                    for i in range(len(val_emb_no))
                    if i * stride_test + config.window_size - 1 < len(val_dates_all)
                ]
            )
            mv = len(val_window_dates)
            hdb_val_labels = np.asarray(hdb_val_labels[:mv])

            assignment = fit_nber_assignment(
                hdb_val_labels, val_window_dates, usrec, lead=0, lag=2,
            )
            mlflow.set_tag("nber_assignment", str(assignment))
            nber_res = nber_overlap_frozen(
                hdb_test_labels, test_window_dates, usrec, assignment,
                lead=0, lag=2,
            )
            metrics["nber_f1"] = nber_res.f1
            metrics["nber_precision"] = nber_res.precision
            metrics["nber_recall"] = nber_res.recall
            metrics["nber_matched_cluster"] = float(nber_res.matched_cluster)

            # Also report legacy max-F1 metric explicitly tagged as biased,
            # so reviewers can compare and the gap is visible in MLflow.
            legacy = nber_overlap(hdb_test_labels, test_window_dates, usrec, lead=0, lag=2)
            metrics["nber_f1_legacy_maxF1"] = legacy.f1
        except FileNotFoundError as exc:
            logger.warning("Skipping NBER overlap: %s", exc)
            mlflow.set_tag("nber_status", "snapshot_missing")

        # ---- Layer 2: Bai-Perron alignment on first PC of test panel ----
        try:
            from sklearn.decomposition import PCA as _PCA
            test_panel_arr = test_scaled[
                config.window_size - 1 : config.window_size - 1 + m * stride_test : stride_test
            ]
            if len(test_panel_arr) >= 10:
                pc1 = _PCA(n_components=1).fit_transform(test_panel_arr).ravel()
                bp = bai_perron_alignment(hdb_test_labels, pc1, penalty=10.0, tolerance=2)
                metrics["bai_perron_f1"] = bp["f1"]
                metrics["bai_perron_n_breakpoints"] = float(bp["n_breakpoints"])
        except Exception as exc:  # pragma: no cover
            logger.warning("Bai-Perron alignment failed: %s", exc)

        # ---- Layer 2: crisis windows coverage ----
        crisis = crisis_window_coverage(hdb_test_labels, test_window_dates)
        for crisis_name, info in crisis.items():
            metrics[f"crisis_{crisis_name}_dominant_cluster"] = float(info.get("dominant_cluster", -1))

        # ---- Layer 3: interpretability ----
        # Build a per-window panel: mean of each window across W timesteps, in feature space.
        feat_cols = list(filled.columns)
        per_window_panel = pd.DataFrame(
            test_windows[test_no_idx][:m].mean(axis=1),
            index=test_window_dates,
            columns=feat_cols,
        )
        moments = regime_conditional_moments(per_window_panel, hdb_test_labels)
        durations = regime_durations(hdb_test_labels)
        trans = transition_matrix(hdb_test_labels)
        metrics["n_regime_transitions_hdbscan"] = float(
            sum(1 for i in range(1, len(hdb_test_labels)) if hdb_test_labels[i] != hdb_test_labels[i - 1])
        )
        if not durations.empty:
            metrics["mean_regime_duration_months"] = float(durations["mean_duration"].mean())

        # ---- Module 4: explanations ----
        explanations = explain_assignment(
            per_window_panel,
            hdb_test_labels,
            probabilities=hdb_test_probs,
            top_k=config.explain_top_k,
            membership_source="soft",
        )
        explain_df = explanations_to_frame(explanations)

        principal_artifacts.update({
            "hdb_grid_log": hdb_log,
            "moments": moments,
            "durations": durations,
            "transition_matrix": trans,
            "explanations": explain_df,
            "test_dates": test_window_dates,
            "test_labels": hdb_test_labels,
            "test_probs": hdb_test_probs,
        })
    except _SkipClustering:
        pass
    except Exception as exc:  # pragma: no cover
        logger.exception("Principal UMAP+HDBSCAN pipeline failed: %s", exc)
        mlflow.set_tag("principal_pipeline_status", "failed")

    # --- 15-18. Clustering-dependent statistics — skipped when clustering disabled. ---
    if config.run_clustering:
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

    return metrics, history, principal_artifacts


def run_full_pipeline(
    config: ExperimentConfig,
    model_dir: Path | str | None = None,
    aux_dir: Path | str | None = None,
) -> dict:
    """End-to-end pipeline used by SageMaker entrypoint and CLI alike.

    Persists:
        - {model_dir}/metrics.json     — flat metrics dict
        - {aux_dir}/history.json       — per-epoch losses
        - {aux_dir}/explanations.parquet, moments.parquet,
          transition_matrix.parquet, durations.parquet,
          hdbscan_grid.json (when principal pipeline succeeds)

    Returns:
        The metrics dict.
    """
    import json

    model_dir = Path(model_dir) if model_dir else Path(config.results_dir) / "model"
    aux_dir = Path(aux_dir) if aux_dir else Path(config.results_dir) / "aux"
    model_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    metrics, history, artifacts = run_experiment(config)

    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (aux_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    config.to_yaml(model_dir / "config.yaml")

    _persist_principal_artifacts(artifacts, aux_dir)

    # Best-effort MLflow log_artifact (no-op outside an active run).
    try:
        if mlflow.active_run() is not None:
            for fname in (
                "metrics.json",
            ):
                fp = model_dir / fname
                if fp.exists():
                    mlflow.log_artifact(str(fp))
            for fname in (
                "history.json",
                "explanations.parquet",
                "moments.parquet",
                "transition_matrix.parquet",
                "durations.parquet",
                "hdbscan_grid.json",
                "test_labels.parquet",
                "embeddings/Z_train.parquet",
                "embeddings/Z_val.parquet",
                "embeddings/Z_test.parquet",
            ):
                fp = aux_dir / fname
                if fp.exists():
                    mlflow.log_artifact(str(fp))
    except Exception:  # pragma: no cover
        logger.debug("MLflow log_artifact skipped", exc_info=True)

    logger.info("Persisted metrics, history, and principal artifacts to %s / %s", model_dir, aux_dir)
    return metrics


def _persist_principal_artifacts(artifacts: dict, aux_dir: Path) -> None:
    import json as _json

    if not artifacts:
        return
    try:
        # ---- Embeddings: Z_{train,val,test}.parquet ---------------------
        # Schema: date (datetime64[ns]), z_0 .. z_{d-1} (float32).
        # These are the canonical inputs for downstream clustering ablations
        # (UMAP vs t-SNE × KMeans vs HDBSCAN), so persist them unconditionally.
        emb_dir = aux_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        import pandas as _pd
        for split in ("train", "val", "test"):
            entry = artifacts.get(f"embeddings_{split}")
            if entry is None:
                continue
            Z, dates = entry
            d = Z.shape[1]
            df = _pd.DataFrame(
                Z.astype("float32"),
                columns=[f"z_{i}" for i in range(d)],
            )
            df.insert(0, "date", _pd.DatetimeIndex(dates))
            df.to_parquet(emb_dir / f"Z_{split}.parquet", index=False)

        moments = artifacts.get("moments")
        if moments is not None:
            moments.to_parquet(aux_dir / "moments.parquet")
        durations = artifacts.get("durations")
        if durations is not None:
            durations.to_parquet(aux_dir / "durations.parquet")
        trans = artifacts.get("transition_matrix")
        if trans is not None:
            trans.to_parquet(aux_dir / "transition_matrix.parquet")
        explain_df = artifacts.get("explanations")
        if explain_df is not None:
            explain_df.to_parquet(aux_dir / "explanations.parquet")
        grid = artifacts.get("hdb_grid_log")
        if grid is not None:
            (aux_dir / "hdbscan_grid.json").write_text(_json.dumps(grid, indent=2, default=str))
        labels = artifacts.get("test_labels")
        dates = artifacts.get("test_dates")
        probs = artifacts.get("test_probs")
        if labels is not None and dates is not None:
            import pandas as _pd
            df = _pd.DataFrame({"date": list(dates), "label": list(labels)})
            if probs is not None:
                df["probability"] = list(probs)
            df.to_parquet(aux_dir / "test_labels.parquet")
    except Exception:  # pragma: no cover
        logger.exception("Failed to persist some principal artifacts")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    tracking_uri = f"file:./{config.results_dir}/mlruns"
    experiment_id = setup_mlflow(tracking_uri, config.experiment_name)

    run_name = f"W{config.window_size}_d{config.latent_dim}_K{config.n_clusters}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        log_config(config)
        metrics, history, artifacts = run_experiment(config)

        # Log per-epoch metrics
        for epoch, (tl, vl) in enumerate(
            zip(history["train_losses"], history["val_losses"]),
        ):
            log_epoch_metrics(epoch, tl, vl)

        # Log evaluation metrics
        log_evaluation_metrics(metrics)

        # Persist principal artifacts to aux dir + log to MLflow.
        aux_dir = Path(config.results_dir) / "aux" / run_name
        aux_dir.mkdir(parents=True, exist_ok=True)
        _persist_principal_artifacts(artifacts, aux_dir)
        for fname in ("explanations.parquet", "moments.parquet", "transition_matrix.parquet",
                      "durations.parquet", "hdbscan_grid.json", "test_labels.parquet"):
            fp = aux_dir / fname
            if fp.exists():
                mlflow.log_artifact(str(fp))

    logger.info("Experiment complete. Run: %s", run_name)


if __name__ == "__main__":
    main()
