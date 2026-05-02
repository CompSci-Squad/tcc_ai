"""Single-config end-to-end pipeline: train iTransformer-AE, extract embeddings,
evaluate (PCA+KMeans + UMAP+HDBSCAN + NBER/Bai-Perron validation + baselines).

Used by both the CLI (`tcc train single`) and the SageMaker entrypoint.
"""

from __future__ import annotations

import json
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
from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
)
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


class _SkipClustering(Exception):
    """Sentinel to short-circuit the UMAP+HDBSCAN block when run_clustering=False."""


def _compute_naive_baseline_mse(
    train_loader: DataLoader, test_loader: DataLoader,
) -> float:
    """Predict the train-mean window for every test window; report MSE."""
    total = None
    count = 0
    for batch in train_loader:
        x = batch[0]
        if total is None:
            total = torch.zeros_like(x[0])
        total += x.sum(dim=0)
        count += x.shape[0]
    train_mean = total / count

    mse_sum = 0.0
    n = 0
    for batch in test_loader:
        x = batch[0]
        pred = train_mean.unsqueeze(0).expand_as(x)
        mse_sum += float(((x - pred) ** 2).mean(dim=(1, 2)).sum())
        n += x.shape[0]
    return mse_sum / max(n, 1)


def _compute_model_test_mse(
    model: iTransformerAE, test_loader: DataLoader, device: torch.device,
) -> float:
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
    """Train + evaluate one config end-to-end. Returns (metrics, history, artifacts)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    set_global_seed(config.seed)

    # Data: ETL-v2 parquet skips stationarity/dropna/ffill (already done upstream).
    if config.data_format == "etl_v2_parquet":
        panel_df, mask_df = load_etl_v2_panel(
            config.data_path, config.mask_path, expected_sha256=config.data_sha256,
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
        logger.info(
            "Loaded ETL-v2 panel: %d series, mask=%s",
            panel_df.shape[1], mask_df is not None,
        )
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

    # D7 policy (pre_analysis_plan addendum 2026-04-29):
    #   train/val keep all windows, surface mask, apply masked MSE in trainer.
    #   test drops windows whose target row has any imputed cell (D7.a).
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

    model = iTransformerAE.from_config(config, n_series)
    trainer = Trainer(model, config, train_loader, val_loader, device)
    history = trainer.train()
    trainer.checkpoint.load_best(model)

    train_emb = trainer.extract_embeddings(train_loader)
    val_emb = trainer.extract_embeddings(val_loader)
    test_emb = trainer.extract_embeddings(test_loader)

    W = config.window_size
    train_all_dates = pd.DatetimeIndex(train_df.index[W - 1 : W - 1 + len(train_windows)])
    val_all_dates = pd.DatetimeIndex(val_df.index[W - 1 : W - 1 + len(val_windows)])
    test_all_dates = pd.DatetimeIndex(test_df.index[W - 1 : W - 1 + len(test_windows)])
    train_emb_dates = train_all_dates[train_ds.kept_indices]
    val_emb_dates = val_all_dates[val_ds.kept_indices]
    test_emb_dates = test_all_dates[test_ds.kept_indices]

    # Persist embeddings even if downstream UMAP+HDBSCAN raises.
    artifacts: dict[str, object] = {
        "embeddings_train": (train_emb, train_emb_dates),
        "embeddings_val": (val_emb, val_emb_dates),
        "embeddings_test": (test_emb, test_emb_dates),
    }

    collapse_info = check_embedding_collapse(train_emb)
    eff_rank = compute_effective_rank(train_emb)
    isotropy = compute_isotropy(train_emb)
    model_test_mse = _compute_model_test_mse(model, test_loader, device)
    naive_baseline_mse = _compute_naive_baseline_mse(train_loader, test_loader)
    n_eff_train = compute_effective_n(len(train_emb), W)
    n_eff_test = compute_effective_n(len(test_emb), W)

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

    non_overlap_idx = extract_non_overlapping_indices(
        n_windows=len(train_emb), window_size=W,
    )
    train_emb_no = train_emb[non_overlap_idx]
    pca, n_pca = fit_adaptive_pca(
        train_emb_no, config.latent_dim,
        variance_threshold=config.pca_variance_threshold,
        n_max=config.n_pca_max,
    )
    pca_var_explained = float(np.sum(pca.explained_variance_ratio_))
    metrics["n_pca_components"] = float(n_pca)
    metrics["pca_variance_explained"] = pca_var_explained
    train_pca = apply_pca(train_emb_no, pca)

    # When D7.a drops every test window, test_emb is empty. Run still completes:
    # train+val embeddings are exported, AE-only metrics already logged.
    test_split_empty = len(test_emb) == 0
    if test_split_empty:
        logger.warning(
            "TEST split is empty after D7.a target-row filter; "
            "skipping test-side PCA/cluster evaluation."
        )
        metrics["test_split_empty"] = 1.0
        test_emb_no = np.zeros((0, train_emb.shape[1]), dtype=train_emb.dtype)
        test_pca = np.zeros((0, n_pca), dtype=train_pca.dtype)
        test_no_idx = np.array([], dtype=int)
    else:
        test_no_idx = extract_non_overlapping_indices(
            n_windows=len(test_emb), window_size=W,
        )
        test_emb_no = test_emb[test_no_idx]
        test_pca = apply_pca(test_emb_no, pca)

    val_no_idx = extract_non_overlapping_indices(
        n_windows=len(val_emb), window_size=W,
    )
    val_emb_no = val_emb[val_no_idx]
    val_pca = apply_pca(val_emb_no, pca)

    test_labels: np.ndarray | None = None
    if config.run_clustering:
        k_selection = select_k(train_pca, k_range=[3, 4, 5])
        metrics["best_k"] = float(k_selection["best_k"])
        for k_val, sil_val in k_selection["scores"].items():
            metrics[f"train_silhouette_K{k_val}"] = sil_val

        km = fit_kmeans(train_pca, config.n_clusters, random_state=config.seed)
        test_labels = km.predict(test_pca)
        for name, val in compute_clustering_metrics(test_pca, test_labels).items():
            metrics[f"test_{name}"] = val
        val_labels = km.predict(val_pca)
        for name, val in compute_clustering_metrics(val_pca, val_labels).items():
            metrics[f"val_{name}"] = val
        metrics["test_regime_transitions"] = float(compute_regime_transitions(test_labels))

        # Combined K selection (Silhouette + BIC/GMM); pre_projeto §4.4.
        if len(train_pca) >= 10:
            k_combined = select_k_combined(
                train_pca, k_range=[3, 4, 5], random_state=config.seed,
            )
            metrics["best_k_combined"] = float(k_combined["best_k"])
            for k_val, v in k_combined["combined"].items():
                metrics[f"k_combined_score_K{k_val}"] = float(v)
    else:
        logger.info("Skipping KMeans/K-selection block (run_clustering=False).")

    # Principal pipeline: UMAP -> HDBSCAN -> NBER/Bai-Perron -> explanations
    # (pre_projeto §4.3 Modules 2-4 + §4.4 validation layers 2-3).
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

        try:
            import hdbscan as _hdbscan
            hdb_test_labels, hdb_test_probs = _hdbscan.approximate_predict(
                hdb_best.clusterer, test_umap,
            )
            hdb_val_labels, _ = _hdbscan.approximate_predict(
                hdb_best.clusterer, val_umap,
            )
        except Exception:  # pragma: no cover
            from tcc_itransformer.evaluation.density_clustering import fit_hdbscan
            _refit = fit_hdbscan(
                test_umap, min_cluster_size=hdb_best.min_cluster_size,
                min_samples=hdb_best.min_samples,
            )
            hdb_test_labels = _refit.labels
            hdb_test_probs = _refit.probabilities
            _refit_val = fit_hdbscan(
                val_umap, min_cluster_size=hdb_best.min_cluster_size,
                min_samples=hdb_best.min_samples,
            )
            hdb_val_labels = _refit_val.labels

        n_test_clusters = int(
            len(set(hdb_test_labels)) - (1 if -1 in hdb_test_labels else 0)
        )
        metrics["hdbscan_test_n_clusters"] = float(n_test_clusters)
        metrics["hdbscan_test_noise_fraction"] = float(np.mean(hdb_test_labels == -1))

        stride_test = W
        test_dates_all = test_df.index
        test_window_dates = pd.DatetimeIndex(
            [
                test_dates_all[i * stride_test + W - 1]
                for i in range(len(test_emb_no))
                if i * stride_test + W - 1 < len(test_dates_all)
            ]
        )
        m = len(test_window_dates)
        hdb_test_labels = np.asarray(hdb_test_labels[:m])
        hdb_test_probs = np.asarray(hdb_test_probs[:m])

        # Layer 2 - NBER overlap (Hungarian on VAL, frozen on TEST).
        # Q5 Tier 1 fix: legacy nber_overlap selected the cluster with maximum
        # F1 on TEST (post-hoc selection bias). We now fit cluster->regime on
        # VAL only and apply verbatim to TEST. Legacy max-F1 also reported,
        # tagged as biased, so the gap is visible in MLflow.
        try:
            usrec = load_usrec(config.nber_usrec_path)
            val_dates_all = val_df.index
            val_window_dates = pd.DatetimeIndex(
                [
                    val_dates_all[i * stride_test + W - 1]
                    for i in range(len(val_emb_no))
                    if i * stride_test + W - 1 < len(val_dates_all)
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

            legacy = nber_overlap(hdb_test_labels, test_window_dates, usrec, lead=0, lag=2)
            metrics["nber_f1_legacy_maxF1"] = legacy.f1
        except FileNotFoundError as exc:
            logger.warning("Skipping NBER overlap: %s", exc)
            mlflow.set_tag("nber_status", "snapshot_missing")

        # Layer 2 - Bai-Perron alignment on the first PC of the test panel.
        try:
            from sklearn.decomposition import PCA as _PCA

            test_panel_arr = test_scaled[
                W - 1 : W - 1 + m * stride_test : stride_test
            ]
            if len(test_panel_arr) >= 10:
                pc1 = _PCA(n_components=1).fit_transform(test_panel_arr).ravel()
                bp = bai_perron_alignment(
                    hdb_test_labels, pc1, penalty=10.0, tolerance=2,
                )
                metrics["bai_perron_f1"] = bp["f1"]
                metrics["bai_perron_n_breakpoints"] = float(bp["n_breakpoints"])
        except Exception as exc:  # pragma: no cover
            logger.warning("Bai-Perron alignment failed: %s", exc)

        crisis = crisis_window_coverage(hdb_test_labels, test_window_dates)
        for crisis_name, info in crisis.items():
            metrics[f"crisis_{crisis_name}_dominant_cluster"] = float(
                info.get("dominant_cluster", -1),
            )

        feat_cols = list(train_df.columns)
        per_window_panel = pd.DataFrame(
            test_windows[test_no_idx][:m].mean(axis=1),
            index=test_window_dates,
            columns=feat_cols,
        )
        moments = regime_conditional_moments(per_window_panel, hdb_test_labels)
        durations = regime_durations(hdb_test_labels)
        trans = transition_matrix(hdb_test_labels)
        metrics["n_regime_transitions_hdbscan"] = float(
            sum(1 for i in range(1, len(hdb_test_labels))
                if hdb_test_labels[i] != hdb_test_labels[i - 1])
        )
        if not durations.empty:
            metrics["mean_regime_duration_months"] = float(
                durations["mean_duration"].mean(),
            )

        explanations = explain_assignment(
            per_window_panel, hdb_test_labels,
            probabilities=hdb_test_probs,
            top_k=config.explain_top_k,
            membership_source="soft",
        )
        explain_df = explanations_to_frame(explanations)

        artifacts.update({
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

    if config.run_clustering and test_labels is not None:
        stability = clustering_stability(
            train_pca, config.n_clusters, n_runs=5, random_state=config.seed,
        )
        metrics["clustering_stability_ari"] = stability

        # W=24 has too few non-overlapping test windows for inferential claims.
        if config.window_size == 24:
            mlflow.set_tag("analysis_type", "exploratory")
            mlflow.set_tag("power_warning", "W=24: n_eff too low for inference")

        if len(test_pca) >= 3 and len(np.unique(test_labels)) >= 2:
            kw_results = kruskal_wallis_per_dim(test_pca, test_labels)
            metrics["kw_n_significant"] = float(kw_results["n_significant"])
            metrics["kw_mean_effect_size"] = float(np.mean(kw_results["effect_sizes"]))

            mw_results = pairwise_mann_whitney(test_pca, test_labels)
            metrics["mw_mean_effect_size"] = float(
                np.mean(np.abs(mw_results["effect_sizes"])),
            )

        baseline_results = run_all_baselines(
            train_windows=train_windows,
            eval_windows=test_windows[test_no_idx] if test_windows.ndim == 3 else test_windows,
            n_components=n_pca,
            k=config.n_clusters,
            random_state=config.seed,
        )
        for bname, bresult in baseline_results.items():
            metrics[f"baseline_{bname}_silhouette"] = bresult["silhouette"]

        # Primary headline test: iTransformer vs raw PCA.
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

        if n_eff_train >= 20 and len(train_pca) >= 3:
            from sklearn.metrics import silhouette_score

            def _sil_fn(data: np.ndarray) -> float:
                _km = fit_kmeans(data, config.n_clusters, random_state=config.seed)
                _labels = _km.predict(data)
                if len(np.unique(_labels)) < 2:
                    return 0.0
                return float(silhouette_score(data, _labels))

            boot_result = moving_block_bootstrap(
                _sil_fn, train_pca,
                block_length=max(1, W // 2),
                n_bootstrap=5000, random_state=config.seed,
            )
            metrics["bootstrap_silhouette_ci_lower"] = boot_result["ci_lower"]
            metrics["bootstrap_silhouette_ci_upper"] = boot_result["ci_upper"]

    return metrics, history, artifacts


def _persist_principal_artifacts(artifacts: dict, aux_dir: Path) -> None:
    """Write embeddings, moments, durations, transitions, explanations to parquet/json."""
    if not artifacts:
        return
    try:
        emb_dir = aux_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            entry = artifacts.get(f"embeddings_{split}")
            if entry is None:
                continue
            Z, dates = entry
            d = Z.shape[1]
            df = pd.DataFrame(
                Z.astype("float32"),
                columns=[f"z_{i}" for i in range(d)],
            )
            df.insert(0, "date", pd.DatetimeIndex(dates))
            df.to_parquet(emb_dir / f"Z_{split}.parquet", index=False)

        for key, fname in [
            ("moments", "moments.parquet"),
            ("durations", "durations.parquet"),
            ("transition_matrix", "transition_matrix.parquet"),
            ("explanations", "explanations.parquet"),
        ]:
            obj = artifacts.get(key)
            if obj is not None:
                obj.to_parquet(aux_dir / fname)

        grid = artifacts.get("hdb_grid_log")
        if grid is not None:
            (aux_dir / "hdbscan_grid.json").write_text(
                json.dumps(grid, indent=2, default=str),
            )

        labels = artifacts.get("test_labels")
        dates = artifacts.get("test_dates")
        probs = artifacts.get("test_probs")
        if labels is not None and dates is not None:
            df = pd.DataFrame({"date": list(dates), "label": list(labels)})
            if probs is not None:
                df["probability"] = list(probs)
            df.to_parquet(aux_dir / "test_labels.parquet")
    except Exception:  # pragma: no cover
        logger.exception("Failed to persist some principal artifacts")


def run_full_pipeline(
    config: ExperimentConfig,
    model_dir: Path | str | None = None,
    aux_dir: Path | str | None = None,
) -> dict:
    """Train + evaluate + persist artifacts. Used by SageMaker entrypoint and CLI."""
    model_dir = Path(model_dir) if model_dir else Path(config.results_dir) / "model"
    aux_dir = Path(aux_dir) if aux_dir else Path(config.results_dir) / "aux"
    model_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    metrics, history, artifacts = run_experiment(config)

    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str),
    )
    (aux_dir / "history.json").write_text(
        json.dumps(history, indent=2, default=str),
    )
    config.to_yaml(model_dir / "config.yaml")
    _persist_principal_artifacts(artifacts, aux_dir)

    try:
        if mlflow.active_run() is not None:
            for fp in [model_dir / "metrics.json"]:
                if fp.exists():
                    mlflow.log_artifact(str(fp))
            for fname in (
                "history.json", "explanations.parquet", "moments.parquet",
                "transition_matrix.parquet", "durations.parquet",
                "hdbscan_grid.json", "test_labels.parquet",
                "embeddings/Z_train.parquet", "embeddings/Z_val.parquet",
                "embeddings/Z_test.parquet",
            ):
                fp = aux_dir / fname
                if fp.exists():
                    mlflow.log_artifact(str(fp))
    except Exception:  # pragma: no cover
        logger.debug("MLflow log_artifact skipped", exc_info=True)

    logger.info(
        "Persisted metrics, history, and principal artifacts to %s / %s",
        model_dir, aux_dir,
    )
    return metrics


def run_single_with_mlflow(config: ExperimentConfig) -> dict:
    """CLI entry: open MLflow run, train, log per-epoch losses + final metrics."""
    tracking_uri = f"file:./{config.results_dir}/mlruns"
    experiment_id = setup_mlflow(tracking_uri, config.experiment_name)
    run_name = f"W{config.window_size}_d{config.latent_dim}_K{config.n_clusters}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        log_config(config)
        metrics, history, artifacts = run_experiment(config)
        for epoch, (tl, vl) in enumerate(
            zip(history["train_losses"], history["val_losses"]),
        ):
            log_epoch_metrics(epoch, tl, vl)
        log_evaluation_metrics(metrics)

        aux_dir = Path(config.results_dir) / "aux" / run_name
        aux_dir.mkdir(parents=True, exist_ok=True)
        _persist_principal_artifacts(artifacts, aux_dir)
        for fname in (
            "explanations.parquet", "moments.parquet", "transition_matrix.parquet",
            "durations.parquet", "hdbscan_grid.json", "test_labels.parquet",
        ):
            fp = aux_dir / fname
            if fp.exists():
                mlflow.log_artifact(str(fp))

    logger.info("Experiment complete. Run: %s", run_name)
    return metrics
