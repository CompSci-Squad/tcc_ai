"""Run all 4 baselines per (W, d, K) cell + locked 7-metric panel CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from tcc_itransformer.config import ExperimentConfig
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
from tcc_itransformer.evaluation.clustering import compute_clustering_metrics
from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.panel_metrics import (
    PANEL_COLUMNS,
    compute_panel_metrics,
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


def run_baselines(config_dir: Path) -> None:
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

    if ref_config.data_format == "etl_v2_parquet":
        panel_df, _mask = load_etl_v2_panel(
            ref_config.data_path, ref_config.mask_path,
            expected_sha256=ref_config.data_sha256,
        )
        train_df, val_df, test_df = split_by_date(
            panel_df, ref_config.train_end, ref_config.val_end,
        )
    else:
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

    tracking_uri = f"file:./{ref_config.results_dir}/mlruns"
    experiment_id = setup_mlflow(tracking_uri, "baselines")

    for config in configs:
        w, d, k = config.window_size, config.latent_dim, config.n_clusters

        train_windows = create_windows(train_scaled, w)
        val_windows = create_windows(val_scaled, w)
        test_windows = create_windows(test_scaled, w)

        test_no_idx = extract_non_overlapping_indices(
            n_windows=len(test_windows), window_size=w,
        )
        val_no_idx = extract_non_overlapping_indices(
            n_windows=len(val_windows), window_size=w,
        )
        n_eff_test = compute_effective_n(len(test_windows), w)

        # Map non-overlapping window indices back to TIMESTAMPS (window-end).
        test_dates_full = test_df.index[w - 1 :]
        val_dates_full = val_df.index[w - 1 :]
        test_dates_no = pd.DatetimeIndex(test_dates_full[test_no_idx])
        val_dates_no = pd.DatetimeIndex(val_dates_full[val_no_idx])

        run_name = f"baseline_W{w}_d{d}_K{k}"
        logger.info("Running baselines: %s (n_eff_test=%d)", run_name, n_eff_test)

        baseline_results = run_all_baselines(
            train_windows=train_windows,
            eval_windows=test_windows[test_no_idx],
            val_windows=val_windows[val_no_idx],
            n_components=d, k=k, random_state=config.seed,
        )

        usrec_csv = (
            Path(config.nber_usrec_path) if config.nber_usrec_path else None
        )
        panel_rows: list[dict] = []
        for bname, res in baseline_results.items():
            panel = compute_panel_metrics(
                val_labels=res.get("val_labels"),
                val_dates=val_dates_no if res.get("val_labels") is not None else None,
                test_labels=res["labels"],
                test_dates=test_dates_no,
                Y_test=res["embeddings"],
                test_signal=res["embeddings"],
                usrec_csv=usrec_csv,
                is_density_clusterer=False,
            )
            panel_rows.append({
                "baseline": bname,
                "window_size": w, "latent_dim": d, "n_clusters": k,
                "seed": config.seed,
                **{key: panel.get(key, float("nan")) for key in PANEL_COLUMNS},
                "nber_precision": panel.get("nber_precision", float("nan")),
                "nber_recall": panel.get("nber_recall", float("nan")),
                "nber_assignment": panel.get("nber_assignment", ""),
                "crisis_n_canonical_in_test": panel.get(
                    "crisis_n_canonical_in_test", float("nan"),
                ),
                "test_silhouette": res["silhouette"],
            })

        panel_out_dir = Path(config.results_dir) / "baselines"
        panel_out_dir.mkdir(parents=True, exist_ok=True)
        panel_csv = panel_out_dir / f"baselines_panel_{run_name}.csv"
        pd.DataFrame(panel_rows).to_csv(panel_csv, index=False)
        logger.info("Wrote baseline panel -> %s", panel_csv)

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_params({
                "window_size": w, "latent_dim": d,
                "n_clusters": k, "seed": config.seed,
            })
            if w == 24:
                mlflow.set_tag("analysis_type", "exploratory")
                mlflow.set_tag(
                    "power_warning", "W=24: n_eff too low for inference",
                )

            metrics: dict[str, float] = {"n_eff_test": float(n_eff_test)}
            for bname, result in baseline_results.items():
                metrics[f"{bname}_silhouette"] = result["silhouette"]
                bm = compute_clustering_metrics(result["embeddings"], result["labels"])
                for mname, mval in bm.items():
                    metrics[f"{bname}_{mname}"] = mval

            for prow in panel_rows:
                bname = prow["baseline"]
                for pcol in PANEL_COLUMNS:
                    val = prow.get(pcol)
                    if (
                        val is not None
                        and isinstance(val, (int, float))
                        and not (isinstance(val, float) and np.isnan(val))
                    ):
                        metrics[f"{bname}_{pcol}"] = float(val)

            names = list(baseline_results.keys())
            for i, na in enumerate(names):
                for nb in names[i + 1:]:
                    ra, rb = baseline_results[na], baseline_results[nb]
                    if len(ra["embeddings"]) >= 3 and len(rb["embeddings"]) >= 3:
                        perm = permutation_test_silhouette(
                            ra["embeddings"], ra["labels"],
                            rb["embeddings"], rb["labels"],
                            n_permutations=5000, random_state=config.seed,
                        )
                        metrics[f"perm_{na}_vs_{nb}_delta"] = perm["observed_diff"]
                        metrics[f"perm_{na}_vs_{nb}_p"] = perm["p_value"]

            log_evaluation_metrics(metrics)

        logger.info(
            "  %s",
            "  ".join(
                f"{name}={res['silhouette']:.4f}"
                for name, res in baseline_results.items()
            ),
        )

    logger.info("All baselines complete.")
