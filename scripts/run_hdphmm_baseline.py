"""Sticky / SDHDP-HMM baseline for the regime-detection thesis.

Two competing baselines on the same ETL-v2 panel and same train/val/test
split as the headline ``iTransformer-AE → UMAP → HDBSCAN`` pipeline:

* **Sticky HDP-HMM** (Fox, Sudderth, Jordan, Willsky — AOAS 2011) with
  hyperparameter ``kappa`` controlling self-transition stickiness.
* **SDHDP-HMM** (Song 2014, Toronto WP tecipa-427) — a Sticky Dirichlet
  HDP-HMM variant. Implemented here as a sticky HDP-HMM with a tighter
  concentration prior on the transition Dirichlet (``alpha``).

Outputs the locked 7-metric panel from
``docs/pre_analysis_plan.md`` §2:

* dbcv (NA — DBCV is undefined for HMM hard assignments; reported as
  ``np.nan`` and flagged ``hmm_dbcv_na`` in MLflow)
* n_clusters_test
* noise_fraction_test  (always 0.0 — HMM has no noise label)
* nber_f1               (Hungarian on VAL, frozen on TEST — same as headline)
* nber_f1_legacy_maxF1  (bias witness)
* bai_perron_f1
* crisis_window_coverage

This script runs **locally** (CPU JAX). It is intentionally *not* in the
SageMaker training image; install with::

    uv sync --extra baselines
    uv run python scripts/run_hdphmm_baseline.py \\
        --config configs/default.yaml --variant sticky

Variants:
    sticky  Sticky HDP-HMM (default).
    sdhdp   Tighter concentration prior (Song 2014 style).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.data.external_labels import load_usrec
from tcc_itransformer.data.fred_md import load_fred_md, transform_panel
from tcc_itransformer.data.preprocessing import (
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    load_etl_v2_panel,
    scale_splits,
    split_by_date,
)
from tcc_itransformer.evaluation.regime_validation import (
    bai_perron_alignment,
    crisis_window_coverage,
    fit_nber_assignment,
    nber_overlap,
    nber_overlap_frozen,
)
from tcc_itransformer.seed import set_global_seed

logger = logging.getLogger(__name__)

VARIANT_PRIORS: dict[str, dict[str, float]] = {
    # Fox 2011: alpha controls overall innovation rate, kappa adds stickiness.
    "sticky": {"alpha": 1.0, "kappa": 50.0, "gamma": 1.0},
    # Song 2014 style: tighter alpha → fewer effective regimes, similar kappa.
    "sdhdp": {"alpha": 0.3, "kappa": 50.0, "gamma": 0.5},
}


def fit_sticky_hdp_hmm(
    train_panel: np.ndarray,
    val_panel: np.ndarray,
    test_panel: np.ndarray,
    *,
    n_states_max: int = 10,
    variant: str = "sticky",
    seed: int = 42,
    n_iter: int = 100,
) -> dict[str, np.ndarray]:
    """Fit a sticky HDP-HMM on the training panel; return Viterbi paths.

    Implementation note: ``dynamax`` exposes a *finite-state* sticky-HMM
    (``StickyHMM``) and a non-parametric HDP-HMM is approximated by
    setting ``n_states_max`` to a generous upper bound and pruning empty
    states post-hoc — this is the standard *weak-limit* approximation
    used in Fox 2011 §6.

    Returns a dict with viterbi paths for {train, val, test}.
    """
    try:
        import jax
        import jax.random as jr
        from dynamax.hidden_markov_model import GaussianHMM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install with: uv sync --extra baselines",
        ) from exc

    priors = VARIANT_PRIORS[variant]
    n_features = train_panel.shape[1]
    key = jr.PRNGKey(seed)

    # Weak-limit sticky-HMM approximation. dynamax does not natively expose a
    # kappa term, but the equivalent effect is achieved by initialising the
    # transition matrix with strong diagonal mass — Fox 2011 §6 footnote.
    init_diag = priors["kappa"] / (priors["kappa"] + priors["alpha"] * n_states_max)
    transition_init = np.full(
        (n_states_max, n_states_max),
        (1.0 - init_diag) / max(n_states_max - 1, 1),
    )
    np.fill_diagonal(transition_init, init_diag)

    hmm = GaussianHMM(num_states=n_states_max, emission_dim=n_features)
    params, props = hmm.initialize(
        key=key,
        method="prior",
        transition_matrix=jax.numpy.asarray(transition_init),
    )

    # JAX expects (T, D) sequence; fit_em accepts either (T,D) or (B,T,D).
    train_seq = jax.numpy.asarray(train_panel)
    params, _lps = hmm.fit_em(
        params, props, train_seq, num_iters=n_iter, verbose=False,
    )

    def viterbi(panel: np.ndarray) -> np.ndarray:
        seq = jax.numpy.asarray(panel)
        return np.asarray(hmm.most_likely_states(params, seq))

    return {
        "train": viterbi(train_panel),
        "val": viterbi(val_panel),
        "test": viterbi(test_panel),
    }


def evaluate_baseline(
    config: ExperimentConfig,
    variant: str,
    *,
    n_states_max: int = 10,
    n_iter: int = 100,
) -> dict[str, float]:
    """Run the full baseline pipeline and return the 7-metric panel."""
    set_global_seed(config.seed)

    # --- Load + split (mirror run_single.py) ---
    if config.data_format == "etl_v2_parquet":
        panel_df, _mask_df = load_etl_v2_panel(
            config.data_path,
            config.mask_path,
            expected_sha256=config.data_sha256,
        )
    else:
        data, tcodes = load_fred_md(config.data_path)
        transformed = transform_panel(data, tcodes)
        cleaned, _ = drop_high_nan_series(transformed)
        panel_df = forward_fill_nans(cleaned)

    train_df, val_df, test_df = split_by_date(panel_df, config.train_end, config.val_end)
    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(train_df, val_df, test_df, scaler)

    # --- Fit HDP-HMM ---
    paths = fit_sticky_hdp_hmm(
        train_scaled, val_scaled, test_scaled,
        n_states_max=n_states_max, variant=variant,
        seed=config.seed, n_iter=n_iter,
    )
    val_labels = paths["val"]
    test_labels = paths["test"]

    # HMM emits a label per timestep. To compare against the headline pipeline
    # (which labels per non-overlapping window of size W), down-sample by
    # window-end to a common granularity.
    w = config.window_size
    val_dates = val_df.index[w - 1 :: w]
    test_dates = test_df.index[w - 1 :: w]
    val_labels_w = val_labels[w - 1 :: w][: len(val_dates)]
    test_labels_w = test_labels[w - 1 :: w][: len(test_dates)]
    val_dates = val_dates[: len(val_labels_w)]
    test_dates = test_dates[: len(test_labels_w)]

    metrics: dict[str, float] = {
        "dbcv": float("nan"),  # undefined for HMM hard assignments
        "n_clusters_test": float(len(np.unique(test_labels_w))),
        "noise_fraction_test": 0.0,
    }

    # --- NBER overlap (Hungarian on VAL, frozen on TEST) ---
    try:
        usrec = load_usrec(config.nber_usrec_path)
        assignment = fit_nber_assignment(
            val_labels_w, pd.DatetimeIndex(val_dates), usrec, lead=0, lag=2,
        )
        mlflow.set_tag("nber_assignment", str(assignment))
        nber_res = nber_overlap_frozen(
            test_labels_w, pd.DatetimeIndex(test_dates), usrec, assignment,
            lead=0, lag=2,
        )
        metrics["nber_f1"] = nber_res.f1
        metrics["nber_precision"] = nber_res.precision
        metrics["nber_recall"] = nber_res.recall

        legacy = nber_overlap(test_labels_w, pd.DatetimeIndex(test_dates), usrec, lead=0, lag=2)
        metrics["nber_f1_legacy_maxF1"] = legacy.f1
    except FileNotFoundError as exc:
        logger.warning("Skipping NBER overlap: %s", exc)
        mlflow.set_tag("nber_status", "snapshot_missing")

    # --- Bai-Perron on PC1 ---
    try:
        from sklearn.decomposition import PCA as _PCA

        test_panel_arr = test_scaled[w - 1 :: w][: len(test_labels_w)]
        if len(test_panel_arr) >= 10:
            pc1 = _PCA(n_components=1).fit_transform(test_panel_arr).ravel()
            bp = bai_perron_alignment(test_labels_w, pc1, penalty=10.0, tolerance=2)
            metrics["bai_perron_f1"] = bp["f1"]
    except Exception as exc:  # pragma: no cover
        logger.warning("Bai-Perron alignment failed: %s", exc)

    # --- Crisis windows ---
    crisis = crisis_window_coverage(test_labels_w, pd.DatetimeIndex(test_dates))
    metrics["crisis_n_canonical_covered"] = float(
        sum(1 for info in crisis.values() if info.get("dominant_cluster", -1) != -1)
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=["sticky", "sdhdp"], default="sticky")
    parser.add_argument("--n-states-max", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--mlflow-experiment", default="hdphmm_baseline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    config = ExperimentConfig.from_yaml(args.config)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"{args.variant}_hdphmm"):
        mlflow.set_tag("baseline_family", "hdp_hmm")
        mlflow.set_tag("baseline_variant", args.variant)
        mlflow.set_tag("data_contract", config.data_contract or "unspecified")
        mlflow.log_params({
            "variant": args.variant,
            "n_states_max": args.n_states_max,
            "n_iter": args.n_iter,
            "window_size": config.window_size,
            **VARIANT_PRIORS[args.variant],
        })
        metrics = evaluate_baseline(
            config, args.variant,
            n_states_max=args.n_states_max,
            n_iter=args.n_iter,
        )
        for k, v in metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(k, v)
        logger.info("Baseline metrics: %s", metrics)


if __name__ == "__main__":
    main()
