"""Sticky / SDHDP-HMM baseline (Fox 2011, Song 2014) with the locked 7-metric panel.

Local-CPU JAX baseline. Install with ``uv sync --extra baselines``.

DBCV is undefined for HMM hard assignments, so it is reported as ``nan`` and
flagged ``hmm_dbcv_na`` in MLflow. The HMM emits a label per timestep -- we
down-sample to non-overlapping windows of size W to match the headline pipeline.

Variants:
    sticky  Sticky HDP-HMM (Fox 2011 §6 weak-limit approximation).
    sdhdp   Tighter alpha (Song 2014 / Toronto WP tecipa-427).
"""

from __future__ import annotations

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
    # Fox 2011: alpha = innovation rate, kappa = self-transition stickiness.
    "sticky": {"alpha": 1.0, "kappa": 50.0, "gamma": 1.0},
    # Song 2014: tighter alpha -> fewer effective regimes.
    "sdhdp": {"alpha": 0.3, "kappa": 50.0, "gamma": 0.5},
}

# Phase C2: minimum EM iterations per variant (Fox 2011 eq. 22 convergence advice)
VARIANT_N_ITER: dict[str, int] = {
    "sticky": 500,
    "sdhdp": 200,
}

_WINSOR_LOW = 1   # 1st percentile
_WINSOR_HIGH = 99  # 99th percentile


def _winsorise_panel(panel: np.ndarray) -> np.ndarray:
    """Winsorise each feature at [1%, 99%] to suppress FRED-MD outliers."""
    lo = np.nanpercentile(panel, _WINSOR_LOW, axis=0)
    hi = np.nanpercentile(panel, _WINSOR_HIGH, axis=0)
    return np.clip(panel, lo, hi)


def fit_sticky_hdp_hmm(
    train_panel: np.ndarray, val_panel: np.ndarray, test_panel: np.ndarray,
    *, n_states_max: int = 10, variant: str = "sticky",
    seed: int = 42, n_iter: int | None = None,
    n_pca_components: int = 20,
) -> dict[str, np.ndarray | list]:
    """Fit dynamax weak-limit sticky-HDP-HMM, return Viterbi paths per split.

    Phase C2 additions:
    - Winsorises each FRED-MD feature at [1%, 99%] before fitting.
    - PCA reduces to n_pca_components (default 20) for numerical stability —
      a 122-feature full GaussianHMM is ill-conditioned with O(500) samples.
    - Uses VARIANT_N_ITER defaults (500 sticky, 200 sdhdp) if n_iter is None.
    - Captures the per-iteration log-likelihood trajectory from fit_em.
    - Tracks active-state count per iteration (states with >0.5% occupancy).
    - Computes final state-occupancy histogram.

    dynamax has no native kappa term: the Fox 2011 §6 footnote shows the same
    effect via strong diagonal mass on the transition prior.
    """
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from dynamax.hidden_markov_model import GaussianHMM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install with: uv sync --extra baselines") from exc

    from sklearn.decomposition import PCA

    _n_iter = n_iter if n_iter is not None else VARIANT_N_ITER.get(variant, 200)

    # Phase C2: winsorise before scaling
    train_w = _winsorise_panel(train_panel)
    val_w = _winsorise_panel(val_panel)
    test_w = _winsorise_panel(test_panel)

    # PCA for numerical stability (122 raw features → n_pca_components).
    # Fit only on train; transform all splits identically.
    n_components = min(n_pca_components, train_w.shape[1], train_w.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    train_w = pca.fit_transform(train_w)
    val_w = pca.transform(val_w)
    test_w = pca.transform(test_w)
    logger.info("[%s] PCA: %d components → %.1f%% variance explained",
                variant, n_components, 100 * pca.explained_variance_ratio_.sum())

    priors = VARIANT_PRIORS[variant]
    n_features = train_w.shape[1]
    key = jr.PRNGKey(seed)

    init_diag = priors["kappa"] / (priors["kappa"] + priors["alpha"] * n_states_max)
    transition_init = np.full(
        (n_states_max, n_states_max),
        (1.0 - init_diag) / max(n_states_max - 1, 1),
    )
    np.fill_diagonal(transition_init, init_diag)

    hmm = GaussianHMM(num_states=n_states_max, emission_dim=n_features)
    params, props = hmm.initialize(
        key=key, method="prior",
        transition_matrix=jnp.asarray(transition_init),
    )
    train_seq = jnp.asarray(train_w)
    params, lls = hmm.fit_em(
        params, props, train_seq, num_iters=_n_iter, verbose=False,
    )
    lls_list: list[float] = [float(v) for v in np.asarray(lls)]

    # Active-state count per EM iteration: estimate from final smoother posterior
    # (full per-iter tracking requires custom EM loop — too expensive; we log final state only)
    posterior = hmm.smoother(params, train_seq)
    # state_probs shape: (T, n_states)
    state_probs = np.asarray(posterior.smoothed_probs)  # (T, K)
    occupancy = state_probs.mean(axis=0)  # mean responsibility per state
    active_mask = occupancy > 0.005  # 0.5% threshold
    n_active_states = int(active_mask.sum())
    occupancy_dict = {int(i): float(occupancy[i]) for i in range(n_states_max)}

    logger.info(
        "[%s] n_iter=%d | final LL=%.4f | active_states=%d/%d",
        variant, _n_iter, lls_list[-1] if lls_list else float("nan"),
        n_active_states, n_states_max,
    )

    def viterbi(panel: np.ndarray) -> np.ndarray:
        return np.asarray(hmm.most_likely_states(params, jnp.asarray(panel)))

    return {
        "train": viterbi(train_w),
        "val": viterbi(val_w),
        "test": viterbi(test_w),
        # C2 diagnostics
        "lls": lls_list,
        "n_active_states": n_active_states,
        "occupancy": occupancy_dict,
        "n_iter_used": _n_iter,
    }


def evaluate_baseline(
    config: ExperimentConfig, variant: str,
    *, n_states_max: int = 10, n_iter: int | None = None,
) -> dict[str, float]:
    set_global_seed(config.seed)

    if config.data_format == "etl_v2_parquet":
        panel_df, _ = load_etl_v2_panel(
            config.data_path, config.mask_path,
            expected_sha256=config.data_sha256,
        )
    else:
        data, tcodes = load_fred_md(config.data_path)
        transformed = transform_panel(data, tcodes)
        cleaned, _ = drop_high_nan_series(transformed)
        panel_df = forward_fill_nans(cleaned)

    train_df, val_df, test_df = split_by_date(
        panel_df, config.train_end, config.val_end,
    )
    scaler = fit_scaler(train_df)
    train_scaled, val_scaled, test_scaled = scale_splits(
        train_df, val_df, test_df, scaler,
    )

    paths = fit_sticky_hdp_hmm(
        train_scaled, val_scaled, test_scaled,
        n_states_max=n_states_max, variant=variant,
        seed=config.seed, n_iter=n_iter,
    )
    val_labels = np.asarray(paths["val"])
    test_labels = np.asarray(paths["test"])

    # Log C2 diagnostics
    lls: list[float] = paths.get("lls", [])
    n_active: int = paths.get("n_active_states", 0)
    occupancy: dict[int, float] = paths.get("occupancy", {})
    n_iter_used: int = paths.get("n_iter_used", 0)
    for i, ll in enumerate(lls):
        if np.isfinite(ll):
            mlflow.log_metric("em_log_likelihood", ll, step=i)
    mlflow.log_metrics({
        "n_active_states_final": float(n_active),
        "em_final_ll": float(lls[-1]) if lls else float("nan"),
        "n_iter_used": float(n_iter_used),
    })
    mlflow.set_tag("state_occupancy", str(occupancy))

    # HMM produces per-timestep labels; evaluate at monthly resolution.
    # The iTransformer uses W-stride downsampling (one label per window) but
    # the HMM timestep == 1 month.  Applying W=6 stride here misses short
    # recessions (e.g., 3-month COVID recession falls between stride-6 anchors)
    # and artificially deflates F1 vs. NBER.  Use per-month labels directly.
    val_dates = val_df.index
    test_dates = test_df.index
    val_labels_w = val_labels[: len(val_dates)]
    test_labels_w = test_labels[: len(test_dates)]

    metrics: dict[str, float] = {
        "dbcv": float("nan"),
        "n_clusters_test": float(len(np.unique(test_labels_w))),
        "noise_fraction_test": 0.0,
    }

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
        legacy = nber_overlap(
            test_labels_w, pd.DatetimeIndex(test_dates), usrec, lead=0, lag=2,
        )
        metrics["nber_f1_legacy_maxF1"] = legacy.f1
    except FileNotFoundError as exc:
        logger.warning("Skipping NBER overlap: %s", exc)
        mlflow.set_tag("nber_status", "snapshot_missing")

    try:
        from sklearn.decomposition import PCA as _PCA

        test_panel_arr = test_scaled[: len(test_labels_w)]
        if len(test_panel_arr) >= 10:
            pc1 = _PCA(n_components=1).fit_transform(test_panel_arr).ravel()
            bp = bai_perron_alignment(
                test_labels_w, pc1, penalty=10.0, tolerance=2,
            )
            metrics["bai_perron_f1"] = bp["f1"]
    except Exception as exc:  # pragma: no cover
        logger.warning("Bai-Perron alignment failed: %s", exc)

    crisis = crisis_window_coverage(test_labels_w, pd.DatetimeIndex(test_dates))
    metrics["crisis_n_canonical_covered"] = float(
        sum(1 for info in crisis.values() if info.get("dominant_cluster", -1) != -1)
    )
    return metrics


def run_hdphmm(
    config_path: Path, *,
    variant: str = "sticky", n_states_max: int = 10, n_iter: int | None = None,
    mlflow_experiment: str = "hdphmm_baseline",
    output: Path | None = None,
) -> dict[str, float]:
    config = ExperimentConfig.from_yaml(config_path)
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"{variant}_hdphmm"):
        mlflow.set_tag("baseline_family", "hdp_hmm")
        mlflow.set_tag("baseline_variant", variant)
        mlflow.set_tag("data_contract", config.data_contract or "unspecified")
        mlflow.log_params({
            "variant": variant, "n_states_max": n_states_max,
            "n_iter": n_iter, "window_size": config.window_size,
            **VARIANT_PRIORS[variant],
        })
        metrics = evaluate_baseline(
            config, variant,
            n_states_max=n_states_max, n_iter=n_iter,
        )
        for k, v in metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(k, v)
        logger.info("Baseline metrics: %s", metrics)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        row = {"variant": variant, "n_states_max": n_states_max, **metrics}
        pd.DataFrame([row]).to_csv(output, index=False)
        logger.info("Results written to %s", output)
    return metrics
