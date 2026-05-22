#!/usr/bin/env python3
"""Phase E: Alternative Encoder Battery.

Runs every registered alternative encoder through the same clustering-ablation
downstream used for the iTransformer winner, then aggregates a
``results/phase_e_comparison.csv`` ranking all encoders by the 7-metric panel.

Encoders are partitioned into three tiers:
- **zero-shot**: MOMENT, MOIRAI  (no training needed)
- **trainable**: TS2Vec, PatchTST, TimesNet, TF-C  (self-supervised on training split)
- **classical**: HamiltonHMM, MS-VAR, BOCPD  (econometric baselines)

Usage
-----
    cd tcc_ai
    source ../tcc.env
    uv run python scripts/run_phase_e_encoders.py

    # run only a subset
    uv run python scripts/run_phase_e_encoders.py \\
        --encoders moment,ts2vec,hamilton_hmm

    # skip clustering permutation/bootstrap for speed (dev mode)
    uv run python scripts/run_phase_e_encoders.py --fast

Optional deps (install before first run):
    uv add --optional phase_e momentfm ts2vec "transformers>=4.40.0,<5.0.0" hmmlearn
    # MOIRAI (not on PyPI):
    #   uv run pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

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
from tcc_itransformer.encoders.alt import REGISTRY
from tcc_itransformer.pipelines.clustering_ablation import run_ablation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── default paths ─────────────────────────────────────────────────────────────
CONFIG = ROOT / "configs/sagemaker_ae_only_W6_d7_K4_b1.yaml"
USREC_CSV = ROOT / "data/snapshots/nber_usrec.csv"
PHASE_E_ROOT = ROOT / "results/phase_e"
OUTPUT_CSV = ROOT / "results/phase_e_comparison.csv"

# iTransformer winner results for cross-encoder comparison baseline.
ITR_ABLATION_CSV = ROOT / "results/clustering_ablation/W6_d7_K4_b1/summary.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (bypasses Splits flat format, returns 3D windows)
# ─────────────────────────────────────────────────────────────────────────────

def _load_3d_windows(
    cfg: dict,
) -> tuple[
    np.ndarray,  # train_windows [n, T, N]
    np.ndarray,  # val_windows   [n, T, N]
    np.ndarray,  # test_windows  [n, T, N]
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """Load + preprocess the FRED-MD panel, return un-flattened windows."""
    W = int(cfg["window_size"])

    if cfg.get("data_format") == "etl_v2_parquet":
        panel_df, _ = load_etl_v2_panel(cfg["data_path"], cfg.get("mask_path"))
        train_df, val_df, test_df = split_by_date(panel_df, cfg["train_end"], cfg["val_end"])
    else:
        data, tcodes = load_fred_md(cfg["data_path"])
        transformed = transform_panel(data, tcodes)
        cleaned, _ = drop_high_nan_series(transformed)
        filled = forward_fill_nans(cleaned)
        train_df, val_df, test_df = split_by_date(filled, cfg["train_end"], cfg["val_end"])

    scaler = fit_scaler(train_df)
    train_s, val_s, test_s = scale_splits(train_df, val_df, test_df, scaler)

    train_w = create_windows(train_s, W)   # [n, T, N]
    val_w = create_windows(val_s, W)
    test_w = create_windows(test_s, W)

    # Date index: last timestep in each window
    train_dates = pd.DatetimeIndex(train_df.index[W - 1 :])
    val_dates = pd.DatetimeIndex(val_df.index[W - 1 :])
    test_dates = pd.DatetimeIndex(test_df.index[W - 1 :])

    return (
        train_w.astype(np.float32),
        val_w.astype(np.float32),
        test_w.astype(np.float32),
        train_dates,
        val_dates,
        test_dates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_encoder(
    name: str,
    cfg: dict,
    train_w: np.ndarray,
    val_w: np.ndarray,
    test_w: np.ndarray,
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    n_clusters: int,
    fast: bool,
    seed: int,
) -> pd.DataFrame | None:
    """Train (if needed), encode, save parquets, run ablation for one encoder."""
    EncoderCls = REGISTRY[name]

    if not EncoderCls.is_available():
        logger.warning("[%s] skipped — dependencies not installed.", name)
        return None

    logger.info("=" * 60)
    logger.info("ENCODER: %s (tier=%s)", name, EncoderCls.tier)
    logger.info("=" * 60)

    enc = EncoderCls()
    emb_dir = PHASE_E_ROOT / name / "embeddings"
    abl_dir = PHASE_E_ROOT / name / "ablation"

    t0 = time.perf_counter()
    try:
        enc.fit(train_w, seed=seed)
        Z_tr = enc.encode(train_w)
        Z_va = enc.encode(val_w)
        Z_te = enc.encode(test_w)
    except Exception:
        logger.exception("[%s] encode failed — skipping.", name)
        return None

    fit_secs = time.perf_counter() - t0
    logger.info("[%s] encode done in %.1fs  Z_test shape=%s", name, fit_secs, Z_te.shape)

    # Persist embeddings
    enc.save_split_parquets(Z_tr, Z_va, Z_te, train_dates, val_dates, test_dates, emb_dir)

    # Run clustering ablation
    n_perm = 100 if fast else 1000
    n_boot = 100 if fast else 1000

    try:
        summary_df = run_ablation(
            embeddings_dir=emb_dir,
            output_dir=abl_dir,
            n_clusters=n_clusters,
            seed=seed,
            usrec_csv=USREC_CSV if USREC_CSV.exists() else None,
            n_perm=n_perm,
            n_boot=n_boot,
        )
    except Exception:
        logger.exception("[%s] run_ablation failed — skipping.", name)
        return None

    summary_df["encoder"] = name
    summary_df["encoder_tier"] = EncoderCls.tier
    summary_df["d_out"] = EncoderCls.d_out
    summary_df["fit_wall_secs"] = fit_secs

    abl_total = time.perf_counter() - t0
    logger.info("[%s] ablation done in %.1fs  cells=%d", name, abl_total, len(summary_df))
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E: alternative encoder battery")
    parser.add_argument(
        "--encoders",
        type=str,
        default=",".join(REGISTRY.keys()),
        help="Comma-separated list of encoder names to run (default: all registered).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
        help="Path to experiment config YAML (default: sagemaker_ae_only_W6_d7_K4_b1.yaml).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce permutation/bootstrap reps to 100 for quick iteration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested = [e.strip() for e in args.encoders.split(",") if e.strip()]
    unknown = [e for e in requested if e not in REGISTRY]
    if unknown:
        logger.error("Unknown encoders: %s.  Available: %s", unknown, list(REGISTRY.keys()))
        sys.exit(1)

    cfg_path = args.config
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    n_clusters = int(cfg.get("n_clusters", 4))

    logger.info("Loading data (config=%s)...", cfg_path.name)
    train_w, val_w, test_w, train_dates, val_dates, test_dates = _load_3d_windows(cfg)
    logger.info(
        "Data loaded: train=%s  val=%s  test=%s  T=%d  N=%d",
        train_w.shape[0], val_w.shape[0], test_w.shape[0],
        train_w.shape[1], train_w.shape[2],
    )

    PHASE_E_ROOT.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for name in requested:
        df = _run_encoder(
            name=name,
            cfg=cfg,
            train_w=train_w,
            val_w=val_w,
            test_w=test_w,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            n_clusters=n_clusters,
            fast=args.fast,
            seed=args.seed,
        )
        if df is not None:
            all_frames.append(df)
        else:
            skipped.append(name)

    if not all_frames:
        logger.error("No encoders produced results. Exiting.")
        sys.exit(1)

    # Append iTransformer baseline rows from existing ablation CSV if present.
    if ITR_ABLATION_CSV.exists():
        itr_df = pd.read_csv(ITR_ABLATION_CSV)
        itr_df["encoder"] = "itransformer_b1"
        itr_df["encoder_tier"] = "foundation"
        itr_df["d_out"] = 7  # winner d=7
        itr_df["fit_wall_secs"] = float("nan")
        all_frames.append(itr_df)
        logger.info("Appended iTransformer B1 baseline (%d rows).", len(itr_df))
    else:
        logger.warning("iTransformer ablation CSV not found: %s", ITR_ABLATION_CSV)

    combined = pd.concat(all_frames, ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote %d rows → %s", len(combined), OUTPUT_CSV)

    # ── Summary table ──────────────────────────────────────────────────────
    summary_cols = [
        "encoder", "encoder_tier", "d_out", "cell",
        "nber_f1", "bai_perron_f1", "crisis_window_coverage",
        "dbcv", "noise_fraction_test",
    ]
    available = [c for c in summary_cols if c in combined.columns]
    summary = (
        combined[available]
        .sort_values("nber_f1", ascending=False, na_position="last")
        .head(30)
    )
    logger.info("\n%s", summary.to_string(index=False))

    if skipped:
        logger.warning("Skipped (deps missing or failed): %s", skipped)

    logger.info("Phase E complete. Results: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
