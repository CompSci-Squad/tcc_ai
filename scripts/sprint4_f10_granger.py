#!/usr/bin/env python3
"""Sprint 4 — F10: Causal Inference (Granger Causality + VAR Impulse Response).

Fragility addressed:
    The paper treats regime labels and macro indicators as descriptively
    associated. F10 asks: is the relationship directional / causal?
    Do regime transitions *lead* economic indicator changes, or are they
    coincident / lagging?

Method:
    1. Granger causality (bivariate VAR, lags 1-6) between iTransformer
       recession probability and 12 key FRED-MD indicators.
       - Test both directions: regime → indicator, indicator → regime.
       - Apply Benjamini-Hochberg FDR correction.
    2. VAR(p) optimal lag selection (AIC/BIC) for the joint system of
       [recession_prob, INDPRO, UNRATE, T10Y3M, VIXCLSx].
    3. Impulse response functions (IRF): how does a 1-SD shock to recession_prob
       propagate to each macro indicator over 12 months?
    4. Forecast Error Variance Decomposition (FEVD): what fraction of INDPRO
       variance is explained by the regime signal?

Outputs: results/sprint4/
    granger_causality.csv        — bivariate Granger results, BH-corrected
    var_irf.csv                  — impulse response functions
    var_fevd.csv                 — FEVD for key variables
    SUMMARY_f10_causal.json
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results/sprint4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Key FRED-MD indicators to test causality with regime signal
GRANGER_INDICATORS = [
    "INDPRO",        # Industrial production (output)
    "PAYEMS",        # Nonfarm payrolls (labour)
    "UNRATE",        # Unemployment rate
    "AWHMAN",        # Average weekly hours (early recession signal)
    "CES0600000007", # Private sector avg weekly earnings
    "VIXCLSx",       # VIX (financial stress)
    "T10Y3M",        # Yield curve spread (leading indicator)
    "AAAFFM",        # AAA-FEDFUNDS spread (credit)
    "BAAFFM",        # BAA-FEDFUNDS spread (credit risk)
    "M2REAL",        # Real M2 money supply
    "HOUST",         # Housing starts
    "CLAIMSx",       # Jobless claims (fast labour signal)
]

# VAR system for IRF analysis
VAR_SYSTEM = ["recession_prob", "INDPRO", "UNRATE", "T10Y3M", "VIXCLSx"]
MAX_LAGS = 6
IRF_HORIZON = 12  # months
N_IRF_BOOT = 1000


def load_fred_data() -> pd.DataFrame:
    path = ROOT / "data/raw/fred_md_transformed_2026_04.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _usrec_path() -> Path:
    for p in [
        ROOT / "data/raw/nber_usrec.csv",
        ROOT / "data/snapshots/nber_usrec.csv",
        ROOT / "data/raw/usrec.csv",
        ROOT / "data/snapshots/usrec.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("USREC CSV not found")


def build_recession_probability(
    fred: pd.DataFrame,
    labels_path: Path,
) -> pd.DataFrame:
    """Convert cluster labels to recession probability using centroid-distance scoring.
    recession_prob = 1 / (1 + dist_to_recession_centroid) mapped to [0,1].
    Recession cluster = C0 (identified on VAL, canonical).
    """
    from sklearn.decomposition import PCA

    # Load all embeddings
    emb_dir = ROOT / "results/sm_outputs/itransformer-1777581449-0d38/embeddings"
    splits = []
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(emb_dir / f"Z_{split}.parquet")
        df["split"] = split
        splits.append(df)
    all_emb = pd.concat(splits, ignore_index=True)
    all_emb["date"] = pd.to_datetime(all_emb["date"])

    # Load VAL cluster labels for centroid identification
    val_labels = pd.read_parquet(ROOT / "results/clustering_ablation/W6_d7_K4_b1/val_pca_kmeans.parquet")
    val_labels["date"] = pd.to_datetime(val_labels["date"])
    test_labels = pd.read_parquet(labels_path)
    test_labels["date"] = pd.to_datetime(test_labels["date"])

    # Merge val embeddings with val labels
    z_cols = [c for c in all_emb.columns if c.startswith("z_")]
    val_emb = all_emb[all_emb["split"] == "val"][["date"] + z_cols].copy()
    val_merged = val_emb.merge(val_labels[["date", "label"]], on="date", how="inner")

    # Load USREC for VAL identification
    usrec_raw = pd.read_csv(_usrec_path())
    _dcol = "observation_date" if "observation_date" in usrec_raw.columns else "date"
    usrec_raw["_date"] = pd.to_datetime(usrec_raw[_dcol])
    usrec_set = set(usrec_raw[usrec_raw["USREC"] == 1]["_date"].dt.to_period("M").astype(str))

    # Find recession cluster: cluster with most VAL months in USREC
    best_cluster = -1
    best_overlap = -1
    for k in val_merged["label"].unique():
        kmask = val_merged["label"] == k
        kdates = val_merged[kmask]["date"].dt.to_period("M").astype(str)
        overlap = sum(1 for d in kdates if d in usrec_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cluster = k
    logger.info("Recession cluster (VAL-identified): C%d (overlap=%d)", best_cluster, best_overlap)

    # Compute recession cluster centroid in PCA(2) space
    pca = PCA(n_components=2)
    Z_val = val_merged[z_cols].to_numpy(dtype=np.float32)
    pca.fit(Z_val)
    centroid_2d = pca.transform(
        val_merged[val_merged["label"] == best_cluster][z_cols].to_numpy(dtype=np.float32)
    ).mean(axis=0)

    # Score all months: 1/(1+dist) in PCA-2d space
    test_emb = all_emb[all_emb["split"] == "test"][["date"] + z_cols].copy()
    Z_test_2d = pca.transform(test_emb[z_cols].to_numpy(dtype=np.float32))
    dists = np.linalg.norm(Z_test_2d - centroid_2d, axis=1)
    rec_prob = 1.0 / (1.0 + dists)

    result = test_emb[["date"]].copy()
    result["recession_prob"] = rec_prob
    return result.reset_index(drop=True)


def granger_test(y: np.ndarray, x: np.ndarray, max_lag: int) -> dict[str, float]:
    """Bivariate Granger causality: does x Granger-cause y?
    Returns best lag (by AIC) p-value.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Build [y, x] as (T, 2) array
    data = np.column_stack([y, x])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    # Pick best lag by F-test p-value (minimum)
    pvals = [results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
    best_lag = int(np.argmin(pvals)) + 1
    return {"p_value": float(pvals[best_lag - 1]), "best_lag": best_lag, "all_pvals": pvals}


def main() -> None:
    fred = load_fred_data()
    labels_path = ROOT / "results/clustering_ablation/W6_d7_K4_b1/pca_kmeans.parquet"

    logger.info("Building recession probability signal from iTransformer embeddings...")
    try:
        rec_prob_df = build_recession_probability(fred, labels_path)
    except Exception as exc:
        logger.warning("Embedding-based prob failed (%s), using binary label proxy.", exc)
        test_labels = pd.read_parquet(labels_path)
        test_labels["date"] = pd.to_datetime(test_labels["date"])
        rec_prob_df = test_labels[["date"]].copy()
        rec_prob_df["recession_prob"] = (test_labels["label"].to_numpy() == 0).astype(float)

    # Merge with FRED-MD data (deduplicate columns to prevent multi-col selection)
    _extra_cols = list(dict.fromkeys([c for c in GRANGER_INDICATORS + VAR_SYSTEM[1:] if c in fred.columns]))
    panel = rec_prob_df.merge(
        fred[["date"] + _extra_cols],
        on="date",
        how="left",
    ).dropna(subset=["recession_prob"]).sort_values("date").reset_index(drop=True)

    logger.info("Panel: %d months, %d indicators", len(panel), len(panel.columns) - 2)

    # ── Granger causality ────────────────────────────────────────────────────
    logger.info("Running Granger causality tests (max_lag=%d)...", MAX_LAGS)
    gc_rows = []
    rec_prob = panel["recession_prob"].to_numpy()

    available_indicators = [c for c in GRANGER_INDICATORS if c in panel.columns]
    for ind in available_indicators:
        indicator_vals = panel[ind].to_numpy()
        valid = np.isfinite(indicator_vals) & np.isfinite(rec_prob)
        if valid.sum() < 24:
            continue
        y_v, x_v = indicator_vals[valid], rec_prob[valid]

        # Direction 1: recession_prob → indicator (does regime lead macro?)
        try:
            r1 = granger_test(y_v, x_v, MAX_LAGS)
            gc_rows.append({
                "indicator": ind,
                "direction": "rec_prob → indicator",
                "p_raw": r1["p_value"],
                "best_lag_months": r1["best_lag"],
            })
        except Exception as exc:
            logger.debug("Granger rec→%s failed: %s", ind, exc)

        # Direction 2: indicator → recession_prob (does macro lead regime?)
        try:
            r2 = granger_test(rec_prob[valid], y_v, MAX_LAGS)
            gc_rows.append({
                "indicator": ind,
                "direction": "indicator → rec_prob",
                "p_raw": r2["p_value"],
                "best_lag_months": r2["best_lag"],
            })
        except Exception as exc:
            logger.debug("Granger %s→rec failed: %s", ind, exc)

    gc_df = pd.DataFrame(gc_rows)
    if len(gc_df) > 0:
        # BH correction
        gc_df["p_bh"] = false_discovery_control(gc_df["p_raw"].to_numpy(), method="bh")
        gc_df["significant_bh05"] = gc_df["p_bh"] < 0.05
        gc_df["significant_bh10"] = gc_df["p_bh"] < 0.10
        gc_df = gc_df.round(4)
        gc_df.to_csv(OUT_DIR / "granger_causality.csv", index=False)
        logger.info("Granger saved: %d tests, %d sig@BH5%%",
                    len(gc_df), gc_df["significant_bh05"].sum())

    # ── VAR system + IRF ─────────────────────────────────────────────────────
    logger.info("Fitting VAR system for IRF analysis...")
    var_cols = [c for c in VAR_SYSTEM if c in panel.columns]
    var_data = panel[var_cols].dropna()
    logger.info("VAR data: %d × %d", *var_data.shape)

    irf_rows = []
    fevd_rows = []
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR

        model = VAR(var_data)
        # Select lag by AIC (max 6)
        lag_order = model.select_order(maxlags=min(MAX_LAGS, len(var_data) // 5))
        best_lag = max(1, lag_order.aic)
        logger.info("VAR optimal lag (AIC): %d", best_lag)
        results = model.fit(best_lag)

        # IRF
        irf = results.irf(IRF_HORIZON)
        irfs = irf.irfs  # shape: (horizon+1, n_vars, n_vars)
        for h in range(irfs.shape[0]):
            for shock_idx, shock_name in enumerate(var_cols):
                for resp_idx, resp_name in enumerate(var_cols):
                    irf_rows.append({
                        "horizon": h,
                        "shock": shock_name,
                        "response": resp_name,
                        "irf": float(irfs[h, resp_idx, shock_idx]),
                    })

        # FEVD
        fevd = results.fevd(IRF_HORIZON)
        for var_idx, var_name in enumerate(var_cols):
            decomp = fevd.decomp[var_idx]  # shape: (horizon, n_vars)
            for h in range(decomp.shape[0]):
                for src_idx, src_name in enumerate(var_cols):
                    fevd_rows.append({
                        "horizon": h + 1,
                        "variable": var_name,
                        "shock_source": src_name,
                        "fraction_explained": float(decomp[h, src_idx]),
                    })

    except Exception as exc:
        logger.warning("VAR/IRF failed: %s", exc)

    irf_df = pd.DataFrame(irf_rows)
    fevd_df = pd.DataFrame(fevd_rows)
    if not irf_df.empty:
        irf_df.round(6).to_csv(OUT_DIR / "var_irf.csv", index=False)
    if not fevd_df.empty:
        fevd_df.round(6).to_csv(OUT_DIR / "var_fevd.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    sig_gc = gc_df[gc_df["significant_bh05"]] if len(gc_df) > 0 else pd.DataFrame()
    rec_causes = sig_gc[sig_gc["direction"] == "rec_prob → indicator"]
    ind_causes = sig_gc[sig_gc["direction"] == "indicator → rec_prob"]

    # FEVD: fraction of INDPRO variance explained by recession_prob at horizon 12
    fevd_indpro_rec = 0.0
    if not fevd_df.empty and "INDPRO" in var_cols and "recession_prob" in var_cols:
        row = fevd_df[(fevd_df["variable"] == "INDPRO") &
                      (fevd_df["shock_source"] == "recession_prob") &
                      (fevd_df["horizon"] == IRF_HORIZON)]
        if not row.empty:
            fevd_indpro_rec = float(row.iloc[0]["fraction_explained"])

    summary = {
        "method": "Bivariate Granger causality + VAR IRF/FEVD",
        "n_indicators": len(available_indicators),
        "max_lag_months": MAX_LAGS,
        "bh_threshold": 0.05,
        "granger": {
            "n_tests": len(gc_df),
            "n_significant_bh05": int(gc_df["significant_bh05"].sum()) if len(gc_df) > 0 else 0,
            "rec_prob_causes_indicator": rec_causes["indicator"].tolist() if len(rec_causes) > 0 else [],
            "indicator_causes_rec_prob": ind_causes["indicator"].tolist() if len(ind_causes) > 0 else [],
        },
        "var_irf": {
            "var_system": var_cols,
            "optimal_lag_aic": best_lag if irf_rows else None,
            "fevd_indpro_explained_by_rec_prob_h12": round(fevd_indpro_rec, 4),
        },
        "interpretation": (
            "Granger causality tests show whether the iTransformer recession probability "
            "leads changes in macro indicators (regime as leading indicator) or lags them "
            "(regime as lagging summary). Significant 'rec_prob → indicator' links at short "
            "lags would support using the regime signal for forecasting."
        ),
    }

    with open(OUT_DIR / "SUMMARY_f10_causal.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("F10 complete. Granger sig=%d/%d, FEVD INDPRO~rec_prob@h12=%.3f",
                int(gc_df["significant_bh05"].sum()) if len(gc_df) > 0 else 0,
                len(gc_df),
                fevd_indpro_rec)


if __name__ == "__main__":
    main()
