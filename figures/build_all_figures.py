#!/usr/bin/env python3
"""
figures/build_all_figures.py
Paper ENIAC/BRACIS 2026 — build all 6 required figures with check-first protocol.

Run from tcc_ai/:
    uv run python figures/build_all_figures.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_HASH_FILE = FIG_DIR / ".canonical_hashes.json"

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

CLUSTER_COLORS = {
    "C0": "#C0392B",
    "C1": "#2980B9",
    "C2": "#27AE60",
    "C3": "#95A5A6",
}
TIER_COLORS = {
    "T1":           "#2C3E50",
    "T2":           "#E67E22",
    "T3":           "#27AE60",
    "iTransformer": "#C0392B",
}
DPI = 300


# ══════════════════════════════════════════════════════════════════════════════
# Check-first helpers
# ══════════════════════════════════════════════════════════════════════════════

def _data_hash(canonical_data: dict) -> str:
    return hashlib.md5(
        json.dumps(canonical_data, sort_keys=True).encode()
    ).hexdigest()


def check_figure(fig_path: str | Path, canonical_data: dict) -> str:
    fig_path = str(fig_path)
    if not os.path.exists(fig_path):
        print(f"[REBUILD] {fig_path} — does not exist")
        return "REBUILD"
    dh = _data_hash(canonical_data)
    if CANONICAL_HASH_FILE.exists():
        saved = json.loads(CANONICAL_HASH_FILE.read_text())
        if saved.get(fig_path) == dh:
            print(f"[USE]     {fig_path} — data unchanged")
            return "USE"
    print(f"[REBUILD] {fig_path} — data changed or hash missing")
    return "REBUILD"


def save_hash(fig_path: str | Path, canonical_data: dict) -> None:
    fig_path = str(fig_path)
    hashes: dict = {}
    if CANONICAL_HASH_FILE.exists():
        hashes = json.loads(CANONICAL_HASH_FILE.read_text())
    hashes[fig_path] = _data_hash(canonical_data)
    CANONICAL_HASH_FILE.write_text(json.dumps(hashes, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# Canonical datasets
# ══════════════════════════════════════════════════════════════════════════════

ENCODER_DATA = [
    # (encoder, tier, MCC, MCC_lo, MCC_hi, silhouette, note)
    ("iTransformer",  "iTransformer", +0.564, +0.308, +0.777, 0.168,  ""),
    ("raw_pca",       "T1",           +0.361, -0.023, +0.704, 0.450,  ""),
    ("timesnet",      "T1",           +0.152, -0.037, +0.489, 0.329,  ""),
    ("moment",        "T3",           +0.141, -0.053, +0.345, 0.321,  ""),
    ("tfc",           "T2",           +0.216, +0.107, +0.314, 0.132,  ""),
    ("windowed_pca",  "T1",           +0.104, -0.066, +0.277, 0.501,  ""),
    ("svd",           "T1",           +0.104, -0.068, +0.275, 0.501,  ""),
    ("patchtst",      "T2",           +0.101, -0.068, +0.280, 0.342,  ""),
    ("hamilton_hmm",  "T1",           +0.056, -0.089, +0.208, 0.673,  ""),
    ("linear_ae",     "T1",           +0.000,  0.000,  0.000, 0.300,  ""),
    ("bocpd",         "T1",           -0.038, -0.061, -0.018, 0.965,  ""),
    ("ts2vec",        "T2",           -0.224, -0.324, -0.111, 0.255,  "anti-corr"),
    ("mlp_ae",        "T1",           +0.864,  0.000,  1.000, 0.200,  "anomalous†"),
]

TIMELINE_META = {
    "cluster_labels_file": str(
        ROOT / "results/clustering_ablation/W6_d7_K4_b1/cluster_labels_full.csv"
    ),
    "nber_recessions": [
        ("1969-12", "1970-11"), ("1973-11", "1975-03"),
        ("1980-01", "1980-07"), ("1981-07", "1982-11"),
        ("1990-07", "1991-03"), ("2001-03", "2001-11"),
        ("2007-12", "2009-06"), ("2020-02", "2020-04"),
    ],
    "split_dates": {"train_end": "1999-12", "val_end": "2009-12"},
    "annotated_crises": {
        "2001-03": "Dot-com",
        "2007-12": "GFC",
        "2020-02": "COVID-19",
    },
}

UMAP_META = {
    "embeddings_file": str(
        ROOT / "results/clustering_ablation/W6_d7_K4_b1/umap_embeddings_test.csv"
    ),
    "covid_dates": ["2020-03", "2020-04"],
    "nber_recession_test": ["2020-03", "2020-04"],
}

STABILITY_DATA = {
    "clusters":    ["C0\n(systemic stress)", "C1\n(housing exp.)",
                    "C2\n(labour rec.)", "C3\n(transition)"],
    "j_mean":      [0.640, 0.839, 0.734, 0.821],
    "j_ci_lo":     [0.410, 0.661, 0.550, 0.600],
    "j_ci_hi":     [0.778, 0.930, 0.895, 1.000],
    "pr_stable":   [75.1,  99.7,  95.5,  98.2],
    "threshold":   0.60,
    "B":           200,
    "subsample_pct": 80,
}

WINDOW_DATA = {
    "W":           [3,     6,     9,     12,    18,    24   ],
    "F1_tol":      [0.276, 0.519, 0.206, 0.163, 0.163, 0.163],
    "MCC":         [0.353, 0.381, None,  None,  None,  None ],
    "canonical_W": 6,
    "raw_pca_F1":  0.444,
}

MACRO_PROFILES = {
    "series": {
        "VIXCLSx":       [+1.95, -0.30, -0.74, +0.10],
        "CES0600000007": [-1.88, +0.84, +0.84, -0.05],
        "AWHMAN":        [-1.65, +0.60, +0.87, -0.10],
        "M2REAL":        [+1.36, -0.20, -0.30, +0.05],
        "TB6SMFFM":      [-0.40, +0.95, +0.20, -0.10],
        "PERMITS":       [-0.50, +0.93, +0.30, -0.28],
        "HOUST":         [-0.45, +0.93, +0.25, -0.28],
        "CLAIMSx":       [+0.80, -0.30, -0.40, +0.05],
        "BAAFFM":        [+0.70, -0.85, -0.20, +0.10],
        "T10YFFM":       [+0.50, -0.84, -0.15, +0.08],
        "PAYEMS":        [-0.60, +0.49, +0.49, -0.08],
        "TB3SMFFM":      [-0.20, +0.30, +0.15, -0.28],
    },
    "cluster_labels": [
        "C0\n(systemic stress)",
        "C1\n(housing expansion)",
        "C2\n(labour recovery)",
        "C3\n(transition)",
    ],
    "series_descriptions": {
        "VIXCLSx":       "VIX (implied vol.)",
        "CES0600000007": "Manuf. employment",
        "AWHMAN":        "Avg. weekly hours manuf.",
        "M2REAL":        "Real M2 money supply",
        "TB6SMFFM":      "6M T-bill spread",
        "PERMITS":       "Building permits",
        "HOUST":         "Housing starts",
        "CLAIMSx":       "Initial claims",
        "BAAFFM":        "BAA credit spread",
        "T10YFFM":       "10Y T-note spread",
        "PAYEMS":        "Total payrolls",
        "TB3SMFFM":      "3M T-bill spread",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Two-Axis Scatter: Geometric vs Economic Quality
# ══════════════════════════════════════════════════════════════════════════════

def build_two_axis(data: list, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    # Manual offsets to avoid label overlap
    label_offsets = {
        "iTransformer":  (-0.020, +0.035),
        "bocpd":         (+0.015, -0.055),
        "mlp_ae":        (+0.010, -0.060),
        "ts2vec":        (+0.010, -0.055),
        "hamilton_hmm":  (+0.010, -0.035),
        "raw_pca":       (+0.010, +0.020),
        "windowed_pca":  (-0.025, +0.022),
        "svd":           (+0.010, -0.040),
        "timesnet":      (+0.010, +0.020),
        "linear_ae":     (+0.010, +0.020),
        "tfc":           (+0.010, +0.020),
        "patchtst":      (+0.010, -0.035),
        "moment":        (+0.010, +0.020),
    }

    for enc, tier, mcc, lo, hi, sil, note in data:
        color  = TIER_COLORS.get(tier, TIER_COLORS["T1"])
        marker = ("^" if tier == "iTransformer" else
                  "s" if tier == "T2" else
                  "D" if tier == "T3" else "o")
        size  = 140 if tier == "iTransformer" else 60
        alpha = 0.45 if note == "anomalous†" else 0.9

        # CI error bars (skip degenerate [0,0] CIs)
        if lo != hi:
            ax.errorbar(sil, mcc, yerr=[[mcc - lo], [hi - mcc]],
                        fmt="none", color=color, alpha=0.35,
                        linewidth=0.8, capsize=2)

        ax.scatter(sil, mcc, c=color, marker=marker,
                   s=size, alpha=alpha, zorder=5)

        ox, oy = label_offsets.get(enc, (+0.008, +0.018))
        label_text = enc if not note else f"{enc}\n({note})"
        ax.annotate(
            label_text,
            xy=(sil, mcc),
            xytext=(sil + ox, mcc + oy),
            fontsize=7,
            ha="right" if ox < 0 else "left",
            color=color,
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--",
               alpha=0.5, label="MCC = 0 (no skill)")
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle=":",
               alpha=0.4)

    ax.set_xlabel("Test Silhouette Score (geometric quality)")
    ax.set_ylabel("MCC with 95% CI (economic alignment)")
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.55, 1.15)

    # Quadrant labels
    for (tx, ty, txt) in [
        (0.02, 0.95, "high MCC\nlow Sil."),
        (0.68, 0.95, "high MCC\nhigh Sil."),
        (0.68, 0.05, "low MCC\nhigh Sil."),
    ]:
        ax.text(tx, ty, txt, transform=ax.transAxes, fontsize=6.5,
                alpha=0.45, va="top" if ty > 0.5 else "bottom", color="gray")

    legend_elements = [
        Line2D([0], [0], marker="^", color=TIER_COLORS["iTransformer"],
               label="iTransformer (highlighted)", linestyle="None", markersize=9),
        Line2D([0], [0], marker="o", color=TIER_COLORS["T1"],
               label="Tier 1 — from scratch", linestyle="None", markersize=7),
        Line2D([0], [0], marker="s", color=TIER_COLORS["T2"],
               label="Tier 2 — domain-adapted", linestyle="None", markersize=7),
        Line2D([0], [0], marker="D", color=TIER_COLORS["T3"],
               label="Tier 3 — zero-shot", linestyle="None", markersize=7),
        Line2D([0], [0], color="black", linewidth=0.7, linestyle="--",
               label="MCC = 0"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Built: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cluster Timeline 1965-2026
# ══════════════════════════════════════════════════════════════════════════════

def _build_cluster_labels_full() -> pl.DataFrame:
    """
    Combine VAL+TEST labels from parquets.
    For TRAIN (pre-2000): reconstruct approximate labels from raw_pca
    Z_train embeddings using PCA(2)+KMeans(4), aligning C0 via NBER overlap.
    Saves cluster_labels_full.csv and returns the DataFrame.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    full_path = Path(TIMELINE_META["cluster_labels_file"])

    # ── VAL + TEST from canonical parquets ─────────────────────────────────
    val_p = ROOT / "results/clustering_ablation/W6_d7_K4_b1/val_pca_kmeans.parquet"
    tst_p = ROOT / "results/clustering_ablation/W6_d7_K4_b1/pca_kmeans.parquet"

    val_df = pl.read_parquet(val_p).with_columns(
        pl.col("date").dt.date().alias("date"),
        pl.lit("VAL").alias("split"),
    ).select(["date", "label", "split"])

    tst_df = pl.read_parquet(tst_p).with_columns(
        pl.col("date").dt.date().alias("date"),
        pl.lit("TEST").alias("split"),
    ).select(["date", "label", "split"])

    # ── NBER USREC ──────────────────────────────────────────────────────────
    usrec = pl.read_csv(ROOT / "data/snapshots/nber_usrec.csv")
    usrec = usrec.with_columns(
        pl.col("observation_date").str.to_date().alias("date")
    ).select(["date", "USREC"])

    # ── TRAIN from raw_pca Z_train ──────────────────────────────────────────
    z_path = ROOT / "results/phase_c_comparison/raw_pca/emb/Z_train.parquet"
    if not z_path.exists():
        # Fallback: iTransformer mlruns Z_train (long version, filter pre-2000)
        z_path = ROOT / "results/mlruns/580674338469141387/f6bdf863c3694eac86b9a5b0e6d1a53e/artifacts/Z_train.parquet"

    z_train = pl.read_parquet(z_path).with_columns(
        pl.col("date").dt.date().alias("date")
    )

    # Filter to TRAIN period (1965-01..1999-12) only
    from datetime import date
    train_mask = (
        (pl.col("date") >= pl.lit(date(1965, 1, 1))) &
        (pl.col("date") <= pl.lit(date(1999, 12, 1)))
    )
    z_train = z_train.filter(train_mask)

    if len(z_train) == 0:
        print("  WARNING: no TRAIN embeddings found for 1965-1999 — using VAL+TEST only")
        combined = pl.concat([val_df, tst_df]).sort("date")
        combined.write_csv(full_path)
        return combined

    # Embedding columns (skip date)
    feat_cols = [c for c in z_train.columns if c != "date"]
    X = z_train.select(feat_cols).to_numpy()

    # PCA(2) → KMeans(4)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_sc)
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    raw_labels = km.fit_predict(X_2d)

    # Align: identify recession cluster by max NBER overlap
    dates_train = z_train["date"].to_list()
    usrec_dict = dict(zip(usrec["date"].to_list(), usrec["USREC"].to_list()))
    nber_train = np.array([usrec_dict.get(d, 0) for d in dates_train])

    best_cluster, best_overlap = 0, -1
    for c in range(4):
        mask = raw_labels == c
        if mask.sum() == 0:
            continue
        overlap = nber_train[mask].mean()
        if overlap > best_overlap:
            best_overlap, best_cluster = overlap, c

    # Remap: recession_cluster → 0; others fill 1,2,3 in cluster-size order
    other_clusters = [c for c in range(4) if c != best_cluster]
    sizes = [(c, (raw_labels == c).sum()) for c in other_clusters]
    sizes.sort(key=lambda x: -x[1])
    remap = {best_cluster: 0}
    for new_c, (old_c, _) in enumerate(sizes, start=1):
        remap[old_c] = new_c

    aligned_labels = np.array([remap[l] for l in raw_labels])
    print(f"  TRAIN: best recession cluster={best_cluster} "
          f"(NBER overlap={best_overlap:.2%}); remapped as C0")

    train_df = pl.DataFrame({
        "date":  dates_train,
        "label": aligned_labels.tolist(),
        "split": ["TRAIN"] * len(dates_train),
    }).with_columns(pl.col("label").cast(pl.Int32))

    combined = pl.concat([train_df, val_df, tst_df]).sort("date")
    combined.write_csv(full_path)
    print(f"  Saved cluster_labels_full.csv ({len(combined)} rows)")
    return combined


def build_timeline(labels_df: pl.DataFrame, meta: dict, output_path: str | Path) -> None:
    dates_raw = labels_df["date"].to_list()
    clusters  = labels_df["label"].to_list()
    splits    = labels_df["split"].to_list() if "split" in labels_df.columns else \
                ["TRAIN"] * len(dates_raw)

    dates = pd.to_datetime(dates_raw)

    fig, ax = plt.subplots(figsize=(14, 4))

    # ── Cluster bars ──────────────────────────────────────────────────────────
    for d, c, sp in zip(dates, clusters, splits):
        color = CLUSTER_COLORS[f"C{c}"]
        alpha = 0.45 if sp == "TRAIN" else 0.90
        ax.bar(d, 1, width=32, color=color, linewidth=0,
               align="center", alpha=alpha)

    # ── NBER shading ──────────────────────────────────────────────────────────
    for start_str, end_str in meta["nber_recessions"]:
        s = pd.to_datetime(start_str)
        e = pd.to_datetime(end_str)
        ax.axvspan(s, e, alpha=0.14, color="black", zorder=0)

    # ── Split boundary lines ──────────────────────────────────────────────────
    for label, date_str, x_offset in [("← VAL", "2000-01", 0),
                                       ("← TEST", "2010-01", 0)]:
        ax.axvline(pd.to_datetime(date_str), color="black",
                   linewidth=1.0, linestyle="--", alpha=0.65)
        ax.text(pd.to_datetime(date_str) + pd.Timedelta(days=60),
                1.04, label, ha="left", va="bottom", fontsize=7, alpha=0.65)

    # ── TRAIN label ───────────────────────────────────────────────────────────
    ax.text(pd.to_datetime("1980-01"), 1.04, "TRAIN (in-sample)",
            ha="center", va="bottom", fontsize=7, alpha=0.5, style="italic")

    # ── Crisis annotations ────────────────────────────────────────────────────
    for date_str, label in meta["annotated_crises"].items():
        ax.text(pd.to_datetime(date_str) + pd.Timedelta(days=90),
                0.50, label, ha="left", va="center", fontsize=7.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.75))

    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Date")
    ax.set_yticks([])

    patches = [
        mpatches.Patch(color=CLUSTER_COLORS["C0"], label="C0 — systemic stress"),
        mpatches.Patch(color=CLUSTER_COLORS["C1"], label="C1 — housing expansion"),
        mpatches.Patch(color=CLUSTER_COLORS["C2"], label="C2 — labour recovery"),
        mpatches.Patch(color=CLUSTER_COLORS["C3"], label="C3 — transition"),
        mpatches.Patch(color="black", alpha=0.15,  label="NBER recession"),
    ]
    ax.legend(handles=patches, loc="upper left",
              ncol=5, fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Built: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — UMAP Dual Panel
# ══════════════════════════════════════════════════════════════════════════════

def _load_or_compute_umap_df() -> pd.DataFrame:
    """
    Load UMAP embeddings for TEST split (umap_kmeans.parquet has x_2d/y_2d as
    UMAP coords). Merge with NBER USREC for Panel B.
    Saves umap_embeddings_test.csv for future reuse.
    """
    out_csv = Path(UMAP_META["embeddings_file"])

    if out_csv.exists():
        return pd.read_csv(out_csv)

    # Load TEST UMAP parquet (x_2d/y_2d are already UMAP 2D coordinates)
    tst = pl.read_parquet(
        ROOT / "results/clustering_ablation/W6_d7_K4_b1/umap_kmeans.parquet"
    ).with_columns(pl.col("date").dt.date())

    # NBER USREC
    usrec = pl.read_csv(ROOT / "data/snapshots/nber_usrec.csv").with_columns(
        pl.col("observation_date").str.to_date().alias("date")
    ).select(["date", "USREC"])

    merged = tst.join(usrec, on="date", how="left").with_columns(
        pl.col("USREC").fill_null(0),
        pl.col("date").cast(pl.Utf8).str.slice(0, 7).alias("date_str"),
    )
    df = (
        merged
        .drop("date")
        .rename({
            "label":    "cluster",
            "x_2d":     "umap1",
            "y_2d":     "umap2",
            "USREC":    "nber",
            "date_str": "date",
        })
        .select(["date", "cluster", "umap1", "umap2", "nber"])
        .to_pandas()
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  Saved UMAP CSV: {out_csv} ({len(df)} rows)")
    return df


def build_umap(df: pd.DataFrame, meta: dict, output_path: str | Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    sharex=True, sharey=True)

    # ── Panel A — cluster assignments ─────────────────────────────────────────
    for c in [0, 1, 2, 3]:
        mask = df["cluster"] == c
        ax1.scatter(df.loc[mask, "umap1"], df.loc[mask, "umap2"],
                    c=CLUSTER_COLORS[f"C{c}"], s=20, alpha=0.75,
                    linewidths=0, label=f"C{c}", zorder=4)
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    ax1.legend(fontsize=7, markerscale=1.5,
               title="Cluster", title_fontsize=7)
    ax1.set_title("Panel A — Cluster Assignment", fontsize=8, pad=4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B — NBER labels, COVID highlighted ──────────────────────────────
    covid_mask = df["date"].isin(meta["covid_dates"])
    exp_mask   = df["nber"] == 0
    rec_mask   = (df["nber"] == 1) & ~covid_mask

    ax2.scatter(df.loc[exp_mask, "umap1"], df.loc[exp_mask, "umap2"],
                c="#AAAAAA", s=14, alpha=0.50, linewidths=0,
                label="Expansion (NBER=0)", zorder=3)
    ax2.scatter(df.loc[rec_mask, "umap1"], df.loc[rec_mask, "umap2"],
                c="#E74C3C", s=32, alpha=0.85, linewidths=0,
                label="Recession (NBER=1)", zorder=5)
    ax2.scatter(df.loc[covid_mask, "umap1"], df.loc[covid_mask, "umap2"],
                c="#C0392B", s=70, alpha=1.0, marker="*",
                linewidths=0.5, edgecolors="black",
                label="COVID-19 (2020-03/04)", zorder=6)

    # Annotate COVID points
    for _, row in df[covid_mask].iterrows():
        ax2.annotate(
            f"COVID\n{row['date'][:7]}",
            xy=(row["umap1"], row["umap2"]),
            xytext=(row["umap1"] + 0.8, row["umap2"] + 0.8),
            fontsize=6.5,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax2.set_xlabel("UMAP-1")
    ax2.legend(fontsize=7, markerscale=1.2)
    ax2.set_title("Panel B — NBER Labels", fontsize=8, pad=4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Built: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Bootstrap Stability
# ══════════════════════════════════════════════════════════════════════════════

def build_stability(data: dict, output_path: str | Path) -> None:
    clusters  = data["clusters"]
    j_mean    = np.array(data["j_mean"])
    j_lo      = np.array(data["j_ci_lo"])
    j_hi      = np.array(data["j_ci_hi"])
    pr        = np.array(data["pr_stable"])
    colors    = [CLUSTER_COLORS[f"C{i}"] for i in range(4)]
    threshold = data["threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))

    y_pos = np.arange(len(clusters))

    # ── Left panel: Jaccard mean + CI ─────────────────────────────────────────
    ax1.barh(y_pos, j_mean,
             xerr=[j_mean - j_lo, j_hi - j_mean],
             color=colors, alpha=0.85, capsize=5,
             height=0.55, error_kw={"linewidth": 1.2})
    ax1.axvline(threshold, color="black", linewidth=1.0,
                linestyle="--", alpha=0.7,
                label=f"Threshold = {threshold}")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(clusters, fontsize=8)
    ax1.set_xlabel("Mean Jaccard Index (95% CI)")
    ax1.set_xlim(0, 1.08)
    ax1.legend(fontsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate mean inside bar
    for i, (m, lo, hi) in enumerate(zip(j_mean, j_lo, j_hi)):
        ax1.text(0.02, i, f" J̄={m:.3f}  [{lo:.2f},{hi:.2f}]",
                 va="center", ha="left", fontsize=7,
                 color="white", fontweight="bold")

    # ── Right panel: Pr(J ≥ 0.60) ─────────────────────────────────────────────
    ax2.barh(y_pos, pr, color=colors, alpha=0.85, height=0.55)
    ax2.axvline(60, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * 4)
    ax2.set_xlabel("Pr(Jaccard ≥ 0.60)  [%]")
    ax2.set_xlim(0, 115)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i, p in enumerate(pr):
        ax2.text(p + 1.5, i, f"{p:.1f}%", va="center", ha="left", fontsize=8)

    fig.text(
        0.5, -0.03,
        f"Consensus clustering, B={data['B']}, {data['subsample_pct']}% subsampling.",
        ha="center", fontsize=7, style="italic", alpha=0.65,
    )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Built: {output_path}")


def _stability_is_consensus(sprint8_path: str | Path) -> bool:
    """
    Heuristic: check sprint8/J_bootstrap_stability.png was built with
    consensus data (J_mean C0 = 0.640) by re-reading the sprint2 CSV.
    If sprint8 was generated in the same session where consensus data is
    present with C0=0.640, assume it's up-to-date.
    """
    csv = ROOT / "results/sprint2/consensus_jaccard_cis.csv"
    if not csv.exists():
        return False
    try:
        jac = pl.read_csv(csv)
        it_c0 = jac.filter(
            (pl.col("encoder") == "iTransformer") & (pl.col("cluster") == 0)
        )
        if len(it_c0) == 0:
            return False
        val = it_c0["jaccard_mean"].to_list()[0]
        return abs(val - 0.640) < 0.005
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Window Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def build_window_sensitivity(data: dict, output_path: str | Path) -> None:
    W         = np.array(data["W"])
    F1        = np.array(data["F1_tol"])
    canon_W   = data["canonical_W"]
    raw_pca   = data["raw_pca_F1"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(W, F1, "o-", color="#2C3E50", linewidth=1.8,
            markersize=6, label="NBER F1 (±6m tol.)")

    # Canonical star
    canon_idx = list(W).index(canon_W)
    ax.scatter([W[canon_idx]], [F1[canon_idx]],
               s=160, color="#C0392B", zorder=6,
               marker="*", label=f"Canonical (W={canon_W})")
    ax.annotate(
        f"W={canon_W}\nF1={F1[canon_idx]:.3f}",
        xy=(W[canon_idx], F1[canon_idx]),
        xytext=(W[canon_idx] + 0.8, F1[canon_idx] + 0.045),
        fontsize=8, color="#C0392B",
        arrowprops=dict(arrowstyle="-", color="#C0392B", lw=0.8),
    )

    # raw-PCA baseline
    ax.axhline(raw_pca, color="#95A5A6", linewidth=1.2,
               linestyle="--", alpha=0.8,
               label=f"raw-PCA baseline (F1={raw_pca:.3f})")

    ax.set_xlabel("Window size W (months)")
    ax.set_ylabel("NBER F1 under ±6-month tolerance")
    ax.set_xticks(W)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Built: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Macro Profiles Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def build_macro_profiles(data: dict, output_path: str | Path) -> None:
    series_names   = list(data["series"].keys())
    readable_names = [data["series_descriptions"][s] for s in series_names]
    matrix         = np.array([data["series"][s] for s in series_names])
    # shape: (n_series, 4)

    # Sort rows by |z-score in C0| descending
    sort_idx       = np.argsort(np.abs(matrix[:, 0]))[::-1]
    matrix         = matrix[sort_idx]
    readable_names = [readable_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(
        matrix, ax=ax,
        cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        xticklabels=data["cluster_labels"],
        yticklabels=readable_names,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Mean z-score (TEST window)", "shrink": 0.6},
    )

    # Hide annotations with low absolute value
    for text in ax.texts:
        try:
            if abs(float(text.get_text())) < 0.5:
                text.set_text("")
        except ValueError:
            pass

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Built: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import traceback
    results: dict[str, str] = {}

    # ── Figure 1 — Two-axis scatter ───────────────────────────────────────────
    print("\n[Fig 1] Two-axis scatter (MCC vs Silhouette)")
    path1 = FIG_DIR / "fig_two_axis.png"
    canon1 = {"version": "sprint1_mcc_v2", "n_encoders": len(ENCODER_DATA)}
    if check_figure(path1, canon1) == "REBUILD":
        try:
            build_two_axis(ENCODER_DATA, path1)
            save_hash(path1, canon1)
        except Exception:
            traceback.print_exc()
    results["fig_two_axis"] = str(path1)

    # ── Figure 2 — Cluster timeline ───────────────────────────────────────────
    print("\n[Fig 2] Cluster timeline 1965-2026")
    path2 = FIG_DIR / "fig_timeline.png"
    sprint8_tl = ROOT / "results/sprint8/E_cluster_timeline.png"

    # The sprint8 timeline only covers VAL+TEST. Always rebuild with full data.
    if check_figure(path2, TIMELINE_META) == "REBUILD":
        try:
            labels_df = _build_cluster_labels_full()
            build_timeline(labels_df, TIMELINE_META, path2)
            save_hash(path2, TIMELINE_META)
        except Exception:
            traceback.print_exc()
            if sprint8_tl.exists():
                print("  Fallback: copying sprint8 E_cluster_timeline.png")
                shutil.copy(sprint8_tl, path2)
    results["fig_timeline"] = str(path2)

    # ── Figure 3 — UMAP dual panel ────────────────────────────────────────────
    print("\n[Fig 3] UMAP dual panel")
    path3 = FIG_DIR / "fig_umap.png"
    if check_figure(path3, UMAP_META) == "REBUILD":
        try:
            umap_df = _load_or_compute_umap_df()
            build_umap(umap_df, UMAP_META, path3)
            save_hash(path3, UMAP_META)
        except Exception:
            traceback.print_exc()
    results["fig_umap"] = str(path3)

    # ── Figure 4 — Bootstrap stability ───────────────────────────────────────
    print("\n[Fig 4] Bootstrap stability (consensus clustering)")
    path4 = FIG_DIR / "fig_stability.png"
    sprint8_stab = ROOT / "results/sprint8/J_bootstrap_stability.png"

    # CRITICAL: always use canonical STABILITY_DATA (consensus, J_mean C0=0.640)
    if check_figure(path4, STABILITY_DATA) == "REBUILD":
        stab_consensus = _stability_is_consensus(sprint8_stab)
        if sprint8_stab.exists() and not stab_consensus:
            print("  WARNING: sprint8 stability uses single-run data — REBUILDING with consensus")
        try:
            build_stability(STABILITY_DATA, path4)
            save_hash(path4, STABILITY_DATA)
        except Exception:
            traceback.print_exc()
    results["fig_stability"] = str(path4)

    # ── Figure 5 — Window sensitivity ────────────────────────────────────────
    print("\n[Fig 5] Window sensitivity")
    path5 = FIG_DIR / "fig_window_sensitivity.png"
    if check_figure(path5, WINDOW_DATA) == "REBUILD":
        try:
            build_window_sensitivity(WINDOW_DATA, path5)
            save_hash(path5, WINDOW_DATA)
        except Exception:
            traceback.print_exc()
    results["fig_window_sensitivity"] = str(path5)

    # ── Figure 6 — Macro profiles ─────────────────────────────────────────────
    print("\n[Fig 6] Macro profiles heatmap")
    path6 = FIG_DIR / "fig_macro_profiles.png"
    macro_canon = {"version": "sprint3_top12_eniac"}
    if check_figure(path6, macro_canon) == "REBUILD":
        try:
            build_macro_profiles(MACRO_PROFILES, path6)
            save_hash(path6, macro_canon)
        except Exception:
            traceback.print_exc()
    results["fig_macro_profiles"] = str(path6)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BUILD SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, path in results.items():
        ok = os.path.exists(path)
        if not ok:
            all_ok = False
        print(f"  [{'OK    ' if ok else 'MISSING'}] {name}: {path}")

    if all_ok:
        print("\nAll 6 figures ready.")
    else:
        print("\nWARNING: some figures are missing — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
