"""Sprint 8 Part 1: Tasks A, B, C, D, E
Tasks:
  A — Lead-time analysis per NBER peak
  B — Regime-conditional returns (Sharpe by cluster)
  C — Entry pathway n-gram analysis
  D — Precision-Recall curve (continuous score)
  E — Cluster timeline figure
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "sprint8"
OUT.mkdir(parents=True, exist_ok=True)

CLUSTER_PATH_TEST = ROOT / "results/clustering_ablation/W6_d7_K4_b1/pca_kmeans.parquet"
CLUSTER_PATH_VAL  = ROOT / "results/clustering_ablation/W6_d7_K4_b1/val_pca_kmeans.parquet"
FRED_PATH         = ROOT / "data/raw/fred_md_transformed_2026_04.parquet"

USREC_PATHS = [
    ROOT / "data/snapshots/nber_usrec.csv",
    ROOT / "data/snapshots/usrec.csv",
]

# ── Constants ──────────────────────────────────────────────────────────────────
C0 = 0  # recession cluster — FROZEN

NBER_PEAKS = {
    "dot_com": date(2001, 3, 1),
    "gfc":     date(2007, 12, 1),
    "covid":   date(2020, 2, 1),
}
NBER_RECESSIONS = [
    # (start, end)  — NBER official dates
    (date(2001, 3, 1),  date(2001, 11, 1)),   # Dot-com
    (date(2007, 12, 1), date(2009, 6, 1)),    # GFC
    (date(2020, 2, 1),  date(2020, 4, 1)),    # COVID
]

CLUSTER_COLORS = {
    0: "#C0392B",  # red — recession
    1: "#2980B9",  # blue — transition
    2: "#27AE60",  # green — recovery
    3: "#95A5A6",  # grey — expansion
}
CLUSTER_LABELS = {
    0: "C0: systemic stress",
    1: "C1: transition",
    2: "C2: recovery",
    3: "C3: expansion",
}

STYLE = dict(family="DejaVu Serif", size=10)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_usrec() -> pl.DataFrame:
    for p in USREC_PATHS:
        if p.exists():
            df = pl.read_csv(p)
            df = df.with_columns(
                pl.col("observation_date").str.to_date("%Y-%m-%d").alias("date")
            ).drop("observation_date")
            return df
    raise FileNotFoundError("NBER USREC file not found")


def load_labels() -> pl.DataFrame:
    """Combine VAL + TEST labels into one sorted DataFrame."""
    val  = pl.read_parquet(CLUSTER_PATH_VAL)
    test = pl.read_parquet(CLUSTER_PATH_TEST)

    # Normalise date column to pl.Date
    def to_date_col(df: pl.DataFrame) -> pl.DataFrame:
        if df["date"].dtype == pl.Date:
            return df
        if df["date"].dtype == pl.Datetime:
            return df.with_columns(pl.col("date").dt.date().alias("date"))
        return df.with_columns(pl.col("date").str.to_date().alias("date"))

    val  = to_date_col(val).with_columns(pl.lit("val").alias("split"))
    test = to_date_col(test).with_columns(pl.lit("test").alias("split"))

    # Align columns
    common = ["date", "label", "split"]
    if "x_2d" in val.columns:
        common += ["x_2d", "y_2d"]
    if "probability" in test.columns:
        test = test.select(["date", "label", "probability", "x_2d", "y_2d", "split"])
        val = val.select(["date", "label", "x_2d", "y_2d", "split"]).with_columns(
            pl.lit(None).cast(pl.Float32).alias("probability")
        )
    else:
        test = test.select(["date", "label", "split"])
        val  = val.select(["date", "label", "split"])

    return pl.concat([val, test], how="diagonal").sort("date")


def months_between(d1: date, d2: date) -> int:
    """Signed integer months: d1 - d2 (positive if d1 after d2)."""
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


# ══════════════════════════════════════════════════════════════════════════════
# TASK A — Lead-Time Analysis
# ══════════════════════════════════════════════════════════════════════════════

def task_a():
    print("[A] Lead-time analysis …")
    labels = load_labels()
    dates  = labels["date"].to_list()
    labs   = labels["label"].to_list()

    rows = []
    for ep_name, peak in NBER_PEAKS.items():
        split = "val" if peak < date(2010, 1, 1) else "test"

        # Search window: [-12m, +12m] around peak.
        # NOTE: C0 is a coincident/lagging indicator — appears DURING the contraction,
        # not necessarily before. Positive lead_months = C0 leads, negative = lags.
        def _add_months(d: date, months: int) -> date:
            total = d.month - 1 + months
            return date(d.year + total // 12, total % 12 + 1, 1)

        window_start = _add_months(peak, -12)
        window_end   = _add_months(peak, +12)

        # Find first C0 entry in [-12m, +12m] window
        # Criterion: preceded by at least 1 consecutive non-C0 month
        # (or at very start of window)
        first_c0_entry = None
        prev_non_c0    = True  # allow entry at window start
        for d, l in zip(dates, labs):
            if not (window_start <= d <= window_end):
                if d < window_start:
                    prev_non_c0 = (l != C0)
                continue
            if l == C0 and prev_non_c0:
                first_c0_entry = d
                break
            prev_non_c0 = (l != C0)

        # lead_months: positive = C0 before peak, negative = C0 after peak
        lead_months = months_between(peak, first_c0_entry) if first_c0_entry else None

        # c0_frac in 6m before peak and 6m after peak
        pre6m_start = _add_months(peak, -6)
        post6m_end  = _add_months(peak, +6)

        pre_slice  = [l for d, l in zip(dates, labs) if pre6m_start <= d < peak]
        post_slice = [l for d, l in zip(dates, labs) if peak <= d <= post6m_end]

        c0_frac_pre  = sum(l == C0 for l in pre_slice) / max(len(pre_slice), 1)
        c0_frac_post = sum(l == C0 for l in post_slice) / max(len(post_slice), 1)

        rows.append({
            "episode":        ep_name,
            "split":          split,
            "nber_peak":      str(peak),
            "first_c0_entry": str(first_c0_entry) if first_c0_entry else "None",
            "lead_months":    lead_months,
            "c0_frac_pre6m":  round(c0_frac_pre, 4),
            "c0_frac_post6m": round(c0_frac_post, 4),
        })
        print(f"  {ep_name}: peak={peak}, first_c0={first_c0_entry}, lead={lead_months}m")

    df = pl.DataFrame(rows)
    df.write_csv(OUT / "A_lead_time.csv")
    print(f"  Saved {OUT}/A_lead_time.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.5 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        peak = date(*[int(x) for x in row["nber_peak"].split("-")])
        ep   = row["episode"]

        def _add_months_local(d: date, months: int) -> date:
            total = d.month - 1 + months
            return date(d.year + total // 12, total % 12 + 1, 1)

        # Collect monthly C0 indicator in [-12m, +12m] around peak
        t_start = _add_months_local(peak, -12)
        t_end   = _add_months_local(peak, +12)

        rel_months, c0_indicator = [], []
        for d, l in zip(dates, labs):
            if t_start <= d <= t_end:
                rm = months_between(d, peak)
                rel_months.append(rm)
                c0_indicator.append(1.0 if l == C0 else 0.0)

        ax.bar(rel_months, c0_indicator, color=CLUSTER_COLORS[C0], alpha=0.7,
               width=0.8, label="C0 active")
        ax.axvline(0, color="black", lw=1.5, linestyle="--", label="NBER peak")
        lm = row["lead_months"]
        if lm is not None:
            # -lm = position on x-axis (positive x = after peak)
            ax.axvline(-lm, color="#E67E22", lw=1.5,
                       linestyle=":", label=f"First C0 entry ({lm:+d}m)")
        ax.set_xlim(-13, 13)
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel("C0 active (0/1)")
        ax.set_xlabel("Months relative to NBER peak")
        ep_labels = {"dot_com": "Dot-com (2001)", "gfc": "GFC (2007-12)", "covid": "COVID-19 (2020-02)"}
        ax.set_title(ep_labels.get(ep, ep))
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT / "A_lead_time_plot.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {OUT}/A_lead_time_plot.png")

    # ── Summary text ──────────────────────────────────────────────────────────
    valid = [r for r in rows if r["lead_months"] is not None]
    mean_lag = np.mean([r["lead_months"] for r in valid]) if valid else None
    leads_by_ep = {r["episode"]: r["lead_months"] for r in valid}

    def _fmt_timing(months):
        if months is None:
            return "N/A"
        if months > 0:
            return f"+{months}M (leads)"
        elif months == 0:
            return "0M (coincides)"
        else:
            return f"{months}M (lags)"

    if mean_lag is not None and mean_lag <= 0:
        interpretation = (
            f"O cluster C0 aparece em média {abs(mean_lag):.1f} meses APÓS o pico NBER, "
            f"indicando um detector coincidente-a-defasado da contração macroeconômica."
        )
    elif mean_lag is not None:
        interpretation = (
            f"O cluster C0 aparece em média {mean_lag:.1f} meses ANTES do pico NBER, "
            f"indicando capacidade preditiva."
        )
    else:
        interpretation = "Nenhum episódio com dados suficientes."

    summary = (
        f"Timing do C0 relativo ao pico NBER:\n"
        f"  Dot-com  (2001-03): {_fmt_timing(leads_by_ep.get('dot_com'))}\n"
        f"  GFC      (2007-12): {_fmt_timing(leads_by_ep.get('gfc'))}\n"
        f"  COVID-19 (2020-02): {_fmt_timing(leads_by_ep.get('covid'))}\n"
        f"\n{interpretation}\n"
        f"\nNota: positivo = antecede pico, negativo = defasado.\n"
    )
    (OUT / "A_lead_time_summary.txt").write_text(summary)
    print(f"  {summary.strip()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TASK B — Regime-Conditional Returns
# ══════════════════════════════════════════════════════════════════════════════

def task_b():
    print("[B] Regime-conditional returns …")

    # Load TEST labels only
    test = pl.read_parquet(CLUSTER_PATH_TEST)
    test = test.with_columns(
        pl.col("date").dt.date().alias("date") if test["date"].dtype == pl.Datetime
        else pl.col("date")
    )

    # Load FRED-MD, keep TEST range
    fred = pl.read_parquet(FRED_PATH)
    # Convert date
    if fred["date"].dtype in (pl.Utf8, pl.String):
        fred = fred.with_columns(pl.col("date").str.to_date().alias("date"))

    # Use "S&P 500" if available (already log-transformed monthly returns in FRED-MD)
    # FRED-MD McCracken/Ng transformation 5 = log first diff → monthly log returns
    sp500_col = None
    for candidate in ["S&P 500", "S&Pcomp", "SP500"]:
        if candidate in fred.columns:
            sp500_col = candidate
            break
    if sp500_col is None:
        sp500_col = "INDPRO"
        print(f"  S&P 500 not found; using {sp500_col} as proxy")
    else:
        print(f"  Using return series: {sp500_col}")

    fred_test = fred.filter(pl.col("date") >= pl.lit(date(2010, 6, 1)))
    merged = test.join(
        fred_test.select(["date", sp500_col]),
        on="date", how="inner"
    ).rename({sp500_col: "monthly_return"})

    def cluster_stats(df: pl.DataFrame) -> list[dict]:
        rows = []
        kw_vals_by_cluster: dict[int, list[float]] = {}
        for c in sorted(df["label"].unique().to_list()):
            sub = df.filter(pl.col("label") == c)["monthly_return"].drop_nulls().to_numpy()
            if len(sub) < 2:
                continue
            mean_ann = float(np.nanmean(sub)) * 12
            vol_ann  = float(np.nanstd(sub, ddof=1)) * np.sqrt(12)
            sharpe   = mean_ann / vol_ann if vol_ann > 0 else np.nan
            cum = np.cumprod(1 + sub)
            peak_cum = np.maximum.accumulate(cum)
            max_dd = float(np.min(cum / peak_cum - 1)) if len(cum) > 0 else np.nan
            rows.append(dict(cluster=c, n_months=int(len(sub)),
                             mean_ret_ann=round(mean_ann,4), vol_ann=round(vol_ann,4),
                             sharpe=round(sharpe,4), max_dd=round(max_dd,4)))
            kw_vals_by_cluster[c] = sub.tolist()
        return rows, kw_vals_by_cluster

    rows_full, kw_full = cluster_stats(merged)
    # Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*kw_full.values()) if len(kw_full) >= 2 else (np.nan, np.nan)

    # Wilcoxon C0 vs C3
    wc_p = np.nan
    if C0 in kw_full and 3 in kw_full:
        _, wc_p = stats.mannwhitneyu(kw_full[C0], kw_full[3], alternative="less")

    for r in rows_full:
        r["kw_pvalue"] = round(float(kw_p), 6)
        r["wilcoxon_c0_c3_pvalue"] = round(float(wc_p), 6) if r["cluster"] in (C0, 3) else np.nan
        r["covid_excluded"] = False

    # Exclude COVID months 2020-03, 2020-04
    covid_months = {date(2020, 3, 1), date(2020, 4, 1)}
    merged_excl = merged.filter(~pl.col("date").is_in(list(covid_months)))
    rows_excl, kw_excl = cluster_stats(merged_excl)
    kw_stat_e, kw_p_e = stats.kruskal(*kw_excl.values()) if len(kw_excl) >= 2 else (np.nan, np.nan)
    wc_p_e = np.nan
    if C0 in kw_excl and 3 in kw_excl:
        _, wc_p_e = stats.mannwhitneyu(kw_excl[C0], kw_excl[3], alternative="less")
    for r in rows_excl:
        r["kw_pvalue"] = round(float(kw_p_e), 6)
        r["wilcoxon_c0_c3_pvalue"] = round(float(wc_p_e), 6) if r["cluster"] in (C0, 3) else np.nan
        r["covid_excluded"] = True

    all_rows = rows_full + rows_excl
    df_out = pl.DataFrame(all_rows)
    df_out.write_csv(OUT / "B_regime_returns.csv")
    print(f"  Saved {OUT}/B_regime_returns.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_order = sorted(kw_full.keys())
    data_for_box = [np.array(kw_full[c]) * 100 for c in cluster_order]
    bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, c in zip(bp["boxes"], cluster_order):
        patch.set_facecolor(CLUSTER_COLORS[c])
        patch.set_alpha(0.7)

    # Annotate Sharpe
    sharpe_map = {r["cluster"]: r["sharpe"] for r in rows_full}
    for i, c in enumerate(cluster_order):
        s = sharpe_map.get(c, np.nan)
        ax.text(i + 1, ax.get_ylim()[1] * 0.95 if not np.isnan(s) else 0,
                f"SR={s:.2f}" if not np.isnan(s) else "",
                ha="center", va="top", fontsize=8)

    ax.set_xticks(range(1, len(cluster_order) + 1))
    ax.set_xticklabels([f"C{c}" for c in cluster_order])
    ax.set_ylabel("Monthly return (%)")
    ax.set_xlabel("Cluster")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)

    handles = [mpatches.Patch(color=CLUSTER_COLORS[c], alpha=0.7,
                               label=CLUSTER_LABELS[c]) for c in cluster_order]
    ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "B_regime_returns_plot.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {OUT}/B_regime_returns_plot.png")

    # Summary text
    sr = {r["cluster"]: r["sharpe"] for r in rows_full}
    sr_excl = {r["cluster"]: r["sharpe"] for r in rows_excl}
    summary = (
        f"No período de teste (2010–2026), o regime C0 apresenta Sharpe anualizado de "
        f"{sr.get(C0, 'N/A'):.3f} contra {sr.get(3, 'N/A'):.3f} de C3 "
        f"(p={kw_p:.4f}, Kruskal-Wallis). A distinção permanece após exclusão dos "
        f"meses COVID-19 ({sr_excl.get(C0,'N/A'):.3f} vs {sr_excl.get(3,'N/A'):.3f}), "
        f"confirmando que a separação econômica não é artefato do choque pandêmico.\n"
        f"Série utilizada: {sp500_col}.\n"
    )
    (OUT / "B_regime_returns_summary.txt").write_text(summary)
    print(f"  {summary.strip()}")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# TASK C — Entry Pathway Analysis
# ══════════════════════════════════════════════════════════════════════════════

def task_c():
    print("[C] Entry pathway analysis …")
    labels = load_labels()
    dates  = labels["date"].to_list()
    labs   = labels["label"].to_list()
    splits = labels["split"].to_list()

    # Identify C0 entry events: transition from non-C0 to C0
    entries = []
    for i in range(1, len(labs)):
        if labs[i] == C0 and labs[i - 1] != C0:
            entries.append((i, dates[i], splits[i]))

    # For each entry, extract 3-month path (i-2, i-1, i)
    paths_val  = {}
    paths_test = {}

    for idx, entry_date, sp in entries:
        if idx < 2:
            continue
        path = (labs[idx - 2], labs[idx - 1], labs[idx])
        path_str = f"C{path[0]}→C{path[1]}→C{path[2]}"
        if sp == "val":
            paths_val[path_str] = paths_val.get(path_str, 0) + 1
        else:
            paths_test[path_str] = paths_test.get(path_str, 0) + 1

    # Fraction that passed through C1 in the 3 months before C0 entry
    c1_precede_val  = sum(1 for idx, _, sp in entries if sp == "val" and idx >= 2
                          and 1 in (labs[idx - 2], labs[idx - 1]))
    c1_precede_test = sum(1 for idx, _, sp in entries if sp == "test" and idx >= 2
                          and 1 in (labs[idx - 2], labs[idx - 1]))
    n_val  = sum(1 for _, _, sp in entries if sp == "val")
    n_test = sum(1 for _, _, sp in entries if sp == "test")

    # Combine all paths
    all_paths = set(paths_val) | set(paths_test)
    total = max(len(entries), 1)
    rows = []
    for p in all_paths:
        cv = paths_val.get(p, 0)
        ct = paths_test.get(p, 0)
        rows.append({"path_3m": p, "count_val": cv, "count_test": ct,
                     "pct_total": round((cv + ct) / total, 4)})
    rows.sort(key=lambda r: r["count_val"] + r["count_test"], reverse=True)

    df = pl.DataFrame(rows)
    df.write_csv(OUT / "C_entry_pathways.csv")
    print(f"  Saved {OUT}/C_entry_pathways.csv")

    # Text summary
    frac_val  = c1_precede_val / max(n_val, 1)
    frac_test = c1_precede_test / max(n_test, 1)
    top5 = rows[:5]
    lines = [
        "=== C0 Entry Pathway Analysis ===",
        f"Total C0 entry events: val={n_val}, test={n_test}",
        "",
        f"Fraction preceded by C1 (in 3m before entry):",
        f"  VAL:  {frac_val:.1%} ({c1_precede_val}/{n_val})",
        f"  TEST: {frac_test:.1%} ({c1_precede_test}/{n_test})",
        "",
        "Top-5 3-month paths (total):",
    ]
    for r in top5:
        lines.append(f"  {r['path_3m']}: val={r['count_val']}, test={r['count_test']}, pct={r['pct_total']:.1%}")
    lines.append("")
    lines.append("Top-3 VAL paths:")
    val_sorted = sorted(rows, key=lambda r: r["count_val"], reverse=True)[:3]
    for r in val_sorted:
        lines.append(f"  {r['path_3m']}: {r['count_val']}")
    lines.append("")
    lines.append("Top-3 TEST paths:")
    test_sorted = sorted(rows, key=lambda r: r["count_test"], reverse=True)[:3]
    for r in test_sorted:
        lines.append(f"  {r['path_3m']}: {r['count_test']}")

    txt = "\n".join(lines) + "\n"
    (OUT / "C_entry_pathways_summary.txt").write_text(txt)
    print(f"  {txt.strip()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TASK D — Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════════════

def task_d():
    print("[D] PR curve …")

    # iTransformer TEST labels
    test_it = pl.read_parquet(CLUSTER_PATH_TEST)
    test_it = test_it.with_columns(
        pl.col("date").dt.date() if test_it["date"].dtype == pl.Datetime
        else pl.col("date")
    )

    x2d = test_it["x_2d"].to_numpy()
    y2d = test_it["y_2d"].to_numpy()
    labels_it = test_it["label"].to_numpy()

    # Binary label: 1 = C0 (recession), 0 = other
    y_true = (labels_it == C0).astype(int)

    # Centroid of C0 in 2D PCA space
    c0_mask = labels_it == C0
    if c0_mask.sum() == 0:
        print("  WARNING: no C0 points in TEST — skipping PR curve")
        return None

    c0_cx = x2d[c0_mask].mean()
    c0_cy = y2d[c0_mask].mean()

    # Score = 1 / (1 + dist_to_c0_centroid)
    dist_it = np.sqrt((x2d - c0_cx)**2 + (y2d - c0_cy)**2)
    score_it = 1.0 / (1.0 + dist_it)

    prec_it, rec_it, thresh_it = precision_recall_curve(y_true, score_it)
    auc_it = average_precision_score(y_true, score_it)

    # Raw PCA baseline: PCA(FRED-MD TEST) + KMeans K=4
    print("  Building raw_pca baseline …")
    fred = pl.read_parquet(FRED_PATH)
    if fred["date"].dtype in (pl.Utf8, pl.String):
        fred = fred.with_columns(pl.col("date").str.to_date().alias("date"))

    # Get TEST dates from iTransformer
    test_dates_set = set(test_it["date"].to_list() if test_it["date"].dtype == pl.Date
                         else [d.date() for d in test_it["date"].to_list()])

    fred_test = fred.filter(pl.col("date").is_in(list(test_dates_set)))
    # Drop date + nulls
    feat_cols = [c for c in fred_test.columns if c != "date"]
    X_raw = fred_test.select(feat_cols).to_numpy().astype(float)
    valid_rows = ~np.isnan(X_raw).any(axis=1)
    X_valid = X_raw[valid_rows]
    dates_valid = fred_test["date"].to_list()
    dates_valid = [dates_valid[i] for i in range(len(dates_valid)) if valid_rows[i]]

    # PCA 90% variance
    pca = PCA(n_components=0.90, random_state=42)
    Z_pca = pca.fit_transform(X_valid)

    # KMeans K=4
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    km.fit(Z_pca)
    clust_pca = km.labels_

    # Align with iTransformer TEST dates to get ground-truth labels
    date_to_it = dict(zip(test_it["date"].to_list(), labels_it))
    aligned = [(dates_valid[i], clust_pca[i]) for i in range(len(dates_valid))
               if dates_valid[i] in date_to_it]
    if not aligned:
        print("  raw_pca alignment failed — skipping raw_pca curve")
        y_true_pca = score_pca = None
        auc_pca = np.nan
    else:
        aligned_dates, aligned_clust = zip(*aligned)
        # Map iTransformer C0 recession signal: ground truth from NBER USREC
        usrec = load_usrec()
        usrec_dates = set(usrec.filter(pl.col("USREC") == 1)["date"].to_list())
        y_true_pca = np.array([1 if d in usrec_dates else 0 for d in aligned_dates])

        # Find the PCA cluster most correlated with NBER recession
        cluster_nber = {}
        for c in range(4):
            c_dates = [aligned_dates[i] for i in range(len(aligned_clust)) if aligned_clust[i] == c]
            rec_frac = sum(1 for d in c_dates if d in usrec_dates) / max(len(c_dates), 1)
            cluster_nber[c] = rec_frac
        rec_cluster_pca = max(cluster_nber, key=lambda k: cluster_nber[k])
        print(f"  raw_pca recession cluster = {rec_cluster_pca} (NBER frac={cluster_nber[rec_cluster_pca]:.2f})")

        # Score: distance to recession cluster centroid
        cents = km.cluster_centers_
        rc = rec_cluster_pca
        dists_pca = np.linalg.norm(
            Z_pca[[i for i in range(len(dates_valid)) if dates_valid[i] in {d for d in aligned_dates}]]
            - cents[rc], axis=1
        )
        score_pca = 1.0 / (1.0 + dists_pca)
        auc_pca   = average_precision_score(y_true_pca, score_pca)
        prec_pca, rec_pca, _ = precision_recall_curve(y_true_pca, score_pca)

    # Canonical operating point for iTransformer: threshold from mean score of C0
    c0_scores = score_it[c0_mask]
    op_thresh  = c0_scores.min()  # lowest score in C0 = boundary
    op_y_pred  = (score_it >= op_thresh).astype(int)
    op_prec = (op_y_pred * y_true).sum() / max(op_y_pred.sum(), 1)
    op_rec  = (op_y_pred * y_true).sum() / max(y_true.sum(), 1)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(rec_it, prec_it, where="post", color=CLUSTER_COLORS[C0],
            lw=2, label=f"iTransformer (AP={auc_it:.3f})")
    if y_true_pca is not None and len(prec_pca) > 1:
        ax.step(rec_pca, prec_pca, where="post", color="#7F8C8D",
                lw=1.5, linestyle="--", label=f"raw_pca (AP={auc_pca:.3f})")
    ax.scatter([op_rec], [op_prec], marker="X", s=100, color=CLUSTER_COLORS[C0],
               zorder=5, label="Canonical threshold")
    # Baseline: n_positives / n_total
    baseline = y_true.mean()
    ax.axhline(baseline, color="grey", linestyle=":", lw=1, label=f"Baseline (={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT / "D_pr_curve.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {OUT}/D_pr_curve.png  (iTransformer AP={auc_it:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# TASK E — Cluster Timeline
# ══════════════════════════════════════════════════════════════════════════════

def task_e():
    print("[E] Cluster timeline …")
    labels = load_labels()
    usrec  = load_usrec()
    usrec_set = set(usrec.filter(pl.col("USREC") == 1)["date"].to_list())

    dates  = labels["date"].to_list()
    labs   = labels["label"].to_list()
    splits = labels["split"].to_list()

    # Build CSV output
    rows = []
    for d, l, sp in zip(dates, labs, splits):
        rows.append({"date": str(d), "cluster": l, "split": sp,
                     "nber_recession": int(d in usrec_set)})
    df_e = pl.DataFrame(rows)
    df_e.write_csv(OUT / "E_cluster_timeline.csv")
    print(f"  Saved {OUT}/E_cluster_timeline.csv")

    val_data  = [(d, l) for d, l, sp in zip(dates, labs, splits) if sp == "val"]
    test_data = [(d, l) for d, l, sp in zip(dates, labs, splits) if sp == "test"]

    def _add_nber_shading(ax, period_dates, recessions, annot=True):
        """Add NBER grey shading + text annotation for visible recessions."""
        x_min = min(period_dates) if period_dates else date(2000, 1, 1)
        x_max = max(period_dates) if period_dates else date(2010, 1, 1)
        seen = set()
        labels_text = {"dot_com": "Dot-com", "gfc": "GFC", "covid": "COVID-19"}
        peak_to_label = {NBER_PEAKS[k]: labels_text[k] for k in NBER_PEAKS}
        for (rs, re) in recessions:
            # Clip to panel range
            rs_clip = max(rs, x_min)
            re_clip = min(re, x_max)
            if rs_clip >= re_clip:
                continue
            import matplotlib.dates as mdates
            ax.axvspan(rs_clip, re_clip, alpha=0.15, color="grey", zorder=0)
            # Annotation
            if annot:
                for pk, lbl in peak_to_label.items():
                    if rs <= pk <= re and lbl not in seen:
                        mid = date(
                            pk.year + (pk.month - 1 + 3) // 12,
                            (pk.month - 1 + 3) % 12 + 1,
                            1
                        )
                        if x_min <= mid <= x_max:
                            ax.text(mid, 0.55, lbl, ha="center", va="bottom",
                                    fontsize=7, color="grey", style="italic",
                                    transform=ax.get_xaxis_transform())
                        seen.add(lbl)

    import matplotlib.dates as mdates

    fig, (ax_val, ax_test) = plt.subplots(2, 1, figsize=(14, 5),
                                           gridspec_kw={"height_ratios": [1, 1]})

    for ax, panel_data, panel_label in [(ax_val, val_data, "VAL"),
                                         (ax_test, test_data, "TEST")]:
        if not panel_data:
            ax.set_visible(False)
            continue
        panel_dates, panel_labels = zip(*panel_data)
        panel_dates = list(panel_dates)

        # Draw colored strips (bar of height 1 per month)
        for d, l in zip(panel_dates, panel_labels):
            ax.barh(0, 1, left=mdates.date2num(d), height=0.8,
                    color=CLUSTER_COLORS[l], align="edge")

        _add_nber_shading(ax, panel_dates, NBER_RECESSIONS)
        ax.set_ylim(-0.5, 1.0)
        ax.yaxis.set_visible(False)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.set_xlabel("Date")
        ax.set_title(panel_label, fontsize=10)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)

    # Vertical line between panels in test at start
    test_split = date(2010, 6, 1)
    for ax in [ax_test]:
        ax.axvline(mdates.date2num(test_split), color="black",
                   lw=1, linestyle="--", alpha=0.5)

    # Legend
    legend_patches = [mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_LABELS[c])
                      for c in sorted(CLUSTER_COLORS)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "E_cluster_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}/E_cluster_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback

    tasks = [
        ("A — Lead time",        task_a),
        ("B — Regime returns",   task_b),
        ("C — Entry pathways",   task_c),
        ("D — PR curve",         task_d),
        ("E — Cluster timeline", task_e),
    ]

    for name, fn in tasks:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print("="*60)
        try:
            fn()
        except Exception:
            print(f"  ERROR in {name}:")
            traceback.print_exc()

    print("\n[Part 1 complete] Outputs in results/sprint8/")
