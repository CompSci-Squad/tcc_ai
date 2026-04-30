"""A1 confound check for the W6_d7_K4 winner.

Refits PCA(7)+HDBSCAN on the FULL panel (train+val+test embeddings) and runs
chi-square independence tests of cluster assignment against three potential
confounders:

  1. NBER recession label (USREC).
  2. Pre-2008 binary flag (date < 2008-01-01).
  3. |delta INDPRO| quartile bin.

Also refits HDBSCAN after dropping 2020-Q2 (Apr/May/Jun 2020) and reports the
adjusted Rand index of the new labels against the original labels on the
remaining months.

Output: results/diagnostics/confound_check.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from tcc_itransformer.data.external_labels import load_usrec
from tcc_itransformer.evaluation.density_clustering import optimize_hdbscan_dbcv

logger = logging.getLogger(__name__)


def _load_embeddings(emb_dir: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(emb_dir / f"Z_{split}.parquet")
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)
    z_cols = [c for c in full.columns if c.startswith("z_")]
    return full[z_cols].to_numpy(), pd.DatetimeIndex(full["date"])


def _chi2(labels: np.ndarray, group: np.ndarray, name: str) -> dict:
    mask = ~pd.isna(group) & (labels != -1)
    if mask.sum() < 10:
        return {"name": name, "n": int(mask.sum()), "chi2": np.nan, "p": np.nan, "dof": 0}
    tab = pd.crosstab(pd.Series(labels[mask], name="cluster"), pd.Series(group[mask], name=name))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"name": name, "n": int(mask.sum()), "chi2": np.nan, "p": np.nan,
                "dof": 0, "table": tab.to_string()}
    chi2, p, dof, _ = chi2_contingency(tab.values)
    n = tab.values.sum()
    r, c = tab.shape
    cramers_v = float(np.sqrt(chi2 / (n * max(min(r - 1, c - 1), 1))))
    return {"name": name, "n": int(mask.sum()), "chi2": float(chi2), "p": float(p),
            "dof": int(dof), "cramers_v": cramers_v, "table": tab.to_string()}


def _indpro_dquart(dates: pd.DatetimeIndex, panel_path: Path) -> np.ndarray:
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index("date")["INDPRO"]
    aligned = panel.reindex(dates)
    delta = aligned.diff().abs()
    quart = pd.qcut(delta, 4, labels=False, duplicates="drop")
    return quart.to_numpy()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--usrec-csv", required=True)
    p.add_argument("--panel-parquet", default="data/raw/fred_md_transformed_balanced_2026_04.parquet")
    p.add_argument("--output", default="results/diagnostics/confound_check.md")
    p.add_argument("--n-pcs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    emb_dir = Path(args.embeddings_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Z, dates = _load_embeddings(emb_dir)
    logger.info("Full panel: n=%d  range=%s..%s", len(Z), dates[0].date(), dates[-1].date())

    pca = PCA(n_components=args.n_pcs, random_state=args.seed)
    X2 = pca.fit_transform(Z)
    res, _ = optimize_hdbscan_dbcv(X2)
    labels = res.labels
    logger.info("HDBSCAN full: k=%d noise=%.3f DBCV=%.4f mcs=%d ms=%d",
                res.n_clusters, res.noise_fraction, res.dbcv,
                res.min_cluster_size, res.min_samples)

    # Confounders.
    usrec = load_usrec(Path(args.usrec_csv))
    nber = usrec.reindex(dates).fillna(0).astype(int).to_numpy()
    pre2008 = (dates < pd.Timestamp("2008-01-01")).astype(int)
    indpro_q = _indpro_dquart(dates, Path(args.panel_parquet))

    results = [
        _chi2(labels, nber, "cluster x NBER"),
        _chi2(labels, pre2008, "cluster x pre-2008"),
        _chi2(labels, indpro_q, "cluster x |dINDPRO| quartile"),
    ]

    # Drop 2020-Q2 and refit.
    drop_mask = ~((dates.year == 2020) & (dates.month.isin([4, 5, 6])))
    Z_drop = Z[drop_mask]
    pca_drop = PCA(n_components=args.n_pcs, random_state=args.seed)
    X2_drop = pca_drop.fit_transform(Z_drop)
    res_drop, _ = optimize_hdbscan_dbcv(X2_drop)
    labels_drop = res_drop.labels
    overlap_idx = np.where(drop_mask)[0]
    ari = adjusted_rand_score(labels[overlap_idx], labels_drop)

    lines = [
        "# A1 Confound check (W6_d7_K4 full-panel refit)",
        "",
        f"- Embeddings: `{emb_dir}`",
        f"- n_months={len(Z)}  range={dates[0].date()}..{dates[-1].date()}",
        f"- PCA -> {args.n_pcs}D, HDBSCAN selected: k={res.n_clusters}, noise={res.noise_fraction:.3f}, "
        f"DBCV={res.dbcv:.4f}, mcs={res.min_cluster_size}, ms={res.min_samples}",
        "",
        "## Chi-square independence tests",
        "",
        "| Confounder | n | dof | chi2 | p | Cramer's V |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        v = r.get("cramers_v", float("nan"))
        lines.append(f"| {r['name']} | {r['n']} | {r['dof']} | {r['chi2']:.3f} | {r['p']:.4g} | {v:.3f} |")
    lines += ["", "### Contingency tables", ""]
    for r in results:
        if "table" in r:
            lines += [f"**{r['name']}**", "", "```", r["table"], "```", ""]

    lines += [
        "## Stability without 2020-Q2",
        "",
        f"- Refit HDBSCAN on n={drop_mask.sum()} (dropped Apr/May/Jun 2020).",
        f"- Refit selected: k={res_drop.n_clusters}, noise={res_drop.noise_fraction:.3f}, "
        f"DBCV={res_drop.dbcv:.4f}, mcs={res_drop.min_cluster_size}, ms={res_drop.min_samples}",
        f"- ARI(full vs no-2020Q2) = **{ari:.4f}** (1.0 = identical, 0.0 = random).",
        "",
        "## Decision rule",
        "",
        "Per `panel-remediation-plan.md`: if cluster <-> pre-2008 dominates "
        "(p < 0.01 and Cramer's V > 0.4), much of the writeup needs rethinking.",
    ]

    pre2008_r = next(r for r in results if r["name"] == "cluster x pre-2008")
    triggered = (pre2008_r["p"] < 0.01) and (pre2008_r.get("cramers_v", 0) > 0.4)
    verdict = ("**TRIGGERED** -- pre-2008 confound dominates cluster assignment."
               if triggered else "**Not triggered** -- pre-2008 not a dominating confound.")
    lines.insert(2, f"\n## Verdict\n\n{verdict}\n")

    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
