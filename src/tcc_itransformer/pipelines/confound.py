"""A1 confound check: is the cluster assignment driven by NBER, pre-2008, or |dINDPRO|?

Refits PCA(2)+HDBSCAN on the full panel (train+val+test embeddings) and runs
chi-square independence tests against three potential confounders. Also refits
HDBSCAN after dropping 2020-Q2 (Apr/May/Jun) and reports the ARI of the new
labels against the original labels on the remaining months.

Decision rule (panel-remediation-plan): if cluster<->pre-2008 dominates
(p < 0.01 and Cramer's V > 0.4), much of the writeup needs rethinking.
"""

from __future__ import annotations

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
    parts = [
        pd.read_parquet(emb_dir / f"Z_{split}.parquet")
        for split in ("train", "val", "test")
    ]
    full = pd.concat(parts, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)
    z_cols = [c for c in full.columns if c.startswith("z_")]
    return full[z_cols].to_numpy(), pd.DatetimeIndex(full["date"])


def _chi2(labels: np.ndarray, group: np.ndarray, name: str) -> dict:
    mask = ~pd.isna(group) & (labels != -1)
    if mask.sum() < 10:
        return {"name": name, "n": int(mask.sum()),
                "chi2": np.nan, "p": np.nan, "dof": 0}
    tab = pd.crosstab(
        pd.Series(labels[mask], name="cluster"),
        pd.Series(group[mask], name=name),
    )
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"name": name, "n": int(mask.sum()),
                "chi2": np.nan, "p": np.nan, "dof": 0,
                "table": tab.to_string()}
    chi2, p, dof, _ = chi2_contingency(tab.values)
    n = tab.values.sum()
    r, c = tab.shape
    cramers_v = float(np.sqrt(chi2 / (n * max(min(r - 1, c - 1), 1))))
    return {"name": name, "n": int(mask.sum()), "chi2": float(chi2),
            "p": float(p), "dof": int(dof),
            "cramers_v": cramers_v, "table": tab.to_string()}


def _indpro_dquart(dates: pd.DatetimeIndex, panel_path: Path) -> np.ndarray:
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    aligned = panel.set_index("date")["INDPRO"].reindex(dates)
    delta = aligned.diff().abs()
    return pd.qcut(delta, 4, labels=False, duplicates="drop").to_numpy()


def run_confound_check(
    embeddings_dir: Path,
    usrec_csv: Path,
    *,
    panel_parquet: Path = Path(
        "data/raw/fred_md_transformed_balanced_2026_04.parquet",
    ),
    output: Path = Path("results/diagnostics/confound_check.md"),
    n_pcs: int = 2,
    seed: int = 42,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    Z, dates = _load_embeddings(embeddings_dir)
    logger.info(
        "Full panel: n=%d  range=%s..%s",
        len(Z), dates[0].date(), dates[-1].date(),
    )

    pca = PCA(n_components=n_pcs, random_state=seed)
    X2 = pca.fit_transform(Z)
    res, _ = optimize_hdbscan_dbcv(X2)
    labels = res.labels
    logger.info(
        "HDBSCAN full: k=%d noise=%.3f DBCV=%.4f mcs=%d ms=%d",
        res.n_clusters, res.noise_fraction, res.dbcv,
        res.min_cluster_size, res.min_samples,
    )

    usrec = load_usrec(usrec_csv)
    nber = usrec.reindex(dates).fillna(0).astype(int).to_numpy()
    pre2008 = (dates < pd.Timestamp("2008-01-01")).astype(int)
    indpro_q = _indpro_dquart(dates, panel_parquet)

    results = [
        _chi2(labels, nber, "cluster x NBER"),
        _chi2(labels, pre2008, "cluster x pre-2008"),
        _chi2(labels, indpro_q, "cluster x |dINDPRO| quartile"),
    ]

    drop_mask = ~((dates.year == 2020) & (dates.month.isin([4, 5, 6])))
    Z_drop = Z[drop_mask]
    pca_drop = PCA(n_components=n_pcs, random_state=seed)
    X2_drop = pca_drop.fit_transform(Z_drop)
    res_drop, _ = optimize_hdbscan_dbcv(X2_drop)
    overlap_idx = np.where(drop_mask)[0]
    ari = adjusted_rand_score(labels[overlap_idx], res_drop.labels)

    lines = [
        "# A1 Confound check (full-panel refit)",
        "",
        f"- Embeddings: `{embeddings_dir}`",
        f"- n_months={len(Z)}  range={dates[0].date()}..{dates[-1].date()}",
        f"- PCA -> {n_pcs}D, HDBSCAN selected: k={res.n_clusters}, "
        f"noise={res.noise_fraction:.3f}, DBCV={res.dbcv:.4f}, "
        f"mcs={res.min_cluster_size}, ms={res.min_samples}",
        "",
        "## Chi-square independence tests",
        "",
        "| Confounder | n | dof | chi2 | p | Cramer's V |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        v = r.get("cramers_v", float("nan"))
        lines.append(
            f"| {r['name']} | {r['n']} | {r['dof']} | {r['chi2']:.3f} | "
            f"{r['p']:.4g} | {v:.3f} |"
        )
    lines += ["", "### Contingency tables", ""]
    for r in results:
        if "table" in r:
            lines += [f"**{r['name']}**", "", "```", r["table"], "```", ""]

    lines += [
        "## Stability without 2020-Q2",
        "",
        f"- Refit HDBSCAN on n={drop_mask.sum()} (dropped Apr/May/Jun 2020).",
        f"- Refit selected: k={res_drop.n_clusters}, "
        f"noise={res_drop.noise_fraction:.3f}, DBCV={res_drop.dbcv:.4f}, "
        f"mcs={res_drop.min_cluster_size}, ms={res_drop.min_samples}",
        f"- ARI(full vs no-2020Q2) = **{ari:.4f}** "
        f"(1.0 = identical, 0.0 = random).",
        "",
        "## Decision rule",
        "",
        "Per `panel-remediation-plan.md`: if cluster <-> pre-2008 dominates "
        "(p < 0.01 and Cramer's V > 0.4), much of the writeup needs rethinking.",
    ]

    pre2008_r = next(r for r in results if r["name"] == "cluster x pre-2008")
    triggered = (
        pre2008_r["p"] < 0.01 and pre2008_r.get("cramers_v", 0) > 0.4
    )
    verdict = (
        "**TRIGGERED** -- pre-2008 confound dominates cluster assignment."
        if triggered
        else "**Not triggered** -- pre-2008 not a dominating confound."
    )
    lines.insert(2, f"\n## Verdict\n\n{verdict}\n")

    output.write_text("\n".join(lines))
    logger.info("Wrote %s", output)
    print("\n".join(lines))
    return output
