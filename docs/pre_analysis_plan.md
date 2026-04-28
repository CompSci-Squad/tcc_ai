# Pre-Analysis Plan — iTransformer Embedding Evaluation
## Date: 2026-04-17
## Author: TCC Team
## Instituto Mauá de Tecnologia — TCC 2026

### Primary Hypothesis
H1: The iTransformer autoencoder produces latent embeddings whose K-Means
clustering yields higher silhouette scores than PCA applied directly to
the raw windowed macroeconomic features (Baseline B1).

### Primary Configuration
- Window: W=12 (one year of monthly data — economically motivated)
- Latent dim: d_latent=8 (middle of range, avoids boundary effects)
- d_model: 64 (constrained by dataset size)
- K: Selected by argmax silhouette on TRAIN non-overlapping set, K∈{3,4,5}
- PCA: Adaptive, ≥90% variance explained, max = d_latent−1 components
- Seed: 42

### Primary Test
- Metric: Silhouette score on TEST non-overlapping set
- Test: Permutation test (10,000 permutations), one-sided
- H0: Silhouette(iTransformer) ≤ Silhouette(B1: Raw PCA)
- H1: Silhouette(iTransformer) > Silhouette(B1: Raw PCA)
- Alpha: 0.05 (no correction — single pre-registered test)

### Secondary Analyses (exploratory, BH-corrected)
- All other 35 (W, d, K) combinations
- KW tests per dimension for each config
- Pairwise Mann-Whitney comparisons
- Baselines B0 (Random), B2 (Linear AE), B3 (Windowed PCA)
- Alpha for secondary: 0.05 with Benjamini-Hochberg correction

### Statistical Power Note
- Effective test set sizes: W=6 → n≈8, W=12 → n≈4, W=24 → n≈2
- W=24 results are descriptive only — insufficient power for inference
- Block bootstrap CIs reported only where viable (W=6 train)

### Operational Definition: Economic Regime
A contiguous temporal interval during which macroeconomic indicators exhibit
statistically homogeneous dynamics, as measured by cluster membership stability
over consecutive non-overlapping windows. Regimes are validated qualitatively
against NBER recession dates and ICSS structural breakpoints.

### Success / Failure Criteria

| Criterion | Threshold | Action if failed |
|-----------|-----------|-----------------|
| Silhouette (primary config, test) | > 0 | Report as null finding |
| Permutation test p-value | < 0.05 | Report as null finding |
| Effective rank of embeddings | > 2.0 | Investigate collapse |
| PCA variance explained (train) | ≥ 90% | Increase n_pca_max |
| Clustering stability ARI | > 0.7 | Report instability |
| KW significant dims (≥ d/2) | BH p < 0.05 | Report as weak separation |

### Quality Gates
Before reporting any configuration's results:
1. Embedding collapse check: per-dimension variance > 1e-4
2. Reconstruction loss beats naive baseline (mean-prediction MSE)
3. Effective sample size ≥ 20 for bootstrap CIs; otherwise report point estimate only
4. W=24 analyses always tagged `analysis_type=exploratory` with power warning

### Data Exclusion Criteria
- Series with > 10% missing values after stationarity transforms are excluded
- Windows containing any NaN after preprocessing are dropped
- No post-hoc exclusion of configurations with unfavorable results

### Reporting Commitment
All results are reported, including null findings and failed quality gates.
Negative results are given equal space as positive results.

### Sample Size Justification
The FRED-MD dataset provides ~770 monthly observations (1959–2024). After
train/val/test split and non-overlapping windowing:
- W=6 train: ~70 non-overlapping windows
- W=12 train: ~35 non-overlapping windows
- W=24 train: ~17 non-overlapping windows

Power for non-parametric tests is limited, especially for W=24. We acknowledge
this limitation explicitly and use block bootstrap where feasible.

---

## Addendum (2026-04-28) — Principal pipeline alignment with pre_projeto_tcc.md

After reviewing `docs/pre_projeto_tcc.md §4.3` against the originally-registered
plan, the following changes were adopted **before** any test-set evaluation
was performed (no peeking):

### Option A — Principal pipeline = UMAP + HDBSCAN

The pre_projeto specifies UMAP for dimensionality reduction and HDBSCAN for
density-based clustering as the **principal** path; PCA + K-Means is retained
as a **baseline** for the permutation test. Rationale:

- HDBSCAN does not require K to be pre-specified and produces a noise label
  for windows that do not belong to any cluster — this matches the
  operational definition of "regime" (contiguous homogeneous interval).
- UMAP preserves topological structure better than PCA in low-dimensional
  embeddings, per the literature cited in the pre_projeto.
- DBCV (Density-Based Clustering Validation) is the registered evaluation
  metric for HDBSCAN; we use `hdbscan.HDBSCAN.relative_validity_` as the
  standard library proxy.

### Updated success criteria (additive)

| Criterion | Threshold | Source |
|---|---|---|
| HDBSCAN noise fraction (TRAIN) | ≤ 0.4 | pre_projeto §4.3 |
| HDBSCAN cluster count (TRAIN) | ≥ 2 | sanity |
| DBCV (`relative_validity_`) | > 0 | pre_projeto §4.3 |
| NBER overlap F1 (lead=0, lag=2) | reported | pre_projeto §4.4 |
| Crisis windows covered | ≥ 3 of {dotcom, GFC, COVID} | pre_projeto §4.4 |

### Validation modules (pre_projeto §4.4)

All implemented in `evaluation/regime_validation.py`:

- NBER USREC overlap (precision/recall/F1 with lead/lag)
- Bai–Perron break alignment via `ruptures.Pelt` / `Dynp`
- Conditional moments per regime (mean, std, skew, kurt by series)
- Markov transition matrix (excluding noise)
- Regime durations (n_runs, mean/median/max)
- Module 4 (`evaluation/explain.py`): `{regime, soft_membership, top_features}`

### Stationarity (pre_projeto §4.2)

Joint **ADF + KPSS** rule (`data/stationarity.py`):
- Stationary iff ADF rejects unit-root **and** KPSS fails to reject stationarity.

### Statistical CIs

Bootstrap CIs use the **BCa** method (`evaluation/statistical_tests.py:_bootstrap_ci`):
z₀ from empirical mass + jackknife acceleration. The original plan's
"percentile bootstrap" wording is superseded.

### Reproducibility infrastructure

All experiments are reproducible via:

```bash
make sm-build && make sm-push   # build + push training image to ECR
make sm-train                   # 1 SageMaker job (default config)
make sm-sweep                   # 36 jobs (configs/sweep/*.yaml)
```

Outputs are persisted under `s3://tcc-regime-etl-sagemaker/jobs/<job>/output/`
and (optionally) logged to a SageMaker-managed MLflow tracking server
(`var.enable_mlflow=true` in `tcc_iac/infra/mlflow.tf`).

---
