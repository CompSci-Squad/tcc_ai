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

## Addendum (2026-04-29) — Thesis-defense lock-in (Q5 Tier 1+3, Q2 HPO)

Adopted **before** running any sweep on the ETL-v2 contract. Supersedes the
2026-04-17 plan where they conflict.

### 1. Frozen primary metric

- **Primary**: `hdbscan_train_dbcv` — DBCV (Density-Based Clustering
  Validation, Moulavi et al. 2014) computed via
  `hdbscan.HDBSCAN.relative_validity_` on the **TRAIN** non-overlapping
  embedding split.
- **Decision rule (single test, alpha = 0.05, no correction)**:
  - The headline pipeline (`iTransformer-AE → UMAP → HDBSCAN`) is
    declared *successful* iff `dbcv > 0` on TRAIN **and** the same
    HDBSCAN parameters yield `dbcv > 0` on TEST when refit with
    train-frozen `(min_cluster_size, min_samples)`.
  - Otherwise the result is reported as **null** ("the AE+UMAP latent
    space does not yield a denser-than-baseline regime structure on
    out-of-sample data") and we fall back to the baselines below.
- **Falsifiability statement**: a TEST DBCV ≤ 0 with a non-degenerate
  cluster count (≥ 2) is treated as evidence *against* the headline
  hypothesis. We commit to publishing this outcome.

### 2. Confirmatory metric panel (frozen, 7 numbers)

Reported for every cell of every sweep, on TEST, in this order:

1. `dbcv` — DBCV (primary).
2. `hdbscan_test_n_clusters` — model parsimony.
3. `hdbscan_test_noise_fraction` — coverage.
4. `nber_f1` — F1 vs. NBER USREC, mapping fit on **VAL**, frozen for
   TEST. See §3.
5. `nber_f1_legacy_maxF1` — legacy post-hoc-best F1, reported only as a
   bias witness; never used for decisions.
6. `bai_perron_f1` — alignment of cluster transitions with structural
   breaks on PC1 (tolerance = 2 months).
7. `crisis_window_coverage` — fraction of canonical crises
   (dotcom/GFC/COVID) whose dominant cluster matches the recession-mapped
   one from §3.

Any additional metrics computed (silhouette, Calinski-Harabasz,
Davies-Bouldin, ARI vs. baselines, …) are **exploratory** and reported
under Benjamini-Hochberg correction (`evaluation/statistical_tests.py`).

### 3. Cluster→regime mapping (Q5 Tier 1 — selection-bias fix)

The 2026-04-17 plan's NBER overlap selected the cluster with maximal F1
on TEST (post-hoc), inflating the score. As of 2026-04-29:

- `evaluation/regime_validation.fit_nber_assignment(val_labels, val_dates,
  usrec)` builds a frozen mapping `{cluster_id: 0|1}` on the VAL split
  (majority vote over the tolerance-expanded recession indicator, lead=0,
  lag=2).
- `evaluation/regime_validation.nber_overlap_frozen(...)` applies that
  mapping verbatim on TEST and is the **only** value reported as
  `nber_f1` from this date onward.
- The legacy `nber_overlap` is retained for backward compatibility but
  marked deprecated; its output appears in MLflow as
  `nber_f1_legacy_maxF1` for transparency.
- The mapping itself is logged as MLflow tag `nber_assignment`.

### 4. Two-stage HPO grid (Q2)

The current "sweep" is a 36-cell architectural grid `(W, d_lat, K)` and
performs no HPO over learning_rate / dropout / etc. Two-stage protocol:

- **Stage 1 (LR × dropout, 12 cells, local CPU)**: at the primary
  configuration `(W=12, d_lat=8)`, sweep
  `learning_rate ∈ {1e-4, 3e-4, 1e-3}` × `dropout ∈ {0.0, 0.1, 0.2, 0.3}`.
  Selection metric: **VAL** reconstruction MSE. Winner is the
  `(lr, dropout)` minimising VAL MSE; written into a frozen
  `configs/stage1_winner.yaml`.
- **Stage 2 (architectural, 36 cells, SageMaker)**: existing
  `configs/sweep/*.yaml`, but each config inherits the stage-1 winner
  values for `learning_rate` and `dropout`. Selection metric: **DBCV on
  TRAIN** (primary). Reported with the 7-metric panel.
- We do not sweep `d_model`, `n_heads`, `n_layers` or `batch_size` for
  this thesis; they are documented as fixed hyperparameters with the
  rationale "dataset-size-bound" (≤ 770 monthly rows).

### 5. Locked baselines (Q4)

All baselines consume the **same dates and splits** as the headline
pipeline and are evaluated with the same 7-metric panel:

1. `iTransformer-AE → UMAP → HDBSCAN` (headline; primary).
2. **Sticky HDP-HMM** (Fox–Sudderth–Jordan–Willsky, AOAS 2011) on the
   stationary panel directly; implemented via `dynamax`. K is
   non-parametric.
3. **SDHDP-HMM** (Song 2014, Toronto WP tecipa-427) on the same panel.
4. (Backlog) `Z → PCA → KMeans` and `Raw-PCA → KMeans` retained from the
   2026-04-17 plan as lower-bound floors; reported when present but not
   required for the thesis defense.

### 6. Data lineage (Q5 Tier 3)

Every training run logs the following as MLflow tags:

- `data_contract` — e.g. `etl_v2_balanced_2026_04`.
- `data_format` — `etl_v2_parquet` or `fred_md_csv`.
- `data_sha256` — SHA-256 of the panel parquet, **asserted** at load
  time (mismatch raises `ValueError`); see `data.preprocessing.load_etl_v2_panel`.
- `git_sha` — repository commit.
- `nber_assignment` — see §3.

Container image is pinned by digest (`@sha256:...`) at submission time
via `sm_jobs/launch_training.resolve_image_digest()`.

### 7. Limitations declared up-front (Q5 Tier 2 deferrals)

- **Vintage realism**: ETL imputes (EM-PCA) on the entire panel including
  TEST months. We document this as a limitation and reframe results as
  *regime characterization*, not *real-time early warning*. ALFRED
  vintages are out of scope.
- **Single seed**: we do not run N-seed Procrustes-aligned ARI for
  embedding stability. Documented limitation; one fixed seed (42) for
  reproducibility.
- **Effective sample size**: NBER F1 on TEST is computed from
  ~16 non-overlapping windows for `W=12`. We report bootstrap CIs where
  `n_eff ≥ 20`; otherwise point estimates only and tag the cell
  `analysis_type=exploratory`.

---

## Addendum (2026-04-29) — D7 imputation policy (loss masking + target-row eval filter)

Adopted **before** running the architectural sweep on the ETL-v2 contract.
Closes the gap between the academic plan (`pre_projeto_tcc.md` §4.2: "séries com
cobertura temporal incompleta serão tratadas por imputação ou exclusão") and the
implementation, where imputation was silently included in both the reconstruction
loss and the test-set evaluation.

### 1. Problem statement

The ETL panel `fred_md_transformed_balanced_2026_04.parquet` ships a companion
boolean mask `fred_md_mask_balanced_2026_04.parquet` flagging cells that were
imputed (EM-PCA tail/head fill, see `tcc_etl/src/tcc_etl/transform/imputation.py`).
Empirical imputation rates over the splits used in this thesis:

| Split | Months | Imputed cell rate | Rows with ≥1 imputed cell |
|-------|-------:|------------------:|--------------------------:|
| TRAIN | 718    | 0.07%             | very small                |
| VAL   | 36     | 1.02%             | moderate                  |
| TEST  | 50     | ~0.7% by cell     | **~100% of rows**         |

The TEST tail concentrates a small set of persistently late-publishing series
(e.g. `CP3M`, `CONSPI`, `COMPAPFFx`). Treating imputed cells as if they were
observed contaminates two surfaces:

1. **Training signal**: the AE is asked to reconstruct values that are themselves
   regression artefacts of the rest of the panel, biasing the encoder toward the
   imputer's linear structure.
2. **Held-out evaluation**: a strict "drop any window whose target row touches
   imputation" rule wipes ~100% of the TEST split (39/39 windows on the
   2026-04-29 smoke), making the principal evaluation impossible.

### 2. Policy decision (principal)

The two policy levers below are **active by default** for the headline pipeline
and apply uniformly to baselines (so the comparison stays fair). They are
orthogonal: (c) governs the loss, (a) governs which evaluation windows count.

- **(c) Masked reconstruction loss** — `loss_mask_imputed: true`.
  Implemented in `model/losses.masked_reconstruction_loss(x, x_hat, mask)`.
  Imputed cells are excluded from the MSE numerator and denominator (per-window).
  The training objective therefore measures reconstruction quality only on
  *observed* cells, which is the property the thesis claims (§4.3 Módulo 1).
  The `FREDMDWindowDataset(return_mask=True)` yields 3-tuples `(x, mask, idx)`;
  the trainer auto-routes to the masked loss when a 3-tuple batch is present.

- **(a) Target-row eval filter** — `eval_drop_imputed_target: true` with
  tolerance `eval_min_observed_fraction: 0.95`. A window is admitted into the
  embedding-export and downstream-evaluation set iff
  `mean(observed cells in target row) ≥ 0.95`, i.e. at most 5% of the 122
  features in the target month may be imputed. Rationale: the target row is the
  "regime label timestamp" carried into UMAP/HDBSCAN/NBER overlap; allowing a
  small late-publishing tail (1-2 series) preserves nearly all TEST windows
  (smoke result: 35/39 retained) without admitting rows that are mostly
  reconstructed.

The legacy flag `drop_imputed_windows` (binary, drops any window with any
imputed cell anywhere) is retained for backward compatibility and is **off by
default**; it is not used in any reported result.

### 3. Configuration surface (frozen)

`ExperimentConfig` (`src/tcc_itransformer/config.py`):

```yaml
# D7 imputation policy (principal: c + a)
loss_mask_imputed: true          # (c) masked MSE during train/val
eval_drop_imputed_target: true   # (a) drop test windows whose target row is mostly imputed
eval_min_observed_fraction: 0.95 # tolerance for (a); 1.0 = strict, 0.0 = disabled
```

These three flags are present in `configs/default.yaml` and
`configs/sagemaker_ae_only.yaml` and propagate to every cell of the architectural
sweep.

### 4. Robustness appendix (declared, not principal)

To bound the sensitivity of the headline result to this policy choice, the
sweep cell at the primary configuration `(W=12, d_lat=8, K=4)` will additionally
be re-run with:

- **(b) Unfiltered evaluation** — `eval_drop_imputed_target: false`,
  `loss_mask_imputed: true`. Reports the 7-metric panel on the full 50-month
  TEST split. Treated as a *bias witness*: divergence between (a) and (b)
  bounds the impact of imputation contamination on TEST. Reported in the
  thesis appendix; never used to overturn the principal decision in §1
  of the 2026-04-29 lock-in addendum.
- **Strict-target** — `eval_min_observed_fraction: 1.0`. Reported only if it
  retains `≥ 20` windows; otherwise tagged `analysis_type=exploratory` per the
  Q5 Tier 2 effective-sample-size rule.

### 5. Impact on the locked metric panel

- The primary metric `hdbscan_train_dbcv` is unchanged in definition; it is now
  computed on embeddings whose AE was trained under masked loss (the only
  change is a more honest training signal on TRAIN, where the imputation rate
  is 0.07%, so the numerical impact is small — but the methodological claim
  is now defensible).
- TEST-side metrics (`dbcv` refit, `nber_f1`, `bai_perron_f1`,
  `crisis_window_coverage`, `hdbscan_test_n_clusters`,
  `hdbscan_test_noise_fraction`) are computed on the **filtered** TEST set
  (policy (a), tolerance 0.95). Effective sample size reported alongside.
- VAL reconstruction MSE remains the Stage-1 HPO selection metric (§4 of the
  prior addendum), now computed under masked loss. **Note**: the smoke run
  surfaced VAL MSE ≈ 16 vs. TRAIN MSE ≈ 0.94, driven by COVID-March-2020
  outliers in the 2019–2021 VAL window when the scaler is fit on pre-COVID
  TRAIN. This is a separate issue (encoder robustness, not imputation), is
  documented under Limitations §7 of the prior addendum, and will be revisited
  before Stage-1 HPO is locked.

### 6. Falsifiability and pre-registration

Both flags and the tolerance are pre-registered here and frozen prior to any
sweep on the ETL-v2 contract. Changing them mid-sweep would constitute
researcher-degrees-of-freedom and is forbidden. Should the sweep produce a
null result under (c)+(a), the unfiltered (b) appendix is the only auxiliary
analysis admissible without a new pre-registration entry.

---

## Addendum 2026-04-30 — Train/Val/Test split locked (Option B)

The smoke-run anomaly noted in §5 of the 2026-04-29 D7 addendum (VAL MSE ≈ 16
vs. TRAIN MSE ≈ 0.94) was diagnosed and resolved before any HPO was launched.

### 1. Diagnosis

Under the previous split (`train_end=2018-12-01`, `val_end=2021-12-01`), the
VAL window 2019-01..2021-12 contained **March 2020**, which is an 8–15σ
outlier in industrial production, payrolls, and hours-worked under the
StandardScaler fit on pre-COVID TRAIN. Squared-error loss on those few
months dominated the VAL aggregate, yielding a 17× VAL/TRAIN gap that was
not a generalization signal but a single-event distortion of the metric.

### 2. Decision rule

Pre-registered: address the **cause** (COVID on the wrong split) before the
**symptom** (loss explodes). Loss-level mitigations (Huber, winsorization,
masked-outlier weights) were not adopted because they would impose a
methodological footnote ("we suppressed the largest macro shock in 75 years
on the very split used for model selection") that weakens the thesis.

### 3. Locked split (effective immediately)

| Split | Range | n months |
|---|---|---|
| TRAIN | 1965-01-01 → **2017-12-01** | 636 |
| VAL   | 2018-01-01 → **2019-12-01** | 24 |
| TEST  | 2020-01-01 → 2026-04-01     | 76 |

Codified at:
- `tcc_ai/src/tcc_itransformer/config.py` — `train_end="2017-12-01"`, `val_end="2019-12-01"` (defaults).
- `tcc_ai/configs/{default,sagemaker_ae_only}.yaml`.
- All 12 `configs/sweep_stage1/*.yaml` and 36 `configs/sweep/*.yaml` regenerated from the new defaults.

### 4. Validation (smoke run, primary architecture W=12, d_lat=8, K=4)

| Metric | Old split (val_end=2021) | New split (val_end=2019) |
|---|---|---|
| TRAIN MSE | 0.94 | 0.948 |
| VAL MSE   | ~16  | **0.631** |
| TEST MSE  | n/a  | 3.918 (vs. naive 4.040) |
| best_epoch / stopped | 36 / 46 | 36 / 46 |
| effective_rank | n/a | 6.63 / 8 |

VAL MSE collapsed by ~25×. TEST MSE > VAL MSE is expected and correct: TEST
contains COVID + post-COVID, the largest macro shock in the panel; the AE
still beats the naive mean-predictor on it.

### 5. Implications for the panel

- **Primary metric** unchanged: `hdbscan_train_dbcv`, decision rule
  `dbcv > 0` on TRAIN and TEST.
- **NBER mapping**: still fit on VAL, frozen on TEST. VAL is now a quiet
  late-expansion window (2018–2019, no NBER-dated recession) — the
  Hungarian assignment will lean on expansionary regimes; this is a
  property of the locked split, not a bug.
- **n_eff**: TEST contains 76 months but the COVID shock concentrates
  effective independent observations near `n_eff_test ≈ 4` per the smoke
  run's clustering-free metrics. BCa CIs on TEST will be tagged
  `analysis_type=exploratory` whenever `n_eff < 20`, per the Q5 Tier 2 rule.
- **Stage-1 HPO selection metric** remains VAL MSE under masked loss (D7.c).
  It is now well-defined; no escalation to VAL-DBCV needed.

### 6. Falsifiability

The split is frozen prior to any sweep. Changing `train_end` or `val_end`
post-hoc — including to "include COVID in VAL for robustness" — constitutes
researcher-degrees-of-freedom and is forbidden. A robustness appendix may
report a single re-run with `val_end=2021-12-01` for transparency, clearly
labelled as such.
