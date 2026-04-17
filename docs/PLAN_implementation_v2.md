# iTransformer Embedding Repo — Detailed Implementation Plan v2

---
goal: "Implement a scientifically rigorous, reproducible iTransformer autoencoder pipeline for FRED-MD economic regime identification"
version: 2.0
date_created: 2026-04-17
last_updated: 2026-04-17
status: 'Planned'
tags: [feature, architecture, ml-pipeline, scientific-experiment]
---

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan was generated through a **6-agent structured debate** (see Appendix A) and supersedes the original `PLAN_itransformer_repo.md` with corrections to statistical methodology, phasing, and infrastructure.

---

## Table of Contents

1. [Multi-Agent Debate Summary](#1-multi-agent-debate-summary)
2. [Requirements & Constraints](#2-requirements--constraints)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Repository Structure](#4-repository-structure)
5. [Implementation Phases](#5-implementation-phases)
6. [Statistical Testing Framework](#6-statistical-testing-framework)
7. [Test Organization & Trackability](#7-test-organization--trackability)
8. [Sweep Strategy](#8-sweep-strategy)
9. [Thesis Artifact Generation](#9-thesis-artifact-generation)
10. [Risk Mitigation Matrix](#10-risk-mitigation-matrix)
11. [Appendix A: Agent Debate Record](#appendix-a-agent-debate-record)
12. [Appendix B: Pre-Analysis Plan Template](#appendix-b-pre-analysis-plan-template)

---

## 1. Multi-Agent Debate Summary

Six specialized agents debated the implementation strategy. Their roles, key arguments, and resolutions:

| # | Agent | Role | Key Contribution | Accepted? |
|---|-------|------|-------------------|-----------|
| 1 | **Plan Mode** (Primary Designer) | Architecture & phasing | MLflow local, Pydantic+YAML, tracer bullet phasing | ✅ All accepted |
| 2 | **Critical Thinking** (Skeptic) | Failure mode analysis | Overlapping windows destroy stats, FRED-MD shrinks, AE may collapse/memorize | ✅ All risks incorporated |
| 3 | **Context Architect** (Constraint Guardian) | Non-functional constraints | `uv` + pinned deps, data versioning, GPU determinism, CI/CD | ✅ All accepted |
| 4 | **Scientific Paper Research** (Rigor Enforcer) | Statistical framework | Hierarchical testing, block bootstrap, 4 baselines, adaptive PCA, pre-registration | ✅ Mostly accepted (BY→BH corrected) |
| 5 | **Implementation Plan** (Feasibility Analyst) | Task atomicity | 12 training runs not 36, parallel phase opportunities | ✅ Accepted |
| 6 | **Doublecheck** (Verification) | Fact-checking all claims | Corrected BY→BH, verified n_eff formula, confirmed W=24 issue | ✅ All corrections applied |

### Key Corrections from Debate

| Original Plan | Correction | Agent Source |
|--------------|------------|-------------|
| All metrics on overlapping windows | Evaluate ONLY on non-overlapping windows | Agent 2 + 4 |
| Fixed PCA components = 6 | Adaptive PCA: min(latent_dim−1, components_for_90%_var) | Agent 4 + 6 |
| BH correction only | BH (not BY) — justified via PRDS condition | Agent 6 overruled Agent 4 |
| 36 training runs | **12 training runs** — K is post-hoc clustering, not model training | Agent 5 + 6 |
| requirements.txt with `>=` | `uv` + `pyproject.toml` with exact pins + committed `uv.lock` | Agent 3 |
| No baselines | **4 mandatory baselines**: Random, Raw PCA, Linear AE, PCA on windowed features | Agent 4 |
| No data versioning | Committed FRED-MD snapshot with SHA-256 verification | Agent 3 |
| No NaN handling strategy | Explicit NaN strategy: drop series with >10% missing, forward-fill remainder | Agent 2 |
| W=24 treated as equal to W=6,12 | **W=24 is exploratory only** — n_eff < 2 on val/test | Agent 4 + 6 + 7 |
| No pre-registration | Pre-analysis plan committed before experiments | Agent 4 |
| No regime definition | Operational definition required before clustering | Agent 2 |

---

## 2. Requirements & Constraints

### Functional Requirements

- **REQ-001**: Implement iTransformer-based autoencoder (variate-as-token inversion) in PyTorch
- **REQ-002**: Load and transform FRED-MD monthly dataset (tcode transformations 1–7)
- **REQ-003**: Produce latent embeddings via bottleneck layer, apply PCA + K-Means for regime clustering
- **REQ-004**: Run hyperparameter sweep: W∈{6,12,24} × d_latent∈{6,7,8,9} × K∈{3,4,5}
- **REQ-005**: Track all experiment results with MLflow local — queryable, comparable, thesis-defense-ready
- **REQ-006**: Statistical validation pipeline: KW tests, permutation tests, bootstrap CIs, effect sizes
- **REQ-007**: 4 baseline comparisons to justify iTransformer over simpler methods
- **REQ-008**: All results reproducible from a single `make sweep` command

### Non-Functional Constraints

- **CON-001**: FRED-MD dataset is ~840×128 (≈120–127 usable series after cleaning)
- **CON-002**: Training samples after windowing: ~330 overlapping (W=6), effective n ≈ T/W
- **CON-003**: Single university GPU (assume RTX 3060, 6GB VRAM), ~8–15 min per training run
- **CON-004**: Must be self-contained — no cloud dependencies for reproducibility
- **CON-005**: Professor must be able to clone repo, run `uv sync && make test`, see results
- **CON-006**: Thesis committee will challenge statistical validity — all claims must be pre-registered or labeled exploratory

### Security & Data Integrity

- **SEC-001**: No API keys in code — FRED-MD is public data, no auth required
- **SEC-002**: Data snapshot integrity verified via SHA-256 hash at load time
- **SEC-003**: No pickle deserialization of untrusted data — use safetensors or torch.save with weights_only=True

### Guidelines & Patterns

- **GUD-001**: All statistical claims on non-overlapping evaluation subsets only
- **GUD-002**: Every p-value accompanied by effect size and confidence interval
- **GUD-003**: Primary analysis pre-registered; all other configs labeled "exploratory"
- **GUD-004**: matplotlib + seaborn only (no Plotly) for thesis figures
- **GUD-005**: Notebooks for exploration only — all production code in `src/`
- **PAT-001**: Pydantic BaseModel for all configs with validation
- **PAT-002**: Seed management via single `set_global_seed()` function called once at entry point
- **PAT-003**: No circular imports — dependency flow: data → model → training → analysis → tracking

---

## 3. Architecture Decisions

### 3.1 Experiment Tracking: MLflow Local

**Decision**: MLflow with local file-backed tracking (`file:./results/mlruns`)

**Justification** (Agent 1 proposal, Agent 6 verified):
- Zero external dependencies — professor clones, runs `mlflow ui`, sees everything
- Handles 36 evaluations (12 training runs × 3 K values) trivially
- Programmatic access: `mlflow.search_runs()` → DataFrame for statistical analysis
- Auto-logs git commit hash per run for traceability
- `mlflow ui` provides a professional dashboard for thesis defense

**Rejected alternatives**:
| Tool | Reason |
|------|--------|
| W&B | Cloud dependency, requires account — friction for thesis review (Agent 1) |
| CSV + Git tags | Doesn't handle nested metrics, artifacts, or configs well (Agent 1) |
| Hydra logging | Only stores configs/outputs, no comparison UI (Agent 1) |

### 3.2 Config Management: Pydantic + YAML

**Decision**: Pydantic BaseModel with YAML serialization

**Justification** (Agent 1 proposal, Agent 6 verified):
- Validation at load time (typo in `d_modle: 64` → immediate error)
- IDE autocomplete and type checking
- `config.model_dump()` → direct MLflow param logging
- 36 YAML files generated from grid script — human-readable, git-diffable

### 3.3 Dependency Management: uv + pyproject.toml

**Decision**: `uv` package manager with exact pins in `pyproject.toml` and committed `uv.lock`

**Justification** (Agent 3):
- Faster than pip/poetry, no resolver conflicts with PyTorch
- `uv.lock` is the reproducibility anchor — ensures exact same packages
- `uv sync --frozen` in CI fails on drift

### 3.4 Statistical Framework: Hierarchical Testing with BH-FDR

**Decision**: Pre-registered primary analysis (single config, single test, α=0.05) + BH-corrected exploratory analyses

**Justification** (Agent 4 proposed BY, Agent 6 corrected to BH):
- BH is valid under PRDS (positive regression dependency), which holds here
- BY would be ~4× more conservative, destroying power on an already power-starved study
- Hierarchical structure separates primary (1 test) from exploratory (≤35 tests)

### 3.5 Adaptive PCA

**Decision**: `n_components = min(latent_dim − 1, components_for_90%_variance, 5)`

**Justification** (Agent 4 proposed, Agent 6 verified):
- When `latent_dim = PCA components`, PCA is a rotation — a no-op for K-Means
- Adaptive selection ensures PCA always does real dimensionality reduction
- Cap at `latent_dim − 1` guarantees at least 1 dimension is removed
- Fit on train non-overlapping embeddings only

---

## 4. Repository Structure

```
tcc_ai/
├── pyproject.toml                        # Package metadata + exact dependency pins
├── uv.lock                               # COMMITTED — reproducibility anchor
├── Makefile                              # make train, make sweep, make test, make ui
├── .python-version                       # Pinned CPython (e.g., 3.11.9)
├── .gitignore
│
├── data/
│   ├── raw/                              # gitignored — downloaded files
│   │   └── .gitkeep
│   ├── processed/                        # gitignored — post-transform
│   │   └── .gitkeep
│   └── snapshots/                        # COMMITTED — immutable thesis data
│       ├── fred_md_2026_04.csv           # Raw FRED-MD as downloaded
│       └── fred_md_2026_04.sha256        # SHA-256 hash for integrity check
│
├── configs/
│   ├── default.yaml                      # Base config with defaults
│   └── sweep/                            # 36 generated YAML files (one per eval combo)
│       ├── W6_d6_K3.yaml
│       ├── W6_d6_K4.yaml
│       └── ...                           # (36 files total)
│
├── src/
│   └── tcc_itransformer/
│       ├── __init__.py
│       ├── seed.py                       # Single source of truth for all RNG seeds
│       ├── config.py                     # Pydantic ExperimentConfig
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fred_md.py                # FRED-MD download, tcode, outlier removal
│       │   ├── preprocessing.py          # StandardScaler, NaN strategy
│       │   └── dataset.py                # WindowedDataset (PyTorch Dataset)
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── layers.py                 # VariateEmbedding, TransformerEncoderBlock
│       │   ├── encoder.py                # iTransformerEncoder
│       │   ├── decoder.py                # Mirror decoder
│       │   ├── autoencoder.py            # Full AE: encode(x)→z, forward(x)→(x_hat, z)
│       │   └── losses.py                 # MSE reconstruction + naive baseline
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py                # Training loop, early stopping, checkpoints
│       │   └── callbacks.py              # EarlyStopping, LRScheduler, MLflowLogger
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── embedding_quality.py      # Reconstruction, geometry checks, isotropy
│       │   ├── clustering.py             # Adaptive PCA, K-Means, intrinsic metrics
│       │   ├── statistical_tests.py      # KW, Mann-Whitney, permutation, block bootstrap
│       │   ├── baselines.py              # 4 baseline methods: Random, Raw PCA, Linear AE, Windowed PCA
│       │   └── effective_sample_size.py  # n_eff computation, autocorrelation analysis
│       │
│       ├── tracking/
│       │   ├── __init__.py
│       │   └── mlflow_utils.py           # MLflow wrappers (no model/data imports)
│       │
│       └── utils/
│           ├── __init__.py
│           └── viz.py                    # matplotlib/seaborn plotting functions
│
├── scripts/
│   ├── download_data.py                  # One-shot: download FRED-MD → data/snapshots/
│   ├── generate_sweep_configs.py         # Generate 36 YAML files from grid
│   ├── run_single.py                     # Run one config end-to-end
│   ├── run_sweep.py                      # Run all 12 training + 36 evaluations
│   ├── run_baselines.py                  # Run all 4 baselines
│   ├── export_results.py                 # MLflow → LaTeX tables + figures
│   └── log_environment.py               # Capture hardware/software provenance
│
├── tests/
│   ├── conftest.py                       # Shared fixtures, GPU skip marker, mock data
│   ├── unit/
│   │   ├── test_data.py                  # FRED-MD parsing, tcode, outlier removal
│   │   ├── test_preprocessing.py         # Scaler, NaN handling, window creation
│   │   ├── test_model.py                 # Forward pass shapes, grads, param count
│   │   ├── test_losses.py                # MSE computation, baseline loss
│   │   ├── test_clustering.py            # Adaptive PCA, K-Means, metric computation
│   │   ├── test_statistical.py           # KW, permutation, bootstrap, BH correction
│   │   └── test_baselines.py             # All 4 baselines produce valid output
│   ├── integration/
│   │   ├── test_training.py              # Loss decreases, early stopping, checkpoint
│   │   └── test_pipeline_e2e.py          # Data → model → embeddings → PCA → K-Means
│   └── quality/
│       ├── test_embedding_quality.py     # Reconstruction < baseline, no collapse, rank > 2
│       └── test_clustering_quality.py    # Silhouette > 0, KW significant, stability > 0.7
│
├── notebooks/
│   ├── 00_eda.ipynb                      # Full EDA of FRED-MD data
│   └── 01_embedding_analysis.ipynb       # Post-training embedding quality + regime viz
│
├── results/
│   ├── mlruns/                           # MLflow tracking (partially gitignored)
│   └── figures/                          # Generated thesis figures
│
└── docs/
    ├── skills-index.md
    ├── PLAN_itransformer_repo.md         # Original plan (v1)
    ├── PLAN_implementation_v2.md         # This file
    ├── pre_analysis_plan.md              # Pre-registration (committed BEFORE experiments)
    └── environment.json                  # Hardware/software provenance (auto-generated)
```

### Module Dependency Rules (No Circular Imports)

```
data/ ──→ model/ ──→ training/ ──→ evaluation/ ──→ tracking/
  │                      │              │
  └──── config.py ◄──────┘              │
  └──── seed.py ◄────────────────────────┘
```

- `model/` NEVER imports `data/` — receives tensors, not datasets
- `training/` NEVER imports `evaluation/` — embeddings extracted after training
- `tracking/` NEVER imports `model/` or `training/` — receives dicts of metrics
- `evaluation/` NEVER imports `training/` — receives frozen embeddings

---

## 5. Implementation Phases

### Phase 0: Foundation (Est: 1 day)

**Goal**: Project skeleton, dependencies, seed management, config system, data snapshot.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-001 | Initialize project with `uv init`, create `pyproject.toml` with pinned deps | `pyproject.toml`, `uv.lock` | None | S | `uv sync` succeeds, all deps installed |
| TASK-002 | Create Makefile with targets: `train`, `sweep`, `test`, `lint`, `ui` | `Makefile` | None | S | `make test` runs (even if 0 tests) |
| TASK-003 | Create `.gitignore` (data/raw, data/processed, results/mlruns/*/artifacts, __pycache__) | `.gitignore` | None | S | Git status clean after setup |
| TASK-004 | Implement `src/tcc_itransformer/seed.py` — `set_global_seed()` with torch deterministic mode | `src/tcc_itransformer/seed.py` | TASK-001 | S | Import succeeds, `CUBLAS_WORKSPACE_CONFIG` set |
| TASK-005 | Implement `src/tcc_itransformer/config.py` — Pydantic `ExperimentConfig` with validation | `src/tcc_itransformer/config.py` | TASK-001 | M | Config loads from YAML, rejects invalid values |
| TASK-006 | Create `configs/default.yaml` with base hyperparameters | `configs/default.yaml` | TASK-005 | S | `ExperimentConfig.model_validate(yaml)` succeeds |
| TASK-007 | Create `scripts/generate_sweep_configs.py` — generates 36 YAML files | `scripts/generate_sweep_configs.py`, `configs/sweep/*.yaml` | TASK-005 | S | 36 YAML files generated, all valid configs |
| TASK-008 | Create `scripts/log_environment.py` — hardware/software provenance capture | `scripts/log_environment.py` | TASK-001 | S | `docs/environment.json` generated |
| TASK-009 | Create `tests/conftest.py` — mock data fixtures, GPU skip marker | `tests/conftest.py` | TASK-001 | S | `pytest --co` lists fixtures |
| TASK-010 | Write and commit `docs/pre_analysis_plan.md` (pre-registration) | `docs/pre_analysis_plan.md` | None | M | Committed with git hash BEFORE any experiment code |

**Phase 0 gate**: `uv sync && pytest --co` succeeds, 36 config files exist, pre-analysis plan committed.

---

### Phase 1: Data Pipeline (Est: 2–3 days)

**Goal**: FRED-MD loading, tcode transformation, NaN handling, windowed dataset.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-011 | Download FRED-MD snapshot, compute SHA-256, commit to `data/snapshots/` | `scripts/download_data.py`, `data/snapshots/` | None | S | SHA-256 file exists, matches CSV |
| TASK-012 | Implement `fred_md.py` — `load_fred_md()`, `remove_outliers()`, `apply_tcode()`, `transform_panel()` | `src/tcc_itransformer/data/fred_md.py` | TASK-011 | L | tcodes extracted from row 2, all 7 codes implemented |
| TASK-013 | Implement NaN handling strategy in `preprocessing.py` — drop series with >10% missing, forward-fill remainder, `fit_scaler()` on train only | `src/tcc_itransformer/data/preprocessing.py` | TASK-012 | M | Scaler fit on train only, NaN count = 0 after preprocessing |
| TASK-014 | Implement `dataset.py` — `FREDMDWindowDataset` with configurable stride | `src/tcc_itransformer/data/dataset.py` | TASK-013 | M | Dataset returns `(W, N)` tensors, length correct for stride |
| TASK-015 | Implement non-overlapping index extraction utility | `src/tcc_itransformer/evaluation/effective_sample_size.py` | None | S | `extract_non_overlapping_indices(n, W)` returns correct indices |
| TASK-016 | Unit tests for data pipeline | `tests/unit/test_data.py`, `tests/unit/test_preprocessing.py` | TASK-012, TASK-013, TASK-014 | M | All 10+ tests pass (tcode, outlier, scaler leakage, dataset shape) |

**Phase 1 gate**: `pytest tests/unit/test_data.py tests/unit/test_preprocessing.py` all pass. Data loads, transforms, windows correctly.

**Verification checklist**:
- [ ] tcodes extracted from row 2, NOT row 1
- [ ] tcode=5 on known series matches manual Δlog calculation
- [ ] Scaler fit on train only → val column means ≈ ≠ 0
- [ ] Outlier removal: |x − median| > 10×IQR → NaN
- [ ] Series with >10% NaN dropped (document which ones)
- [ ] Forward-fill applied AFTER outlier removal, BEFORE tcode
- [ ] SHA-256 of snapshot verified at load time
- [ ] Dataset length: stride=1 → T−W; stride=W → floor((T−W)/W)+1

---

### Phase 2: Model Architecture (Est: 3–4 days)

**Goal**: iTransformer encoder, mirror decoder, autoencoder wrapper. All shapes verified.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-017 | Read iTransformer source code (thuml/Time-Series-Library) | `docs/api_reference.md` (notes) | None | M | Exact API documented: inversion, embedding, attention shapes |
| TASK-018 | Implement `layers.py` — `VariateEmbedding(W→d_model)`, `TransformerEncoderBlock` | `src/tcc_itransformer/model/layers.py` | TASK-017 | L | Input (B,N,W) → Output (B,N,d_model), attention across N tokens |
| TASK-019 | Implement `encoder.py` — `iTransformerEncoder` with variate inversion + mean pooling | `src/tcc_itransformer/model/encoder.py` | TASK-018 | M | Input (B,W,N) → z (B,latent_dim). Transpose, embed, attend, pool, project. |
| TASK-020 | Implement `decoder.py` — mirror decoder (linear expand → FFN blocks → project to W) | `src/tcc_itransformer/model/decoder.py` | TASK-018 | M | Input z (B,latent_dim) → x_hat (B,W,N) |
| TASK-021 | Implement `autoencoder.py` — `iTransformerAE` with `forward(x)→(x_hat, z)` and `encode(x)→z` | `src/tcc_itransformer/model/autoencoder.py` | TASK-019, TASK-020 | M | Round-trip: x_hat.shape == x.shape, z.shape == (B, latent_dim) |
| TASK-022 | Implement `losses.py` — `reconstruction_loss(x, x_hat)`, `naive_baseline_loss(x, train_mean)` | `src/tcc_itransformer/model/losses.py` | None | S | MSE computation correct, baseline computes mean prediction MSE |
| TASK-023 | Unit tests for model | `tests/unit/test_model.py`, `tests/unit/test_losses.py` | TASK-021, TASK-022 | M | All shape tests pass, grads flow, no NaN, param count < 500K (d_model=32) |

**Phase 2 gate**: `pytest tests/unit/test_model.py` all pass.

**Verification checklist**:
- [ ] `z.shape == (B, latent_dim)` for all (B, W, N) inputs
- [ ] `x_hat.shape == x.shape == (B, W, N)`
- [ ] No NaN in forward pass with random input
- [ ] All parameters have `.grad` after backward pass
- [ ] `n_heads` divides `d_model` evenly (validated in config)
- [ ] Param count < 500K for d_model=32, < 2M for d_model=64
- [ ] Variate inversion: input (B,W,N) → transposed to (B,N,W) before embedding

---

### Phase 3: Training Loop (Est: 2 days)

**Goal**: Complete training pipeline with early stopping, checkpointing, MLflow logging.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-024 | Implement `callbacks.py` — `EarlyStopping`, `ModelCheckpoint`, `MLflowCallback` | `src/tcc_itransformer/training/callbacks.py` | TASK-005 | M | Early stopping triggers at patience, checkpoint saves best model |
| TASK-025 | Implement `mlflow_utils.py` — experiment setup, param/metric logging, artifact logging | `src/tcc_itransformer/tracking/mlflow_utils.py` | None | S | MLflow run created, params logged, `mlflow ui` shows run |
| TASK-026 | Implement `trainer.py` — training loop with AdamW, CosineAnnealingLR, gradient clipping | `src/tcc_itransformer/training/trainer.py` | TASK-021, TASK-024, TASK-025 | L | Loss decreases over epochs, val loss tracked, training completes |
| TASK-027 | Implement `scripts/run_single.py` — full pipeline for one config | `scripts/run_single.py` | TASK-026, TASK-014 | M | `python scripts/run_single.py --config configs/default.yaml` runs to completion |
| TASK-028 | Integration tests for training | `tests/integration/test_training.py` | TASK-026 | M | Loss decreases, no NaN, early stopping triggers, checkpoint restores |

**Phase 3 gate**: `pytest tests/integration/test_training.py` passes. A single training run completes with MLflow tracking.

---

### Phase 4: Tracer Bullet — End-to-End Pipeline (Est: 1 day)

**Goal**: Run the FULL pipeline with a dummy/simple config to validate everything connects. This is the **most critical phase** for de-risking.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-029 | Run end-to-end: FRED-MD → preprocess → window → train (5 epochs) → extract embeddings → PCA → K-Means → 1 MLflow run logged | None (orchestration test) | TASK-027 | M | MLflow shows 1 run with: train_loss, val_loss, silhouette, effective_rank |
| TASK-030 | Integration test: `test_pipeline_e2e.py` — validates full pipeline on mock data | `tests/integration/test_pipeline_e2e.py` | TASK-027, TASK-014 | M | Full pipeline completes on synthetic (50, 20) data in <60s CPU |

**Phase 4 gate**: End-to-end pipeline runs. MLflow UI shows a complete experiment run.

---

### Phase 5: Evaluation Modules (Est: 3–4 days) — **PARALLEL with Phase 2–3**

These modules depend only on numpy arrays (embeddings + labels), NOT on the model. Can be developed and tested using synthetic data in parallel with model development.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-031 | Implement `embedding_quality.py` — reconstruction MSE, naive baseline, collapse check, effective rank, isotropy | `src/tcc_itransformer/evaluation/embedding_quality.py` | None | M | All metrics compute on synthetic (50, 6) data |
| TASK-032 | Implement `clustering.py` — adaptive PCA, K-Means, silhouette, DB, CH, stability (ARI across seeds) | `src/tcc_itransformer/evaluation/clustering.py` | None | L | Adaptive PCA selects n_components < latent_dim. K-Means stable ARI > 0.9 on synthetic data. |
| TASK-033 | Implement `statistical_tests.py` — KW per dim, Mann-Whitney pairwise, BH correction, permutation test, effect sizes (η²_H, rank-biserial) | `src/tcc_itransformer/evaluation/statistical_tests.py` | None | L | KW produces H stat + p-value + η². BH correction reduces p-values. Permutation test runs 10K perms. |
| TASK-034 | Implement `baselines.py` — random, raw PCA, linear AE, windowed PCA baselines | `src/tcc_itransformer/evaluation/baselines.py` | None | L | All 4 baselines produce silhouette scores on synthetic data |
| TASK-035 | Implement `effective_sample_size.py` — `compute_effective_n()`, `extract_non_overlapping_indices()` | `src/tcc_itransformer/evaluation/effective_sample_size.py` | None | S | n_eff formula correct, matches T/W for overlapping windows |
| TASK-036 | Implement block bootstrap in `statistical_tests.py` — `moving_block_bootstrap()` with BCa intervals | (added to TASK-033 file) | TASK-033 | M | Bootstrap CI covers true parameter on known distribution |
| TASK-037 | Unit tests for ALL evaluation modules | `tests/unit/test_clustering.py`, `tests/unit/test_statistical.py`, `tests/unit/test_baselines.py` | TASK-031–036 | L | All tests pass on synthetic data, no model required |

**Phase 5 gate**: `pytest tests/unit/test_clustering.py tests/unit/test_statistical.py tests/unit/test_baselines.py` all pass.

---

### Phase 6: EDA Notebook (Est: 2 days) — **PARALLEL with Phases 2–5**

Depends only on Phase 1 (data pipeline). Can run in parallel with all model/evaluation work.

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-038 | EDA notebook: FRED-MD load, missing data heatmap, distributions, stationarity (ADF+KPSS), split visualization, correlation heatmaps, window statistics, baseline PCA, regime proxy (ICSS) | `notebooks/00_eda.ipynb` | TASK-012, TASK-013 | XL | 10 sections complete, all figures saved to `results/figures/` |

**EDA sections** (from original plan):
1. Setup & Imports
2. FRED-MD Load — shape, head, tcode summary, **series dropped** (documenting NaN handling)
3. Missing Data — null % heatmap, structural vs random
4. Distributions (train only) — histograms top-12 by kurtosis, Jarque-Bera table
5. Stationarity — ADF + KPSS per series, flag ambiguous
6. Split Visualization — timeline + KS test (val vs train, test vs train)
7. Correlation — hierarchical heatmaps train vs val
8. Window Statistics — mean/std distributions for W=6,12,24
9. Baseline PCA on raw features — scree plot, PC1 vs PC2, interpret
10. Summary table — confirmed/refined findings

---

### Phase 7: Quality Tests (Est: 1–2 days)

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-039 | Quality tests: `test_embedding_quality.py` — reconstruction < baseline, no collapse, rank > 2 | `tests/quality/test_embedding_quality.py` | TASK-031, TASK-029 | M | Tests defined (may require trained model to pass) |
| TASK-040 | Quality tests: `test_clustering_quality.py` — silhouette > 0, KW significant on ≥ceil(d/2) dims, stability ARI > 0.7 | `tests/quality/test_clustering_quality.py` | TASK-032, TASK-033, TASK-029 | M | Tests defined with clear pass/fail thresholds |

**Note**: Quality tests may FAIL initially — that's expected. They serve as scientific gates, not engineering gates. A failing quality test means the model needs tuning, not that the code is wrong.

---

### Phase 8: Sweep Infrastructure & Execution (Est: 2–3 days)

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-041 | Implement `scripts/run_sweep.py` — orchestrates 12 training runs + 36 evaluations | `scripts/run_sweep.py` | TASK-027 | L | Runs 12 (W,d) training combos, applies K={3,4,5} post-hoc, logs all 36 to MLflow |
| TASK-042 | Implement `scripts/run_baselines.py` — runs all 4 baselines for comparison | `scripts/run_baselines.py` | TASK-034 | M | 4 baseline results logged to MLflow per (W,d,K) combo |
| TASK-043 | Run full sweep + baselines on GPU | None (execution) | TASK-041, TASK-042 | XL | 36 evaluation runs + 4×36 baseline runs complete in MLflow |
| TASK-044 | Run statistical comparison: iTransformer vs each baseline (permutation tests) | (uses TASK-033 code) | TASK-043 | M | Permutation p-values for all 4 baselines on primary config |

---

### Phase 9: Embedding Analysis Notebook (Est: 2 days)

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-045 | Analysis notebook: load trained models, extract embeddings, geometry checks, PCA scree, K selection, regime visualization, statistical validation, baseline comparison, summary table | `notebooks/01_embedding_analysis.ipynb` | TASK-043 | XL | All 8 sections complete, thesis-ready figures |

**Analysis sections**:
1. Load trained model + data
2. Extract embeddings (train/val/test) — forward pass, no grad
3. Embedding geometry — per-dim variance heatmap, effective rank, isotropy
4. Adaptive PCA — scree plot, variance explained per (W,d) config
5. K selection — silhouette vs K on train non-overlapping, select best K
6. Regime visualization — time series of labels + NBER recession overlay
7. Statistical validation — KW table + effect sizes + pairwise comparisons
8. Baseline comparison table — iTransformer vs 4 baselines with permutation p-values
9. Summary metrics table — all primary + exploratory results

---

### Phase 10: Thesis Artifacts (Est: 1–2 days)

| Task ID | Task | Files | Dependencies | Size | Completion Criteria |
|---------|------|-------|-------------|------|---------------------|
| TASK-046 | `scripts/export_results.py` — MLflow → LaTeX tables, comparison figures | `scripts/export_results.py` | TASK-043 | L | LaTeX tables for embedding geometry, clustering, baselines, KW tests |
| TASK-047 | Generate all thesis figures — cluster scatter, regime timeline, scree plots, bootstrap distributions | `results/figures/` | TASK-045 | M | ≥6 publication-ready figures (matplotlib, 300dpi) |
| TASK-048 | Final README with reproduction instructions | `README.md` | All | M | `git clone && uv sync && make sweep` runs the full experiment |

---

### Phase Dependency Graph

```
Phase 0 (Foundation)
  ├──→ Phase 1 (Data)
  │      ├──→ Phase 4 (Tracer Bullet) ──→ Phase 7 (Quality Tests)
  │      ├──→ Phase 6 (EDA) ─────────────────────────┐
  │      └──→ Phase 8 (Sweep) ──→ Phase 9 (Analysis) ├──→ Phase 10 (Artifacts)
  │              ↑                                     │
  ├──→ Phase 2 (Model) ──→ Phase 3 (Training) ────────┘
  │
  └──→ Phase 5 (Evaluation) ───── can start in PARALLEL with Phase 2 ─────┘
```

**Critical path**: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 8 → Phase 9 → Phase 10

**Parallel opportunities**:
- Phase 5 (evaluation modules) runs **entirely in parallel** with Phases 2–3
- Phase 6 (EDA notebook) runs in parallel with Phases 2–5
- Phase 7 (quality tests) can be WRITTEN in parallel, but only PASS after Phase 4

---

## 6. Statistical Testing Framework

### 6.1 Complete Pipeline: Embeddings → Thesis Results

```
STEP 0: PRE-REGISTRATION (Phase 0, TASK-010)
  └─ Lock primary analysis BEFORE seeing results
     Primary config: W=12, d_latent=8, K=argmax_sil(train_nol)
     Primary test: Silhouette(iTransformer) > Silhouette(B1: Raw PCA)
     All other configs → exploratory

STEP 1: EXTRACT EMBEDDINGS
  └─ For each (W, d) config:
     z_train, z_val, z_test = model.encode(windows)
     z_train_nol = z_train[::W]  (non-overlapping)
     z_test_nol = z_test[::W]

STEP 2: EMBEDDING GEOMETRY GATE (stop if failed)
  └─ a) Per-dim variance > 1e-4
     b) Effective rank > 2
     c) Isotropy: mean pairwise cosine < 0.9
     d) Reconstruction MSE < naive baseline MSE
     → If ANY fails → do NOT proceed to clustering for that config

STEP 3: ADAPTIVE PCA
  └─ Fit on z_train_nol (non-overlapping!)
     n_components = min(latent_dim−1, components_for_90%_var, 5)
     Transform all splits with train PCA
     Report: explained variance per component

STEP 4: CLUSTERING (non-overlapping sets only)
  └─ Fit K-Means on z_train_nol_pca for each K∈{3,4,5}
     Predict labels for val_nol and test_nol
     random_state=42, n_init=20

STEP 5: BOOTSTRAP CIs (where viable)
  └─ Moving Block Bootstrap on train_nol
     Block length: ceil(1.5 × W)
     B=10,000, BCa intervals
     ONLY viable for W=6 (n_eff≈55), MARGINAL for W=12 (n_eff≈27)
     W=24: POINT ESTIMATES ONLY (document limitation)

STEP 6: STATISTICAL TESTS (non-overlapping, n_eff corrected)
  └─ a) KW test per embedding dim → BH across d dims
     b) Pairwise Mann-Whitney → BH corrected
     c) Permutation test: silhouette(iTransformer) > silhouette(baseline)
     d) Temporal consistency: count regime transitions
     e) ALL tests on non-overlapping subsets

STEP 7: BASELINE COMPARISONS
  └─ B0: Random embeddings → K-Means
     B1: Raw PCA on features → K-Means
     B2: Linear AE (single layer) → K-Means
     B3: PCA on flattened windows → K-Means
     Permutation test: Δsilhouette(iTransformer − each baseline)

STEP 8: MULTIPLE COMPARISON CORRECTION
  └─ Level 1 (Primary): Single pre-registered test, α=0.05, no correction
     Level 2 (Confirmatory): Primary config only → BH across d KW tests
     Level 3 (Exploratory): All other configs → BH, α=0.10, labeled exploratory

STEP 9: EFFECT SIZES + REPORTING
  └─ η²_H for KW, rank-biserial for Mann-Whitney, Δ_sil with CI
     APA-style tables with all required statistics
```

### 6.2 Effective Sample Size Table

| Split | W=6 | W=12 | W=24 |
|-------|-----|------|------|
| Train (nominal) | ~330 | ~324 | ~312 |
| **Train (n_eff)** | **~55** | **~27** | **~13** |
| Val (nominal) | ~30 | ~24 | ~12 |
| **Val (n_eff)** | **~5** | **~2** | **~1** |
| Test (nominal) | ~45 | ~39 | ~27 |
| **Test (n_eff)** | **~8** | **~4** | **~2** |
| Bootstrap viable? | ✅ Yes | ⚠️ Marginal | ❌ No |
| Statistical tests? | ✅ Yes | ⚠️ Low power | ❌ Point estimates only |

**Formula**: $n_{\text{eff}} \approx \lfloor T / W \rfloor$ where $T$ is the number of months in the split.

### 6.3 What to Report in Thesis Tables

**Table 1: Embedding Geometry** (per config)

| W | d_latent | Recon MSE | Naive MSE | MSE Ratio | Eff. Rank | Isotropy | Collapsed Dims |
|---|----------|-----------|-----------|-----------|-----------|----------|----------------|

**Table 2: Primary Config Results** (W=12, d_latent=8, K=best)

| K | Sil [95% CI] | DB [95% CI] | CH | PCA var% | n_eff |
|---|--------------|-------------|----|----------|-------|

**Table 3: Baseline Comparison** (primary config)

| Method | Silhouette | Δ vs iTransformer | Perm p | Significant? |
|--------|-----------|-------------------|--------|--------------|
| iTransformer | X.XX ± CI | — | — | — |
| Random (B0) | X.XX | −X.XX | 0.XXX | Yes/No |
| Raw PCA (B1) | X.XX | −X.XX | 0.XXX | Yes/No |
| Linear AE (B2) | X.XX | −X.XX | 0.XXX | Yes/No |
| Windowed PCA (B3) | X.XX | −X.XX | 0.XXX | Yes/No |

**Table 4: KW Test per Dimension** (primary config)

| Dim | H stat | p (raw) | p (BH-adj) | η²_H | Significant? |
|-----|--------|---------|------------|------|--------------|

**Table 5: Sensitivity Across All Configs** (exploratory, BH-corrected)

| W | d | K | PCA_k | Sil | Best Baseline Sil | Δ | p (BH) | n_eff |
|---|---|---|-------|-----|--------------------|---|--------|-------|

**Table 6: Effective Sample Sizes & Bootstrap Viability**

| W | Split | Nominal n | n_eff | Block length | Bootstrap? |
|---|-------|-----------|-------|-------------|-----------|

---

## 7. Test Organization & Trackability

### 7.1 Test Categories

```
tests/
├── unit/              # Fast, no GPU, no I/O — validates code correctness
│   ├── test_data.py           # 10+ tests: tcode, outlier, scaler, dataset
│   ├── test_preprocessing.py  # 5+ tests: NaN handling, scaler leakage
│   ├── test_model.py          # 8+ tests: shapes, grads, param count
│   ├── test_losses.py         # 3+ tests: MSE, baseline loss
│   ├── test_clustering.py     # 6+ tests: adaptive PCA, K-Means, metrics
│   ├── test_statistical.py    # 8+ tests: KW, bootstrap, BH, permutation
│   └── test_baselines.py      # 4+ tests: each baseline produces valid output
├── integration/       # Slower, may use GPU — validates connections
│   ├── test_training.py       # 4+ tests: loss decrease, early stop, checkpoint
│   └── test_pipeline_e2e.py   # 1 test: full pipeline on mock data
└── quality/           # Scientific gates — may FAIL until model is tuned
    ├── test_embedding_quality.py   # 4 tests: recon < baseline, no collapse, rank, PCA var
    └── test_clustering_quality.py  # 4 tests: sil > 0, KW significant, stability, K valid
```

### 7.2 Trackability Across Configurations

**Problem**: When hyperparameters change, old results must remain comparable.

**Solution**: MLflow is the results database. Every run is tagged with:

```python
mlflow.log_params({
    "W": config.window_size,
    "d_latent": config.latent_dim,
    "d_model": config.d_model,
    "K": config.n_clusters,
    "seed": config.seed,
    "pca_components": n_selected,  # adaptive
    "n_eff_train": n_eff_train,
    "n_eff_test": n_eff_test,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
})
```

**Comparison workflow**:
```python
# In scripts/export_results.py
runs = mlflow.search_runs(experiment_names=["itransformer-autoencoder"])

# Compare W=6 vs W=12 vs W=24 on silhouette
pivot = runs.pivot_table(
    values="metrics.silhouette",
    index="params.d_latent",
    columns="params.W",
    aggfunc="mean"
)
```

### 7.3 Regression Detection

Quality tests serve as regression gates. If a code refactor changes model behavior:

```python
# tests/quality/test_embedding_quality.py
def test_reconstruction_beats_baseline(trained_model, test_loader, train_mean):
    """CRITICAL: model MSE must be lower than predicting train mean."""
    model_mse = reconstruction_mse(trained_model, test_loader)
    baseline_mse = naive_baseline_mse(test_loader, train_mean)
    assert model_mse < baseline_mse, (
        f"Model MSE ({model_mse:.4f}) >= Baseline MSE ({baseline_mse:.4f}). "
        "Reconstruction is worse than predicting the mean."
    )
```

### 7.4 CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "0.5.x"
      - run: uv sync --frozen
      - run: uv run pytest tests/unit/ -v --cov=src --cov-fail-under=70
        env:
          CUBLAS_WORKSPACE_CONFIG: ":4096:8"
```

GPU tests (`@pytest.mark.gpu`) are skipped in CI and run locally.

---

## 8. Sweep Strategy

### 8.1 Grid Definition

| Axis | Values | Count |
|------|--------|-------|
| Window size W | 6, 12, 24 | 3 |
| Latent dim d | 6, 7, 8, 9 | 4 |
| Clusters K | 3, 4, 5 | 3 |
| **Total evaluations** | | **36** |
| **Unique training runs** | W × d = 3 × 4 | **12** |

K is post-hoc (clustering only), so only 12 models need training. Each trained model produces embeddings that are evaluated with K=3,4,5 → 36 total evaluations.

### 8.2 Fixed Hyperparameters (Not Swept)

| Param | Value | Justification |
|-------|-------|---------------|
| d_model | 64 | Small dataset (~330 effective samples) — 512 would overfit |
| n_heads | 4 | Divides d_model=64 evenly |
| n_layers | 2 | Small dataset — more layers = more overfitting risk |
| dropout | 0.1 | Standard regularization |
| batch_size | 32 | Fits in 6GB VRAM for all W values |
| learning_rate | 1e-3 | AdamW default, fine-tuned via scheduler |
| weight_decay | 1e-4 | Mandatory regularization for small dataset |
| patience | 10 | Early stopping on val loss |
| grad_clip | 1.0 | Prevents gradient explosion |
| seed | 42 | Reproducibility (also run stability with seeds 0–9) |

### 8.3 Primary vs Exploratory

| Category | Configs | Statistical treatment |
|----------|---------|----------------------|
| **Primary** (pre-registered) | W=12, d_latent=8, K=best_sil_train | α=0.05, no correction, full statistical battery |
| **Confirmatory** | W=6 and W=12 configs | BH-corrected, α=0.05 |
| **Exploratory** | W=24 configs (all 12) | BH-corrected, α=0.10, labeled "exploratory — low statistical power" |

### 8.4 Execution

```bash
# Train 12 models (sequential, single GPU)
# Estimated: 12 × 10 min = ~2 hours
make sweep

# Run baselines (fast, CPU-only)
# Estimated: <10 min
make baselines

# Export results to LaTeX
make export
```

---

## 9. Thesis Artifact Generation

### Required Figures (minimum 6, matplotlib+seaborn)

| # | Figure | Source Phase | Script |
|---|--------|-------------|--------|
| 1 | FRED-MD missing data heatmap | Phase 6 (EDA) | `notebooks/00_eda.ipynb` |
| 2 | PCA scree plot (raw features vs embeddings) | Phase 9 | `notebooks/01_embedding_analysis.ipynb` |
| 3 | Embedding 2D scatter (PC1 vs PC2, colored by date/regime) | Phase 9 | `notebooks/01_embedding_analysis.ipynb` |
| 4 | Silhouette vs K selection plot | Phase 9 | `notebooks/01_embedding_analysis.ipynb` |
| 5 | Regime timeline with NBER recession overlay | Phase 9 | `notebooks/01_embedding_analysis.ipynb` |
| 6 | Baseline comparison bar chart (silhouette scores ± CI) | Phase 10 | `scripts/export_results.py` |
| 7 | Per-dim KW effect size heatmap | Phase 10 | `scripts/export_results.py` |
| 8 | Bootstrap distribution of silhouette (histogram) | Phase 9 | `notebooks/01_embedding_analysis.ipynb` |

### Required Tables (minimum 6, auto-generated LaTeX)

See Section 6.3 above (Tables 1–6).

---

## 10. Risk Mitigation Matrix

| Risk | Likelihood | Impact | Mitigation | Owner Phase |
|------|-----------|--------|------------|-------------|
| **Overlapping windows inflate metrics** | 95% | CRITICAL | Evaluate ALL stats on non-overlapping subsets only | Phase 5 |
| **FRED-MD shrinks below 128 series** | 90% | HIGH | NaN strategy: drop >10% missing, forward-fill rest, document dropped series | Phase 1 |
| **Autoencoder collapse (constant embedding)** | 70% | HIGH | Geometry gate: per-dim var > 1e-4, effective rank > 2, isotropy check | Phase 5 |
| **Autoencoder memorizes temporal order** | 60% | HIGH | Baseline comparison: if linear AE matches iTransformer, nonlinearity adds nothing | Phase 8 |
| **W=24 has no statistical power** | 100% | MEDIUM | Label W=24 as "exploratory only" with explicit limitation statement | Phase 6/9 |
| **Val set too small for reliable K selection** | 80% | MEDIUM | Select K on train non-overlapping, report sensitivity to K±1 | Phase 5 |
| **K-Means non-determinism** | 50% | MEDIUM | `random_state=42`, `n_init=20`, stability test (ARI > 0.7 across 10 seeds) | Phase 5 |
| **GPU non-determinism breaks reproducibility** | 75% | MEDIUM | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, env logging | Phase 0 |
| **sklearn version changes KMeans behavior** | 40% | MEDIUM | Pin exact version in `uv.lock`, explicitly set `n_init=20` | Phase 0 |
| **COVID OOD spike in val loss** | 100% | LOW | Expected and documented — quantify domain shift in thesis | Phase 9 |
| **PCA no-op when components = latent_dim** | Fixed | N/A | Adaptive PCA: n_components = min(latent_dim−1, 90%_var, 5) | Phase 5 |
| **No baseline → unfalsifiable thesis claim** | Fixed | N/A | 4 mandatory baselines with permutation tests | Phase 8 |
| **Data changes after download (FRED-MD is live)** | Fixed | N/A | Committed snapshot with SHA-256 integrity check | Phase 1 |

---

## Appendix A: Agent Debate Record

### Agent 1 — Plan Mode (Primary Designer)
**Key positions**: MLflow local, Pydantic+YAML, tracer bullet phasing, results manifest with config hashing.
**Strongest argument**: "Build the statistical testing pipeline on synthetic data IN PARALLEL with model development. This is the single biggest scheduling improvement."

### Agent 2 — Critical Thinking (Skeptic/Challenger)
**Top 5 Failure Modes identified**:
1. Overlapping windows destroy statistical validity (n_eff ≈ T/W, not nominal n)
2. FRED-MD won't actually be 840×128 (discontinued series, NaN after tcode)
3. Autoencoder may collapse or memorize temporal ordering
4. 36-combo grid can't be reliably ranked on ~30 val samples
5. Reproducibility is performative if GPU determinism and version pinning aren't enforced

**Missing from original plan**: NaN handling strategy, baseline comparisons, regime definition, split sensitivity, PCA components coupling.

### Agent 3 — Context Architect (Constraint Guardian)
**Key contributions**: `uv` + exact pins, data snapshot with SHA-256, seed.py as single source of truth, `CUBLAS_WORKSPACE_CONFIG` in env, hardware provenance logging, CI/CD with GPU skip markers.

### Agent 4 — Scientific Paper Research (Rigor Enforcer)
**Key contributions**: Complete 9-step statistical pipeline, block bootstrap specification (only viable for W=6), 4 mandatory baselines, hierarchical testing with pre-registration, adaptive PCA, 6 thesis table templates, effect size reporting (η²_H, rank-biserial), pre-analysis plan template.
**W=24 analysis**: n_eff < 2 on val/test → NO statistical tests, point estimates only.

### Agent 5 — Implementation Plan (Feasibility Analyst)
**Key contributions**: Train 12 models not 36 (K is post-hoc), parallel phase identification, critical path analysis.
**GPU time estimate**: 12 runs × ~10 min = ~2 hours (single RTX 3060).

### Agent 6 — Doublecheck (Verification Agent)
**Verification results**:

| Claim | Verdict |
|-------|---------|
| MLflow > W&B for thesis | PARTIALLY VERIFIED — defensible, not absolute |
| n_eff ≈ T/W | VERIFIED — math correct |
| Block bootstrap impossible W≥12 on test | PARTIALLY VERIFIED — technically possible on overlapping (3 blocks) but useless |
| BY instead of BH | DISPUTED → **BH is sufficient** under PRDS condition |
| Train 12 not 36 | VERIFIED — K is purely post-hoc |
| FRED-MD shrinks below 128 | VERIFIED — typically 120–127 usable series |
| W=24 no statistical tests | VERIFIED — n_eff < 2 on both val and test |
| Pydantic+YAML > Hydra | PARTIALLY VERIFIED — simpler for thesis scale |
| sklearn KMeans changed 1.1→1.2 | PARTIALLY VERIFIED — actual change was in 1.4 (n_init default) |
| PCA no-op at full rank | VERIFIED — Euclidean distance is rotation-invariant |

---

## Appendix B: Pre-Analysis Plan Template

Save as `docs/pre_analysis_plan.md` and commit **BEFORE running any experiments** (TASK-010).

```markdown
# Pre-Analysis Plan — iTransformer Embedding Evaluation
## Date: [GIT COMMIT DATE]
## Author: [NAME]
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
```

---

*End of Implementation Plan v2. Generated through multi-agent debate on 2026-04-17.*
