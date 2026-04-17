# Gap Remediation Plan — iTransformer Thesis Project

> Generated from 6-agent debate (Scientific Rigor, Documentation, Notebooks/Viz, Test Coverage, DevOps, Thesis Defense)

---

## Executive Summary

**Overall readiness: ~55% — NOT defense-ready.**

30 of 48 tasks are DONE. 3 are PARTIAL. 15 are MISSING. The ML pipeline code is solid (~4,327 LOC), but the **statistical testing framework is never called**, zero experiments have been run, no notebooks exist, no data snapshot is committed, and the path from MLflow → thesis tables/figures is unbuilt.

### Agent Consensus Summary

| Agent | CRITICAL Gaps | IMPORTANT Gaps | NICE-TO-HAVE |
|-------|---------------|----------------|--------------|
| 1: Scientific Rigor | 7 | 10 | 6 |
| 2: Documentation | 8 | 10 | 9 |
| 3: Notebooks/Viz | 4 blocking | 6 | 1 |
| 4: Test Coverage | 8 | 9 | 5 |
| 5: DevOps/Repro | 7 | 8 | 5 |
| 6: Thesis Defense | 6 CRITICAL | 9 IMPORTANT | 5 |

---

## TASK-by-TASK Status (48 Tasks)

| Status | Count | Tasks |
|--------|-------|-------|
| ✅ DONE | 30 | 001-006, 008-010, 012-016, 018-028, 030-035, 037, 041-042 |
| ⚠️ PARTIAL | 3 | 007, 017, 036 |
| ❌ MISSING | 15 | 011, 029, 038, 039, 040, 043, 044, 045, 046, 047, 048 |

---

## Implementation Phases (Prioritized)

### PHASE A: Critical Pipeline Fixes (BLOCKS EVERYTHING)

These must be done first — without them, no experiment can run and no thesis table can be generated.

#### A1. Wire Statistical Tests into Pipeline Scripts
**Agent consensus**: Agents 1 + 6 both flagged this as the #1 problem.
**Impact**: The statistical testing framework (~400 LOC) has zero consumers. The scientific core of the thesis is dead code.

**Files to modify:**
- `scripts/run_single.py` — Add after clustering:
  - Call `kruskal_wallis_per_dim()` on test non-overlapping embeddings
  - Call `pairwise_mann_whitney()` 
  - Call `permutation_test_silhouette()` vs baseline
  - Call `moving_block_bootstrap()` for CIs (when n_eff viable)
  - Call `clustering_stability()` across 5 seeds
  - Compute and log `reconstruction_mse()` and `naive_baseline_loss()` at eval time
  - Compute and log `compute_effective_n()`, `n_eff`
  - Log PCA variance explained
  - **Fix: Evaluate on TEST set** (currently only uses val set — violates pre-registration)
  - Add W=24 exploratory labeling (`mlflow.set_tag("analysis_type", "exploratory")`)
- `scripts/run_sweep.py` — Same additions, plus log `git_commit` per run
- `scripts/run_baselines.py` — Fix:
  - Extract non-overlapping indices (currently uses overlapping → violates GUD-001)
  - Apply adaptive PCA before KMeans (same post-processing as iTransformer for fair comparison)
  - Run `permutation_test_silhouette()` between iTransformer and each baseline
  - Compute and log Δ_silhouette with CI

**Priority**: 🔴 CRITICAL | **Complexity**: L | **Est**: 2-3 days

#### A2. Bootstrap BCa (not Percentile)
**Agent 1 GAP-01**: Plan requires BCa intervals; code uses plain `np.percentile()`.
**File**: `src/tcc_itransformer/evaluation/statistical_tests.py`
**Fix**: Replace percentile bootstrap with `scipy.stats.bootstrap(method='BCa')` or implement BCa manually (jackknife acceleration `a` + bias-correction `z0`).
**Also**: Add bootstrap viability warning when `n_eff < 20` (Agent 1 GAP-18).

**Priority**: 🔴 CRITICAL | **Complexity**: M

#### A3. Add CIs to All Effect Sizes
**Agent 1 GAP-02/04/05/17**: Every effect size needs a CI (GUD-002).
**Files**: `src/tcc_itransformer/evaluation/statistical_tests.py`
- `permutation_test_silhouette()` — Add bootstrap CI for Δ_silhouette
- `kruskal_wallis_per_dim()` — Add bootstrap CI for η²_H per dimension
- `pairwise_mann_whitney()` — Add bootstrap CI for rank-biserial

**Priority**: 🔴 CRITICAL | **Complexity**: M

#### A4. Data Snapshot + Download Script
**Agents 1 + 5 + 6**: No data snapshot committed. Cannot run any experiment.
**Files to create:**
- `scripts/download_data.py` — Download FRED-MD CSV, compute SHA-256, save to `data/snapshots/`
- `data/snapshots/fred_md_2026_04.csv` — Committed CSV
- `data/snapshots/fred_md_2026_04.sha256` — Committed hash
**Also**: Call `verify_sha256()` at start of `load_fred_md()` (Agent 1 GAP-14).

**Priority**: 🔴 CRITICAL | **Complexity**: M

#### A5. Generate + Commit Sweep Configs
**Agents 5 + 6**: `configs/sweep/` has only `.gitkeep`. `make sweep` crashes.
**Action**: Run `python scripts/generate_sweep_configs.py` → commit 36 YAML files.

**Priority**: 🔴 CRITICAL | **Complexity**: S

---

### PHASE B: Quality Tests + Test Gaps

#### B1. Create `tests/quality/test_embedding_quality.py` (TASK-039)
**Tests** (all `@pytest.mark.quality`):
- `test_reconstruction_beats_baseline` — model_mse < naive_baseline_mse
- `test_no_embedding_collapse` — all per-dim variances > 1e-4
- `test_effective_rank_above_2` — effective_rank > 2.0
- `test_pca_variance_explained` — first n_components explain > 90%

**New fixtures needed** in `tests/quality/conftest.py`:
- `trained_model` (session scope) — small iTransformerAE trained ~50 epochs on mock data
- `test_loader`, `val_loader` 
- `train_mean`

**Priority**: 🟡 IMPORTANT | **Complexity**: M

#### B2. Create `tests/quality/test_clustering_quality.py` (TASK-040)
**Tests**:
- `test_silhouette_above_zero` — silhouette > 0
- `test_kw_significant_dimensions` — ≥ ceil(d/2) dims significant after BH
- `test_clustering_stability` — ARI > 0.7 across 5 seeds
- `test_valid_k_range` — best_k ∈ {3, 4, 5}

**Priority**: 🟡 IMPORTANT | **Complexity**: M

#### B3. Missing Unit Tests (Agent 4)
| Test | File | What | Priority |
|------|------|------|----------|
| `test_all_params_have_grad` | test_model.py | After backward, all params have .grad | 🔴 CRITICAL |
| `test_n_heads_divides_d_model_validation` | test_model.py | Config rejects bad n_heads/d_model | 🔴 CRITICAL |
| `test_param_count_bounds` | test_model.py | <500K for d=32, <2M for d=64 | 🔴 CRITICAL |
| `test_verify_sha256` | test_data.py | SHA-256 match/mismatch | 🔴 CRITICAL |
| `test_mlflow_run_created` | test_training.py | MLflow run logged | 🟡 IMPORTANT |
| `test_config_from_yaml` | (new) test_config.py | YAML round-trip, validation | 🟡 IMPORTANT |
| `test_reconstruction_mse` | (new) test_embedding_quality.py | Unit test for metric | 🟡 IMPORTANT |
| `test_check_embedding_collapse` | (new) test_embedding_quality.py | Unit test | 🟡 IMPORTANT |
| `test_compute_effective_rank` | (new) test_embedding_quality.py | Unit test | 🟡 IMPORTANT |
| `test_compute_effective_n` | (new) test_effective_sample_size.py | Unit test | 🟡 IMPORTANT |

**Priority**: Mixed | **Complexity**: S each

---

### PHASE C: Documentation & Diagrams

#### C1. Update README.md (TASK-048)
**Currently**: Single line `# tcc_ai`.
**Needs** (~150 lines):
- Project title + academic context (university, TCC 2026)
- Abstract (2-3 sentences)
- Architecture overview diagram (Mermaid)
- Installation: `uv sync`
- Reproduction: `make download-data && make sweep`
- Project structure tree
- Configuration guide
- Results summary placeholder
- Link to pre-analysis plan
- Citation/BibTeX stub
- License reference

**Priority**: 🔴 CRITICAL | **Complexity**: M

#### C2. Create `docs/api_reference.md` (TASK-017)
**Contents**:
- iTransformer architecture overview
- Tensor shape flow table (B,W,N) through entire AE
- Public API reference (all modules)
- Attention mechanism details
- Adaptive PCA formula + justification
- Statistical testing hierarchy

**Priority**: 🟡 IMPORTANT | **Complexity**: L

#### C3. Architecture Diagrams (4 Mermaid diagrams)
1. **Pipeline flow** (data → model → embeddings → PCA → KMeans → stats → MLflow)
2. **iTransformer architecture** (variate-as-token inversion with shapes)
3. **Module dependency** (data → model → training → evaluation → tracking)
4. **Statistical testing hierarchy** (primary → exploratory → baseline)

**Where**: Embed in README.md + docs/api_reference.md

**Priority**: 🟡 IMPORTANT | **Complexity**: M

#### C4. Makefile Comments
Add 1-line comment per target. Fix `export` target pointing to nonexistent script.

**Priority**: 🟡 IMPORTANT | **Complexity**: S

#### C5. pyproject.toml Metadata
Add: `authors`, `license`, `readme`, `urls.repository`.

**Priority**: 🟡 IMPORTANT | **Complexity**: S

#### C6. Pre-Analysis Plan Gaps (Agent 1 GAP-06, Agent 2 DOC-006)
Add to `docs/pre_analysis_plan.md`:
- Success/failure criteria
- Quality gate thresholds
- Data exclusion criteria (>10% NaN threshold)
- "Report all results including null findings" statement
- Sample size justification / power limitation acknowledgment

**Priority**: 🟡 IMPORTANT | **Complexity**: S

---

### PHASE D: Notebooks (XL effort, depends on Phase A)

#### D1. Create `notebooks/00_eda.ipynb` (TASK-038)
**10 sections, ~35 cells** (Agent 3 provided cell-by-cell spec):
1. Setup & Imports
2. FRED-MD Load (shape, head, tcodes, dropped series documentation)
3. Missing Data (null % heatmap, structural vs random)
4. Distributions (train only — top-12 by kurtosis, Jarque-Bera table)
5. Stationarity (ADF + KPSS per series, flag ambiguous)
6. Split Visualization (timeline + KS test val vs train, test vs train)
7. Correlation (hierarchical heatmaps train vs val)
8. Window Statistics (mean/std distributions for W=6,12,24)
9. Baseline PCA on raw features (scree plot, PC1 vs PC2)
10. Summary table

**Requires**: Data snapshot (Phase A4)
**Priority**: 🟡 IMPORTANT | **Complexity**: XL

#### D2. Create `notebooks/01_embedding_analysis.ipynb` (TASK-045)
**9 sections, ~30 cells**:
1. Load trained model + data
2. Extract embeddings (train/val/test, no_grad)
3. Embedding geometry (per-dim variance heatmap, effective rank, isotropy)
4. Adaptive PCA (scree plot, variance explained per config)
5. K selection (silhouette vs K on train non-overlapping)
6. Regime visualization (labels + NBER recession overlay)
7. Statistical validation (KW table + effect sizes + pairwise)
8. Baseline comparison table (4 baselines with permutation p-values)
9. Summary metrics table

**Requires**: Trained models (Phase A fully done + experiments run)
**Priority**: 🟡 IMPORTANT | **Complexity**: XL

---

### PHASE E: Visualization Expansion

#### E1. Publication-Ready Viz Refactor
**File**: `src/tcc_itransformer/utils/viz.py`
**Issues (Agent 3)**:
- Font sizes too small (need ≥11pt body, ≥14pt titles)
- No colorblind-safe palette (need Okabe-Ito or similar)
- PNG only → add PDF export
- Hard-coded (10,6) figure size → vary by plot type
- No error handling for malformed input

**Priority**: 🟡 IMPORTANT | **Complexity**: M

#### E2. New Visualization Functions (11 missing)
| # | Function | For | Priority |
|---|----------|-----|----------|
| 1 | `plot_missing_data_heatmap()` | EDA notebook §3 | 🔴 CRITICAL |
| 2 | `plot_dist_histograms_grid()` | EDA notebook §4 | 🔴 CRITICAL |
| 3 | `plot_stationarity_summary()` | EDA notebook §5 | 🟡 IMPORTANT |
| 4 | `plot_correlation_heatmaps()` | EDA notebook §7 | 🔴 CRITICAL |
| 5 | `plot_window_statistics()` | EDA notebook §8 | 🟡 IMPORTANT |
| 6 | `plot_dim_variance_heatmap()` | Analysis notebook §3 | 🟡 IMPORTANT |
| 7 | `plot_silhouette_vs_k()` | Analysis notebook §5 | 🔴 CRITICAL |
| 8 | `plot_regime_timeline_nber()` | Analysis notebook §6 (NBER overlay) | 🔴 CRITICAL |
| 9 | `plot_statistical_results_table()` | Analysis notebook §7 | 🟡 IMPORTANT |
| 10 | `plot_pairwise_heatmap()` | Analysis notebook §7 | 🟡 IMPORTANT |
| 11 | `plot_baseline_comparison_bar()` | Phase 10 export | 🔴 CRITICAL |

**Estimated**: ~400-600 LOC
**Priority**: Mixed | **Complexity**: M-L total

---

### PHASE F: Thesis Artifacts (Phase 10)

#### F1. Create `scripts/export_results.py` (TASK-046)
**What it does**: MLflow → LaTeX tables + figures
- Query MLflow runs: `mlflow.search_runs()`
- Generate Tables 1-6 in LaTeX format
- Generate all thesis figures (call viz functions)
- Save to `results/figures/` and `results/tables/`

**Priority**: 🟡 IMPORTANT (needed for thesis, but after experiments) | **Complexity**: L

#### F2. Temporal Consistency Metric (Agent 1 GAP-22)
**New function**: Count regime transitions between consecutive non-overlapping windows.
**File**: `src/tcc_itransformer/evaluation/clustering.py`
**Purpose**: Sanity check — too many transitions = noise, too few = trivial.

**Priority**: 🟢 NICE-TO-HAVE | **Complexity**: S

---

### PHASE G: DevOps & Reproducibility

#### G1. Makefile New Targets
Add: `download-data`, `generate-sweep`, `help`, `reproduce` (= clean → download-data → generate-sweep → test → sweep)

**Priority**: 🟡 IMPORTANT | **Complexity**: S

#### G2. Fix numpy Version Constraint (Agent 5)
`pyproject.toml`: Change `numpy>=1.26.0,<2.1.0` → `numpy>=1.26.0,<2.0.0` (numpy 2.0+ has breaking changes for torch/pandas).

**Priority**: 🔴 CRITICAL | **Complexity**: S

#### G3. GitHub Actions CI (optional)
Create `.github/workflows/test.yml`: checkout → setup-uv → uv sync --frozen → make lint → make test

**Priority**: 🟢 NICE-TO-HAVE | **Complexity**: S

#### G4. Git Commit Logging in MLflow
Modify `mlflow_utils.py` or `run_single.py` to log `git_commit` as MLflow tag.

**Priority**: 🟡 IMPORTANT | **Complexity**: S

#### G5. `__init__.py` Exports
Add `__all__` to:
- `src/tcc_itransformer/__init__.py` (top-level package)
- `src/tcc_itransformer/utils/__init__.py`

**Priority**: 🟡 IMPORTANT | **Complexity**: S

---

## Implementation Order (Recommended)

```
WAVE 1 (Foundation — do FIRST, enables everything else):
  A4 → Data snapshot + download script
  A5 → Generate + commit sweep configs
  G2 → Fix numpy constraint

WAVE 2 (Pipeline integrity — makes experiments valid):
  A1 → Wire statistical tests into scripts
  A2 → BCa bootstrap
  A3 → CIs for all effect sizes

WAVE 3 (Quality assurance):
  B1 → Quality test: embeddings
  B2 → Quality test: clustering
  B3 → Missing unit tests

WAVE 4 (Documentation):
  C1 → README.md
  C2 → API reference
  C3 → Architecture diagrams
  C4 → Makefile comments
  C5 → pyproject.toml metadata
  C6 → Pre-analysis plan gaps
  G4 → Git commit in MLflow
  G5 → __init__.py exports

WAVE 5 (Visualization + Notebooks):
  E1 → Viz refactor (publication-ready)
  E2 → 11 new viz functions
  D1 → EDA notebook

WAVE 6 (Experiments — requires trained models):
  [RUN EXPERIMENTS: make sweep && make baselines]

WAVE 7 (Post-experiment artifacts):
  D2 → Analysis notebook
  F1 → export_results.py
  G1 → Makefile new targets
  G3 → CI pipeline (optional)
```

---

## Gap Count Summary

| Priority | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 16 | Blocks thesis defense or produces invalid results |
| 🟡 IMPORTANT | 22 | Significantly weakens thesis or missing plan requirements |
| 🟢 NICE-TO-HAVE | 8 | Polish and completeness |
| **TOTAL** | **46** | |

---

## Cross-Agent Agreements (High Confidence)

These findings were independently flagged by ≥3 agents:

1. **Statistical tests are dead code** (Agents 1, 4, 6) — implemented but never called from scripts
2. **No data snapshot** (Agents 1, 5, 6) — blocks all experiments
3. **README is empty** (Agents 2, 5, 6) — professor can't reproduce
4. **Quality tests don't exist** (Agents 1, 4, 6) — TASK-039/040 completely missing
5. **Bootstrap uses percentile, not BCa** (Agents 1, 6) — plan explicitly requires BCa
6. **run_single.py evaluates on val, not test** (Agents 1, 6) — violates pre-registration
7. **Baselines use overlapping windows** (Agents 1, 6) — unfair comparison, violates GUD-001
8. **No notebooks exist** (Agents 2, 3, 6) — both EDA and analysis missing
9. **export_results.py doesn't exist** (Agents 3, 5, 6) — no MLflow → LaTeX path
10. **viz.py not publication-ready** (Agent 3) — font sizes, palettes, export formats
