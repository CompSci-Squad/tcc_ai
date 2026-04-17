---
goal: 'iTransformer Autoencoder on FRED-MD: Full Thesis Implementation'
version: 1.0
date_created: 2026-04-17
last_updated: 2026-04-17
owner: Agent 5 — Feasibility Analyst
status: 'Planned'
tags: [feature, architecture, ml-pipeline, thesis, pytorch, clustering]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Definitive atomic task breakdown for a college thesis implementing an iTransformer autoencoder on FRED-MD macroeconomic data (840×128), with PCA + K-Means clustering on learned embeddings. 36 hyperparameter combinations (W×d×K = 3×4×3). Tracking via MLflow local. Config via Pydantic + YAML. Dependencies via uv + pyproject.toml with exact pins.

## 1. Requirements & Constraints

- **REQ-001**: All code must be reproducible — fixed seeds (`torch.manual_seed`, `np.random.seed`, `random.seed`) set from config
- **REQ-002**: Train/val/test temporal split: 1990–2018 / 2019–2021 / 2022–2025 — NO leakage
- **REQ-003**: Scaler fit on train split ONLY, applied to val/test via transform
- **REQ-004**: PCA fit on train embeddings ONLY, applied to val/test via transform
- **REQ-005**: Adaptive PCA: `n_components = min(latent_dim - 1, components_for_90%_var)`
- **REQ-006**: Non-overlapping windows for all evaluation metrics (overlapping only for training)
- **REQ-007**: Block bootstrap CIs (n=1000) on all clustering metrics due to temporal autocorrelation
- **REQ-008**: Hierarchical testing: KW first → pairwise Wilcoxon only if KW significant → BH correction
- **REQ-009**: 4 baselines: naive mean, PCA-only-KMeans, linear AE, random embeddings
- **REQ-010**: All 36 sweep combos must be logged to MLflow with full metrics
- **SEC-001**: No API keys or credentials in code — FRED-MD data committed as snapshot
- **CON-001**: Single GPU budget — sweep must run sequentially with model reloading
- **CON-002**: Small dataset (~330 train windows) — model ≤500K params for d_model=32
- **CON-003**: Data snapshot committed with SHA-256 checksum for versioning
- **GUD-001**: matplotlib + seaborn ONLY for visualization (no Plotly)
- **GUD-002**: MLflow local tracking (no remote server)
- **GUD-003**: Pydantic v2 models for config validation, YAML for human editing
- **PAT-001**: PyTorch native training loop — no Lightning (project scope)
- **PAT-002**: uv for dependency management with uv.lock committed

## 2. Implementation Steps

### Phase 0 — Foundation: Project Scaffold & Environment

- GOAL-001: Create the complete filesystem skeleton, dependency environment, configuration system, and data snapshot so that ALL subsequent phases can execute without any setup questions.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-001 | **Create `pyproject.toml`** with project metadata, exact pinned dependencies (torch>=2.2,<2.3; scikit-learn>=1.4,<1.5; numpy>=1.26,<2.0; pandas>=2.2,<2.3; matplotlib>=3.8,<3.9; seaborn>=0.13,<0.14; scipy>=1.13,<1.14; pytest>=8.0; pytest-cov; pyyaml>=6.0; tqdm; pydantic>=2.0,<3.0; mlflow>=2.12,<2.13; ruff), pytest config section `[tool.pytest.ini_options]` with `testpaths=["tests"]`, `markers=["slow", "quality"]`, ruff config section. File: `pyproject.toml` | S | — | | |
| TASK-002 | **Run `uv lock` and commit `uv.lock`**. Verify reproducible install with `uv sync`. Completion: `uv sync` exits 0 and `python -c "import torch; print(torch.__version__)"` prints version. Files: `uv.lock` | S | TASK-001 | | |
| TASK-003 | **Create directory skeleton** — all empty `__init__.py` files. Dirs: `src/`, `src/data/`, `src/model/`, `src/training/`, `src/evaluation/`, `src/utils/`, `tests/`, `tests/unit/`, `tests/integration/`, `tests/quality/`, `configs/`, `scripts/`, `notebooks/`, `notebooks/figures/`, `data/raw/`, `data/processed/`, `results/`, `plan/`, `docs/`. Every `src/` and `tests/` subdir gets an `__init__.py`. Completion: `find src tests -name __init__.py \| wc -l` ≥ 12 | S | — | | |
| TASK-004 | **Download FRED-MD snapshot** to `data/raw/current.csv`. Compute SHA-256: `sha256sum data/raw/current.csv > data/raw/current.csv.sha256`. Commit both. Add `data/raw/README.md` documenting download date, URL, and checksum. Completion: sha256 file exists and matches recomputed hash | S | TASK-003 | | |
| TASK-005 | **Create `src/utils/config.py`** with Pydantic v2 models: `DataConfig(url, csv_path, train_end, val_end, stride, seed)`, `ModelConfig(n_series, d_model, n_heads, n_layers, latent_dim, dropout, window_size)`, `TrainingConfig(batch_size, lr, weight_decay, max_epochs, patience, grad_clip)`, `ClusteringConfig(k_range, pca_variance_threshold, kmeans_n_init, random_state)`, `ExperimentConfig(data, model, training, clustering, mlflow_tracking_uri)`. Add `@classmethod def from_yaml(cls, path: Path) -> ExperimentConfig` using `yaml.safe_load` + Pydantic `model_validate`. Add `def set_all_seeds(seed: int)` function setting `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic=True`. Completion: `ExperimentConfig.from_yaml("configs/base.yaml")` returns validated config without error | M | TASK-001, TASK-003 | | |
| TASK-006 | **Create `configs/base.yaml`** with all fields matching Pydantic schema. Values: `data.csv_path="data/raw/current.csv"`, `data.train_end="2018-12-01"`, `data.val_end="2021-12-01"`, `data.stride=1`, `data.seed=42`, `model.d_model=64`, `model.n_heads=4`, `model.n_layers=2`, `model.latent_dim=6`, `model.dropout=0.1`, `model.window_size=12`, `training.batch_size=32`, `training.lr=1e-3`, `training.weight_decay=1e-4`, `training.max_epochs=200`, `training.patience=10`, `training.grad_clip=1.0`, `clustering.k_range=[3,4,5]`, `clustering.pca_variance_threshold=0.9`, `clustering.kmeans_n_init=20`, `clustering.random_state=42`, `mlflow_tracking_uri="mlruns"`. Create `configs/window_6.yaml`, `configs/window_12.yaml`, `configs/window_24.yaml` that override only `model.window_size`. Completion: all 4 YAML files parse without error via `ExperimentConfig.from_yaml()` | S | TASK-005 | | |
| TASK-007 | **Create `tests/conftest.py`** with shared fixtures: `mock_fred_md_csv` (tmp_path CSV with 102 rows × 22 cols: row1=headers, row2=tcodes as ints, rows 3-102=float data with 2 NaN injections and 1 outlier spike at 1e6), `mock_config` (ExperimentConfig with d_model=16, n_series=20, W=6, latent_dim=4, n_layers=1), `mock_model` (iTransformerAE from mock_config), `mock_embeddings` (np.random.RandomState(42).randn(50, 6)), `mock_labels` (np.array cycling [0,1,2] for 50 points), `device` fixture returning `torch.device("cuda" if available else "cpu")`. Completion: `pytest --collect-only` shows fixtures available | S | TASK-005 | | |
| TASK-008 | **Create `.gitignore`** with entries: `__pycache__/`, `*.pyc`, `.venv/`, `mlruns/`, `data/processed/`, `results/`, `*.egg-info/`, `.pytest_cache/`, `notebooks/figures/*.png`, `*.pt`, `*.pth`, `wandb/`. Completion: file exists | S | — | | |
| TASK-009 | **Create `README.md`** with project title, abstract (2 sentences), quickstart (`uv sync && python scripts/train.py --config configs/base.yaml`), project structure tree, and reference to `docs/PLAN_itransformer_repo.md`. Completion: file renders valid Markdown | S | TASK-003 | | |

**Phase 0 parallelization:** TASK-001, TASK-003, TASK-008 can run in parallel. TASK-002 depends on TASK-001. TASK-004 depends on TASK-003. TASK-005 depends on TASK-001+TASK-003. TASK-006 depends on TASK-005. TASK-007 depends on TASK-005.

**Phase 0 critical path:** TASK-001 → TASK-002 → TASK-005 → TASK-006 → TASK-007

**Phase 0 gate test:** `uv sync && python -c "from src.utils.config import ExperimentConfig; c = ExperimentConfig.from_yaml('configs/base.yaml'); print(c.model.d_model)"` prints `64`

---

### Phase 1 — Data Pipeline

- GOAL-002: Implement FRED-MD loading, tcode transformation, outlier removal, scaling, and PyTorch Dataset with full unit test coverage.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-010 | **Create `src/data/fred_md.py`** — implement `load_fred_md(csv_path: str) -> tuple[pd.DataFrame, pd.Series]`: read CSV, extract row 2 as tcodes (Series indexed by column name), parse rows 3+ as data with `sasdate` column parsed as `pd.Timestamp` index. Implement `remove_outliers(series: pd.Series, iqr_multiplier: float = 10.0) -> pd.Series`: set values where `\|x - median\| > iqr_multiplier * IQR` to `NaN`. Implement `apply_tcode(series: pd.Series, tcode: int) -> pd.Series` for codes 1-7. Implement `transform_panel(data: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame`: apply `remove_outliers` then `apply_tcode` per column, drop rows with any NaN. Completion: `transform_panel` on real FRED-MD returns DataFrame with ~800 rows × 128 cols, no NaN | M | TASK-003, TASK-004 | | |
| TASK-011 | **Create `src/data/preprocessing.py`** — implement `fit_scaler(train_data: pd.DataFrame) -> StandardScaler`: fit `sklearn.preprocessing.StandardScaler` on train. Implement `scale_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: transform all three with the SAME train-fitted scaler. Implement `split_by_date(data: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: temporal split using index comparison. Completion: val array mean ≠ 0.0 (proves no leakage), train array mean ≈ 0.0 | S | TASK-010 | | |
| TASK-012 | **Create `src/data/dataset.py`** — implement `FREDMDWindowDataset(Dataset)`: `__init__(self, data: np.ndarray, window_size: int, stride: int = 1)` stores data (T, N). `__len__` returns `(T - window_size) // stride + 1`. `__getitem__(idx)` returns `torch.FloatTensor` of shape `(window_size, N)`. Implement `create_dataloaders(train, val, test, window_size, stride, batch_size) -> tuple[DataLoader, DataLoader, DataLoader]`: train loader shuffled, val/test not shuffled. Completion: `len(dataset)` matches formula for W=6,12,24 with stride=1 and stride=W | S | TASK-011 | | |
| TASK-013 | **Create `tests/unit/test_data.py`** — 10 tests: `test_tcode_extraction_from_row2` (verify tcodes match injected row2 values), `test_outlier_removal_spike` (1e6 value → NaN), `test_outlier_removal_preserves_normal` (within 3*IQR unchanged), `test_tcode_1_noop` (level returns identical), `test_tcode_2_diff` (matches `pd.Series.diff()`), `test_tcode_5_log_diff` (matches `np.diff(np.log(x))`), `test_tcode_7_pct_change` (matches formula), `test_dataset_length_stride1` (T=100, W=6 → 95), `test_dataset_length_strideW` (T=100, W=6, stride=6 → 16), `test_dataset_item_shape` (returns (W, N) tensor). Completion: all 10 pass | M | TASK-010, TASK-012, TASK-007 | | |
| TASK-014 | **Create `tests/unit/test_preprocessing.py`** — 4 tests: `test_scaler_fit_train_only` (scaler.mean_ shape matches train cols), `test_no_leakage_val_mean` (scaled val mean per col ≠ 0, absolute mean > 0.01 for at least one col), `test_split_by_date_sizes` (train ends at train_end, val starts after), `test_scaled_train_mean_near_zero` (abs(mean) < 0.05). Completion: all 4 pass | S | TASK-011, TASK-007 | | |

**Phase 1 parallelization:** TASK-010 is the root. TASK-011 depends on TASK-010. TASK-012 depends on TASK-011. TASK-013 and TASK-014 can run in parallel after their deps are met.

**Phase 1 critical path:** TASK-010 → TASK-011 → TASK-012 → TASK-013

**Phase 1 gate test:** `pytest tests/unit/test_data.py tests/unit/test_preprocessing.py -v` — all 14 tests pass

---

### Phase 2 — Model Architecture

- GOAL-003: Implement iTransformer autoencoder (encoder + bottleneck + decoder) with correct tensor shapes and gradient flow, validated by unit tests.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-015 | **Create `src/model/layers.py`** — implement `VariateEmbedding(nn.Module)`: `__init__(window_size, d_model)` with `nn.Linear(window_size, d_model)` + `nn.LayerNorm(d_model)`. `forward(x: (B, N, W)) -> (B, N, d_model)`. Implement `TransformerEncoderBlock(nn.Module)`: `__init__(d_model, n_heads, d_ff=None, dropout=0.1)` with `nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)`, `nn.LayerNorm(d_model)` ×2, FFN as `nn.Sequential(Linear(d_model, d_ff), GELU, Dropout, Linear(d_ff, d_model), Dropout)` where `d_ff = d_ff or 4*d_model`. `forward(x: (B, N, d_model)) -> (B, N, d_model)` with pre-norm residual connections. Completion: shapes match docstrings, `d_model % n_heads == 0` enforced in init | M | TASK-003 | | |
| TASK-016 | **Create `src/model/encoder.py`** — implement `iTransformerEncoder(nn.Module)`: `__init__(n_series, window_size, d_model, n_heads, n_layers, latent_dim, dropout)`. Forward: (1) transpose input `(B, W, N) → (B, N, W)`, (2) `VariateEmbedding → (B, N, d_model)`, (3) `n_layers × TransformerEncoderBlock → (B, N, d_model)`, (4) mean pool dim=1 → `(B, d_model)`, (5) `nn.Linear(d_model, latent_dim)` → `z: (B, latent_dim)`. Return z. Completion: `encoder(x).shape == (B, latent_dim)` for x of shape `(B, W, N)` | M | TASK-015 | | |
| TASK-017 | **Create `src/model/decoder.py`** — implement `iTransformerDecoder(nn.Module)`: `__init__(n_series, window_size, d_model, n_layers, latent_dim, dropout)`. Forward: (1) `nn.Linear(latent_dim, d_model) → (B, d_model)`, (2) `unsqueeze(1).expand(-1, n_series, -1) → (B, N, d_model)`, (3) `n_layers × FFN blocks` (same structure as encoder FFN, each: `Linear(d_model, 4*d_model) → GELU → Dropout → Linear(4*d_model, d_model) → Dropout + residual + LayerNorm`), (4) `nn.Linear(d_model, window_size) → (B, N, W)`, (5) transpose → `(B, W, N)`. Return x_hat. Completion: `decoder(z).shape == (B, W, N)` for z of shape `(B, latent_dim)` | M | TASK-015 | | |
| TASK-018 | **Create `src/model/autoencoder.py`** — implement `iTransformerAE(nn.Module)`: `__init__` composes `iTransformerEncoder` + `iTransformerDecoder` (shared config from `ModelConfig`). `forward(x) -> (x_hat, z)`. `encode(x) -> z` (no decoder). `@classmethod from_config(cls, config: ModelConfig) -> iTransformerAE`. Completion: `model(x)` returns tuple of correct shapes, `model.encode(x).shape == (B, latent_dim)` | S | TASK-016, TASK-017 | | |
| TASK-019 | **Create `tests/unit/test_model.py`** — 8 tests: `test_variate_embedding_shape` (B=4, N=20, W=6, d=16 → (4,20,16)), `test_encoder_block_shape` (in/out same), `test_encoder_output_shape` (B=4 → (4, latent_dim)), `test_decoder_output_shape` ((B, latent_dim) → (B, W, N)), `test_autoencoder_roundtrip_shape` (x_hat.shape == x.shape), `test_no_nan_forward` (no NaN in x_hat or z), `test_gradient_flows` (backward on MSE loss → all `param.grad is not None`), `test_param_count_under_500k` (d_model=32 → count ≤ 500_000). Completion: all 8 pass | M | TASK-018, TASK-007 | | |

**Phase 2 parallelization:** TASK-016 and TASK-017 can run in parallel (both depend on TASK-015). TASK-018 waits for both. TASK-019 waits for TASK-018.

**Phase 2 critical path:** TASK-015 → TASK-016 → TASK-018 → TASK-019

**Phase 2 gate test:** `pytest tests/unit/test_model.py -v` — all 8 tests pass

---

### Phase 3 — Training Loop & Baselines

- GOAL-004: Implement the training loop with early stopping, checkpointing, MLflow logging, naive baseline computation, and linear AE baseline. Integration tests validate training mechanics.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-020 | **Create `src/training/losses.py`** — implement `reconstruction_loss(x: Tensor, x_hat: Tensor) -> Tensor`: MSE over all dims. Implement `naive_baseline_mse(dataloader: DataLoader, train_mean: np.ndarray, device) -> float`: compute MSE of predicting train_mean for every window. `train_mean` shape is `(N,)` broadcast to every timestep. Completion: naive baseline on mock data returns float > 0 | S | TASK-012 | | |
| TASK-021 | **Create `src/training/trainer.py`** — implement `Trainer` class: `__init__(model, train_loader, val_loader, config: TrainingConfig, device, mlflow_run)`. `train()` method: AdamW optimizer (lr, weight_decay from config), CosineAnnealingLR scheduler (T_max=max_epochs), gradient clipping (max_norm=grad_clip), epoch loop computing train_loss and val_loss, early stopping (patience on val_loss, track best_val_loss and counter), checkpoint best model to `results/best_model.pt` (state_dict + config + epoch + val_loss), MLflow logging per epoch (train_loss, val_loss, lr). Returns dict with `best_val_loss`, `best_epoch`, `train_losses`, `val_losses`. Implement `extract_embeddings(model, dataloader, device) -> np.ndarray`: no_grad forward, concatenate z across batches. Completion: trainer.train() runs 5 epochs on mock data without error, returns dict | L | TASK-018, TASK-020, TASK-005 | | |
| TASK-022 | **Create `src/training/baselines.py`** — implement `LinearAutoencoder(nn.Module)`: `__init__(input_dim, latent_dim)` with single `nn.Linear` encoder and decoder (flatten window to input_dim = W*N, compress to latent_dim, reconstruct). `forward(x) -> (x_hat, z)`. Implement `RandomEmbeddings`: class that generates `np.random.RandomState(seed).randn(n_samples, latent_dim)`. Implement `PCAOnlyBaseline`: takes raw windowed data, flattens to (n_samples, W*N), fits PCA on train, transforms all splits, runs K-Means. Completion: all three baseline classes instantiate and produce embeddings of correct shape | M | TASK-012, TASK-005 | | |
| TASK-023 | **Create `tests/integration/test_training.py`** — 5 tests: `test_loss_decreases_5_epochs` (epoch 5 train_loss < epoch 1 train_loss on overfit-friendly mock data of 20 identical samples), `test_no_nan_loss` (10 steps, loss.isnan() → False), `test_early_stopping_triggers` (set patience=2, feed flat val_loss → stops before max_epochs), `test_checkpoint_saves_and_restores` (save, reload state_dict, embeddings identical within 1e-6), `test_mlflow_logs_metrics` (mock MLflow, verify log_metric called with "val_loss"). Completion: all 5 pass | M | TASK-021, TASK-007 | | |

**Phase 3 parallelization:** TASK-020 and TASK-022 can start once their deps are ready. TASK-021 depends on TASK-020. TASK-023 depends on TASK-021.

**Phase 3 critical path:** TASK-020 → TASK-021 → TASK-023

**Phase 3 gate test:** `pytest tests/integration/test_training.py -v` — all 5 pass. MLflow tracking URI directory `mlruns/` created.

---

### Phase 4 — EDA Notebook

- GOAL-005: Produce the complete Exploratory Data Analysis notebook with all 10 sections, matplotlib/seaborn only, figures saved to `notebooks/figures/`.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-024 | **Create `notebooks/00_eda.ipynb`** — §0 Setup+Imports (import src.data modules, set seeds, configure matplotlib), §1 FRED-MD Load (parse CSV, show shape, head(5), tcodes distribution bar chart), §2 Missing Data (null % heatmap using seaborn, temporal null pattern line plot), §3 Distributions (histograms of top-12 series by kurtosis on train split, Jarque-Bera test table), §4 Stationarity (ADF + KPSS p-values per series, flag ambiguous cases in table, color-code), §5 Split Visualization (timeline with vertical split lines + KS test table: val vs train, test vs train per series), §6 Correlation (hierarchical clustermap of train, separate clustermap of val, visual comparison), §7 Window Statistics (boxplots of window means/stds for W=6,12,24), §8 Baseline PCA on Raw Data (scree plot, PC1 vs PC2 scatter colored by date, variance explained table), §9 Regime Proxy (ICSS structural breakpoints overlaid with NBER recession bars), §10 Summary Table (key findings in DataFrame). All figures saved to `notebooks/figures/eda_*.png` at 300 DPI. Completion: notebook runs top-to-bottom without error, ≥10 PNG files in figures dir | XL | TASK-010, TASK-011 | | |

**Phase 4 parallelization:** TASK-024 is independent of Phases 2-3 — can run in parallel with model development once Phase 1 is complete.

**Phase 4 gate test:** `jupyter nbconvert --execute notebooks/00_eda.ipynb` exits 0, `ls notebooks/figures/eda_*.png | wc -l` ≥ 10

---

### Phase 5 — Evaluation Modules

- GOAL-006: Implement all embedding quality, clustering, and statistical testing modules with the adaptive PCA rule, block bootstrap, and hierarchical testing framework.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-025 | **Create `src/evaluation/embedding_quality.py`** — implement `reconstruction_mse(model, dataloader, device) -> float`, `naive_baseline_mse(dataloader, train_mean) -> float`, `check_embedding_collapse(embeddings: np.ndarray, threshold: float = 1e-5) -> dict` returning `{per_dim_variance: np.ndarray, collapsed_dims: list[int], is_collapsed: bool}`, `compute_effective_rank(embeddings: np.ndarray) -> float` using `exp(entropy(normalized_singular_values))`, `compute_isotropy(embeddings: np.ndarray) -> float` as mean pairwise cosine similarity (sample 1000 pairs if n>100). Completion: all functions return correct types on mock_embeddings fixture | M | TASK-003 | | |
| TASK-026 | **Create `src/evaluation/clustering.py`** — implement `adaptive_pca(embeddings_train: np.ndarray, latent_dim: int, variance_threshold: float = 0.9) -> tuple[PCA, int]`: fit PCA with `n_components = min(latent_dim - 1, components_for_threshold_var)`, return fitted PCA and actual n_components used. `apply_pca(embeddings: np.ndarray, pca: PCA) -> np.ndarray`. `fit_kmeans(embeddings_pca: np.ndarray, k: int, n_init: int = 20, random_state: int = 42) -> KMeans`. `compute_clustering_metrics(embeddings_pca: np.ndarray, labels: np.ndarray) -> dict` returning `{silhouette, davies_bouldin, calinski_harabasz}`. `select_k(embeddings_pca: np.ndarray, k_range: list[int], n_init: int, random_state: int) -> dict` returning `{best_k, scores: dict[int, float]}` by argmax silhouette. `clustering_stability(embeddings_pca: np.ndarray, k: int, n_runs: int = 10) -> float` returning mean ARI across runs with different seeds. Completion: `adaptive_pca` with latent_dim=6, threshold=0.9 returns n_components ≤ 5 | M | TASK-003 | | |
| TASK-027 | **Create `src/evaluation/statistical_tests.py`** — implement `kruskal_wallis_per_dim(embeddings: np.ndarray, labels: np.ndarray) -> dict` returning `{dim_i: {H_stat, p_value, p_corrected}}` with BH correction via `scipy.stats.false_discovery_control`. `pairwise_wilcoxon(embeddings: np.ndarray, labels: np.ndarray) -> dict` returning per-dim per-pair `{(i,j): {stat, p_value, p_corrected}}`, only computed if KW is significant for that dim. `temporal_consistency_score(labels: np.ndarray, dates: np.ndarray) -> dict` returning `{n_transitions, transition_rate, mean_regime_length}`. `block_bootstrap_ci(embeddings: np.ndarray, labels: np.ndarray, metric_fn: Callable, n_bootstrap: int = 1000, block_size: int = 12, ci: float = 0.95, seed: int = 42) -> dict` returning `{mean, ci_lower, ci_upper, std}` using circular block bootstrap. Completion: KW returns dict with all dims, p_corrected ≤ 1.0 | L | TASK-003 | | |

**Phase 5 parallelization:** TASK-025, TASK-026, TASK-027 are ALL independent — run in parallel.

**Phase 5 critical path:** All three tasks same length relative to each other. TASK-027 is largest (L).

**Phase 5 gate test:** `python -c "from src.evaluation.clustering import adaptive_pca; from src.evaluation.statistical_tests import block_bootstrap_ci; print('OK')"` — imports succeed

---

### Phase 6 — Quality & Statistical Tests

- GOAL-007: Implement the complete test suite that validates embedding quality, clustering quality, and statistical significance — these are the thesis acceptance criteria.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-028 | **Create `tests/quality/test_embedding_quality.py`** — 4 tests (all marked `@pytest.mark.quality`): `test_reconstruction_beats_naive_baseline` (train a tiny model 50 epochs on mock data, assert model_mse < naive_mse), `test_no_embedding_collapse` (all per_dim_variance > 1e-5), `test_effective_rank_above_2` (effective_rank > 2.0), `test_isotropy_below_threshold` (isotropy < 0.9, i.e., embeddings not all pointing same direction). Completion: all 4 pass on mock data | M | TASK-025, TASK-021, TASK-007 | | |
| TASK-029 | **Create `tests/quality/test_clustering_quality.py`** — 5 tests (all `@pytest.mark.quality`): `test_silhouette_positive` (silhouette > 0 on mock embeddings with planted clusters), `test_adaptive_pca_components` (n_components ≤ latent_dim - 1), `test_kw_significant_on_half_dims` (on planted clusters: ≥ ceil(d/2) dims have BH-corrected p < 0.05), `test_clustering_stability_ari` (ARI > 0.7 on planted clusters), `test_k_selection_returns_valid_k` (best_k ∈ k_range). Completion: all 5 pass | M | TASK-026, TASK-027, TASK-007 | | |
| TASK-030 | **Create `tests/quality/test_baselines.py`** — 3 tests: `test_linear_ae_produces_embeddings` (shape (n, latent_dim)), `test_random_embeddings_shape` (shape (n, latent_dim)), `test_pca_only_baseline_produces_labels` (labels.shape == (n,), unique labels ∈ k_range). Completion: all 3 pass | S | TASK-022, TASK-007 | | |

**Phase 6 parallelization:** TASK-028, TASK-029, TASK-030 can all run in parallel.

**Phase 6 critical path:** Blocked by Phases 3 and 5 completing. TASK-028 and TASK-029 are equal length (M).

**Phase 6 gate test:** `pytest tests/quality/ -v -m quality` — all 12 tests pass

---

### Phase 7 — Sweep Infrastructure

- GOAL-008: Implement the sweep script that runs all 36 hyperparameter combinations sequentially on one GPU, logs everything to MLflow, and produces a consolidated results CSV.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-031 | **Create `scripts/train.py`** — CLI entry point: `argparse` with `--config` (path to YAML). Flow: (1) load config via `ExperimentConfig.from_yaml`, (2) `set_all_seeds`, (3) load/transform/split/scale data, (4) create dataloaders (overlapping stride=1 for train, non-overlapping stride=W for val/test eval), (5) instantiate model, (6) start MLflow run with `mlflow.start_run`, log all config params, (7) train, (8) extract embeddings (non-overlapping), (9) adaptive PCA + K-Means + metrics, (10) log metrics to MLflow, (11) save model checkpoint. Completion: `python scripts/train.py --config configs/base.yaml` runs to completion, `mlruns/` populated | L | TASK-021, TASK-026, TASK-027 | | |
| TASK-032 | **Create `scripts/evaluate.py`** — CLI entry point: `argparse` with `--config`, `--checkpoint`. Flow: (1) load config, (2) load data + preprocess, (3) load model from checkpoint, (4) extract embeddings (non-overlapping stride=W), (5) adaptive PCA, (6) K-Means with select_k, (7) ALL metrics: reconstruction MSE, naive baseline MSE, effective rank, isotropy, silhouette (with block bootstrap CI), DB, CH, KW test, pairwise Wilcoxon, temporal consistency, (8) print results table, (9) save to `results/eval_results.json`. Completion: produces valid JSON with all metric keys | M | TASK-031 | | |
| TASK-033 | **Create `scripts/sweep.py`** — CLI entry point: `argparse` with `--base-config`. Grid: `window_sizes=[6,12,24]`, `latent_dims=[6,7,8,9]`, `k_values=[3,4,5]` → 36 combos. Flow per combo: (1) deep-copy base config, (2) override `model.window_size`, `model.latent_dim`, `clustering.k_range=[k]`, (3) set `mlflow.set_experiment("sweep")`, (4) `mlflow.start_run(run_name=f"W{w}_d{d}_K{k}")`, (5) full train → evaluate pipeline, (6) log all metrics + params, (7) `mlflow.end_run()`, (8) append row to `results/sweep_results.csv`. After all combos: (9) select best combo per objective (lowest val_recon_mse, highest val_silhouette, highest KW_significant_dims_ratio), (10) log summary to `results/sweep_summary.json`. Memory management: `del model; torch.cuda.empty_cache()` between runs. Estimated runtime: ~36 × 5-15 min = 3-9 hours on single GPU. Completion: `results/sweep_results.csv` has 36 rows, `results/sweep_summary.json` exists | L | TASK-031, TASK-032 | | |

**Phase 7 parallelization:** TASK-031 must complete first. TASK-032 depends on TASK-031. TASK-033 depends on both.

**Phase 7 critical path:** TASK-031 → TASK-032 → TASK-033

**Phase 7 gate test:** `python scripts/sweep.py --base-config configs/base.yaml` (run with reduced epochs for smoke test: override max_epochs=3). Verify `results/sweep_results.csv` has 36 rows.

---

### Phase 8 — Thesis Artifacts (Figures & Tables)

- GOAL-009: Generate all publication-quality figures and tables needed for the thesis document from trained models and sweep results.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-034 | **Create `notebooks/01_embedding_analysis.ipynb`** — §0 Load best model from sweep + data, §1 Extract embeddings (train/val/test, non-overlapping), §2 Embedding geometry (per-dim variance bar chart, effective rank annotation, isotropy annotation), §3 PCA on embeddings (scree plot with 90% variance line, 2D scatter colored by date with regime labels), §4 K selection (silhouette vs K line plot with CI error bars, DB vs K, CH vs K), §5 Regime visualization (time series of cluster labels overlaid with NBER recession bars), §6 Statistical validation (KW test results table as heatmap, pairwise Wilcoxon significant pairs matrix), §7 Baseline comparison (grouped bar chart: iTransformer vs Linear AE vs PCA-only vs Random for silhouette/DB/CH), §8 Summary metrics table. All figs → `notebooks/figures/analysis_*.png` 300 DPI. Completion: notebook runs top-to-bottom, ≥8 PNG files | L | TASK-033 | | |
| TASK-035 | **Create `scripts/generate_tables.py`** — read `results/sweep_results.csv` and `results/eval_results.json`. Produce LaTeX tables: (1) `results/tables/sweep_results.tex` — full 36-combo table with W, d, K, val_MSE, silhouette±CI, DB, CH, KW_sig_ratio, (2) `results/tables/best_config.tex` — single-row table of best combo, (3) `results/tables/baseline_comparison.tex` — 4 baselines × metrics table, (4) `results/tables/statistical_tests.tex` — KW H-stat + BH-corrected p per dimension. Completion: 4 `.tex` files exist and compile in LaTeX without error | M | TASK-033 | | |
| TASK-036 | **Create `scripts/generate_figures.py`** — read sweep results + best model. Produce: (1) `results/figures/training_curves.png` — train/val loss over epochs for best config, (2) `results/figures/sweep_heatmap.png` — W×d heatmap of best silhouette across K, (3) `results/figures/embedding_2d.png` — PCA 2D scatter train+val+test with cluster coloring, (4) `results/figures/regime_timeline.png` — cluster labels over time with recession bars, (5) `results/figures/baseline_comparison.png` — grouped bar chart, (6) `results/figures/bootstrap_ci.png` — silhouette CIs for each combo. All 300 DPI, matplotlib+seaborn. Completion: 6 PNG files in `results/figures/` | M | TASK-033 | | |
| TASK-037 | **Create `src/utils/viz.py`** — shared visualization utilities: `set_thesis_style()` (set font sizes, figure size defaults, seaborn style "whitegrid"), `save_figure(fig, name, dpi=300)` (saves to both `notebooks/figures/` and `results/figures/`), `plot_training_curves(train_losses, val_losses, title)`, `plot_pca_scatter(embeddings_2d, labels, dates, title)`, `plot_silhouette_vs_k(k_range, scores, cis, title)`, `plot_regime_timeline(labels, dates, recessions, title)`, `plot_baseline_comparison(metrics_dict, title)`. Completion: `set_thesis_style()` runs without error, creates consistent styling | M | TASK-003 | | |

**Phase 8 parallelization:** TASK-037 can be done early (only needs TASK-003). TASK-034, TASK-035, TASK-036 can run in parallel (all depend on TASK-033 results).

**Phase 8 critical path:** TASK-033 → TASK-034 (longest in this phase)

**Phase 8 gate test:** `ls results/figures/*.png | wc -l` ≥ 6, `ls results/tables/*.tex | wc -l` ≥ 4

---

### Phase 9 — Final Integration & Validation

- GOAL-010: Run the complete pipeline end-to-end, verify data integrity, validate all thesis acceptance criteria, and produce the final reproducibility check.

| Task | Description | Size | Depends | Completed | Date |
|------|-------------|------|---------|-----------|------|
| TASK-038 | **Full test suite run**: `pytest tests/ -v --cov=src --cov-report=html`. Completion: all tests pass, coverage report generated, coverage > 80% on `src/` | S | All TASK-013 through TASK-030 | | |
| TASK-039 | **Data integrity check**: re-compute SHA-256 of `data/raw/current.csv`, assert matches committed checksum. Verify transform_panel output dimensions. Verify train/val/test split date boundaries. Script: `scripts/verify_data.py`. Completion: script exits 0 | S | TASK-004, TASK-010 | | |
| TASK-040 | **Reproducibility check**: run `scripts/train.py --config configs/base.yaml` twice with same seed. Assert embeddings are bit-identical (`np.allclose(z1, z2, atol=1e-7)`). Script: `scripts/verify_reproducibility.py`. Completion: assertion passes | M | TASK-031 | | |
| TASK-041 | **Thesis acceptance criteria validation**: create `scripts/validate_thesis.py` that loads best model from sweep, checks: (1) model_mse < naive_mse ✓, (2) effective_rank > 2 ✓, (3) silhouette > 0 ✓, (4) ≥ ceil(d/2) KW-significant dims ✓, (5) clustering ARI stability > 0.7 ✓, (6) iTransformer silhouette > random baseline silhouette ✓, (7) block bootstrap 95% CI for silhouette does not include 0 ✓. Print pass/fail for each. Completion: ≥ 5 of 7 criteria pass (allowing 2 to be flagged for discussion in thesis) | M | TASK-033 | | |

**Phase 9 parallelization:** TASK-038, TASK-039, TASK-040, TASK-041 can all run in parallel.

**Phase 9 critical path:** All are independent; longest is TASK-040 and TASK-041 (M).

**Phase 9 gate test:** All 4 scripts exit 0. Thesis validation prints ≥ 5/7 PASS.

## 3. Alternatives

- **ALT-001**: PyTorch Lightning instead of raw training loop — rejected because project scope is small (single model, single GPU), and a raw loop gives full visibility for the thesis defense
- **ALT-002**: Weights & Biases instead of MLflow — rejected per Agent 1 decision for local-first, no cloud dependency
- **ALT-003**: Optuna instead of grid sweep — rejected because 36 combos is exhaustively enumerable; Optuna adds complexity without benefit at this scale
- **ALT-004**: UMAP instead of PCA — rejected per Agent 4 decision; PCA is deterministic and reproducible, UMAP has stochastic components
- **ALT-005**: Hydra instead of Pydantic+YAML — rejected per Agent 1; Pydantic gives validation at parse time, Hydra is heavier for this scope

## 4. Dependencies

- **DEP-001**: `torch>=2.2,<2.3` — core deep learning framework (iTransformer encoder/decoder)
- **DEP-002**: `scikit-learn>=1.4,<1.5` — PCA, KMeans, StandardScaler, silhouette_score, davies_bouldin_score, calinski_harabasz_score
- **DEP-003**: `scipy>=1.13,<1.14` — kruskal, wilcoxon, false_discovery_control, entropy
- **DEP-004**: `pandas>=2.2,<2.3` — FRED-MD CSV parsing, temporal indexing
- **DEP-005**: `numpy>=1.26,<2.0` — array operations, random seeding
- **DEP-006**: `matplotlib>=3.8,<3.9` — all figures
- **DEP-007**: `seaborn>=0.13,<0.14` — statistical plots, heatmaps
- **DEP-008**: `pydantic>=2.0,<3.0` — config validation
- **DEP-009**: `pyyaml>=6.0` — YAML config loading
- **DEP-010**: `mlflow>=2.12,<2.13` — experiment tracking
- **DEP-011**: `pytest>=8.0` + `pytest-cov` — testing
- **DEP-012**: `tqdm` — progress bars
- **DEP-013**: `ruff` — linting/formatting
- **DEP-014**: `uv` — dependency management (system tool, not a pip package)

## 5. Files

- **FILE-001**: `pyproject.toml` — project metadata, dependencies, tool config
- **FILE-002**: `uv.lock` — pinned dependency lock file
- **FILE-003**: `src/utils/config.py` — Pydantic config models + seed setting
- **FILE-004**: `src/data/fred_md.py` — FRED-MD loading, tcodes, outlier removal
- **FILE-005**: `src/data/preprocessing.py` — StandardScaler, temporal splits
- **FILE-006**: `src/data/dataset.py` — PyTorch Dataset + DataLoader factory
- **FILE-007**: `src/model/layers.py` — VariateEmbedding, TransformerEncoderBlock
- **FILE-008**: `src/model/encoder.py` — iTransformerEncoder
- **FILE-009**: `src/model/decoder.py` — iTransformerDecoder
- **FILE-010**: `src/model/autoencoder.py` — iTransformerAE
- **FILE-011**: `src/training/losses.py` — reconstruction + naive baseline losses
- **FILE-012**: `src/training/trainer.py` — Trainer class with early stopping + MLflow
- **FILE-013**: `src/training/baselines.py` — LinearAE, Random, PCA-only baselines
- **FILE-014**: `src/evaluation/embedding_quality.py` — reconstruction, collapse, effective rank, isotropy
- **FILE-015**: `src/evaluation/clustering.py` — adaptive PCA, K-Means, metrics, stability
- **FILE-016**: `src/evaluation/statistical_tests.py` — KW, Wilcoxon, block bootstrap, temporal consistency
- **FILE-017**: `src/utils/viz.py` — shared thesis-style plotting utilities
- **FILE-018**: `configs/base.yaml`, `configs/window_6.yaml`, `configs/window_12.yaml`, `configs/window_24.yaml`
- **FILE-019**: `scripts/train.py`, `scripts/evaluate.py`, `scripts/sweep.py`
- **FILE-020**: `scripts/generate_tables.py`, `scripts/generate_figures.py`
- **FILE-021**: `scripts/verify_data.py`, `scripts/verify_reproducibility.py`, `scripts/validate_thesis.py`
- **FILE-022**: `notebooks/00_eda.ipynb`, `notebooks/01_embedding_analysis.ipynb`
- **FILE-023**: `tests/conftest.py` + all test files (unit/integration/quality)
- **FILE-024**: `data/raw/current.csv`, `data/raw/current.csv.sha256`, `data/raw/README.md`

## 6. Testing

- **TEST-001**: Unit tests — data pipeline (`tests/unit/test_data.py`): 10 tests verifying tcode extraction, outlier removal, transformations, dataset shapes. Gate: Phase 1.
- **TEST-002**: Unit tests — preprocessing (`tests/unit/test_preprocessing.py`): 4 tests verifying scaler fitting, no leakage, split correctness. Gate: Phase 1.
- **TEST-003**: Unit tests — model (`tests/unit/test_model.py`): 8 tests verifying shapes, NaN-free forward, gradient flow, param count. Gate: Phase 2.
- **TEST-004**: Integration tests — training (`tests/integration/test_training.py`): 5 tests verifying loss decrease, early stopping, checkpointing, MLflow logging. Gate: Phase 3.
- **TEST-005**: Quality tests — embedding quality (`tests/quality/test_embedding_quality.py`): 4 tests verifying reconstruction beats baseline, no collapse, effective rank, isotropy. Gate: Phase 6.
- **TEST-006**: Quality tests — clustering quality (`tests/quality/test_clustering_quality.py`): 5 tests verifying silhouette > 0, adaptive PCA, KW significance, stability, k selection. Gate: Phase 6.
- **TEST-007**: Quality tests — baselines (`tests/quality/test_baselines.py`): 3 tests verifying baseline embedding shapes and labels. Gate: Phase 6.
- **TEST-008**: Full coverage run (`pytest tests/ --cov=src`): target > 80% line coverage. Gate: Phase 9.

## 7. Risks & Assumptions

- **RISK-001**: Overlapping windows (stride=1) inflate apparent sample size — mitigation: all evaluation uses non-overlapping stride=W, block bootstrap for CIs
- **RISK-002**: Small validation set (36 non-overlapping months) makes silhouette noisy — mitigation: bootstrap CIs (n=1000, block_size=12), report CI width in thesis
- **RISK-003**: COVID-era data in val set is distribution-shifted — mitigation: document reconstruction loss spike, interpret as meaningful OOD signal
- **RISK-004**: PCA fitted on train may explain < 90% variance on val/test — mitigation: log explained variance on all splits, flag if < 80% as OOD evidence
- **RISK-005**: K-Means non-determinism across seeds — mitigation: n_init=20, clustering stability ARI > 0.7, report ARI in thesis
- **RISK-006**: d_model=512 (original iTransformer) would massively overfit ~330 samples — mitigation: use d_model ∈ {32, 64} only
- **RISK-007**: Sweep runtime (36 combos × 200 epochs) may exceed available GPU time — mitigation: early stopping reduces actual epochs, estimate 3-9 hours
- **ASSUMPTION-001**: FRED-MD CSV format is stable (row 2 = tcodes, sasdate column) — validated by SHA-256 checksum
- **ASSUMPTION-002**: Single GPU (e.g., RTX 3060 or better) available for sweep — no distributed training needed at this scale
- **ASSUMPTION-003**: All 128 FRED-MD series survive outlier removal + tcode transforms with < 5% rows dropped

## 8. Related Specifications / Further Reading

- [iTransformer Paper (Liu et al., ICLR 2024)](https://arxiv.org/abs/2310.06625)
- [Official iTransformer Implementation](https://github.com/thuml/iTransformer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [FRED-MD Specification (McCracken & Ng 2016)](http://www.columbia.edu/~sn2294/papers/freddata.pdf)
- [sklearn Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html)
- [Block Bootstrap for Dependent Data](https://en.wikipedia.org/wiki/Block_bootstrapping)
- [Existing Plan Document](../docs/PLAN_itransformer_repo.md)

---

## Appendix A: Critical Path Summary

```
TASK-001 → TASK-002 → TASK-005 → TASK-006                     (Phase 0: config)
                    ↘ TASK-003 → TASK-004                       (Phase 0: scaffold)
                                ↘ TASK-010 → TASK-011 → TASK-012 (Phase 1: data)
                                           ↘ TASK-024            (Phase 4: EDA, parallel)
TASK-015 → TASK-016 ↘
           TASK-017 → TASK-018 → TASK-019                       (Phase 2: model)
TASK-020 → TASK-021 → TASK-023                                  (Phase 3: training)
TASK-025, TASK-026, TASK-027 (all parallel)                     (Phase 5: evaluation)
TASK-028, TASK-029, TASK-030 (all parallel)                     (Phase 6: quality tests)
TASK-031 → TASK-032 → TASK-033                                  (Phase 7: sweep)
TASK-034, TASK-035, TASK-036 (all parallel after TASK-033)      (Phase 8: artifacts)
TASK-038, TASK-039, TASK-040, TASK-041 (all parallel)           (Phase 9: validation)

OVERALL CRITICAL PATH:
TASK-001 → TASK-005 → TASK-010 → TASK-011 → TASK-012 → TASK-020 → TASK-021 → TASK-031 → TASK-033 → TASK-034
```

## Appendix B: Sweep Execution Strategy (36 Combos, 1 GPU)

```
Sequential execution order (memory-safe):
  for W in [6, 12, 24]:          # outer: changes dataset shape
    rebuild dataloaders           # only 3 times total
    for d in [6, 7, 8, 9]:       # middle: changes model architecture
      for K in [3, 4, 5]:        # inner: only affects post-hoc clustering
        train model(W, d)        # reuse model across K values
        extract embeddings
        for k in [K]:
          run clustering(k)
          log metrics to MLflow
        del model
        torch.cuda.empty_cache()

Optimization: for each (W, d) pair, train ONCE, then evaluate 3 K values.
This reduces training runs from 36 → 12 (one per W×d pair).
Effective combos: 12 training runs + 36 clustering evaluations.
```

## Appendix C: Task Count Summary

| Phase | Tasks | Parallelizable | Estimated Total |
|-------|-------|---------------|-----------------|
| Phase 0 — Foundation | 9 | TASK-001,003,008 parallel | ~6h |
| Phase 1 — Data Pipeline | 5 | TASK-013,014 parallel | ~5h |
| Phase 2 — Model | 5 | TASK-016,017 parallel | ~6h |
| Phase 3 — Training+Baselines | 4 | TASK-020,022 parallel | ~6h |
| Phase 4 — EDA Notebook | 1 | Parallel with Phase 2-3 | ~8h |
| Phase 5 — Evaluation | 3 | All 3 parallel | ~4h |
| Phase 6 — Quality Tests | 3 | All 3 parallel | ~3h |
| Phase 7 — Sweep | 3 | Sequential | ~10h |
| Phase 8 — Artifacts | 4 | TASK-034,035,036 parallel | ~6h |
| Phase 9 — Validation | 4 | All 4 parallel | ~3h |
| **TOTAL** | **41** | | |