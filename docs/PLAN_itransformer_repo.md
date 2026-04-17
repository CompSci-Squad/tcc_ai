# iTransformer Embedding Repo — Implementation Plan
**TCC 2026 | Instituto Mauá de Tecnologia**
**Date:** 2026-04-16

---

## Research Basis

**iTransformer (Liu et al., ICLR 2024 Spotlight — arXiv:2310.06625)**
- Encoder-only Transformer. Key inversion: each variate (feature) = one token, not each time step.
- Token embed: linear projection of window W time-steps → d_model (original: d_model=512, n_heads=8, e_layers=3).
- Self-attention captures cross-variate correlations. FFN per variate token → series representation.
- Official impl: https://github.com/thuml/iTransformer

**Our Adaptation (encoder → autoencoder for embeddings):**
- Input shape: (B, W, N) — W∈{6,12,24} months, N=128 FRED-MD series
- Invert → (B, N, W): N=128 tokens each from W dims
- Linear embed: W → d_model (use d_model∈{32,64} — NOT 512, small dataset)
- L Transformer encoder blocks → per-variate embeddings (B, N, d_model)
- Mean pool across N → global z ∈ ℝᵈ, d∈{6..9}
- Mirror decoder → reconstruct (B, W, N)
- Loss: MSE reconstruction

**Data constraints (critical):**
- 840 rows × 128 cols (FRED-MD monthly)
- Train: 1990–2018 (~336 months) | Val: 2019–2021 (~36 months) | Test: 2022–2025 (~48 months)
- Rolling stride=1 → ~330/30/42 overlapping samples (temporally correlated — see Risk Flags)
- Small dataset → mandatory: dropout, weight_decay, early stopping

**Embedding quality framework (multi-layer):**
1. Reconstruction: model MSE < naive baseline (predict train mean)
2. Geometry: no collapse (per-dim variance > ε), effective rank > 2
3. Clustering intrinsic: Silhouette > 0, Davies-Bouldin, Calinski-Harabasz
4. Statistical: KW test across cluster labels on embedding dims (significant)
5. Temporal: regime labels align with ICSS breakpoints (qualitative)

---

## Repository Structure

```
tcc-itransformer/
├── src/
│   ├── data/
│   │   ├── fred_md.py           # FRED-MD load, tcode transforms, outlier removal
│   │   ├── preprocessing.py     # StandardScaler per series (fit on train only)
│   │   └── dataset.py           # PyTorch Dataset, rolling window sampler
│   ├── model/
│   │   ├── layers.py            # VariateEmbedding, TransformerEncoderBlock
│   │   ├── encoder.py           # iTransformer encoder (N-token attention)
│   │   ├── decoder.py           # Mirror decoder for reconstruction
│   │   └── autoencoder.py       # Full AE: encoder + bottleneck + decoder
│   ├── training/
│   │   ├── trainer.py           # Train loop, early stopping, LR scheduler
│   │   ├── losses.py            # MSE + naive baseline
│   │   └── callbacks.py         # Checkpoint, logging
│   ├── evaluation/
│   │   ├── embedding_quality.py # Reconstruction metrics + geometry checks
│   │   ├── clustering.py        # PCA, K-Means, Silhouette, DB, CH
│   │   └── statistical_tests.py # KW, Wilcoxon, BH correction (scipy)
│   └── utils/
│       ├── config.py            # Dataclass / YAML config
│       └── viz.py               # matplotlib/seaborn (NO Plotly)
├── notebooks/
│   ├── 00_eda.ipynb             # Full EDA of FRED-MD data
│   └── 01_embedding_analysis.ipynb  # Post-training embedding quality
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── test_data.py         # FRED-MD parsing, tcode, outlier removal
│   │   ├── test_model.py        # Forward pass shapes, loss, grad flow
│   │   └── test_preprocessing.py # Scaler, window creation
│   ├── integration/
│   │   └── test_training.py     # Train loop sanity checks
│   └── quality/
│       ├── test_embedding_quality.py  # Reconstruction < baseline, no collapse
│       └── test_clustering_quality.py # Silhouette > 0, KW significant
├── configs/
│   ├── base.yaml
│   ├── window_6.yaml
│   ├── window_12.yaml
│   └── window_24.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── sweep.py                 # W × d × K grid search (36 combos)
├── requirements.txt
├── pyproject.toml               # pytest config + package metadata
└── README.md
```

---

## Phase 0: Documentation Discovery (ALWAYS FIRST)

Read actual source before writing any code.

**Tasks:**
1. Read iTransformer model source:
   - https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py
   - Extract: exact DataEmbedding class, Encoder class, input shape handling
   - Note: how (B, seq_len, n_vars) → inverted → (B, n_vars, seq_len)
2. Read attention layers:
   - https://github.com/thuml/Time-Series-Library/blob/main/layers/SelfAttention_Family.py
   - Extract: AttentionLayer constructor params
3. Read FRED-MD spec (McCracken & Ng 2016):
   - http://www.columbia.edu/~sn2294/papers/freddata.pdf
   - Confirm: row 2 = tcode, row 1 = headers, row 3+ = data
   - Confirm: tcode transformation table (codes 1–7)
4. Read sklearn docs: PCA, KMeans, silhouette_score, davies_bouldin_score, calinski_harabasz_score

**Output:** `docs/api_reference.md` — exact API signatures, no invented methods.

**Anti-patterns:**
- Do NOT invent `.encode()` if not in iTransformer source
- Do NOT use tcode from row 1 — it is row 2
- Do NOT fit scaler on full dataset — train split only

---

## Phase 1: Data Pipeline (`src/data/`)

### 1A. `fred_md.py`

```python
def load_fred_md(url_or_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
        data: DataFrame, index=sasdate (parsed), columns=series_id
              rows from row 3 onward (NOT tcode row)
        tcodes: Series, index=series_id, values=int tcode
    Critical: Row 2 = tcode. Extract BEFORE parsing data rows.
    sasdate format: M/D/YYYY → pd.Timestamp
    """

def remove_outliers(series: pd.Series) -> pd.Series:
    """Set |x - median| > 10 * IQR to NaN. Pre-transformation."""

def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """
    1: level | 2: Δx | 3: Δ²x | 4: log(x) | 5: Δlog(x) | 6: Δ²log(x) | 7: Δ(x/x_{t-1}−1)
    """

def transform_panel(data: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    """Outlier removal then tcode transform per series."""
```

### 1B. `preprocessing.py`

```python
def fit_scaler(train_data: pd.DataFrame) -> StandardScaler:
    """Fit on train ONLY. No leakage."""

def scale_splits(train, val, test, scaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform all splits with train scaler."""
```

### 1C. `dataset.py`

```python
class FREDMDWindowDataset(Dataset):
    """
    Args: data (T, N), window_size ∈ {6,12,24}, stride int
    Returns: x (window_size, N), idx int
    """
```

**Verification checklist:**
- [ ] tcodes extracted from row 2, not row 1
- [ ] outlier_removal: spike > 10*IQR → NaN; normal values → unchanged
- [ ] tcode=5 on known series matches manual Δlog calc
- [ ] Scaler fit train only → val mean ≠ 0
- [ ] Dataset len: stride=W → (T-W)//W+1; stride=1 → T-W

---

## Phase 2: Model (`src/model/`)

### 2A. `layers.py`

```python
class VariateEmbedding(nn.Module):
    """
    Input:  (B, N, W) — N variates, W time steps each
    Output: (B, N, d_model)
    Uses:   nn.Linear(W, d_model) + LayerNorm
    """

class TransformerEncoderBlock(nn.Module):
    """
    MultiHeadSelfAttention + FFN + residuals + LayerNorm.
    Attention across N variate tokens: (B, N, d_model).
    """
```

### 2B. `encoder.py`

```python
class iTransformerEncoder(nn.Module):
    """
    Input:  (B, W, N)
    1. Transpose → (B, N, W)
    2. VariateEmbedding → (B, N, d_model)
    3. L × TransformerEncoderBlock → (B, N, d_model)
    4. Mean pool N → (B, d_model)
    5. Linear(d_model → latent_dim) → z: (B, latent_dim)

    Hyperparams for small data:
        d_model ∈ {32, 64}  (NOT 512 — overfits ~330 samples)
        n_layers ∈ {1, 2}   (NOT 3+)
        n_heads = 4
        dropout = 0.1–0.3
        latent_dim ∈ {6..9}
    """
```

### 2C. `decoder.py`

```python
class iTransformerDecoder(nn.Module):
    """
    Input:  z: (B, latent_dim)
    1. Linear(latent_dim → d_model) → (B, d_model)
    2. Expand/repeat → (B, N, d_model)
    3. L × FFN per variate (or Transformer blocks)
    4. Linear(d_model → W) → (B, N, W)
    5. Transpose → (B, W, N)
    """
```

### 2D. `autoencoder.py`

```python
class iTransformerAE(nn.Module):
    """forward(x) → (x_hat, z) | encode(x) → z"""
```

**Verification checklist:**
- [ ] z.shape == (B, latent_dim)
- [ ] x_hat.shape == x.shape == (B, W, N)
- [ ] No NaN in forward
- [ ] Gradients flow to all params
- [ ] Param count < 500K for d_model=32
- [ ] n_heads divides d_model evenly

---

## Phase 3: Training (`src/training/`)

```python
class Trainer:
    """
    Optimizer: AdamW (weight_decay mandatory)
    Scheduler: CosineAnnealingLR
    Early stopping: patience=10 on val MSE
    Gradient clipping: max_norm=1.0
    Checkpoint: best model by val loss
    """

def reconstruction_loss(x, x_hat) -> Tensor:
    """MSE over (B, W, N). Ignore NaN positions."""

def naive_baseline_loss(x, train_mean) -> Tensor:
    """MSE of predicting train mean. Reference for quality test."""
```

**Verification checklist:**
- [ ] Loss decreases over first 5 epochs
- [ ] Val loss tracked separately (no leakage)
- [ ] Early stopping triggers at patience
- [ ] No NaN loss after 10 steps

---

## Phase 4: EDA Notebook (`notebooks/00_eda.ipynb`)

```
§0: Setup & Imports
§1: FRED-MD Load — parse CSV, tcodes, transforms, show shape/head
§2: Missing Data — null % heatmap, patterns over time
§3: Distributions (train only) — histograms top-12 by kurtosis, JB test table
§4: Stationarity — ADF + KPSS per series, flag ambiguous (CPIAUCSL)
§5: Split Visualization — timeline + KS test val vs train, test vs train
§6: Correlation — heatmaps train vs val (hierarchical order)
§7: Window Statistics — distribution of window means/stds for W=6,12,24
§8: Baseline PCA — scree plot, PC1 vs PC2 scatter, confirm PC1=risk-on/off
§9: Regime Proxy — ICSS breakpoints vs GFC/dot-com/COVID alignment
§10: Summary — table confirming/refining CLAUDE.md EDA findings
```

**Rules:** matplotlib + seaborn ONLY. All figures → `notebooks/figures/`.

---

## Phase 5: Evaluation (`src/evaluation/`)

### `embedding_quality.py`

```python
def reconstruction_mse(model, dataloader, device) -> float
def naive_baseline_mse(dataloader, train_mean) -> float

def check_embedding_collapse(embeddings, threshold=1e-5) -> dict:
    """Returns: {per_dim_variance, collapsed_dims, is_collapsed}"""

def compute_effective_rank(embeddings) -> float:
    """exp(H(singular value distribution)). Low = collapsed."""

def compute_isotropy(embeddings) -> float:
    """Mean pairwise cosine similarity. Near 0 = good."""
```

### `clustering.py`

```python
def fit_pca(embeddings_train, n_components=6) -> PCA  # train only
def apply_pca(embeddings, pca) -> np.ndarray           # no refit
def fit_kmeans(embeddings_pca, k, n_init=20, random_state=42) -> KMeans
def compute_clustering_metrics(embeddings_pca, labels) -> dict:
    """Returns: {silhouette, davies_bouldin, calinski_harabasz}"""
def select_k(embeddings_pca_val, k_range=[3,4,5]) -> dict  # argmax silhouette
def clustering_stability(embeddings_pca, k, n_runs=10) -> float  # mean ARI
```

### `statistical_tests.py`

```python
def kruskal_wallis_per_dim(embeddings, labels) -> dict:
    """KW test per dim across K clusters. BH-corrected p-values."""

def pairwise_wilcoxon(embeddings, labels) -> dict:
    """Pairwise Wilcoxon between regime pairs. BH corrected."""

def temporal_consistency_score(labels, dates) -> dict:
    """Count regime transitions. Too many = unstable model."""
```

---

## Phase 6: Tests (`tests/`)

### conftest.py fixtures

```python
@pytest.fixture
def mock_fred_md_df():     # 100×20 DataFrame with tcode row
@pytest.fixture
def mock_model():          # iTransformerAE(n_series=20, W=6, d_model=16, latent_dim=6)
@pytest.fixture
def mock_embeddings():     # np.random.randn(50, 6)
```

### Unit Tests — Data (`tests/unit/test_data.py`)

```python
def test_tcode_extraction()            # Row 2 parsed correctly
def test_outlier_removal_spike()       # Spike > 10*IQR → NaN
def test_outlier_removal_preserves_median()
def test_tcode_1_noop()
def test_tcode_2_diff()                # Matches pd.Series.diff()
def test_tcode_5_log_diff()            # Matches np.diff(np.log(x))
def test_dataset_length_non_overlapping()
def test_dataset_length_overlapping()
def test_dataset_item_shape()          # Shape == (W, N)
def test_no_scaler_leakage()           # Val mean ≠ 0
```

### Unit Tests — Model (`tests/unit/test_model.py`)

```python
def test_encoder_output_shape()        # (B, latent_dim)
def test_decoder_output_shape()        # (B, W, N)
def test_autoencoder_roundtrip_shape() # x_hat.shape == x.shape
def test_no_nan_forward()
def test_gradient_flows()              # All params have grad after backward
def test_param_count()                 # < 500K for d_model=32
```

### Integration Tests (`tests/integration/test_training.py`)

```python
def test_loss_decreases()              # Epoch 3 loss < epoch 1 loss
def test_no_nan_loss()                 # 10 steps, never NaN
def test_early_stopping()             # Stops at patience with flat val loss
def test_checkpoint_restore()          # Save/load → embeddings identical
```

### Quality Tests (`tests/quality/`)

```python
# test_embedding_quality.py
def test_reconstruction_beats_baseline()  # CRITICAL: model_mse < naive_mse
def test_no_embedding_collapse()          # All dims var > 1e-5
def test_effective_rank_above_threshold() # Effective rank > 2
def test_pca_explained_variance()         # 6 PCs explain > 80% of embedding var

# test_clustering_quality.py
def test_silhouette_positive()            # CRITICAL: silhouette > 0
def test_kw_significant_on_half_dims()    # ceil(d/2) dims significant (BH p<0.05)
def test_clustering_stability()           # ARI across seeds > 0.7
def test_k_selection_returns_valid_k()    # K ∈ {3, 4, 5}
```

---

## Phase 7: Embedding Analysis Notebook (`notebooks/01_embedding_analysis.ipynb`)

```
§0: Load trained model + data
§1: Extract embeddings (train/val/test) — forward pass, no grad
§2: Embedding geometry — per-dim variance heatmap, effective rank, isotropy
§3: PCA on embeddings — scree plot, 2D scatter colored by date
§4: K selection — silhouette vs K plot, select best K
§5: Regime visualization — time series of labels + NBER recession overlay
§6: Statistical validation — KW table + pairwise Wilcoxon
§7: Baseline comparison — iTransformer silhouette vs PCA-KMeans on raw features
§8: Summary metrics table — all metrics in one place
```

---

## Phase 8: Configs & Sweep

### `configs/base.yaml`

```yaml
data:
  url: "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
  train_end: "2018-12-01"
  val_end: "2021-12-01"
  stride: 1

model:
  d_model: 64
  n_heads: 4
  n_layers: 2
  latent_dim: 6
  dropout: 0.1

training:
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-4
  max_epochs: 200
  patience: 10
  grad_clip: 1.0

clustering:
  k_range: [3, 4, 5]
  pca_components: 6
  kmeans_n_init: 20
```

### `scripts/sweep.py` grid

```
W ∈ {6, 12, 24} × d ∈ {6, 7, 8, 9} × K ∈ {3, 4, 5} = 36 combos
Select: best W by val reconstruction loss
        best d by val silhouette
        best K by val silhouette
Save: results/sweep_results.csv
```

---

## Critical Risk Flags (Scientific Critique)

**Risk 1: Overlapping windows → temporal autocorrelation**
- Problem: stride=1 → highly correlated samples → misleading val metrics
- Fix: report metrics on BOTH overlapping (training) and non-overlapping (evaluation)
- Test: non-overlapping val silhouette must still be > 0

**Risk 2: Small val set (36 non-overlapping months)**
- Problem: silhouette on 36 points is noisy
- Fix: bootstrap CIs (n=1000 resamples) on all clustering metrics
- Report: silhouette ± 95% CI in paper

**Risk 3: COVID val set is most OOD**
- Reconstruction loss will spike on val — expected and meaningful
- Document: how much loss spikes relative to train (domain shift quantification)

**Risk 4: PCA fitted only on train embeddings**
- Check: explained variance of val/test under train PCA > 80%
- If < 80%: val embeddings lie outside train support → document as OOD signal

**Risk 5: K-Means non-determinism**
- Always: random_state=42 AND run stability (ARI across 10 seeds)
- Report ARI in paper alongside clustering metrics

---

## Execution Order

```
Phase 0  → Read docs (iTransformer source, FRED-MD spec, sklearn APIs)
Phase 1  → Data pipeline + unit tests
Phase 2  → Model + unit tests
Phase 3  → Training + integration tests
Phase 4  → EDA notebook (uses Phase 1)
Phase 5  → Evaluation modules
Phase 6  → All quality tests (require Phase 2 + 5)
Phase 7  → Embedding analysis notebook (requires trained model)
Phase 8  → Configs + sweep script
```

---

## requirements.txt

```
torch>=2.2
scikit-learn>=1.4
numpy>=1.26
pandas>=2.2
polars>=0.20
matplotlib>=3.8
seaborn>=0.13
scipy>=1.13
pytest>=8.0
pytest-cov
jupyter
pyyaml>=6.0
tqdm
```

---

*Research sources used in plan:*
- [iTransformer paper](https://arxiv.org/abs/2310.06625)
- [Official impl — thuml/iTransformer](https://github.com/thuml/iTransformer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [Deep Clustering + Autoencoder (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206254/)
- [ML Testing best practices — Made With ML](https://madewithml.com/courses/mlops/testing/)
- [sklearn clustering metrics](https://scikit-learn.org/stable/modules/clustering.html)
- [Embedding quality — Zilliz](https://zilliz.com/ai-faq/how-do-i-measure-the-quality-of-an-embedding-model)
