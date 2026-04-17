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
