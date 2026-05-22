# Notation and Acronym Table

**Version:** 1.0  
**Date:** 2026-05-30

---

## 1. Mathematical Notation

| Symbol | Definition |
|---|---|
| $T$ | Total number of months in the panel |
| $F$ | Number of FRED-MD series (122 after balanced filter) |
| $W$ | Sliding window length in months (W=6 in operational config) |
| $d$ | Latent embedding dimension (d=7 in operational config) |
| $K$ | Number of clusters (K=4 in operational config) |
| $\mathbf{X}_t \in \mathbb{R}^F$ | Feature vector at month t (after tcode transform + z-score) |
| $\mathbf{W}_t \in \mathbb{R}^{W \times F}$ | Window matrix: $[\mathbf{X}_{t-W+1}, \ldots, \mathbf{X}_t]$ |
| $\mathbf{z}_t \in \mathbb{R}^d$ | Latent embedding of window $\mathbf{W}_t$ |
| $c_t \in \{0, 1, \ldots, K-1, -1\}$ | Cluster label at month t (-1 = noise, HDBSCAN only) |
| $\hat{y}_t \in \{0,1\}$ | Binary recession prediction (1 = recession, derived from cluster assignment) |
| $y_t \in \{0,1\}$ | Ground-truth NBER USREC indicator at month t |
| $\mathbf{A} \in \{0,1\}^K$ | Frozen cluster-to-regime assignment vector (from VAL) |
| $\mu_\text{train}$, $\sigma_\text{train}$ | TRAIN-period mean and std for standardization |
| $\text{DBCV}$ | Density-Based Clustering Validation index (Moulavi et al. 2014) |
| $J(A, B)$ | Jaccard similarity: $\|A \cap B\| / \|A \cup B\|$ |

---

## 2. Acronyms

| Acronym | Full form |
|---|---|
| AE | Autoencoder |
| ARI | Adjusted Rand Index |
| BOCPD | Bayesian Online Changepoint Detection (Adams & MacKay 2007) |
| CFNAI | Chicago Fed National Activity Index |
| CFNAI-MA3 | 3-month moving average of CFNAI |
| CLI | Composite Leading Indicator (OECD) |
| DBCV | Density-Based Clustering Validation (Moulavi et al. 2014) |
| DLC | Deep Learning Container (AWS SageMaker) |
| DR | Dimensionality Reduction |
| ETL | Extract, Transform, Load |
| FRED | Federal Reserve Economic Data |
| FRED-MD | FRED Monthly Database (McCracken & Ng 2016) |
| GFC | Global Financial Crisis (2007–2009) |
| HDP-HMM | Hierarchical Dirichlet Process Hidden Markov Model |
| HDBSCAN | Hierarchically Density-Based Spatial Clustering of Applications with Noise |
| HMM | Hidden Markov Model |
| IAC | Inter-Annotator Consistency |
| iTransformer | Inverted Transformer encoder (Liu et al. 2023, adapted for SSL) |
| KMeans | K-Means clustering (Lloyd's algorithm) |
| MLP | Multi-Layer Perceptron |
| MLP-AE | MLP-based Autoencoder (3-layer encoder + 3-layer decoder) |
| NBER | National Bureau of Economic Research |
| OECD | Organisation for Economic Co-operation and Development |
| PCA | Principal Component Analysis |
| PELT | Pruned Exact Linear Time (changepoint algorithm, Killick et al. 2012) |
| ROC | Receiver Operating Characteristic |
| SAM | AWS Serverless Application Model |
| SSL | Self-Supervised Learning |
| SVD | Singular Value Decomposition |
| TCC | Trabalho de Conclusão de Curso (Brazilian capstone thesis) |
| TFC | Time-Frequency Consistency (Chang et al. 2023) |
| t-SNE | t-distributed Stochastic Neighbor Embedding (van der Maaten & Hinton 2008) |
| UMAP | Uniform Manifold Approximation and Projection (McInnes et al. 2018) |
| USREC | US Recession Indicator (NBER, binary monthly) |
| VAE | Variational Autoencoder |

---

## 3. Methods Taxonomy (Encoder Types)

| Encoder | Type | Training |
|---|---|---|
| iTransformer | trainable_ssl | Contrastive/reconstruction SSL on TRAIN |
| MLP-AE | trainable_autoencoder | Reconstruction loss on TRAIN |
| Linear-AE | trainable_autoencoder | Reconstruction loss on TRAIN |
| SVD | fixed_transform | No training (singular value decomposition) |
| Raw-PCA | fixed_transform | No training (PCA on last time step) |
| Windowed-PCA | fixed_transform | No training (PCA on full window) |
| MOMENT | zero_shot_foundation | Pre-trained on large corpus; no fine-tuning |
| PatchTST | trainable | Pre-trained + fine-tuned (time series classification) |
| TS2Vec | trainable_ssl | Contrastive SSL on TRAIN |
| TimesNet | trainable | Temporal convolution network |
| TFC | trainable_ssl | Time-Frequency Consistency SSL |
| HamiltonHMM | probabilistic_unsupervised | Expectation-Maximization on TRAIN |
| BOCPD | probabilistic_online | Bayesian online algorithm (no training) |

---

## 4. Split Labels

| Label | Period | Months |
|---|---|---|
| TRAIN | 1965-01 to 1999-12 | 420 |
| VAL | 2000-01 to 2009-12 | 120 |
| TEST | 2010-01 to 2026-04 | 194 |

---

## 5. Evaluation Window Labels

| Label | Description |
|---|---|
| dot-com | 2001-03 to 2001-11 (NBER: peak 2001-03, trough 2001-11) |
| GFC | 2007-12 to 2009-06 (NBER: peak 2007-12, trough 2009-06) |
| COVID | 2020-02 to 2020-04 (NBER: peak 2020-02, trough 2020-04) |

Note: dot-com and GFC are in VAL; only COVID is in TEST (n=2 USREC months in TEST after windowing).
