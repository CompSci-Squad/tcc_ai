# Thesis Reframe — Honest-Ablation Outline

> Written 2026-04-30 night, after Phase A + B (panel-remediation-plan.md) returned a TRIGGERED falsification verdict.
> Purpose: re-anchor the TCC narrative around a defensible negative result.

## New title (proposal)

> **Detecção não supervisionada de regimes macroeconômicos no FRED-MD: um estudo de benchmark com o iTransformer como ablação negativa.**

Alternative (more agnostic): "Avaliação empírica de codificadores de séries temporais para detecção não supervisionada de regimes macroeconômicos."

## Central thesis

A pipeline simples — janelas deslizantes + redução de dimensionalidade linear (PCA/SVD) ou MLP-autoencoder de uma camada + HDBSCAN — produz embeddings com **DBCV de 0.67–0.79 no TEST**, enquanto o iTransformer treinado sob protocolo de pré-registro idêntico atinge no máximo **0.17**. O ganho prometido pela arquitetura transformer-de-variáveis (Liu et al. 2024) **não se materializa** no problema de regime detection no FRED-MD. Documentamos também um **confounder pré/pós-2008** (Cramer's V ≈ 0.43) que afeta qualquer pipeline avaliado neste painel.

## Chapter map (Capítulos 3–5)

### Cap. 3 — Metodologia (reescrita)

1. **3.1** Painel FRED-MD via ETL própria (`tcc_etl`). Contrato v2: parquet balanceado + máscara, 122 séries, 1965-01..2026-04, transformações de estacionariedade pré-aplicadas.
2. **3.2** Pré-registro de avaliação ([`docs/pre_analysis_plan.md`](../docs/pre_analysis_plan.md)):
   - Split temporal **B1**: TRAIN 1965-01..1999-12 / VAL 2000-01..2009-12 (dotcom + GFC = 26 meses NBER) / TEST 2010-01..2026-04 (COVID = 2 meses NBER).
   - Painel locked de 7 métricas no TEST: `dbcv`, `n_clusters_test`, `noise_fraction_test`, `nber_f1` (Hungarian-on-VAL congelado), `nber_f1_legacy_maxF1` (testemunha de viés), `bai_perron_f1` (tolerância ±2 meses), `crisis_window_coverage` (dotcom/GFC/COVID).
   - Critério de desempate de modelos pré-registrado: `(K asc, params asc, best_epoch asc)`.
   - Inferência: silhouette com permutation p-value (B=1000) e bootstrap CI 95% (n=1000).
3. **3.3** Cinco encoders avaliados: `random`, `truncated_svd`, `linear_ae`, `mlp_ae` (1 hidden, 128), `iTransformer-AE` (W=6, d_lat=7, K=4) — todos seguidos pelo **mesmo** downstream UMAP+HDBSCAN ou KMeans.
4. **3.4** Diagnósticos de validade interna: confound check (χ² + ARI sob remoção 2020-Q2).

### Cap. 4 — Resultados

1. **4.1 Resultado principal — falsificação do iTransformer.** Tabela B1-split:

   | Encoder | TEST DBCV | TEST silhouette | NBER F1 | Bai-Perron F1 |
   |---|---|---|---|---|
   | random | NaN (KMeans) | 0.006 | 0.000 | 0.000 |
   | linear-AE | 0.253 | 0.176 | 0.000 | 0.000 |
   | raw-PCA | NaN (KMeans) | 0.279 | 0.000 | **0.909** |
   | windowed-PCA | NaN (KMeans) | 0.281 | 0.000 | **0.909** |
   | truncated-SVD + HDBSCAN | **0.666** | – | – | – |
   | MLP-AE + HDBSCAN | **0.789** | – | – | – |
   | iTransformer (W6_d7_K4) | 0.166 | 0.348 | 0.571 | 0.791 |

   Interpretação: iTransformer só vence em uma métrica (NBER F1 com KMeans K=4). Em DBCV é dominado por SVD e MLP-AE em > 4×. Em Bai-Perron F1, raw-PCA e windowed-PCA (linear, sem treino) empatam ou excedem.

2. **4.2 Evidência de overfit ao TRAIN pré-2000.** Diagnóstico de treinamento do iTransformer no split B1: `train_loss = 0.93`, `val_loss = 57.0` no early-stop (epoch 72). O modelo não generaliza da janela de calibração 1965-99 para 2000-09 — a escala de choque do GFC é inédita no TRAIN, e o StandardScaler ajustado em TRAIN amplifica o desvio.

3. **4.3 Confounder pré-2008.** χ² entre cluster e "pré-2008 vs pós-2008" tem **Cramer's V ≈ 0.43** (`p = 0.0012`). χ² entre cluster e NBER **não significativo** (`p = 0.83`). ARI entre clustering completo e clustering refit sem 2020-Q2 = **0.29** (instabilidade alta a outliers COVID). Conclusão: clusters refletem majoritariamente "antes/depois da Grande Crise Financeira" + outliers de pandemia, não regimes econômicos persistentes.

4. **4.4 Teto de desempenho do downstream.** Sweep HDBSCAN×{UMAP n_neighbors, t-SNE perplexity}: melhor DBCV alcançável ≈ **0.33** sobre embeddings iTransformer — confirma que o problema é o encoder, não os hiperparâmetros do clustering.

5. **4.5 Significância estatística.** Para a célula `pca_hdbscan` no iTransformer: silhouette=0.198, permutation `p=0.005` (B=1000), bootstrap CI95 = [0.62, 0.71]. Resultados estatisticamente significativos mesmo quando substantivamente fracos.

### Cap. 5 — Discussão

1. **5.1 Por que o iTransformer falha aqui:**
   - **Tamanho amostral.** ~700 janelas mensais de treino ≪ regime usual de transformers (10⁵–10⁶ tokens).
   - **Estrutura linear dominante.** PCA/SVD captam a maior parte da variância de FRED-MD (literatura de fatores de Stock & Watson 2002). O ganho marginal de uma arquitetura não-linear cara é negativo.
   - **Distributional shift estrutural.** O StandardScaler em TRAIN+modelo treinado em 1965-99 não reconhece a escala 2008+, e ainda menos COVID. Modelos lineares "fazem extrapolação suave" e degradam graciosamente; o iTransformer explode.

2. **5.2 Implicações.** Não recomendamos iTransformer como encoder default para regime detection em painéis macroeconômicos públicos. Recomendamos pipeline **windowed-PCA + HDBSCAN** como linha-base honesta — mais barata, mais reprodutível, mais estável.

3. **5.3 Limitações:**
   - Confounder pré-2008 afeta toda interpretação substantiva — discutir como ameaça à validade externa.
   - NBER F1 finalmente não-zero no split B1 (max 0.571), mas COVID-only no TEST limita poder estatístico para a métrica primária.
   - Não testamos arquiteturas auto-supervisionadas mais novas (TS2Vec, TF-C) — registrar como trabalho futuro.

4. **5.4 Trabalho futuro (Phase C + D do plano de remediação):**
   - Painel multi-label (Sahm, Chauvet-Piger MSI, CFNAI MA3, OECD CLI Bry-Boschan).
   - HDP-HMM com winsorização + ≥500 iterações Gibbs (a versão atual colapsou em 1 estado — não é baseline justa).
   - Concordância de quebras Bai-Perron em séries-headline (INDPRO, PAYEMS, UNRATE, T10Y3M).
   - Estabilidade de clusters via bootstrap (Ben-Hur 2002).

## Anchors / arquivos de evidência

- Plano de remediação (verdict no topo): [tcc_ai/plan/panel-remediation-plan.md](panel-remediation-plan.md).
- B3 SageMaker job ID: `itransformer-1777581449-0d38` — output em `s3://tcc-regime-etl-sagemaker/jobs/.../output.tar.gz`; embeddings extraídos em `tcc_ai/results/sm_outputs/itransformer-1777581449-0d38/embeddings/`.
- Falsificação (B2): [tcc_ai/results/falsification.csv](../results/falsification.csv).
- Painel iTransformer B1: [tcc_ai/results/clustering_ablation/W6_d7_K4_b1/summary.csv](../results/clustering_ablation/W6_d7_K4_b1/summary.csv).
- Painel baselines B1: [tcc_ai/results/baselines/baselines_panel_baseline_W6_d7_K4.csv](../results/baselines/baselines_panel_baseline_W6_d7_K4.csv).
- Confound check: [tcc_ai/results/diagnostics/confound_check.md](../results/diagnostics/confound_check.md).
- Sweep param HDBSCAN/DR: [tcc_ai/results/clustering_ablation/param_sweep.csv](../results/clustering_ablation/param_sweep.csv).
- Pré-registro: [tcc_ai/docs/pre_analysis_plan.md](../docs/pre_analysis_plan.md).
- Log de sessão (auditoria): [.github/SESSION_LOG.md](../../.github/SESSION_LOG.md) — entrada "2026-04-30 (night)".
