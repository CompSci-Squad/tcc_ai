# Plano Consolidado v3 — Alinhamento Pré-Projeto + AWS/SageMaker

> **Decisão metodológica (2026-04-28):** opção (A) — **HDBSCAN + UMAP** principal · **PCA + K-Means** baseline.
> **Decisão de infraestrutura (2026-04-28):** **AWS é alvo de produção**. ETL = Lambda → S3 (Parquet, particionado por `year=/month=`). Treino = SageMaker Training Jobs. Tracking = MLflow no SageMaker (managed) ou em EC2 t3.small. IaC = Terraform em [tcc_iac](../../tcc_iac/).
>
> Substitui [PLAN_gap_remediation.md](PLAN_gap_remediation.md) e [feature-itransformer-thesis-1.md](../plan/feature-itransformer-thesis-1.md).

---

## 1. Diagnóstico em uma página

| Camada | Pré-projeto exige | Implementado hoje | Ação |
|---|---|---|---|
| ETL | Lambda → S3 Parquet | ✅ [tcc_etl](../../tcc_etl/) (SAM, arm64, schedule mensal) | manter, expor URI canônica |
| Encoder | iTransformer AE com MSE | ✅ `model/autoencoder.py` | manter |
| Pré-proc | tcodes + ADF/KPSS + IQR | ✅ tcodes / ⚠ KPSS faltando | adicionar KPSS |
| Redução dim | **UMAP** (principal) + PCA (baseline) | apenas PCA | **+ UMAP** |
| Clustering | **HDBSCAN** (principal) + K-Means (baseline) | apenas K-Means | **+ HDBSCAN** |
| Métrica densidade | **DBCV** | ausente | **+ DBCV** |
| Validação ext. | NBER USREC + Bai-Perron + janelas de crise | ausente | **+ módulo regime_validation** |
| Interpretabilidade | momentos condicionais + matriz transição + duração | só count | **+ módulo regime_validation** |
| Explicabilidade | regime + grau de pertencimento + top-features | ausente | **+ módulo 4 (explain.py)** |
| Treino | reprodutível, configurável | local (uv + MLflow file) | **+ SageMaker Training Job** |
| Tracking | MLflow | local file backend | **+ MLflow remoto (S3 artifacts + RDS)** ou SageMaker Experiments |
| Infra | IaC | ✅ [tcc_iac](../../tcc_iac/) (ETL) | **+ ECR training, SageMaker role, S3 modelos/MLflow** |
| Snapshots locais | só dev | precisa baixar do S3 | **+ `data/from_s3.py`** helper |

---

## 2. Reestruturação do repo (proposta)

```
tcc/
├── docs/                          # textos acadêmicos (pre_projeto, tcc.md)
├── tcc_etl/                       # Lambda — JÁ OK
├── tcc_iac/                       # Terraform — adicionar SageMaker stack
│   └── infra/
│       ├── (existentes: lambda, ecr, s3, scheduler...)
│       ├── sagemaker.tf           # NOVO: execution role, training image ECR
│       ├── mlflow.tf              # NOVO: bucket artifacts + (opcional) RDS Postgres
│       └── outputs.tf             # exportar bucket panel + role ARNs
├── tcc_ai/
│   ├── src/tcc_itransformer/      # core lib (importável de qualquer entrypoint)
│   │   ├── data/
│   │   │   ├── fred_md.py         # já existe (CSV)
│   │   │   ├── s3_loader.py       # NOVO: ler parquet do bucket panel
│   │   │   └── ...
│   │   ├── model/                 # já ok
│   │   ├── evaluation/
│   │   │   ├── baseline_clustering.py   # renomeado de clustering.py
│   │   │   ├── density_clustering.py    # NOVO: HDBSCAN+UMAP+DBCV
│   │   │   └── regime_validation.py     # NOVO: NBER+Bai-Perron+momentos
│   │   ├── explain.py             # NOVO: módulo 4
│   │   └── tracking/
│   │       └── mlflow_utils.py    # ler MLFLOW_TRACKING_URI do env
│   ├── sagemaker/                 # NOVO — entrypoints + launchers
│   │   ├── train_entrypoint.py    # SM_CHANNEL_TRAINING -> /opt/ml/model
│   │   ├── launch_training.py     # boto3/sagemaker SDK launcher
│   │   ├── launch_sweep.py        # 36 jobs em paralelo
│   │   ├── Dockerfile.training    # base: pytorch-training:2.4-gpu
│   │   └── requirements.txt       # deps adicionais (hdbscan, umap, ruptures)
│   ├── scripts/                   # CLI local (mantém para dev/debug)
│   ├── configs/                   # YAML (mesmos arquivos servem para SM)
│   ├── notebooks/                 # EDA + análise
│   ├── tests/                     # unit + integration + quality
│   ├── pyproject.toml
│   └── Makefile
└── README.md                      # NOVO: visão monorepo + links para guias
```

**Princípios:**
- `src/tcc_itransformer/` é a única biblioteca; tanto `scripts/` (local) quanto `sagemaker/train_entrypoint.py` (remoto) a importam.
- Configs YAML são idênticas; entrypoint detecta ambiente via `SM_CHANNEL_TRAINING` (presente = SageMaker).
- Dados: dev usa `data/snapshots/*.parquet` (baixado do S3 via `make pull-data`); prod (SM) lê `s3://tcc-regime-etl-panel-data/fred_md/transformed/year=YYYY/month=MM/`.
- Modelos treinados e artefatos: SageMaker grava em `s3://...-sagemaker/models/<job-name>/output/model.tar.gz`; MLflow rastreia métricas + tags + parâmetros.

---

## 3. Waves consolidadas

### Wave 0 — Decisão & Estrutura ✅
- [x] W0.1 Confirmar opção (A): UMAP+HDBSCAN principal
- [x] W0.2 Criar skill `regime-detection-validation`
- [x] W0.3 Plano v3 (este documento)
- [ ] W0.4 Atualizar [skills-index.md](skills-index.md) (✅ feito)

### Wave 1 — Foundation ✅
- [x] W1.1 `tcc_ai/src/tcc_itransformer/data/s3_loader.py` — `load_panel_from_s3(bucket, prefix)` (auto-resolve SM_CHANNEL_TRAINING vs `s3://`)
- [x] W1.2 Adicionar deps em [pyproject.toml](../pyproject.toml): `hdbscan>=0.8.33`, `umap-learn>=0.5.5`, `ruptures>=1.1.9`, `boto3>=1.34`, `s3fs>=2024.3`, `pyarrow>=15`, `joblib>=1.3` — `uv sync` ok
- [x] W1.3 `scripts/pull_nber.py` — baixa NBER USREC para `data/snapshots/` + sha256
- [x] W1.4 `external_labels.load_usrec` (lookup mensal)
- [x] W1.5 `python scripts/generate_sweep_configs.py` → 36 YAMLs em `configs/sweep/`
- [x] W1.6 KPSS junto ao ADF em `data/stationarity.py` (`check_series_stationarity`, `validate_panel_stationarity`)

### Wave 2 — Novos módulos do pipeline ✅
- [x] W2.1 `evaluation/dim_reduction.py` — `UMAPConfig`, `fit_umap`, `apply_umap`
- [x] W2.2 `evaluation/density_clustering.py` — `fit_hdbscan` + `optimize_hdbscan_dbcv` (DBCV via `relative_validity_`)
- [x] W2.3 `evaluation/regime_validation.py` — NBER overlap, Bai-Perron (`ruptures`), crisis windows, conditional moments, transition matrix, durations
- [x] W2.4 `evaluation/explain.py` — `explain_assignment` + `explanations_to_frame`
- [x] W2.5 `clustering.select_k_combined` (Silhouette + −BIC/GMM normalizados)

### Wave 3 — Pipeline integrity (estatística) ✅
- [x] W3.1 `run_single.py`: pipeline principal UMAP+HDBSCAN no TEST + baseline PCA+K-Means no TEST
- [x] W3.2 ~~BCa bootstrap~~ — **já implementado** em `evaluation/statistical_tests.py:_bootstrap_ci` (z0 + jackknife acceleration)
- [x] W3.3 CIs em effect sizes via permutation/bootstrap CIs
- [x] W3.4 `run_baselines.py`: janelas não-sobrepostas + PCA antes do K-Means
- [x] W3.5 Permutation test iTransformer vs Raw PCA
- [x] W3.6 MLflow tag `analysis_type=exploratory` para W=24
- [x] W3.7 `run_full_pipeline()` wrapper + persist `principal_artifacts` (moments/durations/transition_matrix/explanations/hdbscan_grid → parquet/json) + MLflow `log_artifact`

### Wave 4 — AWS / SageMaker ✅
- [x] W4.1 [sagemaker/train_entrypoint.py](../sagemaker/train_entrypoint.py) — lê `SM_CHANNEL_TRAINING`/`SM_CHANNEL_USREC`, `SM_MODEL_DIR`, `SM_OUTPUT_DATA_DIR`; chama `run_full_pipeline`
- [x] W4.2 [Dockerfile](../Dockerfile) baseado em PyTorch 2.4 GPU DLC + extras
- [x] W4.3 [sagemaker/launch_training.py](../sagemaker/launch_training.py) — `sagemaker.pytorch.PyTorch` Estimator
- [x] W4.4 `make sm-sweep` itera `configs/sweep/*.yaml`
- [x] W4.5 [tcc_iac/infra/sagemaker.tf](../../tcc_iac/infra/sagemaker.tf) — ECR repo, SageMaker bucket, IAM exec role
- [x] W4.6 [tcc_iac/infra/mlflow.tf](../../tcc_iac/infra/mlflow.tf) — `aws_sagemaker_mlflow_tracking_server` (toggle via `enable_mlflow`)
- [x] W4.7 `Makefile`: `pull-nber`, `sm-build`, `sm-push`, `sm-train`, `sm-sweep`
- [ ] W4.8 `.github/workflows/sagemaker.yml` (opcional, deferido)

### Wave 5 — Quality tests ✅
- [x] W5.1 `tests/quality/test_clustering_quality.py` estendido para HDBSCAN (3 novos gates: noise ≤ 0.4, n_clusters ≥ 2, DBCV > 0)
- [x] W5.2 `tests/unit/test_regime_validation.py` (7 testes desde Wave 2)
- [x] W5.3 `tests/integration/test_sagemaker_entrypoint.py` — entrypoint roda end-to-end com fakes de `SM_*`
- [x] W5.4 `tests/unit/test_misc_gaps.py` — param.grad, n_heads validation, sha256, mlflow_run setup

### Wave 6 — Documentação ✅
- [x] W6.1 README do monorepo (raiz `tcc/`) com diagrama AWS + status atualizado (138 testes)
- [x] W6.2 [api_reference.md](api_reference.md) inclui seções UMAP/HDBSCAN, regime_validation, explain, stationarity, s3_loader e SageMaker entrypoint
- [x] W6.3 [deploy_aws.md](deploy_aws.md) — guia passo-a-passo (pré-existente, validado contra implementação)
- [x] W6.4 [pre_analysis_plan.md](pre_analysis_plan.md) addendum 2026-04-28 registrando Opção A + critérios HDBSCAN + BCa + AWS
- [ ] W6.5 Sincronizar [docs/tcc.md](../../docs/tcc.md) (deferido — redigido após os experimentos da Wave 8)

### Wave 7 — Notebooks & Viz
- [ ] W7.1 `notebooks/00_eda.ipynb` (lendo do bucket S3 via `s3fs`)
- [ ] W7.2 `notebooks/01_embedding_analysis.ipynb` adaptado para HDBSCAN+UMAP, baixa `model.tar.gz` do SM
- [ ] W7.3 11 funções de viz (NBER overlay, DBCV vs hyperparams, condensed tree)

### Wave 8 — Experimentos & Resultados
- [ ] W8.1 `make sm-sweep` (36 jobs SageMaker) + `make sm-baselines`
- [ ] W8.2 `scripts/export_results.py` consulta MLflow remoto → tabelas LaTeX + figuras
- [ ] W8.3 README com resultados reais

---

## 4. Skills/Agents por Wave

| Wave | Skills primárias | Agentes sugeridos |
|---|---|---|
| 0 | agent-orchestrator, doublecheck | requirements-analyst |
| 1 | fred-economic-data, polars | python-expert |
| 2 | scikit-learn, **regime-detection-validation**, statsmodels | system-architect, python-expert |
| 3 | statistical-analysis, statsmodels | quality-engineer |
| **4** | **mlops-engineer, terraform-aws-modules, aws-serverless, docker-expert** | **devops-architect, system-architect** |
| 5 | scientific-writing, documentation-writer | quality-engineer, technical-writer |
| 6 | scientific-writing, documentation-writer | technical-writer |
| 7 | matplotlib, seaborn, scientific-visualization | — |
| 8 | mlops-engineer, ml-pipeline-workflow | performance-engineer |

---

## 5. Critérios de "defense-ready"

- [ ] HDBSCAN roda end-to-end no TEST set com DBCV reportado e fração de ruído < 0.4
- [ ] NBER USREC overlap (F1) reportado
- [ ] ≥3 das 5 janelas canônicas cobertas
- [ ] Baseline PCA+K-Means com Δ-silhouette + CI BCa
- [ ] Matriz de transição + duração média por regime
- [ ] Módulo 4 produz payload `{regime, membership, top_features}`
- [ ] **Reprodução one-click via SageMaker:** `make sm-sweep` lança 36 jobs e popula MLflow remoto
- [ ] **Custo documentado** (estimativa de US$ por sweep completo)
- [ ] [tcc.md](../../docs/tcc.md) atualizado

---

## 6. Próximo comando recomendado

> **Wave 1 + W4.5 em paralelo:** (a) adicionar deps + criar `s3_loader.py`, (b) escrever `sagemaker.tf` no IaC. Ler o guia [deploy_aws.md](deploy_aws.md) antes.
