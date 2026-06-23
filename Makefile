.PHONY: train sweep test lint ui baselines hdphmm-baseline export install clean download-data generate-sweep generate-sweep-stage1 generate-sweep-stage2 help reproduce pull-nber sm-build sm-push sm-train sm-train-local sm-sweep sm-sweep-parallel sm-poll multi-label hdphmm-proper bai-perron-headline cluster-stability phase-c freeze-config ablation-aggregate reproduce-thesis phase-e phase-e-fast phase-e-install sprint1 sprint2 sprint3 sprints sprint4-f2 sprint4-f10 sprint5-f3 sprint6-f4 sprint7-f5 sprints-deferred all-sprints

# AWS / SageMaker variables (override on CLI: make sm-build AWS_ACCOUNT=...)
AWS_ACCOUNT ?= $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
AWS_REGION  ?= us-east-1
ECR_REPO    ?= tcc-regime-etl-itransformer
IMAGE_TAG   ?= latest
IMAGE_URI   := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO):$(IMAGE_TAG)
SM_BUCKET       ?= tcc-regime-etl-sagemaker
SM_DATA_BUCKET  ?= tcc-regime-etl-panel-data
SM_DATA_PREFIX  ?= fred_md/transformed/year=2026/month=04
SM_ROLE         ?= arn:aws:iam::$(AWS_ACCOUNT):role/LabRole
# Vocareum Pvoclabs2 (verified 2026-04-30 via simulate-principal-policy) explicitly
# denies CreateTrainingJob for m4/m6i/m7i/m5.2xlarge+. Allowed CPU: m5.large,
# m5.xlarge, t3.*, c5.large, c5.xlarge. Use m5.xlarge (4 vCPU / 16GB) for AE.
SM_INSTANCE     ?= ml.m5.xlarge
SM_CONFIG       ?= configs/sagemaker_ae_only.yaml
MLFLOW_URI      ?=

# Install all dependencies
install:
	uv sync

# Download FRED-MD data snapshot with SHA-256 verification
download-data:
	uv run tcc data download

# Pull NBER USREC recession indicator snapshot
pull-nber:
	uv run tcc data pull-nber

# Generate 36 sweep configuration YAML files (stage-2: W x d x K).
generate-sweep:
	uv run tcc configs gen-stage2

# Stage-1 HPO: 12-cell LR x dropout grid at primary (W=12, d_lat=8).
generate-sweep-stage1:
	uv run tcc configs gen-stage1

# Stage-2 HPO: regenerate W x d x K configs with frozen stage-1 winner.
# Usage: make generate-sweep-stage2 STAGE1_WINNER=configs/stage1_winner.yaml
generate-sweep-stage2:
	uv run tcc configs gen-stage2 --frozen-stage1 $(or $(STAGE1_WINNER),configs/stage1_winner.yaml)

# Train a single configuration
train:
	uv run tcc train single --config configs/default.yaml

# Run full hyperparameter sweep across all configs
sweep:
	uv run tcc train sweep

# Run baseline comparisons (PCA-only, random, etc.)
baselines:
	uv run tcc eval baselines

# Run unit tests only
# Run unit tests only
test:
	uv run pytest tests/unit/ -v

# Run all tests except quality gates
test-all:
	uv run pytest tests/ -v --ignore=tests/quality

# Run scientific quality gate tests (requires trained model)
test-quality:
	uv run pytest tests/quality/ -v -m quality

# Check code style and formatting
lint:
	uv run ruff check src/ tests/ sm_jobs/
	uv run ruff format --check src/ tests/ sm_jobs/

# Auto-fix code style and formatting
format:
	uv run ruff check --fix src/ tests/ sm_jobs/
	uv run ruff format src/ tests/ sm_jobs/

# Launch MLflow UI for experiment tracking
ui:
	uv run mlflow ui --backend-store-uri file:./results/mlruns --port 5000

# Export results to LaTeX tables and figures
export:
	uv run tcc analysis export

# Log environment details (Python version, packages, GPU)
env-log:
	uv run tcc data env-log

# Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ---------------- SageMaker / AWS ----------------
# Build training image locally
sm-build:
	docker build -f Dockerfile -t $(ECR_REPO):$(IMAGE_TAG) .

# Push training image to ECR (creates repo if missing, logs into both registries)
sm-push: sm-build
	aws ecr describe-repositories --repository-names $(ECR_REPO) --region $(AWS_REGION) >/dev/null 2>&1 \
		|| aws ecr create-repository --repository-name $(ECR_REPO) --region $(AWS_REGION)
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin 763104351884.dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(ECR_REPO):$(IMAGE_TAG) $(IMAGE_URI)
	docker push $(IMAGE_URI)

# Launch a single SageMaker training job (ETL-v2 contract by default)
sm-train:
	uv run python sm_jobs/launch_training.py \
		--config $(SM_CONFIG) \
		--bucket $(SM_BUCKET) \
		--data-bucket $(SM_DATA_BUCKET) \
		--role $(SM_ROLE) \
		--region $(AWS_REGION) \
		--instance-type $(SM_INSTANCE) \
		--mlflow-uri "$(MLFLOW_URI)" \
		--image-uri $(IMAGE_URI) \
		--data-prefix $(SM_DATA_PREFIX) \
		--usrec-prefix raw/USREC.csv

# Local end-to-end smoke test of the SM entrypoint (no AWS calls).
# Mounts data via SM_CHANNEL_TRAINING env var pointing at local parquet.
sm-train-local:
	SM_CHANNEL_TRAINING=$(PWD)/data/raw \
	SM_CHANNEL_USREC=$(PWD)/data/snapshots \
	SM_MODEL_DIR=$(PWD)/results/sm_local/model \
	SM_OUTPUT_DATA_DIR=$(PWD)/results/sm_local/output \
	uv run python sm_jobs/train_entrypoint.py --config $(SM_CONFIG)

# Sticky / SDHDP-HMM baseline (Q4). Local CPU JAX. Requires the `baselines` extra:
#   uv sync --extra baselines
# Phase C2: defaults now 500 iters (sticky) / 200 iters (sdhdp) with winsorisation.
#   Override with N_ITER=100 to replicate the old collapsed-state result.
hdphmm-baseline:
	uv run tcc eval hdphmm \
		--config $(or $(CONFIG),configs/default.yaml) \
		--variant $(or $(VARIANT),sticky) \
		--n-states-max $(or $(N_STATES_MAX),10) \
		--n-iter $(or $(N_ITER),0)

# ---- Phase C targets ----

# C1: Multi-label validation panel (Chauvet-Piger, Sahm, CFNAI-MA3, OECD CLI).
# Requires: uv sync --extra labels && export FRED_API_KEY=<key>
# Override CLUSTERING_PARQUET for other pipelines.
CLUSTERING_PARQUET ?= results/clustering_ablation/W6_d7_K4_b1/pca_kmeans.parquet
multi-label:
	uv run tcc eval multi-label \
		--clustering-parquet $(CLUSTERING_PARQUET) \
		--output results/diagnostics/multi_label_panel.csv

# C2: Re-run HDP-HMM baseline with proper n_iter + winsorisation.
hdphmm-proper:
	uv sync --extra baselines
	uv run tcc eval hdphmm \
		--config $(or $(CONFIG),configs/default.yaml) \
		--variant sticky \
		--n-states-max 10 \
		--n-iter 0
	uv run tcc eval hdphmm \
		--config $(or $(CONFIG),configs/default.yaml) \
		--variant sdhdp \
		--n-states-max 10 \
		--n-iter 0

# C3: Bai-Perron break agreement on 4 headline series.
# Requires: export FRED_API_KEY=<key>
bai-perron-headline:
	uv run tcc eval bai-perron-headline \
		--clustering-parquet $(CLUSTERING_PARQUET) \
		--output results/diagnostics/bai_perron_headline.csv

# C4: Cluster stability bootstrap (Ben-Hur 2002 Jaccard + ARI).
# Uses iTransformer Z_test embeddings as the correct feature space.
EMB_DIR ?= results/sm_outputs/itransformer-1777581449-0d38/embeddings
cluster-stability:
	uv run tcc eval cluster-stability \
		--ablation-dir results/clustering_ablation/W6_d7_K4_b1 \
		--emb-dir $(EMB_DIR) \
		--output results/diagnostics/cluster_stability.csv

# Run all Phase C steps (C1+C3 require FRED_API_KEY; C2 requires --extra baselines).
phase-c: multi-label hdphmm-proper bai-perron-headline cluster-stability
	@echo "Phase C complete. Results in results/diagnostics/"

.PHONY: multi-label hdphmm-proper bai-perron-headline cluster-stability phase-c

# Launch one SageMaker job per sweep config (sequential; SM handles parallel slots).
# Override SM_SWEEP_DIR=configs/sweep_stage1 for the LR×dropout stage.
SM_SWEEP_DIR    ?= configs/sweep
sm-sweep:
	@for cfg in $(SM_SWEEP_DIR)/*.yaml; do \
		echo "=== Launching $$cfg ==="; \
		uv run python sm_jobs/launch_training.py \
			--config $$cfg \
			--bucket $(SM_BUCKET) \
			--data-bucket $(SM_DATA_BUCKET) \
			--role $(SM_ROLE) \
			--region $(AWS_REGION) \
			--instance-type $(SM_INSTANCE) \
			--data-prefix $(SM_DATA_PREFIX) \
			--usrec-prefix raw/USREC.csv \
			--mlflow-uri "$(MLFLOW_URI)" \
			--image-uri $(IMAGE_URI) || exit 1; \
	done

# Parallel sweep: submit all jobs with bounded concurrency, then poll until all finish.
# Override MAX_PARALLEL=N (default 4) and SM_SWEEP_DIR.
# Each submission's stdout/stderr is captured to logs/sm_sweep/<config>.submit.log.
# Job names are appended to .sm_sweep_jobs.txt for the poll loop.
# Poll status table is written to logs/sm_sweep/poll.log (tail -f to watch).
MAX_PARALLEL    ?= 4
SM_JOBS_FILE    ?= .sm_sweep_jobs.txt
SM_LOG_DIR      ?= logs/sm_sweep
sm-sweep-parallel:
	@mkdir -p $(SM_LOG_DIR)
	@rm -f $(SM_JOBS_FILE) $(SM_LOG_DIR)/poll.log
	@n=$$(ls $(SM_SWEEP_DIR)/*.yaml | wc -l); \
		echo "=== Submitting $$n jobs (max $(MAX_PARALLEL) in flight) ==="; \
		echo "=== Per-job logs: $(SM_LOG_DIR)/<cfg>.submit.log  Poll log: $(SM_LOG_DIR)/poll.log ==="
	@ls $(SM_SWEEP_DIR)/*.yaml | xargs -n1 -P $(MAX_PARALLEL) -I {} sh -c '\
		cfg={} ; \
		name=$$(basename $$cfg .yaml) ; \
		log=$(SM_LOG_DIR)/$$name.submit.log ; \
		uv run python sm_jobs/launch_training.py \
			--config $$cfg \
			--bucket $(SM_BUCKET) \
			--data-bucket $(SM_DATA_BUCKET) \
			--role $(SM_ROLE) \
			--region $(AWS_REGION) \
			--instance-type $(SM_INSTANCE) \
			--data-prefix $(SM_DATA_PREFIX) \
			--usrec-prefix raw/USREC.csv \
			--mlflow-uri "$(MLFLOW_URI)" \
			--image-uri $(IMAGE_URI) \
			--no-wait > $$log 2>&1 ; \
		jn=$$(grep "^JOB_NAME=" $$log | cut -d= -f2) ; \
		if [ -n "$$jn" ]; then echo "$$jn  $$cfg" >> $(SM_JOBS_FILE) ; echo "submitted: $$jn ($$name)" ; \
		else echo "FAILED: $$name -> see $$log" ; fi'
	@echo "=== Submission phase done. Polling every 60s -> $(SM_LOG_DIR)/poll.log ==="
	@$(MAKE) sm-poll

# Poll all jobs in $(SM_JOBS_FILE) until terminal. Status table -> poll.log; concise summary on stdout.
sm-poll:
	@test -s $(SM_JOBS_FILE) || (echo "no jobs in $(SM_JOBS_FILE)"; exit 1)
	@mkdir -p $(SM_LOG_DIR)
	@while :; do \
		pending=0; done_=0; failed=0; \
		ts=$$(date '+%H:%M:%S'); \
		printf "\n=== %s ===\n" "$$ts" >> $(SM_LOG_DIR)/poll.log; \
		for jn in $$(awk '{print $$1}' $(SM_JOBS_FILE)); do \
			st=$$(aws sagemaker describe-training-job --training-job-name $$jn --region $(AWS_REGION) --no-cli-pager --query TrainingJobStatus --output text 2>/dev/null || echo "Unknown"); \
			printf "  %-44s %s\n" "$$jn" "$$st" >> $(SM_LOG_DIR)/poll.log; \
			case "$$st" in Completed) done_=$$((done_+1)) ;; Failed|Stopped) failed=$$((failed+1)) ;; *) pending=$$((pending+1)) ;; esac; \
		done; \
		total=$$(wc -l < $(SM_JOBS_FILE)); \
		printf "[%s] %d/%d done, %d failed, %d running\n" "$$ts" "$$done_" "$$total" "$$failed" "$$pending"; \
		if [ $$pending -eq 0 ]; then echo "=== all jobs terminal (done=$$done_ failed=$$failed) ==="; break; fi; \
		sleep 60; \
	done

# Full reproduction pipeline: download → generate → test → sweep
reproduce: clean download-data generate-sweep test sweep

# ---- Phase D targets ----

# D2: Compute data SHA-256 and inject into a .frozen.yaml copy of the config.
# Usage: make freeze-config CONFIG=configs/sagemaker_ae_only_W6_d7_K4_b1.yaml
FREEZE_CONFIG ?= configs/sagemaker_ae_only_W6_d7_K4_b1.yaml
freeze-config:
	uv run tcc data freeze-config $(FREEZE_CONFIG)

# D3: Aggregate all encoder × downstream results into results/encoders_panel.csv.
ablation-aggregate:
	uv run python scripts/aggregate_ablation.py

# D5: Thesis-specific end-to-end reproduction (winner config only, no full sweep).
# Requires: data snapshots present + clustering ablation results present.
reproduce-thesis: pull-nber download-data ablation-aggregate phase-c export
	@echo "=== Thesis reproduction complete. Check results/encoders_panel.csv ==="

# ── Phase E: alternative encoder battery ──────────────────────────────────────
PHASE_E_ENCODERS ?= moment,ts2vec,patchtst,timesnet,tfc,hamilton_hmm,ms_var,bocpd

# Install optional Phase E Python dependencies.
# momentfm v0.1.5 (git) is compatible with transformers>=4.54.1 — all encoders in one env.
# MOIRAI (uni2ts) is not on PyPI — installed separately here.
phase-e-install:
	uv sync --extra phase_e
	uv run pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
	@echo "Phase E deps installed (momentfm + PatchTST-compatible transformers + MOIRAI)."

# Run full Phase E battery (all encoders, full permutation/bootstrap reps).
# Prereqs: run `make download-data pull-nber` first if data/raw/ is empty.
phase-e:
	uv run python scripts/run_phase_e_encoders.py --encoders $(PHASE_E_ENCODERS)
	uv run python scripts/aggregate_ablation.py
	@echo "Phase E complete — see results/phase_e_comparison.csv"

# Quick iteration mode (100 perm/boot reps instead of 1000).
phase-e-fast:
	uv run python scripts/run_phase_e_encoders.py --encoders $(PHASE_E_ENCODERS) --fast
	@echo "Phase E (fast) complete — see results/phase_e_comparison.csv"

# ── Methodological sprint experiments (F1, F6-F9) ─────────────────────────────
# Sprint 1 (F1 + F7): MCC/PR-AUC/Brier for all 13 encoders + stress-score ablation
sprint1:
	uv run python scripts/sprint1_metrics_and_stress.py
	@echo "Sprint 1 complete — results/sprint1/"

# Sprint 2 (F8): Consensus clustering on iTransformer embeddings (B=200)
sprint2:
	uv run python scripts/sprint2_consensus_clustering.py
	@echo "Sprint 2 complete — results/sprint2/"

# Sprint 3 (F6 + F9): Macro profiles per cluster + tier-stratified benchmark
sprint3:
	uv run python scripts/sprint3_macro_profiles_tiers.py
	@echo "Sprint 3 complete — results/sprint3/"

# Run all three sprint experiments sequentially
sprints: sprint1 sprint2 sprint3
	@echo "All sprints complete."

# Sprint 4 (F2): Bai-Perron breakpoint pre-registration on all 122 FRED-MD series
sprint4-f2:
	uv run python scripts/sprint4_f2_bai_perron.py
	@echo "Sprint 4 F2 complete — results/sprint4/"

# Sprint 4 (F10): Causal inference — Granger causality + VAR IRF/FEVD
sprint4-f10:
	uv run python scripts/sprint4_f10_granger.py
	@echo "Sprint 4 F10 complete — results/sprint4/"

# Sprint 5 (F3): Window sensitivity W∈{3,6,9,12,18,24}
sprint5-f3:
	uv run python scripts/sprint5_f3_window_sensitivity.py
	@echo "Sprint 5 F3 complete — results/sprint5/"

# Sprint 6 (F4): InfoNCE contrastive training vs MSE-only
sprint6-f4:
	uv run python scripts/sprint6_f4_infonce.py
	@echo "Sprint 6 F4 complete — results/sprint6/"

# Sprint 7 (F5): Domain adaptation of Phase E encoders
sprint7-f5:
	uv run python scripts/sprint7_f5_domain_adaptation.py
	@echo "Sprint 7 F5 complete — results/sprint7/"

# Run all deferred fragility experiments (F2, F3, F4, F5, F10)
sprints-deferred: sprint4-f2 sprint4-f10 sprint5-f3 sprint6-f4 sprint7-f5
	@echo "All deferred fragility sprints complete."

# Run everything (Sprint 1-3 + deferred F2/F3/F4/F5/F10)
all-sprints: sprints sprints-deferred
	@echo "Full sprint suite complete."

# Show available targets
help:
	@echo "Available targets:"
	@echo "  install        - Install dependencies"
	@echo "  download-data  - Download FRED-MD snapshot"
	@echo "  generate-sweep - Generate sweep configs"
	@echo "  train          - Train single config"
	@echo "  sweep          - Run full sweep"
	@echo "  baselines      - Run baseline comparisons"
	@echo "  test           - Run unit tests"
	@echo "  test-all       - Run all tests"
	@echo "  test-quality   - Run quality gate tests"
	@echo "  lint           - Check code style"
	@echo "  format         - Auto-fix code style"
	@echo "  ui             - Launch MLflow UI"
	@echo "  export         - Export results to LaTeX"
	@echo "  reproduce      - Full reproduction pipeline"
	@echo "  freeze-config  - Inject data SHA-256 into config .frozen.yaml"
	@echo "  ablation-aggregate - Aggregate all results → results/encoders_panel.csv"
	@echo "  reproduce-thesis   - Thesis-specific end-to-end reproduction (no full sweep)"
	@echo "  phase-e-install    - Install Phase E optional deps (momentfm, ts2vec, hmmlearn…)"
	@echo "  phase-e            - Run Phase E alternative encoder battery (full)"
	@echo "  phase-e-fast       - Run Phase E with reduced perm/boot reps (dev mode)"
	@echo "  clean          - Remove cache files"
	@echo ""
	@echo "SageMaker:"
	@echo "  pull-nber      - Download NBER USREC snapshot"
	@echo "  sm-build       - Build training Docker image"
	@echo "  sm-push        - Push image to ECR"
	@echo "  sm-train       - Launch one SageMaker training job"
	@echo "  sm-sweep       - Launch one job per sweep config"
