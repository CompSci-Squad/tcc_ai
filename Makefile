.PHONY: train sweep test lint ui baselines export install clean download-data generate-sweep help reproduce pull-nber sm-build sm-push sm-train sm-sweep

# AWS / SageMaker variables (override on CLI: make sm-build AWS_ACCOUNT=...)
AWS_ACCOUNT ?= $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
AWS_REGION  ?= us-east-1
ECR_REPO    ?= tcc-regime-etl-itransformer
IMAGE_TAG   ?= latest
IMAGE_URI   := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO):$(IMAGE_TAG)
SM_BUCKET   ?= tcc-regime-etl-sagemaker
SM_ROLE     ?= arn:aws:iam::$(AWS_ACCOUNT):role/LabRole
SM_INSTANCE ?= ml.m5.xlarge   # cheap CPU box for AE training (~$0.23/hr); override for sweep
SM_CONFIG   ?= configs/sagemaker_ae_only.yaml
MLFLOW_URI  ?=

# Install all dependencies
install:
	uv sync

# Download FRED-MD data snapshot with SHA-256 verification
download-data:
	uv run python scripts/download_data.py

# Pull NBER USREC recession indicator snapshot
pull-nber:
	uv run python scripts/pull_nber.py

# Generate 36 sweep configuration YAML files
generate-sweep:
	uv run python scripts/generate_sweep_configs.py

# Train a single configuration
train:
	uv run python scripts/run_single.py --config configs/default.yaml

# Run full hyperparameter sweep across all configs
sweep:
	uv run python scripts/run_sweep.py

# Run baseline comparisons (PCA-only, random, etc.)
baselines:
	uv run python scripts/run_baselines.py

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
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

# Auto-fix code style and formatting
format:
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

# Launch MLflow UI for experiment tracking
ui:
	uv run mlflow ui --backend-store-uri file:./results/mlruns --port 5000

# Export results to LaTeX tables and figures
export:
	uv run python scripts/export_results.py

# Log environment details (Python version, packages, GPU)
env-log:
	uv run python scripts/log_environment.py

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

# Launch a single SageMaker training job
sm-train:
	uv run python sm_jobs/launch_training.py \
		--config $(SM_CONFIG) \
		--bucket $(SM_BUCKET) \
		--role $(SM_ROLE) \
		--region $(AWS_REGION) \
		--instance-type $(SM_INSTANCE) \
		--mlflow-uri "$(MLFLOW_URI)" \
		--image-uri $(IMAGE_URI) \
		--data-prefix raw \
		--usrec-prefix raw/USREC.csv

# Launch one SageMaker job per sweep config (sequential; SM handles parallel slots)
sm-sweep:
	@for cfg in configs/sweep/*.yaml; do \
		echo "=== Launching $$cfg ==="; \
		uv run python sm_jobs/launch_training.py \
			--config $$cfg \
			--bucket $(SM_BUCKET) \
			--role $(SM_ROLE) \
			--region $(AWS_REGION) \
			--instance-type $(SM_INSTANCE) \
			--mlflow-uri "$(MLFLOW_URI)" \
			--image-uri $(IMAGE_URI) || exit 1; \
	done

# Full reproduction pipeline: download → generate → test → sweep
reproduce: clean download-data generate-sweep test sweep

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
	@echo "  clean          - Remove cache files"
	@echo ""
	@echo "SageMaker:"
	@echo "  pull-nber      - Download NBER USREC snapshot"
	@echo "  sm-build       - Build training Docker image"
	@echo "  sm-push        - Push image to ECR"
	@echo "  sm-train       - Launch one SageMaker training job"
	@echo "  sm-sweep       - Launch one job per sweep config"
