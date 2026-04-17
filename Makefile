.PHONY: train sweep test lint ui baselines export install clean download-data generate-sweep help reproduce

# Install all dependencies
install:
	uv sync

# Download FRED-MD data snapshot with SHA-256 verification
download-data:
	uv run python scripts/download_data.py

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
