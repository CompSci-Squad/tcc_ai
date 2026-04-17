.PHONY: train sweep test lint ui baselines export install clean

install:
	uv sync

train:
	uv run python scripts/run_single.py --config configs/default.yaml

sweep:
	uv run python scripts/run_sweep.py

baselines:
	uv run python scripts/run_baselines.py

test:
	uv run pytest tests/unit/ -v

test-all:
	uv run pytest tests/ -v --ignore=tests/quality

test-quality:
	uv run pytest tests/quality/ -v

lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format:
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

ui:
	uv run mlflow ui --backend-store-uri file:./results/mlruns --port 5000

export:
	uv run python scripts/export_results.py

env-log:
	uv run python scripts/log_environment.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
