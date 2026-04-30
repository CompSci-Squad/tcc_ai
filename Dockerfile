# SageMaker training image for tcc_itransformer.
# Base: AWS Deep Learning Container PyTorch 2.9.0 CPU + Python 3.12.
# CPU-only because Vocareum quota for GPU instances = 0.
# Tag listed by `aws ecr list-images --registry-id 763104351884 --repository-name pytorch-training`.
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v1.9

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# IMPORTANT: do NOT set WORKDIR /opt/ml/code and do NOT COPY any source or
# pyproject.toml into /opt/ml/code. The SageMaker framework toolkit extracts
# `sourcedir.tar.gz` to /opt/ml/code at runtime; pre-existing files there
# either suppress that extraction or cause the toolkit to `pip install .`
# against the baked pyproject (Python-version mismatch), which shadows the
# real entrypoint and yields:
#   "/opt/conda/bin/python3.11: can't open file '/opt/ml/code/train_entrypoint.py'"
WORKDIR /tmp

# Install only the extra deps not present in the DLC.
RUN pip install --upgrade pip && \
    pip install \
        "hdbscan>=0.8.33" \
        "umap-learn>=0.5.5" \
        "ruptures>=1.1.9" \
        "statsmodels>=0.14.6" \
        "mlflow>=2.10" \
        "pydantic>=2.6" \
        "pyyaml>=6.0" \
        "pyarrow>=15.0.0" \
        "s3fs>=2024.3.0" \
        "joblib>=1.3.0"

# Source code is shipped at runtime by the SageMaker SDK as sourcedir.tar.gz
# and extracted to /opt/ml/code. Do NOT bake source copies or symlinks here.
ENV PYTHONPATH=/opt/ml/code/src:/opt/ml/code
