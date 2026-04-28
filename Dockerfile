# SageMaker training image for tcc_itransformer.
# Base: AWS Deep Learning Container PyTorch 2.4 GPU + Python 3.11.
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu121-ubuntu22.04-sagemaker

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SAGEMAKER_PROGRAM=train_entrypoint.py

WORKDIR /opt/ml/code

# Install only the extra deps not present in the DLC.
COPY pyproject.toml ./pyproject.toml
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

COPY src ./src
COPY scripts ./scripts
COPY sagemaker ./sagemaker
COPY configs ./configs

ENV PYTHONPATH=/opt/ml/code/src:/opt/ml/code

# SageMaker invokes: python <SAGEMAKER_PROGRAM>
RUN ln -sf /opt/ml/code/sagemaker/train_entrypoint.py /opt/ml/code/train_entrypoint.py
