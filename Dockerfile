# Cancer Mutation Pathogenicity Predictor
# Multi-omics deep learning for predicting pathogenicity of cancer mutations
#
# Build:  docker build -t cancer-pathogenicity .
# Train:  docker run --gpus all -v ./data:/app/data -v ./results:/app/results cancer-pathogenicity
# Eval:   docker run --gpus all -v ./data:/app/data -v ./results:/app/results cancer-pathogenicity python scripts/evaluate.py --checkpoint results/checkpoints/best_model.ckpt
# Infer:  docker run --gpus all -v ./data:/app/data -v ./results:/app/results cancer-pathogenicity python scripts/inference.py --checkpoint results/checkpoints/best_model.ckpt --input data/my_variants.tsv

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Cancer Mutation Pathogenicity Team"
LABEL description="Multi-omics deep learning for cancer mutation pathogenicity prediction"
LABEL version="0.1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir matplotlib-venn

# Copy project source code
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

# Create data and results directories
RUN mkdir -p data/raw data/processed data/splits \
    results/figures results/tables results/checkpoints results/logs \
    notebooks paper

# Default entry point: training
ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--config", "configs/default.yaml"]
