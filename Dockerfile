# =============================================================================
# Dockerfile - Predictive Healthcare MEDS API
# =============================================================================
# Multi-stage build for a lean production image.
# Stage 1: Install dependencies
# Stage 2: Copy application code and run

# ── Stage 1: Dependencies ────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Application ─────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="Healthcare ML Team"
LABEL description="Predictive Healthcare MEDS API"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/meds_format data/external \
    results/models results/metrics results/predictions results/logs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_CHECKPOINT=results/models/lstm_final.pt
ENV MODEL_CONFIG=configs/model_config.yaml

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
