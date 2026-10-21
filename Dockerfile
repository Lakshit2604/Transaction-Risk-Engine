# Multi-stage production Dockerfile for Transaction Risk Engine
# Supports running either FastAPI service (port 8000) or Streamlit Dashboard (port 8501)

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install system dependencies (libgomp1 required for LightGBM OpenMP multithreading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY reports/ ./reports/
COPY models/ ./models/

# Expose API and Dashboard ports
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command: Run FastAPI service using Uvicorn
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT}
