# syntax=docker/dockerfile:1
# Multi-stage build: Node builds the frontend, Python serves it [doc 14 §4].
# Base image tags: VERIFY against official docs at build time (doc 14 §4).
# Checkpoints/datasets/runs are NEVER baked in — they live in the /data volume [doc 14 §5].

ARG NODE_IMAGE=node:22-slim
# D54: runtime standardized on Python 3.12 across laptop, CI and the image
# (doc 13 §4 "3.11+" superseded by author decision 2026-10-20). VERIFY tag
# against official python images at build time [doc 14 §4].
ARG PY_BASE=python:3.12-slim

# --- Frontend build ---------------------------------------------------------
FROM ${NODE_IMAGE} AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Shared backend base -----------------------------------------------------
FROM ${PY_BASE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAINHUB_HOME=/data \
    TRAINHUB_FRONTEND_DIST=/app/frontend-dist
WORKDIR /app
COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app
# Runtime deps + scikit-learn (training extra). torch is per-target below [doc 14 §4].
RUN pip install --no-cache-dir ".[training]"
COPY --from=frontend-build /build/dist ./frontend-dist

# Non-root service user [doc 08 §10]; /data is the persistent volume [doc 14 §7].
RUN useradd --create-home --uid 10001 trainhub \
    && mkdir -p /data \
    && chown -R trainhub:trainhub /data
USER trainhub
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1
# --no-access-log: uvicorn access lines contain client IPs; never log them [doc 10 §1].
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]

# --- CPU target (laptop) [doc 14 §2] -----------------------------------------
FROM base AS cpu
# CPU-only torch (178 MB wheel vs ~555 MB for the PyPI CUDA build) [D70].
RUN pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# --- GPU target (server) [doc 14 §2] ------------------------------------------
# Training slice (S8) switches this target to the CUDA runtime base + CUDA
# PyTorch wheels. VERIFY exact CUDA base tag and torch wheel pairing against
# official docs, matched to the server's NVIDIA driver (doc 14 §4).
FROM base AS gpu
