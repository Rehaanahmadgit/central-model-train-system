# syntax=docker/dockerfile:1
# Multi-stage build: Node builds the frontend, Python serves it [doc 14 §4].
# Base image tags: VERIFY against official docs at build time (doc 14 §4).
# Checkpoints/datasets/runs are NEVER baked in — they live in the /data volume [doc 14 §5].

ARG NODE_IMAGE=node:22-slim
ARG PY_BASE=python:3.11-slim

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
RUN pip install --no-cache-dir .
COPY --from=frontend-build /build/dist ./frontend-dist

# Non-root service user [doc 08 §10]; /data is the persistent volume [doc 14 §7].
RUN useradd --create-home --uid 10001 trainhub \
    && mkdir -p /data \
    && chown -R trainhub:trainhub /data
USER trainhub
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# --- CPU target (laptop) [doc 14 §2] -----------------------------------------
FROM base AS cpu

# --- GPU target (server) [doc 14 §2] ------------------------------------------
# Training slice (S8) switches this target to the CUDA runtime base + CUDA
# PyTorch wheels. VERIFY exact CUDA base tag and torch wheel pairing against
# official docs, matched to the server's NVIDIA driver (doc 14 §4).
FROM base AS gpu
