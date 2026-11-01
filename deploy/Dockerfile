# syntax=docker/dockerfile:1
# Multi-stage build: Node builds the frontend, Python serves it [doc 14 §4].
# Base image tags: VERIFY against official docs at build time (doc 14 §4).
# Checkpoints/datasets/runs are NEVER baked in — they live in the /data volume [doc 14 §5].

ARG NODE_IMAGE=node:22-slim
# D54: runtime standardized on Python 3.12 across laptop, CI and the image.
# python:3.12-slim verified: pullable on docker.io, currently Python 3.12.x.
ARG PY_BASE=python:3.12-slim
ARG VERSION=0.1.0
ARG GIT_COMMIT=unknown

# --- Frontend build ---------------------------------------------------------
FROM ${NODE_IMAGE} AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- torch-base: CPU torch by default ------------------------------------------
# torch is installed BEFORE the app code so an app-only change does not
# invalidate this large layer [doc 14 §4]. CPU wheel: torch==2.7.1+cpu (D80).
FROM ${PY_BASE} AS torch-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# --- torch-base-gpu: CUDA torch from PyPI, no build-args ------------------------
# torch 2.7.1 on PyPI is the CUDA 12.6 build: it depends on the nvidia-*-cu12
# wheels, which bundle the CUDA runtime into site-packages. The client needs
# only the NVIDIA driver + Container Toolkit on the host — no host CUDA toolkit.
# VERIFY at build time (done 2026-10-20): torch==2.7.1 on PyPI declares
# nvidia-cuda-runtime-cu12==12.6.77 etc. Driver >= 525.60.13 is required.
# The index URL is hardcoded per target — the GPU image never takes build-args.
FROM ${PY_BASE} AS torch-base-gpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://pypi.org/simple

# --- Shared backend runtime: app code AFTER torch -------------------------------
FROM torch-base AS base
ARG VERSION=0.1.0
ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAINHUB_HOME=/data \
    TRAINHUB_FRONTEND_DIST=/app/frontend-dist \
    TRAINHUB_BIND_HOST=0.0.0.0
WORKDIR /app
# OCI labels: version + git commit [doc 14 §4, DIST-09].
LABEL org.opencontainers.image.title="TrainHub" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.description="Central ML training platform (no database, no user data)"
COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app
# Runtime deps + scikit-learn (training extra). torch is already in the base.
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

# --- CPU target (laptop) [doc 14 §2] -------------------------------------------
FROM base AS cpu

# --- GPU target (server) [doc 14 §2, §4] -----------------------------------------
# Same runtime as `base`, rebuilt on top of the CUDA torch image. Docker cannot
# rebase `base` onto a different torch stage without build-args, so the runtime
# steps are repeated verbatim [doc 14 §4].
FROM torch-base-gpu AS gpu
ARG VERSION=0.1.0
ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAINHUB_HOME=/data \
    TRAINHUB_FRONTEND_DIST=/app/frontend-dist \
    TRAINHUB_BIND_HOST=0.0.0.0
WORKDIR /app
LABEL org.opencontainers.image.title="TrainHub" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.description="Central ML training platform (no database, no user data)"
COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app
RUN pip install --no-cache-dir ".[training]"
COPY --from=frontend-build /build/dist ./frontend-dist
RUN useradd --create-home --uid 10001 trainhub \
    && mkdir -p /data \
    && chown -R trainhub:trainhub /data
USER trainhub
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]

