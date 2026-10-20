# TrainHub

Central ML training platform: upload a dataset, pick a model, get automatic
verification, automatic pretrained-checkpoint selection, one-click training and a
real-time audit-grade dashboard. Specification: `doc/00`–`doc/15`.

**Two absolute constraints [doc 01 §3]:**
1. **No database.** All state is files under `TRAINHUB_HOME` [doc 04].
2. **No client/user information.** No accounts, logins, emails, IPs or user agents anywhere [doc 10 §1].

## Layout

- `backend/` — FastAPI (Python 3.12), Pydantic v2, ruff, mypy strict, pytest
- `frontend/` — React + TypeScript strict + Vite + Tailwind v4
- `deploy/` — Dockerfile (`cpu`/`gpu` targets), compose files [doc 14]
- `doc/` — the specification (source of truth)
- `PROGRESS.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md` — living build-tracking files
- `Makefile` — the single quality gate (`make verify`) [doc 16 FAST-01]

## Local development (no Docker needed)

Local runs keep all state in `backend/.dev-data` (gitignored). The Docker image
instead mounts the volume at `/data` and sets `TRAINHUB_HOME=/data`.

```bash
First-time setup:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,training]"
.venv/bin/pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
cd ../frontend
npm install
```

Then run both together from the repo root:

```bash
make dev          # backend on :8080 + frontend on :5173 (Ctrl+C stops both)
```

Or run them in two terminals:

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8080 --reload
cd frontend && npm run dev
```

Open `http://localhost:5173`, click **New training**, try the sample dataset and
watch the full flow: verify → train → live dashboard → evaluation. Check
`/healthz` and `/readyz`. `make dev` keeps all state in `backend/.dev-data`.

## With Docker

```bash
docker compose -f deploy/docker-compose.dev.yml up      # laptop, CPU (bind-mounted source)
docker compose -f deploy/docker-compose.yml up          # single-container CPU stack, data volume at /data
```

The compose plugin is required; the image targets are `cpu` and `gpu` [doc 14 §4].
Base tags and CUDA wheel pairings must be verified against official docs at build time.

## Tests and quality gates

One command runs everything [doc 16 FAST-01]:

```bash
make verify          # ruff + mypy + fast tests + frontend build
make test-fast       # pytest only, excludes slow/gpu tests (< 60 s)
make test-full       # everything; run at milestone end
make test-gpu        # GPU smoke tests; GPU server only
```

In CI the same gate runs as `make verify PYTHON=python`.

`trainhub doctor` and the `@pytest.mark.gpu` smoke tests arrive with the training slices [doc 14 §3].
