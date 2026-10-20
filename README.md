# TrainHub

Central ML training platform: upload a dataset, pick a model, get automatic
verification, automatic pretrained-checkpoint selection, one-click training and a
real-time audit-grade dashboard. Specification: `doc/00`–`doc/15`.

**Two absolute constraints [doc 01 §3]:**
1. **No database.** All state is files under `TRAINHUB_HOME` [doc 04].
2. **No client/user information.** No accounts, logins, emails, IPs or user agents anywhere [doc 10 §1].

## Layout

- `backend/` — FastAPI (Python 3.11+), Pydantic v2, ruff, mypy strict, pytest
- `frontend/` — React + TypeScript strict + Vite + Tailwind v4
- `deploy/` — Dockerfile (`cpu`/`gpu` targets), compose files [doc 14]
- `doc/` — the specification (source of truth)
- `PROGRESS.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md` — living build-tracking files

## Local development (no Docker needed)

Local runs keep all state in `backend/.dev-data` (gitignored). The Docker image
instead mounts the volume at `/data` and sets `TRAINHUB_HOME=/data`.

```bash
# Backend
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env            # optional; sets TRAINHUB_HOME=./.dev-data, demo run
.venv/bin/uvicorn app.main:app --port 8080

# Frontend (proxies /api and /ws to :8080)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` (frontend) or `http://localhost:8080/api/v1/docs` (API).
With `TRAINHUB_DEMO_FAKE_RUN=true` the backend streams a hello-world run into the
live demo chart (Sidebar → Runs). Check `/healthz` and `/readyz`.

## With Docker

```bash
docker compose -f deploy/docker-compose.dev.yml up      # laptop, CPU (bind-mounted source)
docker compose -f deploy/docker-compose.yml up          # single-container CPU stack, data volume at /data
```

The compose plugin is required; the image targets are `cpu` and `gpu` [doc 14 §4].
Base tags and CUDA wheel pairings must be verified against official docs at build time.

## Tests and quality gates

```bash
cd backend
.venv/bin/ruff check app tests
.venv/bin/mypy app
.venv/bin/pytest
cd ../frontend && npm run build
```

`trainhub doctor` and the `@pytest.mark.gpu` smoke tests arrive with the training slices [doc 14 §3].
