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
make dev                    # backend :8080 + frontend :5173 (Ctrl+C stops both)
make dev BACKEND_PORT=8081  # if 8080 is taken on your machine
```

Or run them in two terminals:

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8080 --reload
cd frontend && npm run dev
```

Open `http://localhost:5173`, click **New training**, try the sample dataset and
watch the full flow: verify → train → live dashboard → evaluation. Check
`/healthz` and `/readyz`. `make dev` keeps all state in `backend/.dev-data`.

## Run on your server in 3 steps

1. **Install prerequisites** — Docker, the Compose plugin, and (for the `-gpu` image)
   the NVIDIA Container Toolkit + a recent NVIDIA driver. See
   [docs/DOCKER_FOR_BEGINNERS.md](docs/DOCKER_FOR_BEGINNERS.md).
2. **Run the image** (CPU here; add `--gpus all` and use `-gpu` on a GPU server):

   ```bash
   docker run -d --name trainhub -p 8080:8080 -v trainhub_data:/data trainhub:0.1.0-cpu
   ```

3. **Open your browser** at `http://<server-ip>:8080` and click **New training** →
   **Try sample dataset** to see the whole flow in under a minute.

Zero-config: defaults are `TRAINHUB_HOME=/data`, `TRAINHUB_DEVICE=auto`, port 8080,
non-root user. All state lives in the `trainhub_data` volume — it survives
`docker rm` and image upgrades.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TRAINHUB_HOME` | `/data` (image) | Workspace root; all state lives here [doc 04] |
| `TRAINHUB_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` [doc 14 §3] |
| `TRAINHUB_ACCESS_TOKEN` | unset | Optional shared token (set it if not localhost) [doc 10 §2] |
| `TRAINHUB_BIND_HOST` | `0.0.0.0` (image) | Address to bind; `127.0.0.1` = local only |
| `TRAINHUB_MAX_CPU_JOBS` | `1` | Concurrent CPU training jobs |
| `TRAINHUB_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

### Upgrading

`docker pull` the new tag and recreate the container; the `/data` volume keeps
your datasets, checkpoints and runs (back it up first):

```bash
docker pull trainhub:0.2.0-cpu
docker stop trainhub && docker rm trainhub
docker run -d --name trainhub -p 8080:8080 -v trainhub_data:/data trainhub:0.2.0-cpu
```

### Offline / air-gapped servers

If the server has no internet for checkpoint downloads, copy the weights to it
and import them once (sha256 is verified against the catalog):

```bash
docker exec trainhub trainhub checkpoints import ./weights.bin --catalog-id rtdetr-l@1.0
```

Or pre-copy a `checkpoints/store` folder into the volume and run
`docker exec trainhub trainhub checkpoints rescan`.

## With Docker Compose

```bash
cd deploy && cp .env.example .env   # then edit (set TRAINHUB_ACCESS_TOKEN)
docker compose up -d
```

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
