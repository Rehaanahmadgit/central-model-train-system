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

1. **Install prerequisites** — Docker Engine + Compose plugin, and (for the `-gpu`
   image) a current NVIDIA driver + the NVIDIA Container Toolkit. Verify GPU
   passthrough works before installing TrainHub:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
   ```

   See [docs/DOCKER_FOR_BEGINNERS.md](docs/DOCKER_FOR_BEGINNERS.md) and the
   [GPU compatibility matrix](#gpu-compatibility).

2. **Run the image.** On a GPU server (the normal production case):

   ```bash
   docker run -d --name trainhub --gpus all \
     -p 127.0.0.1:8080:8080 -v trainhub_data:/data \
     trainhub:0.1.0-gpu
   ```

   On a CPU-only machine, drop `--gpus all` and use `trainhub:0.1.0-cpu`.
   `-p 127.0.0.1:8080:8080` binds to localhost only; see
   [remote access](#remote-access) to expose it safely.

3. **Open your browser** at `http://localhost:8080`, then run the one-line health
   check:

   ```bash
   docker exec trainhub trainhub doctor
   ```

   Click **New training → Try sample dataset** to see the whole flow in under a
   minute. All state lives in the `trainhub_data` volume — it survives
   `docker rm` and image upgrades.

Zero-config: defaults are `TRAINHUB_HOME=/data`, `TRAINHUB_DEVICE=auto`, port 8080,
non-root user (uid 10001). The client needs no Python, PyTorch or CUDA toolkit on
the host — the image bundles the CUDA runtime.

### Remote access

By default bind to localhost (`127.0.0.1:8080:8080`) so the UI is not exposed to
the network. To reach it from your own computer, either:

- use an SSH tunnel (recommended): `ssh -L 8080:localhost:8080 user@server`, or
- expose the port (the access token is already on, see below):

  ```bash
  docker run -d --name trainhub --gpus all \
    -p 8080:8080 -v trainhub_data:/data trainhub:0.1.0-gpu
  ```

  Put a TLS reverse proxy in front for anything internet-facing.

### Access token (ON by default)

On first boot TrainHub generates a random token, stores it in
`/data/access-token` (mode 0600) and prints it **once** in the container logs
(`docker logs trainhub | grep "FIRST BOOT"`). Every `/api` call and WebSocket needs
it; the browser asks for it once and keeps it in that browser only. A wrong token
shows a clear "rejected (401)" message.

- Read it again: `docker exec trainhub cat /data/access-token`
- Lost it / leaked it: `docker exec trainhub trainhub --reset-token`, then
  `docker restart trainhub`
- Use your own value: set `TRAINHUB_ACCESS_TOKEN` (not written to disk).
- Opt out (trusted localhost only): `-e TRAINHUB_AUTH_DISABLED=true`. Every boot then
  logs a loud `SECURITY` warning and the UI shows a banner.

### Limits and download safety

- Upload cap: `TRAINHUB_MAX_UPLOAD_SIZE_BYTES` (default 50 GiB). Larger uploads get
  HTTP 413 with the size and the limit in the message.
- Checkpoint downloads accept `https://` sources only, never loopback / private /
  link-local (cloud-metadata) addresses, and follow at most 3 redirects (each
  re-checked). A refused source offers **Import from file**.

### GPU compatibility

TrainHub does **not** support every GPU. The tested matrix is:

| Host | NVIDIA driver | Container Toolkit | Image | Status |
|---|---|---|---|---|
| Linux x86_64 + NVIDIA GPU | ≥ 525.60.13 (CUDA 12.6) | required | `-gpu` (torch 2.7.1, CUDA 12.6) | **tested** (RTX 2050, 4 GB) |
| Linux x86_64, no GPU | any | not needed | `-cpu` | **tested** |

GPU image runtime: torch 2.7.1 + CUDA 12.6 (`nvidia-cuda-runtime-cu12==12.6.77`).
Any NVIDIA GPU supported by CUDA 12.6 should work, but only the configuration
above is actually validated. Untested: AMD/Apple Silicon/Windows GPU containers,
older drivers (< 525.60.13), and multi-GPU slots.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TRAINHUB_HOME` | `/data` (image) | Workspace root; all state lives here [doc 04] |
| `TRAINHUB_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` [doc 14 §3] |
| `TRAINHUB_ACCESS_TOKEN` | auto-generated | Shared token; unset = generated on first boot into `/data/access-token` [doc 10 §2] |
| `TRAINHUB_AUTH_DISABLED` | `false` | `true` turns auth off (loud warning each boot) |
| `TRAINHUB_MAX_UPLOAD_SIZE_BYTES` | 50 GiB | Total upload cap (HTTP 413 above it) |
| `TRAINHUB_BIND_HOST` | `0.0.0.0` (image) | Address to bind; `127.0.0.1` = local only |
| `TRAINHUB_MAX_CPU_JOBS` | `1` | Concurrent CPU training jobs |
| `TRAINHUB_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

### Upgrading

`docker pull` the new tag and recreate the container; the `/data` volume keeps
your datasets, checkpoints and runs (back it up first):

```bash
docker pull trainhub:0.2.0-gpu
docker stop trainhub && docker rm trainhub
docker run -d --name trainhub --gpus all -p 127.0.0.1:8080:8080 \
  -v trainhub_data:/data trainhub:0.2.0-gpu
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
cd deploy && cp .env.example .env   # optional; the token is auto-generated on first boot
docker compose up -d                 # CPU image (no GPU)
docker compose --profile gpu up -d   # GPU image (--gpus all passthrough)
```

The UI binds to `127.0.0.1:8080` by default; set `TRAINHUB_PORT=8080:8080` in
`deploy/.env` to expose it to the network (the access token is on by default).

## Build & publish the image

Build locally (no GPU needed to build):

```bash
docker build -f deploy/Dockerfile --target cpu \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) -t trainhub:0.1.0-cpu .
docker build -f deploy/Dockerfile --target gpu \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) -t trainhub:0.1.0-gpu .
```

Publish to Docker Hub (replace `TODO(user)` with your namespace):

```bash
docker login                                            # interactive; never paste tokens in shell history
docker tag trainhub:0.1.0-cpu TODO(user)/trainhub:0.1.0-cpu
docker tag trainhub:0.1.0-gpu TODO(user)/trainhub:0.1.0-gpu
docker push TODO(user)/trainhub:0.1.0-cpu
docker push TODO(user)/trainhub:0.1.0-gpu
```

Or GitHub Container Registry:

```bash
echo "$CR_PAT" | docker login ghcr.io -u TODO(user) --password-stdin
docker tag trainhub:0.1.0-gpu ghcr.io/TODO(user)/trainhub:0.1.0-gpu
docker push ghcr.io/TODO(user)/trainhub:0.1.0-gpu
```

Tag policy: `X.Y.Z-cpu` / `X.Y.Z-gpu` for releases, plus a moving `latest-gpu`
only after a real GPU-server test. Registry name, token and digests are
`TODO(user)` until supplied — nothing is published automatically.

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
