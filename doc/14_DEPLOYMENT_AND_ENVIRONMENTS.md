# 14 — Deployment and Environments (Laptop dev → GPU server via Docker image)

## 1. Decision (author)
- **Now:** build and test on a laptop with a weak/no GPU. Nothing is hosted.
- **Later:** when finished, build a Docker image, push to a registry, `pull` it on a GPU server, and train there. Checkpoints are downloaded automatically on the server by the checkpoint resolver (doc 07) the first time they are needed and are reused locally afterwards.
- Same code in both places. Differences are only configuration (env vars) and the base image.

## 2. Environments
| | Dev (laptop) | Prod (GPU server) |
|---|---|---|
| Hardware | CPU or small GPU | NVIDIA GPU(s) |
| Image | `trainhub:<ver>-cpu` | `trainhub:<ver>-gpu` |
| `TRAINHUB_DEVICE` | `cpu` (or `auto`) | `auto` (uses CUDA) |
| Models/datasets | tiny fixtures, tiny models, few epochs | real datasets and models |
| Workspace | local folder / Docker volume | dedicated volume with enough disk |
| Checkpoint downloads | optional (small models only) or mock | real, cached in volume |

## 3. Rules so GPU code works on the server although it cannot be fully tested on the laptop
1. **Device abstraction:** only one place (`core/device.py`) decides the device from `TRAINHUB_DEVICE` (`auto|cpu|cuda|cuda:N`) and what is actually available. Plugins receive the device from `RunContext`; they never hard-code `"cuda"`.
2. **Graceful no-GPU:** system stats use `pynvml` only if available; otherwise the UI shows "No GPU detected" instead of failing. GPU slot logic degrades to CPU concurrency limit.
3. **Resource estimate in verification (doc 06):** if the selected model needs a GPU and none is present, show a clear warning/blocked reason ("This model needs a GPU; this machine has none"), unless a CPU-capable tiny preset is chosen.
4. **Dev presets:** each plugin exposes a `dev_preset` (tiny image size, 1–2 epochs, small batch) so the whole flow can be exercised on a laptop in minutes.
5. **Mark GPU-only tests** with `@pytest.mark.gpu`; CI/laptop skips them; run them once on the server with `pytest -m gpu` as a smoke test.
6. **`trainhub doctor` command** (CLI inside the image): prints Python/torch/CUDA versions, whether CUDA is usable, driver and GPU names, free disk in the workspace, write permission, ability to reach each checkpoint source host, and whether `TRAINHUB_HOME` has leftover `.part` files. Run it first on every new machine.

## 4. Docker images
- Multi-stage Dockerfile: (1) Node stage builds the frontend to static files; (2) Python stage installs pinned dependencies and copies backend + built frontend (served by the API or by nginx).
- Two build targets: `cpu` (slim Python + CPU PyTorch wheels) and `gpu` (CUDA runtime base + CUDA PyTorch wheels). **VERIFY** exact base-image tags and PyTorch wheel/CUDA combination against the official docs at build time and match them to the server's NVIDIA driver version.
- Runs as a non-root user; `HEALTHCHECK` calls `/healthz`; `TRAINHUB_HOME=/data` declared as a volume; no secrets baked in; `.dockerignore` excludes datasets, checkpoints, `.git`, `node_modules`, workspace folders.
- **Never bake into the image:** datasets, checkpoints/pretrained weights, run outputs, tokens. (They would make the image huge and tie data to a version.)
- Label images with version and git commit (`org.opencontainers.image.revision`).

## 5. Checkpoints on the server (the behaviour you asked for)
- Weights live in the **data volume** (`/data/checkpoints/store/...`), not in the image.
- First run that needs a model: resolver downloads once (lock → `.part` → sha256 check → atomic move → update `index.json`). Every later run, even after upgrading to a new image version, finds it in the volume and does **not** download again. This is why the volume must persist across `docker compose pull && up -d`.
- **Prefetch before first training:** `docker compose run --rm trainhub trainhub checkpoints prefetch rtdetr-l` (or `POST /checkpoints/{id}/prefetch`) so the first user does not wait.
- **Offline / restricted server:** provide `trainhub checkpoints import <file> --catalog-id <id>`: verifies sha256 against `catalog.yaml`, moves the file into the store, updates `index.json`. Alternative: pre-copy the `checkpoints/store` folder from another machine into the volume, then run `trainhub checkpoints rescan`.
- The catalog (`catalog.yaml`) ships in the image as defaults; an operator override file in the volume (`/data/checkpoints/catalog.override.yaml`) can add/replace entries without rebuilding the image. (`TODO(user)`: real URLs and sha256 values.)

## 6. Release flow
1. Build locally or in CI: `docker build --target gpu -t <registry>/trainhub:1.0.0-gpu .` (also `-cpu`).
2. Push to a **private** registry (e.g. GHCR, Docker Hub private repo, or a self-hosted registry).
3. On the server (one-time): NVIDIA driver, Docker + Compose plugin, NVIDIA Container Toolkit; `docker login <registry>`.
4. Deploy: `docker compose pull && docker compose up -d`.
5. Verify: `docker compose exec trainhub trainhub doctor` and open `/readyz`.
6. Upgrade: tag a new version, pull, `up -d`. The data volume persists. Persisted JSON files carry `schema_version`; the app migrates on startup (doc 04 §3 rule 7). **Back up the volume before upgrading.**
7. Rollback: set the previous tag in compose and `up -d`. Keep the last 2–3 tags.

## 7. Example compose for the server (illustrative; VERIFY syntax against current Docker docs)
```yaml
services:
  trainhub:
    image: <registry>/trainhub:1.0.0-gpu
    restart: unless-stopped
    environment:
      TRAINHUB_HOME: /data
      TRAINHUB_DEVICE: auto
      # TRAINHUB_ACCESS_TOKEN: set via .env file, never commit
    volumes:
      - trainhub_data:/data
    ports:
      - "8080:8080"      # put behind a reverse proxy/VPN
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: "8gb"      # PyTorch DataLoader workers need shared memory
volumes:
  trainhub_data:
```
For the laptop use a `docker-compose.dev.yml` with the `-cpu` image, a bind-mounted source folder, hot reload, and no GPU section.

## 8. Moving between laptop and server
- Datasets: upload again through the UI on the server (content hash dedups). Do not copy half-processed `tmp/` folders.
- Dev runs are not needed on the server. If a run must move, copy its whole `runs/{id}/` folder while the app is stopped (files are self-contained) and restart so the index rebuilds.
- Never copy the laptop workspace over the server workspace.

## 9. Server sizing hints (guidelines, adjust to real models)
- Disk: datasets + checkpoints + run artifacts grow quickly; keep workspace on a dedicated volume with alerts at 85%.
- RAM: enough for DataLoader workers; set `shm_size`.
- One training job per GPU (doc 08); CPU-only jobs limited by `TRAINHUB_MAX_CPU_JOBS`.

## 10. Checklists
**Before first server run:** `nvidia-smi` works on host → works inside container (`docker run --gpus all ... nvidia-smi`) → `trainhub doctor` all green → prefetch checkpoints → run the tiny smoke dataset → run `pytest -m gpu` once.
**Before every release:** privacy grep passes, crash-safety tests pass, image built for both targets, `docker compose up` from an empty volume works, upgrade from previous version with an existing volume works.
