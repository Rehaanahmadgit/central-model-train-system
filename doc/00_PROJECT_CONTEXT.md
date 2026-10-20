# TrainHub — Project Context (READ THIS FIRST)

> Working name: **TrainHub** (Central ML Training Platform). Rename freely.

## 1. One-line summary
A central, production-grade web platform where an ML engineer uploads a dataset, picks a model, gets automatic dataset verification, automatic pretrained-checkpoint selection, and then trains the model while watching a real-time audit-grade dashboard (metrics, graphs, logs, hardware, parameters).

## 2. Two non-negotiable design constraints (decided by the author)
1. **NO DATABASE.** No PostgreSQL, MySQL, SQLite, MongoDB, etc. All state lives in the **filesystem** (JSON / JSONL / Parquet / YAML files under one workspace directory) plus in-process memory. See `04_STORAGE_LAYOUT.md`.
2. **NO CLIENT / USER INFORMATION.** The platform has no accounts, no login, no user profiles, no organizations, no emails, no roles, no IP/user-agent logging, no analytics about people. The only data it holds is the **datasets the user uploads** and the **technical records of runs** (config, metrics, logs, hardware, versions). Audit records describe *what happened to which dataset/run/checkpoint*, never *who*.

Consequences: no Redis/Celery/Postgres in the stack; single-node design; protection comes from deployment (private network / reverse proxy), plus an optional single shared access token from an environment variable (this identifies nobody).

## 3. Core user flow (the heart of the product)
1. User opens the app and creates/opens a **Project** (just a folder with a name).
2. User **uploads a dataset** (zip / folder / CSV) and **selects a model** (Random Forest, XGBoost, ResNet, YOLO, RT-DETR...).
3. User clicks **Start Verification**.
4. Backend verification pipeline runs (format, labels, class count, corrupt files, splits, imbalance, leakage, bbox validity, ...).
5. Important outputs (class info, stats, dataset fingerprint, detected task, normalized manifest, recommended checkpoint) are **cached on disk** keyed by dataset hash, so the next pipeline stage never recomputes them.
6. If verification passes: UI shows "Dataset verified successfully" and a **Next → Start Training** button.
7. User clicks **Start Training**.
8. Checkpoint resolver: **use the local cache if present (hash verified), otherwise download once** into the checkpoint folder.
9. Training job starts in a separate subprocess managed by the in-process Job Manager.
10. UI shows **real-time dashboard**: loss/accuracy/mAP curves, logs, LR, GPU/CPU/RAM, ETA, all hyperparameters, events (an *audit tracker*, not just charts).
11. After training: evaluation report, error analysis, export (ONNX etc.), model registry (a folder of exported models with metadata files).

## 4. Key design principles
- Production-level backend: async API, supervised subprocess workers, atomic file writes, crash recovery on restart.
- Everything reproducible and auditable (dataset hash, config, code version, library versions, hardware, timestamps) — without storing personal data.
- Verification decides things (task type, checkpoint) — the user should not need to know internals.
- Plugin architecture: each model family is a plugin implementing a common interface.
- Build in phases; do NOT try to support every model/task at once.

## 5. Tech stack (default decisions)
| Layer | Choice |
|---|---|
| API | FastAPI (async), Pydantic v2 |
| Persistence | **Filesystem only** (JSON, JSONL, Parquet, YAML), atomic writes, file locks |
| Job execution | In-process **Job Manager** (asyncio) + `multiprocessing`/subprocess training workers; GPU slots via file locks |
| Live updates | In-process async broadcaster → WebSocket (SSE fallback); replay from `events.jsonl` |
| Cache | Files on disk (`cache/`) + small in-memory LRU |
| Frontend | React + TypeScript + Vite, Tailwind, ECharts/Recharts, TanStack Query |
| ML | PyTorch, scikit-learn, XGBoost, Ultralytics/RT-DETR (as plugins) |
| Tuning | Optuna with in-memory/file-based study (Phase 3) |
| Deploy | Docker image (cpu and gpu targets) built once, pulled on the GPU server; Docker Compose; one persistent data volume (doc 14) |
| Monitoring | Prometheus metrics endpoint, structured JSON logs (no personal data) |

## 6. Document index
- 01_PRD.md — product requirements
- 02_GOALS_AND_METRICS.md — goals, non-goals, success metrics
- 03_ARCHITECTURE.md — system architecture (no DB)
- 04_STORAGE_LAYOUT.md — filesystem layout and file formats (replaces the old DB schema)
- 05_API_SPEC.md — REST + WebSocket contract
- 06_VERIFICATION_PIPELINE.md — verification checks and cache
- 07_CHECKPOINT_MANAGEMENT.md — pretrained selection, download, cache
- 08_TRAINING_ENGINE.md — plugin interface, job lifecycle, realtime events
- 09_UI_UX_SPEC.md — screens and components
- 10_SECURITY_AND_PRODUCTION.md — security, privacy, reliability, ops
- 11_TESTING_STRATEGY.md — tests
- 12_ROADMAP.md — phases and milestones
- 13_LLM_BUILD_INSTRUCTIONS.md — how the building LLM must work
- 14_DEPLOYMENT_AND_ENVIRONMENTS.md — laptop (CPU) dev, Docker images, GPU server deploy, checkpoint caching in the data volume
- 15_MASTER_PROMPT.md — the prompt to paste into the building LLM (reading protocol, anti-hallucination rules, UI/product brief)

## 7. Glossary
- **Run**: one training execution with a fixed dataset version + config. Lives in `runs/{run_id}/`.
- **Dataset version**: immutable snapshot identified by content hash.
- **Verification report**: structured JSON result of all checks.
- **Checkpoint registry**: `catalog.yaml` (what exists) + `index.json` (what is cached locally).
- **Trainer plugin**: class implementing validate/prepare/train/evaluate/export for a model family.
- **Workspace**: the single data directory (`TRAINHUB_HOME`) that holds all state.
