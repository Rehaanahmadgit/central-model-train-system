# 03 — System Architecture (No Database, No User Data)

## 1. High-level diagram (logical)
```
 [React UI] <--HTTPS REST--> [FastAPI app process]  <-- single process owns all state -->  [Workspace on disk]
     ^                          |   |                                                   (JSON / JSONL / Parquet / YAML)
     |                          |   +-- Job Manager (asyncio): queue, GPU slots, supervision
  WebSocket                     |   +-- Event Broadcaster (in-memory pub/sub) + events.jsonl writer
     |                          |   +-- Verification Runner (process pool)
     +--------------------------+
                                v
                  [Training subprocess] (one per run, own process group)
                  sends events over a pipe -> Job Manager
                                |
                         [Checkpoint store on disk]
```
Reverse proxy (nginx/Caddy) in front for TLS and optional shared-token check.

## 2. Components
### 2.1 API service (FastAPI)
- Async; handles uploads, verification requests, run control, queries, WebSocket.
- **No heavy ML in the API event loop.** Verification and training run in separate processes.
- No authentication layer for users. Optional global token check as a FastAPI dependency (`TRAINHUB_ACCESS_TOKEN`).

### 2.2 Job Manager (in the API process)
- Holds the in-memory queue of runs and verifications; persists state to `run.json` / `verification.json` on every transition.
- Assigns GPU slots using file locks (`locks/gpu-{idx}.lock`); one training job per GPU; CPU-only jobs limited by a concurrency setting.
- Spawns each training job as a **subprocess** (`multiprocessing` spawn context or `subprocess`), supervises heartbeat, forwards events, enforces timeouts, kills process group on stop.
- On startup runs **crash recovery** (doc 04 §3 rule 6).

### 2.3 Verification Runner
A `ProcessPoolExecutor` (CPU) executing the check pipeline; progress events go back to the Job Manager through a queue.

### 2.4 Event Broadcaster
- In-memory pub/sub per run (`asyncio` queues) feeding connected WebSockets.
- Every event is also appended to `runs/{id}/events.jsonl` with a monotonically increasing `seq`.
- Reconnecting clients send `last_seq`; the server replays from the file (using `events.idx` offsets) then switches to live. A page refresh therefore restores everything.

### 2.5 Storage
Filesystem only, see doc 04. A small in-memory index (projects, runs, datasets) is built at startup from the files and updated on writes; it is a rebuildable cache, never the source of truth.

### 2.6 Frontend
React SPA: wizard flow, live run dashboard, run list/compare, checkpoints, model registry, audit view, system health. REST for data, WebSocket for live updates.

## 3. Key flows

### 3.1 Upload + verification flow
1. UI uploads chunks → `tmp/uploads/{upload_id}`; on complete, API safely extracts, computes the **dataset hash**, moves to `datasets/{hash}/raw` (if the hash already exists, reuse it — dedup).
2. UI calls `POST /verifications` with `dataset_hash`, `model_family`, optional `task_type`.
3. Job Manager computes `cache_key = sha256(dataset_hash|model_family|verifier_version)`. If `cache/verif/{cache_key}.json` exists → return the stored report immediately.
4. Otherwise a worker runs the pipeline; each check result is appended to `checks.jsonl` and broadcast live.
5. On finish: `report.json` (derived facts: classes, counts, splits, stats, task type, recommended checkpoint, recommended hyperparameters) and `manifest.parquet` are written atomically; cache pointer is created.
6. UI shows pass → **Start Training** enabled.

### 3.2 Training flow
1. UI calls `POST /runs` with `verification_id` + config. API checks the verification passed and is not stale (dataset hash exists, verifier version matches).
2. Job Manager writes `run.json` (`queued`), appends audit line, puts the run in the queue.
3. When a slot frees: state `preparing` → checkpoint resolver (doc 07) → load `report.json` and `manifest.parquet` (no recomputation).
4. State `running`: trainer plugin in subprocess emits events → pipe → Job Manager → `events.jsonl` + WebSocket.
5. End: evaluation, artifacts written, state `completed` / `failed` / `stopped`.

### 3.3 Run state machine
`queued → preparing → running ⇄ paused → evaluating → completed`
Any state → `failed` or `stopped`. After a server restart, running states become `interrupted` (can `resume`). `retry` creates a new run with `parent_run_id`.

## 4. Caching strategy
| What | Where | Key |
|---|---|---|
| Verification derived facts + manifest | `cache/verif/` + `verifications/{id}/` + `datasets/{hash}/manifest.parquet` | hash of dataset + model_family + verifier_version |
| Checkpoint files | `checkpoints/store/` content-addressed | sha256 |
| Checkpoint index | `checkpoints/index.json` | catalog id |
| Live run snapshot | in-memory (rebuilt from `events.jsonl` after restart) | run id |
| List indexes | in-memory | rebuilt from files |
Invalidation is automatic: a different dataset hash or verifier version = different key.

## 5. Plugin architecture
Each trainer plugin implements the interface in doc 08 and registers itself with metadata: supported tasks, formats, hyperparameter JSON Schema (UI forms are generated from it), compatible checkpoints.

## 6. Repository layout (suggested monorepo)
```
trainhub/
  backend/app/
    api/            # routers
    core/           # config, logging, optional token dependency
    storage/        # atomic file io, jsonl writer, locks, indexes, recovery
    schemas/        # pydantic
    services/       # business logic
    realtime/       # broadcaster + websocket
    verification/   # checks, runner, report
    checkpoints/    # catalog, resolver, downloader
    training/       # job manager, plugin base, plugins/, subprocess entry
  backend/tests/
  frontend/src/     # pages, components, hooks, api, store
  deploy/           # docker-compose, nginx
  docs/             # these documents
```

## 7. Deployment
Docker Compose: `api` (with GPU access), `frontend`+`nginx`, one volume mounted at `TRAINHUB_HOME`, optional `prometheus`/`grafana`. Dev machines without GPU run CPU-only plugins.
Scale-out (not in v1): shared filesystem + a second Job Manager would need leader election — deliberately out of scope; do not add a database to solve it.

## 8. Failure handling
- Training subprocess crash/OOM: parent detects exit/heartbeat loss → state `failed` with reason; API unaffected; resume offered from `last.ckpt`.
- API process restart: crash recovery marks running runs `interrupted`, re-queues queued ones, repairs JSONL tails.
- Partial download: only in `.tmp/*.part`; never in `index.json`.
- Disk full: preflight check before upload/download/training; cleanup per retention policy; alert.
- Corrupt JSON file: loader quarantines it to `*.corrupt` and rebuilds from events/other files where possible; surfaced in system health.
