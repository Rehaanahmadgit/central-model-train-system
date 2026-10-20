# TrainHub — Decisions

Every decision with date, reason, source doc section. `PENDING` = proposed, awaiting user approval.

## 2026-10-17
| # | Decision | Reason / source | Status |
|---|---|---|---|
| D1 | Monorepo `trainhub/` with `backend/`, `frontend/`, `deploy/`, `doc/` | doc 03 §6 (suggested layout) | accepted |
| D2 | Python 3.11+ (local: 3.12.3), FastAPI async, Pydantic v2, ruff + mypy strict, pytest | doc 13 §4 | accepted |
| D3 | React + TypeScript strict + Vite + Tailwind v4 + TanStack Query (later) | doc 00 §5, doc 09 §5 | accepted |
| D4 | Drop `user` filter from Runs list — conflicts with C2; priority 00/01/13 > 09 | master prompt §1 | **user-approved** (docs 08/09 on disk still show old text — user said fixed versions were saved but they did not land; resolution applied anyway) |
| D5 | `stop` → state `stopped`, `USER_STOP` removed from failure classifier | doc 03 §3.3 | **user-approved** (same on-disk caveat as D4) |
| D6 | File renamed `15_MASTER_PROMPT (1).md` → `15_MASTER_PROMPT.md` | doc 00 §6 | **done** |
| D7 | Phase 1 DL plugin = detection (RT-DETR/YOLO-style) + scikit-learn plugin | doc 08 §2, doc 14 §8 | **user-approved** |
| D8 | Charts = ECharts (docs allow ECharts/Recharts) | doc 00 §5, doc 09 §4 | **user-approved** |
| D9 | WebSocket paths literal `/ws/...` | doc 05 §8 | **user-approved** |
| D10 | API port 8080 | doc 14 §7 | accepted |
| D11 | Verification ID `ver_{YYYYMMDD}_{HHMMSS}_{4hex}` | doc 04 §4 | **user-approved** |
| D12 | Verification states `queued\|running\|passed\|passed_with_warnings\|failed` | doc 06 §1 | **user-approved** |
| D13 | Plugin metadata gains `description: str` | master prompt §4.2 | **user-approved** |
| D14 | Ambiguous format → `requires_input` + `POST /verifications/{id}/resolve` | doc 06 §3 | **user-approved** |
| D15 | Upload resume client-driven; add `GET /uploads/{id}` | doc 05 §1 | **user-approved** |
| D16 | `format_detected` written by verification, copied to `meta.json` | doc 04 §1 vs 06 §2 | **user-approved** |
| D17 | Checkpoint URLs/sha256 stay `TODO(user)` until owner supplies verified values | doc 07 §2 | accepted |
| D18 | SPA served by a last-added catch-all GET route (not StaticFiles mount at "/" — the mount shadowed routers in this Starlette version); unknown `api/` or `ws` paths keep the 404 error envelope | doc 14 §4, doc 05 §0 | accepted (verified live) |
| D19 | `/healthz` + `/readyz` at root (HEALTHCHECK calls `/healthz`); API docs under `/api/v1/docs` | doc 14 §4, doc 05 §7 | accepted |
| D20 | TestClient dev dependency = `httpx2` (starlette in this stack deprecates httpx for TestClient) | real warning observed | accepted |
| D21 | Theme preference stored in browser `localStorage` only; never sent to the server | doc 10 §1 | accepted |
| D22 | uvicorn access log disabled by default (contains client IPs) | doc 10 §1 | accepted |
| D23 | Error envelope codes `HTTP_ERROR`, `VALIDATION_ERROR`, `INTERNAL` | doc 05 §0 (codes not enumerated) — ASSUMPTION | accepted |
| D24 | TypeScript pinned 5.9.3 (latest 5.x), not 7.0.2 | toolchain stability with Vite 8 | accepted |
| D25 | Versions pinned from live registries 2026-10-17 (fastapi 0.143.3, pydantic 2.14.1, uvicorn 0.54.0, ruff 0.17.1, mypy 2.4.0, pytest 9.1.1, httpx2 2.13.1, vite 8.3.4, react 19.3.1, tailwind 4.3.3, TS 5.9.3, @vitejs/plugin-react 6.1.2) | doc 13 §4 pin versions | accepted |
| D26 | Docker: `docker compose` plugin missing on laptop → `docker compose config` and image builds NOT RUN this slice; base tags `python:3.11-slim` / `node:22-slim` marked VERIFY | doc 14 §4 | accepted (VERIFY at build time) |
| D27 | Recovery enforces `schema_version == 1` on `run.json`; anything else (incl. missing) → quarantine `*.corrupt` | doc 04 §2/§3 rule 7 | accepted |
| D28 | Recovery collects `queued` runs ordered by `queued_at` into `RecoveryReport.requeued_run_ids`; the Job Manager (slice 8) performs the actual re-queue | doc 04 §3 rule 6, doc 03 §2.2 | accepted |
| D29 | Audit writer rejects forbidden fields incl. `user_id`/`org_id`/`started_by` (doc 13 §1.2) — never written, raises ValueError | doc 04 §2, doc 13 §1.2 | accepted |
| D30 | ID collision caveat: `run_..._{4hex}` has 16 bits of randomness [doc 04 §4] → collisions possible only within the same second; run creation (slice 8) will retry on existing ID | doc 04 §4 | accepted |
| D31 | Recovery report logged at startup; JSONL tail repair covers `runs/*/events.jsonl`, `verifications/*/checks.jsonl`, `audit/*.jsonl` | doc 04 §3 rule 6 | accepted |
| D32 | Project id = `proj_{YYYYMMDD}_{HHMMSS}_{4hex}` (format NOT SPECIFIED in doc 04 §4; non-sequential, not user-derived) | doc 04 §4 | accepted |
| D33 | Optional `project_id` added to dataset `meta.json` + `POST /uploads`; `GET /datasets?project=` filters on it (doc 05 §1 lists a project filter but doc 04 §1 has no linkage field) | doc 04 §1, doc 05 §1 | accepted — additive field, all doc 04 fields preserved |
| D34 | Chunk checksum via header `X-Chunk-Sha256` (doc 05 §1 says "checksum header" without naming it) | doc 05 §1 | accepted |
| D35 | Upload dir persists `upload.json` (init state across restarts; not in doc 04 §1 layout) | doc 05 §1, D15 | accepted |
| D36 | Error codes added: PROJECT_NOT_FOUND/CORRUPT, INVALID_ID, INVALID_FILENAME, UPLOAD_NOT_FOUND/CORRUPT, INVALID_CHUNK_INDEX, CHUNK_CHECKSUM_MISMATCH, INCOMPLETE_UPLOAD, UPLOAD_SIZE_MISMATCH, UPLOAD_TOO_LARGE, ARCHIVE_UNSAFE, ARCHIVE_TOO_LARGE, EMPTY_DATASET, DATASET_NOT_FOUND, DATASET_IN_USE (doc 05 §0 gives only examples) | doc 05 §0 | accepted |
| D37 | Dataset hash = sha256 over sorted `relpath file_sha256` lines ("sorted manifest of file hashes") | doc 04 §4 | accepted |
| D38 | Audit actions `project.created/updated/deleted` added (doc 05 §9 requires audit lines for state-changing endpoints; project actions not enumerated in doc 04 §2) | doc 05 §9 | accepted |
| D39 | Upload accepts zip/tar(gz,bz2,xz)/text-CSV/parquet by magic bytes; limits 100k files, 20 GiB total, zip ratio 200 @ 100 MiB floor (defaults are ASSUMPTIONS) | doc 10 §3 | accepted |
| D40 | WS metric batching: when >1 metric is pending, one envelope `{"type":"metric","data":{"batch":[...]}}` is sent (extension of the single-metric shape) | doc 05 §8 "batching metrics" | accepted |
| D41 | WS heartbeat sent as a JSON envelope `{"type":"heartbeat"}` (envelope types in doc 05 §8 list no heartbeat; protocol-level ping is not exposed by Starlette) | doc 05 §8 | accepted |
| D42 | `uvicorn.error` level raised to WARNING: its INFO connection lines include client IPs [doc 10 §1] | doc 10 §1 | accepted |
| D43 | `/api/v1/system` v1: running/queued jobs from `run.json` scan, disk usage, GPU placeholder "No GPU detected", recovery warnings from startup report | doc 05 §7, doc 14 §3.2 | accepted |
| D44 | Demo fake run: config flags `TRAINHUB_DEMO_FAKE_RUN`/`..._STEPS`/`..._INTERVAL`; deterministic sine-based metrics; `dataset_hash` = 64 zeros (clearly synthetic) | doc 12 Phase 0 | accepted |
| D45 | Frontend: echarts 6.1.0 pinned; demo page lazy-loaded so ECharts is its own chunk; `useRunStream` keeps last seq in localStorage (client-side only) | doc 09 §5, doc 10 §1 | accepted |
| D46 | Verification pool uses `multiprocessing` **fork** context; spawn/forkserver re-import the pytest entry point and deadlock. Workers do file I/O + PIL only | code comment in `app/main.py`; doc 13 §3 | accepted (was undocumented) |
| D47 | Worker writes `checks.jsonl`/`progress.jsonl`; the API process is the single writer of `verification.json` | code comment in `app/services/verifications.py`; doc 04 §3 rule 3 | accepted (was undocumented) |

## 2026-10-20 — audit + doc 16 alignment
| # | Decision | Reason / source | Status |
|---|---|---|---|
| D48 | `make verify` = ruff + mypy + pytest fast tier + frontend `tsc/build`; it becomes the only gate | doc 16 FAST-01 | accepted (implemented + green) |
| D49 | Fix the finished-verification WebSocket by explicitly `await websocket.close()` before returning; Starlette 1.8 no longer auto-closes on endpoint return (verified in `.venv/.../starlette/routing.py:70-86` and by the hanging test) | doc 05 §8 "replay then close" | accepted (implemented + regression test passes) |
| D50 | Docker Compose plugin is now available (v5.6.0); `docker compose config` passes for both deploy files. Image builds stay deferred to D1/D2 | doc 16 FAST-04, DIST-02 | verified, builds NOT RUN |
| D51 | Keep `TRAINHUB_ACCESS_TOKEN` **optional** (doc 01 §4 wins over doc 16 DIST-07) but add a loud startup warning + UI banner when bound to a non-localhost address and unset | doc 01 §4, doc 10 §2, doc 16 DIST-07 | **user-approved 2026-10-20**; implementation lands with the DIST-07 slice |
| D52 | Workspace root is `/data` (not `/data/trainhub` from doc 04 §1) in the image; local default stays `./.dev-data` | doc 16 DIST-04 + master prompt §5 vs doc 04 §1 | **user-approved 2026-10-20** (doc 04 §1 superseded) |
| D53 | The fake hello-world run stays in-process until the Job Manager (S8); M0's "subprocess" wording is not satisfied yet | doc 12 Phase 0 | accepted as a known gap |
| D54 | **Python runtime standardized on 3.12 everywhere** — laptop venv 3.12.3, CI `python-version: 3.12`, Docker `python:3.12-slim`; `requires-python = ">=3.12"`, ruff `py312`, mypy `3.12`. Dependency compatibility verified empirically: all pinned deps installed in the 3.12 venv and the full fast suite (150 tests) passes. doc 13 §4 "3.11+" superseded by author decision. `python:3.12-slim` tag still VERIFY at image build time (doc 14 §4). Side effect: ruff enabled UP047 → `paginate` now uses PEP 695 type params (`def paginate[T]`) | author decision 2026-10-20, doc 13 §4, doc 14 §4 | accepted |
| D55 | Git repo initialized (`main`), baseline commit `a8614b6` (129 files, slices S1–S5). `.vscode/` (machine-specific task files) gitignored; `.env.example` stays tracked via `!**/.env.example` | author decision 2026-10-20 | accepted |
