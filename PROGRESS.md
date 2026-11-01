# TrainHub — Progress

Living checklist for the build order from doc 13 §5 (12 slices) + Phase 0 exit from doc 12.
Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

## Audit 2026-10-20 (read-only, doc 16 rules)
- `ruff check` ✓ · `mypy app` ✓ · 148 tests pass in 10.74 s; **1 test hangs and never completes** (`test_verification_ws_on_finished_stream_closes`, backend/tests/test_verifications_api.py:189) — fixed in S5.1.
- Frontend `npm run build` ✓ · `docker compose config` ✓ for both deploy files · image build **NOT RUN** (FAST-04).
- Doc drift: this file said S5 was "NEXT", but S5 code already exists (runner, 5 checks, API, WS, checklist UI).
- No `Makefile`/`scripts/verify.sh` yet → FAST-01 not met.

## S5.1 2026-10-20 — quality gate + doc 16 alignment
- `make verify` (ruff + mypy + fast tests + frontend build) green: **150 passed in 8.25 s** [FAST-01].
- `test-fast` / `test-full` / `test-gpu` targets + `slow`/`gpu` pytest markers [FAST-02].
- WS close fix (D49): finished verification streams now close cleanly; the previously hanging test passes.
- Audit lines `verification.started` / `verification.finished` [doc 10 §6] + test.
- Python 3.12 standardized everywhere (D54); `paginate` modernized to PEP 695.
- CI = one `verify` job running the same `make verify PYTHON=python` gate.
- Git: repo initialized on `main`; baseline commit `a8614b6`; S5.1 commit follows (D55).

## Phase 0 — Foundations (doc 12)
- [~] M0 exit: repo, compose, CI, health, logging, fake metric and recovery tests done. Gap: the fake run is in-process asyncio, **not** a subprocess as doc 12 Phase 0 requires; real subprocess arrives with the Job Manager (S8). No live `kill -9` drill; recovery is covered by unit tests only.

## Slices (doc 13 §5)
- [x] S1 — Compose stack (api + frontend + volume), health endpoints, CI, config, structured logging. **DONE 2026-10-17.** ruff ✓ mypy ✓ 15 tests ✓ frontend build ✓ live smoke ✓ (healthz/readyz/SPA/404 envelope). `docker compose config` ✓ 2026-10-20 (both files); image build NOT RUN (FAST-04).
- [x] S2 — Storage module: atomic writes, JSONL appender, locks, IDs, startup recovery skeleton, in-memory index (+ heavy crash-safety tests, doc 11 §3b). **DONE 2026-10-17.** ruff ✓ mypy ✓ 60 tests ✓. Recovery wired into app startup lifespan [doc 03 §2.2].
- [x] S3 — Projects + chunked upload + safe extraction + dataset hash/dedup + delete. **DONE 2026-10-17.** ruff ✓ mypy ✓ 104 tests ✓ live smoke ✓ (project → upload → complete → list → delete).
- [x] S4 — Event broadcaster + `events.jsonl` + WebSocket (snapshot, deltas, `last_seq` resume) + fake publisher + live chart. **DONE 2026-10-17.** ruff ✓ mypy ✓ 126 tests ✓ frontend build ✓ live WS smoke ✓ (snapshot → replay → live metrics, monotonic seq). Phase 0 M0 demo: `TRAINHUB_DEMO_FAKE_RUN=true` streams hello-world metrics.
- [x] S5 — Verification runner + 5 core checks + API + WS + streaming checklist UI. **DONE 2026-10-20.** 8/8 tests pass (hang fixed in S5.1); audit lines added.
- [x] S5.1 — `make verify` + FAST-02 test tiers/markers + WS close fix + verification audit lines + Python 3.12 unification + git init/commit. **DONE 2026-10-20.** `make verify` green: ruff ✓ mypy ✓ 150 fast tests ✓ frontend build ✓.
- [x] S6 — Verification cache + normalized manifest + remaining checks + task detection + recommendations. **DONE 2026-10-20.** 23 checks (doc 06 §4), manifest parquet, `report.json` derived facts, `cache/verif/{key}.json` pointer with cache hit on POST, task detection with reason, hparam + checkpoint recommendations, summary cards + warning acknowledgement in the UI, `tests/fixtures/make_datasets.py` spec table (18 datasets, all outcomes asserted). `make verify` green: **185 tests in 14.44 s**.

- [x] S7 — Checkpoint catalog + resolver + selection v1. **DONE 2026-10-20.** `catalog.yaml` (TODO(user) placeholders, unverified entries refused), `CheckpointResolver` (flock → disk preflight → resumable `.part` → sha256 → atomic move → `index.json`), `/checkpoints` API (list/prefetch/evict/reload), `trainhub checkpoints list|prefetch|import|rescan` CLI, recommended-checkpoint card in the Verify UI. 25 new tests (local HTTP server with Range, sha mismatch, concurrent lock, import, evict, corrupt cache). **Docker sanity passed** (see below).
- [x] S8 — Plugin base + 2 REAL trainers + Job Manager + supervised subprocess + telemetry. **DONE 2026-10-20.** `TrainingEngine` removed the in-process fake run; `JobManager` spawns `python -m app.training.worker_main` in its own process group, reads events off stdout (single events.jsonl writer in the parent), heartbeat every 5 s / 60 s → `failed(worker_lost)`, pause/resume/stop over stdin, CPU concurrency semaphore (`max_cpu_jobs`). Two real trainers: `random_forest` (sklearn, tabular) and `cnn` (tiny PyTorch, image classification) on `manifest.parquet`, `dev_preset` = 2 epochs/32 px. `/runs` + `/models` API. Checkpoint resolution at start only when `checkpoint_catalog_id` is set. `make verify` green: **218 tests in 36.81 s**. GPU slot locks deferred to the GPU slice.
- [x] S9 — Live Run page. **DONE 2026-10-20.** State badge + elapsed/ETA/progress, Pause/Resume/Stop, metric chart (smoothing + zoom), windowed log terminal (last 400 lines, search + level filter), system panel (CPU/RAM, "No GPU detected"), events timeline, Run Info panel (params, dataset hash, checkpoint+sha, seed, env versions, hardware, timestamps), connection indicator, reload restores full state via snapshot + `last_seq` replay. Runs list → LiveRun. Also: two-level gate (`make verify` + `make verify-full`, CI runs full), deterministic fork lock test, ProcessPool shutdown fix, torch 2.7.1 pin upgrade. `make verify` green: **218 tests in 50.73 s**; `make verify-full` green: **220 tests in 54.78 s**. Note: true log virtualization deferred to S11.
- [x] S10 — New Training Wizard end to end. **DONE 2026-10-20.** 5-step stepper (always visible): Dataset (drag-drop + chunked upload with speed/ETA + pause/resume + "Try sample dataset" via `POST /datasets/sample` DIST-11), Model (searchable cards from `/models` with needs-GPU badge), Verify (reuses streaming checklist; Next only on pass, warnings need acknowledgement), Configure (form from hparam schema, Basic/Advanced, checkpoint override, train-from-scratch), Train (summary → `POST /runs` → LiveRun). Home "New training" opens the wizard. Token-aware API client (`lib/api.ts`). Plugin catalog import-free. 7 slow tests marked; `make verify` **215 tests / 31.0 s**, `make verify-full` **222 tests / 68.8 s**. Playwright e2e written, NOT RUN (Chromium not installed).
- [x] S11 — Pause/resume/stop + periodic checkpoints + `interrupted` recovery + failure classification. **DONE 2026-10-20.** UI buttons drive the real subprocess (pause→resume→stop end-to-end test); per-epoch `last.ckpt`/`best.ckpt` with full state (model+optimizer+epoch) and exact resume via `POST /runs/{id}/retry`; `interrupted` runs get a Resume button; failure reasons (WORKER_LOST/OOM/NAN_LOSS/DATA_ERROR/UNKNOWN) shown with fix hints; NaN guard + early stopping (patience 3). Log terminal stays windowed (D89: cap 200 lines, virtualization not worth it). `make verify` **215 tests / 27.6 s**; `make verify-full` **225 tests / 102.9 s**.
- [x] S12 — Evaluation report + MVP acceptance (PRD §10). **DONE 2026-10-20.** Confusion matrix + per-class table (CNN) / basic metrics (RF) written to `artifacts/evaluation.json`, served by `GET /runs/{id}/evaluation`, shown on the run page "Evaluation" tab. Three CNN bug fixes (best epoch, resume-empty-range, patience=0). MVP acceptance table below. `make dev` for local run. DEFERRED to Phase 2 (D99): export, model registry, compare, Optuna, notifications.

## Distribution workstream (doc 16 §2)
- [ ] DIST-01 cpu/gpu + latest tags — Dockerfile has `cpu`/`gpu` targets, but `gpu` is empty and no tag/publish script exists; registry `TODO(user)`.
- [ ] DIST-02 one-command `docker run` — absent from README/deploy.
- [~] DIST-03 compose + `.env.example` — both compose files exist; `deploy/.env.example` missing; README has no copy-paste block.
- [~] DIST-04 zero-config start — `TRAINHUB_HOME=/data` is set in the image and device/port defaults exist; no env-var table in README.
- [ ] DIST-05 `trainhub doctor` — no CLI entry point, no doctor code.
- [ ] DIST-06 README "3 steps" + air-gapped note — absent.
- [ ] DIST-07 access-token warning + UI banner — only the config field exists; no middleware, warning, or banner.
- [x] DIST-08 no telemetry/phone-home — by inspection, no telemetry/update-check code or dependency.
- [~] DIST-09 small/reproducible image — multi-stage, non-root, healthcheck, pinned direct deps; no labels, SBOM/licence list, or transitive lock.
- [ ] DIST-10 upgrade path docs — absent.
- [ ] DIST-11 sample dataset + tiny model preset — absent in backend and frontend.
- [ ] DIST-12 version/driver compatibility table — absent.

## Fast dev loop (doc 16 §3)
- [x] FAST-01 one `make verify` — created at repo root; CI runs the same gate (`make verify PYTHON=python`).
- [x] FAST-02 `test-fast` / `test-full` / `test-gpu` — Makefile targets + `slow`/`gpu` markers registered in pyproject.
- [x] FAST-03 one verify + one targeted run per slice — followed this slice (one `make verify`, no command spam).
- [x] FAST-04 no Docker builds mid-slice — image build NOT RUN.
- [x] FAST-05/06 file and reply discipline — state files updated, no spec-index repeat.

## MVP acceptance (doc 01 §10) — evidence
| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Upload → model → verify → pass/fail details | ✅ | `test_verification_full_flow_passes`, `test_samples.py`, e2e `wizard.spec.ts` |
| 2 | Pass → Start Training → job starts | ✅ | `test_random_forest_trains_end_to_end` (real subprocess) |
| 3 | Checkpoint downloaded once, reused | ⚠️ resolver only | `test_second_resolve_uses_cache_without_http`, `test_two_resolvers_download_exactly_once`; no trainer consumes a pretrained weight yet (D100) |
| 4 | Live loss/accuracy + logs; refresh restores state | ✅ | `test_ws_stream.py` snapshot/resume; S9 LiveRun |
| 5 | `run.json` full record, no personal data | ✅ | `test_random_forest_trains_end_to_end` (env fields), `test_workspace_contains_no_personal_data` |
| 6 | Killed subprocess → `failed` + reason; API up | ✅ | `test_kill_worker_marks_failed_worker_lost` |
| 7 | Restart mid-training → `interrupted`, resume | ✅ | `test_restart_marks_running_run_interrupted` + `test_exact_resume_from_last_checkpoint` |
| 8 | Privacy grep: zero emails/IPs/user-agents | ✅ | `test_workspace_contains_no_personal_data` (incl. IPv6) |

## MVP acceptance (doc 01 §10)
1. [x] Upload → model → verification → pass/fail details (S3+S5+S6).
2. [x] Pass → Start Training → job starts (S8: POST /runs spawns a real subprocess).
6. [x] Killed trainer subprocess → `failed(worker_lost)`; API stays up (S8 slow test).
2. [ ] Pass → Start Training → job starts.
3. [ ] Checkpoint downloaded once, reused (log + `index.json`).
4. [ ] Live loss/accuracy + logs; refresh restores state.
5. [ ] `run.json` complete, no personal data.
6. [ ] Killed trainer subprocess → `failed` + reason; API up.
7. [x] Restart mid-training → `interrupted`, resume from `last.ckpt` (S11: recovery marks interrupted; retry `resume_from_last` restores CNN model+optimizer+epoch; RF resume rejected with RESUME_UNAVAILABLE).
8. [ ] Privacy grep: zero emails/IPs/user-agents in workspace + logs.

## S6 detail
- Pipeline: 23 checks in doc 06 §2 order; dependency skip chaining; per-check timeout.
- Artifacts per dataset/verification: `manifest.parquet` (11-column Arrow schema), `report.json`, `checks.jsonl`, `progress.jsonl`, `cache/verif/{sha256(dataset_hash|model_family|verifier_version)}.json`.
- Checkpoint recommendations need a catalog; without `checkpoints/catalog.yaml` the API returns `[]` + `checkpoint_catalog_configured: false` (catalog ships in S7) — no invented ids.
- `app/samples/datasets.py` is importable from app code so DIST-11 can reuse it; `tests/fixtures/make_datasets.py` re-exports it.

## S7 detail + Docker sanity (2026-10-20)
- `docker build -f deploy/Dockerfile --target cpu` → success; `docker run -p 18080:8080 -v trainhub_s7_data:/data` → `/healthz` `{"status":"ok"}`, `/readyz` ok, SPA 200, `/api/v1/checkpoints` 200, `docker exec trainhub-s7 trainhub checkpoints list` lists both catalog entries, uid 10001, `/data` layout created.
- Exception to FAST-04 (user-requested, not milestone D1): documented as D63.
- Container logs were empty because uvicorn's startup line shares a logger with IP-bearing connection lines; app now logs one own startup line (home/device/frontend) — no personal data.

## D1 milestone (doc 16) — 2026-10-20
- `trainhub doctor` (DIST-05) with plain-words output + tests; `make dev BACKEND_PORT=…`.
- DIST-07 token: middleware + WS `?token=` + startup warning + UI banner + tests.
- CPU image built (`trainhub:0.1.0-cpu`, 1.93 GB, torch 694 MB); `docker exec … trainhub doctor` shows all OK/WARN; Playwright e2e passes against the container (sample → RF → verify → train → completed); audit trail in volume confirms it.
- OCI labels (version + git commit) fixed and verified; `python:3.12-slim` verified.
- README "3 steps" + env table + upgrade + offline import; `deploy/.env.example`; `docs/DOCKER_FOR_BEGINNERS.md`.
- `-gpu` target still not published (D2). `make verify`: 226 passed / 33.3 s.

## Non-goal guards (checked every turn)
- No database introduced. No client/user info introduced.

## D2 — GPU image + distribution (2026-10-20)
- Pretrained checkpoint now actually loaded into CNN (`cnn.py:_load_pretrained`), proven by 8 CPU tests (commit 2789836). D100 superseded.
- GPU Dockerfile target implemented: torch 2.7.1 from PyPI (CUDA 12.6 build, verified via wheel metadata). No host CUDA toolkit needed.
- Compose: CPU + GPU services (GPU behind `profiles: ["gpu"]`, `gpus: all`), `shm_size: 8gb`, default port bind `127.0.0.1:8080:8080`.
- `@pytest.mark.gpu` smoke test added (`tests/test_gpu.py`).
- Fixed corrupted `backend/.env.example`, README fence bug, stale torch comments.
- Validation: `make verify` 231 passed / 15 deselected; `make verify-full` 245 passed / 1 skipped. CPU image built + run (healthz/readyz/doctor/SPA/persistence across recreate OK). GPU image built. Host-native GPU validation on RTX 2050: CUDA available, 2048x2048 matmul OK, `pytest -m gpu` 1 passed.
- Container GPU (`--gpus all`) validation BLOCKED on this host: NVIDIA Container Toolkit not installed, no passwordless sudo (D113).

## Non-goal guards (checked every turn)
- No database introduced. No client/user info introduced.

## Slice A — honest evaluation (2026-11-01)
- RT-DETR: `test` split held out (never trained on, never used for checkpoint selection); `unspecified` → train; no-val datasets get a deterministic seeded 80/20 split. Final metrics are read off `best.ckpt`.
- `evaluation.json` now has `val` + `test` (or `test: null` + `test_reason: "no test split"`), per-split `warning` when a split has < 30 images, and the old top-level keys still carry the val numbers (D116, D117).
- Evaluation tab: val mAP50 / test mAP50 tiles, per-class AP for both splits side by side, plain-words notes for "no test split" and "small split".
- Live Run: a finished run no longer shows a "Reconnecting" badge (terminal state stops the reconnect loop) (D119).
- Metric chart: loss on the primary axis, mAP/accuracy on a fixed 0-1 right axis, lr on its own axis (D118). Bug reproduced first (mAP used ~7% of the plot height).
- Tests: `backend/tests/test_rtdetr.py` (split held out, deterministic fallback, report shape, `test: null`, small-split warning, best-checkpoint reload), `frontend/e2e/chart-axes.spec.ts` + terminal-state cases in `runstream-logic.spec.ts`.
- `make verify` green.

## Slice B — predictions on images (2026-11-01)
- RT-DETR writes `artifacts/predictions.json` from the best weights: normalized GT + predictions (score >= `score_threshold`, default 0.25), tp/fp/fn/wrong_class per image, `error_score`, class names, per-split counts (D120, D121).
- New endpoints: `GET /api/v1/runs/{id}/predictions` and `GET /api/v1/runs/{id}/images?path=…` — listed-path + dataset-root confinement (`resolve()` + `is_relative_to`), token-aware like every `/api` route (D122).
- Live Run "Predictions" tab: 12 worst images, GT green / prediction red dashed overlay with labels and scores, split + class + confidence filters (client side) (D123).
- Risk checks: class ids stay in the sorted class-name space; normalized vs pixel boxes; horizontal flip only in train mode; overlay scaled by the real image size. Browser check caught a real NaN bug (GT rows vs prediction rows) — fixed and unit-tested.
- Tests: `match_counts` known answer (hand-computed counts), IoU 0.5 boundary, payload shape, image-endpoint traversal/unlisted/missing/token cases, `e2e/predictions-logic.spec.ts`, and a slow CPU `dev_preset` run asserting `predictions.json` (val + test).
- `make verify` green. Not run: GPU/full-mode training, real container e2e.

## Run-again fixes (2026-11-01)
- Header "Best": falls back to `evaluation.json` val mAP50 when `run.json.best` is still the placeholder, and says so ("from evaluation report") (D124).
- GPU %: 0% is a real reading, a finished run without samples says "n/a (run finished)" (D125).
- Predictions gallery defaults to confidence 0.5 while still showing the stored threshold (D126).
- Config types: schema-driven numeric inputs, `coerceConfig` before POST, and `normalize_config` in `create_run` (rejects with `INVALID_CONFIG` 400); RT-DETR `validate_config` no longer crashes on non-numeric values (D127).
- "Edit & run" Train step was empty (no dataset object) — fixed, the run can be started from that path (D128).
- Tests: `backend/tests/test_training.py` (normalize_config coercion/rejection, run.json types via API, 400 on bad config), `frontend/e2e/run-summary-logic.spec.ts` (best fallback, GPU labels, config coercion/parsing). Repro + fix verified in the browser.
