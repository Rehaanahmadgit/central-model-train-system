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
| D56 | Normalized manifest written as **parquet** via `pyarrow==26.0.0` (verified on PyPI, cp312 manylinux wheels). Rows: path, split, label, class_id, num_boxes, boxes(JSON), width, height, color_mode, file_sha256, annotation_path | doc 06 §6, doc 04 §1 | accepted |
| D57 | `POST /verifications` on a cache hit returns **202** with `{id: <existing>, status, cached: true}` and creates no new verification (doc 05 §3 says 202 + `cached: true`). `GET`'s `cached` reflects the record's own `from_cache` flag, which is always false for a freshly computed report | doc 05 §3, US-06 | accepted |
| D58 | `recommended_checkpoints` is `[]` until `checkpoints/catalog.yaml` exists; the response also carries `checkpoint_catalog_configured` so the UI can explain the gap instead of inventing ids | doc 07 §2 (`TODO(user)` values), doc 13 §3 | accepted |
| D59 | Check thresholds not fixed by the docs are explicit ASSUMPTIONS: `min_samples_per_class` <2 → error, <10 → warn; `dataset_size` <300 → warn; `image_resolution_stats` <32 px → warn; tiny-box share >30% → warn; tabular missing >10% → warn | doc 06 §4 (examples only) | accepted |
| D60 | Task detection heuristics live in `app/verification/tasks.py` and always produce `task_detection_reason` (doc 06 §5). Mask directory → segmentation (not trainable yet) | doc 06 §5 | accepted |
| D61 | `app/core/device.py` is the only device decision point; with no PyTorch installed it reports CPU + a truthful reason, so `resource_estimate` can say "needs a GPU, none here" | doc 14 §3.1/§3.3 | accepted |
| D62 | Synthetic dataset builder lives in `app/samples/datasets.py` (app-importable for DIST-11); `tests/fixtures/make_datasets.py` re-exports it to keep the doc 11 §2 path | doc 11 §2, DIST-11 | accepted |
| D63 | **Docker sanity run inside a slice** (explicit FAST-04 exception, user-requested): `docker build --target cpu` + `docker run` + `/healthz` curl once after S7. Not milestone D1 (no sample-data flow in-container yet) | user instruction 2026-10-20, doc 16 FAST-04 | accepted |
| D64 | `POST /verifications` accepts `preset: auto|dev|full`. `auto` = dev preset on CPU, full on GPU. The no-GPU warning fires only for the **full** preset or a family with no CPU path; a CPU dev preset passes [doc 14 §3.3/§3.4]. Dev preset = 2 epochs, small batch | doc 14 §3.3 | accepted |
| D65 | Cache key stays the documented `sha256(dataset_hash|model_family|verifier_version)`; the `preset` is stored in the pointer and a mismatch is a miss, so a dev verdict cannot be replayed for a full-preset request | doc 06 §6 | accepted |
| D66 | Checkpoint downloads use **stdlib urllib** (Range/retries) — no HTTP client in the runtime image. `httpx2` stays dev-only and pinned (`2.13.1`); app code never imports it | doc 13 §3, doc 10 §9 | accepted |
| D67 | An entry whose `sha256` is a `TODO(user)` placeholder is **unverified**; the resolver refuses to download or import it (`CHECKPOINT_SHA_UNVERIFIED`). Nothing is ever downloaded without a matching hash [doc 10 §4] | doc 07 §2, doc 10 §4 | accepted |
| D68 | The shipped catalog is copied from `app/checkpoints/default_catalog.yaml` into `<home>/checkpoints/catalog.yaml` at startup if absent; `catalog.override.yaml` merges on top [doc 14 §5] | doc 14 §5 | accepted |
| D69 | App emits one startup INFO line (home/device/frontend) because uvicorn's own startup line is suppressed along with IP-bearing connection lines. No personal data in it | doc 10 §1, DIST-05 spirit | accepted |
| D70 | Training deps: `scikit-learn==1.9.1` in a `training` extra (PyPI, cp312 manylinux 9.2 MB). `torch` is **not** in pyproject because the PyPI Linux wheel bundles CUDA (~555 MB); the CPU image installs `torch==2.6.0+cpu` (178 MB) from https://download.pytorch.org/whl/cpu — the newest cp312 linux x86_64 CPU wheel there (2.7+ dropped that tag). The GPU image swaps in the CUDA build. Runtime image ≈ +0.6 GB unpacked for torch on CPU vs ≈ +1.5 GB for the PyPI CUDA wheel | verified against PyPI + pytorch CPU index 2026-10-20 | accepted |
| D71 | Training subprocess protocol: worker → parent over stdout (one JSON event per line: metric/log/state/heartbeat/result); parent → worker over stdin (`{"action":"pause|resume|stop"}`). The parent is the single events.jsonl + run.json writer; the worker never touches them [doc 04 §3 rule 3] | doc 08 §3 step 5 | accepted |
| D72 | `stopped` is a user-requested terminal state with `failure: null` (D5); a worker killed by a signal is classified `WORKER_LOST` even when the kill races ahead of the 60 s heartbeat monitor | doc 08 §3.8, doc 05 §8 | accepted |
| D73 | `POST /runs` requires verification status `passed`/`passed_with_warnings` (not the derived `can_train` field, which is not persisted) | doc 06 §9 | accepted |
| D74 | GPU slot locks are deferred to the GPU slice; on CPU the Job Manager uses an asyncio semaphore sized `max_cpu_jobs` [doc 14 §3.2] | doc 14 §3.2 | accepted |
| D75 | uvicorn access log disabled two ways: `access_logger.disabled` when `TRAINHUB_LOG_ACCESS=false` (programmatic) AND `--no-access-log` in the Docker CMD + compose dev command. Client IPs are never logged [doc 10 §1] | doc 10 §1 | accepted |
| D76 | The in-process fake run (S4 demo) is removed entirely; `TRAINHUB_DEMO_FAKE_RUN*` settings are gone. M0's "fake metric" demo is superseded by the real trainers | doc 12 Phase 0, user instruction | accepted |

## 2026-10-20 — S9 + gate/process/torch fixes
| # | Decision | Reason / source | Status |
|---|---|---|---|
| D77 | Two gate levels: `make verify` (fast, default per slice) and `make verify-full` (full suite incl. `@slow` crash tests). **CI runs `verify-full`.** Milestones must run `verify-full`; per-slice runs use `verify` | doc 16 FAST-02, user instruction | accepted |
| D78 | `test_locks::test_per_run_lock_prevents_interleaving` was flaky because `[os.fork() for _ in range(2)]` let the first child execute the *second* fork too, spawning a third writer (300 records). Now the parent forks each child sequentially and waits — deterministic 200 records, one block transition | observed flake, doc 11 §3b | accepted |
| D79 | Hung ProcessPool root cause: `ProcessPoolExecutor.shutdown(wait=False)` left fork-context verification workers unjoined, which could hang the interpreter at shutdown. Fixed to `shutdown(wait=True, cancel_futures=True)` | observed hang, doc 13 §3 | accepted |
| D80 | **Correction to D70**: my HTML grep of the pytorch CPU index was wrong. `pip download` proves newer cp312 CPU wheels exist: `torch==2.7.1+cpu` (175.8 MB) and `torch==2.9.1+cpu` (184.4 MB), both `manylinux_2_28_x86_64`. Pin upgraded to **2.7.1**. GPU/CPU plan: CPU image installs `torch==2.7.1+cpu` from the pytorch CPU index; the GPU image will install `torch==2.7.1+cu126` (verified: 822.0 MB cp312 manylinux_2_28) on a CUDA 12.6 runtime base (driver ≥ 525, VERIFY at build time) | verified via `pip download --only-binary=:all:` on both indexes 2026-10-20 | accepted |
| D81 | S9 Live Run page: stat tiles + ECharts (smoothing toggle, inside+slider zoom) + log terminal with search/level filter (windowed to last 400) + system panel ("No GPU detected" when `gpu_util` is null) + events timeline + Run Info panel from `GET /runs/{id}`. Worker samples CPU/RAM from `/proc` (no psutil) every heartbeat interval; GPU values stay null on CPU | doc 09 §3, doc 14 §3.2, design brief | accepted |

## 2026-10-20 — S10 wizard + distribution extras
| # | Decision | Reason / source | Status |
|---|---|---|---|
| D82 | `POST /datasets/sample` (DIST-11) generates a synthetic dataset via `app.samples.datasets.build_sample_zip` and installs it through the normal, audited path (safe extraction, content hash, dedup, meta, audit). Kinds: `tabular` (default), `classification`, `yolo` | DIST-11, doc 11 §2 | accepted |
| D83 | Token support lives in one frontend module `lib/api.ts`: `X-Access-Token` header on all fetches, `?token=` on WS, token in `localStorage` per viewer (never sent anywhere but this origin), 401 → one prompt → retry | FR-20, doc 05 §0, doc 10 §1 | accepted |
| D84 | Plugin static metadata moved into `training/registry.py` (import-free). `GET /models` no longer imports torch, so the catalog/wizard load instantly (4.39 s → 0.35 s for the models test). Heavy modules import only at training start | doc 08 §1, doc 05 §2 | accepted |
| D85 | Tests >3 s marked `@pytest.mark.slow` (7 total: 4 real-trainer e2e + checkpoint-retry + 2 crash tests). Fast gate back to **215 tests / 31.0 s**; `verify-full` = **222 tests / 68.8 s** | doc 16 FAST-02, user instruction | accepted |
| D86 | Playwright e2e smoke (`make e2e`, `frontend/e2e/wizard.spec.ts`, `@playwright/test@1.64.1` declared only) written but **NOT RUN** — Chromium is not installed and was not downloaded without approval | doc 11 §1, user instruction | accepted |
| D87 | 5-step wizard (Dataset → Model → Verify → Configure → Train) with always-visible stepper; Home "New training" opens it; on Start Training the app navigates straight to the LiveRun page | doc 09 §2, user instruction | accepted |

## 2026-10-20 — S11 reliability + resume
| # | Decision | Reason / source | Status |
|---|---|---|---|
| D88 | `@playwright/test@1.64.1` is a devDependency with an **exact** pin (not `^`); `package-lock.json` synced via `npm install -D`, then `npm ci` in a clean dir passes. Chromium (browser) is only for `make e2e`, never baked into the image | user instruction, doc 16 DIST-09 | accepted |
| D89 | Log terminal stays **windowed** (useRunStream caps at 200 lines; ≤400 rendered). No react-window: virtualizing ~200 rows adds a dependency + complexity for no measurable gain at this cap. If unbounded logs ever arrive, the cap — not DOM size — is the binding constraint | user instruction, doc 09 §3.3 | accepted |
| D90 | **Exact resume** for CNN: `last.ckpt` stores `{schema_version, epoch, model_state, optimizer_state, best_accuracy}`; `resume()` restores model + optimizer + epoch. RandomForest saves no checkpoints (single fit), so `resume_from_last` for RF is a 409 `RESUME_UNAVAILABLE` — documented limitation | doc 08 §4, user instruction | accepted |
| D91 | NaN/Inf guard (train + val loss) → `NAN_LOSS` failure; early stopping with `patience=3`, `min_delta=0.0` on `val/accuracy` | doc 08 §5 | accepted |
| D92 | `POST /runs/{id}/retry` with `{resume_from_last, name?, config?}`: new run linked by `parent_run_id`, inherits parent config (overridable), and the worker calls `plugin.resume()` on the parent's `last.ckpt` when `resume_from` is set | doc 05 §5 | accepted |
| D93 | Pause→resume→stop and exact-resume are tested at the **API level** (pytest `@slow`), not a second Playwright test — browser pause timing is flaky and adds no coverage beyond the real-subprocess API test | user instruction ("agar 15s ke andar ho, warna API level") | accepted |

## 2026-10-20 — S12 evaluation + MVP acceptance
| # | Decision | Reason / source | Status |
|---|---|---|---|
| D94 | CNN `TrainResult.best.epoch` now reports the real best epoch (tracked on improvement), never 0 on a fresh run | bug fix (user) | accepted |
| D95 | `epoch` is initialized before the training loop, so a resume where `start_epoch > epochs` returns a clean `completed` instead of `UnboundLocalError` | bug fix (user) | accepted |
| D96 | `patience`/`min_delta` no longer use `or` fallbacks: `patience=0` disables early stopping, `None` → default 3. Help text added to the CNN hparam schema so the Configure form explains both | bug fix (user) | accepted |
| D97 | `make dev` starts backend (:8080, `--reload`) + frontend (:5173) in one command with `trap 'kill 0'` so Ctrl+C stops both. README "Local development" rewritten with exact first-time + run commands | user instruction | accepted |
| D98 | Evaluation report: CNN writes a confusion matrix + per-class precision/recall/F1 + accuracy to `artifacts/evaluation.json`; RF writes basic accuracy/rmse. Served via `GET /runs/{id}/evaluation` and shown on the run page "Evaluation" tab | doc 05 §5, doc 12 S12 | accepted |
| D99 | **DEFERRED to Phase 2** (out of MVP): export (ONNX/TorchScript), model registry, run comparison, Optuna hyperparameter tuning, notifications. Documented here so scope stays honest | doc 12 Phase 2 | accepted |
| D100 | MVP acceptance criterion 3 (checkpoint downloaded once) is verified at the **resolver** level (lock/sha/index tests), but no trainer consumes a real pretrained checkpoint yet — the two MVP trainers are from-scratch by design (D70, doc 14 §3.4). Honest status: resolver proven, end-to-end pretrained download deferred to the GPU/D1 milestone | PRD §10.3 | accepted |
