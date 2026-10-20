# TrainHub — Progress

Living checklist for the build order from doc 13 §5 (12 slices) + Phase 0 exit from doc 12.
Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

## Audit 2026-10-20 (read-only, doc 16 rules)
- `ruff check` ✓ · `mypy app` ✓ · 148 tests pass in 10.74 s; **1 test hangs and never completes** (`test_verification_ws_on_finished_stream_closes`, backend/tests/test_verifications_api.py:189).
- Frontend `npm run build` ✓ · `docker compose config` ✓ for both deploy files · image build **NOT RUN** (FAST-04).
- Doc drift: this file said S5 was "NEXT", but S5 code already exists (runner, 5 checks, API, WS, checklist UI).
- No `Makefile`/`scripts/verify.sh` yet → FAST-01 not met.

## Phase 0 — Foundations (doc 12)
- [~] M0 exit: repo, compose, CI, health, logging, fake metric and recovery tests done. Gap: the fake run is in-process asyncio, **not** a subprocess as doc 12 Phase 0 requires; real subprocess arrives with the Job Manager (S8). No live `kill -9` drill; recovery is covered by unit tests only.

## Slices (doc 13 §5)
- [x] S1 — Compose stack (api + frontend + volume), health endpoints, CI, config, structured logging. **DONE 2026-10-17.** ruff ✓ mypy ✓ 15 tests ✓ frontend build ✓ live smoke ✓ (healthz/readyz/SPA/404 envelope). `docker compose config` ✓ 2026-10-20 (both files); image build NOT RUN (FAST-04).
- [x] S2 — Storage module: atomic writes, JSONL appender, locks, IDs, startup recovery skeleton, in-memory index (+ heavy crash-safety tests, doc 11 §3b). **DONE 2026-10-17.** ruff ✓ mypy ✓ 60 tests ✓. Recovery wired into app startup lifespan [doc 03 §2.2].
- [x] S3 — Projects + chunked upload + safe extraction + dataset hash/dedup + delete. **DONE 2026-10-17.** ruff ✓ mypy ✓ 104 tests ✓ live smoke ✓ (project → upload → complete → list → delete).
- [x] S4 — Event broadcaster + `events.jsonl` + WebSocket (snapshot, deltas, `last_seq` resume) + fake publisher + live chart. **DONE 2026-10-17.** ruff ✓ mypy ✓ 126 tests ✓ frontend build ✓ live WS smoke ✓ (snapshot → replay → live metrics, monotonic seq). Phase 0 M0 demo: `TRAINHUB_DEMO_FAKE_RUN=true` streams hello-world metrics.
- [~] S5 — Verification runner + 5 core checks + API + WS + streaming checklist UI. Code exists (`app/verification/*`, `app/api/verifications.py`, `frontend/src/pages/VerifyDemo.tsx`); 7/8 tests pass, 1 hangs; no audit line for verification start/finish [doc 10 §6].
- [ ] S5.1 — `make verify` + FAST-02 test tiers/markers + fix the hanging WS close + verification audit lines. **NEXT**
- [ ] S6 — Verification cache + normalized manifest + remaining checks + task detection + recommendations.

- [ ] S7 — Checkpoint catalog + resolver + selection v1.
- [ ] S8 — Plugin base + first trainer plugin + Job Manager + subprocess supervision + telemetry + GPU slot locks.
- [ ] S9 — Live Run page (charts, logs, system stats, info panel).
- [ ] S10 — New Training Wizard end to end.
- [ ] S11 — Pause/resume/stop + periodic checkpoints + `interrupted` recovery + failure classification.
- [ ] S12 — Evaluation report + MVP acceptance (PRD §10) incl. privacy grep test.

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
- [ ] FAST-01 one `make verify` — **not created**; README lists three manual commands.
- [ ] FAST-02 `test-fast` / `test-full` / `pytest -m gpu` — no Makefile targets, no `markers` in pyproject.
- [x] FAST-03 one verify + one targeted run per slice — followed in this audit (one lint/type/test pass, targeted reruns only to diagnose the hang).
- [x] FAST-04 no Docker builds mid-slice — image build NOT RUN; only `compose config` was run.
- [x] FAST-05/06 file and reply discipline — state files updated, no spec index repeated.

## MVP acceptance (doc 01 §10) — all open
1. [ ] Upload → model → verification → pass/fail details.
2. [ ] Pass → Start Training → job starts.
3. [ ] Checkpoint downloaded once, reused (log + `index.json`).
4. [ ] Live loss/accuracy + logs; refresh restores state.
5. [ ] `run.json` complete, no personal data.
6. [ ] Killed trainer subprocess → `failed` + reason; API up.
7. [ ] Restart mid-training → `interrupted`, resume from `last.ckpt`.
8. [ ] Privacy grep: zero emails/IPs/user-agents in workspace + logs.

## Non-goal guards (checked every turn)
- No database introduced. No client/user info introduced.
