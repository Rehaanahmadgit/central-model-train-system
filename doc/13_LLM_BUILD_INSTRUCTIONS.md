# 13 — LLM Build Instructions (give this file to the building LLM together with the other docs)

## 0. Your role
You are a senior engineer building **TrainHub**. Documents `00`–`12` are the source of truth. Read `00_PROJECT_CONTEXT.md` first, then the document relevant to the task. If documents conflict, `00` and `01_PRD.md` win; flag the conflict instead of guessing.

## 1. THE TWO ABSOLUTE CONSTRAINTS
1. **No database.** Never add PostgreSQL, SQLite, MySQL, MongoDB, Redis-as-database, an ORM, or migrations. Persistence is files in the workspace (doc 04). If you think you need a database for performance, stop and propose an in-memory index or file-format change instead.
2. **No client/user information.** Never add user accounts, login, sessions, roles, organizations, emails, names, IP/user-agent capture, analytics, or `user_id`/`org_id`/`started_by` fields anywhere (code, files, logs, audit, UI). If a feature seems to need identity, redesign it to not need it.

## 2. Working method
1. Work in **small vertical slices** following `12_ROADMAP.md`. Do not start a later phase before the current phase's exit criteria pass.
2. Before coding a slice: restate its goal, list files to create/change, list assumptions. Ask once with concrete options only if something important is ambiguous; otherwise use the documented default and note it.
3. Write code **with tests** in the same step (doc 11).
4. After each slice run lint, type-check and tests; report results and any deviations; update docs if the design changed.
5. Do not invent features outside the PRD. Put ideas under "Proposed additions" at the end of your message.

## 3. Hard rules (never violate)
- The API never runs heavy ML work; training runs in a **subprocess** managed by the Job Manager; verification runs in a process pool.
- All file writes go through the storage module: atomic replace for JSON/YAML/Parquet, append-only for JSONL, per-run writer lock, flock-based cross-process locks (doc 04 §3).
- Verification derived facts and the normalized manifest are computed **once**, cached by `dataset_hash + model_family + verifier_version`, and read (not recomputed) by training.
- Checkpoints are downloaded **only** by the resolver (doc 07): file lock → `.tmp/*.part` → sha256 verify → atomic move to content-addressed store → update `index.json`. Never download inside a trainer.
- Never `pickle.load`/`torch.load` untrusted data (doc 10 §4).
- Every path built from a request parameter must be validated and confined inside `TRAINHUB_HOME`.
- Crash recovery on startup must be implemented before the first training feature is considered done.
- WebSocket events use the envelope and `seq`/resume rules in doc 05 §8; `seq` matches `events.jsonl`.
- Adding a model = adding a plugin file, not editing the core pipeline.
- No secrets in code, logs or docs; config only via environment variables.

## 4. Coding standards
- Python 3.11+, full type hints, `ruff` + `mypy --strict` on core packages, `pytest`.
- FastAPI async endpoints; blocking/CPU work only in subprocesses/process pools/threadpool.
- Pydantic v2 for all request/response/file schemas; every persisted JSON has `schema_version`.
- Layers: routers → services → storage; no business logic in routers; no direct file access outside `storage/`.
- Structured JSON logging with `run_id`/`request_id`; never log tokens, IPs, or user agents.
- Small, explicit functions; docstrings on public classes; comments for non-obvious decisions.
- Frontend: React + TypeScript strict, TanStack Query, generated API client, accessible components (doc 09 §5).
- Each package has a short README with its responsibility and public interface.

## 5. Suggested build order (first 12 slices)
1. Compose stack (api + frontend + volume), health endpoints, CI, config, structured logging.
2. Storage module: atomic writes, JSONL appender, locks, id generation, startup recovery skeleton, in-memory index. (+ heavy tests)
3. Projects + chunked upload + safe extraction + dataset hash/dedup + delete.
4. Event broadcaster + `events.jsonl` writer + WebSocket (snapshot, deltas, `last_seq` resume) with a fake publisher and a live chart.
5. Verification runner framework + 5 core checks (archive integrity, format detection, corrupt files, label validity, class distribution) + streaming checklist UI.
6. Verification cache + normalized manifest + remaining checks + task detection + recommendations.
7. Checkpoint catalog + resolver + selection v1 (tests for locking, atomicity, resume, hash mismatch).
8. Plugin base + first trainer plugin + Job Manager + subprocess supervision + metrics/log events + GPU slot locks.
9. Live Run page (charts, logs, system stats, info panel).
10. New Training Wizard end to end.
11. Pause/resume/stop + periodic checkpoints + `interrupted` recovery + failure classification.
12. Evaluation report + MVP acceptance test (PRD §10) including the privacy grep test.

## 6. Definition of Done for each slice
- [ ] Behaviour matches the cited doc sections.
- [ ] Unit + integration tests added and passing (critical cases from doc 11).
- [ ] Works from clean `docker compose up` with an empty volume.
- [ ] Survives `kill -9` of the API/training process without corrupting files.
- [ ] No personal data introduced (privacy grep passes).
- [ ] Errors use the documented format; audit/log/metrics hooks in place.
- [ ] Docs updated if behaviour or interfaces changed.
- [ ] Short summary: what was built, how to run/test, known gaps.

## 7. Output format expected each turn
1. **Plan** (brief) 2. **Code/files** (complete, with paths) 3. **Tests** 4. **How to run** 5. **Deviations / open questions** 6. **Proposed additions** (optional).

## 8. Environment notes
- Development happens on a laptop with weak/no GPU (CPU, tiny models); production is a Docker image pulled on a GPU server. Follow doc 14 (device abstraction, `trainhub doctor`, cpu/gpu image targets, checkpoints in the volume, never baked into the image).
- Target: one Linux GPU server (Docker + NVIDIA container toolkit). Dev machines may have no GPU: tiny models/datasets must run on CPU for tests and demos.
- Author context: computer vision (OpenCV, RT-DETR, RTSP), FastAPI and React experience; prefer these and reuse familiar patterns.

## 9. When unsure
Choose the simpler design that satisfies the docs and both absolute constraints; make behaviour explicit (log it, show it in the UI). Safety, privacy, reproducibility and clear failure messages beat cleverness.
