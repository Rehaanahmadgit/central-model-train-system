# Architecture (as built)

Long spec: `doc/03_ARCHITECTURE.md`, `doc/04_STORAGE_LAYOUT.md`. This file lists what exists in code.

- **Backend** `backend/app` (FastAPI, py3.12): `api/` routers (projects, datasets, uploads, verifications, runs, models, checkpoints, licences, system, health), `services/`, `storage/` (atomic writes, JSONL, locks, recovery), `realtime/` (broadcaster + WS with `seq` resume), `core/` (config, auth token middleware, device, doctor).
- **Verification** `verification/`: runner + `checks/*` (24 checks, `VERIFIER_VERSION` 1.2.0 = cache key), `manifest.py` (parquet), `recommend.py` (hparam + checkpoint recommendation).
- **Checkpoints** `checkpoints/`: `default_catalog.yaml` (+ override in volume), `catalog.py`, `licences.py` (acceptance record `<home>/licences/accepted.json`), `resolver.py` (licence gate → FileLock → disk preflight → `.part` resume → sha256 → atomic move → `index.json`).
- **Training** `training/`: `job_manager.py` spawns `worker_main` subprocess per run (own process group, heartbeat); `engine.py` shared `TrainEngine`/`EngineHooks`; plugins `rtdetr.py`, `cnn.py`, `random_forest.py`; registry row per plugin.
- **Frontend** `frontend/src` (React + TS + Vite + Tailwind): Wizard (Dataset, Model, Verify, Configure, Train), LiveRun, CheckpointsPage, `LicenceCard`, `CheckList`.
- **Storage**: files only under `TRAINHUB_HOME` (`/data` in the image): datasets, runs, checkpoints/store + index.json, licences/accepted.json, cache/verif.
- **Tooling**: `make verify` (ruff + mypy + fast pytest + frontend build); Playwright e2e in `frontend/e2e`.
