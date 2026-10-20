# 12 — Roadmap

Principle: ship a thin vertical slice first, then widen. Each phase ends with a demo and acceptance checks.

## Phase 0 — Foundations (Week 1–2)
- Repo/monorepo, Docker Compose (api, frontend/nginx, one data volume), CI, linting, config, logging.
- Dockerfile with `cpu` and `gpu` targets, `docker-compose.dev.yml`, device abstraction, `trainhub doctor` (doc 14).
- Storage module (atomic writes, JSONL appender, locks, startup recovery), projects, person-free audit log. No database.
- Plugin base classes and registry skeleton; in-process event broadcaster + `events.jsonl`; WebSocket skeleton with `seq` + resume.
- **Exit:** create project, hello-world job runs in a subprocess and streams a fake metric to a live chart; `kill -9` recovery works.

## Phase 1 — MVP vertical slice (Week 3–8)
- Chunked dataset upload, dataset versions with content hash.
- Verification pipeline v1: format detection + structural + content + basic stats for **one task family** (choose: image classification or detection) + tabular basic.
- Verification cache + normalized manifest.
- Checkpoint catalog + resolver (download-once, sha256, lock, atomic rename) + rule-based selection.
- Trainer plugins: 1 deep learning plugin + 1 scikit-learn plugin.
- Run lifecycle, subprocess training, per-epoch checkpoint, stop.
- Live Run page: loss/metric charts, logs, system stats, run info panel.
- New Training Wizard (5 steps).
- **Exit (MVP acceptance):** criteria in PRD §10 pass on a real dataset.

## Phase 2 — Reliability and insight (Week 9–13)
- First GPU-server deployment from the pushed image; prefetch/import of checkpoints; `pytest -m gpu` smoke run (doc 14).
- Pause/resume, exact resume from checkpoint, early stopping, NaN guard, heartbeat/worker_lost handling.
- Evaluation report (confusion matrix, per-class, curves), error-analysis gallery, model card.
- Runs list, filters, **Compare** page.
- Export (ONNX/TorchScript) + Model Registry with stages.
- Notifications (UI + operator-configured webhook/Slack; no personal emails).
- Checkpoint management UI, prefetch, LRU eviction.
- Prometheus/Grafana + alerts, structured logging, Sentry.
- **Exit:** chaos drills pass; two engineers use it for real work for a week.

## Phase 3 — Smart and scalable (Week 14–20)
- More task types (segmentation, more detection/classification models, more tabular).
- Hyperparameter tuning with Optuna + trial dashboard.
- OOM auto-recovery, data-aware recommended hyperparameters v2 (learn from history).
- Checkpoint selection v2 (historical performance).
- GPU-hour tracking per run (no per-person quotas), GPU scheduler (queue priority).
- Dataset explorer with label overlays; near-duplicate detection.
- Optional shared access token, SDK/CLI (`trainhub run --dataset ... --model ...`).
- **Exit:** 20+ runs/week stable; no file corruption in chaos drills.

## Phase 4 — Scale and polish (Week 21+)
- Multi-GPU (DDP) on one node. Multi-node only if truly needed (shared filesystem + leader election; still no database).
- Data versioning integrations (DVC), MLflow bridge, W&B-style report sharing.
- Drift/monitoring hooks, model serving handoff (BentoML/Triton export).
- Audit hash-chaining (tamper evidence).
- Public plugin SDK + docs.

## Milestone summary
| Milestone | Target | Demo |
|---|---|---|
| M0 Foundations | W2 | fake live chart |
| M1 Verify | W5 | dataset → verification report in UI |
| M2 Train MVP | W8 | upload → verify → train → live dashboard |
| M3 Reliable | W13 | resume, compare, export, alerts |
| M4 Smart | W20 | tuning, scheduler, more tasks |

## Risks to watch per phase
P1: scope creep on model types → keep to 2 plugins. P2: live-stream scaling → throttle + downsample early. P3: GPU scheduling complexity → start with simple FIFO + one-job-per-GPU.

## Backlog / ideas
Active-learning hints (show mislabeled-looking samples), auto data augmentation suggestions, synthetic data hooks, cost estimator before training ("this run ≈ 40 min on RTX…"), template library (defect detection, face recognition, tabular churn), Slack bot to start/monitor runs, public/private run sharing links.
