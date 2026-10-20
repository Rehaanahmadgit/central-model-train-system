# 05 — API Specification

Base path: `/api/v1`. JSON everywhere. **No user accounts.** If `TRAINHUB_ACCESS_TOKEN` is set, every request must send header `X-Access-Token: <token>` (WebSocket: `?token=`); the token identifies nobody. Lists are paginated: `?page=1&page_size=20` → `{items, total, page, page_size}`.

Error format (always):
```json
{"error": {"code": "VERIFICATION_FAILED", "message": "Human readable", "details": {}}}
```

## 1. Projects and datasets
| Method | Path | Notes |
|---|---|---|
| GET/POST | /projects | project = named folder |
| GET/PATCH/DELETE | /projects/{id} | |
| POST | /uploads | init chunked upload → `{upload_id, chunk_size}` |
| PUT | /uploads/{upload_id}/chunks/{n} | one chunk, idempotent, checksum header |
| POST | /uploads/{upload_id}/complete | extract safely, compute hash → `{dataset_hash, ...}` (dedup if exists) |
| GET | /datasets | list (filters: project) |
| GET | /datasets/{hash} | meta: size, files, format_detected, verification status summary |
| DELETE | /datasets/{hash} | removes files, writes audit line (409 if used by running run) |

## 2. Models / plugins
| Method | Path | Notes |
|---|---|---|
| GET | /models | trainer plugins: family, tasks, hyperparameter JSON Schema |
| GET | /models/{family} | detail + default config |

## 3. Verification
| Method | Path | Notes |
|---|---|---|
| POST | /verifications | `{dataset_hash, model_family, task_type?}` → 202 `{id, status}`; returns cached result immediately if cache hit (`cached: true`) |
| GET | /verifications/{id} | status + checks + derived summary |
| GET | /verifications/{id}/checks | list |
| WS | /ws/verifications/{id} | live check-by-check progress |

Completed response:
```json
{
  "id": "ver_...", "status": "passed_with_warnings", "cached": false,
  "task_type": "object_detection",
  "derived": {
    "num_classes": 5, "classes": ["a","b"], "class_counts": {"a": 1200},
    "num_images": 6400, "splits": {"train": 5000, "val": 1000, "test": 400},
    "image_stats": {"min_wh": [320,240], "max_wh": [4000,3000]},
    "recommended_checkpoints": [{"catalog_id": "rtdetr-l@1.0", "reason": "COCO-pretrained, detection, medium dataset", "local": true}],
    "recommended_hparams": {"imgsz": 640, "batch_size": 16}
  },
  "checks": [{"name":"corrupt_files","status":"pass"},{"name":"class_imbalance","status":"warn","message":"Class b is 3% of data"}],
  "can_train": true
}
```

## 4. Checkpoints
| Method | Path | Notes |
|---|---|---|
| GET | /checkpoints | catalog + local availability + disk usage |
| POST | /checkpoints/{id}/prefetch | download ahead of time |
| DELETE | /checkpoints/{id}/local | evict a cached file (409 if in use) |

(Catalog entries are edited in `catalog.yaml`; the API reloads it via `POST /checkpoints/reload`.)

## 5. Runs
| Method | Path | Notes |
|---|---|---|
| POST | /runs | `{verification_id, name, config, checkpoint_catalog_id?, from_scratch?}`. 409 if verification not passed/stale |
| GET | /runs | filters: project, state, model, date |
| GET | /runs/{id} | full `run.json` incl. env |
| POST | /runs/{id}/pause · resume · stop | |
| POST | /runs/{id}/retry | new run linked by `parent_run_id`; `resume_from_last: true` optional |
| GET | /runs/{id}/metrics?names=&from_step=&downsample= | time series (from `metrics.parquet` / events) |
| GET | /runs/{id}/logs?after_seq=&limit= | |
| GET | /runs/{id}/system-stats | |
| GET | /runs/{id}/events | |
| GET | /runs/{id}/artifacts | list; download via `GET /runs/{id}/artifacts/{name}` |
| GET | /runs/{id}/evaluation | |
| DELETE | /runs/{id} | removes run folder (not allowed while running) |
| POST | /runs/compare | `{run_ids:[...]}` |
| WS | /ws/runs/{id} | live stream |

## 6. Export and model registry (files)
| Method | Path | Notes |
|---|---|---|
| POST | /runs/{id}/export | `{format:"onnx"}` → job; output under `models/` or run artifacts |
| GET | /models-registry | list `models/*/*/model.json` |
| POST | /models-registry | register a run's weights as `{name, version}` |
| PATCH | /models-registry/{name}/{version}/stage | none/staging/production/archived |

## 7. System and audit
| Method | Path | Notes |
|---|---|---|
| GET | /system | running/queued jobs, GPU util/mem, disk usage, workspace size, recovery warnings |
| GET | /audit | filter by action/entity/date; reads `audit/*.jsonl` (no personal data exists to show) |
| GET | /healthz · /readyz | liveness / readiness (workspace writable, disk space) |
| GET | /metrics | Prometheus format |

## 8. WebSocket protocol
Connect `/ws/runs/{id}`. Server sends `snapshot` first, then deltas.

Envelope:
```json
{"type": "snapshot|metric|log|system|state|event|epoch_end|error", "run_id": "...", "ts": "ISO8601", "seq": 1234, "data": {}}
```
- `metric`: `{"name":"train/loss","value":0.42,"step":310,"epoch":3}`
- `log`: `{"level":"INFO","message":"Epoch 3/50 started"}`
- `system`: `{"gpu_util":87,"gpu_mem_mb":10230,"cpu":45,"ram_mb":8200}`
- `state`: `{"from":"preparing","to":"running"}`

Rules: `seq` strictly increasing per run (same value as in `events.jsonl`). After reconnect the client sends `{"type":"resume","last_seq":N}`; server replays from `events.jsonl` then continues live. Server throttles to ~10 msgs/s, batching metrics. Heartbeat ping every 20 s. If the run is finished, replay then close.

## 9. Idempotency and limits
- `POST /runs` supports `Idempotency-Key` header (keys kept in memory for 10 min and in `run.json`).
- Request size and upload limits from config; rate limiting per IP **in the reverse proxy only** (the app does not store IPs).
- State-changing endpoints append a (person-free) line to `audit/`.
