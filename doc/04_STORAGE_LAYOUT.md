# 04 — Storage Layout (No Database)

There is **no database**. All persistent state is files under one workspace directory, `TRAINHUB_HOME` (default `/data/trainhub`, a Docker volume). No file contains personal or client information.

## 1. Directory layout
```
$TRAINHUB_HOME/
  projects/
    {project_id}/
      project.json                      # {id, name, description, created_at}
  datasets/
    {dataset_hash}/                     # content-addressed, immutable
      meta.json                         # {hash, name, size_bytes, file_count, format_detected, created_at, source_filename}
      raw/                              # extracted dataset files
      manifest.parquet                  # normalized manifest (path, label/boxes, split) - created by verification
  verifications/
    {verification_id}/
      verification.json                 # {id, dataset_hash, model_family, task_type, verifier_version, status, started_at, finished_at}
      checks.jsonl                      # one line per check result (append as they finish)
      report.json                       # derived facts + summary (see doc 06 §6)
  cache/
    verif/{cache_key}.json              # pointer {verification_id, created_at}; cache_key = sha256(dataset_hash|model_family|verifier_version)
  checkpoints/
    catalog.yaml                        # what pretrained weights exist (hand-maintained, versioned in git)
    index.json                          # what is cached locally {catalog_id: {path, sha256, size, status, last_used_at}}
    store/{sha256[:2]}/{sha256}/model.ext   # content-addressed weight files
    .tmp/                               # in-progress downloads (*.part)
    locks/                              # file locks
  runs/
    {run_id}/
      run.json                          # the run record (state, config, env, timestamps, checkpoint used, best metric)
      events.jsonl                      # append-only event stream: metrics, logs, system stats, state changes (each line has seq)
      events.idx                        # optional: seq -> byte offset index for fast resume
      metrics.parquet                   # compacted metrics (written at end / periodically) for fast charting
      checkpoints/                      # last.ckpt, best.ckpt, epoch_N.ckpt
      artifacts/                        # weights, plots, evaluation.json, error samples, model_card.md, exports/
      stdout.log                        # raw subprocess output
  models/                               # model registry (just folders)
    {model_name}/{version}/
      model.onnx|pt|...                 # exported file
      model.json                        # {name, version, source_run_id, stage, export_formats, notes, created_at}
  audit/
    audit-YYYY-MM.jsonl                 # append-only system audit trail (no personal data)
  locks/
    gpu-{idx}.lock                      # GPU slot locks
    run-{run_id}.lock                   # per-run writer lock
  tmp/
    uploads/{upload_id}/                # chunk staging
```

## 2. File formats
### run.json
```json
{
  "id": "run_20261017_113400_ab12",
  "project_id": "...", "name": "defect-rtdetr-v1",
  "dataset_hash": "sha256...", "verification_id": "...",
  "model_family": "rtdetr", "task_type": "object_detection",
  "checkpoint": {"catalog_id": "rtdetr-l@1.0", "sha256": "...", "source": "cache|download|none"},
  "parent_run_id": null,
  "config": {},                          // full resolved hyperparameters
  "seed": 42,
  "env": {"python": "...", "torch": "...", "cuda": "...", "git_commit": "...", "gpu": "...", "hostname": "..."},
  "state": "running",
  "failure": {"reason": null, "details": null},
  "queued_at": "...", "started_at": "...", "finished_at": null,
  "best": {"metric": "val/mAP50", "value": 0.0, "epoch": 0},
  "schema_version": 1
}
```
### events.jsonl (one JSON per line)
`{"seq":1234,"ts":"ISO8601","type":"metric|log|system|state|event|epoch_end|error","data":{...}}`
`seq` is strictly increasing per run and is the basis for WebSocket resume (doc 05 §8).

### audit-YYYY-MM.jsonl
`{"ts":"...","action":"dataset.uploaded|verification.started|run.started|run.stopped|checkpoint.downloaded|model.exported|dataset.deleted","entity_type":"run","entity_id":"...","details":{}}`
**Forbidden fields:** user name, email, IP address, user agent, device id, any client identifier.

## 3. Rules for safe file-based persistence
1. **Atomic writes** for JSON/YAML/Parquet: write to `name.tmp` in the same directory, `fsync`, then `os.replace`. Never edit in place.
2. **Append-only** for `events.jsonl`, `checks.jsonl`, `audit-*.jsonl`: open with append mode, one write per line, flush on a short timer (~1 s) or N lines; fsync at state changes and on shutdown.
3. **Single writer per run:** only the Job Manager process writes a run's files (workers send events through a pipe). `run.json` updates take `locks/run-{id}.lock`.
4. **Cross-process locks** via `fcntl.flock` (or `portalocker`); locks are released automatically if the process dies.
5. **Content addressing** for datasets and checkpoint files, so duplicates are impossible and integrity is checkable.
6. **Crash recovery on startup:** scan `runs/*/run.json`; any run in `preparing|running|evaluating` → mark `interrupted` (offer Resume from `last.ckpt`); `queued` runs are re-queued in order of `queued_at`; delete `.tmp`/`.part` leftovers older than 1 hour; truncate a half-written last line in JSONL files.
7. **Schema versioning:** every JSON file has `schema_version`; loaders migrate old versions in code.
8. **Listing performance:** at startup and on change, build an in-memory index of projects/runs/datasets from `run.json`/`meta.json` files. It is a cache, rebuildable any time, never a source of truth. Fine for thousands of runs; do not add a database to scale it — paginate and cache instead.
9. **Retention:** configurable cleanup of old `events.jsonl` detail (keep compacted `metrics.parquet`), old non-best checkpoints, orphaned uploads, and old datasets (explicit delete only).
10. **Backups:** the whole workspace is one directory — back it up with filesystem snapshots/rsync; `datasets/*/raw` can be excluded if re-uploadable.

## 4. IDs
Run IDs: `run_{YYYYMMDD}_{HHMMSS}_{4 random hex}`. Verification IDs: `ver_...`. Upload IDs: random UUID4. Dataset ID = content hash (sha256 of a sorted manifest of file hashes). No sequential numeric IDs, no user-derived IDs.

## 5. Privacy by construction
- The server never asks for or stores a name, email, or identity.
- Logs and audit files contain no IP addresses or user agents (configure the reverse proxy not to log them to disk, or to anonymize).
- Uploaded datasets are the user's content: stored only in the workspace, deletable via API (`DELETE /datasets/{hash}`), which removes the files and appends a `dataset.deleted` audit line.
