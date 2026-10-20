# 08 — Training Engine, Plugins and Real-time Telemetry

## 1. Trainer plugin interface
```python
class TrainerPlugin(ABC):
    family: str                      # "rtdetr", "resnet", "random_forest", ...
    supported_tasks: list[str]
    supported_formats: list[str]     # normalized manifest is preferred
    hparam_schema: dict              # JSON Schema -> UI form + validation
    default_hparams: dict

    def validate_config(self, config, facts) -> list[Issue]: ...
    def prepare(self, ctx: RunContext) -> None:
        """Load manifest, build datasets/loaders, build model, load checkpoint, adapt head."""
    def train(self, ctx: RunContext) -> TrainResult:
        """Main loop. Must call ctx.emit_* and honour ctx.should_pause()/should_stop()."""
    def evaluate(self, ctx: RunContext) -> EvalResult: ...
    def export(self, ctx: RunContext, fmt: str) -> ExportResult: ...
    def resume(self, ctx: RunContext, checkpoint_path) -> None: ...
```
`RunContext` provides: config, verification facts, manifest path, checkpoint path, output dir, seed, device info, and emitters:
- `ctx.emit_metric(name, value, step, epoch, split)`
- `ctx.emit_log(level, message)`
- `ctx.emit_event(type, payload)`
- `ctx.save_checkpoint(path, is_best, metrics)`
- `ctx.should_pause()`, `ctx.should_stop()` (check at each batch/epoch boundary)

Plugins are discovered via entry points / a registry module. Adding a model = new file in `training/plugins/` + register. No changes to API or UI (UI form is generated from `hparam_schema`).

## 2. Initial plugins (by phase)
- Phase 1: one detection plugin (RT-DETR/YOLO-style) OR one classification plugin (ResNet/EfficientNet) — pick the one the author knows best — plus a scikit-learn plugin (RandomForest, XGBoost) to prove the plugin model works for classical ML.
- Phase 2+: segmentation, more tabular models, transformers.

## 3. Job lifecycle
1. API writes `run.json` (`queued`) and hands the run to the in-process Job Manager queue.
2. Job Manager claims it (GPU slot lock: `locks/gpu-{idx}.lock`), sets `preparing`, records env (python/torch/cuda versions, git commit, GPU name).
3. Resolves checkpoint (doc 07), loads facts from the cached `report.json` and `manifest.parquet` (no recomputation).
4. Spawns **training subprocess** (own process group). Parent monitors heartbeat, forwards events, enforces timeouts.
5. Subprocess runs `prepare → train`; sends events to parent via a local queue/pipe; parent appends to `events.jsonl` (flush ~1 s, fsync on state changes) and publishes to the in-memory broadcaster for WebSockets.
6. Control signals (pause/resume/stop) are sent to the subprocess over the parent–child pipe; the subprocess checks them at batch boundaries.
7. After training: `evaluate`, upload artifacts, state `completed`.
8. On exception: capture traceback, classify failure (`OOM`, `DATA_ERROR`, `NAN_LOSS`, `WORKER_LOST`, `UNKNOWN`), save, state `failed`. A user-requested stop is NOT a failure: it ends in state `stopped` with `failure` null.
9. Always release GPU slot and clean temp files in `finally`.

## 4. Checkpointing during training
- Save `last.ckpt` every epoch (atomic write), `best.ckpt` on metric improvement, plus optional every-N epochs.
- Checkpoint contains: model, optimizer, scheduler, scaler, epoch, step, RNG states, config hash → exact resume.
- Retention: keep best + last K.

## 5. Reliability features
- **Early stopping** (patience, min_delta, monitored metric).
- **NaN/Inf guard**: stop run and report with last good checkpoint.
- **OOM recovery** (P2): catch CUDA OOM → halve batch size (use gradient accumulation to keep effective batch) → retry once or twice, log event.
- **Mixed precision** toggle; gradient clipping.
- **Determinism**: set seeds; document that full determinism may cost speed.
- **Heartbeat**: subprocess sends a heartbeat message over the pipe every 5 s; missing for 60 s → parent kills and marks `failed(worker_lost)`.
- **Timeouts**: max run duration per org policy.
- **Disk guard**: stop with clear error if disk < threshold.

## 6. Telemetry (what the dashboard shows)
Emitted metrics (names are conventions):
- `train/loss`, `val/loss`, per-task metrics: `val/accuracy`, `val/f1`, `val/mAP50`, `val/mAP50-95`, `val/precision`, `val/recall`
- `lr`, `epoch_time_s`, `samples_per_sec`, `grad_norm`
- System sampler (every 2 s): GPU util, GPU memory, GPU temp, CPU %, RAM, disk I/O (use pynvml, psutil).
- Events: `epoch_start/end`, `checkpoint_saved`, `lr_changed`, `early_stop_triggered`, `paused`, `resumed`, `oom_retry`.
- Static info panel: full resolved hyperparameters, dataset hash/version, checkpoint name+sha, environment, seed, start time, ETA (computed from rolling epoch time).

## 7. Pipeline for data → trainer
Trainer reads the **normalized manifest** and `recommended_hparams` produced by verification. User-config overrides recommended values. Final resolved config is stored in `runs.config` (what actually ran).

## 8. Evaluation outputs (after training)
- Metrics summary, confusion matrix, per-class precision/recall/F1, PR and ROC curves.
- Error analysis: top-N worst predictions (lowest confidence on correct class / highest-confidence mistakes) with image thumbnails saved as artifacts.
- Detection: per-class AP, size-bucket AP, example predictions overlay.
- Tabular: feature importance, residual plots (regression).
- Model card (markdown): data, config, metrics, limitations, environment.

## 9. Hyperparameter tuning (Phase 3)
Optuna study per tuning job; each trial = child run (short budget, pruning enabled). Dashboard shows trial table + parallel-coordinates. Best params can be promoted to a full run.

## 10. Security of training execution
- Run training subprocess as non-root, with resource limits (cgroups/container limits), no outbound network except allowlisted (checkpoint sources handled by the resolver, not the trainer).
- User-supplied custom code is **out of scope for v1**; plugins are trusted code from the repo only.
