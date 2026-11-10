# PRD (living summary)

Source of truth for the long spec is `doc/01_PRD.md`; this file tracks **status per requirement**.
Status values: `verified` (tests + user ran it) · `tested` (automated tests pass, user has not run it) · `code_only` (exists, no test evidence) · `planned` · `blocked`.

Product goal: a public Docker image where a user uploads a dataset, picks a model family/variant, accepts the weights licence, verifies, gets the best pretrained checkpoint selected and downloaded (sha256-checked), trains, evaluates, views predictions and downloads weights, with clear actionable errors.
Hard constraints: no database (files under `TRAINHUB_HOME`), no user information/telemetry.

| ID | Requirement (user story → acceptance) | Status | Evidence |
|---|---|---|---|
| FR-01 | Upload a dataset (chunked, zip) → dataset hash, dedup | verified | S3 tests; user upload (this chat) |
| FR-02 | Verify dataset → streaming checklist with message + fix hint, `can_train` | verified | 24 checks; Slice B tests; user verify (this chat) |
| FR-03 | Verification honesty: zero-box detection data blocks training; full corrupt scan; tiny/out-of-range box warning | tested | Slice B (`ef169eb`), `test_verification_checks.py`, Playwright |
| FR-04 | Pick model family; variants shown with licence card; accept before use | tested | Slice C (`718761b`), `test_licences.py`, `licence-logic.spec.ts` |
| FR-05 | Server-side licence gate: resolver returns 409 `LICENCE_NOT_ACCEPTED` | tested | `test_licences.py` |
| FR-06 | Checkpoint download once, sha256 verified, FileLock, `.part` resume, `weights_only=True` load | tested | S7 resolver tests |
| FR-07 | Best-checkpoint auto-select with reason; hard filters: task, family, class head, VRAM ≤ GPU memory (min_vram 0 ⇒ warning), licence acceptable, verified sha256 (TODO(user) never auto-selected); user can override | tested | Slice D, `test_verification_cache.py`, `wizard.spec.ts`, `pretrained-logic.spec.ts` |
| FR-08 | Auto-download on run start (resolver, FileLock, `.part` resume), run shows "waiting: checkpoint" + progress; failures (offline, sha mismatch without retry loop, disk full size×1.5, partial, concurrent) give a clear message and "Import from file"; unaccepted licence = 409 at run create | tested | Slice D, `test_checkpoints.py` (fake resolver + real local HTTP server); UI progress/Import button NOT RUN against a real download |
| FR-19 | Frozen backbone without pretrained weights is blocked (API 400 `FROZEN_BACKBONE_NEEDS_PRETRAINED`, Configure Next and Start disabled with a red message) | tested | Slice D, `test_checkpoints.py`, `wizard.spec.ts` |
| FR-09 | RT-DETR r18vd training on GPU, val/test evaluation, best checkpoint | verified | user run: apple-orange val mAP50 0.873 / test 0.825 |
| FR-10 | Predictions tab (GT vs predictions, worst samples, confidence filter) | verified | user ran it; `predictions-logic.spec.ts` |
| FR-11 | Weights download (`weights.pth`) | verified | user ran it |
| FR-12 | Run again / Edit & run with prefilled config | verified | user ran Run again |
| FR-13 | Numeric config saved as numbers (Configure, Edit & run, Run again — server also coerces old string values); Predictions default confidence 0.5; header "Best" falls back to evaluation; GPU % n/a on a finished run | tested | commit `a977d7e` + Slice F: `run-summary-logic.spec.ts`, `test_training.py::test_run_again_coerces_string_config_from_an_old_run` |
| FR-14 | Shared TrainEngine (pause/stop/NaN guard/exact resume) | tested | A2 (`7c4dbef`), `test_engine.py`, golden test |
| FR-15 | Access token ON by default (generated on first boot into `<home>/access-token` 0600, printed once in logs); `/api` + WS 401 without it; `--reset-token`; explicit loud opt-out `TRAINHUB_AUTH_DISABLED`; UI token prompt with clear 401 banner | tested | Slice E, `test_auth.py`, `token-prompt.spec.ts` (mocked + real local instance) |
| FR-20 | Upload size cap with clear 413 (declared size and per-chunk); checkpoint `source_uri` https-only, non-public addresses blocked, redirects capped + re-checked | tested | Slice E, `test_uploads_api.py`, `test_checkpoints.py`; DNS-rebinding race VERIFY (sha256 still rejects wrong content) |
| FR-21 | Catalog override merges field by field (old `license`/TODO values ignored with a warning) | tested | `test_checkpoints.py`, commit `561eed7` |
| FR-16 | Configure device auto: GPU → full preset + GPU name/VRAM shown; no GPU → CPU + clear warning banner (`--gpus all` hint for GPU-only models); `/system` reports the real GPU | tested | Slice F, `device-logic.spec.ts`, `wizard.spec.ts`, `test_system.py`; banner on a real GPU host NOT RUN (venv torch is the CPU build) |
| FR-17 | CNN + RandomForest plugins (image classification, tabular) | tested | S8, `make verify` |
| FR-18 | Public image published (cpu/gpu tags) | blocked | registry namespace `TODO(user)`; container GPU run blocked on host (no NVIDIA Container Toolkit) |
