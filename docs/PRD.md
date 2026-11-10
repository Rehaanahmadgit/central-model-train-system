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
| FR-07 | Best-checkpoint auto-select with reason (hard filters) | planned | Slice D |
| FR-08 | Auto-download on run start with waiting state, clear failures, "Import from file" | planned | Slice D |
| FR-09 | RT-DETR r18vd training on GPU, val/test evaluation, best checkpoint | verified | user run: apple-orange val mAP50 0.873 / test 0.825 |
| FR-10 | Predictions tab (GT vs predictions, worst samples, confidence filter) | verified | user ran it; `predictions-logic.spec.ts` |
| FR-11 | Weights download (`weights.pth`) | verified | user ran it |
| FR-12 | Run again / Edit & run with prefilled config | verified | user ran Run again |
| FR-13 | Numeric config saved as numbers; Predictions default confidence 0.5; header "Best"; GPU % n/a on finished run | tested | commit `a977d7e`, `run-summary-logic.spec.ts` (Slice F re-checks) |
| FR-14 | Shared TrainEngine (pause/stop/NaN guard/exact resume) | tested | A2 (`7c4dbef`), `test_engine.py`, golden test |
| FR-15 | Access token ON by default, 401 without token, upload size cap, https-only `source_uri` | planned | Slice E (token middleware exists, default is OFF) |
| FR-16 | Configure device auto: GPU → full preset; no GPU → CPU + warning banner | planned | Slice F |
| FR-17 | CNN + RandomForest plugins (image classification, tabular) | tested | S8, `make verify` |
| FR-18 | Public image published (cpu/gpu tags) | blocked | registry namespace `TODO(user)`; container GPU run blocked on host (no NVIDIA Container Toolkit) |
