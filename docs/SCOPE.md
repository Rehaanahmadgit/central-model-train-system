# Scope

Status values: verified · tested · code_only · planned · blocked.

## In scope
| Item | Status |
|---|---|
| Dataset upload, verification (24 checks), manifest cache | verified |
| RT-DETR (object detection, YOLO + COCO input), CNN (image folder), RandomForest (tabular) | RT-DETR verified; CNN/RF tested |
| Model zoo catalog with licence card + local acceptance | tested |
| Checkpoint resolver (sha256, lock, resume, import offline) | tested |
| Live run page, pause/resume/stop, exact resume | tested |
| Evaluation, Predictions tab, weights export | verified |
| Auto-select + auto-download + failure handling + Import from file (Slice D) | tested |
| Token default ON, upload cap, SSRF guard (Slice E) | tested |
| Device auto + config number types + header fixes (Slice F) | planned |

## Out / deferred
| Item | Status |
|---|---|
| Database, accounts, telemetry, user info | out (hard constraint) |
| Compare page, ONNX/TorchScript export, model registry, Optuna, notifications | deferred (D99) |
| Multi-GPU / multi-node | deferred |
| rtdetr-l, resnet50 weights | blocked: URL/sha256/licence `TODO(user)` |
| Published image | blocked: registry `TODO(user)` |
