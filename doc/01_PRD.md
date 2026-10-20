# 01 — Product Requirements Document (PRD)

**Product:** TrainHub — Central ML Training Platform
**Status:** Draft v2 (no database, no client/user data)
**Owner:** Rehaan

## 1. Problem statement
ML engineers repeatedly rebuild the same plumbing for every project: dataset checks, label format conversion, picking pretrained weights, writing training scripts, logging metrics, watching GPU usage, and tracking what was trained on what. Data mistakes (corrupt files, wrong labels, class imbalance, leakage) are found only after hours of wasted training. There is no single place that verifies data, chooses the right starting checkpoint, trains, and records everything in an auditable way.

## 2. Vision
One platform where an engineer uploads a dataset, picks a model, and gets: automatic verification, automatic pretrained checkpoint selection (no repeated downloads), one-click training, and a live audit-grade view of everything happening — with full reproducibility afterwards. It stores **no database and no personal/client information**.

## 3. Constraints (author decisions)
- C1: No database of any kind. State is files in one workspace directory (doc 04).
- C2: No accounts, logins, roles, organizations, emails, or any information about people or clients. Only datasets (user content) and technical run records are stored.
- C3: Single-node deployment first (one Linux server with GPU).

## 4. Target usage
Engineers on a trusted network open the web app and work directly — no sign-in. (Optional: a single shared access token from an environment variable gates the whole app; it identifies nobody.)

## 5. User stories
### Dataset and verification
- US-01: I can upload a dataset (zip, folder, CSV) and see upload progress (chunked, resumable).
- US-02: I select a model/task and click **Start Verification**.
- US-03: I see each verification check live (pass / warning / fail) with reasons and fix hints.
- US-04: If everything passes I see "Dataset verified successfully" and a **Next / Start Training** button.
- US-05: If verification fails, training cannot start; I see exactly what to fix.
- US-06: Re-uploading an identical dataset skips verification (hash-based cache).

### Checkpoints
- US-07: The system picks a suitable pretrained checkpoint automatically (task, dataset size, domain).
- US-08: I can override it, or choose "train from scratch".
- US-09: A checkpoint is downloaded only once; later runs use the local copy after hash validation.

### Training and monitoring
- US-10: I configure hyperparameters (smart defaults) and click **Start Training**.
- US-11: I see live loss/metric curves, logs, learning rate, GPU/CPU/RAM, ETA.
- US-12: I can pause, resume, and stop a run.
- US-13: Training auto-saves a checkpoint each epoch and keeps the best.
- US-14: If the browser closes or the server restarts, nothing is lost; reopening shows the full state and interrupted runs can resume.
- US-15: I get a UI notification (and optional webhook) on completion or failure.

### Evaluation, registry, audit
- US-16: After training I see confusion matrix, per-class metrics, PR/ROC curves, worst predictions.
- US-17: I can export the model (ONNX / TorchScript) and keep it in the model registry folder with a stage tag.
- US-18: I can compare runs side by side.
- US-19: Every run has an immutable technical record: dataset hash, config, checkpoint used, library/CUDA versions, git commit, hardware, timestamps, seed — and no personal data.
- US-20: I can delete a dataset or run completely from the UI.
- US-21: I can see system health: running/queued jobs, GPU usage, disk usage.

## 6. Functional requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Filesystem persistence with atomic writes, locks, crash recovery (doc 04) | P0 |
| FR-2 | Projects (folders) and content-addressed immutable dataset versions | P0 |
| FR-3 | Chunked/resumable dataset upload with size limits | P0 |
| FR-4 | Verification pipeline (doc 06) with disk cache keyed by dataset hash | P0 |
| FR-5 | Task type detection (classification, detection, segmentation, tabular) | P0 |
| FR-6 | Checkpoint catalog and smart selection, download-once cache (doc 07) | P0 |
| FR-7 | In-process Job Manager, subprocess training workers, lifecycle states (doc 08) | P0 |
| FR-8 | Real-time metrics, logs, system stats via WebSocket with replay | P0 |
| FR-9 | Pause / resume / stop; periodic checkpointing; resume after restart | P1 |
| FR-10 | Append-only system audit trail with no personal data | P0 |
| FR-11 | Evaluation report and error analysis | P1 |
| FR-12 | Model export and file-based model registry | P1 |
| FR-13 | Run comparison | P1 |
| FR-14 | Hyperparameter tuning (Optuna) | P2 |
| FR-15 | Notifications (UI, optional webhook) | P2 |
| FR-16 | Auto OOM recovery (reduce batch size and retry) | P2 |
| FR-17 | Multi-GPU (DDP) and GPU scheduling | P3 |
| FR-18 | Disk/retention management and cleanup policies | P1 |
| FR-19 | Trainer plugin SDK for adding new models | P1 |
| FR-20 | Optional single shared access token (env var) | P1 |

## 7. Non-functional requirements
- **Reliability:** a crashed training subprocess must not crash the API or lose run state; server restart recovers all runs.
- **Performance:** verification of 10k images < 2 min; metrics reach UI < 1 s.
- **Privacy:** no personal/client data stored or logged (doc 10).
- **Reproducibility:** every run can be re-created from its `run.json`.
- **Usability:** first training in under 10 minutes for a new engineer.
- **Portability:** `docker compose up` with one data volume.

## 8. Out of scope (v1)
- Any database, user management, SSO, billing, quotas per person.
- Multi-node clusters (design leaves room, not built).
- Annotation tool; model serving (only export + registry); LLM fine-tuning at scale.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Files as storage can corrupt on crash | Atomic writes, append-only logs, startup recovery (doc 04 §3) |
| Slow listing with many runs | In-memory index rebuilt from files, pagination |
| Too many model types | Phase plan; start with 1–2 plugins |
| Untrusted archives/pickles | Safe extraction, no untrusted pickle loads, sandboxed subprocess |
| Disk fills up | Preflight checks, retention policy, LRU checkpoint eviction, alerts |
| No auth on open network | Deploy on private network/VPN; optional shared token; reverse proxy |
| Wrong auto-checkpoint | Show reasoning, allow override, record choice |

## 10. Acceptance criteria for MVP
1. Upload a classification (or detection) dataset, pick a model, run verification, see pass/fail details.
2. On pass, Start Training appears; clicking it starts a job.
3. Checkpoint is downloaded once and reused on the second run (log + `index.json` prove it).
4. Live loss/accuracy graph and logs update; page refresh restores state.
5. `run.json` contains dataset hash, config, checkpoint, hardware, versions, timestamps — and no personal data.
6. Killing the training subprocess → run marked `failed` with reason; API stays up.
7. Restarting the server mid-training → run shows `interrupted` and can resume from `last.ckpt`.
8. A grep of the whole workspace and logs finds no emails, names, IPs, or user agents.
