# TrainHub — Open Questions

Status as of audit 2026-10-20.

## Resolved by the user
- Q1 design brief — **approved** (colour tweaks may come later).
- Q2 Tailwind — **v4** (4.3.3 pinned), VERIFY latest at scaffold (done).
- Q3 Phase 1 DL plugin — **detection (RT-DETR style)**.
- Q4 ambiguous-format UX — **resolve endpoint** (D14).
- Q5 chart library — **ECharts** (D8).
- Q6 WS paths — **literal `/ws/...`** (D9).
- U5–U15 — **recommended defaults accepted** (D11–D16). U15: generated synthetic sample dataset.
- A1–A7 — accepted.

## Open / awaiting action
- **O1 — docs on disk**: the replacements for `08_TRAINING_ENGINE.md` and `09_UI_UX_SPEC.md` did not land (mtime 12:02, identical to all other docs; both still contain `USER_STOP` (08 §3.8) and the `user` filter (09 §3.4)). Built per your explicit C1/C2 answers — please re-apply your fixes when convenient.
- **O2 — VERIFY at image build time** (D26): exact `python:3.11-slim`, `node:22-slim` tags and the CUDA base + PyTorch wheel pairing (doc 14 §4).
- **O3 — RESOLVED 2026-10-20**: Docker Compose v5.6.0 is installed; `docker compose config` passes for both `deploy/docker-compose.yml` and `deploy/docker-compose.dev.yml` (D50). Image builds remain NOT RUN by FAST-04.
- **O6 — ID collisions** (D30): with 4 random hex chars, same-second run/verification ids can collide. Slice 8 run-creation will retry on an existing `runs/{id}` dir; verification ids will do the same.
- **O7 — format_detected before verification**: D16 (user-approved) says verification writes `format_detected`; the wizard step "detected format shown after upload" (master prompt §4.2) can therefore only show the format after verification runs. If you want a provisional format at upload, say so and I'll add a `detected_hint` field.
- **O8 — D33 project linkage** needs your awareness: datasets carry an optional `project_id` (additive meta.json field). If you prefer datasets strictly global (doc 04 §1 verbatim), I'll drop the field and the `?project=` filter.
- **O9 — polling fallback**: `useRunStream` currently has WS auto-reconnect + `last_seq` resume; the doc 09 §5 polling fallback needs `GET /runs/{id}/events` (slice 8). Noted as a slice-9 gap.
- **O10 — `events.idx`**: replay scans `events.jsonl` linearly; the optional `seq → byte-offset` index [doc 03 §2.4] lands with large-run performance work (slice 9+).

## Non-blocking notes
- O4 — `TRAINHUB_ACCESS_TOKEN` enforcement (FR-20 P1 / DIST-07): **D51 approved by the user 2026-10-20** (optional token + loud startup warning + UI banner on non-localhost bind). Implementation itself lands in a later slice with `core/device.py`.

## New questions from the 2026-10-20 audit
- **O11 — workspace root — RESOLVED by user**: `/data` in the image, `./.dev-data` locally (doc 04 §1 superseded; recorded as D52).
- **O12 — registry name — not blocking**: stays `TODO(user)` for DIST-01; no publish until provided.
- **O13 — GPU image — RESOLVED by user**: the `-gpu` tag is not published until it is tested on a real GPU server (also DIST-12).
- O5 — CSP header deferred until the SPA is served in-container (doc 10 §5); nosniff/frame/referrer/HSTS are already set.

