# 10 — Security, Privacy and Production Readiness

## 1. Privacy by design (hard requirement)
The platform stores **no client or user information**:
- No accounts, passwords, emails, names, roles, organizations, API keys per person, avatars, preferences tied to a person.
- No IP addresses or user-agent strings in app logs, audit files, or run records. Configure nginx/Caddy `access_log off` or anonymized logging (e.g., truncated IP) and never persist raw client IPs.
- No analytics/telemetry SDKs, no third-party trackers, no cookies except (optionally) none at all.
- Stored data = (a) datasets uploaded by the user, (b) technical run records (config, metrics, logs, hardware/software versions), (c) checkpoint files, (d) person-free audit lines.
- Logs must never include dataset *content* paths from outside the workspace, secrets, or tokens.
- Provide delete endpoints for datasets and runs; deletion removes files and appends a person-free audit line.
- Acceptance test: grep the entire workspace + container logs for email patterns, IP patterns, `User-Agent`; expect zero matches.

## 2. Access control without users
- Deploy only on a private network / VPN / localhost behind a reverse proxy with TLS.
- Optional global access token `TRAINHUB_ACCESS_TOKEN` (env var, constant-time compare). It protects the app as a whole and identifies no one. Rotate by changing the env var.
- Admin-only operations (delete checkpoint cache, reload catalog) are not role-gated; keep them behind the same token and, if needed, a separate proxy path rule.
- Never expose the app directly to the public internet without the token + proxy rate limits.

## 3. Upload and data safety
- Max upload size and total workspace size limits from config; chunked upload with per-chunk checksum.
- Archive hardening: block path traversal (`../`), absolute paths, symlinks/hardlinks, device files; limit file count, total uncompressed size, compression ratio (zip-bomb); extract in a restricted temp dir, then move.
- Detect real content types (magic bytes), not extensions.
- Optional ClamAV hook.
- Filesystem permissions: workspace owned by the service user, mode 750; uploaded files never executable; never serve raw dataset directories statically.
- Encryption at rest: use disk/volume encryption (LUKS) if required; TLS in transit.

## 4. ML-specific security
- **Never** `pickle.load` / `torch.load` untrusted files. Use safetensors where possible; `weights_only=True`; pickles only from catalog entries with a verified sha256.
- Trainer plugins are repository code only (no user-uploaded code in v1).
- Training subprocess: non-root, memory/CPU limits (container limits/cgroups), no outbound network access except what the checkpoint resolver (parent process) performs, working directory restricted to the run folder.
- Checkpoint downloads: HTTPS only, hostname allowlist from `catalog.yaml`, sha256 verification mandatory.

## 5. Web security
- Strict input validation (Pydantic), path-parameter sanitization (IDs match `^[A-Za-z0-9_\-]+$`; never join user input into paths without resolving and checking it stays inside the workspace), security headers (HSTS, CSP, X-Content-Type-Options), CORS allowlist, signed/limited artifact downloads, output escaping in the UI.

## 6. Auditability (system, not people)
- Append-only `audit/audit-YYYY-MM.jsonl`: dataset uploaded/deleted, verification started/finished, run state changes, checkpoint downloaded/evicted, export, registry stage changes, catalog reload, crash recovery actions.
- Each run's `run.json` + `events.jsonl` give the full technical history (what ran, with which data/config/checkpoint/versions/hardware).
- Make audit files tamper-evident (optional P3): each line stores `prev_hash`; verify chain via `GET /audit/verify`.
- File permissions: audit files append-only for the service (`chattr +a` where available).

## 7. Reliability (file-based storage done right)
- Atomic writes (`tmp` + `fsync` + `os.replace`), append-only logs, per-run writer lock, crash recovery at startup (doc 04 §3).
- Health: `/healthz` (process alive), `/readyz` (workspace writable, free disk > threshold, GPU visible when required).
- Graceful shutdown: on SIGTERM stop accepting jobs, ask training subprocesses to checkpoint (or `last.ckpt` is already recent), flush and fsync JSONL, mark runs `interrupted` so they can resume.
- Subprocess supervision: heartbeat every 5 s; 60 s silence → kill process group, `failed(worker_lost)`.
- Timeouts and retries with exponential backoff on network calls; resumable downloads.
- Disk safeguards: preflight space check; retention/cleanup jobs; LRU checkpoint eviction (never evict in-use files); alert at 85% usage.
- Backups: snapshot/rsync the workspace; test a restore; `datasets/*/raw` optional.

## 8. Observability
- Structured JSON logs with `run_id`, `request_id`, component — **no personal data**.
- Prometheus `/metrics`: API latency/errors, active/queued jobs, verification duration, checkpoint cache hit rate, download failures, WebSocket connections, GPU utilisation, disk usage.
- Optional OpenTelemetry traces; error tracking (self-hosted Sentry) with PII scrubbing enabled.
- Dashboards (Grafana) and alerts (webhook/Slack/email to an ops address in config): failed-run rate, disk > 85%, no GPU, stuck queue, recovery events.

## 9. Configuration and environments
- 12-factor: env vars via `pydantic-settings` (`TRAINHUB_HOME`, limits, token, GPU settings, retention).
- CI: ruff, mypy, pytest, `pip-audit`, container scan; reproducible Docker builds with pinned lockfile.
- Feature flags for risky features.

## 10. Production checklist
- [ ] TLS + security headers; app reachable only on private network
- [ ] Access token set (if network is shared)
- [ ] Reverse proxy logs anonymized or off; privacy grep test passes
- [ ] Workspace on a dedicated volume with free-space alert
- [ ] Backup + restore tested
- [ ] Crash drills: kill training process, kill API mid-run, fill disk, power-cut simulation (kill -9) → state recovers
- [ ] Load test: concurrent runs + WebSocket clients
- [ ] Archive-upload fuzz/pen test
- [ ] Runbook for common failures
