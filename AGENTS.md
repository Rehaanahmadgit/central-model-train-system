# AGENTS.md — rules for anyone (human or AI) changing this repo

1. **No database, no user data.** All state is files under `TRAINHUB_HOME`. No accounts, emails, IPs, user agents, telemetry. The licence accept record stores only what/when.
2. **Every slice = test + one commit + `make verify` + `git push origin main`**, then stop and report. One `make verify` per slice.
3. **Destructive commands banned:** Docker build/rebuild, `rm -rf`, `git clean`, `git reset --hard`, force-push. No new dependency without telling the user.
4. `ruff format` only on your own files, never the whole repo.
5. Weights only via `CheckpointResolver` (sha256 verified, `weights_only=True`). "Import from file" must always work offline. Never guess sha256/URL/licence: write `TODO(user)` and mark the entry "not offered yet".
6. A plugin is a new file plus a registry row.
7. Anything not run is reported as NOT RUN; unverified facts are VERIFY/TODO(user).
8. **DOCS RULE (definition of done):** update in the same commit — docs/PRD.md (FR entry), docs/SCOPE.md, docs/ARCHITECTURE.md (new module/endpoint/storage path), docs/USE_CASES.md, docs/ROADMAP.md (slice + commit hash), docs/CHECKPOINTS.md (catalog change), capabilities/capabilities.yaml (status only with evidence), CHANGELOG.md (1–2 user-facing lines), DECISIONS.md, PROGRESS.md. Commit message says `docs updated: <files>`; the report says which docs changed. No doc on guesses.
