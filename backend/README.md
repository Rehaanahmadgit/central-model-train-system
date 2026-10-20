# TrainHub backend

FastAPI service for TrainHub [doc 03 §2.1]. No database, no user data [doc 01 §3].

## Responsibility
- HTTP API (`/api/v1`) and WebSocket endpoints per doc 05
- File-based storage with atomic writes under `TRAINHUB_HOME` [doc 04]
- Verification pipeline, checkpoint resolver, training Job Manager [doc 03 §2]

## Layout (doc 03 §6)
```
app/
  api/        routers (health endpoints in slice 1)
  core/       config, logging, device abstraction, path confinement
  storage/    atomic IO, JSONL, locks, indexes, recovery (slice 2+)
  schemas/    pydantic models (slice 3+)
  services/   business logic (slice 3+)
  realtime/   broadcaster + websocket (slice 4)
  verification/  checks + cache (slice 5+)
  checkpoints/   catalog + resolver (slice 7)
  training/      job manager + plugins (slice 8)
```

## Public interface
- `app.main.create_app(settings=None) -> FastAPI` — application factory (tests inject Settings)
- `GET /healthz`, `GET /readyz` — liveness / readiness [doc 05 §7]
- Error envelope `{"error": {"code", "message", "details"}}` [doc 05 §0]

## Commands
```bash
uvicorn app.main:app --port 8080
ruff check app tests && mypy app && pytest
```
