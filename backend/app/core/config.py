"""Application settings via environment variables (12-factor) [doc 10 §9].

Every value comes from the environment with the ``TRAINHUB_`` prefix or a ``.env``
file [doc 14 §2]. Defaults below that are not stated in the docs are marked ASSUMPTION.
"""

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# TRAINHUB_DEVICE: auto | cpu | cuda | cuda:N [doc 14 §3.1]
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")


class Settings(BaseSettings):
    """TrainHub configuration. No secrets or tokens in code, only env vars [doc 13 §3]."""

    model_config = SettingsConfigDict(env_prefix="TRAINHUB_", env_file=".env", extra="ignore")

    # Workspace root; all state lives under it [doc 04 §1].
    # Local dev default: ./.dev-data (relative to CWD, gitignored). The Docker
    # image overrides this with TRAINHUB_HOME=/data (see deploy/Dockerfile).
    home: Path = Path("./.dev-data")
    # Device abstraction, the only place device is decided [doc 14 §3.1].
    device: str = "auto"
    # Optional single shared access token; identifies nobody [doc 01 §3, doc 10 §2].
    access_token: SecretStr | None = None
    # Port matches the compose example [doc 14 §7].
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: str = "INFO"
    # uvicorn access logs contain client IPs; disabled by default [doc 10 §1].
    log_access: bool = False
    # CORS allowlist [doc 10 §5]. Empty = same-origin only.
    cors_origins: list[str] = []
    # Built frontend served by the API when present [doc 14 §4].
    frontend_dist: Path | None = None
    # Chunk size for resumable uploads [doc 05 §1]. ASSUMPTION: 8 MiB default.
    chunk_size_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    # ASSUMPTION: total upload cap [doc 10 §3].
    max_upload_size_bytes: int = Field(default=50 * 1024**3, gt=0)
    # CPU-only job concurrency limit [doc 14 §9].
    max_cpu_jobs: int = Field(default=1, ge=1)
    # /readyz fails below this free disk [doc 10 §7]. ASSUMPTION: 2 GiB default.
    min_free_disk_bytes: int = Field(default=2 * 1024**3, ge=0)
    # Alert threshold [doc 10 §7].
    disk_alert_percent: int = Field(default=85, ge=1, le=100)
    # Safe-extraction limits [doc 10 §3]. Defaults are ASSUMPTIONS.
    max_extract_files: int = Field(default=100_000, ge=1)
    max_extract_total_bytes: int = Field(default=20 * 1024**3, ge=1)
    zip_bomb_ratio: int = Field(default=200, ge=2)
    zip_bomb_floor_bytes: int = Field(default=100 * 1024**2, ge=1)
    # WebSocket cadence [doc 05 §8].
    ws_heartbeat_seconds: float = Field(default=20.0, gt=0)
    ws_throttle_interval: float = Field(default=0.1, gt=0)
    ws_resume_timeout_seconds: float = Field(default=5.0, gt=0)
    # Verification runner [doc 03 §2.3, doc 06 §7]. Defaults are ASSUMPTIONS.
    verify_workers: int = Field(default=2, ge=1, le=8)
    verify_check_timeout_seconds: float = Field(default=300.0, gt=0)
    verify_sample_cap: int = Field(default=5000, ge=1)
    # Training subprocess supervision [doc 08 §5]: heartbeat every 5 s,
    # 60 s of silence → worker_lost failure.
    heartbeat_seconds: float = Field(default=5.0, gt=0)
    worker_lost_seconds: float = Field(default=60.0, gt=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if not _DEVICE_PATTERN.match(value):
            raise ValueError("TRAINHUB_DEVICE must match auto|cpu|cuda|cuda:N [doc 14 §3.1]")
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings; tests build their own instances via the app factory."""
    return Settings()
