"""Settings parsing tests [doc 10 §9, doc 14 §2]."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.home == Path("./.dev-data")  # local dev; Docker sets /data via env
    assert settings.device == "auto"
    assert settings.access_token is None
    assert settings.port == 8080
    assert settings.max_cpu_jobs == 1
    assert settings.disk_alert_percent == 85


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAINHUB_HOME", str(tmp_path))
    monkeypatch.setenv("TRAINHUB_DEVICE", "cpu")
    monkeypatch.setenv("TRAINHUB_ACCESS_TOKEN", "shared-secret")
    monkeypatch.setenv("TRAINHUB_MAX_CPU_JOBS", "2")
    settings = Settings()
    assert settings.home == tmp_path
    assert settings.device == "cpu"
    assert settings.max_cpu_jobs == 2
    assert settings.access_token is not None
    assert settings.access_token.get_secret_value() == "shared-secret"


def test_invalid_device_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(device="tpu")


def test_invalid_port_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(port=70000)
