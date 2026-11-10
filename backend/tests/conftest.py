"""Test fixtures for the TrainHub backend [doc 11 §2]."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _open_test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing tests exercise the API without a token and use local http/file sources.

    Slice E made auth and the https-only source rule default ON; tests opt out here
    (env, the same switch an operator would use) and the Slice E tests opt back in.
    """
    monkeypatch.setenv("TRAINHUB_AUTH_DISABLED", "true")
    monkeypatch.setenv("TRAINHUB_ALLOW_INSECURE_SOURCES", "true")


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    """A disposable TRAINHUB_HOME per test [doc 04 §1]."""
    return tmp_path / "trainhub_home"


@pytest.fixture()
def settings(home: Path) -> Settings:
    return Settings(home=home, log_level="DEBUG")


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
