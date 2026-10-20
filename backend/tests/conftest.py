"""Test fixtures for the TrainHub backend [doc 11 §2]."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


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
