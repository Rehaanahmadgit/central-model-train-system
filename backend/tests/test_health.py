"""Health and readiness tests [doc 05 §7, doc 10 §7]."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.paths import LAYOUT_DIRS
from app.main import create_app


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_creates_workspace_layout(client: TestClient, home: Path) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # The standard layout exists [doc 04 §1].
    for relative in LAYOUT_DIRS:
        assert (home / relative).is_dir()


def test_readyz_not_ready_when_workspace_is_a_file(tmp_path: Path) -> None:
    # A file blocks mkdir; startup must survive and /readyz must say not ready [doc 03 §8].
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    with TestClient(create_app(Settings(home=blocked, log_level="DEBUG"))) as test_client:
        response = test_client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_not_found_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error"}
    assert {"code", "message", "details"} <= set(body["error"])


def test_method_not_allowed_uses_error_envelope(client: TestClient) -> None:
    response = client.post("/healthz")
    assert response.status_code == 405
    assert {"code", "message", "details"} <= set(response.json()["error"])


def test_security_headers_present(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_request_id_header_echoed(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-Id": "req-test-123"})
    assert response.headers["x-request-id"] == "req-test-123"
