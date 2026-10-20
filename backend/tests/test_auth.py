"""Access token enforcement tests [FR-20, DIST-07, doc 10 §2]."""

from fastapi.testclient import TestClient

from app.core.auth import security_state, verify_token
from app.core.config import Settings
from app.main import create_app


def test_token_not_set_means_open(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200


def test_token_set_blocks_api_without_header(tmp_path) -> None:
    from pathlib import Path

    settings = Settings(
        home=Path(tmp_path) / "home",
        access_token="shared-secret",
        log_level="DEBUG",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/models").status_code == 401
        ok = client.get(
            "/api/v1/models", headers={"X-Access-Token": "shared-secret"}
        )
        assert ok.status_code == 200
        bad = client.get("/api/v1/models", headers={"X-Access-Token": "wrong"})
        assert bad.status_code == 401
        # SPA + health stay open so the browser can load and prompt.
        assert client.get("/healthz").status_code == 200
        assert client.get("/").status_code != 401


def test_ws_without_token_is_rejected(tmp_path) -> None:
    from pathlib import Path

    import pytest
    from starlette.websockets import WebSocketDisconnect

    settings = Settings(home=Path(tmp_path) / "home", access_token="tok", log_level="DEBUG")
    with TestClient(create_app(settings)) as client:
        valid_id = "run_20261020_000000_abcd"
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws/runs/{valid_id}") as websocket:
                websocket.receive_json()


def test_ws_with_token_receives_snapshot(tmp_path) -> None:
    from pathlib import Path

    settings = Settings(home=Path(tmp_path) / "home", access_token="tok", log_level="DEBUG")
    with TestClient(create_app(settings)) as client:
        valid_id = "run_20261020_000000_abcd"
        with client.websocket_connect(f"/ws/runs/{valid_id}?token=tok") as websocket:
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.close()


def test_verify_token_constant_time() -> None:
    settings = Settings(access_token="abc")
    assert verify_token(settings, "abc") is True
    assert verify_token(settings, "abd") is False
    assert verify_token(settings, None) is False
    open_settings = Settings()
    assert verify_token(open_settings, None) is True


def test_security_state_warning_on_non_localhost() -> None:
    assert security_state(Settings(bind_host="127.0.0.1"))["warning"] is None
    assert security_state(Settings(bind_host="0.0.0.0"))["warning"] is not None
    assert security_state(Settings(bind_host="0.0.0.0", access_token="x"))["warning"] is None
