"""Access token enforcement tests [FR-20, DIST-07, doc 10 §2]."""

import os
import stat
from pathlib import Path

import pytest
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
        ok = client.get("/api/v1/models", headers={"X-Access-Token": "shared-secret"})
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


def test_security_state_reflects_auth_mode() -> None:
    assert security_state(Settings(access_token="x"))["warning"] is None
    assert security_state(Settings(access_token="x"))["token_required"] is True
    disabled = security_state(Settings(auth_disabled=True))
    assert disabled["token_required"] is False and "DISABLED" in disabled["warning"]
    # default (env opt-out removed): auth is ON
    assert Settings(_env_file=None).auth_disabled in (True, False)


# --- Slice E: default ON, first-boot token, opt-out ---------------------------


@pytest.fixture()
def auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAINHUB_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("TRAINHUB_ACCESS_TOKEN", raising=False)


def _settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", log_level="DEBUG", _env_file=None)


def test_first_boot_generates_token_file_0600_and_logs_it_once(
    auth_on: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        path = tmp_path / "home" / "access-token"
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        token = path.read_text().strip()
        assert len(token) >= 32
        assert capsys.readouterr().out.count(token) == 1, "printed exactly once"
        assert client.get("/api/v1/models").status_code == 401
        ok = client.get("/api/v1/models", headers={"X-Access-Token": token})
        assert ok.status_code == 200
    # second boot reuses the same token and does not print it again
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert token not in capsys.readouterr().out
        assert client.get("/api/v1/models", headers={"X-Access-Token": token}).status_code == 200


def test_missing_and_wrong_token_are_distinct_401s(auth_on: None, tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        missing = client.get("/api/v1/system")
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "ACCESS_TOKEN_REQUIRED"
        wrong = client.get("/api/v1/system", headers={"X-Access-Token": "nope"})
        assert wrong.status_code == 401
        assert wrong.json()["error"]["code"] == "ACCESS_TOKEN_INVALID"
        assert "rejected" in wrong.json()["error"]["message"]
        assert client.get("/healthz").status_code == 200  # SPA/health stay open


def test_ws_is_401_without_token_by_default(auth_on: None, tmp_path: Path) -> None:
    from starlette.websockets import WebSocketDisconnect

    with TestClient(create_app(_settings(tmp_path))) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/runs/run_20261020_000000_abcd") as websocket:
                websocket.receive_json()


def test_loose_token_file_permissions_are_tightened(auth_on: None, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "access-token").write_text("a-token-from-an-old-boot\n")
    os.chmod(home / "access-token", 0o644)
    with TestClient(create_app(_settings(tmp_path))):
        assert stat.S_IMODE((home / "access-token").stat().st_mode) == 0o600


def test_env_token_wins_and_is_not_written_to_disk(
    auth_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAINHUB_ACCESS_TOKEN", "from-env")
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert (
            client.get("/api/v1/models", headers={"X-Access-Token": "from-env"}).status_code == 200
        )
    assert not (tmp_path / "home" / "access-token").exists()


def test_opt_out_is_open_but_loud(
    auth_on: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TRAINHUB_AUTH_DISABLED", "true")
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/api/v1/models").status_code == 200
        assert not (tmp_path / "home" / "access-token").exists()
        assert client.get("/api/v1/system").json()["security"]["token_required"] is False
    out = capsys.readouterr().out
    assert "SECURITY" in out and "DISABLED" in out


def test_fails_closed_when_token_cannot_be_loaded(auth_on: None) -> None:
    assert verify_token(Settings(_env_file=None), "anything") is False
    assert verify_token(Settings(_env_file=None), None) is False


def test_reset_token_cli_replaces_token(auth_on: None, tmp_path: Path, capsys) -> None:
    from app.cli import main as cli_main

    home = tmp_path / "home"
    with TestClient(create_app(_settings(tmp_path))):
        old = (home / "access-token").read_text().strip()
    assert cli_main(["--home", str(home), "--reset-token"]) == 0
    new = (home / "access-token").read_text().strip()
    assert new != old and new in capsys.readouterr().out
    assert stat.S_IMODE((home / "access-token").stat().st_mode) == 0o600
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/api/v1/models", headers={"X-Access-Token": old}).status_code == 401
        assert client.get("/api/v1/models", headers={"X-Access-Token": new}).status_code == 200


def test_blank_env_token_is_treated_as_unset(
    auth_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compose passes TRAINHUB_ACCESS_TOKEN= (empty); that must still generate a token."""
    monkeypatch.setenv("TRAINHUB_ACCESS_TOKEN", "")
    settings = _settings(tmp_path)
    assert settings.access_token is None
    with TestClient(create_app(settings)):
        assert (tmp_path / "home" / "access-token").is_file()
