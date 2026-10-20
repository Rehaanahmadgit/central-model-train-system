"""SPA fallback tests: API routes always win; SPA served for other paths [doc 14 §4]."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _make_dist(root: Path) -> Path:
    dist = root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>TrainHub shell</title>")
    return dist


def test_api_routes_win_over_spa_fallback(tmp_path: Path) -> None:
    settings = Settings(
        home=tmp_path / "home",
        frontend_dist=_make_dist(tmp_path),
        log_level="DEBUG",
    )
    with TestClient(create_app(settings)) as test_client:
        # API routes must not be shadowed by the SPA catch-all.
        assert test_client.get("/healthz").json() == {"status": "ok"}
        assert test_client.get("/readyz").status_code == 200
        # Root and unknown non-API paths serve the SPA shell (client routing).
        assert "TrainHub shell" in test_client.get("/").text
        assert "TrainHub shell" in test_client.get("/some/deep/link").text
        # Unknown API paths keep the documented error envelope [doc 05 §0].
        api_miss = test_client.get("/api/v1/nope")
        assert api_miss.status_code == 404
        assert set(api_miss.json()) == {"error"}


def test_no_frontend_dist_means_api_only(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "home", frontend_dist=None, log_level="DEBUG")
    with TestClient(create_app(settings)) as test_client:
        assert test_client.get("/healthz").json() == {"status": "ok"}
        response = test_client.get("/some/deep/link")
        assert response.status_code == 404
        assert set(response.json()) == {"error"}
