"""Try-it inference tests [Slice G1].

One tiny real RT-DETR run (CPU, dev preset) is trained once for the module; every
test then talks to the real endpoints. Detection runs in a subprocess, so the API
process stays torch-free (checked in a fresh interpreter).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fixtures.make_datasets import png_bytes, to_zip
from helpers import upload_dataset

from app.core.config import Settings
from app.main import create_app

TOL = 1e-4


def _dataset() -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    index = 0
    for split, count in (("train", 8), ("val", 4), ("test", 3)):
        for _ in range(count):
            entries[f"{split}/images/img{index}.png"] = png_bytes((10 + index, 40, 90))
            entries[f"{split}/labels/img{index}.txt"] = b"0 0.25 0.25 0.5 0.5\n"
            index += 1
    entries["data.yaml"] = b"names:\n  - cat\n  - dog\n"
    return entries


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, str]]:
    """(home, run_id) of a finished tiny rtdetr run with predictions.json."""
    patch = pytest.MonkeyPatch()
    patch.setenv("TRAINHUB_AUTH_DISABLED", "true")
    patch.setenv("TRAINHUB_ALLOW_INSECURE_SOURCES", "true")
    home = tmp_path_factory.mktemp("infer") / "home"
    try:
        with TestClient(create_app(Settings(home=home, log_level="WARNING"))) as client:
            completed = upload_dataset(client, "det.zip", to_zip(_dataset()))
            verification = client.post(
                "/api/v1/verifications",
                json={
                    "dataset_hash": completed["dataset_hash"],
                    "model_family": "rtdetr",
                    "preset": "dev",
                },
            ).json()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                body = client.get(f"/api/v1/verifications/{verification['id']}").json()
                if body["status"] != "running":
                    break
                time.sleep(0.1)
            created = client.post(
                "/api/v1/runs",
                json={
                    "verification_id": verification["id"],
                    "name": "infer-run",
                    "config": {
                        "dev_preset": True,
                        "epochs": 1,
                        "batch_size": 2,
                        "imgsz": 128,
                        "freeze_backbone": False,
                        "score_threshold": 0.0,
                    },
                },
            )
            assert created.status_code == 202, created.text
            run_id = created.json()["id"]
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                state = client.get(f"/api/v1/runs/{run_id}").json()["state"]
                if state in ("completed", "failed", "stopped"):
                    break
                time.sleep(0.2)
            assert state == "completed", state
        yield home, run_id
    finally:
        patch.undo()


@pytest.fixture()
def api(trained: tuple[Path, str]) -> Iterator[tuple[TestClient, str, Path]]:
    home, run_id = trained
    with TestClient(create_app(Settings(home=home, log_level="WARNING"))) as client:
        yield client, run_id, home


def _predictions(home: Path, run_id: str) -> dict:
    return json.loads((home / "runs" / run_id / "artifacts" / "predictions.json").read_text())


def _assert_rows_close(got: list[list[float]], want: list[list[float]]) -> None:
    """Same detections within TOL. Matched as a set: an untrained model has many
    near-tied scores whose order may differ by float noise, the values may not."""
    assert len(got) == len(want) and want, "known answer must not be empty"
    unused = list(want)
    for row in got:
        assert len(row) == 6
        for index, candidate in enumerate(unused):
            same_class = int(row[0]) == int(candidate[0])
            if same_class and all(
                abs(x - y) <= TOL + 1e-9 for x, y in zip(row[1:], candidate[1:], strict=False)
            ):
                del unused[index]
                break
        else:
            raise AssertionError(f"no match within {TOL} for {row}")
    assert not unused


def test_samples_match_predictions_json_known_answer(api) -> None:
    client, run_id, home = api
    response = client.post(f"/api/v1/runs/{run_id}/infer-samples?threshold=0")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["class_names"] == _predictions(home, run_id)["class_names"]
    assert body["device_note"] and body["device"]["kind"] in ("cpu", "cuda")
    by_path = {image["path"]: image for image in _predictions(home, run_id)["images"]}
    assert 1 <= len(body["images"]) <= 6
    assert all(image["split"] == "test" for image in body["images"]), "test split preferred"
    for image in body["images"]:
        known = by_path[image["path"]]
        assert (image["width"], image["height"]) == (known["width"], known["height"])
        assert all(len(row) == 5 for row in image["gt"])
        _assert_rows_close(image["pred"], known["pred"])


def test_samples_are_deterministic(api) -> None:
    client, run_id, _ = api
    first = client.post(f"/api/v1/runs/{run_id}/infer-samples").json()
    second = client.post(f"/api/v1/runs/{run_id}/infer-samples").json()
    assert [i["path"] for i in first["images"]] == [i["path"] for i in second["images"]]


def test_uploaded_image_matches_the_same_file_in_predictions(api) -> None:
    client, run_id, home = api
    known = _predictions(home, run_id)["images"][0]
    record = json.loads((home / "runs" / run_id / "run.json").read_text())
    data = (home / "datasets" / record["dataset_hash"] / "raw" / known["path"]).read_bytes()
    response = client.post(
        f"/api/v1/runs/{run_id}/infer?threshold=0",
        files=[("images", ("my photo.png", data, "image/png"))],
    )
    assert response.status_code == 200, response.text
    image = response.json()["images"][0]
    assert image["name"] == "my photo.png" and "gt" not in image
    _assert_rows_close(image["pred"], known["pred"])


def test_threshold_filters_rows(api) -> None:
    client, run_id, _ = api
    low = client.post(f"/api/v1/runs/{run_id}/infer-samples?threshold=0").json()
    high = client.post(f"/api/v1/runs/{run_id}/infer-samples?threshold=0.99").json()
    assert sum(len(i["pred"]) for i in high["images"]) < sum(len(i["pred"]) for i in low["images"])
    assert client.post(f"/api/v1/runs/{run_id}/infer-samples?threshold=2").status_code == 422


def test_rejections_are_precise(api) -> None:
    client, run_id, home = api
    url = f"/api/v1/runs/{run_id}/infer"
    good = png_bytes((1, 2, 3))

    def post(files: list[tuple[str, tuple[str, bytes, str]]]) -> object:
        return client.post(url, files=files)

    wrong = post([("images", ("notes.txt", b"hello", "text/plain"))])
    assert wrong.status_code == 415 and wrong.json()["error"]["code"] == "UNSUPPORTED_IMAGE"
    gif = post([("images", ("a.gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif"))])
    assert gif.status_code == 415
    big = post([("images", ("big.png", good + b"\0" * (10 * 1024 * 1024), "image/png"))])
    assert big.status_code == 413 and big.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    many = post([("images", (f"{i}.png", good, "image/png")) for i in range(9)])
    assert many.status_code == 400 and many.json()["error"]["code"] == "TOO_MANY_IMAGES"
    assert post([]).status_code in (400, 415)
    huge = client.post(url, content=b"x" * 10, headers={"Content-Length": str(200 * 1024 * 1024)})
    assert huge.status_code == 413
    assert client.post("/api/v1/runs/..%2Fetc/infer-samples").status_code in (400, 404)
    assert client.post("/api/v1/runs/run_20200101_000000_abcd/infer-samples").status_code == 404
    assert not list((home / "tmp").glob("infer-*")), "temp images are always deleted"


def test_traversal_in_filename_is_neutralised(api) -> None:
    client, run_id, home = api
    response = client.post(
        f"/api/v1/runs/{run_id}/infer",
        files=[("images", ("../../../evil.png", png_bytes((1, 2, 3)), "image/png"))],
    )
    assert response.status_code == 200
    assert "/" not in response.json()["images"][0]["name"]
    assert not (home.parent / "evil.png").exists()
    assert not list((home / "tmp").glob("infer-*"))


def test_unsupported_family_and_missing_checkpoint(api) -> None:
    client, _, home = api
    for run_id, family in (
        ("run_20200101_000000_aaaa", "cnn"),
        ("run_20200101_000000_bbbb", "rtdetr"),
    ):
        run_dir = home / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps({"id": run_id, "model_family": family, "state": "completed"})
        )
    cnn = client.post("/api/v1/runs/run_20200101_000000_aaaa/infer-samples")
    assert cnn.status_code == 422 and cnn.json()["error"]["code"] == "INFER_UNSUPPORTED"
    nock = client.post("/api/v1/runs/run_20200101_000000_bbbb/infer-samples")
    assert nock.status_code == 409 and nock.json()["error"]["code"] == "NO_BEST_CHECKPOINT"


def test_token_is_required(trained, monkeypatch: pytest.MonkeyPatch) -> None:
    home, run_id = trained
    monkeypatch.delenv("TRAINHUB_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("TRAINHUB_ACCESS_TOKEN", "secret-token")
    with TestClient(create_app(Settings(home=home, log_level="WARNING", _env_file=None))) as c:
        for path in ("infer", "infer-samples"):
            denied = c.post(f"/api/v1/runs/{run_id}/{path}")
            assert denied.status_code == 401, path
        ok = c.post(
            f"/api/v1/runs/{run_id}/infer-samples", headers={"X-Access-Token": "secret-token"}
        )
        assert ok.status_code == 200


def test_api_process_never_imports_torch_for_inference(trained) -> None:
    home, run_id = trained
    code = (
        "import sys; from fastapi.testclient import TestClient; "
        "from app.core.config import Settings; from app.main import create_app; "
        f"c = TestClient(create_app(Settings(home=r'{home}'))); c.__enter__(); "
        f"r = c.post('/api/v1/runs/{run_id}/infer-samples'); "
        "print('STATUS', r.status_code); print('TORCH_IN_API=' + str('torch' in sys.modules))"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**__import__("os").environ, "TRAINHUB_AUTH_DISABLED": "true"},
    )
    assert "STATUS 200" in out.stdout and "TORCH_IN_API=False" in out.stdout, out.stdout[-800:]
