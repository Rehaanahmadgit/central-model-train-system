"""S7 tests: catalog, resolver, index, API, CLI [doc 07, doc 11 §3].

No internet access: downloads go to a local HTTP server on 127.0.0.1 that
implements ``Range``, so resume can be exercised deterministically.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fixtures.make_datasets import build_classification, build_yolo, to_zip

from app.checkpoints.catalog import CatalogError, ensure_catalog, load_catalog
from app.checkpoints.licences import LicenceStore
from app.checkpoints.resolver import (
    CheckpointError,
    CheckpointIndex,
    CheckpointResolver,
)
from app.cli import main as cli_main

PAYLOAD = bytes(range(256)) * 400  # ~100 KB, deterministic
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves one blob with Range support and records request offsets."""

    blob: bytes = PAYLOAD
    ranges: list[str] = []
    fail_first: int = 0
    truncate_first: int = 0  # first response is cut to this many bytes (a partial download)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        type(self).ranges.append(self.headers.get("Range") or "")
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(503)
            self.end_headers()
            return
        range_header = self.headers.get("Range")
        start = 0
        if range_header and range_header.startswith("bytes="):
            start = int(range_header.split("=", 1)[1].split("-", 1)[0] or 0)
        body = type(self).blob[start:]
        if type(self).truncate_first and not start:
            body = body[: type(self).truncate_first]
            type(self).truncate_first = 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.blob) - 1}/{len(self.blob)}"
            )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep test output clean
        return


@pytest.fixture()
def http_server() -> Iterator[str]:
    _RangeHandler.ranges = []
    _RangeHandler.fail_first = 0
    _RangeHandler.truncate_first = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/model.bin"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_catalog(home: Path, entries: list[dict]) -> None:
    ensure_catalog(home)
    (home / "checkpoints" / "catalog.yaml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    _accept_all(home)


def _accept_all(home: Path) -> None:
    """Pretend the user accepted every licence; licence gating is tested in test_licences."""
    store = LicenceStore(home)
    for entry in load_catalog(home):
        store.accept(entry)


def _entry(uri: str, **overrides: object) -> dict:
    entry = {
        "name": "tiny-model",
        "version": "1.0",
        "model_family": "resnet",
        "task_types": ["image_classification"],
        "pretrained_on": "unit-test",
        "source_type": "url",
        "source_uri": uri,
        "sha256": PAYLOAD_SHA,
        "size_bytes": len(PAYLOAD),
        "license": "MIT",
        "filename": "tiny-model.bin",
    }
    entry.update(overrides)
    return entry


# --- catalog ----------------------------------------------------------------


R18VD_SHA = "3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0"


def test_shipped_catalog_only_offers_verified_entries(tmp_path: Path) -> None:
    """Only rtdetr-r18vd has a verified hash + URL; everything else is "not offered yet"."""
    entries = {entry.catalog_id: entry for entry in load_catalog(tmp_path)}
    assert entries, "a default catalog must ship"
    offered = {cid for cid, entry in entries.items() if entry.offered}
    assert offered == {"rtdetr-r18vd@coco-1.0"}
    r18 = entries["rtdetr-r18vd@coco-1.0"]
    assert (r18.sha256, r18.size_bytes) == (R18VD_SHA, 81001853)
    assert r18.source_uri.startswith("https://")
    for entry in entries.values():
        if not entry.offered:
            assert "TODO(user)" in (entry.sha256 or "")


def test_shipped_catalog_never_claims_a_verified_weights_licence(tmp_path: Path) -> None:
    for entry in load_catalog(tmp_path):
        assert entry.weights_license["verified"] is False
        assert entry.commercial_use == "unknown"
        assert entry.min_vram_gb == 0  # 0 = not measured


def test_shipped_catalog_schema_has_every_required_field() -> None:
    from app.checkpoints.catalog import DEFAULT_CATALOG, REQUIRED_SHIPPED_KEYS

    items = yaml.safe_load(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    assert items
    for item in items:
        missing = [key for key in REQUIRED_SHIPPED_KEYS if key not in item]
        assert not missing, f"{item.get('name')}: missing {missing}"
        assert set(item["code_license"]) == {"spdx", "url"}
        assert set(item["weights_license"]) == {"spdx", "url", "verified"}
        assert item["commercial_use"] in (True, False, "unknown")
        assert isinstance(item["formats"], list) and item["formats"]
    ids = [f"{item['name']}@{item['version']}" for item in items]
    assert len(ids) == len(set(ids))


def test_override_replaces_entry(tmp_path: Path) -> None:
    _write_catalog(tmp_path, [_entry("http://example.invalid/a.bin", name="a")])
    override = tmp_path / "checkpoints" / "catalog.override.yaml"
    override.write_text(
        yaml.safe_dump([_entry("http://example.invalid/b.bin", name="a")]), encoding="utf-8"
    )
    entries = load_catalog(tmp_path)
    names = {entry.name: entry for entry in entries}
    assert names["a"].source_uri == "http://example.invalid/b.bin"


def test_malformed_catalog_raises(tmp_path: Path) -> None:
    ensure_catalog(tmp_path)
    (tmp_path / "checkpoints" / "catalog.yaml").write_text("- name: x\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)


# --- resolve / download -----------------------------------------------------


def test_resolve_downloads_verifies_and_indexes(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    events: list[dict] = []
    resolved = CheckpointResolver(tmp_path, progress=events.append).resolve("tiny-model@1.0")

    assert resolved.source == "download"
    assert resolved.sha256 == PAYLOAD_SHA
    assert resolved.path.read_bytes() == PAYLOAD
    expected = tmp_path / "checkpoints" / "store" / PAYLOAD_SHA[:2] / PAYLOAD_SHA / "tiny-model.bin"
    assert resolved.path == expected
    record = CheckpointIndex(tmp_path).get("tiny-model@1.0")
    assert record is not None and record["status"] == "ready"
    assert record["size"] == len(PAYLOAD)
    # Progress is observable for the live log [doc 07 §4.3h].
    assert any(event["event"] == "download_progress" for event in events)
    # No .part leftovers [doc 14 §3.6].
    assert list((tmp_path / "checkpoints" / ".tmp").glob("*.part")) == []


def test_second_resolve_uses_cache_without_http(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    before = len(_RangeHandler.ranges)
    again = CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert again.source == "cache"
    assert len(_RangeHandler.ranges) == before, "a cache hit must not touch the network"


def test_resume_seeks_from_offset(tmp_path: Path, http_server: str) -> None:
    """Explicitly exercise the Range path with a pre-existing part file."""
    _write_catalog(tmp_path, [_entry(http_server)])
    resolver = CheckpointResolver(tmp_path)
    entry = resolver._entry("tiny-model@1.0")  # noqa: SLF001
    part = tmp_path / "checkpoints" / ".tmp" / "manual.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(PAYLOAD[:8192])
    resolver._transfer(entry, part)  # noqa: SLF001
    assert part.read_bytes() == PAYLOAD
    assert _RangeHandler.ranges[-1] == "bytes=8192-", "resume must send a Range request"


def test_sha_mismatch_is_rejected_and_nothing_cached(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server, sha256="0" * 64)])
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_SHA_MISMATCH"
    assert CheckpointIndex(tmp_path).get("tiny-model@1.0") is None
    assert list((tmp_path / "checkpoints" / ".tmp").glob("*.part")) == []


def test_placeholder_sha_refuses_download(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server, sha256="TODO(user): provide sha256")])
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_SHA_UNVERIFIED"
    assert _RangeHandler.ranges == [], "must not download what it cannot verify"


def test_unknown_catalog_id(tmp_path: Path) -> None:
    _write_catalog(tmp_path, [_entry("http://example.invalid/a.bin")])
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path).resolve("nope@9")
    assert excinfo.value.code == "CHECKPOINT_NOT_FOUND"


@pytest.mark.slow
def test_retries_then_succeeds(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    _RangeHandler.fail_first = 2
    resolved = CheckpointResolver(tmp_path, retries=3).resolve("tiny-model@1.0")
    assert resolved.path.read_bytes() == PAYLOAD


def test_retries_exhausted(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    _RangeHandler.fail_first = 5
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path, retries=2).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_DOWNLOAD_FAILED"


# --- lock / concurrency -----------------------------------------------------


def test_two_resolvers_download_exactly_once(tmp_path: Path, http_server: str) -> None:
    """Two jobs wanting the same checkpoint: one downloads, the other waits [doc 07 §7]."""
    _write_catalog(tmp_path, [_entry(http_server)])
    results: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            resolved = CheckpointResolver(tmp_path, retries=3).resolve("tiny-model@1.0")
            results.append(resolved.source)
        except BaseException as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors
    assert sorted(results) == ["cache", "download"]
    assert len(_RangeHandler.ranges) == 1, "exactly one network fetch"


def test_local_file_source(tmp_path: Path) -> None:
    source = tmp_path / "offline.bin"
    source.write_bytes(PAYLOAD)
    _write_catalog(tmp_path, [_entry("file://" + str(source), source_type="file")])
    resolved = CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert resolved.path.read_bytes() == PAYLOAD


# --- import / rescan / evict ------------------------------------------------


def test_import_verifies_then_stores(tmp_path: Path) -> None:
    _write_catalog(tmp_path, [_entry("http://example.invalid/a.bin")])
    incoming = tmp_path / "incoming.bin"
    incoming.write_bytes(PAYLOAD)
    resolved = CheckpointResolver(tmp_path).import_file("tiny-model@1.0", incoming)
    assert resolved.source == "import"
    assert resolved.path.read_bytes() == PAYLOAD
    assert not incoming.exists()


def test_import_rejects_wrong_file(tmp_path: Path) -> None:
    _write_catalog(tmp_path, [_entry("http://example.invalid/a.bin")])
    incoming = tmp_path / "incoming.bin"
    incoming.write_bytes(b"tampered")
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path).import_file("tiny-model@1.0", incoming)
    assert excinfo.value.code == "CHECKPOINT_SHA_MISMATCH"
    assert incoming.exists(), "a rejected import must not delete the operator's file"


def test_evict_removes_file_and_index_entry(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    resolver = CheckpointResolver(tmp_path)
    resolved = resolver.resolve("tiny-model@1.0")
    assert resolver.evict("tiny-model@1.0") is True
    assert not resolved.path.exists()
    assert CheckpointIndex(tmp_path).get("tiny-model@1.0") is None
    assert resolver.evict("tiny-model@1.0") is False


def test_corrupt_local_file_is_redownloaded(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    resolver = CheckpointResolver(tmp_path)
    resolved = resolver.resolve("tiny-model@1.0")
    resolved.path.write_bytes(b"bit rot")
    again = CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert again.source == "download"
    assert again.path.read_bytes() == PAYLOAD


# --- CLI --------------------------------------------------------------------


def test_cli_import_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_catalog(tmp_path, [_entry("http://example.invalid/a.bin")])
    incoming = tmp_path / "incoming.bin"
    incoming.write_bytes(PAYLOAD)
    assert (
        cli_main(
            [
                "--home",
                str(tmp_path),
                "checkpoints",
                "import",
                str(incoming),
                "--catalog-id",
                "tiny-model@1.0",
            ]
        )
        == 0
    )
    assert "tiny-model@1.0" in capsys.readouterr().out

    assert cli_main(["--home", str(tmp_path), "checkpoints", "list"]) == 0
    listed = capsys.readouterr().out
    assert '"local": true' in listed


def test_cli_import_fails_with_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_catalog(tmp_path, [_entry("http://example.invalid/a.bin")])
    incoming = tmp_path / "incoming.bin"
    incoming.write_bytes(b"nope")
    assert (
        cli_main(
            [
                "--home",
                str(tmp_path),
                "checkpoints",
                "import",
                str(incoming),
                "--catalog-id",
                "tiny-model@1.0",
            ]
        )
        == 1
    )
    assert "CHECKPOINT_SHA_MISMATCH" in capsys.readouterr().out


def test_cli_rescan_restores_index(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    (tmp_path / "checkpoints" / "index.json").unlink()
    assert cli_main(["--home", str(tmp_path), "checkpoints", "rescan"]) == 0
    assert CheckpointIndex(tmp_path).get("tiny-model@1.0") is not None


# --- API --------------------------------------------------------------------


def _patch_catalog(home: Path, entries: list[dict]) -> None:
    ensure_catalog(home)
    (home / "checkpoints" / "catalog.yaml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    _accept_all(home)


def test_api_lists_catalog_with_local_flag(client, home: Path) -> None:
    ensure_catalog(home)
    body = client.get("/api/v1/checkpoints").json()
    assert body["total"] >= 1
    assert {item["catalog_id"] for item in body["items"] if item["offered"]} == {
        "rtdetr-r18vd@coco-1.0"
    }
    assert all(item["local"] is False for item in body["items"])
    assert body["disk"]["free_bytes"] > 0


def test_api_prefetch_then_evict(client, home: Path, http_server: str) -> None:
    _patch_catalog(home, [_entry(http_server)])
    response = client.post("/api/v1/checkpoints/tiny-model@1.0/prefetch")
    assert response.status_code == 200, response.text
    assert response.json()["local"] is True

    listed = client.get("/api/v1/checkpoints").json()["items"]
    entry = next(item for item in listed if item["catalog_id"] == "tiny-model@1.0")
    assert entry["local"] is True

    evicted = client.delete("/api/v1/checkpoints/tiny-model@1.0/local")
    assert evicted.status_code == 200
    assert evicted.json()["evicted"] is True


def test_api_prefetch_placeholder_returns_409(client, home: Path) -> None:
    ensure_catalog(home)
    response = client.post("/api/v1/checkpoints/rtdetr-l@1.0/prefetch")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CHECKPOINT_SHA_UNVERIFIED"


def test_api_prefetch_unknown_id_404(client, home: Path) -> None:
    ensure_catalog(home)
    response = client.post("/api/v1/checkpoints/nope@1/prefetch")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CHECKPOINT_NOT_FOUND"


def test_api_reload(client, home: Path) -> None:
    response = client.post("/api/v1/checkpoints/reload")
    assert response.status_code == 200
    assert response.json()["reloaded"] is True


# --- Slice D: failure matrix, import, lock wait, run-start guards -------------


def test_offline_gives_clear_code_and_offers_import(tmp_path: Path) -> None:
    _write_catalog(tmp_path, [_entry("http://127.0.0.1:1/model.bin")])  # nothing listens
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path, retries=1).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_OFFLINE"
    assert "Import from file" in excinfo.value.message
    assert excinfo.value.details["can_import"] is True


def test_sha_mismatch_is_not_retried(tmp_path: Path, http_server: str) -> None:
    _write_catalog(tmp_path, [_entry(http_server, sha256="0" * 64)])
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path, retries=3).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_SHA_MISMATCH"
    assert excinfo.value.details["can_import"] is True
    assert len(_RangeHandler.ranges) == 1, "no retry loop on a hash mismatch"


def test_partial_download_keeps_part_and_next_attempt_resumes(
    tmp_path: Path, http_server: str
) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    _RangeHandler.truncate_first = 8192
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path, retries=1).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_PARTIAL"
    parts = list((tmp_path / "checkpoints" / ".tmp").glob("*.part"))
    assert len(parts) == 1 and parts[0].stat().st_size == 8192

    resolved = CheckpointResolver(tmp_path, retries=1).resolve("tiny-model@1.0")
    assert resolved.path.read_bytes() == PAYLOAD
    assert _RangeHandler.ranges[-1] == "bytes=8192-", "second attempt must resume the .part"


def test_disk_full_preflight_uses_size_times_1_5(
    tmp_path: Path, http_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])
    usage = type("U", (), {"free": int(len(PAYLOAD) * 1.4), "total": 10**9, "used": 0})
    monkeypatch.setattr("app.checkpoints.resolver.shutil.disk_usage", lambda _p: usage)
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_DISK_FULL"
    assert excinfo.value.details["needed_bytes"] == int(len(PAYLOAD) * 1.5)
    assert _RangeHandler.ranges == [], "nothing is downloaded when the disk is too small"
    # exactly enough room (1.5x) passes the preflight
    ok = type("U", (), {"free": int(len(PAYLOAD) * 1.5) + 1, "total": 10**9, "used": 0})
    monkeypatch.setattr("app.checkpoints.resolver.shutil.disk_usage", lambda _p: ok)
    assert CheckpointResolver(tmp_path).resolve("tiny-model@1.0").source == "download"


def test_second_resolver_reports_waiting_for_lock(tmp_path: Path, http_server: str) -> None:
    from app.storage.locks import FileLock

    _write_catalog(tmp_path, [_entry(http_server)])
    events: list[str] = []
    resolver = CheckpointResolver(tmp_path, progress=lambda e: events.append(e["event"]))
    lock = FileLock(tmp_path / "checkpoints" / "locks" / "tiny-model_1.0.lock")
    lock.acquire()  # "another process" is downloading
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(resolver.resolve("tiny-model@1.0").source)
    )
    thread.start()
    deadline = time.monotonic() + 10
    while "waiting_for_lock" not in events and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "waiting_for_lock" in events and not result, "must wait, not download"
    lock.release()
    thread.join(timeout=30)
    assert result == ["download"] and len(_RangeHandler.ranges) == 1


def test_import_stream_verifies_and_stores(tmp_path: Path) -> None:
    _write_catalog(tmp_path, [_entry("http://127.0.0.1:1/x.bin")])
    resolver = CheckpointResolver(tmp_path)
    with pytest.raises(CheckpointError) as bad:
        resolver.import_stream("tiny-model@1.0", iter([b"not the weights"]))
    assert bad.value.code == "CHECKPOINT_SHA_MISMATCH"
    ok = resolver.import_stream("tiny-model@1.0", iter([PAYLOAD[:5000], PAYLOAD[5000:]]))
    assert ok.source == "import" and ok.path.read_bytes() == PAYLOAD
    assert list((tmp_path / "checkpoints" / ".tmp").glob("*.upload")) == []


def test_api_import_from_file_then_resolve_offline(client, home: Path) -> None:
    _patch_catalog(home, [_entry("http://127.0.0.1:1/x.bin")])  # unreachable source
    wrong = client.put("/api/v1/checkpoints/tiny-model@1.0/import", content=b"junk")
    assert wrong.status_code == 409
    assert wrong.json()["error"]["code"] == "CHECKPOINT_SHA_MISMATCH"
    ok = client.put("/api/v1/checkpoints/tiny-model@1.0/import", content=PAYLOAD)
    assert ok.status_code == 200, ok.text
    assert ok.json()["local"] is True
    # now a run can resolve it with no network
    assert CheckpointResolver(home, retries=1).resolve("tiny-model@1.0").source == "cache"


def _passed_verification(client, family: str, entries: dict[str, bytes]) -> str:
    from helpers import upload_dataset

    completed = upload_dataset(client, f"{family}.zip", to_zip(entries))
    verification = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": family, "preset": "dev"},
    ).json()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/verifications/{verification['id']}").json()
        if body["status"] != "running":
            return verification["id"]
        time.sleep(0.1)
    raise AssertionError("verification did not finish")


def test_run_create_without_accepted_licence_is_409(client, home: Path) -> None:
    _patch_catalog(home, [_entry("http://127.0.0.1:1/x.bin", model_family="cnn")])
    (home / "licences" / "accepted.json").unlink()  # the user has NOT accepted
    verification = _passed_verification(client, "cnn", build_classification(per_class=3))
    response = client.post(
        "/api/v1/runs",
        json={
            "verification_id": verification,
            "name": "x",
            "config": {},
            "checkpoint_catalog_id": "tiny-model@1.0",
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "LICENCE_NOT_ACCEPTED"


def test_frozen_backbone_without_checkpoint_is_refused(client) -> None:
    verification = _passed_verification(client, "rtdetr", build_yolo())
    for scratch in (True, False):
        response = client.post(
            "/api/v1/runs",
            json={
                "verification_id": verification,
                "name": "x",
                "config": {"freeze_backbone": True},
                "checkpoint_catalog_id": None,
                "from_scratch": scratch,
            },
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "FROZEN_BACKBONE_NEEDS_PRETRAINED"
    # unfrozen scratch training is allowed
    ok = client.post(
        "/api/v1/runs",
        json={
            "verification_id": verification,
            "name": "x",
            "config": {"freeze_backbone": False, "dev_preset": True},
            "checkpoint_catalog_id": None,
            "from_scratch": True,
        },
    )
    assert ok.status_code == 202, ok.text
    client.post(f"/api/v1/runs/{ok.json()['id']}/stop")


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        ("CHECKPOINT_OFFLINE", "CHECKPOINT_OFFLINE"),
        ("CHECKPOINT_SHA_MISMATCH", "CHECKPOINT_SHA_MISMATCH"),
        ("CHECKPOINT_DISK_FULL", "CHECKPOINT_DISK_FULL"),
        ("CHECKPOINT_PARTIAL", "CHECKPOINT_PARTIAL"),
        ("LICENCE_NOT_ACCEPTED", "LICENCE_NOT_ACCEPTED"),
    ],
)
def test_run_start_failure_matrix_fake_resolver(
    client, home: Path, monkeypatch: pytest.MonkeyPatch, raised: str, code: str
) -> None:
    """A fake resolver raises each failure; the run fails with that exact reason + import flag."""
    _patch_catalog(home, [_entry("http://127.0.0.1:1/x.bin", model_family="cnn")])

    def boom(self, catalog_id, *, task=None):  # noqa: ARG001
        raise CheckpointError(
            raised, f"fake {raised}", {"catalog_id": catalog_id, "can_import": True}
        )

    monkeypatch.setattr("app.training.job_manager.CheckpointResolver.resolve", boom)
    verification = _passed_verification(client, "cnn", build_classification(per_class=3))
    created = client.post(
        "/api/v1/runs",
        json={
            "verification_id": verification,
            "name": "x",
            "config": {},
            "checkpoint_catalog_id": "tiny-model@1.0",
        },
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["id"]
    deadline = time.monotonic() + 30
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/runs/{run_id}").json()
        if body["state"] == "failed":
            break
        time.sleep(0.1)
    assert body["state"] == "failed", body
    assert body["failure"]["reason"] == code
    assert body["failure"]["can_import"] is True
    assert body["failure"]["catalog_id"] == "tiny-model@1.0"
    assert f"fake {raised}" in body["failure"]["details"]


# --- override merge (field-wise) ---------------------------------------------


def test_old_style_override_keeps_default_licence_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Old override (only a TODO 'license') must not wipe the default entry's fields."""
    from app.checkpoints.catalog import catalog_paths, find_entry

    ensure_catalog(tmp_path)
    _, override = catalog_paths(tmp_path)
    override.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "rtdetr-r18vd",
                    "version": "coco-1.0",
                    "license": "TODO(user): provide the license",
                }
            ]
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        entry = find_entry(load_catalog(tmp_path), "rtdetr-r18vd@coco-1.0")
    assert entry is not None
    assert entry.code_license["spdx"] == "Apache-2.0"
    assert entry.offered is True and entry.sha256 == R18VD_SHA
    assert "retired" in caplog.text


def test_override_changes_only_the_fields_it_gives(tmp_path: Path) -> None:
    from app.checkpoints.catalog import catalog_paths, find_entry

    ensure_catalog(tmp_path)
    _, override = catalog_paths(tmp_path)
    override.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "rtdetr-r18vd",
                    "version": "coco-1.0",
                    "min_vram_gb": 3,
                    "weights_license": {"spdx": "Apache-2.0", "verified": True},
                }
            ]
        ),
        encoding="utf-8",
    )
    entry = find_entry(load_catalog(tmp_path), "rtdetr-r18vd@coco-1.0")
    assert entry is not None
    assert entry.min_vram_gb == 3
    assert entry.weights_license["spdx"] == "Apache-2.0" and entry.weights_license["verified"]
    assert entry.weights_license["url"], "url comes from the default"
    assert entry.code_license["spdx"] == "Apache-2.0" and entry.size_bytes == 81001853


# --- Slice E: SSRF guard on source_uri -----------------------------------------


@pytest.fixture()
def strict_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAINHUB_ALLOW_INSECURE_SOURCES", raising=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/w.pth",  # not https
        "ftp://example.com/w.pth",
        "https://127.0.0.1/w.pth",  # loopback
        "https://localhost/w.pth",
        "https://10.1.2.3/w.pth",  # private
        "https://192.168.0.10/w.pth",
        "https://172.16.0.1/w.pth",
        "https://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "https://100.64.0.1/w.pth",  # CGNAT
        "https://0.0.0.0/w.pth",
        "https://[::1]/w.pth",
        "https://[fe80::1]/w.pth",
        "https://[::ffff:127.0.0.1]/w.pth",  # IPv4-mapped loopback
        "https:///no-host",
    ],
)
def test_ssrf_blocks_non_public_and_non_https(strict_sources: None, url: str) -> None:
    from app.checkpoints.resolver import validate_source_url

    with pytest.raises(CheckpointError) as excinfo:
        validate_source_url("m@1", url)
    assert excinfo.value.code == "CHECKPOINT_SOURCE_BLOCKED"
    assert excinfo.value.details["can_import"] is True


def test_ssrf_allows_public_https_host(
    strict_sources: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.checkpoints.resolver import validate_source_url

    monkeypatch.setattr(
        "app.checkpoints.resolver.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("140.82.112.3", 443))],
    )
    validate_source_url("m@1", "https://github.com/x/w.pth")


def test_ssrf_hostname_resolving_to_private_ip_is_blocked(
    strict_sources: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.checkpoints.resolver import validate_source_url

    monkeypatch.setattr(
        "app.checkpoints.resolver.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("10.0.0.7", 443))],
    )
    with pytest.raises(CheckpointError) as excinfo:
        validate_source_url("m@1", "https://rebind.example/w.pth")
    assert "non-public" in excinfo.value.message


def test_redirects_are_capped_and_revalidated(
    strict_sources: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    from app.checkpoints.resolver import MAX_REDIRECTS, _GuardedRedirects

    monkeypatch.setattr(
        "app.checkpoints.resolver.socket.getaddrinfo",
        lambda host, *a, **k: [(2, 1, 6, "", (host if host[0].isdigit() else "140.82.112.3", 443))],
    )
    handler = _GuardedRedirects("m@1")
    request = urllib.request.Request("https://github.com/a")
    with pytest.raises(CheckpointError):  # redirect into the metadata address
        handler.redirect_request(request, None, 302, "Found", {}, "https://169.254.169.254/latest/")
    with pytest.raises(CheckpointError):  # redirect downgraded to http
        handler.redirect_request(request, None, 302, "Found", {}, "http://github.com/a")
    capped = _GuardedRedirects("m@1")
    for _ in range(MAX_REDIRECTS):
        capped.redirect_request(request, None, 302, "Found", {}, "https://github.com/b")
    with pytest.raises(CheckpointError) as excinfo:
        capped.redirect_request(request, None, 302, "Found", {}, "https://github.com/c")
    assert "redirects" in excinfo.value.message


def test_resolve_refuses_blocked_source_without_any_request(
    strict_sources: None, tmp_path: Path, http_server: str
) -> None:
    _write_catalog(tmp_path, [_entry(http_server)])  # http://127.0.0.1:port
    with pytest.raises(CheckpointError) as excinfo:
        CheckpointResolver(tmp_path).resolve("tiny-model@1.0")
    assert excinfo.value.code == "CHECKPOINT_SOURCE_BLOCKED"
    assert _RangeHandler.ranges == [], "no request may reach the blocked host"
    # offline import still works for the same entry
    assert (
        CheckpointResolver(tmp_path).import_stream("tiny-model@1.0", iter([PAYLOAD])).source
        == "import"
    )


def test_shipped_catalog_source_passes_scheme_rule(strict_sources: None, tmp_path: Path) -> None:
    entries = load_catalog(tmp_path)
    offered = [e for e in entries if e.offered]
    assert offered and all(e.source_uri.startswith("https://") for e in offered)
