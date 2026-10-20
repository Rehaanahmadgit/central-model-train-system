"""S7 tests: catalog, resolver, index, API, CLI [doc 07, doc 11 §3].

No internet access: downloads go to a local HTTP server on 127.0.0.1 that
implements ``Range``, so resume can be exercised deterministically.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from app.checkpoints.catalog import CatalogError, ensure_catalog, load_catalog
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
    (home / "checkpoints" / "catalog.yaml").write_text(
        yaml.safe_dump(entries), encoding="utf-8"
    )


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


def test_shipped_catalog_is_placeholder_only(tmp_path: Path) -> None:
    """Defaults must never claim a verified hash [doc 07 §2]."""
    entries = load_catalog(tmp_path)
    assert entries, "a default catalog must ship"
    assert all(not entry.verified for entry in entries)
    assert all("TODO(user)" in (entry.sha256 or "") for entry in entries)


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
    part.write_bytes(PAYLOAD[: 8192])
    resolver._transfer(entry, part)  # noqa: SLF001
    assert part.read_bytes() == PAYLOAD
    assert _RangeHandler.ranges[-1] == "bytes=8192-", "resume must send a Range request"


def test_sha_mismatch_is_rejected_and_nothing_cached(
    tmp_path: Path, http_server: str
) -> None:
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
    assert cli_main(["--home", str(tmp_path), "checkpoints", "import", str(incoming),
                     "--catalog-id", "tiny-model@1.0"]) == 0
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
    assert cli_main(["--home", str(tmp_path), "checkpoints", "import", str(incoming),
                     "--catalog-id", "tiny-model@1.0"]) == 1
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


def test_api_lists_catalog_with_local_flag(client, home: Path) -> None:
    ensure_catalog(home)
    body = client.get("/api/v1/checkpoints").json()
    assert body["total"] >= 1
    assert all(item["verified"] is False for item in body["items"])
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
