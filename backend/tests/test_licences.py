"""Slice C: licence acceptance record + server-side gate [Slice C]."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from app.checkpoints.catalog import ensure_catalog, find_entry, load_catalog
from app.checkpoints.licences import LicenceStore
from app.checkpoints.resolver import CheckpointError, CheckpointResolver

PAYLOAD = b"fake weights " * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _catalog(home: Path, source: Path, **overrides: object) -> None:
    """One offered entry served from a local file (the fake resolver source)."""
    source.write_bytes(PAYLOAD)
    entry = {
        "name": "tiny",
        "version": "1.0",
        "model_family": "rtdetr",
        "variant": "t",
        "task_types": ["object_detection"],
        "formats": ["yolo"],
        "pretrained_on": "unit-test",
        "source_type": "file",
        "source_uri": str(source),
        "sha256": SHA,
        "size_bytes": len(PAYLOAD),
        "filename": "tiny.bin",
        "code_license": {"spdx": "MIT", "url": "https://example.invalid/code"},
        "weights_license": {"spdx": None, "url": None, "verified": False},
        "commercial_use": "unknown",
        "min_vram_gb": 0,
    }
    entry.update(overrides)
    ensure_catalog(home)
    (home / "checkpoints" / "catalog.yaml").write_text(yaml.safe_dump([entry]), encoding="utf-8")


def test_resolver_refuses_download_without_acceptance(tmp_path: Path) -> None:
    _catalog(tmp_path, tmp_path / "src.bin")
    with pytest.raises(CheckpointError) as caught:
        CheckpointResolver(tmp_path).resolve("tiny@1.0")
    assert caught.value.code == "LICENCE_NOT_ACCEPTED"
    assert not (tmp_path / "checkpoints" / "store").exists()  # nothing was downloaded


def test_resolver_downloads_after_acceptance(tmp_path: Path) -> None:
    _catalog(tmp_path, tmp_path / "src.bin")
    entry = find_entry(load_catalog(tmp_path), "tiny@1.0")
    assert entry is not None
    LicenceStore(tmp_path).accept(entry)
    resolved = CheckpointResolver(tmp_path).resolve("tiny@1.0")
    assert resolved.source == "download" and resolved.sha256 == SHA


def test_cached_weights_still_need_acceptance(tmp_path: Path) -> None:
    _catalog(tmp_path, tmp_path / "src.bin")
    entry = find_entry(load_catalog(tmp_path), "tiny@1.0")
    assert entry is not None
    CheckpointResolver(tmp_path, require_licence=False).resolve("tiny@1.0")  # already cached
    with pytest.raises(CheckpointError) as caught:
        CheckpointResolver(tmp_path).resolve("tiny@1.0")
    assert caught.value.code == "LICENCE_NOT_ACCEPTED"


def test_changed_terms_ask_again(tmp_path: Path) -> None:
    _catalog(tmp_path, tmp_path / "src.bin")
    entry = find_entry(load_catalog(tmp_path), "tiny@1.0")
    assert entry is not None
    LicenceStore(tmp_path).accept(entry)
    _catalog(tmp_path, tmp_path / "src.bin", commercial_use=False)
    with pytest.raises(CheckpointError) as caught:
        CheckpointResolver(tmp_path).resolve("tiny@1.0")
    assert caught.value.code == "LICENCE_NOT_ACCEPTED"


def test_api_accept_writes_minimal_record(client, home: Path) -> None:
    _catalog(home, home / "src.bin")
    response = client.post("/api/v1/licences/accept", json={"catalog_id": "tiny@1.0"})
    assert response.status_code == 200, response.text
    saved = json.loads((home / "licences" / "accepted.json").read_text())
    (record,) = saved["accepted"]
    # Only what and when: no user identity, address or anything else.
    assert set(record) == {"catalog_id", "terms_sha256", "accepted_at"}
    entry = find_entry(load_catalog(home), "tiny@1.0")
    assert entry is not None
    assert record["catalog_id"] == "tiny@1.0" and record["terms_sha256"] == entry.terms_sha256
    # Idempotent: accepting twice keeps one record.
    client.post("/api/v1/licences/accept", json={"catalog_id": "tiny@1.0"})
    assert len(json.loads((home / "licences" / "accepted.json").read_text())["accepted"]) == 1


def test_api_prefetch_409_without_accept_then_ok(client, home: Path) -> None:
    _catalog(home, home / "src.bin")
    denied = client.post("/api/v1/checkpoints/tiny@1.0/prefetch")
    assert denied.status_code == 409
    assert denied.json()["error"]["code"] == "LICENCE_NOT_ACCEPTED"
    client.post("/api/v1/licences/accept", json={"catalog_id": "tiny@1.0"})
    allowed = client.post("/api/v1/checkpoints/tiny@1.0/prefetch")
    assert allowed.status_code == 200, allowed.text


def test_api_accept_rejects_unknown_and_not_offered(client, home: Path) -> None:
    ensure_catalog(home)
    assert client.post("/api/v1/licences/accept", json={"catalog_id": "nope@1"}).status_code == 404
    placeholder = client.post("/api/v1/licences/accept", json={"catalog_id": "rtdetr-l@1.0"})
    assert placeholder.status_code == 409
    assert placeholder.json()["error"]["code"] == "CHECKPOINT_NOT_OFFERED"
    bad = client.post("/api/v1/licences/accept", json={"catalog_id": "../etc"})
    assert bad.status_code == 400


def test_api_flags_accepted_and_filters_by_task_and_format(client, home: Path) -> None:
    ensure_catalog(home)
    listed = client.get("/api/v1/checkpoints", params={"task": "object_detection"}).json()
    ids = {item["catalog_id"] for item in listed["items"]}
    assert ids == {"rtdetr-r18vd@coco-1.0", "rtdetr-l@1.0"}
    assert client.get("/api/v1/checkpoints", params={"format": "image_folder"}).json()["total"] == 1
    assert client.get("/api/v1/checkpoints", params={"format": "coco"}).json()["total"] == 0

    item = next(i for i in listed["items"] if i["catalog_id"] == "rtdetr-r18vd@coco-1.0")
    assert item["licence_accepted"] is False and item["offered"] is True
    client.post("/api/v1/licences/accept", json={"catalog_id": "rtdetr-r18vd@coco-1.0"})
    after = client.get("/api/v1/checkpoints", params={"family": "rtdetr"}).json()["items"]
    assert next(i for i in after if i["catalog_id"] == "rtdetr-r18vd@coco-1.0")[
        "licence_accepted"
    ]


def test_api_models_lists_variants(client, home: Path) -> None:
    ensure_catalog(home)
    params = {"task": "object_detection", "format": "yolo"}
    body = client.get("/api/v1/models", params=params).json()
    rtdetr = next(item for item in body["items"] if item["family"] == "rtdetr")
    assert [v["variant"] for v in rtdetr["variants"]] == ["l", "r18vd"]
    r18 = next(v for v in rtdetr["variants"] if v["variant"] == "r18vd")
    assert r18["weights_license"]["verified"] is False
    assert r18["code_license"]["spdx"] == "Apache-2.0"
