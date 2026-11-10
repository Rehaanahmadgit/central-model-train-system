"""Checkpoint resolver [doc 07 §4].

Exactly one place downloads weights, and it always follows the documented
order: lock → free-disk check → ``.tmp/*.part`` (resumable) → sha256 verify →
atomic move into the content-addressed store → atomic ``index.json`` update.
Trainers never download [doc 07 §5].

Only stdlib + the shared storage helpers are used, so the runtime image does
not need an HTTP client dependency.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.checkpoints.catalog import CatalogEntry, find_entry, load_catalog
from app.checkpoints.licences import LicenceStore
from app.storage.atomic import atomic_write_json, read_json
from app.storage.locks import FileLock, LockTimeout

INDEX_SCHEMA_VERSION = 1
CHUNK = 1024 * 256
Progress = Callable[[dict[str, Any]], None]


class CheckpointError(RuntimeError):
    """A resolution failed; ``code`` maps to the API error envelope."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class ResolvedCheckpoint:
    catalog_id: str
    path: Path
    sha256: str
    size_bytes: int
    source: str  # "cache" | "download" | "import"
    filename: str


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


MAX_REDIRECTS = 3


def _insecure_sources_allowed() -> bool:
    """Test/dev escape hatch (env). Never set it on a public instance."""
    return os.environ.get("TRAINHUB_ALLOW_INSECURE_SOURCES", "").lower() in ("1", "true", "yes")


def _blocked(catalog_id: str, why: str) -> CheckpointError:
    return CheckpointError(
        "CHECKPOINT_SOURCE_BLOCKED",
        f"download source for {catalog_id!r} refused: {why}. "
        "Use Import from file with weights you downloaded yourself.",
        {"catalog_id": catalog_id, "can_import": True},
    )


def validate_source_url(catalog_id: str, url: str) -> None:
    """SSRF guard: https only, and every resolved address must be publicly routable.

    Blocks loopback, private (RFC 1918), link-local (incl. the 169.254.169.254 cloud
    metadata address), CGNAT, multicast, reserved and unspecified ranges, IPv4-mapped
    IPv6 included. Residual risk (VERIFY): DNS rebinding between this check and the
    connect; the sha256 check still rejects anything that is not the expected file.
    """
    if _insecure_sources_allowed():
        return
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise _blocked(catalog_id, f"only https sources are allowed (got {parsed.scheme!r})")
    host = parsed.hostname
    if not host:
        raise _blocked(catalog_id, "the URL has no host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        # Cannot resolve = offline (or a bad name): let the normal download path report it.
        raise urllib.error.URLError(exc) from exc
    for info in infos:
        address = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast:
            raise _blocked(catalog_id, f"{host} resolves to a non-public address ({address})")


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Cap redirects and re-validate every hop with the same rules."""

    def __init__(self, catalog_id: str) -> None:
        self._catalog_id = catalog_id
        self._hops = 0

    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        self._hops += 1
        if self._hops > MAX_REDIRECTS:
            raise _blocked(self._catalog_id, f"more than {MAX_REDIRECTS} redirects")
        validate_source_url(self._catalog_id, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class CheckpointIndex:
    """``checkpoints/index.json`` — what is cached locally [doc 04 §1]."""

    def __init__(self, home: Path) -> None:
        self._path = home / "checkpoints" / "index.json"

    def read(self) -> dict[str, dict[str, Any]]:
        if not self._path.is_file():
            return {}
        try:
            data = read_json(self._path)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        entries = data.get("entries")
        return entries if isinstance(entries, dict) else {}

    def write(self, entries: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self._path, {"schema_version": INDEX_SCHEMA_VERSION, "entries": entries})

    def get(self, catalog_id: str) -> dict[str, Any] | None:
        entry = self.read().get(catalog_id)
        return entry if isinstance(entry, dict) else None

    def put(self, catalog_id: str, record: dict[str, Any]) -> None:
        entries = self.read()
        entries[catalog_id] = record
        self.write(entries)

    def remove(self, catalog_id: str) -> None:
        entries = self.read()
        if catalog_id in entries:
            del entries[catalog_id]
            self.write(entries)


class CheckpointResolver:
    """Resolve a catalog id to a verified local file [doc 07 §4]."""

    def __init__(
        self,
        home: Path,
        *,
        retries: int = 3,
        timeout_seconds: float = 30.0,
        lock_timeout_seconds: float | None = None,
        progress: Progress | None = None,
        require_licence: bool = True,
    ) -> None:
        self._require_licence = require_licence
        self._home = home
        self._checkpoints = home / "checkpoints"
        self._store = self._checkpoints / "store"
        self._tmp = self._checkpoints / ".tmp"
        self._locks = self._checkpoints / "locks"
        self._index = CheckpointIndex(home)
        self._retries = max(1, retries)
        self._timeout = timeout_seconds
        self._lock_timeout = lock_timeout_seconds
        self._progress = progress

    # -- public API ---------------------------------------------------------

    def resolve(self, catalog_id: str, *, task: str | None = None) -> ResolvedCheckpoint:
        entry = self._entry(catalog_id)
        if not entry.is_active:
            raise CheckpointError(
                "CHECKPOINT_INACTIVE",
                f"catalog entry {catalog_id!r} is not active",
                {"catalog_id": catalog_id},
            )
        self._check_licence(entry)
        cached = self._usable_from_cache(entry)
        if cached is not None:
            self._emit("cache_hit", entry, {"path": str(cached.path)})
            return cached

        with self._acquire_with_wait_event(entry):
            # Another process may have finished while we waited [doc 07 §4.3b].
            cached = self._usable_from_cache(entry)
            if cached is not None:
                self._emit("cache_hit", entry, {"path": str(cached.path), "after_wait": True})
                return cached
            self._preflight(entry)
            part = self._download(entry)
            final = self._commit(entry, part)
            self._record(entry, final, source="download")
            return ResolvedCheckpoint(
                catalog_id=entry.catalog_id,
                path=final,
                sha256=str(entry.sha256),
                size_bytes=final.stat().st_size,
                source="download",
                filename=entry.resolved_filename,
            )

    def import_file(self, catalog_id: str, source: Path) -> ResolvedCheckpoint:
        """Offline / air-gapped path [doc 14 §5]: verify then move into the store."""
        entry = self._entry(catalog_id)
        if not source.is_file():
            raise CheckpointError(
                "CHECKPOINT_IMPORT_MISSING", f"{source} does not exist", {"path": str(source)}
            )
        if not entry.verified:
            raise CheckpointError(
                "CHECKPOINT_SHA_UNVERIFIED",
                f"catalog entry {catalog_id!r} has no verified sha256; cannot verify the file",
                {"catalog_id": catalog_id},
            )
        digest = _sha256_file(source)
        if digest != entry.sha256:
            raise CheckpointError(
                "CHECKPOINT_SHA_MISMATCH",
                f"sha256 mismatch for {catalog_id!r}",
                {"expected": entry.sha256, "actual": digest},
            )
        with self._checkpoint_lock(entry):
            target = self._store_path(entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), target)
            self._record(entry, target, source="import")
        self._emit("imported", entry, {"path": str(target)})
        return ResolvedCheckpoint(
            catalog_id=entry.catalog_id,
            path=target,
            sha256=digest,
            size_bytes=target.stat().st_size,
            source="import",
            filename=entry.resolved_filename,
        )

    def import_stream(self, catalog_id: str, chunks: Iterable[bytes]) -> ResolvedCheckpoint:
        """Import from an uploaded byte stream ("Import from file" in the UI).

        Streams into ``.tmp`` (disk preflight first), verifies sha256, then moves into
        the store. Never gated on the licence: it is the offline path.
        """
        entry = self._entry(catalog_id)
        self._preflight(entry)
        staging = self._tmp / f"{_safe_name(entry.catalog_id)}.{uuid.uuid4().hex}.upload"
        limit = max(entry.size_bytes, 0) * 2 or None
        written = 0
        try:
            with staging.open("wb") as handle:
                for block in chunks:
                    written += len(block)
                    if limit is not None and written > limit:
                        raise CheckpointError(
                            "CHECKPOINT_IMPORT_TOO_LARGE",
                            f"uploaded file is larger than expected for {catalog_id!r}",
                            {"expected_bytes": entry.size_bytes},
                        )
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if written == 0:
                raise CheckpointError("CHECKPOINT_IMPORT_MISSING", "the uploaded file is empty", {})
            return self.import_file(catalog_id, staging)
        finally:
            staging.unlink(missing_ok=True)

    def evict(self, catalog_id: str) -> bool:
        """Remove a cached file and its index entry [doc 05 §4]."""
        record = self._index.get(catalog_id)
        self._index.remove(catalog_id)
        if not record:
            return False
        path = Path(str(record.get("path", "")))
        if path.is_file() and self._is_inside_store(path):
            path.unlink(missing_ok=True)
            self._prune_empty_dirs(path.parent)
            return True
        return False

    def status(self) -> dict[str, dict[str, Any]]:
        return self._index.read()

    def prefetch(self, catalog_id: str) -> ResolvedCheckpoint:
        return self.resolve(catalog_id)

    # -- internals ----------------------------------------------------------

    def _entry(self, catalog_id: str) -> CatalogEntry:
        entry = find_entry(load_catalog(self._home), catalog_id)
        if entry is None:
            raise CheckpointError(
                "CHECKPOINT_NOT_FOUND",
                f"catalog id {catalog_id!r} is not in the catalog",
                {"catalog_id": catalog_id},
            )
        if not entry.verified:
            raise CheckpointError(
                "CHECKPOINT_SHA_UNVERIFIED",
                (
                    f"catalog entry {catalog_id!r} still has a placeholder sha256. "
                    "Fill it in catalog.yaml (or the volume override) first."
                ),
                {"catalog_id": catalog_id, "sha256": entry.sha256},
            )
        if not entry.source_uri or "TODO(user)" in entry.source_uri:
            raise CheckpointError(
                "CHECKPOINT_SOURCE_UNCONFIGURED",
                f"catalog entry {catalog_id!r} has no real download URL",
                {"catalog_id": catalog_id},
            )
        return entry

    def _check_licence(self, entry: CatalogEntry) -> None:
        """Server-side gate: no weights are used or downloaded before acceptance."""
        if self._require_licence and not LicenceStore(self._home).is_accepted(entry):
            raise CheckpointError(
                "LICENCE_NOT_ACCEPTED",
                (
                    f"licence for {entry.catalog_id!r} has not been accepted. "
                    "Review and accept it in the Model step "
                    "(or POST /api/v1/licences/accept)."
                ),
                {"catalog_id": entry.catalog_id, "terms_sha256": entry.terms_sha256},
            )

    @contextmanager
    def _acquire_with_wait_event(self, entry: CatalogEntry) -> Iterator[None]:
        """Take the per-checkpoint lock; say so when another downloader holds it."""
        probe = FileLock(self._locks / f"{_safe_name(entry.catalog_id)}.lock", timeout=0.0)
        try:
            probe.acquire()
        except LockTimeout:
            self._emit("waiting_for_lock", entry, {})
            lock = self._checkpoint_lock(entry)
            with lock:
                yield
            return
        try:
            yield
        finally:
            probe.release()

    def _checkpoint_lock(self, entry: CatalogEntry) -> FileLock:
        """Per-catalog-id flock: one downloader, the rest wait [doc 07 §7]."""
        return FileLock(
            self._locks / f"{_safe_name(entry.catalog_id)}.lock", timeout=self._lock_timeout
        )

    def _store_path(self, entry: CatalogEntry) -> Path:
        digest = str(entry.sha256)
        return self._store / digest[:2] / digest / entry.resolved_filename

    def _is_inside_store(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self._store.resolve())
        except OSError:  # pragma: no cover - defensive
            return False

    def _usable_from_cache(self, entry: CatalogEntry) -> ResolvedCheckpoint | None:
        record = self._index.get(entry.catalog_id)
        if not record or record.get("status") != "ready":
            return None
        path = Path(str(record.get("path", "")))
        if not path.is_file() or not self._is_inside_store(path):
            return None
        # Fast size check, then hash once per process lifetime [doc 07 §4.2].
        expected_size = int(record.get("size") or 0)
        if expected_size and path.stat().st_size != expected_size:
            return None
        if record.get("sha256") != entry.sha256:
            return None
        verified_at = record.get("verified_at")
        if not verified_at or record.get("mtime") != path.stat().st_mtime:
            if _sha256_file(path) != entry.sha256:
                self._index.remove(entry.catalog_id)
                return None
            record["verified_at"] = _now()
            record["mtime"] = path.stat().st_mtime
        record["last_used_at"] = _now()
        self._index.put(entry.catalog_id, record)
        return ResolvedCheckpoint(
            catalog_id=entry.catalog_id,
            path=path,
            sha256=str(entry.sha256),
            size_bytes=path.stat().st_size,
            source="cache",
            filename=path.name,
        )

    def _preflight(self, entry: CatalogEntry) -> None:
        self._tmp.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self._home)
        needed = max(entry.size_bytes, 0) * 1.5
        if needed and usage.free < needed:
            raise CheckpointError(
                "CHECKPOINT_DISK_FULL",
                (
                    f"not enough free disk for {entry.catalog_id!r}: "
                    f"need {needed / 1e6:.0f} MB, have {usage.free / 1e6:.0f} MB. "
                    "Free some space (Checkpoints page → evict) or use Import from file "
                    "on a disk with room."
                ),
                {"needed_bytes": int(needed), "free_bytes": usage.free, "can_import": True},
            )

    def _download(self, entry: CatalogEntry) -> Path:
        # Deterministic name: a later attempt resumes the same .part (Range) [doc 07 §4].
        part = self._tmp / f"{_safe_name(entry.catalog_id)}.{str(entry.sha256)[:16]}.part"
        last_error: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                self._transfer(entry, part)
                return part
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
                if attempt == self._retries:
                    break
                # Exponential backoff between retries [doc 07 §4.3d].
                time.sleep(min(2 ** (attempt - 1), 8))
        # The .part is kept on purpose: the next attempt continues where this stopped.
        offline = isinstance(last_error, urllib.error.URLError) and not isinstance(
            last_error, urllib.error.HTTPError
        )
        have = part.stat().st_size if part.is_file() else 0
        if offline:
            code = "CHECKPOINT_OFFLINE"
            message = (
                f"cannot reach the download server for {entry.catalog_id!r} "
                f"({getattr(last_error, 'reason', last_error)}). "
                "No internet? Use Import from file with the weights downloaded elsewhere."
            )
        elif have:
            code = "CHECKPOINT_PARTIAL"
            message = (
                f"download of {entry.catalog_id!r} stopped at {have} of "
                f"{entry.size_bytes} bytes ({last_error}). Run again to resume, "
                "or use Import from file."
            )
        else:
            code = "CHECKPOINT_DOWNLOAD_FAILED"
            message = (
                f"could not download {entry.catalog_id!r}: {last_error}. "
                "Try again, or use Import from file."
            )
        raise CheckpointError(
            code,
            message,
            {"catalog_id": entry.catalog_id, "attempts": self._retries, "can_import": True},
        )

    def _transfer(self, entry: CatalogEntry, part: Path) -> None:
        if entry.source_type == "file":
            self._copy_local(entry, part)
        else:
            if part.is_file() and entry.size_bytes and part.stat().st_size > entry.size_bytes:
                part.unlink()  # larger than expected: cannot be a valid prefix
            if not (
                part.is_file() and entry.size_bytes and part.stat().st_size == entry.size_bytes
            ):
                self._http_download(entry, part)
        if entry.size_bytes and part.stat().st_size < entry.size_bytes:
            raise OSError(
                f"incomplete download ({part.stat().st_size} of {entry.size_bytes} bytes)"
            )
        digest = _sha256_file(part)
        if digest != entry.sha256:
            part.unlink(missing_ok=True)
            raise CheckpointError(
                "CHECKPOINT_SHA_MISMATCH",
                (
                    f"sha256 mismatch after downloading {entry.catalog_id!r}: the file is not "
                    "the expected one and was discarded (not retried). Use Import from file "
                    "with a copy you trust."
                ),
                {"expected": entry.sha256, "actual": digest, "can_import": True},
            )

    def _copy_local(self, entry: CatalogEntry, part: Path) -> None:
        source = Path(urllib.parse.urlparse(entry.source_uri).path or entry.source_uri)
        if entry.source_uri.startswith("file://"):
            source = Path(urllib.parse.urlparse(entry.source_uri).path)
        if not source.is_file():
            raise OSError(f"local source {source} does not exist")
        part.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, part)

    def _http_download(self, entry: CatalogEntry, part: Path) -> None:
        parsed = urllib.parse.urlparse(entry.source_uri)
        if parsed.scheme not in ("http", "https"):
            raise OSError(f"unsupported URL scheme {parsed.scheme!r}")
        validate_source_url(entry.catalog_id, entry.source_uri)
        opener = urllib.request.build_opener(_GuardedRedirects(entry.catalog_id))
        offset = part.stat().st_size if part.is_file() else 0
        request = urllib.request.Request(entry.source_uri, method="GET")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with opener.open(request, timeout=self._timeout) as response:
            status = getattr(response, "status", 200)
            if offset and status != 206:
                # Server ignored Range: restart from zero rather than corrupt the file.
                offset = 0
                part.unlink(missing_ok=True)
            total = entry.size_bytes or int(response.headers.get("Content-Length") or 0) + offset
            mode = "ab" if offset else "wb"
            written = offset
            with part.open(mode) as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    written += len(block)
                    self._emit(
                        "download_progress",
                        entry,
                        {
                            "downloaded_bytes": written,
                            "total_bytes": total,
                            "percent": round(100 * written / total, 1) if total else None,
                        },
                    )
                handle.flush()
                os.fsync(handle.fileno())

    def _commit(self, entry: CatalogEntry, part: Path) -> Path:
        target = self._store_path(entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, target)
        return target

    def _record(self, entry: CatalogEntry, path: Path, *, source: str) -> None:
        self._index.put(
            entry.catalog_id,
            {
                "path": str(path),
                "sha256": entry.sha256,
                "size": path.stat().st_size,
                "status": "ready",
                "last_used_at": _now(),
                "verified_at": _now(),
                "mtime": path.stat().st_mtime,
                "source": source,
            },
        )

    def _prune_empty_dirs(self, directory: Path) -> None:
        current = directory
        while current != self._store and self._is_inside_store(current):
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _emit(self, event: str, entry: CatalogEntry, data: dict[str, Any]) -> None:
        if self._progress is None:
            return
        self._progress({"event": event, "catalog_id": entry.catalog_id, **data})


def _safe_name(catalog_id: str) -> str:
    """Filesystem-safe lock name; the catalog id is not a path [doc 10 §5]."""
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in catalog_id
    )
