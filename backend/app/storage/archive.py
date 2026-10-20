"""Safe archive handling [doc 10 §3].

- real content types via magic bytes, never extensions
- path traversal (`../`, absolute paths) blocked
- symlinks/hardlinks/device files rejected
- file-count, total-size and compression-ratio (zip-bomb) limits
- extraction happens into a caller-provided destination dir
"""

import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

KIND_ZIP = "zip"
KIND_TAR = "tar"
KIND_CSV = "csv"
KIND_PARQUET = "parquet"
KIND_UNKNOWN = "unknown"

_COPY_CHUNK = 1024 * 1024
_SPECIAL_MODES = frozenset(
    {0o120000, 0o020000, 0o060000, 0o010000, 0o140000}
)  # symlink, chardev, blockdev, fifo, socket


class ArchiveUnsafeError(ValueError):
    """The archive violates a safety rule (traversal, symlink, bomb, ...)."""


class ArchiveTooLargeError(ValueError):
    """The archive exceeds extraction limits."""


@dataclass(frozen=True)
class ExtractLimits:
    max_files: int = 100_000
    max_total_bytes: int = 20 * 1024**3
    zip_bomb_ratio: int = 200
    zip_bomb_floor_bytes: int = 100 * 1024**2


def detect_kind(path: Path) -> str:
    """Detect the real content type from magic bytes [doc 10 §3]."""
    with path.open("rb") as handle:
        head = handle.read(512)
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return KIND_ZIP
    if head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00":
        return KIND_TAR  # gzip/bzip2/xz-compressed tar
    if len(head) >= 262 and head[257:262] == b"ustar":
        return KIND_TAR
    if head[:4] == b"PAR1":
        return KIND_PARQUET
    if b"\x00" not in head:
        return KIND_CSV  # plain text (CSV and friends)
    return KIND_UNKNOWN


def extract_archive(source: Path, dest: Path, limits: ExtractLimits) -> int:
    """Safely extract a zip or tar archive; returns the number of files extracted."""
    kind = detect_kind(source)
    if kind == KIND_ZIP:
        return _extract_zip(source, dest, limits)
    if kind == KIND_TAR:
        return _extract_tar(source, dest, limits)
    raise ArchiveUnsafeError(f"unsupported archive format (detected: {kind})")


def _validate_member_name(name: str) -> None:
    if not name or "\x00" in name:
        raise ArchiveUnsafeError(f"empty or NUL-containing member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveUnsafeError(f"path traversal in archive member: {name!r}")


def _is_special_mode(mode: int) -> bool:
    return (mode & 0o170000) in _SPECIAL_MODES


def _check_size_limits(
    file_size: int, total: int, count: int, limits: ExtractLimits
) -> tuple[int, int]:
    new_total = total + file_size
    if new_total > limits.max_total_bytes:
        raise ArchiveTooLargeError("archive exceeds the total uncompressed size limit")
    new_count = count + 1
    if new_count > limits.max_files:
        raise ArchiveTooLargeError("archive exceeds the file-count limit")
    return new_total, new_count


def _extract_zip(source: Path, dest: Path, limits: ExtractLimits) -> int:
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > limits.max_files:
            raise ArchiveTooLargeError("archive exceeds the file-count limit")
        total = 0
        count = 0
        for info in infos:
            _validate_member_name(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if _is_special_mode(mode):
                raise ArchiveUnsafeError(f"special file in archive: {info.filename!r}")
            if info.is_dir():
                continue
            if (
                info.file_size >= limits.zip_bomb_floor_bytes
                and info.compress_size > 0
                and info.file_size / info.compress_size > limits.zip_bomb_ratio
            ):
                raise ArchiveUnsafeError(
                    f"suspicious compression ratio in archive member: {info.filename!r}"
                )
            total, count = _check_size_limits(info.file_size, total, count, limits)
        for info in infos:
            if info.is_dir():
                continue
            target = dest.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=_COPY_CHUNK)
    return count


def _extract_tar(source: Path, dest: Path, limits: ExtractLimits) -> int:
    with tarfile.open(source, mode="r:*") as archive:
        members = archive.getmembers()
        if len(members) > limits.max_files:
            raise ArchiveTooLargeError("archive exceeds the file-count limit")
        total = 0
        count = 0
        for member in members:
            _validate_member_name(member.name)
            if member.issym() or member.islnk():
                raise ArchiveUnsafeError(f"link in archive: {member.name!r}")
            if member.ischr() or member.isblk() or member.isfifo():
                raise ArchiveUnsafeError(f"device/special file in archive: {member.name!r}")
            if member.isdir():
                continue
            total, count = _check_size_limits(member.size, total, count, limits)
        for member in members:
            if member.isdir():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            target = dest.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=_COPY_CHUNK)
    return count
