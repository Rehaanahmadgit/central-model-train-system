"""Safe-extraction tests [doc 10 §3, doc 11 §3]."""

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from app.storage.archive import (
    ArchiveTooLargeError,
    ArchiveUnsafeError,
    ExtractLimits,
    detect_kind,
    extract_archive,
)


def _make_zip(path: Path, entries: list[tuple[str, bytes, int]]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = mode << 16
            archive.writestr(info, data)


def _make_tar(path: Path, mode: str, entries: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, mode=mode) as archive:
        for name, data, kind in entries:
            if kind == "file":
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info = tarfile.TarInfo(name)
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                archive.addfile(info)
            else:
                raise ValueError(kind)


def test_detect_kind_magic_bytes(tmp_path: Path) -> None:
    cases = [
        (b"PK\x03\x04rest", "zip"),
        (b"\x1f\x8brest", "tar"),
        (b"PAR1rest", "parquet"),
        (b"a,b,c\n1,2,3\n", "csv"),
        (b"\x00\x01\x02\x03binary", "unknown"),
    ]
    for content, expected in cases:
        path = tmp_path / f"sample-{expected}.bin"
        path.write_bytes(content)
        assert detect_kind(path) == expected


def test_extract_clean_zip(tmp_path: Path) -> None:
    source = tmp_path / "clean.zip"
    _make_zip(
        source,
        [("data/train.csv", b"a,b\n", 0o100644), ("data/test.csv", b"c,d\n", 0o100644)],
    )
    dest = tmp_path / "out"
    dest.mkdir()
    count = extract_archive(source, dest, ExtractLimits())
    assert count == 2
    assert (dest / "data" / "train.csv").read_bytes() == b"a,b\n"
    assert (dest / "data" / "test.csv").read_bytes() == b"c,d\n"


def test_zip_path_traversal_rejected(tmp_path: Path) -> None:
    source = tmp_path / "evil.zip"
    _make_zip(source, [("../evil.txt", b"x", 0o100644)])
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", ExtractLimits())


def test_zip_absolute_path_rejected(tmp_path: Path) -> None:
    source = tmp_path / "evil.zip"
    _make_zip(source, [("/etc/evil.txt", b"x", 0o100644)])
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", ExtractLimits())


def test_zip_symlink_rejected(tmp_path: Path) -> None:
    source = tmp_path / "evil.zip"
    _make_zip(source, [("link.txt", b"", stat.S_IFLNK | 0o777)])
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", ExtractLimits())


def test_zip_bomb_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bomb.zip"
    # 1 MiB of zeros compresses to ~1 KiB → ratio way above any sane limit.
    _make_zip(source, [("big.txt", b"\x00" * (1024 * 1024), 0o100644)])
    limits = ExtractLimits(zip_bomb_ratio=50, zip_bomb_floor_bytes=1024)
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", limits)


def test_zip_too_many_files(tmp_path: Path) -> None:
    source = tmp_path / "many.zip"
    _make_zip(source, [(f"f{i}.txt", b"x", 0o100644) for i in range(3)])
    with pytest.raises(ArchiveTooLargeError):
        extract_archive(source, tmp_path / "out", ExtractLimits(max_files=2))


def test_zip_total_size_exceeded(tmp_path: Path) -> None:
    source = tmp_path / "big.zip"
    _make_zip(source, [("a.txt", b"12345", 0o100644), ("b.txt", b"67890", 0o100644)])
    with pytest.raises(ArchiveTooLargeError):
        extract_archive(source, tmp_path / "out", ExtractLimits(max_total_bytes=9))


def test_tar_path_traversal_rejected(tmp_path: Path) -> None:
    source = tmp_path / "evil.tar"
    _make_tar(source, "w", [("../evil.txt", b"x", "file")])
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", ExtractLimits())


def test_tar_symlink_rejected(tmp_path: Path) -> None:
    source = tmp_path / "evil.tar"
    _make_tar(source, "w", [("link.txt", b"", "symlink")])
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", ExtractLimits())


def test_targz_clean_extracts(tmp_path: Path) -> None:
    source = tmp_path / "clean.tar.gz"
    _make_tar(source, "w:gz", [("data/train.csv", b"a,b\n", "file")])
    dest = tmp_path / "out"
    dest.mkdir()
    count = extract_archive(source, dest, ExtractLimits())
    assert count == 1
    assert (dest / "data" / "train.csv").read_bytes() == b"a,b\n"


def test_unsupported_kind_rejected(tmp_path: Path) -> None:
    source = tmp_path / "binary.bin"
    source.write_bytes(b"\x00\x01\x02binary-with-nuls")
    with pytest.raises(ArchiveUnsafeError):
        extract_archive(source, tmp_path / "out", ExtractLimits())
