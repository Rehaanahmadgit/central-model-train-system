"""JSONL appender / reader / tail-repair tests [doc 04 §3 rules 2+6, doc 11 §3b]."""

from pathlib import Path

import pytest

from app.storage.jsonl import JsonlAppender, JsonlParseError, read_jsonl, repair_tail


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlAppender(path) as appender:
        appender.append({"seq": 1, "type": "metric"})
        appender.append({"seq": 2, "type": "log"})
    assert list(read_jsonl(path)) == [
        {"seq": 1, "type": "metric"},
        {"seq": 2, "type": "log"},
    ]


def test_flushes_every_n_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlAppender(path, flush_lines=2) as appender:
        appender.append({"seq": 1})
        assert path.read_text() == ""  # buffered, not yet flushed
        appender.append({"seq": 2})
        assert path.read_text().strip()  # flush every 2 lines


def test_sync_fsyncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def counting_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("app.storage.jsonl.os.fsync", counting_fsync)
    path = tmp_path / "events.jsonl"
    appender = JsonlAppender(path)
    appender.append({"seq": 1})
    appender.sync()
    appender.close()
    monkeypatch.undo()
    assert calls == 2  # sync() + close()


def test_read_jsonl_skips_half_written_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1}\n{"seq":2}\n{"seq":3')
    assert list(read_jsonl(path)) == [{"seq": 1}, {"seq": 2}]


def test_read_jsonl_raises_on_malformed_nonfinal_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1}\nnot json\n{"seq":3}\n')
    with pytest.raises(JsonlParseError):
        list(read_jsonl(path))


def test_read_jsonl_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(read_jsonl(tmp_path / "missing.jsonl")) == []


def test_repair_tail_truncates_half_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1}\n{"seq":2}\n{"seq":3')
    removed = repair_tail(path)
    assert removed == len('{"seq":3')
    assert path.read_text() == '{"seq":1}\n{"seq":2}\n'


def test_repair_tail_clean_file_removes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1}\n{"seq":2}\n')
    assert repair_tail(path) == 0
    assert path.read_text() == '{"seq":1}\n{"seq":2}\n'


def test_repair_tail_invalid_whole_file_truncates_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1')
    assert repair_tail(path) == len('{"seq":1')
    assert path.read_bytes() == b""


def test_repair_tail_missing_file_returns_zero(tmp_path: Path) -> None:
    assert repair_tail(tmp_path / "missing.jsonl") == 0


def test_appender_resumes_after_repair(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq":1}\n{"seq":2')
    repair_tail(path)
    with JsonlAppender(path) as appender:
        appender.append({"seq": 2})
    records = list(read_jsonl(path))
    assert [r["seq"] for r in records] == [1, 2]


def test_append_after_close_raises(tmp_path: Path) -> None:
    appender = JsonlAppender(tmp_path / "events.jsonl")
    appender.close()
    with pytest.raises(ValueError):
        appender.append({"seq": 1})
