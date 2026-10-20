"""Privacy baseline test [doc 10 §1, doc 01 §10.8].

Scans every text file under TRAINHUB_HOME for email-like strings, IPv4-like
strings and user-agent mentions. The full container-log grep runs at MVP
acceptance; this test keeps the invariant from slice 1.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

# A real-looking address: local part, single @, dotted domain with a 2+ letter TLD.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Valid IPv4 octets only, not digits joined by dots (avoids "1.0.0" style versions).
IPV4_RE = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
)
# Full IPv6 forms (heuristic: at least two ":" groups and only hex/colon chars).
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?![\w:])")
USER_AGENT_RE = re.compile(r"user[-_ ]agent", re.IGNORECASE)


def iter_text_files(root: Path) -> Iterator[tuple[Path, str]]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        try:
            yield path, data.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary files are not text


def test_scanner_catches_planted_personal_data(tmp_path: Path) -> None:
    planted = tmp_path / "planted.json"
    planted.write_text(
        'contact a@b.co from 192.168.1.10 or 2001:db8::1 via "User-Agent: probe"'
    )
    texts = [text for _, text in iter_text_files(tmp_path)]
    assert any(EMAIL_RE.search(text) for text in texts)
    assert any(IPV4_RE.search(text) for text in texts)
    assert any(IPV6_RE.search(text) for text in texts)
    assert any(USER_AGENT_RE.search(text) for text in texts)


def test_workspace_contains_no_personal_data(client: TestClient, home: Path) -> None:
    # Exercise the app first so the workspace has real content.
    client.get("/readyz")
    for path, text in iter_text_files(home):
        assert not EMAIL_RE.search(text), f"email-like string in {path}"
        assert not IPV4_RE.search(text), f"ipv4-like string in {path}"
        assert not IPV6_RE.search(text), f"ipv6-like string in {path}"
        assert not USER_AGENT_RE.search(text), f"user-agent mention in {path}"
