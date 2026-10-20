"""ID generation tests [doc 04 §4]."""

import re
import uuid
from datetime import datetime, timedelta

from app.storage.ids import new_run_id, new_upload_id, new_verification_id

RUN_ID_RE = re.compile(r"^run_\d{8}_\d{6}_[0-9a-f]{4}$")
VER_ID_RE = re.compile(r"^ver_\d{8}_\d{6}_[0-9a-f]{4}$")


def test_run_id_format_and_timestamp() -> None:
    now = datetime(2026, 10, 17, 11, 34, 0)
    run_id = new_run_id(now)
    assert RUN_ID_RE.match(run_id)
    assert run_id.startswith("run_20261017_113400_")


def test_verification_id_format() -> None:
    now = datetime(2026, 10, 17, 11, 34, 0)
    verification_id = new_verification_id(now)
    assert VER_ID_RE.match(verification_id)
    assert verification_id.startswith("ver_20261017_113400_")


def test_upload_id_is_uuid4() -> None:
    upload_id = new_upload_id()
    assert uuid.UUID(upload_id).version == 4


def test_run_ids_are_unique_across_seconds() -> None:
    # The spec format carries only 4 random hex chars [doc 04 §4], so
    # uniqueness comes from the timestamp; ids created at different times differ.
    ids = {
        new_run_id(datetime(2026, 10, 17, 12, 0, 0) + timedelta(seconds=i))
        for i in range(1000)
    }
    assert len(ids) == 1000
