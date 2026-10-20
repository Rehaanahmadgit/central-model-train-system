"""ID generation [doc 04 §4].

- Run ids: ``run_{YYYYMMDD}_{HHMMSS}_{4 random hex}``
- Verification ids: ``ver_{YYYYMMDD}_{HHMMSS}_{4 random hex}`` (D11)
- Upload ids: random UUID4
No sequential numeric ids, no user-derived ids.
"""

import secrets
import uuid
from datetime import datetime


def _stamp(now: datetime) -> str:
    return now.strftime("%Y%m%d_%H%M%S")


def _hex4() -> str:
    return secrets.token_hex(2)  # 4 hex characters


def new_run_id(now: datetime | None = None) -> str:
    return f"run_{_stamp(now or datetime.now())}_{_hex4()}"


def new_verification_id(now: datetime | None = None) -> str:
    # doc 04 §4 leaves the `ver_...` suffix open; it mirrors the run-id format (D11).
    return f"ver_{_stamp(now or datetime.now())}_{_hex4()}"


def new_project_id(now: datetime | None = None) -> str:
    # Project id format is NOT SPECIFIED in doc 04 §4; mirrors the run-id
    # pattern and stays non-sequential and not user-derived (D32).
    return f"proj_{_stamp(now or datetime.now())}_{_hex4()}"


def new_upload_id() -> str:
    return str(uuid.uuid4())
