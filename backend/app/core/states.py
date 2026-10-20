"""Run state machine values [doc 03 §3.3, doc 04 §3 rule 6].

Full transition validation arrives with the Job Manager (slice 8);
these constants are what crash recovery and storage need today.
"""

from typing import Final, Literal

RunState = Literal[
    "queued",
    "preparing",
    "running",
    "paused",
    "evaluating",
    "completed",
    "failed",
    "stopped",
    "interrupted",
]

# States that must be marked `interrupted` by startup crash recovery [doc 04 §3 rule 6].
INTERRUPTIBLE_RUN_STATES: Final = frozenset({"preparing", "running", "evaluating"})
QUEUED: Final = "queued"
INTERRUPTED: Final = "interrupted"
