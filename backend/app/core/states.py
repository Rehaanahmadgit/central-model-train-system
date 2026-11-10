"""Run state machine [doc 03 §3.3, doc 08 §3.8, doc 04 §3 rule 6].

Valid transitions and the failure classifier live in exactly one place so the
Job Manager, crash recovery and the API cannot drift apart.
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

INTERRUPTIBLE_RUN_STATES: Final = frozenset({"preparing", "running", "evaluating"})
QUEUED: Final = "queued"
INTERRUPTED: Final = "interrupted"

# state -> states the job may move into next [doc 08 §3].
VALID_TRANSITIONS: Final = {
    "queued": {"preparing", "failed", "interrupted"},
    "preparing": {"running", "failed", "interrupted"},
    "running": {"paused", "evaluating", "completed", "failed", "stopped", "interrupted"},
    "paused": {"running", "stopped", "interrupted"},
    "evaluating": {"completed", "failed", "interrupted"},
    "completed": set(),
    "failed": set(),
    "stopped": set(),
    "interrupted": {"queued"},  # requeue/resume paths
}

TERMINAL_STATES: Final = frozenset({"completed", "failed", "stopped"})

# Failure reasons [doc 08 §3.8]. A user-requested stop is NOT a failure (D5):
# it ends in `stopped` with `failure: null`.
FAILURE_REASONS: Final = frozenset(
    {
        "OOM",
        "DATA_ERROR",
        "NAN_LOSS",
        "WORKER_LOST",
        "GPU_UNAVAILABLE",
        "UNKNOWN",
        # checkpoint resolution at run start [Slice D]
        "LICENCE_NOT_ACCEPTED",
        "CHECKPOINT_OFFLINE",
        "CHECKPOINT_PARTIAL",
        "CHECKPOINT_DOWNLOAD_FAILED",
        "CHECKPOINT_SHA_MISMATCH",
        "CHECKPOINT_DISK_FULL",
        "CHECKPOINT_SOURCE_BLOCKED",
    }
)


def can_transition(state: str, to: str) -> bool:
    """Whether ``state -> to`` is allowed; unknown states are rejected."""
    return state in VALID_TRANSITIONS and to in VALID_TRANSITIONS[state]


def transition(state: str, to: str) -> str:
    """Return ``to`` if valid, else raise ValueError."""
    if not can_transition(state, to):
        raise ValueError(f"invalid run transition {state!r} -> {to!r}")
    return to


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES
