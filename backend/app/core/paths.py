"""Filesystem layout and path confinement inside TRAINHUB_HOME [doc 04 §1, doc 10 §5]."""

from pathlib import Path

# Standard workspace layout [doc 04 §1]. Created at startup; empty dirs are harmless.
LAYOUT_DIRS: tuple[str, ...] = (
    "projects",
    "datasets",
    "verifications",
    "cache/verif",
    "checkpoints/store",
    "checkpoints/.tmp",
    "checkpoints/locks",
    "runs",
    "models",
    "audit",
    "locks",
    "tmp/uploads",
)


def ensure_workspace(home: Path) -> None:
    """Create the standard directory layout under the workspace root [doc 04 §1]."""
    for relative in LAYOUT_DIRS:
        (home / relative).mkdir(parents=True, exist_ok=True)


def resolve_within(home: Path, *parts: str) -> Path:
    """Join parts under the workspace and reject anything that escapes it [doc 10 §5].

    Every path built from request parameters must go through here (doc 13 §3).
    """
    root = home.resolve()
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("path escapes TRAINHUB_HOME")
    return candidate
