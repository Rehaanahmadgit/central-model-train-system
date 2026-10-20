"""Check results and the shared verification context [doc 06 §2]."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CheckSeverity = Literal["error", "warn", "info"]
CheckStatus = Literal["pass", "warn", "fail", "skipped", "running"]


@dataclass
class CheckResult:
    """One check outcome: name, severity, status, message, details, fix_hint [doc 06 §2]."""

    name: str
    severity: CheckSeverity
    status: CheckStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    fix_hint: str | None = None
    samples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "fix_hint": self.fix_hint,
            "samples": self.samples,
        }


@dataclass
class VerificationContext:
    """Everything a check may need; facts are shared between checks."""

    dataset_root: Path
    dataset_hash: str
    model_family: str
    task_type: str | None
    verifier_version: str
    sample_cap: int
    facts: dict[str, Any] = field(default_factory=dict)
