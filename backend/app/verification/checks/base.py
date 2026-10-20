"""Check base class [doc 06 §2].

A check declares its name, severity and dependencies; a failing
error-severity check short-circuits dependent checks (marked `skipped`).
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from app.verification.context import CheckResult, CheckSeverity, VerificationContext


class Check(ABC):
    name: ClassVar[str]
    severity: ClassVar[CheckSeverity]
    depends_on: ClassVar[list[str]] = []

    @abstractmethod
    def run(self, ctx: VerificationContext) -> CheckResult:
        """Execute the check and return its result."""


def skipped_result(name: str, dependency: str) -> CheckResult:
    return CheckResult(
        name=name,
        severity="info",
        status="skipped",
        message=f"skipped: dependency {dependency!r} failed",
    )
