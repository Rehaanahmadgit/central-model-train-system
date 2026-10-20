"""Statistical checks: class distribution [doc 06 §4].

Imbalance thresholds: min class < 2% of samples OR ratio > 20:1 → warn
with concrete suggestions (weighted loss, resampling, augmentation).
"""

from app.verification.checks.base import Check
from app.verification.context import CheckResult, VerificationContext


class ClassDistributionCheck(Check):
    name = "class_distribution"
    severity = "warn"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        counts = ctx.facts.get("class_counts") or {}
        if not counts:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no class counts available for this dataset",
            )
        total = sum(counts.values())
        values = [int(value) for value in counts.values()]
        minimum = min(values)
        maximum = max(values)
        details = {"counts": dict(counts), "total": total}
        if minimum <= 0:
            imbalance = True
        else:
            imbalance = minimum < 0.02 * total or maximum / minimum > 20
        if not imbalance:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message=f"{len(counts)} classes, {total} samples, distribution is balanced",
                details=details,
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="warn",
            message=(
                f"class imbalance: min class is {minimum} sample(s), "
                f"max is {maximum} (ratio {maximum / max(minimum, 1):.1f}:1)"
            ),
            details=details,
            fix_hint="Consider weighted loss, resampling or augmentation for the minority class.",
        )
