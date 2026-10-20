"""``trainhub doctor`` — plain-words environment check [doc 14 §3.6, DIST-05].

Tells a stranger what is wrong (no GPU, no toolkit, disk full, no internet for
checkpoints, unwritable /data) and how to fix it. Pure functions so it is
testable without torch or a GPU.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.checkpoints.catalog import load_catalog
from app.core.device import resolve_device

# Hosts we must be able to reach to download checkpoints. Unverified catalog
# entries (TODO(user) placeholders) are skipped: there is no real URL to test.
_CHECKPOINT_TIMEOUT = 5.0


@dataclass
class DoctorCheck:
    name: str
    status: str  # "ok" | "warn" | "fail"
    message: str
    fix: str | None = None


@dataclass
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def add(self, name: str, status: str, message: str, fix: str | None = None) -> None:
        self.checks.append(DoctorCheck(name, status, message, fix))

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [check.__dict__ for check in self.checks]}


def _python_version() -> str:
    return sys.version.split()[0]


def _torch_version() -> tuple[str | None, bool, str]:
    """Return (version, cuda_available, reason)."""
    if importlib.util.find_spec("torch") is None:
        return None, False, "torch is not installed (CPU-only image; fine for CPU presets)"
    try:
        import torch

        version = str(torch.__version__)
        try:
            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
        return version, cuda, f"torch {version} installed"
    except Exception as exc:  # pragma: no cover - defensive
        return None, False, f"torch import failed: {type(exc).__name__}"


def check_disk(home: Path) -> tuple[str, str, str | None]:
    """Free disk: warn below the alert threshold, fail below 2 GiB [doc 10 §7]."""
    probe = home
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return "fail", f"cannot read free disk in {home}: {exc}", None
    free_gb = usage.free / 1024**3
    if usage.free < 2 * 1024**3:
        return (
            "fail",
            f"only {free_gb:.1f} GB free in {home}",
            "Free up disk space on the volume, or move the workspace to a bigger disk.",
        )
    if usage.free / usage.total < 0.15:
        return (
            "warn",
            f"{free_gb:.1f} GB free ({usage.free / usage.total:.0%}) — running low",
            "Add disk soon; training and checkpoints need space.",
        )
    return "ok", f"{free_gb:.1f} GB free in {home}", None


def check_write(home: Path) -> tuple[str, str, str | None]:
    """Write permission on the workspace, with the uid-10001 bind-mount hint."""
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        uid = os.getuid() if hasattr(os, "getuid") else None
        return (
            "fail",
            f"cannot write to {home} (running as uid {uid}): {exc}",
            (
                "The /data volume is owned by another user. If you used a bind mount "
                "(docker run -v /host/dir:/data), make that dir writable by the "
                "container user (uid 10001): `mkdir -p /host/dir && "
                "sudo chown 10001:10001 /host/dir`, or use a named volume "
                "(`-v trainhub_data:/data`), which Docker sets up on its own."
            ),
        )
    return "ok", f"{home} is writable", None


def check_part_files(home: Path) -> tuple[str, str, str | None]:
    """Leftover .part downloads from an interrupted transfer [doc 14 §3.6]."""
    parts = sorted((home / "checkpoints" / ".tmp").glob("*.part"))
    if parts:
        return (
            "warn",
            f"{len(parts)} leftover .part download file(s) in checkpoints/.tmp",
            "They are stale. Remove them with: "
            "find $TRAINHUB_HOME/checkpoints/.tmp -name '*.part' -delete",
        )
    return "ok", "no leftover .part files", None


def check_checkpoint_hosts(home: Path) -> tuple[str, str, str | None]:
    """Can we reach the checkpoint hosts in the catalog? [doc 14 §3.6]"""
    entries = load_catalog(home)
    hosts: set[str] = set()
    for entry in entries:
        if not entry.verified or "TODO(user)" in (entry.source_uri or ""):
            continue
        parsed = urllib.parse.urlparse(entry.source_uri)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            hosts.add(parsed.hostname)
    if not hosts:
        return (
            "warn",
            "no verified checkpoint URLs in catalog.yaml, so reachability was not tested",
            "Fill real URLs + sha256 in checkpoints/catalog.override.yaml when you deploy.",
        )
    unreachable: list[str] = []
    for host in sorted(hosts):
        try:
            with socket.create_connection((host, 443), timeout=_CHECKPOINT_TIMEOUT):
                pass
        except OSError:
            unreachable.append(host)
    if unreachable:
        return (
            "fail",
            f"cannot reach checkpoint host(s): {', '.join(unreachable)}",
            "Check outbound HTTPS (firewall/proxy) on this server.",
        )
    return "ok", f"checkpoint host(s) reachable: {', '.join(sorted(hosts))}", None


def run_doctor(home: Path) -> DoctorReport:
    report = DoctorReport()

    # Python / torch / CUDA
    python_version = _python_version()
    torch_version, cuda_available, torch_reason = _torch_version()
    device = resolve_device("auto")
    report.add("python", "ok", f"Python {python_version}")
    if torch_version is None:
        report.add(
            "torch",
            "warn",
            torch_reason,
            "CPU-only presets still work; install torch to train.",
        )
    else:
        report.add(
            "torch",
            "ok" if not cuda_available else "ok",
            f"{torch_reason} — CUDA {'available' if cuda_available else 'not available'}",
        )

    # GPU
    if device.is_gpu:
        report.add("gpu", "ok", f"GPU detected: {device.name}")
    else:
        report.add(
            "gpu",
            "warn",
            f"no GPU detected ({device.reason})",
            "For large detection/classification models, run the -gpu image on a "
            "CUDA server; for a quick test use a CPU dev preset [doc 14 §3.3].",
        )

    # Disk + write + .part + hosts
    status, message, fix = check_disk(home)
    report.add("disk", status, message, fix)
    status, message, fix = check_write(home)
    report.add("write_permission", status, message, fix)
    status, message, fix = check_part_files(home)
    report.add("partial_downloads", status, message, fix)
    status, message, fix = check_checkpoint_hosts(home)
    report.add("checkpoint_hosts", status, message, fix)

    return report


def format_report(report: DoctorReport) -> str:
    """Plain-words output for the terminal."""
    lines: list[str] = ["TrainHub doctor", "==============="]
    for check in report.checks:
        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[check.status]
        lines.append(f"[{icon}] {check.name}: {check.message}")
        if check.fix:
            lines.append(f"      fix: {check.fix}")
    lines.append("===============")
    lines.append("All good." if report.ok else "Problems found — fix the FAIL items above.")
    return "\n".join(lines)
