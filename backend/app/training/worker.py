"""Training subprocess side [doc 08 §3 step 5].

The worker never touches ``events.jsonl`` or ``run.json`` — the Job Manager in
the parent is the single writer [doc 04 §3 rule 3]. The worker talks to the
parent through two pipes:

* **stdout** — one JSON object per event (metric/log/state/heartbeat/result);
* **stdin** — control messages ``{"action":"pause"|"resume"|"stop"}``.

The plugin runs here and receives this ``RunContext``.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from app.core.device import DeviceInfo

# The worker only uses this process-local stdout; never the logging module,
# because its logs must be captured as run events, not app logs.
_stdout: TextIO
_pause_event = threading.Event()
_stop_event = threading.Event()
_control_thread: threading.Thread | None = None
_heartbeat_thread: threading.Thread | None = None
_system_thread: threading.Thread | None = None
_heartbeat_seconds = 5.0


def emit(event_type: str, data: dict[str, Any]) -> None:
    """Write one event to the parent on stdout."""
    payload: dict[str, Any] = {"type": event_type, "data": data}
    _stdout.write(json.dumps(payload, default=str) + "\n")
    _stdout.flush()


def _read_control(stream: TextIO) -> None:
    """Read control messages from the parent on stdin."""
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        action = message.get("action") if isinstance(message, dict) else None
        if action == "pause":
            _pause_event.set()
        elif action == "resume":
            _pause_event.clear()
        elif action == "stop":
            _stop_event.set()
            _pause_event.clear()  # unblock wait_if_paused so stop wins


def _heartbeat() -> None:
    """Emit a heartbeat every N seconds so the parent can detect a dead worker [doc 08 §5]."""
    while not _stop_event.is_set():
        emit("heartbeat", {"ts": datetime.now().isoformat(timespec="milliseconds")})
        time.sleep(_heartbeat_seconds)


def gpu_available() -> bool:
    """True when this process can use CUDA [doc 14 §3.5]; never raises."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch absent or CUDA init failed
        return False


def sample_gpu() -> tuple[float | None, float | None]:
    """Return (gpu_util_percent, gpu_mem_mb) or (None, None).

    Never raises [doc 14 §3.5]. When torch has no CUDA device (CPU image, or a
    GPU container without passthrough) it returns (None, None) so the UI keeps
    showing "No GPU detected" — even if the host has nvidia-smi, because a
    CPU run is not using that GPU. With CUDA available: memory from
    torch.cuda.mem_get_info(), util from `nvidia-smi` via a short subprocess.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
    except Exception:  # pragma: no cover - defensive; sampler must never crash
        return None, None

    mem_mb: float | None = None
    util_pct: float | None = None
    try:
        total, free = torch.cuda.mem_get_info()
        mem_mb = round((total - free) / 1024**2, 1)
    except Exception:  # pragma: no cover - defensive; any failure => null
        mem_mb = None
    try:
        import shutil
        import subprocess

        executable = shutil.which("nvidia-smi")
        if executable is not None:
            output = subprocess.run(  # noqa: S603 - fixed executable, no shell
                [executable, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            first_line = output.stdout.strip().splitlines()[0] if output.stdout.strip() else ""
            if first_line:
                util_pct = round(float(first_line.strip()), 1)
    except Exception:  # pragma: no cover - defensive; any failure => null
        util_pct = None
    return util_pct, mem_mb


def _sample_system() -> None:
    """Emit CPU/RAM/GPU every N seconds for the live system-stats panel [doc 08 §6].

    Reads /proc directly (no psutil dependency). GPU values come from
    ``sample_gpu()`` (torch.cuda + nvidia-smi); absent means null, never a crash.
    The first sample is emitted immediately so even a run that finishes within
    one heartbeat interval leaves one system event behind.
    """
    import os

    proc = Path("/proc")
    clock = os.sysconf("SC_CLK_TCK")
    # Baseline at worker start, so the very first sample already has a real CPU %.
    try:
        stat = (proc / "self" / "stat").read_text().split()
        baseline_ticks = int(stat[13]) + int(stat[14])
    except (OSError, IndexError, ValueError):  # pragma: no cover - defensive
        baseline_ticks = 0
    baseline_wall = time.monotonic()
    while not _stop_event.is_set():
        try:
            stat = (proc / "self" / "stat").read_text().split()
            utime = int(stat[13])
            stime = int(stat[14])
            status = (proc / "self" / "status").read_text()
            ram_kb = 0
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    ram_kb = int(line.split()[1])
                    break
            elapsed = time.monotonic() - baseline_wall
            cpu: float | None = None
            if baseline_ticks > 0 and elapsed > 0:
                delta = (utime + stime) - baseline_ticks
                cpu = round(min(100 * delta / (clock * elapsed), 100.0), 1)
            gpu_util, gpu_mem_mb = sample_gpu()
            emit(
                "system",
                {
                    "gpu_available": gpu_available(),
                    "gpu_util": gpu_util,
                    "gpu_mem_mb": gpu_mem_mb,
                    "cpu": cpu,
                    "ram_mb": round(ram_kb / 1024, 1),
                },
            )
        except (OSError, IndexError, ValueError):  # pragma: no cover - defensive
            pass
        time.sleep(_heartbeat_seconds)


class WorkerRunContext:
    """RunContext passed to plugins; writes events to stdout [doc 08 §1]."""

    def __init__(
        self,
        *,
        home: Path,
        run_id: str,
        dataset_hash: str,
        model_family: str,
        task_type: str | None,
        dataset_root: Path,
        manifest_path: Path,
        facts: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
        checkpoint_path: Path | None,
        device: DeviceInfo,
        seed: int,
    ) -> None:
        self.home = home
        self.run_id = run_id
        self.dataset_hash = dataset_hash
        self.model_family = model_family
        self.task_type = task_type
        self.dataset_root = dataset_root
        self.manifest_path = manifest_path
        self.facts = facts
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.seed = seed

    # -- emitters -----------------------------------------------------------

    def emit_metric(self, name: str, value: float, step: int, epoch: int, split: str) -> None:
        emit(
            "metric",
            {
                "name": name,
                "value": round(float(value), 6),
                "step": step,
                "epoch": epoch,
                "split": split,
            },
        )

    def emit_log(self, level: str, message: str) -> None:
        emit("log", {"level": level, "message": message})

    def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        emit("event", {"name": event_type, **payload})

    def emit_system(self, cpu: float | None, ram_mb: float | None) -> None:
        emit(
            "system",
            {
                "gpu_available": gpu_available(),
                "gpu_util": None,
                "gpu_mem_mb": None,
                "cpu": cpu,
                "ram_mb": ram_mb,
            },
        )

    def save_checkpoint(self, path: Path, is_best: bool, metrics: dict[str, Any]) -> None:
        emit(
            "event",
            {
                "name": "checkpoint_saved",
                "path": str(path),
                "is_best": is_best,
                "metrics": metrics,
            },
        )

    # -- control ------------------------------------------------------------

    def should_pause(self) -> bool:
        return _pause_event.is_set() and not _stop_event.is_set()

    def should_stop(self) -> bool:
        return _stop_event.is_set()

    def wait_if_paused(self) -> None:
        if not self.should_pause():
            return
        emit("state", {"from": "running", "to": "paused"})
        while self.should_pause():
            time.sleep(0.1)
        if not self.should_stop():
            emit("state", {"from": "paused", "to": "running"})


def start_control_and_heartbeat(stdin: TextIO, stdout: TextIO, *, heartbeat_seconds: float) -> None:
    """Wire the module-level pipes; called once by the worker entry point."""
    global _stdout, _control_thread, _heartbeat_thread, _system_thread, _heartbeat_seconds
    _stdout = stdout
    _heartbeat_seconds = heartbeat_seconds
    _pause_event.clear()
    _stop_event.clear()
    _control_thread = threading.Thread(target=_read_control, args=(stdin,), daemon=True)
    _heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    _system_thread = threading.Thread(target=_sample_system, daemon=True)
    _control_thread.start()
    _heartbeat_thread.start()
    _system_thread.start()
