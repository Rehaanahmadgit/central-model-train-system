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


def _sample_system() -> None:
    """Emit CPU/RAM every N seconds for the live system-stats panel [doc 08 §6].

    Reads /proc directly (no psutil dependency). GPU stays None — the parent
    fills real GPU values on the GPU image, and the UI shows "No GPU detected".
    """
    import os

    proc = Path("/proc")
    clock = os.sysconf("SC_CLK_TCK")
    previous: tuple[int, int] | None = None
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
            cpu: float | None = None
            if previous is not None:
                delta = (utime + stime) - previous[0]
                cpu = round(100 * delta / (clock * _heartbeat_seconds), 1)
            previous = (utime + stime, previous[1] if previous else 0)
            emit(
                "system",
                {
                    "gpu_util": None,
                    "gpu_mem_mb": None,
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
        emit("system", {"gpu_util": None, "gpu_mem_mb": None, "cpu": cpu, "ram_mb": ram_mb})

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
