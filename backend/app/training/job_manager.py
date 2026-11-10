"""In-process Job Manager with supervised training subprocesses [doc 08 §3, FR-7].

The parent owns one ``events.jsonl`` writer and ``run.json`` writer per run
[doc 04 §3 rule 3]; the worker only streams events over its stdout pipe. The
parent monitors the heartbeat, forwards events to the broadcaster, and enforces
the state machine.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.checkpoints.resolver import CheckpointError, CheckpointResolver
from app.core.config import Settings
from app.core.device import DeviceInfo, probe_device
from app.core.states import can_transition, transition
from app.realtime.broadcaster import EventBroadcaster
from app.realtime.events import RunEventLog
from app.storage.atomic import atomic_write_json, read_json
from app.storage.audit import append_audit
from app.storage.locks import FileLock, LockTimeout
from app.training import registry

HEARTBEAT = "heartbeat"


class JobNotFoundError(RuntimeError):
    pass


class _JobHandle:
    """Parent-side state for one running training subprocess."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        run_id: str,
        home: Path,
        broadcaster: EventBroadcaster,
        lost_seconds: float,
    ) -> None:
        self.process = process
        self.run_id = run_id
        self.home = home
        self.broadcaster = broadcaster
        self.lost_seconds = lost_seconds
        self.log = RunEventLog(home, run_id)
        self.last_heartbeat = time.monotonic()
        self.terminal: str | None = None
        self.failure: dict[str, Any] | None = None
        self.best: dict[str, Any] = {}
        self.finalized = False
        self.gpu_index: int | None = None
        self.stdout_task: asyncio.Task[Any] | None = None
        self.stderr_task: asyncio.Task[Any] | None = None
        self.monitor_task: asyncio.Task[Any] | None = None
        self.waiter_task: asyncio.Task[Any] | None = None


class JobManager:
    """Spawns, supervises and controls training subprocesses [doc 08 §3]."""

    def __init__(self, home: Path, settings: Settings, broadcaster: EventBroadcaster) -> None:
        self._home = home
        self._settings = settings
        self._broadcaster = broadcaster
        self._handles: dict[str, _JobHandle] = {}
        # CPU concurrency limit [doc 14 §3.2]; GPU runs use per-GPU file locks
        # instead, so a CPU job never eats a GPU slot and vice versa.
        self._slots = asyncio.Semaphore(max(1, settings.max_cpu_jobs))
        self._gpu_locks: dict[int, FileLock] = {}

    # -- public API ---------------------------------------------------------

    async def start(self, run_id: str) -> None:
        """Claim a queued run: preparing → (checkpoint) → spawn → running."""
        record = self._read_run(run_id)
        if record is None:
            raise JobNotFoundError(run_id)
        state = str(record.get("state") or "queued")
        if state != "queued" or not can_transition(state, "preparing"):
            return
        family = str(record.get("model_family") or "")
        raw_config = record.get("config")
        config: dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}
        # dev_preset is the CPU-capable path for GPU-only families [doc 14 §3.4]:
        # a dev_preset run uses the CPU semaphore and never demands a GPU.
        needs_gpu = bool(registry.METADATA.get(family, {}).get("needs_gpu")) and not bool(
            config.get("dev_preset")
        )
        gpu_index: int | None = None
        if needs_gpu:
            # Clear error instead of a silent CPU fallback [doc 14 §3.3].
            # Subprocess probe: the API process must never import torch (Slice H).
            device = DeviceInfo.from_dict(
                await asyncio.to_thread(probe_device, self._settings.device)
            )
            if not device.is_gpu:
                self._set_state(
                    run_id,
                    "failed",
                    failure={"reason": "GPU_UNAVAILABLE", "details": device.reason},
                )
                return
            gpu_index = device.index if device.index is not None else 0
            lock = self._gpu_locks.setdefault(
                gpu_index,
                FileLock(self._home / "locks" / f"gpu-{gpu_index}.lock", timeout=0.2),
            )
            try:
                await asyncio.to_thread(lock.acquire)
            except LockTimeout:
                # One job per GPU; stay queued and retry when a GPU job finishes.
                return
        else:
            await self._slots.acquire()
        try:
            self._set_state(run_id, "preparing", record)
            probe = await asyncio.to_thread(probe_device, self._settings.device)
            env = self._capture_env(record, probe)
            self._write_env(run_id, env)
            await self._resolve_checkpoint(run_id, record)
            process = await self._spawn(run_id, gpu_index=gpu_index)
        except CheckpointError as exc:
            # Clear, actionable failure (offline, sha mismatch, disk full, licence...)
            # instead of UNKNOWN; the UI offers "Import from file" [Slice D].
            self._release_slot(gpu_index)
            self._set_state(
                run_id,
                "failed",
                failure={
                    "reason": exc.code,
                    "details": exc.message,
                    "catalog_id": exc.details.get("catalog_id") or _catalog_id_of(record),
                    "can_import": bool(exc.details.get("can_import")),
                },
            )
            return
        except BaseException:
            self._release_slot(gpu_index)
            self._set_state(run_id, "failed", failure={"reason": "UNKNOWN"})
            raise
        handle = _JobHandle(
            process, run_id, self._home, self._broadcaster, self._settings.worker_lost_seconds
        )
        handle.gpu_index = gpu_index
        self._handles[run_id] = handle
        handle.log.append(
            "log",
            {"level": "INFO", "message": f"training subprocess started (pid={process.pid})"},
        )
        handle.stdout_task = asyncio.create_task(self._pump_stdout(handle))
        handle.stderr_task = asyncio.create_task(self._pump_stderr(handle))
        handle.monitor_task = asyncio.create_task(self._monitor(handle))
        handle.waiter_task = asyncio.create_task(self._wait(handle))

    async def pause(self, run_id: str) -> None:
        await self._control(run_id, "pause")

    async def resume(self, run_id: str) -> None:
        await self._control(run_id, "resume")

    async def stop(self, run_id: str) -> None:
        await self._control(run_id, "stop")

    def is_active(self, run_id: str) -> bool:
        return run_id in self._handles

    async def shutdown(self) -> None:
        """Stop accepting work; ask each child to stop, then wait briefly."""
        for run_id in list(self._handles):
            try:
                await self.stop(run_id)
            except Exception:  # pragma: no cover - best effort on shutdown
                pass
        for handle in list(self._handles.values()):
            if handle.process.returncode is None:
                try:
                    await asyncio.wait_for(handle.process.wait(), timeout=5)
                except TimeoutError:
                    _kill_group(handle.process)

    # -- internals ----------------------------------------------------------

    def _read_run(self, run_id: str) -> dict[str, Any] | None:
        path = self._home / "runs" / run_id / "run.json"
        if not path.is_file():
            return None
        data = read_json(path)
        return data if isinstance(data, dict) else None

    def _set_state(
        self,
        run_id: str,
        to: str,
        current: dict[str, Any] | None = None,
        *,
        failure: dict[str, Any] | None = None,
    ) -> None:
        record = current if current is not None else self._read_run(run_id)
        if record is None:
            return
        from_state = str(record.get("state") or "queued")
        if from_state == to:
            return
        if not can_transition(from_state, to):
            return
        transition(from_state, to)  # validates; side-effect free
        record["state"] = to
        if failure is not None:
            record["failure"] = {
                "reason": failure.get("reason", "UNKNOWN"),
                "details": failure.get("details"),
            }
            for extra in ("catalog_id", "can_import"):
                if extra in failure:
                    record["failure"][extra] = failure[extra]
            record.pop("waiting_for", None)
        elif to == "stopped":
            record["failure"] = {"reason": None, "details": None}
        if to in ("completed", "failed", "stopped"):
            record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        if to in ("running", "paused"):
            record["started_at"] = record.get("started_at") or datetime.now(UTC).isoformat(
                timespec="seconds"
            )
        path = self._home / "runs" / run_id / "run.json"
        with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
            atomic_write_json(path, record)

    def _capture_env(self, record: dict[str, Any], probe: dict[str, object]) -> dict[str, Any]:
        del record
        # Versions come from the device-probe subprocess: the API process stays torch-free.
        torch_version = probe.get("torch_version")
        cuda_version = probe.get("cuda_version")
        return {
            "python": sys.version.split()[0],
            "torch": torch_version if isinstance(torch_version, str) else None,
            "cuda": cuda_version if isinstance(cuda_version, str) else None,
            "device": self._settings.device,
            "git_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "platform": sys.platform,
        }

    def _write_env(self, run_id: str, env: dict[str, Any]) -> None:
        path = self._home / "runs" / run_id / "run.json"
        record = read_json(path)
        if isinstance(record, dict):
            record["env"] = env
            with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
                atomic_write_json(path, record)

    async def _resolve_checkpoint(self, run_id: str, record: dict[str, Any]) -> None:
        """Resolve a pretrained checkpoint at start, only when one is requested.

        Tiny trainers (dev_preset) set ``checkpoint_catalog_id`` to null and
        train from scratch, so nothing is downloaded [doc 14 §3.4].
        """
        checkpoint = record.get("checkpoint") or {}
        catalog_id = checkpoint.get("catalog_id") if isinstance(checkpoint, dict) else None
        if not catalog_id:
            return
        self._log(run_id, "INFO", f"resolving checkpoint {catalog_id} before training")
        throttle = {"at": 0.0}

        def on_progress(event: dict[str, Any]) -> None:
            kind = str(event.get("event"))
            now = time.monotonic()
            if kind == "download_progress" and now - throttle["at"] < 1.0:
                return
            throttle["at"] = now
            status = {
                "waiting_for_lock": "waiting_lock",
                "download_progress": "downloading",
                "cache_hit": "ready",
            }.get(kind)
            if status is None:
                return
            self._write_waiting(
                run_id,
                {
                    "status": status,
                    "catalog_id": str(catalog_id),
                    "percent": event.get("percent"),
                    "downloaded_bytes": event.get("downloaded_bytes"),
                    "total_bytes": event.get("total_bytes"),
                },
            )

        # "waiting: checkpoint" is visible while the run is still `preparing`.
        self._write_waiting(
            run_id, {"status": "resolving", "catalog_id": str(catalog_id), "percent": None}
        )
        resolver = CheckpointResolver(self._home, progress=on_progress)
        resolved = await asyncio.to_thread(resolver.resolve, str(catalog_id))
        path = self._home / "runs" / run_id / "run.json"
        data = read_json(path)
        if isinstance(data, dict):
            data.pop("waiting_for", None)
            data["checkpoint"] = {
                "catalog_id": str(catalog_id),
                "sha256": resolved.sha256,
                "source": resolved.source,
                "path": str(resolved.path),
            }
            with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
                atomic_write_json(path, data)

    def _write_waiting(self, run_id: str, waiting: dict[str, Any]) -> None:
        path = self._home / "runs" / run_id / "run.json"
        with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
            data = read_json(path)
            if isinstance(data, dict):
                data["waiting_for"] = {"what": "checkpoint", **waiting}
                atomic_write_json(path, data)

    def _write_best(self, run_id: str, best: dict[str, Any]) -> None:
        path = self._home / "runs" / run_id / "run.json"
        record = read_json(path)
        if isinstance(record, dict):
            record["best"] = best
            with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
                atomic_write_json(path, record)

    def _log(self, run_id: str, level: str, message: str) -> None:
        handle = self._handles.get(run_id)
        if handle is not None:
            handle.log.append("log", {"level": level, "message": message})

    async def _spawn(
        self, run_id: str, *, gpu_index: int | None = None
    ) -> asyncio.subprocess.Process:
        command = [
            sys.executable,
            "-m",
            "app.training.worker_main",
            "--home",
            str(self._home),
            "--run-id",
            run_id,
        ]
        env = os.environ.copy()
        env.setdefault("TRAINHUB_HOME", str(self._home))
        env["TRAINHUB_HEARTBEAT_SECONDS"] = str(self._settings.heartbeat_seconds)
        if gpu_index is not None:
            # The parent owns GPU slot selection: the worker only ever sees the
            # assigned GPU [doc 14 §3.2 one job per GPU].
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group so we can kill the tree
            cwd=str(self._home),
            env=env,
        )
        return process

    async def _pump_stdout(self, handle: _JobHandle) -> None:
        assert handle.process.stdout is not None
        while True:
            raw = await handle.process.stdout.readline()
            if not raw:
                break
            self._on_worker_event(handle, raw.decode(errors="replace"))

    async def _pump_stderr(self, handle: _JobHandle) -> None:
        assert handle.process.stderr is not None
        while True:
            raw = await handle.process.stderr.readline()
            if not raw:
                break
            message = raw.decode(errors="replace").strip()
            if message:
                record = handle.log.append("log", {"level": "ERROR", "message": message})
                self._broadcaster.publish(handle.run_id, record)

    def _on_worker_event(self, handle: _JobHandle, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return
        event_type = message.get("type")
        data = message.get("data") if isinstance(message.get("data"), dict) else {}

        if event_type == HEARTBEAT:
            handle.last_heartbeat = time.monotonic()
            return

        if event_type == "state":
            to = data.get("to")
            if to in ("completed", "failed", "stopped"):
                handle.terminal = str(to)
                failure_data = data.get("failure")
                handle.failure = dict(failure_data) if isinstance(failure_data, dict) else None
            # record the state in events.jsonl like any other event
        if event_type == "result":
            best_data = data.get("best")
            handle.best = dict(best_data) if isinstance(best_data, dict) else {}
            return  # result is parent bookkeeping, not an audit event

        record = handle.log.append(str(event_type), dict(data))
        self._broadcaster.publish(handle.run_id, record)
        if event_type == "state":
            to = data.get("to")
            # Terminal states are finalized by _finish, which also writes
            # best + finished_at + the audit line; pausing transitions land here
            # so the API reflects `paused` immediately [doc 08 §3].
            if isinstance(to, str) and to not in ("completed", "failed", "stopped"):
                self._set_state(handle.run_id, to)

    async def _monitor(self, handle: _JobHandle) -> None:
        """Heartbeat watchdog: 60 s of silence → kill the group, failed(worker_lost) [doc 08 §5]."""
        while not handle.finalized:
            await asyncio.sleep(1.0)
            if handle.process.returncode is not None:
                return
            if time.monotonic() - handle.last_heartbeat > handle.lost_seconds:
                handle.log.append(
                    "log",
                    {
                        "level": "ERROR",
                        "message": "worker heartbeat lost; killing the process group",
                    },
                )
                _kill_group(handle.process)
                handle.terminal = "failed"
                handle.failure = {
                    "reason": "WORKER_LOST",
                    "details": "no heartbeat within the timeout",
                }
                return

    async def _wait(self, handle: _JobHandle) -> None:
        try:
            await handle.process.wait()
        finally:
            await asyncio.sleep(0.2)  # let the stdout pump drain the last events
            await self._finish(handle)

    async def _finish(self, handle: _JobHandle) -> None:
        if handle.finalized:
            return
        handle.finalized = True
        terminal = handle.terminal or "failed"
        failure = handle.failure
        if terminal == "failed" and failure is None:
            # The worker died without reporting a reason: killed by a signal
            # means the heartbeat watchdog should have caught it, but a fast
            # kill can beat the monitor here — classify it the same way.
            returncode = handle.process.returncode
            if returncode is not None and returncode < 0:
                failure = {
                    "reason": "WORKER_LOST",
                    "details": f"worker killed by signal {-returncode}",
                }
            else:
                failure = {"reason": "UNKNOWN", "details": "worker exited without a terminal state"}
        record = self._read_run(handle.run_id)
        self._set_state(
            handle.run_id,
            terminal,
            current=record,
            failure=failure if terminal == "failed" else None,
        )
        if handle.best:
            self._write_best(handle.run_id, handle.best)
        if terminal == "stopped":
            append_audit(
                self._home, action="run.stopped", entity_type="run", entity_id=handle.run_id
            )
        else:
            append_audit(
                self._home,
                action=f"run.{terminal}",
                entity_type="run",
                entity_id=handle.run_id,
                details={"best": handle.best} if handle.best else None,
            )
        handle.log.append(
            "log", {"level": "INFO", "message": f"run finished with state {terminal}"}
        )
        handle.log.sync()
        handle.log.close()
        self._handles.pop(handle.run_id, None)
        self._release_slot(handle.gpu_index)
        await self._drain_queue()

    def _release_slot(self, gpu_index: int | None) -> None:
        if gpu_index is not None:
            lock = self._gpu_locks.get(gpu_index)
            if lock is not None:
                lock.release()
        else:
            self._slots.release()

    async def _drain_queue(self) -> None:
        """Start queued GPU runs that can now claim a slot, oldest first.

        CPU runs keep their existing create-time path (start() blocks on the
        CPU semaphore); this only retries GPU runs left in ``queued`` because a
        GPU slot was busy [doc 14 §3.2].
        """
        runs_dir = self._home / "runs"
        if not runs_dir.is_dir():
            return
        queued: list[tuple[str, str]] = []
        for path in runs_dir.glob("*/run.json"):
            try:
                data = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("state") == "queued":
                family = str(data.get("model_family") or "")
                if registry.METADATA.get(family, {}).get("needs_gpu"):
                    queued.append((str(data.get("queued_at") or ""), path.parent.name))
        for _, run_id in sorted(queued):
            if run_id in self._handles:
                continue
            try:
                await self.start(run_id)
            except JobNotFoundError:
                continue

    async def _control(self, run_id: str, action: str) -> None:
        handle = self._handles.get(run_id)
        if handle is None:
            raise JobNotFoundError(run_id)
        if handle.process.stdin is None or handle.process.returncode is not None:
            return
        handle.process.stdin.write(json.dumps({"action": action}).encode() + b"\n")
        try:
            await handle.process.stdin.drain()
        except (ConnectionError, BrokenPipeError):
            pass


def _kill_group(process: asyncio.subprocess.Process) -> None:
    """Kill the whole training process group [doc 08 §3 step 5]."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _git_commit() -> str | None:
    try:
        result = subprocess.run(  # noqa: S603 - fixed command, no shell
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _catalog_id_of(record: dict[str, Any]) -> str | None:
    checkpoint = record.get("checkpoint")
    value = checkpoint.get("catalog_id") if isinstance(checkpoint, dict) else None
    return str(value) if value else None
