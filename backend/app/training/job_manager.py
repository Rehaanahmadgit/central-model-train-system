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
from datetime import datetime
from pathlib import Path
from typing import Any

from app.checkpoints.resolver import CheckpointError, CheckpointResolver
from app.core.config import Settings
from app.core.states import can_transition, transition
from app.realtime.broadcaster import EventBroadcaster
from app.realtime.events import RunEventLog
from app.storage.atomic import atomic_write_json, read_json
from app.storage.audit import append_audit
from app.storage.locks import FileLock

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
        # CPU concurrency limit [doc 14 §3.2]; GPU slot locks arrive with the
        # GPU slice. This is the honest degradation for a laptop/CPU image.
        self._slots = asyncio.Semaphore(max(1, settings.max_cpu_jobs))

    # -- public API ---------------------------------------------------------

    async def start(self, run_id: str) -> None:
        """Claim a queued run: preparing → (checkpoint) → spawn → running."""
        record = self._read_run(run_id)
        if record is None:
            raise JobNotFoundError(run_id)
        state = str(record.get("state") or "queued")
        if state != "queued" or not can_transition(state, "preparing"):
            return
        await self._slots.acquire()
        try:
            self._set_state(run_id, "preparing", record)
            env = self._capture_env(record)
            self._write_env(run_id, env)
            await self._resolve_checkpoint(run_id, record)
            process = await self._spawn(run_id)
        except BaseException:
            self._slots.release()
            self._set_state(run_id, "failed", failure={"reason": "UNKNOWN"})
            raise
        handle = _JobHandle(
            process, run_id, self._home, self._broadcaster, self._settings.worker_lost_seconds
        )
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
        elif to == "stopped":
            record["failure"] = {"reason": None, "details": None}
        if to in ("completed", "failed", "stopped"):
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        if to in ("running", "paused"):
            record["started_at"] = record.get("started_at") or datetime.now().isoformat(
                timespec="seconds"
            )
        path = self._home / "runs" / run_id / "run.json"
        with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
            atomic_write_json(path, record)

    def _capture_env(self, record: dict[str, Any]) -> dict[str, Any]:
        del record
        torch_version = None
        cuda_version = None
        try:
            import torch

            torch_version = str(torch.__version__)
            cuda_version = str(torch.version.cuda) if torch.cuda.is_available() else None
        except Exception:  # pragma: no cover - torch may be absent in future images
            pass
        return {
            "python": sys.version.split()[0],
            "torch": torch_version,
            "cuda": cuda_version,
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
        resolver = CheckpointResolver(self._home)
        try:
            resolved = await asyncio.to_thread(resolver.resolve, str(catalog_id))
        except CheckpointError as exc:
            raise RuntimeError(f"checkpoint {catalog_id}: {exc.message}") from exc
        path = self._home / "runs" / run_id / "run.json"
        data = read_json(path)
        if isinstance(data, dict):
            data["checkpoint"] = {
                "catalog_id": str(catalog_id),
                "sha256": resolved.sha256,
                "source": resolved.source,
                "path": str(resolved.path),
            }
            with FileLock(self._home / "locks" / f"run-{run_id}.lock"):
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

    async def _spawn(self, run_id: str) -> asyncio.subprocess.Process:
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
        self._slots.release()

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
