"""Device abstraction — the only place that decides cpu vs cuda [doc 14 §3.1].

Plugins receive the resolved device and never hard-code ``"cuda"``. When
PyTorch is not installed (the current CPU-only image) CUDA is reported as
unavailable with an honest reason instead of guessing.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # "cpu" | "cuda"
    index: int | None
    name: str
    torch_installed: bool
    cuda_available: bool
    reason: str
    memory_gb: float | None = None  # total VRAM of the selected GPU, None = unknown

    @property
    def is_gpu(self) -> bool:
        return self.kind == "cuda"

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DeviceInfo:
        index = data.get("index")
        memory = data.get("memory_gb")
        return cls(
            kind=str(data.get("kind") or "cpu"),
            index=int(index) if isinstance(index, int | float) else None,
            name=str(data.get("name") or "CPU"),
            torch_installed=bool(data.get("torch_installed")),
            cuda_available=bool(data.get("cuda_available")),
            reason=str(data.get("reason") or ""),
            memory_gb=float(memory) if isinstance(memory, int | float) else None,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "index": self.index,
            "name": self.name,
            "torch_installed": self.torch_installed,
            "cuda_available": self.cuda_available,
            "reason": self.reason,
            "memory_gb": self.memory_gb,
        }


def _torch_status() -> tuple[bool, bool, str]:
    if importlib.util.find_spec("torch") is None:
        return (
            False,
            False,
            "PyTorch is not installed in this image, so CUDA cannot be used",
        )
    try:
        torch = importlib.import_module("torch")  # heavy and optional: never at import time

        if torch.cuda.is_available():
            return True, True, f"CUDA runtime {torch.version.cuda}"
        return True, False, "PyTorch is installed but no CUDA device is visible"
    except Exception as exc:  # pragma: no cover - defensive; import can fail at runtime
        return False, False, f"PyTorch import failed: {type(exc).__name__}"


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve ``auto|cpu|cuda|cuda:N`` against what is actually available."""
    torch_installed, cuda_available, torch_reason = _torch_status()
    index: int | None = None
    if requested.startswith("cuda:"):
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError:
            index = None
    wants_cuda = requested == "auto" or requested.startswith("cuda")

    if not wants_cuda:
        return DeviceInfo(
            kind="cpu",
            index=None,
            name="CPU",
            torch_installed=torch_installed,
            cuda_available=cuda_available,
            reason="TRAINHUB_DEVICE requests cpu",
        )
    if cuda_available:
        label = _gpu_name(index)
        return DeviceInfo(
            kind="cuda",
            index=index,
            name=label,
            torch_installed=torch_installed,
            cuda_available=True,
            reason=torch_reason,
            memory_gb=_gpu_memory_gb(index),
        )
    return DeviceInfo(
        kind="cpu",
        index=None,
        name="CPU",
        torch_installed=torch_installed,
        cuda_available=False,
        reason=f"{torch_reason}; falling back to CPU",
    )


def _gpu_memory_gb(index: int | None) -> float | None:
    """Total VRAM in GB, or None when it cannot be read (never guessed)."""
    try:
        torch = importlib.import_module("torch")
        props = torch.cuda.get_device_properties(index or 0)
        return round(float(props.total_memory) / (1024**3), 2)
    except Exception:  # pragma: no cover - no CUDA / odd driver
        return None


def _gpu_name(index: int | None) -> str:
    """Best-effort GPU name without importing torch (used for reports)."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "GPU"
    command = [executable]
    if index is not None:
        command.append(f"--id={index}")
    command += ["--query-gpu=name", "--format=csv,noheader"]
    try:
        output = subprocess.run(  # noqa: S603 - fixed executable, no shell
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "GPU"
    lines = output.stdout.strip().splitlines()
    return lines[0].strip() if lines else "GPU"


@lru_cache(maxsize=4)
def probe_device(requested: str = "auto") -> dict[str, object]:
    """Resolve the device in a throw-away subprocess, result cached.

    The API process must never import torch itself: the verification pool forks
    from it, and forking after torch's thread pools exist can deadlock (seen: the
    Verify step hung once /system imported torch). Slice F.
    """
    try:
        output = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "app.core.device", requested],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        data = json.loads(output.stdout.strip().splitlines()[-1])
        if isinstance(data, dict):
            return data
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return {
        "kind": "cpu",
        "index": None,
        "name": "CPU",
        "torch_installed": False,
        "cuda_available": False,
        "reason": "device probe failed; assuming CPU",
        "memory_gb": None,
    }


def _probe_main(requested: str) -> dict[str, object]:  # pragma: no cover - probe subprocess
    data = resolve_device(requested).as_dict()
    data["torch_version"] = None
    data["cuda_version"] = None
    if importlib.util.find_spec("torch") is not None:
        try:
            torch = importlib.import_module("torch")
            data["torch_version"] = str(torch.__version__)
            if torch.cuda.is_available():
                data["cuda_version"] = str(torch.version.cuda)
        except Exception:
            pass
    return data


if __name__ == "__main__":  # pragma: no cover - used by probe_device
    print(json.dumps(_probe_main(sys.argv[1] if len(sys.argv) > 1 else "auto")))
