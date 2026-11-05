"""Describe the machine a run executed on, so timings can be compared fairly."""

import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

import psutil

_NPU_DRIVERS = {"amdxdna": "AMD XDNA", "intel_vpu": "Intel NPU"}


def _cpu_name() -> str:
    """Read the CPU model, falling back to what the platform reports."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _command(*arguments: str) -> str | None:
    """Run a detection tool if it exists; any failure just means nothing was found."""
    binary = shutil.which(arguments[0])
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, *arguments[1:]], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _accelerator(kind: str, name: str, source: str) -> dict[str, str]:
    """Describe one device."""
    return {"type": kind, "name": name, "source": source}


def _nvidia() -> list[dict[str, str]]:
    """List NVIDIA GPUs."""
    output = _command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    names = (output or "").splitlines()
    return [_accelerator("gpu", name.strip(), "nvidia-smi") for name in names]


def _rocm() -> list[dict[str, str]]:
    """List AMD GPUs through ROCm, which also works inside WSL."""
    accelerators = []
    name = None
    for line in (_command("rocminfo") or "").splitlines():
        if "Marketing Name:" in line:
            name = line.split(":", 1)[1].strip()
        elif "Device Type:" in line and line.split(":", 1)[1].strip() == "GPU":
            accelerators.append(_accelerator("gpu", name or "AMD GPU", "rocminfo"))
    return accelerators


def _apple() -> list[dict[str, str]]:
    """List a Mac's GPUs, and the Neural Engine on Apple silicon."""
    if platform.system() != "Darwin":
        return []
    accelerators = []
    output = _command("system_profiler", "SPDisplaysDataType", "-json")
    for display in json.loads(output or "{}").get("SPDisplaysDataType", []):
        name = display.get("sppci_model", "GPU")
        accelerators.append(_accelerator("gpu", name, "system_profiler"))
    if platform.machine() == "arm64":
        accelerators.append(_accelerator("npu", "Apple Neural Engine", "platform"))
    return accelerators


def _linux_npus() -> list[dict[str, str]]:
    """List NPUs the Linux kernel exposes as accel devices."""
    accelerators = []
    for device in Path("/sys/class/accel").glob("accel*"):
        driver = (device / "device" / "driver").resolve().name
        name = _NPU_DRIVERS.get(driver, driver)
        accelerators.append(_accelerator("npu", name, "sysfs"))
    return accelerators


def _cpu_hints(cpu: str, found: list[dict[str, str]]) -> list[dict[str, str]]:
    """Infer integrated devices from the CPU's name when no tool reported them.

    A hint says the silicon exists, not that anything here can reach it.
    """
    hints = []
    has_gpu = any(entry["type"] == "gpu" for entry in found)
    has_npu = any(entry["type"] == "npu" for entry in found)
    graphics = re.search(r"w/ (Radeon[\w ]*)", cpu)
    if graphics and not has_gpu:
        hints.append(_accelerator("gpu", graphics[1].strip(), "cpu_name"))
    if not has_npu and re.search(r"Ryzen AI|Core(\(TM\))? Ultra", cpu, re.IGNORECASE):
        hints.append(_accelerator("npu", "Integrated NPU", "cpu_name"))
    return hints


def describe_hardware() -> dict[str, object]:
    """Report what is present.

    Being present doesn't mean it was used: each phase records its own `device`.
    """
    cpu = _cpu_name()
    accelerators = [*_nvidia(), *_rocm(), *_apple(), *_linux_npus()]
    return {
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "accelerators": [*accelerators, *_cpu_hints(cpu, accelerators)],
    }
