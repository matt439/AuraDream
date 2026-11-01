"""Read-only checks of this machine before (or instead of) a first load.

    uv run python -m backend.doctor

Each check prints one PASS, WARN or FAIL line in plain words, and a fix under
anything that is not a pass. The exit status is 1 when any check fails, else
0. `install.ps1` runs it after `uv sync`.

Nothing here changes anything or reaches the network beyond loopback: torch
is imported to ask CUDA about the card, but diffusers is never imported and
no model is loaded; `nvidia-smi` is run if it is on the PATH; the Hugging Face
sign-in is the token file's existence only (never read, printed or sent to
the Hub); the port check is the server's own trial bind and loopback probe
(`server_probe`). The figures it compares against are the measured ones in
docs/setup.md's Before you start, never a promise for another machine.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# config is the first project import, before anything that could import
# diffusers (AGENTS.md); nothing here does, but the rule holds everywhere.
from .config import settings
from . import server_probe

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
GIB = 1024 ** 3

# Driver CUDA the cu130 torch wheels need (docs/setup.md step 1).
MIN_DRIVER_CUDA = (13, 0)
DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"
# The measured machine's figures (docs/setup.md, Before you start).
MEASURED_PRIVATE_GIB = 39.9
# The default with the critic, plus room: an estimate (docs/setup.md).
DISK_PLAN_GB = 50
DISK_LOW_GB = 2


@dataclass(frozen=True)
class Check:
    status: str
    title: str
    detail: str
    fix: str = ""


def _gib(n: float) -> str:
    return f"{n / GIB:.1f} GiB"


# --- GPU through torch --------------------------------------------------------

def arch_supported(capability: tuple[int, int], arch_list: Iterable[str]) -> bool:
    """Whether torch's kernels run on a card of `capability`: a `sm_XY` entry
    of the same major and a minor no higher (CUDA binaries run on later minors
    of their major), or a `compute_XY` PTX entry no newer than the card."""
    major, minor = capability
    for arch in arch_list:
        match = re.fullmatch(r"(sm|compute)_(\d+)(\d)[a-z]?", arch)
        if not match:
            continue
        kind, arch_major, arch_minor = match.group(1), int(match.group(2)), int(match.group(3))
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= capability:
            return True
    return False


def _arch_range(arch_list: list[str]) -> str:
    sms = [a for a in arch_list if a.startswith("sm_")]
    return f"{sms[0]} to {sms[-1]}" if sms else ", ".join(arch_list) or "none"


def check_gpu(torch: Any = None) -> list[Check]:
    """Whether torch sees a CUDA card, and whether its build runs on it."""
    if torch is None:
        try:
            import torch  # noqa: PLC0415 - heavy, and only this check needs it
        except Exception as exc:  # noqa: BLE001 - any import failure is the same advice
            return [Check(FAIL, "PyTorch", f"torch could not be imported ({type(exc).__name__}: {exc}).",
                          "From the project folder, run `uv sync` and try again.")]
    version = getattr(torch, "__version__", "?")
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if not cuda:
        return [Check(FAIL, "PyTorch", f"torch {version} is a CPU-only build.",
                      "Run `uv sync` from the project folder; it installs the CUDA 13 build.")]
    if not torch.cuda.is_available():
        return [Check(FAIL, "GPU", f"torch {version} cannot use a CUDA GPU.",
                      f"Update the NVIDIA driver to one that supports CUDA 13 or newer ({DRIVER_URL}), "
                      "reboot, and run this again. Do not install the CUDA Toolkit.")]
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    arch_list = list(torch.cuda.get_arch_list())
    cap = f"{capability[0]}.{capability[1]}"
    if not arch_supported((capability[0], capability[1]), arch_list):
        return [Check(FAIL, "GPU", f"{name} (compute capability {cap}) is older than this torch "
                      f"{version} supports ({_arch_range(arch_list)}).",
                      "AuraDream needs an RTX 20 / GTX 16 series card or newer.")]
    return [Check(PASS, "GPU", f"{name}, compute capability {cap}, supported by torch {version} "
                  f"({_arch_range(arch_list)}).")]


# --- Driver through nvidia-smi ------------------------------------------------

_CUDA_VERSION = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")


def parse_driver_cuda(text: str) -> tuple[int, int] | None:
    """The CUDA version `nvidia-smi`'s header reports ("CUDA Version: 13.0",
    or "CUDA UMD Version: 13.4" from newer drivers), or None."""
    match = _CUDA_VERSION.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def check_driver(which: Callable[[str], str | None] = shutil.which,
                 run: Callable[..., Any] = subprocess.run) -> list[Check]:
    tool = which("nvidia-smi")
    if not tool:
        return [Check(WARN, "Driver", "nvidia-smi was not found, so the driver's CUDA version is unknown.",
                      f"Install the current NVIDIA driver ({DRIVER_URL}) if the GPU check fails.")]
    try:
        result = run([tool], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return [Check(FAIL, "Driver", f"nvidia-smi did not run ({exc}).",
                      f"Reinstall the current NVIDIA driver ({DRIVER_URL}) and reboot.")]
    if result.returncode != 0:
        return [Check(FAIL, "Driver", f"nvidia-smi failed (exit {result.returncode}): the driver is not working.",
                      f"Reinstall the current NVIDIA driver ({DRIVER_URL}) and reboot.")]
    version = parse_driver_cuda(result.stdout or "")
    if version is None:
        return [Check(WARN, "Driver", "nvidia-smi ran, but its CUDA version could not be read.",
                      "Check that `nvidia-smi` shows CUDA 13.0 or higher; the GPU check above is the one that counts.")]
    shown = f"{version[0]}.{version[1]}"
    if version < MIN_DRIVER_CUDA:
        return [Check(FAIL, "Driver", f"the driver supports CUDA {shown}; AuraDream's torch needs 13.0 or higher.",
                      f"Install the current NVIDIA driver ({DRIVER_URL}) and reboot.")]
    return [Check(PASS, "Driver", f"supports CUDA {shown}.")]


# --- RAM ----------------------------------------------------------------------

def memory_status() -> tuple[int, int] | None:
    """(total, available) bytes of RAM, or None where they cannot be read."""
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text())
    except OSError:
        return None


def parse_meminfo(text: str) -> tuple[int, int] | None:
    fields = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            fields[name.strip()] = int(parts[0]) * 1024
    if "MemTotal" not in fields:
        return None
    return fields["MemTotal"], fields.get("MemAvailable", fields.get("MemFree", 0))


def check_ram(read: Callable[[], tuple[int, int] | None] = memory_status) -> list[Check]:
    status = read()
    if status is None:
        return [Check(WARN, "RAM", "the amount of RAM could not be read.")]
    total, available = status
    detail = (f"{_gib(total)} total, {_gib(available)} free now. The default checkpoint's load was "
              f"measured at 27.9 GiB working set and 35.4 GiB private (39.9 GiB in the first render) "
              f"on a 64 GB machine.")
    if total / GIB < MEASURED_PRIVATE_GIB:
        return [Check(WARN, "RAM", detail + " Less RAM than that has not been measured.",
                      "Start with an SDXL checkpoint, such as Animagine XL 4.0, which needs far less.")]
    return [Check(PASS, "RAM", detail)]


# --- Disk ---------------------------------------------------------------------

def _existing(path: Path) -> Path:
    """`path`, or its nearest parent that exists (what it would be made on)."""
    path = Path(path).expanduser()
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return path


def disk_places() -> list[tuple[str, Path]]:
    from huggingface_hub import constants

    from .legacy_names import hf_home_dir

    return [
        ("model downloads (Hugging Face cache)", Path(constants.HF_HUB_CACHE)),
        ("converted copies", hf_home_dir("converted")),
        ("images (outputs)", settings.outputs_dir),
    ]


def check_disk(places: list[tuple[str, Path]] | None = None,
               usage: Callable[[Path], Any] = shutil.disk_usage) -> list[Check]:
    """Free space on each drive the downloads, conversions and images use,
    one line per drive."""
    if places is None:
        places = disk_places()
    drives: dict[str, tuple[list[str], list[Path], Any]] = {}
    checks: list[Check] = []
    for label, path in places:
        base = _existing(path)
        try:
            found = usage(base)
        except OSError as exc:
            checks.append(Check(WARN, "Disk", f"free space for {label} ({path}) could not be read: {exc}."))
            continue
        key = os.path.splitdrive(str(base))[0].upper() or f"{found.total}:{found.free}"
        entry = drives.setdefault(key, ([], [], found))
        entry[0].append(label)
        entry[1].append(path)
    for labels, paths, found in drives.values():
        free_gb = found.free / 1e9
        where = "; ".join(f"{label} in {path}" for label, path in zip(labels, paths))
        detail = f"{free_gb:.0f} GB free for {where}."
        downloads = any(not label.startswith("images") for label in labels)
        if free_gb < DISK_LOW_GB:
            checks.append(Check(FAIL, "Disk", detail, "Free some space on that drive first."))
        elif downloads and free_gb < DISK_PLAN_GB:
            checks.append(Check(WARN, "Disk", detail + f" The default checkpoint with the critic needs roughly "
                                f"{DISK_PLAN_GB} GB (an estimate); an SDXL checkpoint several GB.",
                                "Free space there, or set HF_HOME to a folder on a larger drive before "
                                "the first load (docs/setup.md step 3)."))
        else:
            checks.append(Check(PASS, "Disk", detail))
    return checks


# --- Port ---------------------------------------------------------------------

def check_port(host: str | None = None, port: int | None = None,
               busy: Callable[[str, int], bool] = server_probe.port_busy,
               ours: Callable[[str], bool] = server_probe.auradream_at) -> list[Check]:
    host = settings.host if host is None else host
    port = settings.port if port is None else port
    url = server_probe.page_url(host, port)
    if not port or not busy(host, port):
        return [Check(PASS, "Port", f"{port} is free; the server will be at {url}")]
    if ours(url):
        return [Check(PASS, "Port", f"AuraDream is already running at {url}")]
    return [Check(FAIL, "Port", f"{port} is in use by another program (or reserved by the system).",
                  f"Close that program, or set AURADREAM_PORT to a free port, such as {port + 1}.")]


# --- Hugging Face sign-in -----------------------------------------------------

def check_hf_login(token_path: Path | None = None, environ: dict[str, str] | None = None) -> list[Check]:
    """Whether a Hugging Face token is present: the token file's existence or
    HF_TOKEN being set. Never read, printed or checked with the Hub."""
    environ = dict(os.environ) if environ is None else environ
    if token_path is None:
        from huggingface_hub import constants

        token_path = Path(constants.HF_TOKEN_PATH)
    if environ.get("HF_TOKEN"):
        return [Check(PASS, "Hugging Face", "a token is set in HF_TOKEN (not checked with the Hub here).")]
    if token_path.is_file():
        return [Check(PASS, "Hugging Face", f"signed in: a token file is in {token_path.parent} "
                      "(not checked with the Hub here).")]
    return [Check(WARN, "Hugging Face", "not signed in. Open checkpoints such as Animagine XL 4.0 need no "
                  "sign-in; the default, FLUX.1 Krea [dev], does.",
                  "For a gated checkpoint, accept its licence on its Hugging Face page, then run "
                  "`uv run hf auth login` (docs/setup.md step 6).")]


# --- Running ------------------------------------------------------------------

CHECKS: tuple[Callable[[], list[Check]], ...] = (
    check_driver, check_gpu, check_ram, check_disk, check_port, check_hf_login,
)


def report(checks: Iterable[Check]) -> tuple[list[str], int]:
    """The printed lines and the exit status (1 if anything failed)."""
    lines: list[str] = []
    failed = False
    for check in checks:
        failed = failed or check.status == FAIL
        lines.append(f"{check.status}  {check.title}: {check.detail}")
        if check.fix:
            lines.append(f"      What to do: {check.fix}")
    return lines, 1 if failed else 0


def run_checks(checks: Iterable[Callable[[], list[Check]]] = CHECKS) -> list[Check]:
    found: list[Check] = []
    for check in checks:
        try:
            found.extend(check())
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            found.append(Check(WARN, check.__name__.removeprefix("check_"),
                               f"the check itself failed ({type(exc).__name__}: {exc})."))
    return found


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    argparse.ArgumentParser(
        prog="python -m backend.doctor",
        description="Read-only checks of this machine: GPU, driver, RAM, disk, port, sign-in.",
    ).parse_args(argv)
    print("AuraDream doctor: read-only checks of this machine.", flush=True)
    lines, status = report(run_checks())
    for line in lines:
        print(line, flush=True)
    print("Nothing failed." if status == 0 else "Something failed: see What to do above.", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
