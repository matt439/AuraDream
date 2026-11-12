"""Read-only checks of this machine before (or instead of) a first load.

    uv run python -m backend.doctor

Each check prints one PASS, WARN or FAIL line in plain words, and a fix under
anything that is not a pass. The exit status is 1 when any check fails, else
0. `install.ps1` runs it after `uv sync`.

Nothing here changes anything or reaches the network beyond loopback: torch
is imported to ask CUDA about the card, but diffusers is never imported and
no model is loaded; `nvidia-smi` is run if it is on the PATH; on Linux,
`systemd-inhibit` holds the PC awake around `true`, which lets go as soon as
`true` ends, and on Windows a power request is set and let go at once (no
power setting is read or changed); the Hugging Face sign-in is the token file's existence only
(never read, printed or sent to the Hub); the port check is the server's own
trial bind and loopback probe (`server_probe`); the commit is `git rev-parse`
in this folder, and the running server's own from its status on loopback
(with `AURADREAM_PASSCODE` from this environment where that server asks for
one; the passcode is never printed); a direct-TLS pair is loaded as the
server loads it, read and never changed.
The figures it compares against are the measured ones in docs/setup.md's
Before you start and docs/verified.md, never a promise for another machine.
"""
from __future__ import annotations

import argparse
import ctypes
import getpass
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# config is the first project import, before anything that could import
# diffusers (AGENTS.md); nothing here does, but the rule holds everywhere.
from .config import settings
from . import build_info, hf_sign_in, keep_awake, server_probe, server_transport
from .loader_choices import default_format
from .passcode import MIN_LENGTH as PASSCODE_MIN_LENGTH, asked as passcode_asked, setting_problem as passcode_problem

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
# The default checkpoint's peak on the card: FLUX.1 Krea [dev], fp8 with model
# offload, 12.00 GiB allocated (12.63 reserved) on the RTX 5080
# (docs/verified.md, "RAM at the load, the fixed loader (2026-10-27)").
DEFAULT_PEAK_GIB = 12.0
# Seconds the keep-awake trial may take; a polkit answer is immediate.
INHIBIT_TRIAL_SECONDS = 10


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


def check_vram(torch: Any = None, platform: str | None = None) -> list[Check]:
    """The card's memory against the default checkpoint's measured peak, and
    on Linux below Blackwell which Nunchaku repos run. Nothing when torch
    cannot use a card: the GPU check says why."""
    if torch is None:
        try:
            import torch  # noqa: PLC0415 - heavy, and only the GPU checks need it
        except Exception:  # noqa: BLE001 - the GPU check reports it
            return []
    if not getattr(getattr(torch, "version", None), "cuda", None) or not torch.cuda.is_available():
        return []
    platform = sys.platform if platform is None else platform
    total = int(torch.cuda.get_device_properties(0).total_memory)
    capability = tuple(torch.cuda.get_device_capability(0))
    detail = (f"{_gib(total)} on the card. The default checkpoint, FLUX.1 Krea [dev] at fp8 with model "
              f"offload, peaked at {DEFAULT_PEAK_GIB:.1f} GiB allocated on the measured RTX 5080 "
              "(docs/verified.md).")
    if platform.startswith("linux") and default_format(capability) == "int4":
        detail += (" For AURADREAM_QUANT=nunchaku, use the int4 4-bit repos: nvfp4 needs an RTX 50 card "
                   "(compute capability 10.0 or higher).")
    if total / GIB < DEFAULT_PEAK_GIB:
        return [Check(WARN, "VRAM", detail,
                      "Start with a smaller checkpoint: an SDXL one such as Animagine XL 4.0, Anima, or "
                      "FLUX.2 [klein] 4B (the Models panel, or AURADREAM_MODEL_ID).")]
    return [Check(PASS, "VRAM", detail)]


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
    url = server_probe.page_url(host, port, secure=settings.ssl_certfile is not None)
    if not port or not busy(host, port):
        return [Check(PASS, "Port", f"{port} is free; the server will be at {url}")]
    if ours(url):
        return [Check(PASS, "Port", f"AuraDream is already running at {url}")]
    return [Check(FAIL, "Port", f"{port} is in use by another program (or reserved by the system).",
                  f"Close that program, or set AURADREAM_PORT to a free port, such as {port + 1}.")]


# --- Commit -------------------------------------------------------------------

def check_commit(host: str | None = None, port: int | None = None,
                 checkout: Callable[[], str | None] = build_info.read_commit,
                 busy: Callable[[str, int], bool] = server_probe.port_busy,
                 running: Callable[[str, str | None], str | None] = server_probe.running_commit,
                 passcode: str | None = None) -> list[Check]:
    """This folder's short commit, and the running server's where one
    answers on the port: they differ after an update without a restart.
    A server that asks for a passcode is sent this environment's
    (`settings.passcode`, or `passcode`); without it the row says so, and
    does not fail."""
    host = settings.host if host is None else host
    port = settings.port if port is None else port
    passcode = settings.passcode if passcode is None else passcode
    here = checkout()
    mine = f"this folder is at {here}" if here is not None else "this folder's own is unknown"
    # Asked only of a held port: a refused connection is slow on Windows.
    try:
        there = running(server_probe.page_url(host, port, secure=settings.ssl_certfile is not None), passcode) if port and busy(host, port) else None
    except server_probe.PasscodeRefused as refused:
        detail = f"{mine}. The running server {refused.reason}, so its commit was not read."
        if refused.sent or refused.locked:
            return [Check(WARN, "Commit", detail, refused.fix)]
        return [Check(PASS, "Commit", f"{detail} {refused.fix}")]
    if here is None and there is None:
        return [Check(PASS, "Commit", "unknown: this folder is not a Git checkout, or git was not found.")]
    if there is None:
        return [Check(PASS, "Commit", f"this folder is at {here}.")]
    if here is None:
        return [Check(PASS, "Commit", f"the running server started at {there}; this folder's own is unknown "
                      "(not a Git checkout, or git was not found).")]
    if build_info.same_commit(here, there):
        return [Check(PASS, "Commit", f"this folder and the running server are both at {here}.")]
    return [Check(WARN, "Commit", f"the running server started at {there}, and this folder is now at {here}: "
                  "it is still running the code it started with.",
                  "Restart AuraDream to run this folder's code.")]


# --- LAN settings -------------------------------------------------------------

def check_lan(host: str | None = None, allowed: Iterable[str] | None = None) -> list[Check]:
    """With a bind other machines can reach, whether AURADREAM_ALLOWED_HOSTS
    lets the local probes and the browser's writes through. Nothing on a
    loopback bind."""
    host = settings.host if host is None else host
    names = list(settings.allowed_hosts if allowed is None else allowed)
    if server_probe.is_loopback(host):
        return []
    checks: list[Check] = []
    if "127.0.0.1" not in names:
        checks.append(Check(
            WARN, "LAN", "AURADREAM_ALLOWED_HOSTS does not list 127.0.0.1. A second start, this doctor and "
            "the command line ask the server at 127.0.0.1, so a second start takes AuraDream for "
            "another program.",
            "Add 127.0.0.1 (and localhost) to AURADREAM_ALLOWED_HOSTS beside the LAN names: setting it "
            "replaces the default."))
    if names and all(server_probe.is_loopback(name) for name in names):
        checks.append(Check(
            WARN, "LAN", "AURADREAM_ALLOWED_HOSTS lists only this machine's names, so a browser on "
            "another machine is refused with 400 Invalid host header.",
            "Add the names and addresses other machines use, such as "
            "127.0.0.1,localhost,my-box,192.168.1.20."))
    upper = [name for name in names if name != name.lower()]
    if upper:
        checks.append(Check(
            WARN, "LAN", f"AURADREAM_ALLOWED_HOSTS has capital letters ({', '.join(upper)}). Browsers "
            "send names in lowercase and the checks compare exactly, so those names are refused.",
            "Write the names in lowercase, such as my-box."))
    wild = [name for name in names if "*" in name]
    if wild:
        checks.append(Check(
            WARN, "LAN", f"AURADREAM_ALLOWED_HOSTS has a wildcard ({', '.join(wild)}). The page loads, but "
            "the origin check compares names exactly, so every change from the browser is refused "
            "with 403.",
            "List the exact names and addresses the browser uses, such as my-box,192.168.1.20."))
    if not checks:
        checks.append(Check(PASS, "LAN", f"serving on {host} to {', '.join(names) or 'no names'}."))
    return checks


# --- Passcode -----------------------------------------------------------------

def check_passcode(host: str | None = None, allowed: Iterable[str] | None = None,
                   environ: Mapping[str, str] | None = None) -> list[Check]:
    """Whether `AURADREAM_PASSCODE` is asked, as `main` decides it
    (`passcode.asked`): only on a server another machine can reach. A
    passcode the server would refuse to start with fails; a reachable server
    with none warns. The value is never printed."""
    host = settings.host if host is None else host
    names = list(settings.allowed_hosts if allowed is None else allowed)
    passcode = settings.passcode if environ is None else server_probe.environment_passcode(environ)
    variable = server_probe.PASSCODE_VARIABLE
    problem = passcode_problem(passcode)
    if problem is not None:
        said, _, fix = problem.partition(". ")
        return [Check(FAIL, "Passcode", f"{said}, so the server will not start.", fix)]
    if not passcode_asked(host, names):
        return [Check(PASS, "Passcode", ("set, but not asked" if passcode else "not asked")
                      + ": this server is reached from this machine only.")]
    if passcode is None:
        return [Check(
            WARN, "Passcode", "none is set, and this server can be reached from other machines: anyone who "
            "can open its address can use it, read and delete the gallery, and queue work.",
            f"Set {variable} ({PASSCODE_MIN_LENGTH} characters or more, not a password used anywhere else) "
            "where the server starts, and restart it. For the Linux service that is its environment file.")]
    return [Check(PASS, "Passcode", "asked of every browser and script: this server can be reached from other "
                  "machines. Over plain http it crosses the network unencrypted, so it keeps out people who "
                  "can merely reach the address, and is no protection on a network that is not trusted.")]


def check_tls(pair: tuple[Path | None, Path | None] | None = None) -> list[Check]:
    """The direct-TLS pair as `main.run` loads it (`server_transport.setting_problem`):
    no row without one, a FAIL for a pair the server would refuse to start with.
    The files are read, never changed; a key's password is never asked for."""
    certfile, keyfile = (settings.ssl_certfile, settings.ssl_keyfile) if pair is None else pair
    if certfile is None and keyfile is None:
        return []
    problem = server_transport.setting_problem(certfile, keyfile)
    if problem is not None:
        said, _, fix = problem.partition(". ")
        return [Check(FAIL, "TLS", f"{said.rstrip('.')}, so the server will not start.", fix or said)]
    return [Check(PASS, "TLS", "the certificate and its key load: the server serves https.")]


# --- Keeping the PC awake -----------------------------------------------------

def polkit_rule(user: str) -> str:
    """The polkit rule that lets `user` take the block hold from a service."""
    return (
        "polkit.addRule(function(action, subject) {\n"
        "    if ((action.id == \"org.freedesktop.login1.inhibit-block-sleep\" ||\n"
        "         action.id == \"org.freedesktop.login1.inhibit-block-idle\") &&\n"
        f"        subject.user == \"{user}\") {{\n"
        "        return polkit.Result.YES;\n"
        "    }\n"
        "});"
    )


def _check_power_request(request: keep_awake.Holder) -> list[Check]:
    """Windows' trial: the server's own power request, set and let go at
    once. It changes no power setting."""
    try:
        request.hold()
    except Exception as exc:  # noqa: BLE001 - whatever stopped it is the row's words
        why = str(exc) or type(exc).__name__
        return [Check(WARN, "Keep awake", f"Windows refused a power request ({why}), so this PC may sleep "
                      "in the middle of a load or a render.",
                      "If this PC sleeps when idle, lengthen or turn off its sleep timer (Settings > System > "
                      "Power) while AuraDream is used from another device.")]
    finally:
        request.release()
    return [Check(PASS, "Keep awake", "Windows accepts a power request, so the idle sleep timer waits while "
                  "AuraDream works. The display may still turn off, and a sleep you ask for still happens.")]


def _check_wsl_power_request(program: str | None,
                             build: Callable[[str], keep_awake.Holder] | None) -> list[Check]:
    """WSL's trial: Windows' power request through powershell.exe, set and
    let go at once; systemd-inhibit is never tried, since under WSL it
    holds only the Linux VM."""
    if not program:
        return [Check(WARN, "Keep awake", keep_awake.WSL_NO_HOLD + ".",
                      "Turn WSL interop on ([interop] enabled=true in /etc/wsl.conf, then wsl --shutdown), or "
                      "lengthen or turn off Windows' sleep timer (Settings > System > Power) while AuraDream "
                      "is used from another device.")]
    request = (build or (lambda path: keep_awake.WslPowerRequest("AuraDream doctor: check", program=path)))(program)
    try:
        request.hold()
    except Exception as exc:  # noqa: BLE001 - whatever stopped it is the row's words
        why = str(exc) or type(exc).__name__
        return [Check(WARN, "Keep awake", f"Under WSL, Windows refused a power request through powershell.exe "
                      f"({why}), so this PC may sleep in the middle of a load or a render. systemd-inhibit "
                      "would hold only the Linux VM.",
                      "If this PC sleeps when idle, lengthen or turn off its sleep timer (Settings > System > "
                      "Power) while AuraDream is used from another device.")]
    finally:
        request.release()
    return [Check(PASS, "Keep awake", "Under WSL, Windows accepts a power request made through powershell.exe, "
                  "so Windows' idle sleep timer waits while AuraDream works (systemd-inhibit, which holds only "
                  "the Linux VM, is not used). The display may still turn off.")]


def check_keep_awake(platform: str | None = None, which: Callable[[str], str | None] = shutil.which,
                     run: Callable[..., Any] = subprocess.run, enabled: bool | None = None,
                     user: str | None = None,
                     power: Callable[[], keep_awake.Holder] | None = None,
                     wsl: bool | None = None, exists: Callable[[str], bool] = os.path.exists,
                     wsl_power: Callable[[str], keep_awake.Holder] | None = None) -> list[Check]:
    """Whether the server may hold the PC awake while it works. On Linux, a
    trial block hold around `true`, let go when it ends; under WSL, a trial
    power request through powershell.exe (`wsl_power` builds it, `wsl` None
    asks `keep_awake.under_wsl`); on Windows, a trial power request (`power`
    builds it); each let go at once. Nothing elsewhere."""
    platform = sys.platform if platform is None else platform
    if platform != "win32" and not platform.startswith("linux"):
        return []
    if not (settings.keep_awake if enabled is None else enabled):
        return [Check(PASS, "Keep awake", "switched off (AURADREAM_KEEP_AWAKE=false), so nothing is tried.")]
    if platform == "win32":
        return _check_power_request((power or (lambda: keep_awake.PowerRequest("AuraDream doctor: check")))())
    if keep_awake.under_wsl(exists=exists) if wsl is None else wsl:
        return _check_wsl_power_request(keep_awake.windows_powershell(which=which, exists=exists), wsl_power)
    program = which("systemd-inhibit")
    if not program:
        return [Check(WARN, "Keep awake", "systemd-inhibit was not found, so AuraDream cannot keep this PC "
                      "awake while it works.",
                      "If this PC suspends when idle, install systemd's tools or turn the idle suspend off.")]
    argv = [program, "--what=sleep:idle", "--who=AuraDream doctor", "--why=check", "--mode=block", "true"]
    try:
        done = run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                   timeout=INHIBIT_TRIAL_SECONDS)
    except subprocess.TimeoutExpired:
        said, code = f"it did not finish within {INHIBIT_TRIAL_SECONDS} s", None
    except OSError as exc:
        said, code = str(exc), None
    else:
        code = done.returncode
        said = next((line.strip() for line in f"{done.stderr or ''}\n{done.stdout or ''}".splitlines()
                     if line.strip()), "")
    if code == 0:
        return [Check(PASS, "Keep awake", "systemd-inhibit may hold this PC awake (sleep and idle) while "
                      "AuraDream works.")]
    who = user or getpass.getuser()
    why = said or f"exit code {code}"
    return [Check(WARN, "Keep awake", f"systemd-inhibit refused a block hold ({why}), so this PC may "
                  "suspend in the middle of a load or a render.",
                  "Let this user take the hold with a polkit rule, for example in "
                  "/etc/polkit-1/rules.d/50-auradream-inhibit.rules (as root):\n" + polkit_rule(who))]


# --- Hugging Face sign-in -----------------------------------------------------

def check_hf_login(token_path: Path | None = None, environ: dict[str, str] | None = None) -> list[Check]:
    """Whether a Hugging Face token is present: the token file's existence or
    HF_TOKEN (or the legacy HUGGING_FACE_HUB_TOKEN) being set. Never read, printed or checked with the Hub."""
    environ = dict(os.environ) if environ is None else environ
    if token_path is None:
        token_path = hf_sign_in.default_token_path()
    found = hf_sign_in.token_source(token_path, environ)
    if found == "env":
        return [Check(PASS, "Hugging Face", "a token is set in HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
                      "(not checked with the Hub here).")]
    if found == "file":
        return [Check(PASS, "Hugging Face", f"signed in: a token file is in {token_path.parent} "
                      "(not checked with the Hub here).")]
    return [Check(WARN, "Hugging Face", "not signed in. Open checkpoints such as Animagine XL 4.0 need no "
                  "sign-in; the default, FLUX.1 Krea [dev], does.",
                  "For a gated checkpoint, accept its licence on its Hugging Face page, then "
                  f"{hf_sign_in.login_instructions()} (docs/setup.md step 6).")]


# --- Running ------------------------------------------------------------------

CHECKS: tuple[Callable[[], list[Check]], ...] = (
    check_driver, check_gpu, check_vram, check_ram, check_disk, check_port, check_commit, check_lan,
    check_passcode,
    check_tls,
    check_keep_awake,
    check_hf_login,
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
        description="Read-only checks of this machine: GPU, driver, VRAM, RAM, disk, port, commit, LAN settings, "
                    "the passcode, keeping the PC awake, sign-in.",
    ).parse_args(argv)
    print("AuraDream doctor: read-only checks of this machine.", flush=True)
    lines, status = report(run_checks())
    for line in lines:
        print(line, flush=True)
    print("Nothing failed." if status == 0 else "Something failed: see What to do above.", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
