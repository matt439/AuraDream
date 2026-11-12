"""Keeping the PC awake while AuraDream is busy (Linux through systemd-inhibit,
Windows through a power request, WSL through Windows' power request).

A PC that suspends after a spell of idleness (logind's IdleAction, Windows'
sleep timer) sees no keyboard or mouse while a browser on another machine
drives it, so without this it could suspend in the middle of a load, a
render or a gallery bundle. The controller (`KeepAwake`) holds the
platform's holder while:

- any provider returns a reason (main.py registers the job queue, the card's
  holder, the model load, a switch and a bundle operation): each is a
  callable returning a short reason or None, read on the event loop only -
  `JobManager.active()` is not safe from another thread;
- a counted hold is open (`busy(reason)`, thread-safe): the model load's and
  the critic's downloads run on daemon threads that outlive a cancelled
  request, so they hold for as long as the thread works;
- for `idle_minutes` after the last request a person made
  (`counts_as_activity`: never the page's own timed polls, so an idle open
  tab does not hold the PC awake for ever). Starting the server counts.

On Linux the hold is a child process, `systemd-inhibit --what=sleep:idle
--who=AuraDream --why=<reason> --mode=block cat`, whose `cat` reads a pipe
only this process holds. Releasing closes the pipe (then terminate, then
kill), and so does this process dying in any way, so nothing is ever left
holding the PC awake. A child that exits within START_WAIT was refused (polkit,
or no logind or system bus), as is one that dies within SHORT_LIFE; the
refusal is logged once and tried again after RETRY_AFTER.

On Windows the hold is a power request (`PowerRequest`, kind
"power-request"): a handle from `PowerCreateRequest` carrying the reason,
with `PowerSetRequest(PowerRequestSystemRequired)` set on it. That asks
Windows not to sleep on its idle timer and nothing else: the display may
still turn off, and a sleep someone asks for (the Start menu, the power
button, a lid) still happens. Releasing clears the request and closes the
handle, and Windows closes it when this process dies in any way. The handle
belongs to the process, not a thread, which is why it is this and not
`SetThreadExecutionState`: hold and release run on different threads. No
power setting or plan is read or changed.

Under WSL (`under_wsl`) the server is Linux but the PC that sleeps is
Windows: `systemd-inhibit` there blocks only the VM's logind, never the
Windows host's sleep timer, so it is never taken. The hold is the same power
request, made by Windows' own `powershell.exe` through WSL interop
(`WslPowerRequest`, kind "wsl-power-request"): a small script that sets the
request, says `held`, and reads its stdin, a pipe only this process holds,
until it ends; then it clears the request and exits. Releasing closes the
pipe (then terminate, then kill), and so does this process dying in any
way; Windows closes the handle when powershell.exe ends however it ends.
Without interop or powershell.exe, nothing is held and the status says why
(`NoInhibitor.detail`).

Elsewhere, or on Linux with no systemd-inhibit on the PATH, nothing is held
(`NoInhibitor`, kind "none").

The controller ticks every INTERVAL seconds as an asyncio task, and at once
when a counted hold starts or a request names new work (`wake`, safe from any
thread). The blocking hold and release run through `asyncio.to_thread`.
Nothing here raises into its callers. `hold_while(reason)` is the same
primitive for a command-line process (`backend.generate`), with no controller.

Standard library only: the doctor and the CLI import it without the app.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any, Protocol

log = logging.getLogger("auradream.keep_awake")

INTERVAL = 15.0                     # seconds between checks
DEFAULT_IDLE_MINUTES = 20
WHO = "AuraDream"
REASON = "AuraDream is in use"

# The page's own timers, which an idle open tab keeps sending: the status
# poll (web/app.js, every 3 s), the bundle dialog's status and automatic
# backup reads (web/bundles.js, web/automatic-backup.js; their prefixes are
# `bundle_manager.PREFIX` and `BACKUP_PREFIX`), the job streams and listings
# that follow the status (web/job-stream.js, web/job-queue.js) and an
# inpaint's mask keep-alive (web/inpaint.js, every 20 minutes).
_POLLED_GETS = frozenset({
    "/api/status",
    "/api/gallery/bundles/status",
    "/api/gallery/automatic-backup",
})
# A timed read the page sends only because an earlier one failed, or because
# a status poll said its answer changed (web/model-switch.js's catalogue
# retry and cache facts, web/model-shots.js's retry), carries this header
# (`POLL_HEADER` in web/state.js): a GET with it is the page's own timer,
# never someone using it. Only GETs: a write always counts.
POLL_HEADER = "x-auradream-poll"
# The page's files: a page load is counted by "/" and its API calls, and a
# cached page's revalidations or a thumbnail are not someone using it, nor
# the browser's own check for a newer service worker (/sw.js).
_FILE_PREFIXES = ("/static/", "/outputs/")


def counts_as_activity(method: str, path: str, *, polled: bool = False) -> bool:
    """True for a request that is someone using AuraDream: page loads and API
    calls. Not HEAD or OPTIONS, the page's timed polls (above, and a GET
    marked `polled` by POLL_HEADER), the job streams, the mask keep-alive,
    nor a GET of the page's files."""
    method = (method or "").upper()
    if method in ("HEAD", "OPTIONS"):
        return False
    path = (path or "/").split("?", 1)[0]
    path = path.rstrip("/") or "/"
    if method == "GET":
        if polled:
            return False
        if path in _POLLED_GETS or path == "/api/jobs" or path.startswith("/api/jobs/"):
            return False
        if path.startswith(_FILE_PREFIXES) or path in ("/favicon.ico", "/sw.js"):
            return False
    if method == "POST" and path.startswith("/api/masks/") and path.endswith("/keep"):
        return False
    return True


def scope_counts_as_activity(scope: Any) -> bool:
    """`counts_as_activity` for an ASGI http scope, with its POLL_HEADER
    read (the middleware below and the fake server's recorder share it)."""
    marker = POLL_HEADER.encode("latin-1")
    polled = any(name.lower() == marker for name, _ in scope.get("headers") or ())
    return counts_as_activity(scope.get("method", ""), scope.get("path", "/"), polled=polled)


class ActivityMiddleware:
    """ASGI: note each request `counts_as_activity` passes on the installed
    controller, and after a write check again at once, so a job or a load
    it started is held now rather than at the next tick. Every other
    request, the job streams included, passes straight through."""

    def __init__(self, app: Any, get_controller: Callable[[], KeepAwake | None]) -> None:
        self.app = app
        self.get_controller = get_controller

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        controller = self.get_controller() if scope.get("type") == "http" else None
        method = scope.get("method", "") if controller is not None else ""
        if controller is None or not scope_counts_as_activity(scope):
            await self.app(scope, receive, send)
            return
        controller.note_activity()
        try:
            await self.app(scope, receive, send)
        finally:
            if method.upper() != "GET":
                controller.wake()


# ------------------------------------------------------------ the inhibitor


class Holder(Protocol):
    kind: str

    @property
    def held(self) -> bool: ...

    def hold(self) -> None: ...

    def release(self) -> None: ...


class Inhibitor:
    """A block inhibitor: a systemd-inhibit child running `cat` on a pipe only
    this process holds (the module docstring says why). hold() and release()
    are idempotent and safe from any thread; held is True only while the
    child runs. hold() raises OSError, in words, when the hold was refused.
    popen and monotonic are injectable for tests."""

    kind = "systemd-inhibit"
    START_WAIT = 0.5                # seconds: a child still running by then holds
    STOP_WAIT = 2.0                 # seconds for each step of release()
    RETRY_AFTER = 300.0             # seconds before trying again after a refusal
    SHORT_LIFE = 60.0               # seconds: a child that stopped sooner was refused

    def __init__(self, reason: str = REASON, *, program: str = "systemd-inhibit",
                 popen: Callable[..., Any] | None = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.reason, self.program = reason, program
        self._popen = popen or subprocess.Popen
        self._monotonic = monotonic
        self._proc: Any = None
        self._started = 0.0
        self._refused: tuple[str, float] | None = None
        self._lock = threading.Lock()

    def argv(self) -> list[str]:
        return [self.program, "--what=sleep:idle", f"--who={WHO}", f"--why={self.reason}",
                "--mode=block", "cat"]

    @property
    def held(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def hold(self) -> None:
        with self._lock:
            if self.held:
                return
            if self._proc is not None:      # it died (logind restarted, say): start another
                proc, self._proc = self._proc, None
                if self._monotonic() - self._started < self.SHORT_LIFE:
                    # A refusal that came after START_WAIT: count it as one,
                    # so it is not tried again at every check.
                    why = _first_error_line(proc)
                    _close(proc)
                    self._refused = (f"systemd-inhibit stopped at once (exit code {proc.returncode}): {why}",
                                     self._monotonic())
                    raise OSError(self._refused[0])
                _close(proc)
            if self._refused is not None and self._monotonic() - self._refused[1] < self.RETRY_AFTER:
                raise OSError(self._refused[0])
            try:
                proc = self._popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, close_fds=True)
            except OSError as exc:
                self._refused = (f"systemd-inhibit could not be started: {exc}", self._monotonic())
                raise OSError(self._refused[0]) from None
            try:
                proc.wait(timeout=self.START_WAIT)
            except subprocess.TimeoutExpired:
                self._proc, self._refused, self._started = proc, None, self._monotonic()
                return
            why = _first_error_line(proc)
            _close(proc)
            self._refused = (f"systemd-inhibit refused (exit code {proc.returncode}): {why}", self._monotonic())
            raise OSError(self._refused[0])

    def release(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                if proc.stdin is not None:
                    proc.stdin.close()      # cat reads the end of its input; systemd-inhibit lets go
            except OSError:
                pass
            for stop in (None, proc.terminate, proc.kill):
                try:
                    if stop is not None:
                        stop()
                    proc.wait(timeout=self.STOP_WAIT)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except OSError:
                    break
            _close(proc)


def _first_error_line(proc: Any) -> str:
    try:
        _, err = proc.communicate(timeout=1)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        err = b""
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    return next((text.strip() for text in (err or "").splitlines() if text.strip()), "it said nothing")


def _close(proc: Any) -> None:
    for pipe in (proc.stdin, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except OSError:
            pass


class PowerApi(Protocol):
    """The four Win32 calls a power request needs, each raising OSError in
    words when Windows refuses it. `handle` is whatever `create` returned."""

    def create(self, reason: str) -> Any: ...

    def set(self, handle: Any) -> None: ...

    def clear(self, handle: Any) -> None: ...

    def close(self, handle: Any) -> None: ...


# winnt.h: POWER_REQUEST_CONTEXT_VERSION, POWER_REQUEST_CONTEXT_SIMPLE_STRING
# and POWER_REQUEST_TYPE's PowerRequestSystemRequired (the idle sleep timer;
# 0 would be the display, which is left alone).
_CONTEXT_VERSION = 0
_CONTEXT_SIMPLE_STRING = 0x1
_SYSTEM_REQUIRED = 1


class _ReasonDetailed(ctypes.Structure):
    # ULONG is 32 bits on Windows; c_ulong is not on 64-bit Linux, where
    # the tests still measure this.
    _fields_ = [("module", ctypes.c_void_p), ("id", ctypes.c_uint32),
                ("count", ctypes.c_uint32), ("strings", ctypes.c_void_p)]


class _Reason(ctypes.Union):
    # The localized member is the larger one, so the structure is as large
    # as Windows expects; only the simple string is ever filled.
    _fields_ = [("detailed", _ReasonDetailed), ("simple", ctypes.c_wchar_p)]


class _ReasonContext(ctypes.Structure):
    """REASON_CONTEXT (minwinbase.h)."""

    _fields_ = [("version", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("reason", _Reason)]


class Kernel32Power:
    """`PowerApi` on kernel32, bound at the first call so that importing
    this module, or building a `PowerRequest`, loads no DLL."""

    def __init__(self) -> None:
        self._bound: Any = None

    def _kernel32(self) -> Any:
        if self._bound is None:
            if sys.platform != "win32":
                raise OSError("power requests are a Windows call")
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.PowerCreateRequest.argtypes = [ctypes.c_void_p]
            kernel32.PowerCreateRequest.restype = ctypes.c_void_p
            for name in ("PowerSetRequest", "PowerClearRequest"):
                call = getattr(kernel32, name)
                call.argtypes, call.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            self._bound = kernel32
        return self._bound

    @staticmethod
    def _refused(call: str) -> OSError:
        if sys.platform != "win32":
            return OSError(f"Windows refused {call}")
        code = ctypes.get_last_error()
        text = ctypes.FormatError(code).strip() or "no reason given"
        return OSError(f"Windows refused {call} (error {code}): {text}")

    def create(self, reason: str) -> Any:
        kernel32 = self._kernel32()
        context = _ReasonContext()
        context.version, context.flags = _CONTEXT_VERSION, _CONTEXT_SIMPLE_STRING
        context.reason.simple = reason
        handle = kernel32.PowerCreateRequest(ctypes.byref(context))
        # NULL, or INVALID_HANDLE_VALUE (-1 as a pointer).
        if not handle or handle == ctypes.c_void_p(-1).value:
            raise self._refused("PowerCreateRequest")
        return handle

    def set(self, handle: Any) -> None:
        if not self._kernel32().PowerSetRequest(handle, _SYSTEM_REQUIRED):
            raise self._refused("PowerSetRequest")

    def clear(self, handle: Any) -> None:
        if not self._kernel32().PowerClearRequest(handle, _SYSTEM_REQUIRED):
            raise self._refused("PowerClearRequest")

    def close(self, handle: Any) -> None:
        if not self._kernel32().CloseHandle(handle):
            raise self._refused("CloseHandle")


class PowerRequest:
    """Windows' hold: a power request handle with system-required set (the
    module docstring says what that does and does not stop). hold() and
    release() are idempotent and safe from any thread; held is True from a
    hold that worked until its release. hold() raises OSError, in Windows'
    words, when a call was refused, and does not ask again until RETRY_AFTER
    has passed. release() always forgets the handle, closing it even where
    clearing was refused. api and monotonic are injectable for tests."""

    kind = "power-request"
    RETRY_AFTER = 300.0             # seconds before trying again after a refusal

    def __init__(self, reason: str = REASON, *, api: PowerApi | None = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.reason = reason
        self._api: PowerApi = api if api is not None else Kernel32Power()
        self._monotonic = monotonic
        self._handle: Any = None
        self._refused: tuple[str, float] | None = None
        self._lock = threading.Lock()

    @property
    def held(self) -> bool:
        return self._handle is not None

    def hold(self) -> None:
        with self._lock:
            if self._handle is not None:
                return
            if self._refused is not None and self._monotonic() - self._refused[1] < self.RETRY_AFTER:
                raise OSError(self._refused[0])
            handle = None
            try:
                handle = self._api.create(self.reason)
                self._api.set(handle)
            except Exception as exc:  # noqa: BLE001 - a missing DLL entry is a refusal too
                if handle is not None:
                    try:
                        self._api.close(handle)
                    except Exception:  # noqa: BLE001
                        pass
                self._refused = (str(exc) or type(exc).__name__, self._monotonic())
                raise OSError(self._refused[0]) from None
            self._handle, self._refused = handle, None

    def release(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is None:
                return
            # Closing ends the request whatever clearing said.
            for step in (self._api.clear, self._api.close):
                try:
                    step(handle)
                except Exception as exc:  # noqa: BLE001 - logged, never raised
                    log.warning("Keep awake: letting go of the power request: %s", exc)


# --------------------------------------------------- Windows' hold from WSL

# The binfmt entries WSL registers to run Windows programs; absent when
# interop is off (`[interop] enabled=false` in /etc/wsl.conf). Newer WSL
# names it WSLInterop-late.
WSL_INTEROP = ("/proc/sys/fs/binfmt_misc/WSLInterop", "/proc/sys/fs/binfmt_misc/WSLInterop-late")
# Where Windows PowerShell 5.1 sits when the Windows PATH is not appended
# (`appendWindowsPath=false`): the C: drive's default mount.
WINDOWS_POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

WSL_NO_HOLD = ("Under WSL, systemd-inhibit holds only the Linux VM, never Windows, and Windows' "
               "powershell.exe is not reachable through WSL interop, so nothing keeps this PC awake")


def _kernel_release() -> str:
    uname = getattr(os, "uname", None)
    try:
        return uname().release if uname is not None else ""
    except OSError:
        return ""


def under_wsl(*, release: str | None = None, exists: Callable[[str], bool] = os.path.exists) -> bool:
    """True inside WSL: its interop entry, or "microsoft" in the kernel's
    release (5.15.x-microsoft-standard-WSL2, 4.4.0-19041-Microsoft)."""
    release = _kernel_release() if release is None else release
    return "microsoft" in release.lower() or any(exists(path) for path in WSL_INTEROP)


def windows_powershell(*, which: Callable[[str], str | None] = shutil.which,
                       exists: Callable[[str], bool] = os.path.exists) -> str | None:
    """Windows PowerShell as WSL runs it, or None: interop must be on (its
    binfmt entry), then powershell.exe on the PATH or at its default mount."""
    if not any(exists(path) for path in WSL_INTEROP):
        return None
    found = which("powershell.exe")
    if found:
        return found
    return WINDOWS_POWERSHELL if exists(WINDOWS_POWERSHELL) else None


# What powershell.exe runs. The C# is Add-Type's (C# 5 under Windows
# PowerShell 5.1); REASON_CONTEXT is padded to its union's size, as
# `_ReasonContext` is. The script prints one line, `held` or `refused: <why>`,
# then reads its stdin to the end before clearing the request and closing
# the handle; Windows closes the handle anyway when the process ends.
_WSL_HOLD_CS = r"""
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct ReasonContext {
    public uint Version;
    public uint Flags;
    [MarshalAs(UnmanagedType.LPWStr)] public string Reason;
    public IntPtr Detailed1;
    public IntPtr Detailed2;
}
[DllImport("kernel32.dll", SetLastError = true)]
static extern IntPtr PowerCreateRequest(ref ReasonContext context);
[DllImport("kernel32.dll", SetLastError = true)]
static extern bool PowerSetRequest(IntPtr request, int type);
[DllImport("kernel32.dll", SetLastError = true)]
static extern bool PowerClearRequest(IntPtr request, int type);
[DllImport("kernel32.dll", SetLastError = true)]
static extern bool CloseHandle(IntPtr handle);
static Exception Refused(string call) {
    int code = Marshal.GetLastWin32Error();
    return new Exception("Windows refused " + call + " (error " + code + "): "
        + new System.ComponentModel.Win32Exception(code).Message);
}
public static IntPtr Hold(string reason) {
    ReasonContext context = new ReasonContext();
    context.Version = 0;
    context.Flags = 1;
    context.Reason = reason;
    IntPtr request = PowerCreateRequest(ref context);
    if (request == IntPtr.Zero || request == new IntPtr(-1)) throw Refused("PowerCreateRequest");
    if (!PowerSetRequest(request, 1)) {
        Exception refused = Refused("PowerSetRequest");
        CloseHandle(request);
        throw refused;
    }
    return request;
}
public static void Let(IntPtr request) {
    PowerClearRequest(request, 1);
    CloseHandle(request);
}
""".strip("\n")

_WSL_HOLD_PS = r"""$ErrorActionPreference = 'Stop'
try {
    Add-Type -Namespace AuraDream -Name WslHold -MemberDefinition @'
<CS>
'@
    $request = [AuraDream.WslHold]::Hold('<REASON>')
} catch {
    $e = $_.Exception
    while ($e.InnerException) { $e = $e.InnerException }
    [Console]::Out.WriteLine('refused: ' + ($e.Message -replace '\s+', ' '))
    [Console]::Out.Flush()
    exit 1
}
[Console]::Out.WriteLine('held ' + $PID)
[Console]::Out.Flush()
try {
    while ($null -ne [Console]::In.ReadLine()) { }
} finally {
    [AuraDream.WslHold]::Let($request)
}
"""


def wsl_hold_script(reason: str) -> str:
    """The PowerShell that holds a power request until its stdin ends."""
    # A single-quoted PowerShell string: quotes doubled, line breaks gone.
    safe = " ".join(str(reason).split()).replace("'", "''")
    return _WSL_HOLD_PS.replace("<CS>", _WSL_HOLD_CS).replace("<REASON>", safe)


class WslPowerRequest:
    """Windows' power request, held from WSL by Windows' powershell.exe
    through interop (the module docstring says how and why). hold() and
    release() are idempotent and safe from any thread; held is True from a
    `held` answer while the child runs. hold() raises OSError, in Windows'
    words where it gave some, and does not ask again until RETRY_AFTER has
    passed. popen and monotonic are injectable for tests."""

    kind = "wsl-power-request"
    START_WAIT = 30.0               # seconds for `held`: Add-Type compiles at each start
    STOP_WAIT = 5.0                 # seconds for each step of release()
    RETRY_AFTER = 300.0             # seconds before trying again after a refusal
    SHORT_LIFE = 60.0               # seconds: a holder that stopped sooner was refused

    def __init__(self, reason: str = REASON, *, program: str = WINDOWS_POWERSHELL,
                 popen: Callable[..., Any] | None = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.reason, self.program = reason, program
        self._popen = popen or subprocess.Popen
        self._monotonic = monotonic
        self._proc: Any = None
        self.windows_pid: str | None = None
        self._started = 0.0
        self._refused: tuple[str, float] | None = None
        self._lock = threading.Lock()

    def argv(self) -> list[str]:
        # -EncodedCommand carries the script as one argument, untouched by
        # the quoting WSL and Windows apply to a command line; it reads no
        # file and changes no execution policy.
        encoded = base64.b64encode(wsl_hold_script(self.reason).encode("utf-16-le")).decode("ascii")
        return [self.program, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]

    @property
    def held(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _refuse(self, why: str) -> OSError:
        self._refused = (why, self._monotonic())
        return OSError(why)

    def hold(self) -> None:
        with self._lock:
            if self.held:
                return
            if self._proc is not None:      # it stopped (ended on the Windows side, say)
                proc, self._proc = self._proc, None
                _stop(proc, self.STOP_WAIT)
                if self._monotonic() - self._started < self.SHORT_LIFE:
                    raise self._refuse(f"powershell.exe stopped holding at once (exit code {proc.returncode})")
            if self._refused is not None and self._monotonic() - self._refused[1] < self.RETRY_AFTER:
                raise OSError(self._refused[0])
            try:
                proc = self._popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, close_fds=True)
            except OSError as exc:
                raise self._refuse(f"powershell.exe could not be started through WSL interop: {exc}") from None
            line = _first_line(proc, self.START_WAIT)
            words = (line or "").split()
            if words[:1] == ["held"] and proc.poll() is None:
                # Windows' own process id, for the log line.
                self.windows_pid = words[1] if len(words) > 1 else None
                self._proc, self._refused, self._started = proc, None, self._monotonic()
                log.debug("Keep awake: powershell.exe %s holds a power request.", self.windows_pid)
                return
            _stop(proc, self.STOP_WAIT)
            if line is None:
                raise self._refuse(f"powershell.exe gave no answer within {self.START_WAIT:g} s")
            if line.startswith("refused:"):
                raise self._refuse(line[len("refused:"):].strip() or "Windows refused a power request")
            raise self._refuse(f"powershell.exe stopped before holding (exit code {proc.returncode})"
                               + (f": {line}" if line and not words[:1] == ["held"] else ""))

    def release(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is not None:
                _stop(proc, self.STOP_WAIT)


def _first_line(proc: Any, timeout: float) -> str | None:
    """The child's first line of output, stripped: "" at its end, None when
    none came within `timeout`. Read on a daemon thread, which the child's
    end (`_stop`) lets go."""
    box: list[Any] = []

    def read() -> None:
        try:
            box.append(proc.stdout.readline())
        except (OSError, ValueError):
            box.append(b"")

    reader = threading.Thread(target=read, name="keep-awake-wsl-answer", daemon=True)
    reader.start()
    reader.join(timeout)
    if not box:
        return None
    line = box[0]
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    return (line or "").strip()


def _stop(proc: Any, wait: float) -> None:
    """Close the child's stdin (the end of its input lets go), then
    terminate, then kill; then close its pipes. Never raises."""
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    for stop in (None, proc.terminate, proc.kill):
        try:
            if stop is not None:
                stop()
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
        except OSError:
            break
    for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        try:
            if pipe is not None:
                pipe.close()
        except OSError:
            pass


class NoInhibitor:
    """Neither Linux with systemd-inhibit, Windows, nor WSL with interop:
    nothing is held. `detail` says why where that is worth saying (WSL)."""

    kind = "none"
    held = False

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail

    def hold(self) -> None:
        pass

    def release(self) -> None:
        pass


def platform_inhibitor(reason: str = REASON, *, platform: str | None = None,
                       which: Callable[[str], str | None] = shutil.which,
                       wsl: bool | None = None,
                       exists: Callable[[str], bool] = os.path.exists,
                       ) -> Inhibitor | PowerRequest | WslPowerRequest | NoInhibitor:
    """This machine's inhibitor: a power request on Windows; under WSL the
    same request through powershell.exe, never systemd-inhibit; on other
    Linux systemd-inhibit's where it is on the PATH; else one that holds
    nothing. `wsl` None asks `under_wsl`."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return PowerRequest(reason)
    if platform.startswith("linux"):
        if under_wsl(exists=exists) if wsl is None else wsl:
            program = windows_powershell(which=which, exists=exists)
            return WslPowerRequest(reason, program=program) if program else NoInhibitor(WSL_NO_HOLD)
        program = which("systemd-inhibit")
        if program:
            return Inhibitor(reason, program=program)
    return NoInhibitor()


@contextlib.contextmanager
def hold_while(reason: str = REASON, *, inhibitor: Holder | None = None) -> Iterator[Holder]:
    """Keep the PC awake for the with block (a command line's process),
    released at its end. A refusal is logged; it never raises."""
    held: Holder = inhibitor if inhibitor is not None else platform_inhibitor(reason)
    try:
        held.hold()
    except Exception as exc:  # noqa: BLE001 - keeping awake is best effort
        log.warning("Could not keep the PC awake: %s", exc)
    try:
        yield held
    finally:
        try:
            held.release()
        except Exception:  # noqa: BLE001
            pass


def off_status(idle_minutes: int) -> dict[str, Any]:
    """The status when AURADREAM_KEEP_AWAKE=false: nothing runs."""
    return {"kind": "off", "holding": False, "wanted": False, "reasons": [], "detail": None,
            "idle_minutes": idle_minutes}


# ------------------------------------------------------------ the controller


class KeepAwake:
    """Holds the PC awake while AuraDream is busy (the module docstring says
    when). `start()` and `stop()` run on the event loop; `busy()`, `wake()`
    and `note_activity()` are safe from any thread; providers are read on
    the loop only. monotonic is the idle timer's clock (injectable)."""

    def __init__(self, inhibitor: Holder | None = None, *, idle_minutes: int = DEFAULT_IDLE_MINUTES,
                 monotonic: Callable[[], float] = time.monotonic, interval: float = INTERVAL) -> None:
        self.inhibitor: Holder = inhibitor if inhibitor is not None else NoInhibitor()
        self.idle_minutes = idle_minutes
        self.interval = interval
        self._monotonic = monotonic
        self._providers: list[Callable[[], str | None]] = []
        self._lock = threading.Lock()
        self._counted: Counter[str] = Counter()
        # Starting counts as use: someone started it to use it.
        self._last_activity = monotonic()
        self._failed: set[object] = set()       # what has logged its failure once
        self._reasons: list[str] = []
        self._holding = False
        self._detail: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    # -- inputs

    def add_provider(self, fn: Callable[[], str | None]) -> None:
        self._providers.append(fn)

    @contextlib.contextmanager
    def busy(self, reason: str) -> Iterator[None]:
        """Hold while the with block runs, from any thread: the PC is held
        at once (`wake`), not at the next check."""
        with self._lock:
            self._counted[reason] += 1
        self.wake()
        try:
            yield
        finally:
            with self._lock:
                self._counted[reason] -= 1
                if self._counted[reason] <= 0:
                    del self._counted[reason]
            self.wake()

    def note_activity(self) -> None:
        """Someone used AuraDream (a request `counts_as_activity` passes)."""
        self._last_activity = self._monotonic()
        if not self._reasons:
            self.wake()

    def wake(self) -> None:
        """Check again now rather than at the next tick; safe from any thread,
        and nothing before `start()` or after `stop()`."""
        loop, event = self._loop, self._wakeup
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:            # the loop has closed: the process is ending
            pass

    # -- one check

    def idle_seconds(self) -> float:
        """Age of the last real activity, on the shared monotonic clock."""
        return max(0.0, self._monotonic() - self._last_activity)

    def work_reasons(self) -> list[str]:
        """Live providers and counted workers, excluding the idle grace."""
        found: list[str] = []
        for fn in list(self._providers):
            try:
                reason = fn()
            except Exception as exc:  # noqa: BLE001 - logged once per provider
                if fn not in self._failed:
                    self._failed.add(fn)
                    log.warning("Keep awake: a busy check failed (%s: %s); it is ignored.",
                                type(exc).__name__, exc)
                continue
            if reason:
                found.append(str(reason))
        with self._lock:
            found.extend(self._counted)
        return list(dict.fromkeys(found))

    def reasons(self) -> list[str]:
        """Why to hold now: live work and the grace after a real request."""
        found = self.work_reasons()
        if self.idle_minutes > 0 and self.idle_seconds() < self.idle_minutes * 60:
            found.insert(0, f"in use (a request in the last {self.idle_minutes} minutes)")
        return list(dict.fromkeys(found))

    async def check(self) -> bool:
        """One check: hold or release. True while holding. Never raises."""
        reasons = self.reasons()
        self._reasons = reasons
        was = self._holding
        try:
            if reasons:
                await asyncio.to_thread(self.inhibitor.hold)
            else:
                await asyncio.to_thread(self.inhibitor.release)
        except Exception as exc:  # noqa: BLE001 - logged once until a hold works
            self._detail = str(exc) or type(exc).__name__
            if "hold" not in self._failed:
                self._failed.add("hold")
                log.warning("Could not keep the PC awake: %s. Tried again every %d minutes.",
                            self._detail, round(getattr(self.inhibitor, "RETRY_AFTER", 300) / 60))
        self._holding = bool(self.inhibitor.held)
        if self._holding:
            self._failed.discard("hold")
            self._detail = None
        if self._holding and not was:
            log.info("Keeping the PC awake: %s.", "; ".join(reasons))
        elif was and not self._holding:
            log.info("No longer keeping the PC awake.")
        return self._holding

    def status(self) -> dict[str, Any]:
        """The last check's answer, for /api/status (no subprocess): kind
        ("systemd-inhibit", "power-request", "wsl-power-request" or "none"),
        holding (the inhibitor really holds), wanted (there is a reason to; a
        refused hold is wanted, not holding), reasons, detail (the last
        refusal until a hold works, else why nothing can be held: WSL
        without interop) and idle_minutes."""
        reasons = list(self._reasons)
        detail = self._detail or getattr(self.inhibitor, "detail", None)
        return {"kind": self.inhibitor.kind, "holding": self._holding, "wanted": bool(reasons),
                "reasons": reasons, "detail": detail, "idle_minutes": self.idle_minutes}

    # -- the task

    def start(self) -> asyncio.Task:
        """Start ticking on the running loop."""
        why = getattr(self.inhibitor, "detail", None)
        if why:
            log.warning("Keep awake: %s.", why)
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="keep-awake")
        return self._task

    async def _run(self) -> None:
        wakeup = self._wakeup
        assert wakeup is not None
        while True:
            # Cleared before the check: a wake during it is a second check.
            wakeup.clear()
            try:
                await self.check()
            except Exception as exc:  # noqa: BLE001 - never ends the task
                log.warning("Keep awake: a check failed (%s: %s).", type(exc).__name__, exc)
            try:
                await asyncio.wait_for(wakeup.wait(), self.interval)
            except TimeoutError:
                pass

    async def stop(self) -> None:
        """Stop ticking and let go. Never raises."""
        self._loop = None
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            await asyncio.to_thread(self.inhibitor.release)
        except Exception:  # noqa: BLE001
            pass
        if self._holding:
            log.info("No longer keeping the PC awake.")
        self._holding = False
        self._reasons = []


_CURRENT: KeepAwake | None = None


def install(controller: KeepAwake | None) -> None:
    """Make `controller` the one `busy()` reaches (the server's); None removes it."""
    global _CURRENT
    _CURRENT = controller


def current() -> KeepAwake | None:
    return _CURRENT


@contextlib.contextmanager
def busy(reason: str) -> Iterator[None]:
    """The installed controller's counted hold, from any thread; nothing when
    none is installed (the CLI, the tests, AURADREAM_KEEP_AWAKE=false)."""
    controller = _CURRENT
    if controller is None:
        yield
        return
    with controller.busy(reason):
        yield
