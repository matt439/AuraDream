"""The critic child process, its complete process tree and loopback HTTP transport."""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import critic_runtime

log = logging.getLogger(__name__)


# A cold 4 GB read from disk, then the load; `--fit` may put layers on the CPU
# when the image pipeline is holding VRAM, which is what the long request
# timeout is for (13 tok/s on the CPU against ~150 on the card).
HEALTH_TIMEOUT = 180.0
REQUEST_TIMEOUT = 600.0
STOP_TIMEOUT = 10.0
_HEALTH_POLL = 0.25
# A backstop, not the mechanism: a server this process could not stop (it was
# killed outright) drops the model and its VRAM after this long idle.
_IDLE_SLEEP = 120


# --- Running the server ------------------------------------------------------


def server_command(server: Path, model: Path, port: int) -> list[str]:
    """The llama-server command line; flags as documented by b11206's --help.

    Both paths absolute: `_spawn` starts the server in its own directory, where
    a relative `-m` - or, on Linux, a relative executable - means another file.
    No `--api-key`: a command line is readable by every local user (under WSL,
    `/proc/<pid>/cmdline`), so the key goes in the environment (`_server_env`).
    """
    return [
        str(server.resolve()),
        "-m", str(model.resolve()),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-ui",
        # Never a network fetch of its own: the model is a local file.
        "--offline",
        "-c", str(critic_runtime.CONTEXT),
        "-np", "1",
        "--jinja",
        # Places GPU layers by the VRAM actually free, rather than failing when
        # the pipeline holds some. b11206 adjusts only arguments left unset, so
        # the context stays at `-c` and layers that do not fit go to the CPU.
        "--fit", "on",
        "--reasoning-budget", str(critic_runtime.REASONING_BUDGET),
        "--sleep-idle-seconds", str(_IDLE_SLEEP),
    ]


def _server_env(server: Path, api_key: str) -> dict[str, str]:
    """The server's environment: this process's, less every `LLAMA_*` - a
    user's settings for their own servers (`LLAMA_ARG_TOOLS` would switch on
    its shell and file tools) - plus this start's key as `LLAMA_API_KEY`, which
    b11206 reads in place of `--api-key`."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_")}
    env["LLAMA_API_KEY"] = api_key
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(server.parent), env.get("LD_LIBRARY_PATH")])
        )
    return env


_WIN32: Any = None


def _win32() -> Any:
    """kernel32's job object and process calls, ntdll's resume and iphlpapi's
    TCP table, typed once."""
    global _WIN32
    if _WIN32 is None:
        import ctypes
        from ctypes import wintypes

        class Limits(ctypes.Structure):
            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, its basic part flattened.
            _fields_ = [
                ("per_process_time", ctypes.c_int64),
                ("per_job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority_class", wintypes.DWORD),
                ("scheduling_class", wintypes.DWORD),
                ("io_counters", ctypes.c_uint64 * 6),
                ("process_memory_limit", ctypes.c_size_t),
                ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        class ProcessIds(ctypes.Structure):
            # JOBOBJECT_BASIC_PROCESS_ID_LIST, with room for a wrapper's tree.
            _fields_ = [
                ("assigned", wintypes.DWORD),
                ("listed", wintypes.DWORD),
                ("ids", ctypes.c_size_t * 64),
            ]

        class Row4(ctypes.Structure):
            # MIB_TCPROW_OWNER_PID; the port is in network order.
            _fields_ = [
                ("state", wintypes.DWORD),
                ("address", ctypes.c_ubyte * 4),
                ("port", ctypes.c_ubyte * 4),
                ("remote_address", wintypes.DWORD),
                ("remote_port", wintypes.DWORD),
                ("pid", wintypes.DWORD),
            ]

        class Row6(ctypes.Structure):
            # MIB_TCP6ROW_OWNER_PID.
            _fields_ = [
                ("address", ctypes.c_ubyte * 16),
                ("scope", wintypes.DWORD),
                ("port", ctypes.c_ubyte * 4),
                ("remote_address", ctypes.c_ubyte * 16),
                ("remote_scope", wintypes.DWORD),
                ("remote_port", wintypes.DWORD),
                ("state", wintypes.DWORD),
                ("pid", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")
        iphlpapi = ctypes.WinDLL("iphlpapi")
        handle, info = wintypes.HANDLE, ctypes.c_int
        for name, argtypes, restype in (
            ("CreateJobObjectW", (ctypes.c_void_p, wintypes.LPCWSTR), handle),
            (
                "SetInformationJobObject",
                (handle, info, ctypes.c_void_p, wintypes.DWORD),
                wintypes.BOOL,
            ),
            ("AssignProcessToJobObject", (handle, handle), wintypes.BOOL),
            ("TerminateJobObject", (handle, wintypes.UINT), wintypes.BOOL),
            (
                "QueryInformationJobObject",
                (handle, info, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)),
                wintypes.BOOL,
            ),
            ("CloseHandle", (handle,), wintypes.BOOL),
            ("OpenProcess", (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD), handle),
            ("IsProcessInJob", (handle, handle, ctypes.POINTER(wintypes.BOOL)), wintypes.BOOL),
            ("WaitForSingleObject", (handle, wintypes.DWORD), wintypes.DWORD),
        ):
            function = getattr(kernel32, name)
            function.argtypes, function.restype = argtypes, restype
        # Undocumented, but ntdll's since NT: resumes every thread of a
        # process from the process handle, which is all Popen keeps.
        ntdll.NtResumeProcess.argtypes = (handle,)
        ntdll.NtResumeProcess.restype = ctypes.c_long
        iphlpapi.GetExtendedTcpTable.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL, wintypes.ULONG,
            ctypes.c_int, wintypes.ULONG,
        )
        iphlpapi.GetExtendedTcpTable.restype = wintypes.DWORD
        _WIN32 = SimpleNamespace(
            ctypes=ctypes, wintypes=wintypes, kernel32=kernel32, ntdll=ntdll,
            iphlpapi=iphlpapi, Limits=Limits, ProcessIds=ProcessIds,
            rows={socket.AF_INET: Row4, socket.AF_INET6: Row6},
        )
    return _WIN32


def _new_job(process: _ServerProcess) -> int | None:
    """A job object holding `process`, which kills everything in it when its
    last handle closes; None, and the child alone, if Windows refuses one."""
    win = _win32()
    job = win.kernel32.CreateJobObjectW(None, None)
    if not job:
        log.warning("no job object for llama-server (error %d)", win.ctypes.get_last_error())
        return None
    limits = win.Limits(flags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not (
        win.kernel32.SetInformationJobObject(
            job, 9, win.ctypes.byref(limits), win.ctypes.sizeof(limits)
        )
        and win.kernel32.AssignProcessToJobObject(job, int(process._handle))
    ):
        log.warning(
            "llama-server is not in a job object (error %d); only it is stopped, not "
            "anything it starts", win.ctypes.get_last_error(),
        )
        win.kernel32.CloseHandle(job)
        return None
    return job


def _job_pids(job: int) -> set[int]:
    """The processes still in `job`: empty once every one has exited."""
    win = _win32()
    listed = win.ProcessIds()
    # ERROR_MORE_DATA past 64 processes still fills the first 64.
    win.kernel32.QueryInformationJobObject(
        job, 3, win.ctypes.byref(listed), win.ctypes.sizeof(listed), None
    )
    return {int(pid) for pid in listed.ids[: min(listed.listed, len(listed.ids))]}


def _waitable(pid: int, job: int) -> int | None:
    """A handle to wait on process `pid` of `job` with; None if it is gone,
    or if the pid is no longer that job's."""
    win = _win32()
    # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
    handle = win.kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        return None
    inside = win.wintypes.BOOL()
    if win.kernel32.IsProcessInJob(handle, job, win.ctypes.byref(inside)) and inside:
        return handle
    win.kernel32.CloseHandle(handle)
    return None


def _release(job: int | None, held: dict[int, int]) -> None:
    """Close what a `_ServerProcess` holds on Windows; the job last, whose
    closing kills whatever is still in it."""
    win = _win32()
    for handle in held.values():
        win.kernel32.CloseHandle(handle)
    held.clear()
    if job is not None:
        win.kernel32.CloseHandle(job)


def _session_pids(session: int) -> set[int]:
    """The live processes in a session, by /proc: a POSIX child started in a
    session of its own and everything it starts, whatever process group each
    is in. Empty without /proc.

    A zombie holds nothing and is left out - but only once it is one whole:
    /proc shows a process by its first thread, which can be a zombie while
    the others, still exiting, hold the process's files and sockets."""
    pids = set()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return pids
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as stat:
                # state, ppid, pgrp, session after the parenthesised name.
                fields = stat.read().rpartition(")")[2].split()
            if len(fields) < 4 or fields[3] != str(session):
                continue
            if fields[0] in ("Z", "X") and len(os.listdir(f"/proc/{entry}/task")) < 2:
                continue
        except OSError:
            continue
        pids.add(int(entry))
    return pids


class _ServerProcess(subprocess.Popen):
    """llama-server and whatever it starts, as one tree.

    `LOCAL_DIFFUSION_CRITIC_SERVER` may name a wrapper - a .bat, a venv
    trampoline, a shell script that does not `exec` - whose server is a
    grandchild, which stopping the direct child alone left listening, and
    holding the card, after the gate opened. So `terminate`, `kill` and
    `wait` here reach the whole tree, the same tree `pids()` names for the
    listener check. On Windows it is a job object, joined before the child
    runs - it starts suspended and is resumed once inside - which also kills
    the tree when its handle closes, as a hard kill of this app closes it.
    On POSIX it is a session of the child's own, whose id is the child's
    pid; on Linux every process in it is signalled and waited for, whatever
    group it has moved to, and elsewhere the child's process group. Nothing
    ends that session with the app: an app killed outright, or by its
    terminal's hang-up (which no longer reaches the server), leaves it to
    `--sleep-idle-seconds` to free the card.
    """

    # Popen creates this Windows-only integer Handle before the suspended
    # child is assigned to its job. Its cross-platform stub omits the field.
    _handle: int

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self._job: int | None = None
        # Windows: a handle to each process seen in the job, to wait on.
        self._held: dict[int, int] = {}
        if sys.platform != "win32":
            super().__init__(command, start_new_session=True, **kwargs)
            return
        flags = kwargs.pop("creationflags", 0) | 0x4  # CREATE_SUSPENDED
        super().__init__(command, creationflags=flags, **kwargs)
        try:
            self._job = _new_job(self)
            status = _win32().ntdll.NtResumeProcess(int(self._handle))
            if status < 0:
                raise OSError(f"could not resume the process (NTSTATUS {status & 0xFFFFFFFF:#010x})")
        except BaseException:
            # Never a suspended child left behind: Popen's own kill is
            # TerminateProcess, and closing the job kills it again.
            super().kill()
            _release(self._job, self._held)
            self._job = None
            raise
        weakref.finalize(self, _release, self._job, self._held)

    def pids(self) -> set[int]:
        """The tree's process ids, the child's own included."""
        if self._job is not None:
            return {self.pid} | _job_pids(self._job)
        if sys.platform.startswith("linux"):
            return {self.pid} | _session_pids(self.pid)
        return {self.pid}

    def _hold(self) -> set[int]:
        """The processes still in the job, each with a handle to wait on
        taken when first seen. A process leaves the job's list before its
        handles - the log, the listening socket - are closed; its own handle
        is signalled only after they are."""
        job = self._job
        assert job is not None  # Called only while this process owns a Windows job.
        pids = _job_pids(job)
        for pid in pids - self._held.keys():
            handle = _waitable(pid, job)
            if handle is not None:
                self._held[pid] = handle
        return pids

    def _signal_tree(self, number: int) -> None:
        if sys.platform == "win32":
            if self._job is not None:
                self._hold()
                if _win32().kernel32.TerminateJobObject(self._job, 1):
                    return
            super().kill()
            return
        try:
            os.killpg(self.pid, number)
        except OSError:
            # Nothing left in the group, or nothing this user may signal.
            super().send_signal(number)
        # The rest of the session: a wrapper may have put its server in a
        # group of its own (`set -m`), which `pids()` still counts.
        for pid in _session_pids(self.pid) - {self.pid}:
            try:
                os.kill(pid, number)
            except OSError:
                pass

    def terminate(self) -> None:
        # On Windows both end the job at once, as Popen's own terminate is a
        # kill there too.
        self._signal_tree(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_tree(getattr(signal, "SIGKILL", signal.SIGTERM))

    def _others_alive(self) -> bool:
        if self._job is not None:
            return bool(self._hold())
        if sys.platform == "win32":
            return False
        if sys.platform.startswith("linux"):
            return bool(_session_pids(self.pid))
        try:
            os.killpg(self.pid, 0)
        except OSError:
            return False
        return True

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the child, then for the rest of its tree - whose exit is
        what frees the card when the child is only a wrapper - and on
        Windows until each process has closed its handles, not only left
        the job's list."""
        deadline = None if timeout is None else time.monotonic() + timeout
        code = super().wait(timeout)
        while self._others_alive():
            if deadline is not None and time.monotonic() >= deadline:
                assert timeout is not None
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.05)
        for pid, handle in list(self._held.items()):
            win = _win32()
            left = 0xFFFFFFFF if deadline is None else (deadline - time.monotonic()) * 1000
            if win.kernel32.WaitForSingleObject(handle, max(0, int(left))) == 0x102:
                assert timeout is not None  # An INFINITE wait cannot report WAIT_TIMEOUT.
                raise subprocess.TimeoutExpired(self.args, timeout)  # WAIT_TIMEOUT
            win.kernel32.CloseHandle(handle)
            del self._held[pid]
        return code


def _spawn(command: list[str], log_file: Any, api_key: str) -> _ServerProcess:
    """Start the server as a `_ServerProcess`. The seam the tests replace."""
    server = Path(command[0])
    return _ServerProcess(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=server.parent,
        env=_server_env(server, api_key),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


# A proxy from the environment must never see loopback traffic - it would be
# handed the prompt and the key, or answer for a server it cannot reach.
_LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(
    method: str, url: str, api_key: str | None, body: Any | None, timeout: float
) -> tuple[int, Any]:
    """One HTTP exchange with the server: (status, decoded body). The seam the
    tests replace."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _LOOPBACK.open(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, raw.decode("utf-8", "replace")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# Where a listener takes connections to 127.0.0.1: bound there, to every
# address, or (dual-stack) to IPv6's.
_REACHES_LOOPBACK = {
    socket.AF_INET: frozenset({"127.0.0.1", "0.0.0.0"}),
    socket.AF_INET6: frozenset({"::", "::ffff:127.0.0.1"}),
}


def _windows_listeners(port: int) -> set[int] | None:
    """The pids listening where 127.0.0.1:`port` goes, from the TCP tables."""
    win = _win32()
    ctypes = win.ctypes
    owners: set[int] = set()
    for family, row_type in win.rows.items():
        size = ctypes.c_uint32(0)
        table = None
        for _ in range(4):
            # TCP_TABLE_OWNER_PID_LISTENER; ERROR_INSUFFICIENT_BUFFER says how
            # much the table needs, which may grow between calls.
            code = win.iphlpapi.GetExtendedTcpTable(table, ctypes.byref(size), False, family, 3, 0)
            if code == 0 and table is not None:
                break
            if code not in (0, 122):
                return None
            table = ctypes.create_string_buffer(size.value)
        else:
            return None
        count = ctypes.c_uint32.from_buffer(table).value
        for row in (row_type * count).from_buffer(table, 4):
            address = socket.inet_ntop(family, bytes(row.address))
            if row.port[0] << 8 | row.port[1] == port and address in _REACHES_LOOPBACK[family]:
                owners.add(int(row.pid))
    return owners


def _linux_listeners(port: int) -> set[str] | None:
    """The socket inodes listening where 127.0.0.1:`port` goes, from /proc."""
    inodes: set[str] = set()
    for family, name in ((socket.AF_INET, "tcp"), (socket.AF_INET6, "tcp6")):
        try:
            with open(f"/proc/net/{name}", encoding="ascii") as table:
                lines = table.read().splitlines()[1:]
        except OSError:
            if family == socket.AF_INET:
                return None
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # TCP_LISTEN
                continue
            address, _, hex_port = fields[1].partition(":")
            # Each 32-bit word printed as the host reads it.
            raw = b"".join(
                int(address[i : i + 8], 16).to_bytes(4, sys.byteorder)
                for i in range(0, len(address), 8)
            )
            reaches = socket.inet_ntop(family, raw) in _REACHES_LOOPBACK[family]
            if int(hex_port, 16) == port and reaches:
                inodes.add(fields[9])
    return inodes


def _socket_inodes(pids: set[int]) -> set[str]:
    inodes: set[str] = set()
    for pid in pids:
        try:
            descriptors = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                link = os.readlink(f"/proc/{pid}/fd/{descriptor}")
            except OSError:
                continue
            if link.startswith("socket:["):
                inodes.add(link[len("socket:[") : -1])
    return inodes


def _listener_is_ours(port: int, pids: set[int]) -> bool | None:
    """Whether what listens where a connection to 127.0.0.1:`port` goes is
    the server's own process tree (`pids`): True; False when anything else
    listens there, or nothing can be seen to; None where this platform gives
    no way to tell. The seam the tests replace.

    Windows names each listener's pid; Linux names its socket inode, which
    only the tree's own descriptors can show - another user's are unreadable,
    and theirs is exactly the listener to refuse.
    """
    try:
        if sys.platform == "win32":
            owners = _windows_listeners(port)
            return None if owners is None else bool(owners) and owners <= pids
        if sys.platform.startswith("linux"):
            inodes = _linux_listeners(port)
            return None if inodes is None else bool(inodes) and inodes <= _socket_inodes(pids)
    except (OSError, ValueError) as exc:
        log.warning("could not see who listens on port %d: %s", port, exc)
    return None


def _tail(path: Path, lines: int = 12) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            source.seek(max(0, source.tell() - 4096))
            text = source.read().decode("utf-8", "replace")
    except OSError:
        return "(no server output)"
    return " | ".join(line.strip() for line in text.splitlines()[-lines:] if line.strip())


def _error_message(payload: Any) -> str:
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            return str(error["message"])
    return str(payload)[:300]


# The Ubuntu build's libraries from outside its tree (its RUNPATH is $ORIGIN),
# by the Ubuntu package that installs each.
_APT_PACKAGES = {"libgomp.so.1": "libgomp1", "libssl.so.3": "libssl3", "libcrypto.so.3": "libssl3"}


def _exited(code: int, log_path: Path) -> str:
    tail = _tail(log_path)
    missing = re.search(r"([\w.+-]+\.so[\w.]*): cannot open shared object file", tail)
    if code == 127 and missing:
        library = missing.group(1)
        package = _APT_PACKAGES.get(library)
        if package is not None:
            return (
                f"llama-server could not start: {library} is missing; install it with "
                f"`sudo apt install {package}` (under WSL, inside the distribution)"
            )
        return f"llama-server could not start: {library} is missing: {tail}"
    return f"llama-server exited with code {code} before it was ready: {tail}"


def _wait_until_healthy(process: subprocess.Popen, base: str, log_path: Path) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while True:
        code = process.poll()
        if code is not None:
            raise critic_runtime.CriticError(_exited(code, log_path))
        try:
            # /health is public and answers 503 while the model loads.
            status, _ = _request("GET", base + "/health", None, None, 2.0)
        except (OSError, http.client.HTTPException):
            status = None
        # Alive as well as answering. That alone does not prove the answer
        # is ours: b11206 binds with SO_REUSEADDR, so a stranger's server that
        # took the port in the gap after `_free_port` need not make ours exit.
        # Nor does `_run`'s keyless /v1/models 401: it refuses a server that
        # ignores its key, but a stranger can answer 401 too. The guard is
        # `_run`'s check that the port's listener is the server's own process
        # tree, before anything carries the key - where the platform can tell.
        if status == 200 and process.poll() is None:
            return
        if time.monotonic() >= deadline:
            raise critic_runtime.CriticError(
                f"llama-server was not ready after {HEALTH_TIMEOUT:g} s: {_tail(log_path)}"
            )
        time.sleep(_HEALTH_POLL)


def _stop(process: subprocess.Popen) -> None:
    """Terminate, wait, kill - the whole tree of a `_ServerProcess`, even when
    the child itself has exited, since a wrapper's server outlives it. A kill
    not waited out either is an error: the card may still be taken."""
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise critic_runtime.CriticError(
                f"llama-server (pid {process.pid}) had not exited {STOP_TIMEOUT:g} s after "
                "it was killed; it may still hold GPU memory"
            ) from exc


def _remove_log(path: Path, attempts: int = 5) -> None:
    """Delete the server's log, retried over a few hundred milliseconds as
    `storage._rename_with_retry` renames: on Windows a scanner or an indexer
    may hold a just-closed file (WinError 32). A log still held after that -
    by one of those, or a process outside the tree - is only logged: it is
    not worth the answer it would replace."""
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            if attempt == attempts - 1:
                log.warning("could not remove llama-server's log %s: %s", path, exc)
                return
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            log.warning("could not remove llama-server's log %s: %s", path, exc)
            return
