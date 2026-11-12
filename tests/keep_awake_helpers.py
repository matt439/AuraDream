"""What the keep-awake tests share (`test_keep_awake.py`,
`test_keep_awake_holders.py`): a fake clock and popen for the holders'
children, powershell.exe's stand-in under WSL, the four Win32 calls faked,
an inhibitor that records, and the real `platform_inhibitor` taken before
conftest's autouse stub replaces it for each test. Imported by name.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
from pathlib import Path

from backend import keep_awake

# The real factory, before conftest's autouse stub replaces it for each test.
platform_inhibitor = keep_awake.platform_inhibitor

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Pipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeProc:
    """A systemd-inhibit child: `lives` None runs until its stdin closes (or
    a signal), 0 exits at once with `stderr`, N dies N seconds in."""

    def __init__(self, clock: Clock, lives: float | None = None, *, returncode: int = 1,
                 stderr: bytes = b"", ignores_eof: bool = False) -> None:
        self.clock, self.lives, self.stderr_text = clock, lives, stderr
        self.started = clock()
        self.exit_code = returncode
        self.ignores_eof = ignores_eof
        self.stdin, self.stderr = Pipe(), Pipe()
        self.returncode: int | None = None
        self.signals: list[str] = []

    def poll(self) -> int | None:
        if self.returncode is None:
            if self.stdin.closed and not self.ignores_eof:
                self.returncode = 0
            elif self.lives is not None and self.clock() - self.started >= self.lives:
                self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        code = self.poll()
        if code is None:
            raise subprocess.TimeoutExpired("systemd-inhibit", timeout or 0)
        return code

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        return b"", self.stderr_text

    def terminate(self) -> None:
        self.signals.append("terminate")
        self.returncode = -15

    def kill(self) -> None:
        self.signals.append("kill")
        self.returncode = -9


class Popen:
    def __init__(self, *procs: FakeProc) -> None:
        self.procs = list(procs)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        proc = self.procs.pop(0)
        proc.started = proc.clock()         # its life counts from its start
        return proc


INTEROP = keep_awake.WSL_INTEROP[0]


def _exists(*paths: str):
    return lambda path: path in paths


class Answer:
    """A child's stdout: `line` at the first readline, or (None) nothing
    until the child ends."""

    def __init__(self, proc: WslProc, line: bytes | None) -> None:
        self.proc, self.line, self.closed = proc, line, False

    def readline(self) -> bytes:
        if self.line is not None:
            return self.line
        self.proc.ended.wait(30)
        return b""

    def close(self) -> None:
        self.closed = True


class WslProc(FakeProc):
    """powershell.exe running the hold script: answers `line` (b"held 4242\\r\\n",
    b"refused: ...", b"" for an end before holding, None for silence)."""

    def __init__(self, clock: Clock, line: bytes | None = b"held 4242\r\n", *, lives: float | None = None,
                 returncode: int = 1, ignores_eof: bool = False) -> None:
        super().__init__(clock, lives, returncode=returncode, ignores_eof=ignores_eof)
        self.ended = threading.Event()
        self.stdout = Answer(self, line)
        if line is not None and not line.startswith(b"held"):
            self.returncode = returncode        # it said its word and stopped
            self.ended.set()

    def poll(self) -> int | None:
        code = super().poll()
        if code is not None:
            self.ended.set()
        return code

    def terminate(self) -> None:
        super().terminate()
        self.ended.set()

    def kill(self) -> None:
        super().kill()
        self.ended.set()


def _wsl(*procs: WslProc, clock: Clock | None = None):
    popen = Popen(*procs)
    request = keep_awake.WslPowerRequest("AuraDream is in use", program="/mnt/c/ps/powershell.exe",
                                         popen=popen, monotonic=clock or procs[0].clock)
    return request, popen


class FakePower:
    """The four Win32 calls, recorded; `refuse` names the ones that fail."""

    def __init__(self, *refuse: str) -> None:
        self.refuse = set(refuse)
        self.calls: list[tuple] = []
        self.thread_ids: set[int] = set()
        self.handles = 0

    def _call(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        self.thread_ids.add(threading.get_ident())
        if name in self.refuse:
            raise OSError(f"Windows refused {name} (error 5): Access is denied.")

    def create(self, reason: str):
        self._call("create", reason)
        self.handles += 1
        return 0x100 + self.handles

    def set(self, handle) -> None:
        self._call("set", handle)

    def clear(self, handle) -> None:
        self._call("clear", handle)

    def close(self, handle) -> None:
        self._call("close", handle)

    def named(self) -> list[str]:
        return [call[0] for call in self.calls]


class Recorder:
    """An inhibitor that records, optionally refusing."""

    kind = "systemd-inhibit"
    RETRY_AFTER = 300

    def __init__(self, refuse: str | None = None) -> None:
        self.held = False
        self.refuse = refuse
        self.holds = 0
        self.thread_ids: set[int] = set()

    def hold(self) -> None:
        self.thread_ids.add(threading.get_ident())
        self.holds += 1
        if self.refuse:
            raise OSError(self.refuse)
        self.held = True

    def release(self) -> None:
        self.held = False


async def _until(predicate, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def _run(scenario):
    asyncio.run(asyncio.wait_for(scenario(), 30))
