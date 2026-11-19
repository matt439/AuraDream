"""Linux launcher that keeps the critic session tied to its owning app.

The bootstrap runs in a fresh, single-threaded Python process. It starts
the guard in a separate process group before exec replaces the bootstrap
with the real command, keeping the direct child's PID and exit status.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import time
from pathlib import Path


def session_pids(session: int) -> set[int]:
    """Live processes in a /proc session, including half-exited processes."""
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
            # The first thread can be a zombie before the other threads
            # finish exiting and close the process's files and sockets.
            if fields[0] in ("Z", "X") and len(os.listdir(f"/proc/{entry}/task")) < 2:
                continue
        except OSError:
            continue
        pids.add(int(entry))
    return pids


def signal_session(session: int, pids: set[int], number: int) -> None:
    for pid in pids:
        try:
            descriptor = os.pidfd_open(pid)
            try:
                # Check after opening the handle. A PID from a /proc scan
                # could already have been reused outside our session.
                if os.getsid(pid) == session:
                    signal.pidfd_send_signal(descriptor, number)
            finally:
                os.close(descriptor)
        except (ProcessLookupError, PermissionError):
            pass


def bootstrap_command(command: list[str], owner: int, release: int, receipt: int) -> list[str]:
    return [
        sys.executable, "-I", "-S", str(Path(__file__).resolve()),
        str(owner), str(release), str(receipt), *command,
    ]


def _guard(owner: int, release: int) -> None:
    session = os.getsid(0)
    me = os.getpid()
    watch = select.poll()
    watch.register(owner, select.POLLIN)
    watch.register(release, select.POLLIN)
    # Stay armed until the owner exits or releases us after Stop. A /proc
    # scan alone can miss a wrapper that forks and exits during the scan.
    watch.poll()
    saw_empty = False
    while True:
        pids = session_pids(session) - {me}
        if not pids and saw_empty:
            return
        signal_session(session, pids, signal.SIGKILL)
        saw_empty = not pids
        time.sleep(0.05)


def _start_guard(owner: int, release: int, receipt: int) -> None:
    # Reparent the guard before exec so the actual wrapper cannot wait on
    # it as one of its children. Wait for the starter to install its group.
    starter = os.fork()
    if starter == 0:
        try:
            guard = os.fork()
        except OSError:
            os._exit(1)
        if guard == 0:
            os.close(receipt)
            try:
                os.setpgid(0, 0)
                _guard(owner, release)
            except OSError as exc:
                print(f"Linux critic owner guard failed: {exc}", file=sys.stderr, flush=True)
                os._exit(1)
            os._exit(0)
        try:
            os.setpgid(guard, guard)
            os.write(receipt, f"{guard}\n".encode("ascii"))
        except OSError:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(starter, 0)
    if status:
        raise OSError("the Linux critic owner guard did not start")


def _launch(owner: int, release: int, receipt: int, command: list[str]) -> None:
    os.set_inheritable(receipt, False)
    try:
        _start_guard(owner, release, receipt)
        os.close(owner)
        os.close(release)
        os.execvp(command[0], command)
    except OSError as exc:
        os.write(receipt, str(exc).encode("utf-8", "backslashreplace"))
        sys.exit(1)


if __name__ == "__main__":
    _launch(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
