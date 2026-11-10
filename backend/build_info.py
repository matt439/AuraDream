"""The short Git commit of this checkout, read by asking git once.

`main` reads it at import, beside `STARTED_AT`, so `/api/status` and the
start-up log name the commit the running process started from: a
`git pull` without a restart moves the checkout's commit and leaves the
process's where it was. The doctor reads the checkout's own and compares
the two. Standard library only, and no `config` import.

None whenever it cannot be told: no `.git` in the project folder (a copied
or unpacked tree), git not on the PATH, git refusing the folder, or an
answer that is not a commit. Never an error: nothing depends on it.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Seconds git may take; `rev-parse` on a local checkout answers at once.
GIT_SECONDS = 5
_COMMIT = re.compile(r"[0-9a-f]{4,40}")


def read_commit(root: Path | None = None, run: Callable[..., Any] = subprocess.run) -> str | None:
    """`git rev-parse --short HEAD` in `root` (this checkout by default), or
    None. Only where `root` itself holds `.git` (a folder, or a worktree's
    file): a tree unpacked inside some other repository is not that
    repository's commit."""
    root = ROOT if root is None else root
    if not (root / ".git").exists():
        return None
    options: dict[str, Any] = {}
    if sys.platform == "win32":
        # A server started without a console must not flash one for git.
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=GIT_SECONDS, **options,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    answer = (done.stdout or "").strip()
    return answer if _COMMIT.fullmatch(answer) else None


def same_commit(one: str, other: str) -> bool:
    """Whether two short commits name the same one: git may abbreviate to
    different lengths, so either may be the other's beginning."""
    return one.startswith(other) or other.startswith(one)
