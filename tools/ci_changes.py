"""Select the CPU test matrix from a GitHub push/PR's complete changed range.

Only known documentation paths skip tests. Unknown events, missing history
and empty comparisons keep the full gate; manual runs always request it.
Standard library only: the lint job never installs the application.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath


_PROSE = {".md", ".rst"}
_DOC_IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".pdf"}


def is_documentation(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    # docs/ also contains runnable prototypes, fixtures and Python helpers.
    if path.suffix.lower() in _PROSE:
        return len(path.parts) == 1 or path.parts[0] == "docs"
    return bool(path.parts and path.parts[0] == "docs" and path.suffix.lower() in _DOC_IMAGES)


def _sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value):
        raise ValueError("missing or invalid commit id")
    if not value.strip("0"):
        raise ValueError("new branch has no previous commit")
    return value


def _git(*args: str, cwd: Path) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def full_tests_required(event_name: str, event: dict, *, cwd: Path) -> bool:
    try:
        if event_name == "push":
            base, head = _sha(event["before"]), _sha(event["after"])
        elif event_name == "pull_request":
            pr = event["pull_request"]
            base, head = _sha(pr["base"]["sha"]), _sha(pr["head"]["sha"])
            base = _sha(_git("merge-base", base, head, cwd=cwd).decode("ascii").strip())
        else:
            return True
        # No rename detection: code moved into docs still names the deleted
        # code path. NUL separators preserve spaces, newlines and Unicode.
        diff = _git("diff", "--no-renames", "--name-only", "-z", base, head, "--", cwd=cwd)
        paths = [name.decode("utf-8", errors="surrogateescape") for name in diff.split(b"\0") if name]
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        return True
    return not paths or any(not is_documentation(path) for path in paths)


def main() -> int:
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError):
        event = {}
    full = full_tests_required(os.environ.get("GITHUB_EVENT_NAME", ""), event, cwd=Path.cwd())
    result = f"full_tests={str(full).lower()}"
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(result + "\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
