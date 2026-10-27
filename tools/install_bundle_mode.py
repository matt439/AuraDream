"""Explicit Windows fixed-NTFS shared-download cutover; no retention activation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import bundle_mode  # noqa: E402


def main(arguments: list[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="the existing bundle destination, never the image directory")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="read the marker/outcome without writes or a quiescence claim")
    action.add_argument("--other-servers-stopped", action="store_true", help=(
        "Operator confirmation that EVERY other server using this destination is stopped; "
        "the tool cannot verify host-wide quiescence. Stop other servers before using this flag."
    ))
    args = parser.parse_args(arguments)
    try:
        if args.check:
            known = bundle_mode.discover(args.directory)
            print(json.dumps({"state": "installed" if known is not None else "not_installed",
                              "directory": str(args.directory.resolve()),
                              "installation_id": known.installation_id if known is not None else None,
                              "read_only": True, "machine_verified": False, "retention_enabled": False}))
            return 0
        result = bundle_mode.install(args.directory, other_servers_stopped=args.other_servers_stopped)
    except (OSError, ValueError, RuntimeError) as exc:
        operation = "Read-only shared-mode check failed." if args.check else "Shared-mode cutover did not finish cleanly. Check its marker/outcome before retrying."
        print(f"{operation} {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1
    print(json.dumps({"state": "installed" if result.created else "already_installed",
                      "directory": str(result.mode.directory), "installation_id": result.mode.installation_id,
                      "other_servers_stopped": "operator_confirmed", "machine_verified": False,
                      "supported_platform": "windows-fixed-ntfs", "retention_enabled": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
