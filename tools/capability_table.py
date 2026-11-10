"""Print docs/usage.md's capability table, or rewrite it in place.

    python tools/capability_table.py            # the table, to stdout
    python tools/capability_table.py --write    # replace it in docs/usage.md

The table is `backend.model_info.capability_table()`: the catalogue's rows
and the flags the Models notes say, from the same function. It reads no
weights and reaches no Hub. `tests/test_capabilities.py` fails while the
document differs, so run this with --write after a catalogue or profile
change.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import model_info  # noqa: E402

USAGE = ROOT / "docs" / "usage.md"


def replaced(text: str) -> str:
    """`text` with the lines between the two markers replaced by the table."""
    start, end = model_info.CAPABILITY_TABLE_START, model_info.CAPABILITY_TABLE_END
    if text.count(start) != 1 or text.count(end) != 1 or text.index(end) < text.index(start):
        raise SystemExit(f"{USAGE} needs one '{start}' line before one '{end}' line")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{model_info.capability_table()}\n{end}{tail}"


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    if "--write" not in sys.argv[1:]:
        print(model_info.capability_table())
        return 0
    with USAGE.open(encoding="utf-8", newline="") as handle:
        before = handle.read()
    # Kept in the file's own line endings (a Windows checkout's are CRLF).
    newline = "\r\n" if "\r\n" in before else "\n"
    after = replaced(before.replace("\r\n", "\n")).replace("\n", newline)
    if after != before:
        with USAGE.open("w", encoding="utf-8", newline="") as handle:
            handle.write(after)
    print(f"{USAGE.relative_to(ROOT)}: {'updated' if after != before else 'already current'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
