"""The module maps against the files they map.

`docs/rules.md`'s Layout is the full map, and `CLAUDE.md`/`AGENTS.md` are
kept byte-identical; both drifted by hand before (`backend/metadata.py` was
in neither map), so they are held here rather than by review.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _layout() -> str:
    text = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    return text[text.index("## Layout"):text.index("## Rules")]


def test_claude_and_agents_are_the_same_file():
    assert (ROOT / "CLAUDE.md").read_bytes() == (ROOT / "AGENTS.md").read_bytes()


def test_every_backend_module_and_page_script_is_in_the_layout():
    layout = _layout()
    files = [
        *(p for p in (ROOT / "backend").rglob("*.py") if p.name != "__init__.py"),
        *(ROOT / "web").glob("*.js"),
    ]
    missing = [p.relative_to(ROOT).as_posix() for p in files if p.name not in layout]
    assert missing == []
