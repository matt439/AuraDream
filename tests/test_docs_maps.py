"""The module maps against the files they map.

`docs/rules.md`'s Layout is the full map, and `CLAUDE.md` only imports
`AGENTS.md` (its first line is `@AGENTS.md`); the maps drifted by hand before
(`backend/metadata.py` was in neither), so they are held here rather than by
review. So is `AGENTS.md`'s smoke/port table: a smoke run against the port its row names
finds the fake that row says to start.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _layout() -> str:
    text = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    return text[text.index("## Layout"):text.index("## Rules")]


def _smoke_table() -> dict[str, list[int]]:
    """Each smoke the port table names (`-rewrite` is `smoke-rewrite`) and
    the ports of the rows naming it."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table: dict[str, list[int]] = {}
    for port, smokes in re.findall(r"^\| (\d{4}) \|[^|\n]*\|([^|\n]*)\|$", text, re.M):
        for name in re.findall(r"`([^`]+)`", smokes):
            stem = name.removesuffix(".json")
            stem = "smoke" + stem if stem.startswith("-") else stem
            table.setdefault(stem, []).append(int(port))
    return table


def _first_navigate_port(path: Path) -> int | None:
    for step in json.loads(path.read_text(encoding="utf-8")):
        if isinstance(step, dict) and "navigate" in step:
            found = re.match(r"https?://127\.0\.0\.1:(\d+)/", step["navigate"])
            return int(found.group(1)) if found else None
    return None


def test_claude_imports_agents():
    first = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()[:1]
    assert first == ["@AGENTS.md"]


def test_every_backend_module_and_page_script_is_in_the_layout():
    layout = _layout()
    files = [
        *(p for p in (ROOT / "backend").rglob("*.py") if p.name != "__init__.py"),
        *(ROOT / "web").glob("*.js"),
    ]
    missing = [p.relative_to(ROOT).as_posix() for p in files if p.name not in layout]
    assert missing == []


def test_every_stylesheet_is_in_the_layout_in_link_order():
    layout = _layout()
    links = re.findall(
        r'<link rel="stylesheet" href="/static/styles/([^"]+)">',
        (ROOT / "web" / "index.html").read_text(encoding="utf-8"),
    )
    on_disk = sorted(p.name for p in (ROOT / "web" / "styles").glob("*.css"))
    assert sorted(links) == on_disk
    # The Layout's list of what each sheet holds runs in the link order.
    listing = layout[layout.index("explicit link order"):]
    assert [name for name in on_disk if f"`{name}`" not in listing] == []
    assert sorted(links, key=lambda name: listing.index(f"`{name}`")) == links


def test_the_layout_counts_the_page_scripts():
    scripts = re.findall(
        r'<script src="/static/[^"]+\.js"></script>',
        (ROOT / "web" / "index.html").read_text(encoding="utf-8"),
    )
    assert f"`index.html` loads {len(scripts)} classic" in _layout()


def test_every_smoke_is_in_the_port_table_on_its_port():
    table = _smoke_table()
    smokes = {p.stem: p for p in (ROOT / "tools").glob("smoke*.json")}
    assert sorted(set(table) - set(smokes)) == []
    assert sorted(set(smokes) - set(table)) == []
    wrong = {
        stem: (table[stem], _first_navigate_port(path))
        for stem, path in smokes.items()
        if table[stem] != [_first_navigate_port(path)]
    }
    assert wrong == {}


def test_every_smoke_is_described_under_tests():
    rules = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    tests = rules[rules.index("## Tests"):]
    missing = [p.name for p in (ROOT / "tools").glob("smoke*.json") if p.name not in tests]
    assert missing == []
