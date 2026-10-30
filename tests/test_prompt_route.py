"""P's route back to the prompt box (UI#36): its wiring, help row and name.

Static reads of the page's files. Focus, the caret, the guards against fields,
menus and dialogs, and the narrow scroll are `smoke-shortcut-dispatch`,
`smoke-shortcuts` and `smoke-narrow`'s, in a real browser on the fake.
"""
from __future__ import annotations

from pathlib import Path
import re

WEB = Path(__file__).resolve().parent.parent / "web"


def _route_block() -> str:
    app = (WEB / "app.js").read_text(encoding="utf-8")
    start = app.index("if (!e.repeat && !inMenu && !popup && !keyboardEditingTarget(e.target) && !keyboardCommandModifier(e)) {")
    return app[start:app.index("\n  }\n", start)]


def test_p_shares_the_slash_routes_guards():
    block = _route_block()
    assert "e.key === '/' && !modal" in block
    assert "!e.shiftKey && !modal && e.key.toLowerCase() === 'p'" in block
    assert re.search(r"'p'\) \{ e\.preventDefault\(\); focusPromptBox\(\); return; \}", block)


def test_route_focuses_without_writing_the_box():
    app = (WEB / "app.js").read_text(encoding="utf-8")
    start = app.index("function focusPromptBox() {")
    body = app[start:app.index("\n}\n", start)]
    assert "focus({ preventScroll: true })" in body
    assert "setSelectionRange(box.value.length, box.value.length)" in body
    # The route changes no text, so it adds no Undo step and no box write.
    for writer in ("writeBox", "rewriteBox", "replaceBoxText", ".value ="):
        assert writer not in body
    assert body.count("scrollIntoView(") == 1


def test_help_row_and_prompt_box_name_the_key():
    shortcuts = (WEB / "shortcuts.js").read_text(encoding="utf-8")
    group = shortcuts[shortcuts.index("id: 'prompt',"):]
    group = group[:group.index("\n  },")]
    assert "[['P'], 'Focus the prompt box from the workspace, with the caret at the end." in group
    html = (WEB / "index.html").read_text(encoding="utf-8")
    prompt = re.search(r'<textarea id="prompt"[^>]*>', html, re.S)
    assert prompt and 'aria-keyshortcuts="P"' in prompt.group(0)
