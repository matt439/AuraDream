"""The Bundles dialog's Caps Lock notice (UI#38), as the page source declares it.

Each password grid holds one empty polite live line, positioned out of the
flow so its coming and going moves nothing. The behaviour (shown, hidden on
a false state and on blur, no control moved, nothing stored) is
`tools/smoke-bundles.json`'s.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
PASSWORDS = ("bundle-password", "bundle-password-again", "bundle-unlock")


def _grids(html: str) -> list[str]:
    return re.findall(r'<div class="bundle-password[^"]*">(.*?)</div>', html, re.S)


def test_every_password_grid_holds_one_empty_polite_note():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    grids = _grids(html)
    found = []
    for grid in grids:
        notes = re.findall(r'<p class="caps-lock-note" id="[^"]+" aria-live="polite"></p>', grid)
        assert len(notes) == 1, grid
        found += re.findall(r'<input id="([^"]+)" type="password"', grid)
    assert sorted(found) == sorted(PASSWORDS)


def test_the_note_is_out_of_the_flow_in_its_grid():
    css = (WEB / "styles" / "bundles.css").read_text(encoding="utf-8")
    assert re.search(r"\.bundle-password \{ position: relative; \}", css)
    rule = re.search(r"\n\.caps-lock-note \{(.*?)\}", css, re.S)
    assert rule and "position: absolute" in rule.group(1)
    assert "pointer-events: none" in rule.group(1)


def test_only_the_modifier_state_is_read_and_nothing_is_stored():
    js = (WEB / "bundles.js").read_text(encoding="utf-8")
    start = js.index("let bundleCapsLock")
    end = js.index("function ", js.index("function watchBundleCapsLock") + 1)
    watcher = js[start:end]
    assert "getModifierState('CapsLock')" in watcher
    assert ".value" not in watcher
    assert "Storage" not in watcher and "fetch" not in watcher
