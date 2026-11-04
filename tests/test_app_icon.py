"""The home-screen icon, favicon and manifest: every file index.html and the
manifest name exists at its stated size, and the manifest keeps the page in
the browser (no full-screen home-screen app: Download and Open PNG need
Safari round them)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

WEB = Path(__file__).resolve().parent.parent / "web"
HEAD = (WEB / "index.html").read_text(encoding="utf-8").split("</head>", 1)[0]


def _static(href: str) -> Path:
    assert href.startswith("/static/"), href
    return WEB / href.removeprefix("/static/")


def _links(rel: str) -> list[str]:
    return re.findall(rf'<link rel="{rel}" href="([^"]+)"', HEAD)


def test_head_links_resolve():
    icons = _links("icon")
    assert "/static/icons/icon.svg" in icons
    assert _links("apple-touch-icon") == ["/static/icons/apple-touch-icon.png"]
    assert _links("manifest") == ["/static/manifest.webmanifest"]
    for href in icons + _links("apple-touch-icon") + _links("manifest"):
        assert _static(href).is_file(), href
    assert '<meta name="apple-mobile-web-app-capable"' not in HEAD


def test_png_sizes_match_their_names():
    assert Image.open(WEB / "icons" / "apple-touch-icon.png").size == (180, 180)
    assert Image.open(WEB / "icons" / "favicon-32.png").size == (32, 32)
    # iOS shows a transparent apple-touch-icon on black: the icon is opaque.
    assert Image.open(WEB / "icons" / "apple-touch-icon.png").mode == "RGB"


def test_manifest_stays_in_the_browser():
    manifest = json.loads((WEB / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["display"] == "browser"
    assert manifest["start_url"] == "/" and manifest["scope"] == "/"
    assert manifest["short_name"] == "AuraDream"
    purposes = set()
    for icon in manifest["icons"]:
        path = _static(icon["src"])
        assert path.is_file(), icon["src"]
        purposes.add(icon["purpose"])
        if icon["sizes"] != "any":
            width, height = map(int, icon["sizes"].split("x"))
            assert Image.open(path).size == (width, height), icon["src"]
    assert {"any", "maskable"} <= purposes
