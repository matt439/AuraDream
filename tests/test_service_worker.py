"""The offline page's service worker (web/sw.js, web/offline.html): served
at the site's root so its scope covers "/", never cached by HTTP, never
activity; registered only in a secure context; answering only the page's
navigation, and the offline page standing alone. The behaviour itself, a
silent server's "/" and the page reopening, is smoke-offline's."""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
SW = (WEB / "sw.js").read_text(encoding="utf-8")
OFFLINE = (WEB / "offline.html").read_text(encoding="utf-8")


def test_served_at_the_root_uncached(client):
    response = client.get("/sw.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.headers["cache-control"] == "no-cache"
    assert response.text == SW


def test_registered_only_in_a_secure_context():
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "if ('serviceWorker' in navigator && window.isSecureContext) {" in app
    assert app.count("navigator.serviceWorker.register('/sw.js')") == 1
    index = (WEB / "index.html").read_text(encoding="utf-8")
    assert "sw.js" not in index  # no classic script: app.js registers it


def test_the_worker_answers_only_the_page_and_caches_only_the_offline_page():
    assert "request.mode !== 'navigate'" in SW
    assert "url.pathname !== '/'" in SW
    # One cache entry: the offline page. Nothing the page runs is cached.
    assert re.findall(r"cache\.(add|addAll|put)\(", SW) == ["add"]
    assert "const OFFLINE_URL = '/static/offline.html';" in SW
    assert re.search(r"const CACHE = 'auradream-offline-v\d+';", SW)


def test_the_offline_page_stands_alone():
    # Nothing it shows may need the server it says is gone.
    assert not re.search(r'<(link|img|script)[^>]*\s(href|src)=', OFFLINE)
    assert "url(" not in OFFLINE.replace("url(#", "")
    # Its probe is a page file: never activity, never behind a load.
    assert "fetch('/static/manifest.webmanifest'" in OFFLINE
    assert (WEB / "manifest.webmanifest").is_file()
    assert "location.replace('/')" in OFFLINE
