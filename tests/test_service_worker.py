"""The offline page's service worker (web/sw.js, web/offline.html): served
at the site's root so its scope covers "/", never cached by HTTP, never
activity; registered only in a secure context; answering only the page's
navigation, and the offline page standing alone. The behaviour itself, a
silent server's "/" and the page reopening, is smoke-offline's."""
from __future__ import annotations

import hashlib
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
    # The file as it is on disk: a Windows checkout has CRLF line ends.
    assert response.content == (WEB / "sw.js").read_bytes()


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


# The cache name each offline.html was released under, by the page's hash
# (line ends normalised). The worker caches the page at install only, so a
# changed page under the old name never reaches a browser that has it.
OFFLINE_RELEASES = {
    "auradream-offline-v2": "10cc23d6caf791bfdad8940182be34bfc4547d2879f9bc00b36c9eddf96171db",
}


def test_a_changed_offline_page_has_a_new_cache_name():
    cache = re.search(r"const CACHE = '(auradream-offline-v\d+)';", SW)
    assert cache is not None
    digest = hashlib.sha256((WEB / "offline.html").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert OFFLINE_RELEASES.get(cache[1]) == digest, (
        f"offline.html changed: give sw.js's CACHE a new version and record {digest} under it here"
    )


def test_the_offline_page_announces_only_a_change_of_outcome():
    # UI#54: the live part is written only when its words change; each
    # ask's clock goes in a part no screen reader announces.
    assert OFFLINE.count("aria-live") == 1 and '<div id="state" aria-live="polite">' in OFFLINE
    assert 'id="clock"' in OFFLINE and "state.textContent = " not in OFFLINE.replace(
        "if (state.textContent !== text) state.textContent = text;", "")
    assert "clock.textContent = `Last asked at ${at}`;" in OFFLINE


def test_the_offline_page_stands_alone():
    # Nothing it shows may need the server it says is gone.
    assert not re.search(r'<(link|img|script)[^>]*\s(href|src)=', OFFLINE)
    assert "url(" not in OFFLINE.replace("url(#", "")
    # Its probe is a page file: never activity, never behind a load.
    assert "fetch('/static/manifest.webmanifest'" in OFFLINE
    assert (WEB / "manifest.webmanifest").is_file()
    assert "location.replace('/')" in OFFLINE


def test_notify_when_done_goes_through_the_worker_and_its_tap_opens_the_page():
    # UI#55: Chrome on Android throws on the Notification constructor, so a
    # controlled page asks the registration; the tap is the worker's.
    progress = (WEB / "job-progress.js").read_text(encoding="utf-8")
    assert "registration.showNotification(title, shown)" in progress
    assert ".catch(construct)" in progress
    assert "self.addEventListener('notificationclick'" in SW
    assert "self.clients.openWindow('/')" in SW
    assert "event.notification.close()" in SW
