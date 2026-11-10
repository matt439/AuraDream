"""The phone's Share (UI#42): Picture only beside With settings.

The stored PNG carries the prompt, the settings, the checkpoint and LoRA
names and an inpaint's mask in its text chunks. These read the page's
source for what must stay true of the picture-only path; the files it
makes are checked in a browser by `tools/smoke-phone-gallery.json`.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
VIEWER = (WEB / "phone-viewer.js").read_text(encoding="utf-8")


def _function(name: str) -> str:
    """The source of the top-level function `name`, to the next top-level close."""
    found = re.search(rf"^(?:async )?function {name}\(.*?^}}$", VIEWER, re.M | re.S)
    assert found, name
    return found.group(0)


def test_share_offers_both_files_in_words():
    sheet = _function("openViewerShare")
    assert "label: 'Picture only'" in sheet
    assert "label: 'With settings'" in sheet
    assert "sharePhoneImage(false)" in sheet and "sharePhoneImage(true)" in sheet
    index = (WEB / "index.html").read_text(encoding="utf-8")
    assert re.search(r'id="lbp-share" aria-haspopup="menu"', index)
    assert "$('lbp-share').addEventListener('click', openViewerShare)" in VIEWER


def test_no_share_path_falls_back_to_the_stored_png():
    # The lightbox's Download fetches the stored PNG whole: a Picture only
    # that fell back to it would send the settings after all.
    for name in ("openViewerShare", "pictureOnlyBlob", "shareFile", "sharePhoneImage"):
        assert "downloadLightboxImage" not in _function(name), name
    share = _function("sharePhoneImage")
    assert share.count("saveBlobAs(file, file.name)") == 2
    # Its failure is held as Download's is, so a walk or a later success clears it.
    assert "lbDownloadError = { id: rec.id, text }" in share


def test_picture_only_is_redrawn_and_named_without_the_settings():
    redraw = _function("pictureOnlyBlob")
    assert "canvas.toBlob(resolve, 'image/png')" in redraw
    assert "throw new Error" in redraw
    file = _function("shareFile")
    assert "`auradream-${rec.id}.png`" in file
    assert "await pictureOnlyBlob(stored)" in file
