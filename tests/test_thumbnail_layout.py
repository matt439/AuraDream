"""Thumbnail layout changes preserve originals, recovery, and gallery cleanup."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from PIL import Image

from backend import clear, reindex
from backend.schemas import GenerateRequest


def _make(store, prompt="a fox", *, favorite=False):
    record = store.save_image(
        Image.new("RGB", (64, 48), (40, 70, 110)),
        GenerateRequest(prompt=prompt, width=256, height=256),
        seed=7, duration_ms=123, model_id="test/model",
    )
    if favorite:
        record = store.set_favorite(record.id, True)
    png = store.settings.outputs_dir / record.filename
    return record, png, png.with_suffix(".webp"), png.parent / "thumbnails" / png.with_suffix(".webp").name


def test_startup_moves_legacy_thumbnail_without_changing_image_or_record(store):
    record, png, legacy, thumb = _make(store, favorite=True)
    original_png, original_thumb = png.read_bytes(), thumb.read_bytes()
    thumb.rename(legacy)
    assert store.get_image(record.id).thumb_url == f"/outputs/{legacy.name}"

    store.init_db()
    store.init_db()  # Another startup is harmless.

    assert png.read_bytes() == original_png
    assert thumb.read_bytes() == original_thumb
    assert not legacy.exists()
    assert store.get_image(record.id) == record


def test_startup_leaves_unrelated_thumbnails_and_missing_originals_alone(store):
    _, png, legacy, thumb = _make(store)
    thumb.rename(legacy)
    png.unlink()
    outputs = store.settings.outputs_dir
    unrelated = outputs / "unrelated.webp"
    untracked_png = outputs / "untracked.png"
    untracked_thumb = outputs / "untracked.webp"
    unrelated.write_bytes(b"an unrelated image")
    untracked_png.write_bytes(b"not indexed")
    untracked_thumb.write_bytes(b"not an indexed thumbnail")
    before = {path: path.read_bytes() for path in (legacy, unrelated, untracked_png, untracked_thumb)}

    store.init_db()

    assert {path: path.read_bytes() for path in before} == before
    assert not thumb.exists()


def test_startup_preserves_conflicting_thumbnail_and_prefers_new_location(store):
    record, _, legacy, thumb = _make(store)
    original_thumb = thumb.read_bytes()
    legacy.write_bytes(b"different legacy thumbnail")

    store.init_db()

    assert legacy.read_bytes() == b"different legacy thumbnail"
    assert thumb.read_bytes() == original_thumb
    assert store.get_image(record.id).thumb_url == f"/outputs/thumbnails/{thumb.name}"


def test_locked_legacy_thumbnail_remains_available(store, monkeypatch):
    record, png, legacy, thumb = _make(store)
    thumb.rename(legacy)
    rename = Path.rename

    def locked(path, target):
        if path == legacy:
            raise PermissionError("thumbnail is open")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", locked)
    store.init_db()
    with Image.open(png) as image:
        store.thumbnail_for(image, png.name)

    assert legacy.exists() and not thumb.exists()
    assert store.get_image(record.id).thumb_url == f"/outputs/{legacy.name}"


def test_lost_database_recovery_moves_legacy_thumbnails_but_dry_runs_do_not(store, monkeypatch):
    record, png, legacy, thumb = _make(store)
    original_png, original_thumb = png.read_bytes(), thumb.read_bytes()
    thumb.rename(legacy)
    thumb.parent.rmdir()
    store.settings.db_path.unlink()
    monkeypatch.setattr(reindex, "settings", store.settings)
    monkeypatch.setattr(clear, "settings", store.settings)

    assert store.summarise()["bytes_on_disk"] == len(original_png) + len(original_thumb)
    assert clear.main(["--dry-run"]) == 0
    assert reindex.reindex(store.settings.outputs_dir, dry_run=True)["indexed"] == 1
    assert not store.settings.db_path.exists()
    assert not thumb.parent.exists()
    assert png.read_bytes() == original_png and legacy.read_bytes() == original_thumb

    assert reindex.reindex(store.settings.outputs_dir)["indexed"] == 1
    assert not legacy.exists()
    assert thumb.read_bytes() == original_thumb and png.read_bytes() == original_png
    restored = store.list_images()[0][0]
    assert restored.filename == record.filename
    assert restored.thumb_url == f"/outputs/thumbnails/{thumb.name}"


@pytest.mark.parametrize("operation", ["single", "matching", "history"])
def test_deletion_removes_both_thumbnail_locations_and_keeps_starred_files(store, operation):
    kept, kept_png, kept_legacy, kept_thumb = _make(store, "keep", favorite=True)
    gone, gone_png, gone_legacy, gone_thumb = _make(store, "remove")
    for legacy, thumb in ((kept_legacy, kept_thumb), (gone_legacy, gone_thumb)):
        legacy.write_bytes(thumb.read_bytes())

    if operation == "single":
        assert store.delete_image(gone.id)
    elif operation == "matching":
        assert store.delete_matching(None, keep_favorites=True)["records"] == 1
    else:
        assert store.clear_history(keep_favorites=True)["records"] == 1

    assert all(path.exists() for path in (kept_png, kept_legacy, kept_thumb))
    assert not any(path.exists() for path in (gone_png, gone_legacy, gone_thumb))
    assert [item.id for item in store.list_images()[0]] == [kept.id]


def test_summary_and_cleanup_include_both_thumbnail_folders_and_only_stale_partials(store):
    _, png, legacy, thumb = _make(store, favorite=True)
    legacy.write_bytes(thumb.read_bytes())
    roots = (store.settings.outputs_dir, thumb.parent)
    orphan_thumbs = [root / "orphan.webp" for root in roots]
    stale = [root / "stale.webp.part" for root in roots]
    fresh = [root / "fresh.webp.part" for root in roots]
    for path in orphan_thumbs + stale + fresh:
        path.write_bytes(b"cached bytes")
    old = time.time() - 2 * store.PARTIAL_SETTLE_SECONDS
    for path in stale:
        os.utime(path, (old, old))
    other = store.settings.outputs_dir / "other"
    other.mkdir()
    unrelated = other / "stale.webp.part"
    unrelated.write_bytes(b"not owned by thumbnail cleanup")
    os.utime(unrelated, (old, old))

    summary = store.summarise()
    assert summary["bytes_on_disk"] == sum(path.stat().st_size for path in (png, legacy, thumb, *orphan_thumbs))
    assert summary["partial_files"] == 2
    assert all(path.exists() for path in stale)
    assert store.clear_history(keep_favorites=True)["partials"] == 2
    assert not any(path.exists() for path in stale + orphan_thumbs)
    assert all(path.exists() for path in fresh + [unrelated, png, legacy, thumb])


def test_nested_thumbnail_url_is_served_as_webp(store):
    record, png, _, thumb = _make(store)
    # The suite's main app mounted its import-time directory, so give this
    # HTTP check the scratch gallery explicitly, using the same StaticFiles.
    app = FastAPI()
    app.mount("/outputs", StaticFiles(directory=store.settings.outputs_dir), name="outputs")
    with TestClient(app) as client:
        response = client.get(record.thumb_url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
        assert response.content == thumb.read_bytes()
        assert client.get(record.url).content == png.read_bytes()
