"""A bundle view owns verified input and memory only, including for encrypted archives."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from backend import bundle_archive, bundle_browse_archive, bundle_leases, bundles, storage
from tests.bundle_helpers import bundle_snapshot, save_source


def test_browse_paged_starred_filters_and_settings_are_archive_local(store, monkeypatch):
    images = [save_source(f"Straße %_{number}", number, negative_prompt="rain") for number in range(26)]
    chosen = images[3]
    store.set_favorite(chosen.id, True)
    archive = bundles.export_bundle(expected=26, keep_gallery=True)
    store.set_favorite(chosen.id, False)
    before = bundle_snapshot(store)
    # The archived settings validator must never query the working gallery's
    # thumbnail or calculate a staging directory even for a plain archive.
    def unexpected(*args, **kwargs):
        pytest.fail("browsing reached the live gallery or staging")
    monkeypatch.setattr(storage, "_thumb_path", unexpected)
    monkeypatch.setattr(storage, "bundle_staging_dir", unexpected)
    path = Path(archive["path"])
    with bundle_browse_archive.open_view(path.name, None) as view:
        rows = view.images("", "", False, 0)
        assert rows["total"] == 26 and len(rows["images"]) == 24
        assert rows["images"][0]["id"] == images[-1].id
        assert [row["id"] for row in view.images("", "", False, 24)["images"]] == [images[1].id, images[0].id]
        one = view.images("STRASSE %_", "N-ON", True, 0)
        assert one["total"] == 1
        assert one["images"][0]["id"] == chosen.id
        assert one["images"][0]["settings"]["negative_prompt"] == "rain"
        assert one["images"][0]["settings"]["seed"] == 3
        assert "url" not in one["images"][0]["settings"]
        assert view.images("", "missing run", False, 0)["total"] == 0
        with Image.open(io.BytesIO(view.thumbnail(chosen.id))) as image:
            assert image.format == "WEBP" and max(image.size) <= storage.THUMB_SIDE
            assert "exif" not in image.info and "xmp" not in image.info
        with pytest.raises(KeyError):
            view.thumbnail(5000)
        with pytest.raises(bundle_leases.LeaseBusy):
            with bundle_leases.downloads.exclusive(path):
                pass
    assert not bundle_leases.downloads.leased(path)
    assert bundle_snapshot(store) == before


def test_protected_browse_errors_release_claim_without_creating_plaintext(store, monkeypatch):
    row = save_source("unpublished treasure")
    path = Path(bundles.export_bundle(expected=1, password="secret")["path"])
    files = sorted(file for file in path.parent.rglob("*") if file.is_file())
    with pytest.raises(bundles.BundleError, match="password-protected"):
        with bundle_browse_archive.open_view(path.name, None):
            pass
    with pytest.raises(bundles.BundleError, match="incorrect"):
        with bundle_browse_archive.open_view(path.name, "wrong"):
            pass
    assert not bundle_leases.downloads.leased(path)
    with bundle_browse_archive.open_view(path.name, "secret") as view:
        assert view.images("treasure", "", False, 0)["images"][0]["id"] == row.id
        assert view.thumbnail(row.id).startswith(b"RIFF")
        assert sorted(file for file in path.parent.rglob("*") if file.is_file()) == files
    monkeypatch.setattr(bundle_archive, "_BROWSE_DATABASE_MAX", 1)
    with pytest.raises(bundles.BundleError, match="limit for browsing in memory"):
        with bundle_browse_archive.open_view(path.name, "secret"):
            pass
    assert not bundle_leases.downloads.leased(path)
    assert sorted(file for file in path.parent.rglob("*") if file.is_file()) == files


def test_verified_png_required_before_any_records_and_again_on_read(store, monkeypatch):
    row = save_source()
    path = Path(bundles.export_bundle(expected=1)["path"])
    with bundle_browse_archive.open_view(path.name, None) as view:
        original_read = zipfile.ZipFile.read
        def changed_read(archive, name, *args):
            data = original_read(archive, name, *args)
            return data + b"changed" if name == "images/" + row.filename else data
        monkeypatch.setattr(zipfile.ZipFile, "read", changed_read)
        with pytest.raises(bundles.BundleError, match="changed since it was opened"):
            view.thumbnail(row.id)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with pytest.raises(bundles.BundleError, match="invalid size or checksum"):
        with bundle_browse_archive.open_view(path.name, None):
            pytest.fail("unverified records were exposed")
    assert not bundle_leases.downloads.leased(path)

def test_closed_during_verify_and_oversized_source_release_archive(store, monkeypatch):
    save_source()
    path = Path(bundles.export_bundle(expected=1)["path"])
    calls = iter([False, True])
    with pytest.raises(bundles.BundleError, match="view was closed"):
        with bundle_browse_archive.open_view(path.name, None, cancelled=lambda: next(calls)):
            pytest.fail("cancelled view was exposed")
    assert not bundle_leases.downloads.leased(path)
    monkeypatch.setattr(bundle_browse_archive, "_THUMB_SOURCE_PIXELS", 255 * 255)
    monkeypatch.setattr(bundle_browse_archive.bundle_images, "_inspect_source",
                        lambda *args: pytest.fail("oversized source was decoded"))
    with pytest.raises(bundles.BundleError, match="too large to browse"):
        with bundle_browse_archive.open_view(path.name, None):
            pytest.fail("oversized view was exposed")
    assert not bundle_leases.downloads.leased(path)
