"""Saved archives can move independently of the gallery and its staging."""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend import bundles
from tests.bundle_helpers import save_source, export_all, use_destination


def test_default_directory_follows_the_current_database(store, monkeypatch, tmp_path):
    assert store.settings.bundles_dir is None
    assert store.bundle_directory() == store.settings.db_path.parent / "bundles"
    dest = use_destination(store, monkeypatch, tmp_path / "another-gallery")
    assert store.bundle_directory() == dest.db_path.parent / "bundles"
    assert not store.bundle_directory().exists()  # Looking never creates it.


def test_configured_directory_exports_lists_and_imports_independently(store, monkeypatch, tmp_path):
    directory = tmp_path / "another-drive" / "saved-bundles"
    database, outputs = store.settings.db_path, store.settings.outputs_dir
    monkeypatch.setattr(store, "settings", dataclasses.replace(store.settings, bundles_dir=directory))
    assert bundles.list_bundles() == []
    assert not directory.exists()
    row = save_source()
    archive = export_all()
    path = Path(archive["path"])
    assert path.parent == directory
    assert store.settings.db_path == database and store.settings.outputs_dir == outputs
    assert not (database.parent / "bundles").exists()
    assert bundles.bundle_path(archive["name"]) == path
    assert [item["name"] for item in bundles.list_bundles()] == [archive["name"]]
    assert bundles.import_bundle(path)["imported"] == 1
    assert store.list_images()[0][0].filename == row.filename
    assert store.bundle_staging_dir().parent == outputs.parent
    assert not list(database.parent.glob(".bundle-*"))


@pytest.mark.parametrize("nested", [False, True])
def test_configured_directory_cannot_be_the_images_or_inside_them(store, monkeypatch, nested):
    directory = store.settings.outputs_dir / "archives" if nested else store.settings.outputs_dir
    monkeypatch.setattr(store, "settings", dataclasses.replace(store.settings, bundles_dir=directory))
    with pytest.raises(bundles.BundleError, match="outside the images directory"):
        bundles.list_bundles()
    with pytest.raises(bundles.BundleError, match="outside the images directory"):
        store._bundle_leftovers()


@pytest.mark.parametrize("configured", [False, True])
def test_direct_directory_links_are_refused_before_resolution(store, monkeypatch, tmp_path, configured):
    target = tmp_path / "archive-target"
    target.mkdir()
    protected = target / "keep.zip.part"
    protected.write_bytes(b"another directory's file")
    directory = tmp_path / ("configured-archives" if configured else "bundles")
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(directory))
    else:
        directory.symlink_to(target, target_is_directory=True)
    if configured:
        monkeypatch.setattr(store, "settings", dataclasses.replace(store.settings, bundles_dir=directory))
    with pytest.raises(bundles.BundleError, match="must not be a symbolic link"):
        bundles.list_bundles()
    with pytest.raises(bundles.BundleError, match="must not be a symbolic link"):
        store._bundle_leftovers()
    assert protected.read_bytes() == b"another directory's file"


def test_containment_resolves_parent_links(store, monkeypatch, tmp_path):
    parent = tmp_path / "images-alias"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(store.settings.outputs_dir), str(parent))
    else:
        parent.symlink_to(store.settings.outputs_dir, target_is_directory=True)
    monkeypatch.setattr(store, "settings", dataclasses.replace(store.settings, bundles_dir=parent / "archives"))
    with pytest.raises(bundles.BundleError, match="outside the images directory"):
        bundles.list_bundles()


def test_fake_server_keeps_inherited_bundle_storage_outside_its_reach(tmp_path):
    production = tmp_path / "production-bundles"
    production.mkdir()
    marker = production / "keep.zip.part"
    marker.write_bytes(b"not fake-server data")
    old = time.time() - 7200
    os.utime(marker, (old, old))
    scratch = tmp_path / "fake-gallery"
    # A fresh interpreter checks the effective frozen settings, including
    # the override happening before config imports. No server is started.
    script = """
import runpy
import sys
import uvicorn

def inspect(*args, **kwargs):
    from backend import storage
    print('BUNDLES=' + str(storage.bundle_directory()))
    storage.init_db()
    storage.settle_partials()

uvicorn.run = inspect
sys.argv = ['tools/fake_server.py', '--data', sys.argv[1]]
runpy.run_path('tools/fake_server.py', run_name='__main__')
"""
    clean = {key: value for key, value in os.environ.items() if not key.startswith("AURADREAM_")}
    result = subprocess.run(
        [sys.executable, "-c", script, str(scratch)],
        env={**clean, "AURADREAM_BUNDLES_DIR": str(production)},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "BUNDLES=" + str(scratch / "bundles") in result.stdout
    assert marker.read_bytes() == b"not fake-server data"
