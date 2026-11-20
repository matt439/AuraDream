"""Historical archives through the current import, interrupted publication and retry."""
from __future__ import annotations

import json
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundles
from backend.schemas import GenerateRequest
from tests.bundle_helpers import bundle_snapshot
from tests.historical_gallery_helpers import ARCHIVES, assert_gallery
from tools import storage_interrupt


def populate_destination(store):
    # Occupy every original ID, including the missing edit source's ID at
    # the expected geometry. Its pixels and filename belong to this gallery.
    for image_id in (1, 2, 3):
        with Image.new("RGB", (512, 512), (7, 12, image_id)) as image:
            row = store.save_image(
                image, GenerateRequest(prompt=f"local {image_id}", width=512, height=512),
                seed=900 + image_id, duration_ms=100, model_id="test/local", quantization="fp8",
            )
        assert row.id == image_id
    store.set_favorite(2, True)
    return bundle_snapshot(store)


def retry_after_interruption(store, tmp_path, archive, before, members):
    state = {"gallery": str(tmp_path), "archive": str(archive), "records": 2}
    killed = storage_interrupt.interrupt_worker(tmp_path, state, "import-before-publish")
    rows, files = bundle_snapshot(store)
    # Actual files are installed, but no new row has been published.
    assert rows == before[0]
    assert set(files) == set(before[1]) | members
    assert all(files[name] == data for name, data in before[1].items())
    report = storage_interrupt.restart_worker(tmp_path, state, store, killed)
    assert report["swept"] > 0
    assert report["repeated"]["present"] == 2
    return report["imported"]


@pytest.mark.parametrize(("archive_name", "manifest_identity", "seeds", "binned", "unresolved"), [
    ("legacy-v1", ("local-diffusion-gallery", 1, 3), (101, 102, 103), 1, 0),
    ("edit-v2", ("auradream-gallery", 2, 2), (201, 202), 0, 0),
    ("edit-only-v2", ("auradream-gallery", 2, 1), (202,), 0, 1),
])
def test_historical_import_remaps_collisions_and_repeats_without_changes(
    store, tmp_path, archive_name, manifest_identity, seeds, binned, unresolved,
):
    archive = ARCHIVES / f"{archive_name}.zip"
    before = populate_destination(store)
    with ZipFile(archive) as opened:
        manifest = json.loads(opened.read("manifest.json"))
        assert (manifest["format"], manifest["version"], manifest["records"]) == manifest_identity
        members = {name.removeprefix("images/") for name in opened.namelist() if name.startswith("images/")}
        thumbnails = {name: opened.read(f"images/{name}") for name in members if name.endswith(".webp")}
    if archive_name == "edit-v2":
        imported = retry_after_interruption(store, tmp_path, archive, before, members)
    else:
        imported = bundles.import_bundle(archive)
    count = len(seeds)
    assert (imported["imported"], imported["present"], imported["restored"]) == (count, 0, 0)
    assert (imported["first_id"], imported["last_id"]) == (4, 3 + count)
    assert (imported["binned"], imported["unresolved_references"]) == (binned, unresolved)
    if unresolved:
        records = assert_gallery("edit-v2", None, orphan=True)
    else:
        records = assert_gallery(archive_name, 4)
    assert {seed: records[seed].id for seed in seeds} == {
        seed: image_id for image_id, seed in enumerate(seeds, start=4)
    }
    rows, files = bundle_snapshot(store)
    assert len(rows) == 3 + count and rows[:3] == before[0]
    assert set(files) == set(before[1]) | members
    assert all(files[name] == data for name, data in (before[1] | thumbnails).items())
    # A repeat must respect a local preference change as well as preserve
    # every rewritten chunk and its remapped source.
    store.set_favorite(4, not records[seeds[0]].favorite)
    snapshot = bundle_snapshot(store)
    repeated = bundles.import_bundle(archive)
    assert (repeated["imported"], repeated["present"], repeated["restored"]) == (0, count, 0)
    assert (repeated["first_id"], repeated["last_id"]) == (None, None)
    assert bundle_snapshot(store) == snapshot
