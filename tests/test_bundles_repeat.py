"""Repeat bundle imports: image identity, restored files and local state kept."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from backend import bundle_archive, bundle_images, bundles
from tests.bundle_helpers import save_source, export_all, use_destination, bundle_snapshot, png_metadata


def test_repeat_import_recognises_a_re_encoded_existing_image_by_its_pixels(store, monkeypatch, tmp_path):
    row = save_source("source", 1)
    bundle = Path(export_all()["path"])
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    path = dest.outputs_dir / row.filename
    with Image.open(path) as image:
        image.load()
        image.save(path, compress_level=0)  # different IDAT bytes, same pixels
    assert bundles.import_bundle(bundle)["present"] == 1
    Image.new("RGB", (256, 256), "white").save(path)
    with pytest.raises(bundles.BundleError, match="already uses the filename"):
        bundles.import_bundle(bundle)


def test_re_import_restores_an_image_whose_row_outlived_its_file(store, monkeypatch, tmp_path):
    source = save_source("source", 1)
    conditioned = save_source("conditioned", 2, reference=[source.id])
    bundle = Path(export_all()["path"])
    dest = use_destination(store, monkeypatch, tmp_path)
    save_source("existing", 99)
    assert bundles.import_bundle(bundle)["imported"] == 2
    rows = {r.filename: r for r in store.list_images()[0]}
    path = dest.outputs_dir / conditioned.filename
    original = path.read_bytes()
    # What a clear interrupted between removing files and rows leaves.
    path.unlink()
    snapshot_rows = bundle_snapshot(store)[0]
    result = bundles.import_bundle(bundle)
    assert (result["imported"], result["present"], result["restored"]) == (0, 1, 1)
    assert result["first_id"] is result["last_id"] is None
    assert bundle_snapshot(store)[0] == snapshot_rows  # the row is unchanged
    assert path.read_bytes() == original
    assert png_metadata(path)["reference"] == [rows[source.filename].id]

    # Only the very picture the row recorded, with the same identity.
    path.unlink()
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET content_hash = '0123456789abcdef' WHERE filename = ?", (conditioned.filename,))
    with pytest.raises(bundles.BundleError, match="lost its image"):
        bundles.import_bundle(bundle)
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET content_hash = ?, seed = 7 WHERE filename = ?",
                     (rows[conditioned.filename].content_hash, conditioned.filename))
    with pytest.raises(bundles.BundleError, match="different image metadata"):
        bundles.import_bundle(bundle)
    assert not path.exists()

    # The short hash is of the raw bytes alone: the size must match the row.
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET seed = ?, width = 2 * width WHERE filename = ?",
                     (conditioned.seed, conditioned.filename))
    with pytest.raises(bundles.BundleError, match="lost its image"):
        bundles.import_bundle(bundle)
    with closing(sqlite3.connect(dest.db_path)) as conn, conn:
        conn.execute("UPDATE images SET width = ?, height = ? WHERE filename = ?",
                     (conditioned.width, conditioned.height, conditioned.filename))
    assert bundles.import_bundle(bundle)["restored"] == 1


def test_re_import_restores_only_a_plain_rgb_picture(store, monkeypatch, tmp_path):
    # A palette copy's raw bytes are indices, so its short hash says nothing
    # of the colours; a render is always RGB, so nothing else is restored.
    row = save_source("source", 1)
    image = Image.new("P", (row.width, row.height), 0)
    image.putpalette([255, 0, 0, *[0] * 765])
    image.save(store.settings.outputs_dir / row.filename)
    bundle = Path(export_all()["path"])
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    (dest.outputs_dir / row.filename).unlink()
    with pytest.raises(bundles.BundleError, match="lost its image"):
        bundles.import_bundle(bundle)
    assert not (dest.outputs_dir / row.filename).exists()


@pytest.mark.parametrize("existing", ["palette", "gray"])
def test_repeat_import_tells_apart_pictures_whose_raw_bytes_agree(store, monkeypatch, tmp_path, existing):
    row = save_source("source", 1)

    def indexed(colour):
        image = Image.new("P", (256, 256), 0)
        image.putpalette([*colour, *[0] * 765])
        return image

    # A palette image's bytes are its indices: these three share them all.
    indexed((255, 0, 0)).save(store.settings.outputs_dir / row.filename)
    bundle = Path(export_all()["path"])
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    path = dest.outputs_dir / row.filename
    other = indexed((0, 0, 255)) if existing == "palette" else Image.new("L", (256, 256), 0)
    other.save(path)
    with pytest.raises(bundles.BundleError, match="already uses the filename"):
        bundles.import_bundle(bundle)


def test_repeat_import_keeps_a_known_image_s_own_thumbnail(store, monkeypatch, tmp_path):
    kept = save_source("kept", 1)
    removed = save_source("removed", 2)
    bundle = Path(export_all()["path"])
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 2
    thumbs = dest.outputs_dir / "thumbnails"
    # Another Pillow build, or a regenerated cache: different bytes.
    regenerated = thumbs / Path(kept.filename).with_suffix(".webp").name
    Image.new("RGB", (32, 32), "white").save(regenerated, format="WEBP")
    ours = regenerated.read_bytes()
    assert bundles.import_bundle(bundle)["present"] == 2
    assert regenerated.read_bytes() == ours

    # A new image's thumbnail is still held to its bundle bytes.
    store.delete_image(next(r.id for r in store.list_images()[0] if r.filename == removed.filename))
    (thumbs / Path(removed.filename).with_suffix(".webp").name).write_bytes(b"an unrelated file")
    with pytest.raises(bundles.BundleError, match="another file already uses the filename thumbnails/"):
        bundles.import_bundle(bundle)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "1", "I;16", "LA"])
def test_raw_digest_is_tobytes(mode):
    image = Image.effect_noise((37, 11), 64).convert(mode)
    assert bundle_images._raw_digest(image) == hashlib.sha256(image.tobytes()).hexdigest()
    assert bundle_images._raw_digest(Image.new(mode, (0, 5))) == hashlib.sha256(b"").hexdigest()


def test_repeat_import_neither_decodes_nor_unpacks_what_is_already_there(store, monkeypatch, tmp_path):
    for seed in (1, 2, 3):
        save_source(f"image {seed}", seed)
    bundle = Path(export_all()["path"])
    use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 3
    snapshot = bundle_snapshot(store)
    decoded, staged = [], []
    real_pixels, real_stage = bundle_images._pixels, bundle_archive._write_staged
    monkeypatch.setattr(bundle_images, "_pixels", lambda *a, **k: decoded.append(a) or real_pixels(*a, **k))
    monkeypatch.setattr(bundle_archive, "_write_staged", lambda target, *a: staged.append(target.name) or real_stage(target, *a))
    assert bundles.import_bundle(bundle)["present"] == 3
    assert decoded == [] and staged == []
    assert bundle_snapshot(store) == snapshot


def test_a_present_image_s_bundle_row_is_still_validated_without_a_decode(store, monkeypatch, tmp_path):
    row = save_source("source", 1)
    bundle = Path(export_all()["path"])
    use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle)["imported"] == 1
    with ZipFile(bundle) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    database = tmp_path / "bundle.sqlite3"
    database.write_bytes(members["database.sqlite3"])
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("UPDATE images SET width = 999 WHERE filename = ?", (row.filename,))
        conn.commit()
    members["database.sqlite3"] = database.read_bytes()
    manifest = json.loads(members["manifest.json"])
    manifest["files"]["database.sqlite3"] = {"size": len(members["database.sqlite3"]),
                                             "sha256": hashlib.sha256(members["database.sqlite3"]).hexdigest()}
    members["manifest.json"] = json.dumps(manifest).encode()
    altered = tmp_path / "altered.zip"
    with ZipFile(altered, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    snapshot = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError, match="dimensions disagree"):
        bundles.import_bundle(altered)
    assert bundle_snapshot(store) == snapshot


def test_the_recycle_bin_comes_across_and_a_repeat_import_keeps_the_local_state(store, monkeypatch, tmp_path):
    live = save_source("live", 1)
    binned = save_source("binned", 2)
    store.bin_images([binned.id])
    preview = bundles.preview()
    assert (preview["records"], preview["binned"]) == (2, 1)
    archive = export_all()
    # The bin is exported and cleared with the rest.
    assert (archive["records"], archive["binned"]) == (2, 1)
    assert set(archive["deleted_ids"]) == {live.id, binned.id}
    assert store.list_images(binned=True)[1] == 0

    use_destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["binned"]) == (2, 1)
    assert (imported["first_id"], imported["last_id"]) == (1, 2)
    assert [r.filename for r in store.list_images()[0]] == [live.filename]
    (back,) = store.list_images(binned=True)[0]
    assert back.filename == binned.filename and back.binned_at is not None

    # This gallery's choice wins on a repeat import, as its stars do.
    store.restore_images([back.id])
    store.bin_images([store.list_images(search="live")[0][0].id])
    again = bundles.import_bundle(Path(archive["path"]))
    assert (again["imported"], again["present"], again["binned"]) == (0, 2, 0)
    assert again["first_id"] is again["last_id"] is None
    assert [r.filename for r in store.list_images()[0]] == [binned.filename]
    assert [r.filename for r in store.list_images(binned=True)[0]] == [live.filename]


@pytest.mark.parametrize("binned", [False, True])
def test_import_range_covers_only_new_rows_among_known_and_restored_images(store, monkeypatch, tmp_path, binned):
    present = save_source("present", 1)
    restoring = save_source("restoring", 2)
    new = save_source("new", 3)
    if binned:
        store.bin_images([new.id])
    bundle = Path(export_all()["path"])
    dest = use_destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(bundle)
    rows = {r.filename: r for in_bin in (False, True) for r in store.list_images(binned=in_bin)[0]}
    old_id = rows[new.filename].id
    store.delete_image(old_id)
    (dest.outputs_dir / restoring.filename).unlink()

    result = bundles.import_bundle(bundle)
    assert (result["imported"], result["present"], result["restored"], result["binned"]) == (1, 1, 1, int(binned))
    first, last = result["first_id"], result["last_id"]
    assert first == last and first > old_id
    assert first not in {rows[present.filename].id, rows[restoring.filename].id}
    found, total, _ = store.list_images(search=f"id:{first}-{last}", binned=binned)
    assert total == 1 and found[0].filename == new.filename
    assert store.list_images(search=f"id:{first}-{last}", binned=not binned)[1] == 0

    again = bundles.import_bundle(bundle)
    assert (again["imported"], again["present"], again["restored"]) == (0, 3, 0)
    assert again["first_id"] is again["last_id"] is None


def test_an_empty_bundle_has_no_new_row_range(store):
    result = bundles.import_bundle(Path(export_all()["path"]))
    assert result["imported"] == result["present"] == result["restored"] == 0
    assert result["first_id"] is result["last_id"] is None
