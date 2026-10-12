"""An inpaint's mask carried where its image goes: in a bundle (the members,
the table, the merge by hash, and what is refused), and back through a
reindex (from `masks/`, from another gallery's database, and from the copy
every result carries in its own PNG)."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image

from backend import bundle_images, bundles, reindex, storage, submission
from backend.metadata import PNG_MASK_KEY
from backend.schemas import GenerateRequest
from tests.bundle_helpers import _destination, _export, _metadata, _save, _snapshot


def _mask(size=(256, 256), box=(0, 0, 128, 64)) -> Image.Image:
    image = Image.new("L", size, 0)
    image.paste(255, box)
    return image


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _inpainted(source, digest, *, seed=2, prompt="a red scarf"):
    return storage.save_image(
        Image.new("RGB", (source.width, source.height), (200, 10, 10)),
        GenerateRequest(prompt=prompt, width=source.width, height=source.height, steps=28,
                        refine=source.id, strength=0.6, mask=digest),
        seed=seed, duration_ms=10, model_id="test/model", job_id="job-one", run_id="run-one",
    )


def _inpaint_gallery():
    """A source, an inpaint of it under one mask, and a second mask nothing names."""
    source = _save("source", 1)
    digest, _ = storage.store_mask(_mask())
    unused, _ = storage.store_mask(_mask(box=(0, 0, 16, 16)))
    return source, _inpainted(source, digest), digest, unused


def _tamper(path: Path, member: str, *, data: bytes | None = None, rename: str | None = None) -> Path:
    """A copy of a plain bundle with one member replaced or renamed, its
    manifest entry recomputed so that only the member's own check can fail."""
    with zipfile.ZipFile(path) as archive:
        items = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(items.pop("manifest.json"))
    if rename is not None:
        items[rename] = items.pop(member)
        manifest["files"].pop(member)
        member = rename
    if data is not None:
        items[member] = data
    manifest["files"][member] = {"size": len(items[member]), "sha256": hashlib.sha256(items[member]).hexdigest()}
    out = path.with_name("tampered.zip")
    with zipfile.ZipFile(out, "w") as archive:
        for name, body in items.items():
            archive.writestr(name, body)
        archive.writestr("manifest.json", json.dumps(manifest))
    return out


def _bundle_masks(path: Path, tmp_path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        (tmp_path / "unpacked.sqlite3").write_bytes(archive.read("database.sqlite3"))
    with closing(sqlite3.connect(tmp_path / "unpacked.sqlite3")) as conn:
        return {row[0]: (row[1], row[2]) for row in conn.execute("SELECT hash, width, height FROM masks")}


# ------------------------------------------------------------------ bundles


def test_a_bundle_carries_the_masks_and_the_import_merges_them_by_hash(store, monkeypatch, tmp_path):
    source, made, digest, unused = _inpaint_gallery()
    assert bundles.preview()["masks"] == 2
    archive = _export()
    assert archive["cleared"] and archive["masks"] == 2
    with zipfile.ZipFile(archive["path"]) as bundle:
        names = set(bundle.namelist())
        manifest = json.loads(bundle.read("manifest.json"))
    assert {f"masks/{digest}.png", f"masks/{unused}.png"} <= names
    assert manifest["version"] == 1
    assert _bundle_masks(Path(archive["path"]), tmp_path) == {digest: (256, 256), unused: (256, 256)}
    # Cleared with the gallery: the files and their rows.
    assert list(store.masks_dir().glob("*.png")) == [] and store._tracked_masks() == set()

    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["masks"], imported["masks_missing"]) == (2, 2, 0)
    inpaint = next(r for r in store.list_images()[0] if r.mask)
    new_source = next(r for r in store.list_images()[0] if r.filename == source.filename)
    # The source is renumbered; the mask, named by its pixels, is not.
    assert (inpaint.refine, inpaint.mask) == (new_source.id, digest)
    assert store.mask_size(digest) == (256, 256) and store.mask_path(digest).is_file()
    meta = _metadata(store.settings.outputs_dir / inpaint.filename)
    assert (meta["refine"], meta["mask"], meta["mask_blend"]) == (new_source.id, digest, "overlay")
    # The rewrite of the record leaves the mask's own copy in the PNG.
    with Image.open(store.settings.outputs_dir / inpaint.filename) as handle:
        assert PNG_MASK_KEY in handle.info
    assert submission.missing_masks([inpaint]) == []

    # Again: every image present, every mask the same one already here.
    again = bundles.import_bundle(Path(archive["path"]))
    assert (again["imported"], again["present"], again["masks"], again["masks_missing"]) == (0, 2, 0, 0)


def test_the_identity_of_an_inpaint_ignores_its_text_chunks(store):
    source, made, digest, _ = _inpaint_gallery()
    data = (store.settings.outputs_dir / made.filename).read_bytes()
    # The same PNG with every text chunk taken out: the mask's copy and the
    # record are metadata, never what decides the pixels.
    stripped = bytearray(data[:8])
    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        end = offset + 12 + length
        if data[offset + 4:offset + 8] not in {b"tEXt", b"zTXt", b"iTXt"}:
            stripped += data[offset:end]
        offset = end
    assert PNG_MASK_KEY.encode() in data and PNG_MASK_KEY.encode() not in stripped
    assert bundle_images._critical_digest(io.BytesIO(data)) == bundle_images._critical_digest(io.BytesIO(bytes(stripped)))


def test_a_selection_carries_its_own_masks_and_no_other(store, monkeypatch, tmp_path):
    source, made, digest, unused = _inpaint_gallery()
    archive = bundles.export_bundle(keep_gallery=True, ids=[source.id, made.id])
    assert archive["masks"] == 1
    with zipfile.ZipFile(archive["path"]) as bundle:
        members = [name for name in bundle.namelist() if name.startswith("masks/")]
    assert members == [f"masks/{digest}.png"]
    assert _bundle_masks(Path(archive["path"]), tmp_path) == {digest: (256, 256)}
    # A backup: the gallery keeps both masks.
    assert store._tracked_masks() == {digest, unused}
    # A selection without the inpaint carries no mask at all.
    plain = bundles.export_bundle(keep_gallery=True, ids=[source.id])
    assert plain["masks"] == 0 and _bundle_masks(Path(plain["path"]), tmp_path) == {}


def _without_embedded_mask(store, record) -> None:
    """Rewrite `record`'s PNG without the copy of its mask, its pixels and
    every other chunk as they were."""
    from PIL import PngImagePlugin

    path = store.settings.outputs_dir / record.filename
    with Image.open(path) as image:
        image.load()
        info = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(value, str) and key != PNG_MASK_KEY:
                info.add_itxt(key, value)
        image.save(path, format="PNG", pnginfo=info)


def test_an_import_restores_a_lost_mask_from_its_results_own_png(store, monkeypatch, tmp_path):
    source, made, digest, _ = _inpaint_gallery()
    # Lost in the gallery it came from: no member carries it.
    store.mask_path(digest).unlink()
    archive = bundles.export_bundle(keep_gallery=True, ids=[source.id, made.id])
    assert archive["masks"] == 0

    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["masks"], imported["masks_missing"]) == (2, 1, 0)
    inpaint = next(r for r in store.list_images()[0] if r.mask)
    assert inpaint.mask == digest and store.mask_size(digest) == (256, 256)
    with Image.open(store.mask_path(digest)) as handle:
        assert store.mask_hash(handle) == digest
    assert submission.missing_masks([inpaint]) == []


def test_a_row_whose_mask_the_bundle_lacks_keeps_its_hash_and_is_counted(store, monkeypatch, tmp_path):
    source, made, digest, _ = _inpaint_gallery()
    store.mask_path(digest).unlink()
    # Nor does its PNG hold a copy any more.
    _without_embedded_mask(store, made)
    archive = bundles.export_bundle(keep_gallery=True, ids=[source.id, made.id])
    assert archive["masks"] == 0

    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]))
    assert (imported["imported"], imported["masks"], imported["masks_missing"]) == (2, 0, 1)
    inpaint = next(r for r in store.list_images()[0] if r.mask)
    assert inpaint.mask == digest
    # What ran is still said; running it again is refused by its id.
    assert submission.missing_masks([inpaint]) == [inpaint.id]
    assert f"the mask of #{inpaint.id} is missing" in submission.missing_mask_detail([inpaint.id])


def test_a_password_bundle_carries_the_masks(store, monkeypatch, tmp_path):
    source, made, digest, _ = _inpaint_gallery()
    archive = bundles.export_bundle(keep_gallery=True, password="tea kettle")
    _destination(store, monkeypatch, tmp_path)
    imported = bundles.import_bundle(Path(archive["path"]), "tea kettle")
    assert (imported["masks"], imported["masks_missing"]) == (2, 0)
    assert store.mask_path(digest).is_file()


def test_a_whole_gallery_export_leaves_out_what_is_no_stored_mask(store, monkeypatch, tmp_path):
    source, made, digest, unused = _inpaint_gallery()
    masks = store.masks_dir()
    # A user's own mask copied in, a sync tool's duplicate, an upper-case
    # name, and a hash-named file that is RGB: none is a stored mask.
    strays = ["scarf-mask.png", f"{digest} (1).png", f"{'A' * 64}.png", f"{'b' * 64}.png"]
    for name in strays[:3]:
        (masks / name).write_bytes(_png(_mask()))
    (masks / strays[3]).write_bytes(_png(_mask().convert("RGB")))
    assert sorted(bundles.preview()["masks_left_out"]) == sorted(strays)
    assert bundles.preview()["masks"] == 2
    archive = bundles.export_bundle(keep_gallery=True)
    assert archive["masks"] == 2
    with zipfile.ZipFile(archive["path"]) as bundle:
        members = sorted(name for name in bundle.namelist() if name.startswith("masks/"))
    assert members == sorted([f"masks/{digest}.png", f"masks/{unused}.png"])
    # Left where they were, as a reindex leaves them.
    assert all((masks / name).is_file() for name in strays)


@pytest.mark.parametrize("change", ["repainted", "renamed", "rgb"])
def test_a_mask_that_is_not_the_one_its_name_says_is_refused(store, monkeypatch, tmp_path, change):
    source, made, digest, unused = _inpaint_gallery()
    archive = Path(bundles.export_bundle(keep_gallery=True)["path"])
    member = f"masks/{digest}.png"
    if change == "repainted":
        bad = _tamper(archive, member, data=_png(_mask(box=(0, 0, 200, 200))))
        words = "holds other pixels than its name says"
    elif change == "renamed":
        bad = _tamper(archive, member, rename=f"masks/{'e' * 64}.png")
        words = "holds other pixels than its name says"
    else:
        bad = _tamper(archive, member, data=_png(_mask().convert("RGB")))
        words = "is RGB, not one luminance channel"
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match=words):
        bundles.import_bundle(bad)
    assert _snapshot(store) == snapshot


def test_a_mask_of_another_size_under_the_same_hash_is_refused(store, monkeypatch, tmp_path):
    source, made, digest, _ = _inpaint_gallery()
    archive = Path(bundles.export_bundle(keep_gallery=True)["path"])
    _destination(store, monkeypatch, tmp_path)
    # Only a hand-edited table could say so: the hash covers the size.
    with store._session() as conn:
        conn.execute("INSERT INTO masks VALUES (?, 384, 384, '2026-01-01T00:00:00+00:00')", (digest,))
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="has another size in this gallery"):
        bundles.import_bundle(archive)
    assert _snapshot(store) == snapshot


def test_a_bundle_row_naming_a_path_as_its_mask_is_refused(store, monkeypatch, tmp_path):
    source, made, digest, _ = _inpaint_gallery()
    with store._session() as conn:
        conn.execute("UPDATE images SET mask = '../../secret.png' WHERE id = ?", (made.id,))
    with pytest.raises(bundles.BundleError, match="invalid inpaint mask"):
        bundles.export_bundle(keep_gallery=True)


# ------------------------------------------------------------------ reindex


def _lose_the_database(store):
    for suffix in ("", "-wal", "-shm"):
        Path(str(store.settings.db_path) + suffix).unlink(missing_ok=True)


def test_a_png_only_reindex_rebuilds_the_masks_from_their_files(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made, digest, unused = _inpaint_gallery()
    (store.masks_dir() / f"{'a' * 64}.png").write_bytes(_png(_mask(box=(0, 0, 8, 8))))
    _lose_the_database(store)
    counts = reindex.reindex(store.settings.outputs_dir)
    assert (counts["indexed"], counts["masks"], counts["bad_masks"], counts["unresolved_masks"]) == (2, 2, 1, 0)
    assert store._tracked_masks() == {digest, unused}
    inpaint = next(r for r in store.list_images()[0] if r.mask)
    assert inpaint.mask == digest and store.mask_path(digest).is_file()
    # Idempotent: the masks are held now.
    assert reindex.reindex(store.settings.outputs_dir)["masks"] == 0


def test_a_png_only_reindex_restores_a_lost_mask_from_its_result(store, monkeypatch, capsys):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source, made, digest, unused = _inpaint_gallery()
    pixels = _mask().tobytes()
    _lose_the_database(store)
    for path in store.masks_dir().glob("*.png"):
        path.unlink()
    assert reindex.main([]) == 0
    assert "1 inpaint mask(s) restored" in capsys.readouterr().out
    # The one a result names, from that result's own copy; the unused one
    # was in no PNG and is gone.
    assert store._tracked_masks() == {digest}
    with Image.open(store.mask_path(digest)) as handle:
        assert handle.mode == "L" and handle.tobytes() == pixels


def test_a_mask_found_nowhere_keeps_its_hash_and_is_counted(store, monkeypatch, capsys):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = _save("source", 1)
    digest, _ = storage.store_mask(_mask())
    # A result written while its mask could not be read carries no copy.
    monkeypatch.setattr(storage, "mask_png", lambda mask: None)
    _inpainted(source, digest)
    _lose_the_database(store)
    for path in store.masks_dir().glob("*.png"):
        path.unlink()
    assert reindex.main([]) == 0
    assert "1 inpaint mask(s) found nowhere" in capsys.readouterr().out
    inpaint = next(r for r in store.list_images()[0] if r.refine is not None)
    assert inpaint.mask == digest and submission.missing_masks([inpaint]) == [inpaint.id]


def test_a_database_merge_carries_the_masks_and_the_column(store, monkeypatch, tmp_path):
    import dataclasses

    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    _save("already here", 90)
    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "local-diffusion.db"
    monkeypatch.setattr(storage, "settings", dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere))
    storage.init_db()
    source, made, digest, unused = _inpaint_gallery()
    monkeypatch.setattr(storage, "settings", mine)

    counts = reindex.reindex(elsewhere, database=source_db)
    assert (counts["indexed"], counts["masks"], counts["unresolved_masks"]) == (2, 2, 0)
    rows = {r.filename: r for r in store.list_images(limit=10)[0]}
    inpaint, new_source = rows[made.filename], rows[source.filename]
    assert (inpaint.refine, inpaint.mask, inpaint.strength) == (new_source.id, digest, 0.6)
    assert store.mask_path(digest).is_file()
    meta = _metadata(mine.outputs_dir / made.filename)
    assert (meta["refine"], meta["mask"], meta["mask_blend"]) == (new_source.id, digest, "overlay")
