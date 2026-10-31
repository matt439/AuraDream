"""Stored recipes travel without making them available for generation."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import sqlite3
import zlib
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image, PngImagePlugin

from backend import bundle_images, bundles, reindex, storage
from backend.execution import canonical_execution, execution_key
from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEYS, require_header_metadata
from tests.bundle_helpers import use_destination, png_metadata, save_source, bundle_snapshot


_ABSENT = object()
_COMPILATION = {"compilation": "regional-v1"}
_ATTENTION = {"attention_backend": "cudnn-v1"}
_CACHE = {"step_cache": "fbc-005-v1"}
_ALL = {**_COMPILATION, **_ATTENTION, **_CACHE}
_INVALID = [False, [], {"unknown": None}, {"compilation": "future-v2"}]


def _sql(database, image_id, value):
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("UPDATE images SET execution = ? WHERE id = ?", (value, image_id))
        conn.commit()


def _png(path, value):
    meta = png_metadata(path)
    meta.pop("execution", None)
    if value is not _ABSENT:
        meta["execution"] = value
    bundle_images._rewrite_metadata(path, meta)


def _stored(store, row, recipe):
    # Fixtures represent a future gallery. GenerateRequest deliberately
    # refuses these recipes, so do not use request-validation bypasses.
    _sql(store.settings.db_path, row.id, execution_key(recipe))
    _png(store.settings.outputs_dir / row.filename, recipe)


def _backup(**kwargs):
    return Path(bundles.export_bundle(keep_gallery=True, **kwargs)["path"])


def _reencode(path, raw_metadata=None):
    with Image.open(path) as image:
        chunks = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(value, str):
                chunks.add_text(key, raw_metadata if key == reindex.PNG_METADATA_KEY and raw_metadata is not None else value)
        image.load()
        image.save(path, compress_level=0, pnginfo=chunks)


def _text_chunk(kind, key, raw):
    keyword = key.encode("ascii") + b"\0"
    value = raw.encode("utf-8")
    if kind == b"zTXt":
        payload = keyword + b"\0" + zlib.compress(value)
    elif kind == b"iTXt":
        payload = keyword + b"\0\0\0\0" + value
    else:
        payload = keyword + value
    return len(payload).to_bytes(4, "big") + kind + payload + zlib.crc32(kind + payload).to_bytes(4, "big")


def _placed_text(data, kind, key, raw, *, late):
    """Move just the synthetic app chunk; compressed image bytes stay exact."""
    result = bytearray(data[:8])
    offset = 8
    inserted = False
    replacement = _text_chunk(kind, key, raw)
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        end = offset + length + 12
        chunk_kind = data[offset + 4:offset + 8]
        keyword = reindex._text_keyword(chunk_kind, data[offset + 8:offset + 8 + min(length, 80)])
        if keyword in {name.encode("ascii") for name in (*PNG_METADATA_KEYS, PNG_MASK_KEY)}:
            offset = end
            continue
        if chunk_kind == (b"IEND" if late else b"IDAT") and not inserted:
            result.extend(replacement)
            inserted = True
        result.extend(data[offset:end])
        offset = end
    assert inserted
    return bytes(result)


def _edited_archive(archive, tmp_path, *, sql=_ABSENT, png=_ABSENT, drop_column=False, late_kind=None):
    with ZipFile(archive) as source:
        members = {info.filename: source.read(info) for info in source.infolist()}
    database = tmp_path / "edited.sqlite3"
    database.write_bytes(members["database.sqlite3"])
    with closing(sqlite3.connect(database)) as conn:
        filename = conn.execute("SELECT filename FROM images ORDER BY id LIMIT 1").fetchone()[0]
        if drop_column:
            conn.execute("ALTER TABLE images DROP COLUMN execution")
        elif sql is not _ABSENT:
            conn.execute("UPDATE images SET execution = ?", (sql,))
        conn.commit()
    members["database.sqlite3"] = database.read_bytes()
    if png is not _ABSENT:
        name = "images/" + filename
        with Image.open(io.BytesIO(members[name])) as image:
            meta = json.loads(image.info[reindex.PNG_METADATA_KEY])
        meta["execution"] = png
        members[name] = reindex._with_auradream_metadata(members[name], meta)
    if late_kind is not None:
        name = "images/" + filename
        with Image.open(io.BytesIO(members[name])) as image:
            raw = image.info[reindex.PNG_METADATA_KEY]
        members[name] = _placed_text(members[name], late_kind, reindex.PNG_METADATA_KEY, raw, late=True)
    manifest = json.loads(members["manifest.json"])
    for name, data in members.items():
        if name != "manifest.json":
            manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    members["manifest.json"] = json.dumps(manifest).encode()
    edited = tmp_path / "edited.zip"
    with ZipFile(edited, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    return edited


@pytest.mark.parametrize("route", ["bundle", "database", "png"])
@pytest.mark.parametrize("recipe", [_COMPILATION, _ATTENTION, _CACHE, _ALL])
def test_known_recipes_survive_each_transport(store, monkeypatch, tmp_path, route, recipe):
    row = save_source()
    _stored(store, row, recipe)
    source = store.settings
    original = (source.outputs_dir / row.filename).read_bytes()
    archive = _backup() if route == "bundle" else None
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    save_source("existing", 99)
    if archive is not None:
        assert bundles.import_bundle(archive)["imported"] == 1
    else:
        counts = reindex.reindex(source.outputs_dir, database=source.db_path if route == "database" else None)
        assert counts["indexed"] == 1
    (back,) = [item for item in store.list_images()[0] if item.filename == row.filename]
    assert back.execution == canonical_execution(recipe)
    assert png_metadata(dest.outputs_dir / row.filename)["execution"] == recipe
    with closing(sqlite3.connect(dest.db_path)) as conn:
        assert conn.execute("SELECT execution FROM images WHERE id = ?", (back.id,)).fetchone()[0] == execution_key(recipe)
    assert (source.outputs_dir / row.filename).read_bytes() == original


@pytest.mark.parametrize("route", ["bundle", "database", "png"])
@pytest.mark.parametrize("lost", [False, True])
def test_remapping_or_losing_a_source_keeps_the_execution_recipe(store, monkeypatch, tmp_path, route, lost):
    source_row = save_source("reference", 1)
    derived = save_source("derived", 2, reference=[source_row.id], refine=source_row.id, strength=0.35)
    _stored(store, derived, _ALL)
    source = store.settings
    if route == "bundle":
        archive = _backup(ids=[derived.id] if lost else None)
    elif lost:
        (source.outputs_dir / source_row.filename).unlink()
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    save_source("existing", 99)
    if route == "bundle":
        result = bundles.import_bundle(archive)
    else:
        result = reindex.reindex(source.outputs_dir, database=source.db_path if route == "database" else None)
    records = {item.filename: item for item in store.list_images()[0]}
    back = records[derived.filename]
    meta = png_metadata(dest.outputs_dir / derived.filename)
    assert back.execution == canonical_execution(_ALL) and meta["execution"] == _ALL
    if lost:
        assert result["unresolved_references"] == 2
        assert back.reference is None and back.refine is None and back.strength is None
    else:
        remapped = records[source_row.filename].id
        assert remapped != source_row.id
        assert back.reference == [remapped] and back.refine == remapped
        assert meta["reference"] == [remapped] and meta["refine"] == remapped


@pytest.mark.parametrize(("sql", "png", "expected", "invalid"), [
    (_ABSENT, _COMPILATION, _COMPILATION, False),
    (_COMPILATION, _ABSENT, _COMPILATION, False),
    (_COMPILATION, _COMPILATION, _COMPILATION, False),
    (_ABSENT, _ABSENT, None, False),
    (None, _ABSENT, None, False),
    (_ABSENT, {}, None, False),
    (None, None, None, False),
    ({}, {}, None, False),
    (None, _COMPILATION, None, True),
    ({}, _COMPILATION, None, True),
    (_COMPILATION, None, None, True),
    (_COMPILATION, {}, None, True),
    (_COMPILATION, _ATTENTION, None, True),
])
def test_database_and_png_presence_matrix(store, monkeypatch, tmp_path, sql, png, expected, invalid):
    row = save_source()
    source = store.settings
    path = source.outputs_dir / row.filename
    _png(path, png)
    if sql is _ABSENT:
        with closing(sqlite3.connect(source.db_path)) as conn:
            conn.execute("ALTER TABLE images DROP COLUMN execution")
            conn.commit()
    else:
        _sql(source.db_path, row.id, json.dumps(sql) if sql is not None else None)
    before = path.read_bytes()
    dest = dataclasses.replace(source, db_path=tmp_path / "rebuilt.db")
    monkeypatch.setattr(storage, "settings", dest)
    monkeypatch.setattr(reindex, "settings", dest)
    counts = reindex.merge_database(source.outputs_dir, source.db_path, out=io.StringIO())
    if invalid:
        assert counts["invalid"] == 1 and counts["indexed"] == 0
        assert store.list_images()[1] == 0 and path.read_bytes() == before
    else:
        assert counts["indexed"] == 1
        (back,) = store.list_images()[0]
        assert back.execution == canonical_execution(expected)
        if expected is None or png is not _ABSENT:
            assert path.read_bytes() == before
        else:
            assert png_metadata(path)["execution"] == expected


@pytest.mark.parametrize("origin", ["sql", "png"])
@pytest.mark.parametrize("bad", _INVALID)
def test_invalid_database_execution_is_skipped_before_image_side_effects(store, monkeypatch, tmp_path, origin, bad):
    row = save_source()
    source = store.settings
    if origin == "sql":
        _sql(source.db_path, row.id, json.dumps(bad))
    else:
        _png(source.outputs_dir / row.filename, bad)
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    before = bundle_snapshot(store)

    def forbidden(*args, **kwargs):
        pytest.fail("a refused image reached mask recovery, copying or insertion")

    monkeypatch.setattr(reindex._Masks, "resolve", forbidden)
    monkeypatch.setattr(reindex, "_copy_in", forbidden)
    monkeypatch.setattr(storage, "import_file", forbidden)
    counts = reindex.merge_database(source.outputs_dir, source.db_path, out=io.StringIO())
    assert counts["invalid"] == 1 and counts["indexed"] == 0
    assert bundle_snapshot(store) == before


@pytest.mark.parametrize("bad", _INVALID)
def test_invalid_png_recipe_is_skipped_before_links_masks_or_writes(store, monkeypatch, tmp_path, bad):
    row = save_source()
    source = store.settings
    _png(source.outputs_dir / row.filename, bad)
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    before = bundle_snapshot(store)

    def forbidden(*args, **kwargs):
        pytest.fail("a refused image reached mask recovery, copying or insertion")

    monkeypatch.setattr(reindex._Masks, "resolve", forbidden)
    monkeypatch.setattr(reindex, "_copy_in", forbidden)
    monkeypatch.setattr(storage, "index_file", forbidden)
    counts = reindex.reindex(source.outputs_dir, out=io.StringIO())
    assert counts["invalid_execution"] == 1 and counts["indexed"] == 0
    assert bundle_snapshot(store) == before


@pytest.mark.parametrize("origin", ["sql", "png"])
@pytest.mark.parametrize("bad", _INVALID)
def test_invalid_execution_refuses_export_without_clearing(store, origin, bad):
    row = save_source()
    if origin == "sql":
        _sql(store.settings.db_path, row.id, json.dumps(bad))
    else:
        _png(store.settings.outputs_dir / row.filename, bad)
    before = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError, match="execution"):
        bundles.export_bundle(expected=1)
    assert bundle_snapshot(store) == before


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("origin", ["sql", "png"])
def test_verified_archive_refuses_unknown_execution_even_on_repeat_fast_path(store, monkeypatch, tmp_path, origin, repeat):
    save_source()
    original = _backup()
    bad = {"compilation": "future-v2"}
    damaged = _edited_archive(original, tmp_path, **({"sql": json.dumps(bad)} if origin == "sql" else {"png": bad}))
    use_destination(store, monkeypatch, tmp_path)
    if repeat:
        assert bundles.import_bundle(original)["imported"] == 1

        def no_decode(*args, **kwargs):
            pytest.fail("the repeat fast path decoded pixels")

        monkeypatch.setattr(bundle_images, "_pixels", no_decode)
    before = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError, match="execution"):
        bundles.import_bundle(damaged)
    assert bundle_snapshot(store) == before


def test_archive_with_an_older_database_recovers_a_known_png_recipe(store, monkeypatch, tmp_path):
    row = save_source()
    _stored(store, row, _ALL)
    archive = _edited_archive(_backup(), tmp_path, drop_column=True)
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(archive)["imported"] == 1
    (back,) = store.list_images()[0]
    assert back.execution == canonical_execution(_ALL)
    assert png_metadata(dest.outputs_dir / row.filename)["execution"] == _ALL


@pytest.mark.parametrize("other", [None, _ATTENTION])
def test_identical_pixels_with_different_execution_refuse_the_whole_import(store, monkeypatch, tmp_path, other):
    row = save_source()
    _stored(store, row, _COMPILATION)
    archive = _backup()
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(archive)["imported"] == 1
    (known,) = store.list_images()[0]
    _stored(store, known, other)
    before = bundle_snapshot(store)
    with pytest.raises(bundles.BundleError, match="different execution"):
        bundles.import_bundle(archive)
    assert bundle_snapshot(store) == before
    assert png_metadata(dest.outputs_dir / row.filename).get("execution") == other


@pytest.mark.parametrize("reencoded", [False, True])
@pytest.mark.parametrize("kept_recipe", [None, {}, _ATTENTION, *_INVALID])
def test_repeat_bundle_refuses_a_destination_png_that_disagrees_with_its_own_row(
    store, monkeypatch, tmp_path, reencoded, kept_recipe,
):
    row = save_source()
    _stored(store, row, _COMPILATION)
    archive = _backup()
    dest = use_destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(archive)["imported"] == 1
    path = dest.outputs_dir / row.filename
    _png(path, kept_recipe)
    if reencoded:
        _reencode(path)
    with ZipFile(archive) as source, path.open("rb") as kept:
        assert (bundle_images._critical_digest(kept) == bundle_images._critical_digest(
            io.BytesIO(source.read("images/" + row.filename))
        )) is not reencoded
    before = bundle_snapshot(store)

    def no_decode(*args, **kwargs):
        pytest.fail("contradictory destination metadata reached pixel decoding")

    monkeypatch.setattr(bundle_images, "_pixels", no_decode)
    with pytest.raises(bundles.BundleError, match="execution"):
        bundles.import_bundle(archive)
    assert bundle_snapshot(store) == before


@pytest.mark.parametrize("kept_recipe", [None, {}, _ATTENTION, *_INVALID])
def test_database_reindex_refuses_contradictory_destination_metadata_before_claiming_present(
    store, monkeypatch, tmp_path, kept_recipe,
):
    row = save_source()
    _stored(store, row, _COMPILATION)
    source = store.settings
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    assert reindex.merge_database(source.outputs_dir, source.db_path)["indexed"] == 1
    target = dest.outputs_dir / row.filename
    _png(target, kept_recipe)
    before = bundle_snapshot(store)
    real_open = Image.open

    def header_only(path, *args, **kwargs):
        image = real_open(path, *args, **kwargs)
        if Path(getattr(path, "name", path)) == target:
            image.load = forbidden
        return image

    def forbidden(*args, **kwargs):
        pytest.fail("contradictory destination metadata reached pixel decoding or an image write")

    monkeypatch.setattr(Image, "open", header_only)
    monkeypatch.setattr(reindex._Masks, "resolve", forbidden)
    monkeypatch.setattr(reindex, "_copy_in", forbidden)
    monkeypatch.setattr(storage, "import_file", forbidden)
    counts = reindex.merge_database(source.outputs_dir, source.db_path, out=io.StringIO())
    assert counts["conflicts"] == 1 and counts["present"] == counts["indexed"] == 0
    assert bundle_snapshot(store) == before


@pytest.mark.parametrize("route", ["bundle", "database"])
@pytest.mark.parametrize("damage", ["invalid-json", "non-object-json", "null-json", "not-png", "missing"])
def test_an_uncertain_destination_header_is_refused_without_repair(
    store, monkeypatch, tmp_path, route, damage,
):
    row = save_source()
    _stored(store, row, _COMPILATION)
    source = store.settings
    archive = _backup()
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    assert bundles.import_bundle(archive)["imported"] == 1
    target = dest.outputs_dir / row.filename
    if damage == "missing":
        target.unlink()
    elif damage == "not-png":
        target.write_bytes(b"not a PNG")
    else:
        _reencode(target, raw_metadata={"invalid-json": "{", "non-object-json": "[]", "null-json": "null"}[damage])
    before = bundle_snapshot(store)
    if route == "bundle" and damage == "missing":
        # Missing files retain the existing, deliberate verified restore
        # behavior; contradictory files must never take this route.
        assert bundles.import_bundle(archive)["restored"] == 1
        assert png_metadata(target)["execution"] == _COMPILATION
    elif route == "bundle":
        with pytest.raises(bundles.BundleError, match="execution"):
            bundles.import_bundle(archive)
        assert bundle_snapshot(store) == before
    else:
        counts = reindex.merge_database(source.outputs_dir, source.db_path, out=io.StringIO())
        assert counts["conflicts"] == 1 and counts["present"] == counts["indexed"] == 0
        assert bundle_snapshot(store) == before


@pytest.mark.parametrize("route", ["database", "png"])
@pytest.mark.parametrize("empty", [_ABSENT, None, {}, {"compilation": None, "attention_backend": None, "step_cache": None}])
def test_empty_legacy_execution_does_not_rewrite_in_place_pngs(store, monkeypatch, tmp_path, route, empty):
    row = save_source()
    source = store.settings
    path = source.outputs_dir / row.filename
    _png(path, empty)
    before = path.read_bytes()
    dest = dataclasses.replace(source, db_path=tmp_path / "rebuilt.db")
    monkeypatch.setattr(storage, "settings", dest)
    monkeypatch.setattr(reindex, "settings", dest)
    counts = reindex.reindex(source.outputs_dir, database=source.db_path if route == "database" else None)
    assert counts["indexed"] == 1 and path.read_bytes() == before
    (back,) = store.list_images()[0]
    assert back.execution is None


@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt"])
@pytest.mark.parametrize("key", [*PNG_METADATA_KEYS, PNG_MASK_KEY])
@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("recipe", [_COMPILATION, {"compilation": "future-v2"}])
def test_header_metadata_policy_covers_app_keywords_and_text_encodings(kind, key, late, recipe):
    original = io.BytesIO()
    Image.new("RGB", (1, 1)).save(original, format="PNG")
    stream = io.BytesIO(_placed_text(original.getvalue(), kind, key, json.dumps({"execution": recipe}), late=late))
    stream.seek(17)
    if late:
        with pytest.raises(ValueError, match="metadata must precede PNG image data"):
            require_header_metadata(stream)
    else:
        require_header_metadata(stream)
    assert stream.tell() == 17


@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt"])
def test_header_metadata_scan_allows_unrelated_late_comments_and_never_reads_payloads(kind):
    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 80, "metadata placement check read a payload"
            return super().read(size)

    original = io.BytesIO()
    Image.new("RGB", (1, 1)).save(original, format="PNG")
    data = _placed_text(original.getvalue(), kind, "Comment", "unrelated " * 10000, late=True)
    stream = Bounded(data)
    stream.seek(17)
    require_header_metadata(stream)
    assert stream.tell() == 17


@pytest.mark.parametrize("route", ["database", "png", "export", "import", "repeat-source", "repeat-destination"])
@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt"])
@pytest.mark.parametrize("recipe", [_COMPILATION, {"compilation": "future-v2"}])
def test_late_app_metadata_is_refused_before_transport_writes(store, monkeypatch, tmp_path, route, kind, recipe):
    row = save_source()
    source = store.settings
    path = source.outputs_dir / row.filename
    archive = _backup() if route in {"import", "repeat-source", "repeat-destination"} else None
    if route in {"import", "repeat-source"}:
        assert archive is not None
        damaged = _edited_archive(archive, tmp_path, png=recipe, late_kind=kind)
    else:
        meta = png_metadata(path)
        meta["execution"] = recipe
        path.write_bytes(_placed_text(path.read_bytes(), kind, reindex.PNG_METADATA_KEY, json.dumps(meta), late=True))
    if route == "export":
        before = bundle_snapshot(store)
        with pytest.raises(bundles.BundleError, match="metadata must precede PNG image data"):
            bundles.export_bundle(expected=1)
        assert bundle_snapshot(store) == before
        return
    dest = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", dest)
    if route in {"repeat-source", "repeat-destination"}:
        assert archive is not None
        assert bundles.import_bundle(archive)["imported"] == 1
        if route == "repeat-destination":
            target = dest.outputs_dir / row.filename
            meta = png_metadata(target)
            meta["execution"] = recipe
            target.write_bytes(_placed_text(target.read_bytes(), kind, reindex.PNG_METADATA_KEY, json.dumps(meta), late=True))

        def no_decode(*args, **kwargs):
            pytest.fail("late app metadata reached a pixel decode")

        monkeypatch.setattr(bundle_images, "_pixels", no_decode)
    before = bundle_snapshot(store)
    if route in {"import", "repeat-source", "repeat-destination"}:
        chosen = archive if route == "repeat-destination" else damaged
        assert chosen is not None
        with pytest.raises(bundles.BundleError, match="metadata must precede PNG image data"):
            bundles.import_bundle(chosen)
    else:
        counts = reindex.reindex(source.outputs_dir, database=source.db_path if route == "database" else None, out=io.StringIO())
        assert counts["indexed"] == 0
        assert counts["invalid"] == 1 if route == "database" else counts["invalid_execution"] == 1
    assert bundle_snapshot(store) == before
