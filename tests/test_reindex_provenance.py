"""PNG recovery must never interpret another database's conditioning ids."""
from __future__ import annotations

import dataclasses
import json

import pytest
from PIL import Image

from backend import reindex, storage
from backend.schemas import GenerateRequest


def _save(prompt, seed, **kwargs):
    return storage.save_image(
        Image.new("RGB", (32, 24), "red" if prompt == "source" else "blue"),
        GenerateRequest(prompt=prompt, steps=28, **kwargs),
        seed=seed, duration_ms=10, model_id="test/model",
    )


def _metadata(path):
    with Image.open(path) as image:
        return json.loads(image.info["local-diffusion"])


def _rewrite_metadata(path, meta):
    path.write_bytes(reindex._with_local_diffusion_metadata(path.read_bytes(), meta))


@pytest.mark.parametrize("legacy", [False, True])
def test_png_import_remaps_proven_sources_and_drops_legacy_ids(store, monkeypatch, tmp_path, legacy):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    unrelated = _save("unrelated", 9)
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=tmp_path / "other")
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    # The dependent name sorts first within the same timestamp. Recovery must
    # visit its named source before assigning the dependent's gallery id.
    source = _save("source", 9)
    conditioned = _save("reference", 1, reference=[source.id])
    refined = _save("refined", 2, refine=source.id, strength=0.3)
    assert source.id == unrelated.id
    if legacy:
        for record in (conditioned, refined):
            path = other.outputs_dir / record.filename
            meta = _metadata(path)
            meta.pop("reference_sources")
            meta.pop("refine_source")
            _rewrite_metadata(path, meta)
    before = {p.name: p.read_bytes() for p in other.outputs_dir.glob("*.png")}
    monkeypatch.setattr(storage, "settings", mine)

    counts = reindex.reindex(other.outputs_dir)

    by_prompt = {r.prompt: r for r in store.list_images()[0]}
    assert counts["indexed"] == 3
    assert counts["unresolved_references"] == (2 if legacy else 0)
    imported = by_prompt["source"]
    assert imported.id != source.id
    assert by_prompt["reference"].reference == (None if legacy else [imported.id])
    assert by_prompt["refined"].refine == (None if legacy else imported.id)
    assert by_prompt["refined"].strength == (None if legacy else 0.3)
    for record in (conditioned, refined):
        meta = _metadata(mine.outputs_dir / record.filename)
        rebuilt = by_prompt[record.prompt]
        assert (meta["reference"], meta["refine"], meta["strength"]) == (
            rebuilt.reference, rebuilt.refine, rebuilt.strength,
        )
    assert {p.name: p.read_bytes() for p in other.outputs_dir.glob("*.png")} == before
    assert reindex.reindex(other.outputs_dir)["present"] == 3


def test_rebuild_after_deleted_ids_remaps_png_metadata_too(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    removed = _save("removed", 1)
    source = _save("source", 9)
    refined = _save("refined", 2, refine=source.id, strength=0.3)
    conditioned = _save("reference", 3, reference=[source.id])
    (store.settings.outputs_dir / removed.filename).unlink()
    store.settings.db_path.unlink()
    before = {p.name: p.read_bytes() for p in store.settings.outputs_dir.glob("*.png")}

    dry = reindex.reindex(store.settings.outputs_dir, dry_run=True)
    assert dry["indexed"] == 3 and dry["unresolved_references"] == 0
    assert not store.settings.db_path.exists()
    assert {p.name: p.read_bytes() for p in store.settings.outputs_dir.glob("*.png")} == before
    counts = reindex.reindex(store.settings.outputs_dir)

    rebuilt = {r.filename: r for r in store.list_images()[0]}
    new_source = rebuilt[source.filename]
    assert new_source.id != source.id
    assert counts["unresolved_references"] == 0
    assert rebuilt[refined.filename].refine == new_source.id
    assert rebuilt[conditioned.filename].reference == [new_source.id]
    assert _metadata(store.settings.outputs_dir / refined.filename)["refine"] == new_source.id
    assert _metadata(store.settings.outputs_dir / conditioned.filename)["reference"] == [new_source.id]
    assert (store.settings.outputs_dir / source.filename).read_bytes() == before[source.filename]


@pytest.mark.parametrize("broken", ["missing", "hash", "size", "filename", "pixels"])
def test_unproven_source_is_dropped_instead_of_binding_a_valid_id(store, monkeypatch, broken):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = _save("source", 9)
    refined = _save("refined", 1, refine=source.id, strength=0.3)
    path = store.settings.outputs_dir / refined.filename
    meta = _metadata(path)
    if broken == "missing":
        (store.settings.outputs_dir / source.filename).unlink()
    elif broken == "pixels":
        # The row still claims the original hash, but this is what a pass
        # would actually load through the otherwise-valid gallery id.
        Image.new("RGB", (32, 24), "green").save(store.settings.outputs_dir / source.filename)
    elif broken == "hash":
        meta["refine_source"]["hash"] = "0" * 16
    elif broken == "size":
        meta["refine_source"]["width"] += 1
    else:
        meta["refine_source"]["filename"] = "../outside.png"
    _rewrite_metadata(path, meta)
    with store._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (refined.id,))

    counts = reindex.reindex(store.settings.outputs_dir)

    rebuilt = next(r for r in store.list_images()[0] if r.filename == refined.filename)
    assert counts["unresolved_references"] == 1
    assert rebuilt.refine is None and rebuilt.strength is None
    assert _metadata(path)["refine"] is None


def test_partial_same_gallery_reindex_preserves_verified_links(store, monkeypatch):
    monkeypatch.setattr(reindex, "settings", store.settings)
    source = _save("source", 9)
    refined = _save("refined", 1, refine=source.id, strength=0.3)
    with store._session() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (refined.id,))

    counts = reindex.reindex(store.settings.outputs_dir)

    rebuilt = next(r for r in store.list_images()[0] if r.filename == refined.filename)
    assert counts["unresolved_references"] == 0
    assert (rebuilt.refine, rebuilt.strength) == (source.id, 0.3)


def test_database_merge_dry_run_with_sources_does_not_create_destination_db(store, monkeypatch, tmp_path):
    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    other = dataclasses.replace(mine, db_path=tmp_path / "other.db", outputs_dir=tmp_path / "other")
    monkeypatch.setattr(storage, "settings", other)
    storage.init_db()
    source = _save("source", 9)
    _save("reference", 1, reference=[source.id])
    _save("refined", 2, refine=source.id, strength=0.3)
    monkeypatch.setattr(storage, "settings", mine)
    mine.db_path.unlink()
    before = set(mine.outputs_dir.iterdir())

    counts = reindex.reindex(other.outputs_dir, database=other.db_path, dry_run=True)

    assert counts["indexed"] == 3 and counts["unresolved_references"] == 0
    assert not mine.db_path.exists()
    assert set(mine.outputs_dir.iterdir()) == before
