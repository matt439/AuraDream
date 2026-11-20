"""CPU recovery preserves Tiled Refine intent and proves both source images."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import pytest
from PIL import Image

from backend import reindex, storage, tiled_refine_contract as tiled
from backend.schemas import GenerateRequest
from tests.bundle_helpers import png_metadata


def _save_root(prompt):
    with closing(Image.new("RGB", tiled.ROOT_SIZE, "navy")) as image:
        image.paste("white", (100, 200, 500, 700))
        row = storage.save_image(image, GenerateRequest(prompt=prompt), 7, 1, "root/checkpoint")
        with closing(tiled.prepare_canvas(image)) as prepared:
            return row, tiled.StoredTiledRefine(
                source=row.id, recipe=tiled.RECIPE_NAME, preparation=tiled.PREPARATION,
                root_source=tiled.descriptor_from_image(row.filename, image),
                prepared_rgb_sha256=tiled.rgb_sha256(prepared),
            )


def _save_tiled(stored):
    request = GenerateRequest(
        prompt="contrasting scene", negative_prompt="halos", tiled_refine=stored,
        width=tiled.CANVAS_SIZE[0], height=tiled.CANVAS_SIZE[1], steps=tiled.STEPS,
        guidance_scale=tiled.GUIDANCE_SCALE, strength=tiled.STRENGTH, mu=None,
    )
    with closing(Image.new("RGB", tiled.CANVAS_SIZE, "green")) as image:
        return storage.save_image(image, request, 8, 1, tiled.MODEL_ID, quantization=tiled.QUANTIZATION)


def _destination(mine, monkeypatch, tmp_path):
    settings = replace(mine, db_path=tmp_path / "new.db", outputs_dir=tmp_path / "new")
    monkeypatch.setattr(storage, "settings", settings)
    monkeypatch.setattr(reindex, "settings", settings)
    storage.init_db()
    _save_root("foreign numeric id collision")
    return settings


@pytest.mark.parametrize("database", [False, True])
@pytest.mark.parametrize("damage", ["none", "root", "preparation", "missing", "unresolved", "retained", "retained-hash"])
def test_reindex_remaps_only_proven_tiled_roots(store, monkeypatch, tmp_path, database, damage):
    mine = store.settings
    root, proof = _save_root("root")
    result = _save_tiled(proof)
    if damage == "root":
        proof = proof.model_copy(update={"root_source": proof.root_source.model_copy(update={"hash": "0" * 64})})
    elif damage == "preparation":
        proof = proof.model_copy(update={"prepared_rgb_sha256": "0" * 64})
    elif damage == "missing":
        (mine.outputs_dir / root.filename).unlink()
    elif damage == "unresolved":
        proof = proof.model_copy(update={"source": None})
    if damage in {"root", "preparation", "unresolved"}:
        with store._session() as conn:
            conn.execute("UPDATE images SET tiled_refine = ? WHERE id = ?", (tiled.stored_json(proof), result.id))
        path = mine.outputs_dir / result.filename
        metadata = png_metadata(path)
        metadata["tiled_refine"] = proof.model_dump(mode="json")
        path.write_bytes(reindex._with_auradream_metadata(path.read_bytes(), metadata))
    target = _destination(mine, monkeypatch, tmp_path)
    source_db = mine.db_path if database else None
    if damage in {"retained", "retained-hash"}:
        assert reindex.reindex(mine.outputs_dir, database=source_db)["unresolved_tiled_refines"] == 0
        (target.outputs_dir / result.filename).unlink()
        with store._session() as conn:
            conn.execute("DELETE FROM images WHERE filename = ?", (result.filename,))
            if damage == "retained-hash":
                conn.execute("UPDATE images SET content_hash = ? WHERE filename = ?", ("0" * 16, root.filename))
    if damage == "retained":
        with closing(Image.new("RGB", tiled.ROOT_SIZE, "brown")) as image:
            image.save(target.outputs_dir / root.filename)

    before = {path.name: path.read_bytes() for path in target.outputs_dir.glob("*.png")}
    preview = reindex.reindex(mine.outputs_dir, database=source_db, dry_run=True)
    assert {path.name: path.read_bytes() for path in target.outputs_dir.glob("*.png")} == before
    assert preview == reindex.reindex(mine.outputs_dir, database=source_db)
    assert preview["unresolved_tiled_refines"] == int(damage != "none")
    assert preview["conflicts"] == int(damage == "retained")
    rows = {row.filename: row for row in store.list_images()[0]}
    actual = rows[result.filename].tiled_refine
    assert actual is not None
    assert actual == proof.model_copy(update={"source": rows[root.filename].id if damage == "none" else None})
    assert actual.source != root.id
    assert rows[result.filename].mask is None and rows[result.filename].refine is None
    assert png_metadata(target.outputs_dir / result.filename)["tiled_refine"] == actual.model_dump(mode="json")
    if damage != "none":
        with pytest.raises(ValueError, match="has not been resolved"):
            actual.request()
    else:
        repeat = reindex.reindex(mine.outputs_dir, database=source_db)
        assert (repeat["indexed"], repeat["present"], repeat["conflicts"]) == (0, 2, 0)
        assert store.get_image(rows[result.filename].id).tiled_refine == actual


@pytest.mark.parametrize("damaged_hash", [False, True])
def test_tiled_png_reindex_checks_the_provenance_of_a_root_only_in_the_destination(
    store, monkeypatch, tmp_path, damaged_hash,
):
    mine = store.settings
    root, proof = _save_root("root")
    result = _save_tiled(proof)
    target = _destination(mine, monkeypatch, tmp_path)
    assert reindex.reindex(mine.outputs_dir)["unresolved_tiled_refines"] == 0
    (target.outputs_dir / result.filename).unlink()
    (mine.outputs_dir / root.filename).unlink()
    with store._session() as conn:
        conn.execute("DELETE FROM images WHERE filename = ?", (result.filename,))
        if damaged_hash:
            conn.execute("UPDATE images SET content_hash = NULL WHERE filename = ?", (root.filename,))

    preview = reindex.reindex(mine.outputs_dir, dry_run=True)
    assert preview == reindex.reindex(mine.outputs_dir)
    assert preview["unresolved_tiled_refines"] == int(damaged_hash)
    rows = {row.filename: row for row in store.list_images()[0]}
    expected_source = None if damaged_hash else rows[root.filename].id
    assert rows[result.filename].tiled_refine == proof.model_copy(update={"source": expected_source})


def test_tiled_sql_png_intent_disagreement_refuses_before_mutation(store, monkeypatch, tmp_path):
    mine = store.settings
    _root, proof = _save_root("root")
    result = _save_tiled(proof)
    with store._session() as conn:
        conn.execute("UPDATE images SET tiled_refine = NULL WHERE id = ?", (result.id,))
    target = _destination(mine, monkeypatch, tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="recipes disagree"):
        reindex.reindex(mine.outputs_dir, database=mine.db_path)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert store.settings.db_path == target.db_path
