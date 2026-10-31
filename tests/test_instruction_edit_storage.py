"""Durable edit intent and original-pixel proof without editor execution."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pytest
from PIL import Image

from backend import grid, png_settings
from backend.instruction_edit import edit_key
from backend.metadata import canonical_edit, edit_chunk, edit_from_json, edit_from_sources
from backend.schemas import GenerateRequest
from tests.bundle_helpers import png_metadata, save_source
from tests.instruction_edit_helpers import save_edit, stored_edit


def test_sql_png_completed_identity_and_source_availability(store):
    source = save_source()
    child = save_edit(source)
    spec = stored_edit(source)
    assert child.edit == spec and not child.edit_missing and not child.edit_binned
    assert store.get_image(child.id).edit == spec
    assert store.get_images([child.id])[0].edit == spec
    meta = png_metadata(store.settings.outputs_dir / child.filename)
    assert meta["edit"] == spec.model_dump(mode="json")
    with store._session() as conn:
        assert conn.execute("SELECT edit FROM images WHERE id = ?", (child.id,)).fetchone()[0] == edit_key(spec)
    request = GenerateRequest(prompt=child.prompt, width=512, height=512, steps=child.steps,
                              guidance_scale=child.guidance_scale, mu=None, edit=spec)
    assert grid.cell_key(grid.Cell(request, child.seed)) in store.completed_cells(
        [child.prompt], 512, 512, child.model_id, "fp8",
    )
    store.bin_images([source.id])
    back = store.get_image(child.id)
    assert back.edit == spec and back.edit_missing and back.edit_binned
    store.restore_images([source.id])
    assert not store.get_image(child.id).edit_missing
    Image.new("RGB", (256, 256), "red").save(store.settings.outputs_dir / source.filename)
    back = store.get_image(child.id)
    assert back.edit_missing and not back.edit_binned and back.edit == spec


def test_missing_source_keeps_child_and_proof(store):
    source = save_source()
    child = save_edit(source)
    store.delete_image(source.id)
    back = store.get_image(child.id)
    assert back.edit == child.edit and back.edit_missing
    assert store.list_images()[0][0].edit_missing


def test_source_sql_dimensions_are_part_of_availability(store):
    source = save_source()
    child = save_edit(source)
    with store._session() as conn:
        conn.execute("UPDATE images SET width = 128 WHERE id = ?", (source.id,))
    assert store.get_image(child.id).edit_missing


def test_legacy_migration_and_absence_do_not_invent_edit_metadata(store):
    ordinary = save_source()
    assert "edit" not in png_metadata(store.settings.outputs_dir / ordinary.filename)
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        conn.execute("ALTER TABLE images DROP COLUMN edit")
        conn.commit()
    assert store.get_image(ordinary.id).edit is None
    with store._session() as conn:
        column, = [row for row in conn.execute("PRAGMA table_info(images)") if row["name"] == "edit"]
        assert column["type"] == "TEXT" and column["notnull"] == 0
    assert edit_from_json(None) is None and edit_chunk(None) == {}


@pytest.mark.parametrize("bad", ["", "null", "{}", "[]", "false", '{"source":1,"recipe":"kontext-v1"}'])
def test_malformed_stored_edit_never_becomes_plain_generation(store, bad):
    source = save_source()
    with pytest.raises(ValueError):
        edit_from_json(bad)
    meta = png_metadata(store.settings.outputs_dir / source.filename)
    if bad != "null":  # PNG null is absence; SQL text "null" is not a stored object.
        with pytest.raises(ValueError):
            store.index_file("bad.png", {**meta, "edit": json.loads(bad) if bad else bad},
                             256, 256, datetime.now(timezone.utc))


@pytest.mark.parametrize("sql_present,png_present", [(True, False), (False, True)])
def test_sql_png_absence_is_a_conflict(store, sql_present, png_present):
    source = save_source()
    child = save_edit(source)
    with store._session() as conn:
        row = dict(conn.execute("SELECT * FROM images WHERE id = ?", (child.id,)).fetchone())
    meta = png_metadata(store.settings.outputs_dir / child.filename)
    if not sql_present:
        row.pop("edit")
    if not png_present:
        meta.pop("edit")
    with pytest.raises(ValueError, match="disagree"):
        edit_from_sources(row, meta)


def test_unresolved_descriptor_is_indexed_and_reexportable(store):
    source = save_source()
    child = save_edit(source)
    meta = png_metadata(store.settings.outputs_dir / child.filename)
    meta["edit"]["source"] = None
    recovered = store.index_file("recovered.png", meta, 512, 512, datetime.now(timezone.utc))
    assert recovered.edit.source is None and recovered.edit_missing
    assert recovered.edit.edit_source == child.edit.edit_source
    assert canonical_edit(meta["edit"]) == recovered.edit


@pytest.mark.parametrize("change", [
    {"model": "test/model"}, {"size": "256x256"}, {"negative_prompt": "bad"},
    {"refine": 1}, {"reference": [1]}, {"true_cfg": 2.0}, {"mu": 1.0},
    {"batch": 2}, {"batch": True}, {"lora": "adapter"}, {"strength": 0.5}, {"lora_scale": 0.5},
])
def test_conflicting_stored_pass_refuses_before_inserting(store, change):
    source = save_source()
    child = save_edit(source)
    meta = {**png_metadata(store.settings.outputs_dir / child.filename), **change}
    width, height = map(int, meta["size"].split("x"))
    with pytest.raises(ValueError, match="instruction"):
        store.index_file("bad.png", meta, width, height, datetime.now(timezone.utc))
    assert store.list_images()[1] == 2


@pytest.mark.parametrize("edit", [{"source": 1}, {"recipe": "future"}, False, []])
def test_png_settings_refuse_malformed_edit_before_a1111_fallback(edit):
    with pytest.raises(png_settings.PngSettingsError):
        png_settings.record({"auradream": json.dumps({"edit": edit}),
                             "parameters": "direct\nSteps: 1, Seed: 1, Size: 512x512"}, "edit.png")


def test_png_settings_hold_complete_edit_and_a1111_edit_marker(store):
    source = save_source()
    child = save_edit(source)
    with Image.open(store.settings.outputs_dir / child.filename) as image:
        with pytest.raises(png_settings.PngSettingsError, match="gallery image"):
            png_settings.record(image.info, child.filename)
        with pytest.raises(png_settings.PngSettingsError, match="gallery image"):
            png_settings.parse_parameters(image.info["parameters"])
