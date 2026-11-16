"""A detailer's explanation survives storage without changing inpaint keys."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from backend import gallery_delete, grid, storage
from backend.detailer import stored_json
from backend.metadata import detailer_from_sources
from tests.bundle_helpers import png_metadata, save_source
from tests.detailer_helpers import plan_for, save_detail


def test_sql_png_and_individual_replay_keep_manifest_without_changing_pixel_key(store):
    source = save_source()
    plan = plan_for(source)
    child = save_detail(source, plan)
    assert store.get_image(child.id).detailer == child.detailer
    meta = png_metadata(store.settings.outputs_dir / child.filename)
    assert meta["detailer"] == child.detailer.model_dump(mode="json")
    with store._session() as conn:
        row = dict(conn.execute("SELECT * FROM images WHERE id=?", (child.id,)).fetchone())
        assert row["detailer"] == stored_json(child.detailer)
    assert detailer_from_sources(row, meta) == child.detailer
    cell = grid.cell_from_record(child)
    plain = grid.Cell(cell.request.model_copy(update={"detailer": None}), cell.seed)
    assert grid.cell_key(cell) == grid.cell_key(plain)
    assert grid.run_id([cell], child.model_id, "fp8") == grid.run_id(
        [plain], child.model_id, "fp8",
    )
    assert grid.cell_key(cell) in store.completed_cells([child.prompt], child.width, child.height, child.model_id, "fp8")
    new_seed = grid.cell_from_record(child, seed=55)
    assert new_seed.request.detailer == child.detailer and new_seed.seed == 55
    assert new_seed.request.detailer.plan.base_seed == plan.base_seed


def test_mass_refine_starts_new_ordinary_pass_and_unresolved_replay_refuses(store):
    source = save_source()
    child = save_detail(source)
    cell = grid.cell_from_record(child, refine=child.id, mask=None, strength=0.4)
    assert cell.request.detailer is None and cell.request.mask is None
    unresolved = child.model_copy(update={"refine": None, "mask": None, "strength": None})
    with pytest.raises(ValueError, match="resolved direct source"):
        grid.cell_from_record(unresolved)


def test_applied_cannot_silently_turn_detailer_into_text_to_image(store):
    cell = grid.cell_from_record(save_detail(save_source()))
    with pytest.raises(ValueError, match="cannot run"):
        cell.request.applied(negative_prompt_supported=True, cfg_threshold=1.0, mu_supported=False)
    assert cell.request.applied(negative_prompt_supported=True, cfg_threshold=1.0, mu_supported=False, inpaint_supported=True).detailer == cell.request.detailer


def test_forged_manifest_cannot_be_admitted_or_saved_through_ordinary_inpaint(store):
    cell = grid.cell_from_record(save_detail(save_source()))
    bad_plan = cell.request.detailer.plan.model_copy(update={"padding": 19})
    bad = cell.request.detailer.model_copy(update={"plan": bad_plan})
    forged = cell.request.model_copy(update={"detailer": bad})
    with pytest.raises(ValueError, match="recorded mask"):
        forged.validate_detailer()
    from PIL import Image
    with pytest.raises(ValueError, match="recorded mask"):
        store.save_image(Image.new("RGB", (256, 256)), forged,
                         seed=3, duration_ms=1, model_id="test/model")


def test_migration_legacy_absence_and_corrupt_provenance_refuse(store):
    source = save_source()
    assert "detailer" not in png_metadata(store.settings.outputs_dir / source.filename)
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        conn.execute("ALTER TABLE images DROP COLUMN detailer")
        conn.commit()
    assert store.get_image(source.id).detailer is None
    child = save_detail(source)
    meta = png_metadata(store.settings.outputs_dir / child.filename)
    with store._session() as conn:
        row = dict(conn.execute("SELECT * FROM images WHERE id=?", (child.id,)).fetchone())
    for broken in (None, {}, {**meta["detailer"], "pass_index": 90}):
        with pytest.raises(ValueError):
            detailer_from_sources(row, {**meta, "detailer": broken})
    with pytest.raises(ValueError):
        storage.import_file({**row, "filename": "bad.png", "detailer": json.dumps({})})


def test_partial_result_keeps_future_masks_through_permanent_sweep(store):
    source = save_source()
    plan = plan_for(source)
    child = save_detail(source, plan, pass_index=0)
    assert child.mask == plan.masks[0]
    assert gallery_delete.sweep_masks(settle=False) == 0
    assert all(storage.mask_path(digest).is_file() for digest in plan.masks)
    storage.delete_image(child.id)
    assert gallery_delete.sweep_masks(settle=False) == 2
    assert all(not (storage.masks_dir() / f"{digest}.png").exists() for digest in plan.masks)
