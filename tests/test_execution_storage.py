"""Synthetic metadata/identity fixtures; no backend renders these recipes."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pytest
from PIL import Image

from backend import execution, grid
from backend.metadata import PNG_METADATA_KEY
from backend.schemas import GenerateRequest, ImageRecord, PromptVariant


RECIPES = [
    {"compilation": "regional-v1"}, {"attention_backend": "cudnn-v1"},
    {"step_cache": "fbc-005-v1"},
    {"compilation": "regional-v1", "attention_backend": "cudnn-v1", "step_cache": "fbc-005-v1"},
]
EMPTY = [None, {}, {"compilation": None}, {"attention_backend": None}, {"step_cache": None},
         {"compilation": None, "attention_backend": None, "step_cache": None}, execution.ExecutionSpec()]


def request(**changes):
    return GenerateRequest(**{
        "prompt": "a fox", "width": 256, "height": 256, "steps": 8,
        "guidance_scale": 2.0, "mu": None, "seed": 7, **changes,
    })


def carried(recipe):
    # This constructs metadata only. Normal schema/applied/queue/backend
    # admission continues to reject these specifications before rendering.
    return request().model_copy(update={"execution": execution.ExecutionSpec(**recipe)})


def save(store, req):
    return store.save_image(
        Image.new("RGB", (req.width, req.height), (11, 40, 90)), req,
        seed=7, duration_ms=10, model_id="m", quantization="fp8",
    )


def chunks(store, record):
    with Image.open(store.settings.outputs_dir / record.filename) as image:
        return {key: value for key, value in image.info.items() if isinstance(value, str)}


@pytest.mark.parametrize("value", EMPTY)
def test_empty_codecs_preserve_null_database_and_absent_png(value):
    assert execution.execution_key(value) is None
    assert execution.execution_chunk(value) == {}
    assert execution.execution_from_json(execution.execution_key(value)) is None


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_codecs_round_trip_canonically_without_granting_runtime(recipe):
    spec = execution.ExecutionSpec(**recipe)
    encoded = execution.execution_key(spec)
    assert encoded == json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    assert execution.execution_from_json(encoded) == spec
    assert execution.execution_chunk(spec) == {"execution": recipe}
    spaced = json.dumps({**dict(reversed(list(recipe.items()))), "step_cache": recipe.get("step_cache")})
    assert execution.execution_key(execution.execution_from_json(spaced)) == encoded
    with pytest.raises(ValueError, match="not available"):
        execution.require_default_execution(execution.execution_from_json(encoded))


@pytest.mark.parametrize("encoded", [
    "", "{", "[]", "true", "0", '"regional-v1"', '{"compilation":"future"}',
    '{"extra":null}', '{"attention_backend":false}', 3,
])
def test_stored_unknown_or_malformed_recipes_are_refused(encoded):
    with pytest.raises(ValueError):
        execution.execution_from_json(encoded)


def test_old_or_empty_json_has_the_same_canonical_absence():
    for value in (None, "null", "{}", '{"step_cache":null}'):
        assert execution.execution_from_json(value) is None


@pytest.mark.parametrize("recipe", RECIPES)
def test_synthetic_recipe_survives_row_png_index_and_completed_keys(store, recipe):
    req = carried(recipe)
    made = save(store, req)
    assert made.execution == store.get_image(made.id).execution == req.execution
    meta = json.loads(chunks(store, made)[PNG_METADATA_KEY])
    assert meta["execution"] == recipe
    with store._session() as conn:
        stored = conn.execute("SELECT execution FROM images WHERE id = ?", (made.id,)).fetchone()[0]
    assert stored == execution.execution_key(req.execution)
    recovered = store.index_file("recovered.png", meta, 256, 256, datetime.now(timezone.utc))
    assert recovered.execution == req.execution
    assert ImageRecord.model_validate(made.model_dump()).execution == req.execution
    assert grid.cell_key(grid.Cell(req, 7)) in store.completed_cells([req.prompt], 256, 256, "m", "fp8")
    assert grid.cell_key(grid.Cell(request(), 7)) not in store.completed_cells([req.prompt], 256, 256, "m", "fp8")
    # The record can be inspected and exported, but cannot become an Eager
    # rerun merely because this build has no implementation of its recipe.
    with pytest.raises(ValueError, match="not available"):
        grid.cell_from_record(made)


@pytest.mark.parametrize("value", EMPTY)
def test_empty_recipe_keeps_legacy_png_bytes_and_completed_identity(store, value):
    plain = save(store, request())
    empty = save(store, request(execution=value))
    assert empty.execution is None
    assert chunks(store, plain) == chunks(store, empty)
    assert "execution" not in json.loads(chunks(store, empty)[PNG_METADATA_KEY])
    key = grid.cell_key(grid.Cell(request(), 7))
    assert store.completed_cells(["a fox"], 256, 256, "m", "fp8") == {key}


def test_old_database_gains_nullable_execution_on_next_read_and_is_idempotent(store):
    made = save(store, request())
    with closing(sqlite3.connect(store.settings.db_path)) as conn:
        conn.execute("ALTER TABLE images DROP COLUMN execution")
        conn.commit()
    assert store.get_image(made.id).execution is None
    store.init_db()
    with store._session() as conn:
        columns = [row for row in conn.execute("PRAGMA table_info(images)") if row["name"] == "execution"]
        assert len(columns) == 1 and columns[0]["type"] == "TEXT" and columns[0]["notnull"] == 0
        assert conn.execute("SELECT execution FROM images").fetchone()[0] is None


def test_import_normalizes_valid_database_json_and_preserves_absent_legacy_rows(store):
    made = save(store, request())
    with store._session() as conn:
        source = dict(conn.execute("SELECT * FROM images WHERE id = ?", (made.id,)).fetchone())
    source.update(filename="carried.png", execution=' {"step_cache":null, "compilation":"regional-v1"} ')
    imported = store.import_file(source)
    assert imported.execution == execution.ExecutionSpec(compilation="regional-v1")
    with store._session() as conn:
        assert conn.execute("SELECT execution FROM images WHERE id = ?", (imported.id,)).fetchone()[0] == (
            '{"compilation":"regional-v1"}'
        )
    source.pop("execution")
    source["filename"] = "legacy.png"
    assert store.import_file(source).execution is None


@pytest.mark.parametrize("bad", [{"compilation": "future"}, {"unknown": None}, {"step_cache": False}])
def test_unknown_recipe_refuses_before_save_or_index_or_import_rows(store, bad):
    made = save(store, request())
    meta = json.loads(chunks(store, made)[PNG_METADATA_KEY])
    with store._session() as conn:
        source = dict(conn.execute("SELECT * FROM images WHERE id = ?", (made.id,)).fetchone())
    with pytest.raises(ValueError):
        save(store, request().model_copy(update={"execution": bad}))
    with pytest.raises(ValueError):
        store.index_file("invalid.png", {**meta, "execution": bad}, 256, 256, datetime.now(timezone.utc))
    with pytest.raises(ValueError):
        store.import_file({**source, "filename": "invalid.png", "execution": json.dumps(bad)})
    assert list(store.settings.outputs_dir.glob("*.png")) == [store.settings.outputs_dir / made.filename]
    with store._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_corrupted_database_execution_is_not_silently_read_as_eager(store):
    made = save(store, request())
    with store._session() as conn:
        conn.execute("UPDATE images SET execution = ? WHERE id = ?", ('{"compilation":"future"}', made.id))
    with pytest.raises(ValueError):
        store.get_image(made.id)
    with pytest.raises(ValueError):
        store.completed_cells(["a fox"], 256, 256, "m", "fp8")


def layouts():
    variants = [PromptVariant(label="original", prompt="a fox"), PromptVariant(label="yours", prompt="a wolf")]
    return {
        "uniform": grid.expand(request(), steps="4,8"),
        "noncenter": grid.expand(request(refine=3, strength=0.5, crop_anchor="top-left"), steps="4,8"),
        "mixed": grid.expand(request(), size="256x512,512x256", steps="4,8"),
        "mixed-noncenter": grid.expand(
            request(refine=3, strength=0.5, crop_anchor="top-left"), size="256x512,512x256", steps="4,8",
        ),
        "comparison": grid.expand(request(), steps="4,8", variants=variants),
        "mixed-comparison": grid.expand(request(), size="256x512,512x256", steps="4,8", variants=variants),
    }


# Captured from run_id before execution widened the key, not calculated by
# the implementation under test. Shape may be nonempty AFTER the new slot.
LEGACY_RUN_IDS = {
    "uniform": "b0f43ed489d3", "noncenter": "3ce689ccd499", "mixed": "58fe06d91156",
    "mixed-noncenter": "2273d1082568", "comparison": "c7a32c43ce94", "mixed-comparison": "5a9e86b135ce",
}


@pytest.mark.parametrize("empty", EMPTY)
def test_every_legacy_layout_omits_empty_execution_from_hashed_definition(empty):
    for name, cells in layouts().items():
        expected = LEGACY_RUN_IDS[name]
        assert grid.run_id(cells, "m", "fp8") == expected
        copied = [grid.Cell(cell.request.model_copy(update={"execution": empty}), cell.seed,
                            cell.label, cell.variables) for cell in cells]
        assert grid.run_id(copied, "m", "fp8") == grid.run_id(list(reversed(copied)), "m", "fp8") == expected
        for original, other in zip(cells, copied):
            assert grid.cell_key(original) == grid.cell_key(other)


def test_named_recipes_widen_only_before_shape_and_separate_every_layout():
    for name, cells in layouts().items():
        ids = {LEGACY_RUN_IDS[name]}
        for recipe in RECIPES:
            spec = execution.ExecutionSpec(**recipe)
            copied = [grid.Cell(cell.request.model_copy(update={"execution": spec}), cell.seed) for cell in cells]
            for original, other in zip(cells, copied):
                before, after = grid.cell_key(original), grid.cell_key(other)
                assert before[:12] == after[:12] and before[-2:] == after[-2:]
                assert before[12] is None and after[12] == execution.execution_key(spec)
                assert after[-2] == grid.shape_key(other.request.width, other.request.height, other.request.crop_anchor)
                assert [key[-1] for key in grid.cell_keys(other)] == [other.seed]
            named = grid.run_id(copied, "m", "fp8")
            assert named == grid.run_id(list(reversed(copied)), "m", "fp8")
            assert named not in ids
            ids.add(named)


def test_duplicate_and_prune_comparisons_keep_execution_in_the_key():
    cells = [grid.Cell(request(), 7), grid.Cell(request(), 7), *[grid.Cell(carried(recipe), 7) for recipe in RECIPES]]
    assert grid.duplicate_count(cells, lambda req: req) == 1
    assert grid.prune_done(cells, {grid.cell_key(cells[0])}) == cells[2:]
