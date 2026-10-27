"""Instruction-edit request/identity integration with synthetic source proof."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend import grid, more_seeds, refine_batch, then_refine
from backend.instruction_edit import InstructionEditSpec, StoredInstructionEdit, edit_key
from backend.schemas import GenerateRequest, GenerateSubmission, GridRequest, ImageRecord, ThenRefine


def stored(**changes):
    return StoredInstructionEdit.model_validate({
        "source": 17, "recipe": "kontext-v1",
        "edit_source": {"filename": "source.png", "hash": "ab" * 32, "width": 1024, "height": 1024},
        **changes,
    })


def request(**changes):
    return GenerateRequest(**{
        "prompt": "make it night", "width": 512, "height": 512,
        "edit": stored(), "seed": 7, **changes,
    })


def record(**changes):
    return ImageRecord(**{
        "id": 19, "filename": "result.png", "url": "/outputs/result.png",
        "prompt": "make it night", "negative_prompt": None, "width": 512, "height": 512,
        "steps": 28, "guidance_scale": 2.5, "mu": None, "seed": 7,
        "model_id": "black-forest-labs/FLUX.1-Kontext-dev", "duration_ms": 1,
        "created_at": "2026-10-27T00:00:00", "edit": stored(), "batch": 1,
        "run_id": "edit-run", **changes,
    })


def applied(req):
    return req.applied(negative_prompt_supported=False, cfg_threshold=1, instruction_edit_supported=True)


def test_edit_defaults_are_its_recipe_and_do_not_inherit_running_checkpoint():
    req = request()
    assert (req.steps, req.guidance_scale, req.mu, req.strength) == (28, 2.5, None, None)
    assert applied(req).edit == stored()
    assert applied(req).negative_prompt is None
    assert GenerateRequest.model_validate(req.model_dump()).edit == stored()


@pytest.mark.parametrize("changes", [
    {"reference": [1]}, {"lora": "a.safetensors"}, {"refine": 1}, {"strength": 0.5},
    {"mask": "ab" * 32}, {"outpaint": {"left": 32}}, {"upscaler": "realesrgan-x4plus-anime"},
    {"crop_anchor": "top"}, {"negative_prompt": "blurry"}, {"true_cfg": 1.0}, {"mu": 0.0},
    {"num_images": 2}, {"runs": 2},
])
def test_incompatible_edit_fields_refuse_instead_of_becoming_plain_generation(changes):
    with pytest.raises(ValueError):
        request(**changes)
    copied = request().model_copy(update=changes)
    with pytest.raises(ValueError):
        applied(copied)


@pytest.mark.parametrize("edit", [
    {"source": True, "recipe": "kontext-v1"}, {"source": "17", "recipe": "kontext-v1"},
    {"source": 17, "recipe": "future"}, {"source": 17, "recipe": "kontext-v1", "extra": 1},
])
def test_forged_edit_copy_never_reaches_a_key_or_normalization(edit):
    forged = request().model_copy(update={"edit": edit})
    with pytest.raises(ValueError):
        grid.cell_key(grid.Cell(forged, 7))
    with pytest.raises(ValueError):
        applied(forged)


def test_source_id_alone_has_no_durable_or_renderable_identity():
    req = request(edit={"source": 17, "recipe": "kontext-v1"})
    assert isinstance(req.edit, InstructionEditSpec)
    with pytest.raises(ValueError, match="Bind and verify"):
        grid.cell_key(grid.Cell(req, 7))
    with pytest.raises(ValueError, match="Bind and verify"):
        applied(req)
    with pytest.raises(ValueError, match="Bind and verify"):
        req.check_budget()
    with pytest.raises(ValueError):
        edit_key(req.edit)


def test_normalization_requires_an_explicit_edit_capability_and_preserves_the_pass_kind():
    with pytest.raises(ValueError, match="not available"):
        request().applied(negative_prompt_supported=False, cfg_threshold=1)
    assert applied(request()).edit == stored()


def test_bound_geometry_refuses_resize_or_unresolved_sources():
    with pytest.raises(ValueError, match="frozen source geometry"):
        request(width=1024, height=1024)
    with pytest.raises(ValueError, match="has not been resolved"):
        request(edit=stored(source=None))
    with pytest.raises(ValueError, match="frozen source geometry"):
        applied(request().model_copy(update={"height": 256}))


def test_edit_key_carries_complete_source_and_instruction_and_preserves_legacy_indices():
    req = request()
    key = grid.cell_key(grid.Cell(req, 7))
    assert key[8] is None and key[-2:] == ("512x512@center", 7)
    assert key[11] == edit_key(stored())
    variants = [req]
    for change in (
        {"source": 18},
        {"edit_source": stored().edit_source.model_copy(update={"hash": "cd" * 32})},
        {"edit_source": stored().edit_source.model_copy(update={"filename": "other.png"})},
        {"edit_source": stored().edit_source.model_copy(update={"width": 512, "height": 512})},
    ):
        variants.append(request(edit=stored(**change)))
    variants.append(request(prompt="make it snow"))
    assert len({grid.run_id([grid.Cell(item, 7)], "editor", "fp8") for item in variants}) == len(variants)


def test_replay_keeps_the_original_input_and_more_seeds_grouping_uses_its_proof():
    rec = record()
    replay = grid.cell_from_record(rec)
    assert replay.request.edit == rec.edit and replay.request.edit.source == 17
    assert replay.request.prompt == rec.prompt
    assert more_seeds.settings_key(rec) == grid.cell_key(replay)[:-1]
    different = record(edit=stored(source=18))
    assert more_seeds.batch_key(rec) != more_seeds.batch_key(different)
    unresolved = rec.model_copy(update={"edit": stored(source=None)})
    with pytest.raises(ValueError, match="has not been resolved"):
        grid.cell_from_record(unresolved)


def test_more_seeds_preserves_original_source_and_calls_one_image_at_a_time():
    # A legacy result with no recorded Batch uses the panel's fallback,
    # which must still become one call per image for this edit recipe.
    rec = record(batch=None)
    batch = more_seeds.group([rec])[0]
    more_seeds.plan(batch, count=3, fallback=4, applied=applied, held=set())
    assert batch.status == "queued" and batch.seeds == [8, 9, 10]
    assert len(batch.cells) == 3 and batch.size == 1
    assert all(cell.request.num_images == 1 and cell.request.edit == stored() for cell in batch.cells)


def test_more_seeds_reports_unavailable_normalization_as_a_batch_hold():
    batch = more_seeds.group([record()])[0]

    def unavailable(req):
        return req.applied(negative_prompt_supported=False, cfg_threshold=1)

    more_seeds.plan(batch, count=3, fallback=4, applied=unavailable, held=set())
    assert batch.status == "invalid" and "not available" in batch.detail
    assert batch.cells == []


def test_grids_and_then_refine_refuse_edits_at_schema_and_internal_copy_boundaries():
    with pytest.raises(ValidationError, match="cannot run as a grid"):
        GridRequest(**request().model_dump())
    with pytest.raises(ValueError, match="cannot run as a grid"):
        grid.expand(request())
    with pytest.raises(ValidationError, match="cannot use Then refine"):
        GenerateSubmission(**request().model_dump(), then_refine={})
    spec = ThenRefine(strength=0.3)
    assert "instruction edit" in then_refine.refusal([request()], spec, None)
    with pytest.raises(ValueError, match="instruction edit"):
        then_refine.stand_in(request(), spec)


def test_mass_refine_explicitly_changes_pass_kind_to_the_edited_result():
    rec = record()
    args = dict(running=rec.model_id, profile={}, scale=1, strength=0.3, steps=None, include_foreign=False)
    skipped = refine_batch.plan(rec, skip_passes=True, **args)
    assert skipped.status == "pass" and "instruction edit" in skipped.detail
    out = refine_batch.plan(rec, skip_passes=False, **args)
    assert out.status == "queued"
    assert out.cell.request.edit is None and out.cell.request.refine == rec.id
    assert out.cell.request.strength == 0.3
    # An ordinary child can be made even when the edit's old input is lost.
    unresolved = rec.model_copy(update={"edit": stored(source=None)})
    out = refine_batch.plan(unresolved, skip_passes=False, **args)
    assert out.status == "queued" and out.cell.request.edit is None
