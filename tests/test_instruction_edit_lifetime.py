"""Unavailable edit results still retain honest source lifetime and estimates."""
from __future__ import annotations

from PIL import Image

from backend import gallery_delete, gallery_reports, grid, instruction_edit as edit
from backend import instruction_edit_admission, search
from backend.schemas import GenerateRequest


def _plant(store, *, proof=None, favorite=False):
    source = store.save_image(
        Image.new("RGB", (512, 512), "red"), GenerateRequest(prompt="source", width=512, height=512),
        seed=1, duration_ms=100, model_id="source/model",
    )
    spec, _ = instruction_edit_admission.bind_source(edit.InstructionEditSpec(source=source.id, recipe="kontext-v1"))
    if proof is not None:
        spec = spec.model_copy(update={"edit_source": spec.edit_source.model_copy(update=proof)})
    request = GenerateRequest(prompt="make it night", width=512, height=512, edit=spec)
    child = store.save_image(
        Image.new("RGB", (512, 512), "blue"), request, seed=2, duration_ms=28000,
        model_id=edit.KONTEXT_V1.model_id, batch=1,
    )
    if favorite:
        store.set_favorite(child.id, True)
    return source, child, request


def test_starred_edit_preserves_only_its_proven_source_during_scoped_cleanup(store):
    source, child, _ = _plant(store, favorite=True)
    assert gallery_delete.matching_preview(None)["sources"] == [source.id]
    assert gallery_delete.summarise()["sources_kept"] == 1
    assert gallery_delete.delete_matching(None, keep_favorites=True)["records"] == 0
    assert store.get_image(child.id).edit.source == source.id


def test_forged_source_digest_does_not_protect_an_unrelated_local_source(store):
    source, child, _ = _plant(store, proof={"hash": "ab" * 32}, favorite=True)
    assert store.get_image(child.id).edit_missing
    assert gallery_delete.matching_preview(None)["sources_kept"] == 0
    assert gallery_delete.delete_matching(None, keep_favorites=True)["records"] == 1
    assert store.get_image(source.id) is None
    assert store.get_image(child.id).edit is not None


def test_binning_restoring_and_purging_keeps_edit_intent_and_derives_availability(store):
    source, child, _ = _plant(store)
    gallery_delete.bin_images([source.id])
    held = store.get_image(child.id)
    assert held.edit_missing and held.edit_binned
    assert gallery_delete.bin_summary()["sources_in_use"] == [source.id]
    gallery_delete.restore_images([source.id])
    restored = store.get_image(child.id)
    assert not restored.edit_missing and not restored.edit_binned
    assert restored.edit == held.edit
    gallery_delete.delete_image(source.id)
    gone = store.get_image(child.id)
    assert gone.edit == held.edit and gone.edit_missing and not gone.edit_binned


def test_forged_edit_link_is_not_counted_as_a_binned_source_in_use(store):
    source, _, _ = _plant(store, proof={"hash": "cd" * 32})
    gallery_delete.bin_images([source.id])
    assert gallery_delete.bin_summary()["sources_in_use"] == []


def test_clear_keeps_live_starred_edit_and_proven_source(store):
    source, child, _ = _plant(store, favorite=True)
    assert gallery_delete.clear_history(keep_favorites=True)["records"] == 0
    assert store.get_image(source.id) is not None
    assert store.get_image(child.id).edit.source == source.id


def test_edit_history_never_prices_direct_or_refine_calls(store):
    _, _, request = _plant(store)
    assert gallery_reports.step_cost_ms(512, 512, edit.KONTEXT_V1.model_id) is None
    assert gallery_reports.refine_cost_ms(512, 512, edit.KONTEXT_V1.model_id) == {}
    prices, disk = gallery_reports.execution_estimates([grid.Cell(request=request, seed=2)], edit.KONTEXT_V1.model_id)
    assert prices[(512, 512)] == (None, {}, {}) and disk is None
    assert search.filter_clause("edit", "yes") == ("edit IS NOT NULL", [])
    assert gallery_reports.keep_rates(search="edit:yes", by=["edit"])[0]["edit"] == "kontext-v1"
