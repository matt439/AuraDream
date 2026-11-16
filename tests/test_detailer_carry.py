"""Detailer provenance survives portable remaps without granting a chain replay."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from backend import bundle_images, bundles, png_settings, reindex, storage
from backend.bundle_common import BundleError
from backend.detailer import DetailerPlan, DetectorFacts, Region, StoredDetailer, region_mask
from backend.instruction_edit import descriptor_from_image
from backend.schemas import GenerateRequest
from tests.bundle_helpers import bundle_snapshot, png_metadata, save_source, use_destination


def chain(store):
    source = save_source("original", seed=42)
    with Image.open(store.settings.outputs_dir / source.filename) as picture:
        proof = descriptor_from_image(source.filename, picture)
    detector = DetectorFacts(
        id="owned-fixture", runtime="fixture", version="1", asset_revision="1", asset_sha256="1" * 64,
        preprocessing="fixture RGB pixels",
    )
    regions = [
        Region(id="face", kind="face", box=(24, 24, 72, 72), confidence=0.9, detector=detector.id),
        Region(id="hand", kind="hand", box=(120, 120, 180, 180), confidence=0.8, detector=detector.id),
    ]
    masks = []
    for region in regions:
        with region_mask(256, 256, region, 8, 4) as mask:
            masks.append(store.store_mask(mask)[0])
    plan = DetailerPlan(
        source_id=source.id, source=proof, detectors=[detector], candidates=regions,
        accepted=[region.id for region in regions], masks=masks, padding=8, feather=4,
        threshold=0.5, nms_iou=0.5, base_seed=100,
        recipe={"prompt": "original", "steps": 20, "strength": 0.5, "model_id": "test/model"},
    )
    previous = source
    children = []
    for index, digest in enumerate(masks):
        child = store.save_image(
            Image.new("RGB", (256, 256), (100 + index, 40, 80)),
            GenerateRequest(
                prompt="original", width=256, height=256, steps=20, refine=previous.id,
                mask=digest, strength=0.5, detailer=StoredDetailer(plan=plan, pass_index=index),
            ),
            seed=100 + index, duration_ms=123, model_id="test/model", run_id=plan.group_id,
        )
        children.append(child)
        previous = child
    return source, children


def destination(store, monkeypatch, tmp_path):
    settings = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", settings)
    return settings


def rewrite_detailer(store, record, mutate):
    path = store.settings.outputs_dir / record.filename
    meta = png_metadata(path)
    mutate(meta["detailer"])
    with store._session() as conn:
        conn.execute("UPDATE images SET detailer = ? WHERE id = ?", (json.dumps(meta["detailer"]), record.id))
    bundle_images._rewrite_metadata(path, meta)


def test_whole_bundle_remaps_full_original_proof_and_direct_chain(store, monkeypatch, tmp_path):
    source, children = chain(store)
    archive = Path(bundles.export_bundle(keep_gallery=True)["path"])
    target = destination(store, monkeypatch, tmp_path)
    save_source("occupy id", seed=1)
    result = bundles.import_bundle(archive)
    assert result["imported"] == 3 and result["unresolved_references"] == 0
    imported = {record.filename: record for record in store.list_images()[0]}
    first, last = (imported[child.filename] for child in children)
    original = imported[source.filename]
    assert original.id != source.id
    assert first.refine == original.id and last.refine == first.id
    assert first.detailer.plan.source_id == last.detailer.plan.source_id == original.id
    assert last.detailer.plan.source == children[-1].detailer.plan.source
    assert last.detailer.group_id == children[-1].detailer.group_id
    assert png_metadata(target.outputs_dir / last.filename)["detailer"] == last.detailer.model_dump(mode="json")
    before = bundle_snapshot(store)
    assert bundles.import_bundle(archive)["present"] == 3
    assert bundle_snapshot(store) == before


def test_child_only_bundle_keeps_unresolved_proof_and_rebuilds_every_mask(store, monkeypatch, tmp_path):
    _source, children = chain(store)
    last = children[-1]
    for path in (store.settings.outputs_dir / storage.MASKS_DIR).glob("*.png"):
        path.unlink()
    archive = Path(bundles.export_bundle(keep_gallery=True, ids=[last.id])["path"])
    target = destination(store, monkeypatch, tmp_path)
    save_source("numeric collision", seed=7)
    result = bundles.import_bundle(archive)
    imported = next(record for record in store.list_images()[0] if record.filename == last.filename)
    assert result["imported"] == 1 and imported.refine is None and imported.mask is None
    assert imported.detailer.plan.source_id is None
    assert imported.detailer.plan.source == last.detailer.plan.source
    assert imported.detailer.group_id == last.detailer.group_id
    for digest in last.detailer.plan.masks:
        assert storage.mask_path(digest).is_file()
    assert png_metadata(target.outputs_dir / last.filename)["detailer"] == imported.detailer.model_dump(mode="json")


@pytest.mark.parametrize("database", [False, True])
def test_reindex_remaps_original_and_preserves_manifest_with_missing_mask_files(store, monkeypatch, tmp_path, database):
    source, children = chain(store)
    origin = store.settings
    for path in (origin.outputs_dir / storage.MASKS_DIR).glob("*.png"):
        path.unlink()
    target = destination(store, monkeypatch, tmp_path)
    save_source("unrelated", seed=9)
    counts = reindex.reindex(origin.outputs_dir, database=origin.db_path if database else None)
    records = {record.filename: record for record in store.list_images()[0]}
    last = records[children[-1].filename]
    assert counts["indexed"] == 3 and counts["unresolved_detailers"] == 0
    assert counts["detailer_masks_rebuilt"] == 2
    assert last.detailer.plan.source_id == records[source.filename].id != source.id
    assert last.refine == records[children[0].filename].id
    assert last.detailer.plan.source == children[-1].detailer.plan.source
    assert png_metadata(target.outputs_dir / last.filename)["detailer"] == last.detailer.model_dump(mode="json")
    assert reindex.reindex(origin.outputs_dir, database=origin.db_path if database else None)["present"] == 3


def test_png_only_child_recovery_retains_intent_without_matching_source(store, monkeypatch, tmp_path):
    _source, children = chain(store)
    last = children[-1]
    only_child = tmp_path / "only-child"
    only_child.mkdir()
    shutil.copy2(store.settings.outputs_dir / last.filename, only_child / last.filename)
    destination(store, monkeypatch, tmp_path)
    save_source("numeric collision", seed=9)
    counts = reindex.reindex(only_child)
    imported = next(record for record in store.list_images()[0] if record.filename == last.filename)
    assert counts["indexed"] == 1 and counts["unresolved_detailers"] == 1
    assert imported.detailer.plan.source_id is None and imported.refine is None and imported.mask is None
    assert imported.detailer.plan.source == last.detailer.plan.source
    assert all(storage.mask_path(digest).is_file() for digest in last.detailer.plan.masks)


@pytest.mark.parametrize("database", [False, True])
def test_bad_frozen_mask_refuses_recovery_before_destination_changes(store, monkeypatch, tmp_path, database):
    _source, children = chain(store)
    rewrite_detailer(store, children[-1], lambda stored: stored["plan"]["masks"].__setitem__(0, "0" * 64))
    origin = store.settings
    destination(store, monkeypatch, tmp_path)
    save_source("retained", seed=9)
    before = bundle_snapshot(store)
    with pytest.raises(ValueError, match="geometry"):
        reindex.reindex(origin.outputs_dir, database=origin.db_path if database else None)
    assert bundle_snapshot(store) == before


def test_bad_frozen_mask_refuses_bundle_before_cleanup(store):
    _source, children = chain(store)
    rewrite_detailer(store, children[-1], lambda stored: stored["plan"]["masks"].__setitem__(0, "0" * 64))
    before = bundle_snapshot(store)
    with pytest.raises(BundleError, match="geometry"):
        bundles.export_bundle(expected=3)
    assert bundle_snapshot(store) == before


def test_png_manifest_disagreement_refuses_repeated_import_fast_path(store, monkeypatch, tmp_path):
    _source, children = chain(store)
    archive = Path(bundles.export_bundle(keep_gallery=True)["path"])
    target = destination(store, monkeypatch, tmp_path)
    bundles.import_bundle(archive)
    imported = next(record for record in store.list_images()[0] if record.filename == children[-1].filename)
    meta = png_metadata(target.outputs_dir / imported.filename)
    meta.pop("detailer")
    bundle_images._rewrite_metadata(target.outputs_dir / imported.filename, meta)
    before = bundle_snapshot(store)
    with pytest.raises(BundleError, match="detailer manifests disagree"):
        bundles.import_bundle(archive)
    assert bundle_snapshot(store) == before


def test_png_sampling_read_validates_manifest_without_importing_image_links(store):
    _source, children = chain(store)
    meta = png_metadata(store.settings.outputs_dir / children[-1].filename)
    back = png_settings.record({"auradream": json.dumps(meta)}, "owned.png")
    assert back["detailer"] == meta["detailer"]
    meta["detailer"]["pass_index"] = 999
    with pytest.raises(png_settings.PngSettingsError, match="pass index"):
        png_settings.record({"auradream": json.dumps(meta)}, "owned.png")


def test_in_place_reindex_rewrites_original_source_id_and_unchanged_operation(store, monkeypatch):
    removed = save_source("removed", seed=9)
    source, children = chain(store)
    original_group = children[-1].detailer.group_id
    monkeypatch.setattr(reindex, "settings", store.settings)
    (store.settings.outputs_dir / removed.filename).unlink()
    store.settings.db_path.unlink()
    counts = reindex.reindex(store.settings.outputs_dir)
    records = {record.filename: record for record in store.list_images()[0]}
    last = records[children[-1].filename]
    assert counts["unresolved_detailers"] == 0
    assert last.detailer.plan.source_id == records[source.filename].id != source.id
    assert last.detailer.group_id == original_group


@pytest.mark.parametrize("broken", ["pixels", "unresolved"])
def test_unproven_original_never_maps_to_a_colliding_id(store, monkeypatch, tmp_path, broken):
    source, children = chain(store)
    origin = store.settings
    if broken == "pixels":
        Image.new("RGB", (256, 256), "red").save(origin.outputs_dir / source.filename)
    else:
        for child in children:
            rewrite_detailer(store, child, lambda stored: stored["plan"].__setitem__("source_id", None))
    destination(store, monkeypatch, tmp_path)
    save_source("numeric collision", seed=7)
    counts = reindex.reindex(origin.outputs_dir)
    details = [record.detailer for record in store.list_images()[0] if record.detailer is not None]
    assert len(details) == 2 and counts["unresolved_detailers"] == 2
    assert all(detail.plan.source_id is None for detail in details)
    assert all(detail.plan.source == children[-1].detailer.plan.source for detail in details)
