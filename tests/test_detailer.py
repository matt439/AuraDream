"""Frozen detector output normalization, review order and mask contract."""
from __future__ import annotations

import hashlib
import json

import pytest

from backend.detailer import (
    DetailerPlan, Region, StoredDetailer, iter_rebuilt_masks, normalize_regions,
    operation_id, region_mask, stored_from_json, stored_json,
)
from tests.bundle_helpers import save_source
from tests.detailer_helpers import plan_for


def test_nms_rounding_ties_and_display_order_do_not_depend_on_detector_iteration():
    candidates = [
        {"kind": "hand", "box": (50.2, 10.1, 90.1, 70.1), "confidence": 0.8, "detector": "hands"},
        {"kind": "face", "box": (-2.1, 7.9, 39.2, 61.1), "confidence": 0.9, "detector": "face"},
        {"kind": "face", "box": (0, 8, 40, 62), "confidence": 0.8, "detector": "face"},
        {"kind": "face", "box": (80, 90, 120, 140), "confidence": 0.4, "detector": "face"},
    ]
    result = normalize_regions(candidates, 128, 128)
    assert result == normalize_regions(list(reversed(candidates)), 128, 128)
    assert [r.kind for r in result] == ["face", "hand"]
    assert result[0].box == (0, 7, 40, 62)
    assert result[1].box == (50, 10, 91, 71)


@pytest.mark.parametrize("box", [(0, 0, float("nan"), 8), (3, 5, 2, 8), (300, 0, 400, 20), (True, 0, 8, 8)])
def test_invalid_detector_output_is_an_error_instead_of_empty_success(box):
    with pytest.raises(ValueError, match="detector"):
        normalize_regions([{"kind": "face", "box": box, "confidence": 0.9, "detector": "a"}], 128, 128)


def test_mask_keeps_outside_pixels_and_has_a_frozen_integer_feather():
    region = Region(id="face", kind="face", box=(8, 8, 24, 24), confidence=1.0, detector="a")
    with region_mask(32, 32, region, 2, 3) as mask:
        assert mask.getbbox() == (6, 6, 26, 26)
        assert [mask.getpixel((x, 6 + i)) for i, x in enumerate([6, 7, 8, 9])] == [64, 128, 191, 255]
        assert mask.getpixel((0, 0)) == 0
        assert hashlib.sha256(mask.tobytes()).hexdigest() == "9eb6cf50c0a71ccca21f0c2e1aa76e0434e14644392356f5bb363d13d1dad576"


def test_portable_manifest_json_round_trip_and_identity_ignore_local_id_and_scores(store):
    plan = plan_for(save_source())
    stored = StoredDetailer(plan=plan, pass_index=0)
    assert stored_from_json(stored_json(stored)) == stored
    assert StoredDetailer.model_validate(json.loads(stored_json(stored))) == stored
    changed = plan.model_copy(update={"source_id": None, "candidates": [r.model_copy(update={"confidence": 0.6}) for r in plan.candidates]})
    assert operation_id(changed) == plan.group_id
    assert operation_id(plan.model_copy(update={"accepted": plan.accepted[:1], "masks": plan.masks[:1]})) != plan.group_id
    assert operation_id(plan.model_copy(update={"base_seed": plan.base_seed + 1})) != plan.group_id


def test_manifest_rejects_reordered_regions_duplicate_ids_and_oversized_source(store):
    plan = plan_for(save_source())
    values = plan.model_dump()
    for change in (
        {"accepted": list(reversed(plan.accepted))},
        {"candidates": [plan.candidates[0], plan.candidates[0]]},
        {"source": plan.source.model_copy(update={"width": 100_000})},
    ):
        with pytest.raises(ValueError):
            DetailerPlan.model_validate({**values, **change})


def test_rebuild_proves_each_hash_and_rejects_tampered_geometry(store):
    plan = plan_for(save_source())
    rebuilt = list(iter_rebuilt_masks(plan))
    try:
        assert [digest for digest, _ in rebuilt] == plan.masks
    finally:
        for _, image in rebuilt:
            image.close()
    tampered = plan.model_copy(update={"padding": plan.padding + 1})
    with pytest.raises(ValueError, match="recorded mask"):
        list(iter_rebuilt_masks(tampered))
