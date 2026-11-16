"""Small frozen detail operations over throwaway gallery sources."""
from __future__ import annotations

from PIL import Image

from backend import storage
from backend.detailer import DetailerPlan, DetectorFacts, Region, StoredDetailer, region_mask
from backend.instruction_edit import descriptor_from_image
from backend.schemas import GenerateRequest


def plan_for(source, *, regions=2):
    with Image.open(storage.settings.outputs_dir / source.filename) as image:
        proof = descriptor_from_image(source.filename, image)
    candidates = [
        Region(id=f"region-{i}", kind="face", box=(16 + i * 64, 24, 56 + i * 64, 80),
               confidence=0.9, detector="test-face-v1")
        for i in range(regions)
    ]
    masks = []
    for region in candidates:
        with region_mask(source.width, source.height, region, 8, 4) as picture:
            masks.append(storage.store_mask(picture)[0])
    return DetailerPlan(
        source_id=source.id, source=proof,
        detectors=[DetectorFacts(id="test-face-v1", runtime="fixture", version="1",
                                 asset_revision="1", asset_sha256="a" * 64, preprocessing="RGB unchanged")],
        candidates=candidates, accepted=[r.id for r in candidates], masks=masks,
        padding=8, feather=4, threshold=0.5, nms_iou=0.5, base_seed=12,
        recipe={"prompt": source.prompt, "steps": 10, "guidance_scale": 5.0,
                "strength": 0.6, "model_id": "test/model", "quantization": "fp8"},
    )


def save_detail(source, plan=None, *, pass_index=0, direct_source=None):
    plan = plan or plan_for(source)
    request = GenerateRequest(
        prompt=source.prompt, width=source.width, height=source.height, steps=10, guidance_scale=5.0,
        refine=direct_source or source.id, strength=0.6, mask=plan.masks[pass_index],
        detailer=StoredDetailer(plan=plan, pass_index=pass_index),
    )
    return storage.save_image(
        Image.new("RGB", (source.width, source.height), (200, 10, pass_index)), request,
        seed=plan.base_seed + pass_index, duration_ms=1000, model_id="test/model",
        quantization="fp8", run_id=plan.group_id, batch=1,
    )
