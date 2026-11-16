"""Portable, frozen face/hand region plans; no settings, detector or GPU imports.

The detail operation is a chain of ordinary inpaints. Its manifest explains
how masks were chosen; an individual pass keeps the ordinary inpaint key.
``box-feather-v1`` uses an integer inward ramp, independent of blur libraries.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from typing import Annotated, Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .instruction_edit import InstructionEditSource

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Kind = Literal["face", "hand"]
MAX_CANDIDATES = 128
MAX_REGIONS = 16
MAX_SOURCE_PIXELS = 8_411_136
MAX_MANIFEST_BYTES = 262_144


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def _copied_instances(cls, value: object) -> object:
        if isinstance(value, cls):
            return {**vars(value), **(value.__pydantic_extra__ or {})}
        return value


class AssetFacts(_Contract):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    revision: Annotated[str, Field(min_length=1, max_length=256)]
    sha256: Digest


class DetectorFacts(_Contract):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    runtime: Annotated[str, Field(min_length=1, max_length=128)]
    version: Annotated[str, Field(min_length=1, max_length=128)]
    asset_revision: Annotated[str, Field(min_length=1, max_length=256)]
    asset_sha256: Digest
    preprocessing: Annotated[str, Field(min_length=1, max_length=1024)]
    provider: Literal["CPU"] = "CPU"
    additional_assets: Annotated[list[AssetFacts], Field(max_length=8)] = []


class Region(_Contract):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    kind: Kind
    box: tuple[int, int, int, int]
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    detector: Annotated[str, Field(min_length=1, max_length=128)]

    @model_validator(mode="before")
    @classmethod
    def _json_box(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("box"), list):
            return {**value, "box": tuple(value["box"])}
        return value

    @model_validator(mode="after")
    def _nonempty(self) -> Region:
        x1, y1, x2, y2 = self.box
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("a detail region must be a nonempty nonnegative pixel box")
        return self


def region_order(region: Region) -> tuple[Any, ...]:
    x1, y1, x2, y2 = region.box
    return region.kind, y1, x1, y2, x2, region.detector, region.id


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class DetailerPlan(_Contract):
    """Stored limits are format bounds, never read from current settings.

    ``recipe`` is the resolved sampling request and loader facts at admission.
    Current request ceilings may become tighter without making this unreadable.
    Original source ids can be unresolved after a subset import; proof remains.
    """
    version: Literal["detailer-v1"] = "detailer-v1"
    source_id: Annotated[int, Field(gt=0, le=2**63 - 1)] | None
    source: InstructionEditSource
    detectors: Annotated[list[DetectorFacts], Field(min_length=1, max_length=16)]
    candidates: Annotated[list[Region], Field(min_length=1, max_length=512)]
    accepted: Annotated[list[str], Field(min_length=1, max_length=64)]
    masks: Annotated[list[Digest], Field(min_length=1, max_length=64)]
    padding: Annotated[int, Field(ge=0, le=4096)]
    feather: Annotated[int, Field(ge=0, le=4096)]
    threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    nms_iou: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mask_rule: Literal["box-feather-v1"] = "box-feather-v1"
    base_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    recipe: dict[str, Any]

    @model_validator(mode="after")
    def _consistent(self) -> DetailerPlan:
        if max(self.source.width, self.source.height) > 8192 or self.source.width * self.source.height > 16_777_216:
            raise ValueError("the stored detailer source exceeds the format's image bound")
        ids = [region.id for region in self.candidates]
        detector_ids = [detector.id for detector in self.detectors]
        if len(set(ids)) != len(ids) or len(set(detector_ids)) != len(detector_ids):
            raise ValueError("detailer region and detector ids must be unique")
        if len(self.accepted) != len(self.masks) or len(set(self.accepted)) != len(self.accepted):
            raise ValueError("detailer accepted regions and masks must correspond one to one")
        if not set(self.accepted) <= set(ids):
            raise ValueError("an accepted detail region is missing from the detection")
        if any(region.detector not in detector_ids for region in self.candidates):
            raise ValueError("a detail region names an unknown detector")
        if any(region.box[2] > self.source.width or region.box[3] > self.source.height for region in self.candidates):
            raise ValueError("a detail region lies outside its original source")
        accepted = self.accepted_regions
        if accepted != sorted(accepted, key=region_order):
            raise ValueError("accepted detail regions must follow their displayed canonical order")
        if len(_canonical(self.model_dump(mode="json")).encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ValueError("the detailer manifest is too large")
        return self

    @property
    def accepted_regions(self) -> list[Region]:
        by_id = {region.id: region for region in self.candidates}
        return [by_id[region_id] for region_id in self.accepted]

    @property
    def group_id(self) -> str:
        return operation_id(self)


class StoredDetailer(_Contract):
    plan: DetailerPlan
    pass_index: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _pass_exists(self) -> StoredDetailer:
        if self.pass_index >= len(self.plan.accepted):
            raise ValueError("the detailer pass index is outside its accepted regions")
        return self

    @property
    def expected_mask(self) -> str:
        return self.plan.masks[self.pass_index]

    @property
    def group_id(self) -> str:
        return self.plan.group_id


def stored_json(value: object | None) -> str | None:
    return None if value is None else _canonical(StoredDetailer.model_validate(value).model_dump(mode="json"))


def stored_from_json(value: str | None) -> StoredDetailer | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("stored detailer provenance must be bounded JSON text")
    return StoredDetailer.model_validate_json(value)


def operation_id(plan: DetailerPlan) -> str:
    value = plan.model_dump(mode="json", exclude={"source_id"})
    # Confidence is an observed result. Accepted geometry/masks already freeze
    # the decision; imported local ids and confidence summaries change no pixel.
    for region in value["candidates"]:
        region.pop("confidence", None)
    value["source"].pop("filename", None)
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:24]


def normalize_regions(
    raw: Sequence[Region | Mapping[str, Any]], width: int, height: int,
    *, threshold: float = 0.5, nms_iou: float = 0.5,
) -> list[Region]:
    """Floor left/top, ceil right/bottom, clip, same-kind greedy NMS.

    Malformed outputs are failures, not empty detections. NMS takes confidence
    descending with a total geometry/detector tie break, then display order.
    """
    if len(raw) > MAX_CANDIDATES:
        raise ValueError("the detector returned too many candidate regions")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("detailer source dimensions must be positive integers")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (threshold, nms_iou)):
        raise ValueError("detailer detection thresholds must be finite fractions")
    regions = []
    for candidate in raw:
        value = candidate.model_dump() if isinstance(candidate, Region) else dict(candidate)
        coordinates = value.get("box")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in coordinates
        ):
            raise ValueError("the detector returned an invalid pixel box")
        a, b, c, d = coordinates
        if a >= c or b >= d or c <= 0 or d <= 0 or a >= width or b >= height:
            raise ValueError("the detector returned an empty or off-image box")
        box = max(0, math.floor(a)), max(0, math.floor(b)), min(width, math.ceil(c)), min(height, math.ceil(d))
        value["box"] = box
        value.setdefault("id", hashlib.sha256(_canonical([value.get("kind"), value.get("detector"), box]).encode()).hexdigest()[:16])
        region = Region.model_validate(value)
        if region.confidence >= threshold:
            regions.append(region)
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: (-r.confidence, region_order(r))):
        if not any(region.kind == other.kind and overlap(region.box, other.box) > nms_iou for other in kept):
            kept.append(region)
    return sorted(kept, key=region_order)


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def mask_box(width: int, height: int, region: Region, padding: int) -> tuple[int, int, int, int]:
    """The exact clipped nonzero extent of a box-feather-v1 mask."""
    x1, y1, x2, y2 = region.box
    return max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding)


def region_mask(width: int, height: int, region: Region, padding: int, feather: int) -> Image.Image:
    """Frozen box-feather-v1: clip padded box, then integer inward ramp.

    Pillow draws integer rectangles only; no decoder/blur rounding is involved.
    A changed raster algorithm must receive another mask_rule name.
    """
    if any(type(v) is not int or v < 0 for v in (width, height, padding, feather)) or not width or not height:
        raise ValueError("detailer mask dimensions, padding and feather must be nonnegative integers")
    x1, y1, x2, y2 = region.box
    if x2 > width or y2 > height:
        raise ValueError("the detail region is outside its mask")
    box = mask_box(width, height, region, padding)
    picture = Image.new("L", (width, height))
    draw = ImageDraw.Draw(picture)
    a, b, c, d = box
    draw.rectangle((a, b, c - 1, d - 1), fill=255)
    ramp = min(feather, (c - a - 1) // 2, (d - b - 1) // 2)
    for inset in range(ramp):
        level = (255 * (inset + 1) + (ramp + 1) // 2) // (ramp + 1)
        draw.rectangle((a + inset, b + inset, c - inset - 1, d - inset - 1), outline=level)
    return picture


def rebuilt_masks(plan: DetailerPlan) -> dict[str, Image.Image]:
    """Rebuild and verify the full chain's masks, or refuse corrupt intent."""
    result: dict[str, Image.Image] = {}
    try:
        for digest, picture in iter_rebuilt_masks(plan):
            if digest in result:
                picture.close()
            else:
                result[digest] = picture
        return result
    except BaseException:
        for picture in result.values():
            picture.close()
        raise


def iter_rebuilt_masks(plan: DetailerPlan) -> Iterator[tuple[str, Image.Image]]:
    """Yield one verified full-size mask at a time; caller owns each picture."""
    from .record_fields import mask_hash

    for region, digest in zip(plan.accepted_regions, plan.masks, strict=True):
        picture = region_mask(plan.source.width, plan.source.height, region, plan.padding, plan.feather)
        if mask_hash(picture) != digest:
            picture.close()
            raise ValueError("the detailer geometry does not reproduce its recorded mask")
        yield digest, picture


def verify_masks(plan: DetailerPlan) -> None:
    """Prove geometry before admitting or saving provenance from any route.

    Cache only bounded immutable geometry JSON, never images or model owners.
    Recomputing the key notices mutations to nested lists in copied models.
    """
    encoded = _canonical({
        "width": plan.source.width, "height": plan.source.height,
        "boxes": [region.box for region in plan.accepted_regions],
        "padding": plan.padding, "feather": plan.feather, "masks": plan.masks,
        "rule": plan.mask_rule,
    })
    _verified_geometry(encoded)


@lru_cache(maxsize=64)
def _verified_geometry(encoded: str) -> None:
    from .record_fields import mask_hash

    geometry = json.loads(encoded)
    if geometry["rule"] != "box-feather-v1":
        raise ValueError("the detailer mask rule is unavailable")
    for box, digest in zip(geometry["boxes"], geometry["masks"], strict=True):
        region = Region(id="verify", kind="face", box=tuple(box), confidence=1.0, detector="verify")
        with region_mask(geometry["width"], geometry["height"], region,
                         geometry["padding"], geometry["feather"]) as picture:
            if mask_hash(picture) != digest:
                raise ValueError("the detailer geometry does not reproduce its recorded mask")
