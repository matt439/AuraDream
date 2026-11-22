"""Portable, versioned conditioning for direct image generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

from .controlnet_contract import ControlNetSpec
from .execution import canonical_execution
from .tiled_refine_contract import GUIDANCE_SCALE, MODEL_ID, QUANTIZATION, ROOT_SIZE, STEPS

REGIONAL_UNAVAILABLE = "Regional prompts are unavailable on this backend."
REGIONAL_LOADER_REQUIRED = (
    "Regional prompts require pinned Animagine XL 4.0, FP8/model offload, "
    "Windows RTX 5080 CUDA 0, Eager/Native and diffusers 0.42.1."
)

_Coordinate = Annotated[int, Field(strict=True, ge=0, le=100)]
_Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class PromptRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: _Prompt
    left: _Coordinate
    top: _Coordinate
    right: _Coordinate
    bottom: _Coordinate

    @model_validator(mode="after")
    def _has_area(self) -> PromptRegion:
        if self.right - self.left < 5 or self.bottom - self.top < 5:
            raise ValueError("A region must span at least 5% of the canvas in each direction.")
        return self


class RegionalPrompts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["regional_prompts"]
    method: Literal["sdxl-attention-v1"]
    regions: tuple[PromptRegion, ...] = Field(min_length=1, max_length=3)


SpatialConditioning = Annotated[RegionalPrompts | ControlNetSpec, Field(discriminator="kind")]
_CONDITIONING = TypeAdapter(SpatialConditioning)


def canonical_conditioning(value: object) -> SpatialConditioning | None:
    return None if value is None else _CONDITIONING.validate_python(value)


def conditioning_from_json(value: str | None) -> SpatialConditioning | None:
    return None if value is None else _CONDITIONING.validate_json(value)


def conditioning_json(value: SpatialConditioning | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def conditioning_key(value: SpatialConditioning | None) -> str | None:
    if isinstance(value, ControlNetSpec):
        return json.dumps(value.key(), ensure_ascii=False, separators=(",", ":"))
    return conditioning_json(value)


def conditioning_chunk(value: SpatialConditioning | None) -> dict[str, Any]:
    return {} if value is None else {"spatial_conditioning": value.model_dump(mode="json")}


def _validate_shape(value: SpatialConditioning, width: Any, height: Any, steps: Any, guidance: Any) -> None:
    if (width, height) != ROOT_SIZE or steps != STEPS or guidance != GUIDANCE_SCALE:
        name = "Regional prompts" if isinstance(value, RegionalPrompts) else "ControlNet"
        raise ValueError(f"{name} require a 1024×1024 canvas, 28 steps and CFG 5.")


def validate_request(request: Any) -> SpatialConditioning | None:
    value = canonical_conditioning(request.spatial_conditioning)
    if value is None:
        return None
    _validate_shape(value, request.width, request.height, request.steps, request.guidance_scale)
    if (
        request.num_images != 1 or request.reference or request.crop_anchor != "center"
        or canonical_execution(request.execution) is not None
        or any(getattr(request, key) is not None for key in (
            "refine", "mask", "outpaint", "upscaler", "mask_crop", "lora", "edit", "detailer", "tiled_refine", "mu",
        ))
        or request.true_cfg not in (None, GUIDANCE_SCALE if request.negative_prompt else None)
    ):
        raise ValueError("Spatial conditioning is for Batch 1 direct generation with standard execution.")
    return value


def validate_record(value: SpatialConditioning | None, record: Mapping[str, Any]) -> None:
    if value is None:
        return
    width, height = record.get("width"), record.get("height")
    if width is None and height is None and record.get("size") == "1024x1024":
        width, height = ROOT_SIZE
    _validate_shape(value, width, height, record.get("steps"), record.get("guidance_scale"))
    batch = record.get("batch")
    if (
        record.get("model_id", record.get("model")) != MODEL_ID or record.get("quantization") != QUANTIZATION
        or (batch is not None and (type(batch) is not int or batch != 1))
    ):
        raise ValueError("Spatial conditioning metadata requires Animagine XL 4.0 FP8 and Batch 1.")
    if (
        record.get("reference") not in (None, [], (), "[]")
        or record.get("crop_anchor") not in (None, "center")
        or record.get("execution") not in (None, {}, "{}", "null")
        or record.get("true_cfg") not in (None, GUIDANCE_SCALE if record.get("negative_prompt") else None)
        or any(record.get(key) is not None for key in (
            "refine", "mask", "outpaint", "upscaler", "mask_crop", "lora", "edit", "detailer", "tiled_refine", "mu",
        ))
    ):
        raise ValueError("Spatial conditioning metadata describes another workflow.")


def conditioning_from_sources(
    row: Mapping[str, Any], metadata: Mapping[str, Any] | None,
) -> SpatialConditioning | None:
    stored = conditioning_from_json(row.get("spatial_conditioning"))
    validate_record(stored, row)
    if metadata is None:
        return stored
    embedded = canonical_conditioning(metadata.get("spatial_conditioning"))
    validate_record(embedded, metadata)
    if stored != embedded:
        raise ValueError("spatial conditioning disagrees between the database and PNG")
    return stored
