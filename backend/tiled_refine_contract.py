"""Portable intent for the private first Tiled Refine recipe."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .instruction_edit import InstructionEditSource, PositiveSource

RecipeName = Literal["private-provisional-sdxl-four-region-2048-v1"]
PreparationName = Literal["lanczos-2x-rgb-v1"]
RECIPE_NAME: RecipeName = "private-provisional-sdxl-four-region-2048-v1"
PREPARATION: PreparationName = "lanczos-2x-rgb-v1"
MODEL_ID = "cagliostrolab/animagine-xl-4.0"
REVISION = "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96"
QUANTIZATION = "fp8"
OFFLOAD = "model"
ROOT_SIZE = (1024, 1024)
CANVAS_SIZE = (2048, 2048)
STEPS = 28
GUIDANCE_SCALE = 5.0
STRENGTH = 0.35
HELD = "Tiled Refine is held until the private card qualification and public-flow checks pass."
UNRESOLVED_SOURCE = "The recorded Tiled Refine source has not been resolved to a gallery image."


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def _validate_copied_instances(cls, value: object) -> object:
        if isinstance(value, cls):
            return {**vars(value), **(value.__pydantic_extra__ or {})}
        return value


class TiledRefineSpec(_Contract):
    """Explicit recipe and original gallery id before binding the pixels."""

    source: PositiveSource
    recipe: RecipeName


class StoredTiledRefine(_Contract):
    """Both proofs survive a remap that leaves the root id unresolved."""

    source: PositiveSource | None
    recipe: RecipeName
    root_source: InstructionEditSource
    preparation: PreparationName
    prepared_rgb_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _root_size(self) -> StoredTiledRefine:
        if (self.root_source.width, self.root_source.height) != ROOT_SIZE:
            raise ValueError("Tiled Refine needs an original 1024x1024 RGB source.")
        return self

    def request(self) -> TiledRefineSpec:
        stored = StoredTiledRefine.model_validate(self)
        if stored.source is None:
            raise ValueError(UNRESOLVED_SOURCE)
        return TiledRefineSpec(source=stored.source, recipe=stored.recipe)


def bound_spec(value: object) -> StoredTiledRefine:
    return StoredTiledRefine.model_validate(value)


def stored_json(value: object) -> str:
    return json.dumps(bound_spec(value).model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stored_from_json(value: str) -> StoredTiledRefine:
    return StoredTiledRefine.model_validate_json(value)


def tiled_refine_key(value: object | None) -> str | None:
    return None if value is None else stored_json(value)


def rgb_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _require_root(image: Image.Image) -> None:
    if image.mode != "RGB" or image.size != ROOT_SIZE:
        raise ValueError("Tiled Refine needs an original 1024x1024 RGB source.")


def descriptor_from_image(filename: str, image: Image.Image) -> InstructionEditSource:
    """Prove the original RGB pixels without an EXIF transform or mode conversion."""
    _require_root(image)
    return InstructionEditSource(filename=filename, hash=rgb_sha256(image), width=image.width, height=image.height)


def prepare_canvas(image: Image.Image) -> Image.Image:
    """Return the exact 2x Lanczos preparation from the exploratory recipe."""
    _require_root(image)
    return image.resize(CANVAS_SIZE, resample=Image.Resampling.LANCZOS)
