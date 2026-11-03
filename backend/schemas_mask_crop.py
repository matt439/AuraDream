"""Only the masked area's spec (`mask_crop.py`, TODO.md INP#9).

`MaskCropSpec` is what a request asks for and a row stores: the frozen rule,
the padding around the mask and the working tier. `key()` is its one
spelling (`bbox16:32@1`), which the resume key's mask element and the column
read; `parse_key()` is its exact inverse. The box and working size the rule
decides are results (`mask_crop.region`), never part of the spec. A leaf:
no settings, only `mask_crop`'s constants.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import mask_crop as rules
from .schemas_outpaint import _on_grid

# The request's refusals, in the words the dialog and the CLI show.
MASK_CROP_NEEDS_A_MASK = "Only the masked area needs an inpaint or outpaint: a source and a mask."
MASK_CROP_WITH_EDIT = "An instruction edit has no masked-area crop."

_MASK_CROP_KEY = re.compile(r"^([a-z0-9][a-z0-9-]{0,31}):(\d{1,6})@(\d{1,6})$")


class MaskCropSpec(BaseModel):
    """Only the masked area, as a request asks for it and a row stores it.
    Frozen, so a check cache can key on it; unknown fields are refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: Literal["bbox16"] = rules.RULE
    pad: int = Field(default=rules.PAD, ge=0, le=rules.MAX_PAD)
    tier: Literal[1] = rules.TIER

    @field_validator("pad")
    @classmethod
    def _pad_on_the_grid(cls, v: int) -> int:
        return _on_grid(v, rules.PAD_GRID, "")

    def key(self) -> str:
        """`rule:pad@tier`, e.g. `bbox16:32@1`."""
        return f"{self.rule}:{self.pad}@{self.tier}"

    @classmethod
    def parse_key(cls, text: str) -> "MaskCropSpec":
        """`key()`'s inverse; ValueError for text that is no key, or names a
        rule, padding or tier this build refuses."""
        match = _MASK_CROP_KEY.fullmatch(text) if isinstance(text, str) else None
        if match is None:
            raise ValueError(f"{str(text)[:60]!r} is not a masked-area key")
        rule, pad, tier = match.groups()
        if str(int(pad)) != pad or str(int(tier)) != tier:
            raise ValueError(f"{text!r} is not a masked-area key: its numbers are not in their one spelling")
        try:
            return cls(rule=rule, pad=int(pad), tier=int(tier))  # pyright: ignore[reportArgumentType]
        except ValueError as error:
            raise ValueError(f"{text!r} is not a masked-area key this build runs: {error}") from None


class MaskCropArea(BaseModel):
    """What the grid preview says of a cropped pass's region
    (`mask_crop.region`): the box on the source or canvas, `(x1, y1, x2,
    y2)` with the right and bottom edges exclusive, the working size the
    class runs at, and how much larger the working size is than the box
    (`Crop.scale`: above 1 more detail, below 1 less memory). A result, as
    the row's: never sent back."""

    box: tuple[int, int, int, int]
    working: tuple[int, int]
    scale: float

    @classmethod
    def of(cls, crop: rules.Crop) -> "MaskCropArea":
        return cls(box=crop.box, working=crop.working, scale=round(crop.scale(), 3))
