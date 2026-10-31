"""An outpaint's spec and the mask and preset answers.

`OutpaintSpec` as a row stores it and `Outpaint` as a request asks for it,
with their grid checks (`_on_grid`) and key spelling; `MaskStored`, a stored
mask's answer, and `OutpaintPreset(s)`, the presets route's. A leaf: no
settings, only `outpaint`'s constants. `schemas_generate` imports `Outpaint`
for the request, and `schemas.py` re-exports every model.
"""
from __future__ import annotations

import re
from typing import LiteralString

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from . import outpaint as outpaint_rules


_FILL_NAME = r"^[a-z0-9][a-z0-9-]{0,31}$"
_OUTPAINT_KEY = re.compile(r"^([a-z0-9][a-z0-9-]{0,31}):(\d{1,6}),(\d{1,6}),(\d{1,6}),(\d{1,6})@(\d{1,6})$")


def _on_grid(value: int, grid: int, what: LiteralString) -> int:
    """`value` where it is a multiple of `grid`, else the custom error the
    size uses, with the value and its two neighbours."""
    if value % grid:
        below = value - value % grid
        raise PydanticCustomError(
            # pydantic-core requires LiteralString here, but this identifier
            # deliberately includes an internal integer grid, never a template
            # from the request. Keep the existing machine-readable error code.
            f"multiple_of_{grid}",  # pyright: ignore[reportArgumentType]
            "{value} is not a multiple of {grid}: {below} or {above}" + what,
            {"value": value, "grid": grid, "below": below, "above": below + grid},
        )
    return value


class OutpaintSpec(BaseModel):
    """An outpaint as a row stores it (`ImageRecord.outpaint`, the
    `images.outpaint` column): the fill that made the new area, the padding
    on each side and the band of the original it blends over.

    Plain ints with grid checks only - no bound read from settings, so a row
    written under one `refine_max_side` still parses under another. The
    request's limits are `Outpaint`'s. Frozen, so a check cache can key on
    it. `key()` is its one spelling: the column, the resume key's mask
    element, `run_id` and the CLI's echo all read it."""

    model_config = ConfigDict(frozen=True)

    fill: str = Field(pattern=_FILL_NAME)
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(ge=0)
    bottom: int = Field(ge=0)
    band: int = Field(ge=0)

    @field_validator("left", "top", "right", "bottom")
    @classmethod
    def _sides_on_the_grid(cls, v: int) -> int:
        return _on_grid(v, outpaint_rules.GRID, "")

    @field_validator("band")
    @classmethod
    def _band_on_the_grid(cls, v: int) -> int:
        return _on_grid(v, outpaint_rules.BAND_GRID, ", the latent grid a band ends on")

    def key(self) -> str:
        """`fill:L,T,R,B@band`, e.g. `edge-box3-r16:400,0,400,0@32`."""
        return f"{self.fill}:{self.left},{self.top},{self.right},{self.bottom}@{self.band}"

    @classmethod
    def parse_key(cls, text: str) -> "OutpaintSpec":
        """`key()`'s inverse; ValueError for text that is no key."""
        match = _OUTPAINT_KEY.fullmatch(text) if isinstance(text, str) else None
        if match is None:
            raise ValueError(f"{str(text)[:60]!r} is not an outpaint key")
        fill, left, top, right, bottom, band = match.groups()
        return cls(fill=fill, left=int(left), top=int(top), right=int(right), bottom=int(bottom), band=int(band))

    def sides(self) -> str:
        """`+L/+T/+R/+B`, as the page and the CLI say the padding."""
        return outpaint_rules.sides(self)

    def wellformed(self) -> bool:
        """Whether a request could have made this spec: at least one side
        extended and the band within `MAX_BAND` - what `Outpaint` adds to
        the stored type's grid checks. Every way a spec reaches a row (a
        chunk, a merged database, a bundle) holds it to this, so a row's
        spec always runs again as an `Outpaint`."""
        return bool(self.left or self.top or self.right or self.bottom) and self.band <= outpaint_rules.MAX_BAND


class Outpaint(OutpaintSpec):
    """A request's outpaint: the stored spec with the fill defaulted to
    this build's, the band's default and constant bound, and at least one
    side extended. No side has a bound of its own: the canvas is the
    request's width x height, which `check_budget` holds to the refine
    guard."""

    fill: str = Field(default=outpaint_rules.FILL, pattern=_FILL_NAME)
    left: int = Field(default=0, ge=0)
    top: int = Field(default=0, ge=0)
    right: int = Field(default=0, ge=0)
    bottom: int = Field(default=0, ge=0)
    band: int = Field(default=outpaint_rules.BAND, ge=0, le=outpaint_rules.MAX_BAND)

    @model_validator(mode="after")
    def _extends_a_side(self) -> "Outpaint":
        if not (self.left or self.top or self.right or self.bottom):
            raise ValueError("an outpaint extends at least one side")
        return self


class MaskStored(BaseModel):
    """`POST /api/masks`' answer: the hash an inpaint's `mask` names, and the
    size it was stored at - its source's. The same picture answers the same
    hash, so an upload repeated after every stroke costs nothing new."""

    hash: str
    width: int
    height: int
    # Whether this upload wrote it, or the server already held it.
    created: bool


class OutpaintPreset(BaseModel):
    """One preset's padding over a source (`outpaint.presets`): its name
    ("+25%" or an aspect such as "16:9"), the words for it, the four
    sides, the canvas they make, and what holds it, or None."""

    name: str
    label: str
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    held: str | None = None


class OutpaintPresets(BaseModel):
    """`GET /api/images/{id}/outpaint-presets`' answer: the source's size,
    why no outpaint of it can run where it is off the 16-pixel grid, the
    refine guard's limits the canvas is held to, this build's fill, the
    band's default, its note and bound, the kept rectangle's least side, and every
    preset. Model-free: the server
    decides the numbers, the page holds no copy of the arithmetic."""

    id: int
    width: int
    height: int
    off_grid: str | None = None
    max_side: int
    max_pixels: int
    fill: str
    band: int
    # What the dialog says of `band` (`outpaint.band_note`): the server's
    # words, since which checkpoints the card ran is no copy the page holds.
    band_note: str
    max_band: int
    # The fewest pixels each way the band must leave kept (`outpaint.MIN_KEPT`).
    min_kept: int
    presets: list[OutpaintPreset]
