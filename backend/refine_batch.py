"""Mass refine: what a refine pass of each of several gallery records would be.

Pure, like `tag_swap`: the routes (`api_refine_batch.refine_batch_plan`) resolve the
selection, check what needs the disk or the backend (the record's file, its
adapter, what the gallery already holds) and queue. Here each record becomes
one cell or a reason it cannot, decided from the record alone - never a
refusal of the whole selection over one image.

A cell is the record run again through `grid.cell_from_record` - its prompt
pair, seed, CFG, true CFG, mu, LoRA and variables - as a pass of the record
itself (`refine = record.id`) at the shared strength, the record's size times
the shared scale, and the record's steps unless the request names some. A
record another checkpoint made (only with `include_foreign`) keeps its
prompt pair and seed but takes the running checkpoint's steps, CFG and mu and
no LoRA, as the single Refine dialog substitutes them: a pass starts from
pixels, but those settings are the other schedule's. References never carry:
a pass's image is its source. Not a second cell loop: one record, one cell
per version, in the order given (oldest first), so `grid.expand()` stays the
one loop over axes.

Versions are the single Refine dialog's: seeds counting up from the
record's own through `wrap_seed`, one cell of one image each, so the
gallery's answer (`prune_done`) is per version. The first version is the
cell a press without versions makes, its key and run id unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from . import grid
from .config import settings
from .refine import scaled
from .schemas import ImageRecord, error_detail
from .seeds import wrap_seed

# The statuses this module decides; the routes add `file_missing`,
# `lora_missing`, `rendered` and `deleted`.
QUEUED = "queued"


@dataclass
class Planned:
    """One record's outcome: a cell to queue (status `queued`) or why not.

    `cell` is the first version, at the record's own seed; `more` the
    further versions' cells, seeds counting up (`plan`'s `versions`)."""

    record: ImageRecord
    status: str
    width: int
    height: int
    detail: str | None = None
    notes: list[str] = field(default_factory=list)
    cell: Any = None  # grid.Cell where queued
    foreign: bool = False
    more: list[Any] = field(default_factory=list)

    @property
    def cells(self) -> list[Any]:
        """Every version's cell, the first first; none where not queued."""
        return [] if self.cell is None else [self.cell, *self.more]

    def drop(self, status: str, detail: str | None) -> None:
        """Not queued after all, every version with it."""
        self.status, self.detail = status, detail
        self.cell, self.more = None, []


def target_size(record: ImageRecord, scale: float) -> tuple[int, int]:
    """The pass's size: the record's times `scale`, on the 16-pixel grid
    (`refine.scaled`, which the page's `refineScaled` matches)."""
    return scaled(record.width, record.height, scale)


def too_big(width: int, height: int) -> bool:
    """Past what a refine pass renders: config's `refine_max_side` or
    `refine_max_pixels`, the request's own bound and `check_budget`'s."""
    return (
        max(width, height) > settings.refine_max_side
        or width * height > settings.refine_max_pixels
    )


def plan(
    record: ImageRecord,
    *,
    running: str,
    profile: dict[str, Any],
    scale: float,
    strength: float,
    steps: int | None,
    include_foreign: bool,
    skip_passes: bool,
    versions: int = 1,
) -> Planned:
    """What Mass refine makes of `record` on the `running` checkpoint, whose
    `profile` (`model_defaults.defaults_for`) stands in for another
    checkpoint's steps, CFG and mu. In this order: a refine pass skipped,
    another checkpoint's image not asked for, a size past the guard, then the
    cell - `invalid` where it is no valid request, `no_step` where the
    strength leaves no step of its steps. `versions` cells where queued, at
    the record's seed and the seeds counting up from it: one more than the
    last, wrapped at the ceiling as a batch's are."""
    width, height = target_size(record, scale)
    out = Planned(record=record, status=QUEUED, width=width, height=height)
    foreign = record.model_id != running
    out.foreign = foreign
    if (record.refine is not None or record.edit is not None) and skip_passes:
        out.status = "pass"
        what = (
            "an instruction edit" if record.edit is not None else "an outpaint" if record.outpaint is not None
            else "an inpaint" if record.mask is not None else "a refine pass"
        )
        source = record.edit.source if record.edit is not None else record.refine
        original = f" of #{source}" if source is not None else " with an unresolved original source"
        out.detail = f"{what}{original}; untick Skip refine passes to refine it again"
        return out
    if foreign and not include_foreign:
        out.status = "foreign"
        out.detail = f"made on {record.model_id}; tick Include other checkpoints to refine it here"
        return out
    if too_big(width, height):
        out.status = "too_big"
        out.detail = (
            f"{record.width}×{record.height} at {scale:g}× is {width}×{height}, past the "
            f"{settings.refine_max_side}-pixel side and "
            f"{settings.refine_max_pixels / 1e6:.1f} MP a refine pass takes"
        )
        return out
    changes: dict[str, Any] = {
        # The edited result is the new ordinary Refine source; it is not
        # another instruction applied to that edit's original input.
        "edit": None,
        "refine": record.id,
        "strength": strength,
        "reference": [],
        # The record is the new source; an inpaint's mask was painted on the
        # old one's pixels, at the old size, and means nothing over these;
        # an outpaint's padding goes with it.
        "mask": None,
        "outpaint": None,
        # A new pass has no opt-in GAN choice; the source record may have
        # used one to prepare its own, different source.
        "upscaler": None,
        "crop_anchor": "center",
    }
    if foreign:
        changes.update(
            steps=steps or profile["steps"],
            guidance_scale=profile["guidance_scale"],
            true_cfg=None,
            mu=profile["mu"],
            lora=None,
            lora_scale=1.0,
        )
        out.notes.append(f"made on {record.model_id}: steps, CFG and mu are {running}'s")
        if record.lora:
            out.notes.append(f"LoRA {record.lora} not applied: it was for another checkpoint")
    elif steps is not None:
        changes["steps"] = steps
    try:
        cell = grid.cell_from_record(
            record, width=width, height=height, label=f"#{record.id}", **changes
        )
    except ValidationError as exc:
        out.status, out.detail = "invalid", error_detail(exc)
        return out
    try:
        cell.request.check_budget()
    except ValueError as exc:
        out.status, out.detail = "too_big", str(exc)
        return out
    try:
        cell.request.check_refine_steps()
    except ValueError:
        out.status = "no_step"
        out.detail = (
            f"strength {strength:g} of {cell.request.steps} steps leaves no step; "
            "set Steps or raise the strength"
        )
        return out
    out.cell = cell
    # The same pass at the next seeds: only the seed differs, so what the
    # first one passed the others pass.
    out.more = [
        grid.cell_from_record(
            record, width=width, height=height, label=f"#{record.id} v{n + 1}",
            seed=wrap_seed(record.seed + n), **changes,
        )
        for n in range(1, versions)
    ]
    return out
