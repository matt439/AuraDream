"""More seeds: more images of each selected batch, its seeds continuing.

Pure, like `tag_swap` and `refine_batch`: the routes (`api_more_seeds.more_seeds_plan`)
resolve the selection, read every image of each batch and what the gallery
and the live jobs already hold, check what needs the backend, and queue.
Here the selection is grouped into batches, and each batch's next seeds, the
Batch size they run at and its cells are worked out.

A batch is the images of one run that agree on everything but the seed -
the resume key's elements (`settings_key`), the size, the checkpoint and
the quantization: one cell of a grid. Two picked images of one batch are
one batch: "4 more" is four more of it, not four of each picked image. A
record with no run id falls back to its job, and one with neither is a
batch of its own. A run scope (`MoreSeedsRequest.run`) is every batch of
the run, each its own "4 more".

The new seeds start one past the highest seed the gallery holds of the
batch - picked or not, earlier extensions included, since they join its run
- and step over any seed the gallery or a live job already has at these
settings, so nothing made is made again and two presses do not render the
same images. One generator per image, seeded `seed + i`, so consecutive
seeds are one cell of up to the batch's Batch size: the largest recorded on
its images (`ImageRecord.batch`; a call that ends a batch can hold fewer),
else the page's (`fallback`), held to the pixel budget. Not a second cell loop: each cell is one record through
`grid.cell_from_record` with its seed and `num_images` named, so
`grid.expand()` stays the one loop over axes.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from . import grid
from .config import settings
from .schemas import GenerateRequest, ImageRecord, error_detail
from .seeds import wrap_seed

# The statuses this module decides; the routes add `foreign`,
# `source_deleted`, `lora_missing`, `mask_missing` and `deleted`.
QUEUED = "queued"


def lineage(record: ImageRecord) -> tuple[str, str | int]:
    """What a batch belongs to: its run, else its job, else the image alone."""
    if record.run_id:
        return ("run", record.run_id)
    if record.job_id:
        return ("job", record.job_id)
    return ("image", record.id)


def settings_key(record: ImageRecord) -> tuple:
    """The resume key's elements but the seed, read off a record as
    `storage.completed_cells` reads them off a row: the prompt pair, steps,
    CFG, mu, true CFG, references, the adapter and its weight, the refine
    source and its strength, the mask (an outpaint by its spec, as
    `completed_cells` keys it), the upscaler, rounded alike."""
    return (
        record.prompt,
        record.negative_prompt or None,
        record.steps,
        round(record.guidance_scale, 6),
        None if record.mu is None else round(record.mu, 6),
        None if record.true_cfg is None else round(record.true_cfg, 6),
        grid.reference_key(record.reference),
        grid.lora_key(record.lora, record.lora_scale),
        grid.refine_key(record.refine, record.strength),
        grid.mask_key(record.mask, record.outpaint.key() if record.outpaint is not None else None),
        record.upscaler,
        grid.shape_key(record.width, record.height, record.crop_anchor),
    )


def batch_key(record: ImageRecord) -> tuple:
    """The batch `record` is one image of."""
    return (
        lineage(record), record.model_id, record.quantization, record.width, record.height,
        settings_key(record),
    )


@dataclass
class Batch:
    """One batch of the selection: its picked images, every image the
    gallery holds of it, and what More seeds makes of it."""

    key: tuple
    # The selection's images of it, oldest first.
    picked: list[ImageRecord]
    # Every image the gallery holds of it, the picked ones among them.
    mates: list[ImageRecord] = field(default_factory=list)
    status: str = QUEUED
    detail: str | None = None
    notes: list[str] = field(default_factory=list)
    # The Batch its new images run at, and whether that is the batch's own.
    size: int = 1
    recorded: bool = False
    seeds: list[int] = field(default_factory=list)
    cells: list[grid.Cell] = field(default_factory=list)

    @property
    def template(self) -> ImageRecord:
        """The record its cells are built from, and its row's picture: the
        newest picked image. Every image of a batch carries the same
        reproduction set but the seed, so any would do."""
        return self.picked[-1]

    @property
    def run_id(self) -> str | None:
        """The run its new images join, where it has one."""
        kind, name = self.key[0]
        return name if kind == "run" else None

    @property
    def held(self) -> list[ImageRecord]:
        """The images it has, by id: the gallery's, else the picked ones."""
        return sorted({r.id: r for r in [*self.mates, *self.picked]}.values(), key=lambda r: r.id)


def group(records: Sequence[ImageRecord]) -> list[Batch]:
    """The selection as batches, in the order of their oldest picked image."""
    batches: dict[tuple, Batch] = {}
    for record in records:
        key = batch_key(record)
        batches.setdefault(key, Batch(key=key, picked=[])).picked.append(record)
    return list(batches.values())


def batch_size(held: Sequence[ImageRecord], fallback: int) -> tuple[int, bool]:
    """The Batch a batch's new images run at, and whether it is the batch's
    own: the largest recorded on its images, else `fallback` (the page's).
    The largest, not the newest: each image records the call that made it,
    and the call that ends a batch - 5 more at Batch 2 is 2, 2 and 1 - holds
    fewer. A run has one Batch (`run_id` hashes it), so nothing else differs."""
    recorded = [r.batch for r in held if r.batch]
    return (max(recorded), True) if recorded else (fallback, False)


def fitting(size: int, width: int, height: int, refine: int | None) -> int:
    """The largest Batch up to `size` that `check_budget` lets through at
    this size. A refine pass has no batch budget: it is one call per image."""
    if refine is not None:
        return size
    return max(1, min(size, settings.max_batch_pixels // (width * height)))


def next_seeds(top: int, count: int, taken: Callable[[int], bool]) -> list[int]:
    """`count` seeds after `top`, in order, stepping over the taken ones.
    Ends: `taken` answers for a finite set, so at most that many are skipped."""
    seeds: list[int] = []
    seed = top
    while len(seeds) < count:
        seed = wrap_seed(seed + 1)
        if not taken(seed):
            seeds.append(seed)
    return seeds


def calls(seeds: Sequence[int], size: int) -> list[list[int]]:
    """`seeds` as pipeline calls: consecutive seeds together, at most `size`
    to a call, since a call draws `seed + i` for each of its images."""
    out: list[list[int]] = []
    for seed in seeds:
        if out and len(out[-1]) < size and seed == wrap_seed(out[-1][-1] + 1):
            out[-1].append(seed)
        else:
            out.append([seed])
    return out


def seed_label(seeds: Sequence[int]) -> str:
    return f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {seeds[0]}–{seeds[-1]}"


def plan(
    batch: Batch,
    *,
    count: int,
    fallback: int,
    applied: Callable[[GenerateRequest], GenerateRequest],
    held: set[grid.CellKey],
) -> None:
    """`count` more images of `batch`: its seeds, its Batch and its cells,
    or `invalid` where the record makes no valid request. `held` is every
    resume key the gallery and the live jobs have at the batch's size, as
    `applied` runs them; the seeds chosen are added to it, so a later batch
    of the same settings - another run of the same prompt - steps past them."""
    template = batch.template
    size, batch.recorded = batch_size(batch.held, fallback)
    fit = fitting(size, template.width, template.height, template.refine)
    if not batch.recorded:
        batch.notes.append(f"its Batch was not recorded: the panel's {size}")
    if fit < size:
        batch.notes.append(
            f"Batch {size} is past the pixel budget at {template.width}×{template.height}: {fit} at a time"
        )
    batch.size = fit
    try:
        probe = grid.cell_from_record(template)
    except ValueError as exc:
        batch.status, batch.detail = "invalid", error_detail(exc)
        return
    base = grid.cell_key(grid.Cell(request=applied(probe.request), seed=0))[:-1]
    top = max(r.seed for r in batch.held)
    seeds = next_seeds(top, count, lambda seed: (*base, seed) in held)
    try:
        cells = [
            grid.cell_from_record(template, seed=run[0], num_images=len(run), label=seed_label(run))
            for run in calls(seeds, fit)
        ]
        for cell in cells:
            cell.request.check_budget()
            cell.request.check_refine_steps()
    except ValueError as exc:
        batch.status, batch.detail = "invalid", error_detail(exc)
        return
    held.update((*base, seed) for seed in seeds)
    batch.seeds, batch.cells = seeds, cells


def count_default(batches: Sequence[Batch]) -> int | None:
    """The count a run surface opens More seeds at: the Batch the batches
    recorded (`batch_size`), where every one that recorded a Batch recorded
    the same; None where none did, or where they differ and no one count is
    the run's."""
    recorded = {size for size, own in (batch_size(batch.held, 1) for batch in batches) if own}
    return recorded.pop() if len(recorded) == 1 else None


def run_headline(queued: Sequence[Batch], count: int, favorites: bool) -> str | None:
    """A run scope's account of a press, written here so the page shows the
    server's words: `count` more per cell, the cells, the total and the new
    seeds' range, lowest first to highest last - a grid's cells continue
    from their own highest seeds, often the same ones. Starred is said to
    be ignored where the request had it ticked. None with nothing queued:
    the counts say why."""
    if not queued:
        return None
    images = sum(len(batch.seeds) for batch in queued)
    cells = len(queued)
    each = f"{count} more of each of {cells} cells" if cells > 1 else f"{count} more of its one cell"
    low = min(batch.seeds[0] for batch in queued)
    high = max(batch.seeds[-1] for batch in queued)
    seeds = f"seed {low}" if low == high else f"seeds {low}–{high}"
    starred = " · Starred ignored" if favorites else ""
    return f"{each} = {images} image{'' if images == 1 else 's'} · {seeds}{starred}"
