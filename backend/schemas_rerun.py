"""Gallery records run again as new jobs - Re-render, Swap tags, Mass
refine and More seeds: each one's request, per-image status, result row,
preview and accepted answer.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import sizes
from .config import settings
from .seeds import MAX_BATCH
from .schemas_generate import GenerateRequest, JobAccepted

class RerenderRequest(BaseModel):
    """Render a gallery selection again at another size.

    The selection is the gallery's picked cards (`ids`) where the page has
    picked some, else its own filter - the search box and the Favorites
    tick - taken server-side, because the browser holds only the page it
    has scrolled to. Every record carries the complete reproduction
    set, so each becomes one cell of one job: the prompt, the negative prompt,
    steps, CFG, true CFG, mu, the references, the adapter and its weight, the
    refine source and its strength, the seed and the variables - what the
    placeholders were set to, or which version of a comparison it is - carry
    over and only the size changes. What
    that delivers is honest to state: the pipeline draws its latents at the
    requested shape, so the same seed at 4 MP is a new sample - the prompt and
    the settings carry, the picture does not. For a prompt-variable grid that
    is the whole value; for a seeds-only grid the selection is thrown away.
    """

    # The picked cards, exactly those whatever the filter; a picked image
    # deleted or binned since is left out and named in the answer
    # (`RerenderAccepted.deleted`), as Swap tags reports it.
    ids: list[int] | None = Field(default=None, min_length=1, max_length=settings.max_grid_images)
    search: str | None = Field(default=None, max_length=2000)
    favorites: bool = True
    # The gallery's third filter: off, the selection is this checkpoint's
    # images alone, as the grid shows them; on, another checkpoint's records
    # are selected too and refused below, since their steps and CFG are not
    # this checkpoint's. Picked cards are not scoped by it: another
    # checkpoint's among them are refused all the same.
    all_models: bool = False
    # One size for every record, or `tier`: each picked record at its own
    # aspect's entry at that megapixel tier (`sizes.nearest_preset`), one job
    # per size. Exactly one of the two; a request without `tier` is the
    # one-size Re-render it always was, with the same run ids.
    width: int | None = Field(default=None, ge=256, le=settings.max_side)
    height: int | None = Field(default=None, ge=256, le=settings.max_side)
    tier: float | None = None
    # The ids the preview resolved (`RerenderPreview.ids`): a selection that
    # changed since it is refused rather than queued unseen.
    expected: list[int] | None = Field(default=None, max_length=settings.max_grid_images)

    # Written out: a validator assigned from another model is not run.
    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, value: int | None) -> int | None:
        return value if value is None else GenerateRequest._multiple_of_16.__func__(cls, value)

    @field_validator("ids")
    @classmethod
    def _positive_ids(cls, value: list[int] | None) -> list[int] | None:
        return SwapTagsRequest._positive_ids.__func__(cls, value)

    @model_validator(mode="after")
    def _one_size(self) -> RerenderRequest:
        if self.tier is None:
            if self.width is None or self.height is None:
                raise ValueError("a re-render needs a width and a height, or a tier")
            return self
        if self.width is not None or self.height is not None:
            raise ValueError("a re-render takes a tier or a width and a height, not both")
        if self.tier not in sizes.TIERS:
            raise ValueError(
                f"{self.tier:g} MP is not a tier; the tiers are {', '.join(f'{t:g}' for t in sizes.TIERS)} MP"
            )
        # The page's picked cards only: a filter of mixed aspects is a
        # selection nobody has looked at size by size.
        if self.ids is None:
            raise ValueError("a re-render at a tier takes picked images (ids), not a filter")
        return self


class RerenderGone(BaseModel):
    """A picked image Re-render left out: deleted, or binned since it was
    picked (`detail`, `submission.gone_detail`'s words)."""

    id: int
    detail: str


class RerenderUnmatched(BaseModel):
    """A picked image a tier re-render left out: its size is no preset's
    aspect (`sizes.nearest_preset`), and none is guessed for it."""

    id: int
    width: int
    height: int


class RerenderSize(BaseModel):
    """One size of a tier re-render's plan: one job's worth."""

    width: int
    height: int
    # The images mapped to it, and of them those it would queue now.
    images: int
    queued: int


class RerenderPreview(BaseModel):
    """What a Re-render would queue, size by size, without queueing it."""

    # Every image the selection resolved to, oldest first, unmatched ones
    # included: what the queueing request sends back as `expected`.
    ids: list[int]
    sizes: list[RerenderSize]
    unmatched: list[RerenderUnmatched] = Field(default_factory=list)
    deleted: list[RerenderGone] = Field(default_factory=list)
    queued: int
    # How many jobs it makes: one per size with anything to queue.
    jobs: int
    pruned: int = 0
    planned: int = 0


class RerenderAccepted(JobAccepted):
    """Re-render's job, and the picked images it could not take. A tier
    re-render makes one job per size: the first at the top level, as every
    other submission answers, the rest in `more`."""

    deleted: list[RerenderGone] = Field(default_factory=list)
    # Held by the gallery and by live jobs respectively, never combined.
    pruned: int = 0
    planned: int = 0
    more: list[JobAccepted] = Field(default_factory=list)
    unmatched: list[RerenderUnmatched] = Field(default_factory=list)


class RunAction(BaseModel):
    """A run action's availability, including the reason a press is held."""

    available: bool
    reason: str = ""
    count: int = 0


class FinishRunRequest(BaseModel):
    """Pin the missing cells the user previewed, without a composer payload."""

    expected: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")


class FinishRunPreview(BaseModel):
    run_id: str
    available: bool
    reason: str = ""
    total: int = 0
    cells: int = 0
    images: int = 0
    rendered: int = 0
    queued: int = 0
    # Previously completed cells now missing, redrawn by this submission.
    redrawn: int = 0
    budget_minutes: float | None = None
    pin: str


class RunDetails(BaseModel):
    run_id: str
    model_id: str
    quantization: str | None = None
    width: int
    height: int
    sizes: list[tuple[int, int]] = Field(default_factory=list)
    size_count: int = 1
    batch: int
    comparison: bool = False
    images: int = 0
    binned: int = 0
    cells: int = 0
    actions: dict[str, RunAction]


class FinishRunAccepted(JobAccepted):
    cells: int
    images: int
    pruned: int = 0
    planned: int = 0
    redrawn: int = 0
    budget_minutes: float | None = None


class TagSwap(BaseModel):
    """One row of Swap tags: the comma items to find, what goes in their place."""

    find: str = Field(min_length=1, max_length=1000)
    # Empty removes what was found.
    replace: str = Field(default="", max_length=1000)


class SwapTagsRequest(BaseModel):
    """Run images again with some of their tags swapped, each keeping its own
    seed, size and settings.

    The images are the gallery selection's `ids` where the page has picked
    cards, else the gallery's filter as Re-render takes it (search, Starred,
    the checkpoint scope). `field` is the prompt or the negative prompt;
    `expected`, on the queueing request, is the `ids` the preview answered,
    so a selection that changed in between is refused rather than queued
    unseen. The rows are parsed by `tag_swap.parse_rows`.
    """

    ids: list[int] | None = Field(default=None, min_length=1, max_length=settings.max_grid_images)
    search: str | None = Field(default=None, max_length=2000)
    favorites: bool = True
    all_models: bool = False
    swaps: list[TagSwap] = Field(min_length=1, max_length=16)
    field: Literal["prompt", "negative_prompt"] = "prompt"
    expected: list[int] | None = Field(default=None, max_length=settings.max_grid_images)

    @field_validator("ids")
    @classmethod
    def _positive_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(image_id < 1 for image_id in value):
            raise ValueError("gallery ids start at 1")
        return value

    @model_validator(mode="after")
    def _rows_parse(self) -> "SwapTagsRequest":
        from .tag_swap import parse_rows

        parse_rows([(row.find, row.replace) for row in self.swaps])
        return self


SwapStatus = Literal[
    "queued", "planned", "missing", "unchanged", "rendered", "invalid", "source_deleted",
    "lora_missing", "mask_missing", "outpaint", "duplicate", "by_hand", "structured", "deleted", "foreign",
]


class SwapResult(BaseModel):
    """What Swap tags makes of one image.

    `text` is the swapped field as it would run (the original where nothing
    changed) and `spans` the offsets of what was written in, which the page
    marks; `removed` is what the swap took out of the original, as written
    there, which the new text cannot show (`tag_swap.Outcome`); `detail` says why an image is not queued ("missing: long hair",
    "same as #12", a validation message).
    """

    id: int
    thumb_url: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    status: SwapStatus
    detail: str | None = None
    text: str = ""
    spans: list[tuple[int, int]] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SwapTagsPreview(BaseModel):
    """Swap tags without queueing: every image's status, counted, and the
    first `SWAP_PREVIEW_ROWS` of them listed."""

    # Every image the selection resolved to, oldest first: what the queueing
    # request sends back as `expected`.
    ids: list[int]
    counts: dict[str, int]
    queued: int
    # How many jobs the queued images make: one per distinct size.
    jobs: int
    images: list[SwapResult]
    total: int


class RefineBatchRequest(BaseModel):
    """Mass refine: a refine pass of every selected image, each from its own
    pixels under its own prompt, negative prompt, seed, CFG, mu and LoRA, at
    one shared strength and scale.

    The images are the gallery selection's `ids` where cards are picked,
    else the gallery's filter as Re-render takes it. `strength` None is the
    checkpoint's own; `steps` None keeps each image's own (the profile's for
    another checkpoint's). Another checkpoint's records are refused per
    image unless `include_foreign`, and a record that is itself a refine
    pass is left out while `skip_passes`. `expected`, on the queueing
    request, is the `ids` the preview answered.
    """

    ids: list[int] | None = Field(default=None, min_length=1, max_length=settings.max_grid_images)
    search: str | None = Field(default=None, max_length=2000)
    favorites: bool = True
    all_models: bool = False
    scale: float = Field(default=1.5, ge=1.0, le=2.0)
    strength: float | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    include_foreign: bool = False
    skip_passes: bool = True
    expected: list[int] | None = Field(default=None, max_length=settings.max_grid_images)

    # Written out: a validator assigned from another model is not run.
    @field_validator("ids")
    @classmethod
    def _positive_ids(cls, value: list[int] | None) -> list[int] | None:
        return SwapTagsRequest._positive_ids.__func__(cls, value)

    @field_validator("strength")
    @classmethod
    def _a_fraction(cls, value: float | None) -> float | None:
        return GenerateRequest._a_fraction.__func__(cls, value)


RefineBatchStatus = Literal[
    "queued", "pass", "foreign", "too_big", "no_step", "file_missing", "lora_missing",
    "rendered", "planned", "deleted", "invalid",
]


class RefineBatchResult(BaseModel):
    """What Mass refine makes of one image: its size, the pass's size, and
    why it is not queued where it is not (`detail`); `notes` say what the
    pass changes of how it was made (another checkpoint's steps, CFG and
    mu; a LoRA left out)."""

    id: int
    thumb_url: str | None = None
    width: int | None = None
    height: int | None = None
    target_width: int | None = None
    target_height: int | None = None
    seed: int | None = None
    status: RefineBatchStatus
    detail: str | None = None
    notes: list[str] = Field(default_factory=list)


class RefineBatchPreview(BaseModel):
    """Mass refine without queueing: every image's status, counted, the
    first `REFINE_BATCH_PREVIEW_ROWS` listed, and what the dialog prices
    and caps the scale by."""

    ids: list[int]
    counts: dict[str, int]
    queued: int
    jobs: int
    images: list[RefineBatchResult]
    total: int
    # The strength the passes run at: the request's, else the checkpoint's.
    strength: float
    # Records another checkpoint made, queued or not, and those checkpoints:
    # what the dialog's Include other checkpoints is offered on.
    foreign: int = 0
    models: list[str] = Field(default_factory=list)
    # Records that are refine passes themselves, skipped or not.
    passes: int = 0
    # The largest source the scale applies to (by pixels), [width, height]:
    # the dialog caps its opening scale by it, as the single Refine does by
    # its one source.
    largest: list[int] | None = None
    # Summed per size from refine history; None where any size queued has
    # none, and then how many sizes have none.
    seconds: float | None = None
    unpriced_sizes: int = 0
    # Where the queued passes start on the schedule, lowest and highest:
    # one strength starts a larger pass at more noise on a shifted flow
    # schedule. None before a load, on a schedule with no noise fraction, or
    # past `REFINE_BATCH_SIGMA_KEYS` distinct passes.
    start_sigma_low: float | None = None
    start_sigma_high: float | None = None


class MoreSeedsRequest(BaseModel):
    """More seeds: `count` more images of each selected batch, every setting
    the batch's own, the seeds continuing past its highest.

    A batch is the images of one run that agree on everything but the seed
    (`more_seeds.batch_key`); two picked images of one batch ask for `count`
    more of it, not of each (`more_seeds`). The images are the gallery
    selection's `ids` where cards are picked, else the gallery's filter as
    Re-render takes it - or, with `run`, every live image of that run,
    whatever the filter: `favorites` and `all_models` are then ignored, and
    `ids` or `search` beside it is refused. The new images run at the
    batch's own Batch where its images recorded one, else at `batch` - the
    page's - either held to the pixel budget. `expected`, on the queueing
    request, is the `ids` the preview answered; a run's images are not
    capped, so neither is its `expected`.
    """

    ids: list[int] | None = Field(default=None, min_length=1, max_length=settings.max_grid_images)
    search: str | None = Field(default=None, max_length=2000)
    run: str | None = Field(default=None, min_length=1, max_length=64)
    favorites: bool = True
    all_models: bool = False
    count: int = Field(default=4, ge=1, le=settings.max_grid_images)
    batch: int = Field(default=1, ge=1, le=MAX_BATCH)
    expected: list[int] | None = None
    # A preview-issued token makes an uncertain submit safely replayable.
    # Legacy callers without one retain deliberate fresh-extension semantics.
    intent: str | None = Field(default=None, min_length=1, max_length=128)

    # Written out: a validator assigned from another model is not run.
    @field_validator("ids")
    @classmethod
    def _positive_ids(cls, value: list[int] | None) -> list[int] | None:
        return SwapTagsRequest._positive_ids.__func__(cls, value)

    @model_validator(mode="after")
    def _one_scope(self) -> "MoreSeedsRequest":
        if self.run is not None and (self.ids is not None or (self.search or "").strip()):
            raise ValueError("a run is the whole selection: send run without ids or search")
        if self.run is None and self.expected is not None and len(self.expected) > settings.max_grid_images:
            raise ValueError(f"expected lists at most {settings.max_grid_images} ids")
        return self


MoreSeedsStatus = Literal[
    "queued", "foreign", "invalid", "source_deleted", "lora_missing", "mask_missing", "outpaint", "deleted",
]


class MoreSeedsResult(BaseModel):
    """What More seeds makes of one batch, or of a picked id the gallery no
    longer holds (`deleted`, `id` its own and nothing else known).

    `id` is the batch's newest picked image, whose picture and prompt the
    row shows, and `ids` every picked image of it. `images` and
    `seed_low`/`seed_high` are what the gallery holds of the batch; `first`
    and `last` the new seeds, `skipped` the seeds past `seed_high` stepped
    over as already made or queued. `batch` is the Batch they run at,
    `recorded` whether that is the batch's own rather than the page's.
    """

    id: int
    ids: list[int] = Field(default_factory=list)
    thumb_url: str | None = None
    width: int | None = None
    height: int | None = None
    prompt: str = ""
    images: int = 0
    seed_low: int | None = None
    seed_high: int | None = None
    first: int | None = None
    last: int | None = None
    queued: int = 0
    skipped: int = 0
    batch: int | None = None
    recorded: bool = False
    status: MoreSeedsStatus
    detail: str | None = None
    notes: list[str] = Field(default_factory=list)


class MoreSeedsPreview(BaseModel):
    """More seeds without queueing: every batch's outcome, counted, the
    first `MORE_SEEDS_PREVIEW_ROWS` listed, and the time it would take."""

    # Every image the selection resolved to, oldest first: what the queueing
    # request sends back as `expected`.
    ids: list[int]
    counts: dict[str, int]
    # Images queued, and the batches and jobs they make: one job per run.
    queued: int
    batches: int
    jobs: int
    results: list[MoreSeedsResult]
    total: int
    # Queued batches running at the page's Batch, theirs not recorded.
    unrecorded: int = 0
    # Summed per size from that size's history; None where any size queued
    # has none, and then how many sizes have none.
    seconds: float | None = None
    unpriced_sizes: int = 0
    # The count a run surface opens the dialog at: the Batch the batches
    # recorded (`more_seeds.batch_size`), where every batch that recorded
    # one agrees; None where none did or they differ.
    count_default: int | None = None
    # For a `run` scope with something to queue, the server's one-line
    # account of it: per cell, the cells, the total and the seeds.
    headline: str | None = None
    intent: str | None = None


class SwapTagsAccepted(BaseModel):
    """The jobs Swap tags queued, one per distinct size, the first one's ids
    at the top level as every other submission answers them."""

    job_id: str
    seed: int
    run_id: str
    # The first job's stored price, as `JobAccepted`'s; each in `more` has its own.
    estimate_seconds: float | None = None
    more: list[JobAccepted] = Field(default_factory=list)
    queued: int
    # Images whose cell the gallery holds at queue time, left out without a
    # refusal: the preview's "already rendered" ones plus any rendered since.
    pruned: int = 0
    # Images whose cell a queued or running job already plans.
    planned: int = 0


class RefineBatchAccepted(SwapTagsAccepted):
    """The jobs Mass refine queued, one per distinct size, as Swap tags
    answers them."""


class MoreSeedsAccepted(BaseModel):
    """The jobs More seeds queued, one per run its batches extend, the
    first one's ids at the top level as every other submission answers
    them. `queued` counts images, `batches` the batches they extend."""

    job_id: str
    seed: int
    run_id: str
    estimate_seconds: float | None = None
    more: list[JobAccepted] = Field(default_factory=list)
    queued: int
    batches: int
    # A later group can fail after earlier groups were admitted. Those jobs
    # are acknowledged, with a terminal warning; replay never adds the rest.
    warning: str | None = None
    not_queued: int = 0
    not_queued_batches: int = 0
