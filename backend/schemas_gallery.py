"""What the gallery stores and serves: the record (the reproduction set as
served), run summaries, a page of records, Delete unstarred's preview, and
the recycle bin's bodies and answers.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from .config import settings
from .schemas_generate import OutpaintSpec

class ImageRecord(BaseModel):
    id: int
    filename: str
    url: str
    # A ~384 px WEBP beside the PNG, for the cards and the sheet. Null for rows
    # written before it existed; the client then shows the PNG.
    thumb_url: str | None = None
    prompt: str
    negative_prompt: str | None
    # The prompt-variable assignment that made this cell - `{"size": "big",
    # "ground": "grass"}` - when it came from a prompt grid. The resolved prompt
    # is what ran; this is what varied, so the contact sheet can lay a 2x2
    # night out as a 2x2 rather than a 4-wide strip of finished sentences. Null
    # for a plain prompt and for rows from before it was recorded.
    variables: dict[str, str] | None = None
    width: int
    height: int
    steps: int
    guidance_scale: float
    # The scale the negative prompt was applied at - FLUX.1's separate real
    # CFG, or `guidance_scale` itself on a pipeline whose CFG is real - and
    # None when it was not applied. Read through `effective_true_cfg`, so a
    # row from before the column carries what actually ran.
    true_cfg: float | None = None
    # Persisted since the first release but never served, which left the UI
    # unable to restore it - and on Turbo a different mu is a different image.
    mu: float | None = None
    # The gallery ids this image was conditioned on, in the order they were
    # concatenated. Null for a plain text-to-image row and for anything written
    # before references existed.
    reference: list[int] | None = None
    # The adapter that was applied and the weight it ran at - part of the
    # reproduction set as much as the seed, and null together when none was.
    lora: str | None = None
    lora_scale: float | None = None
    # The gallery image this one is a refine pass of, and the strength it was
    # re-denoised at - null together for a text-to-image row. `steps` is then
    # the schedule's length, of which the pass ran about `steps * strength`.
    refine: int | None = None
    strength: float | None = None
    # True when `refine` names an image no longer in the gallery: the pass is
    # kept, and so is the id, but it cannot be made again. Read, not stored -
    # the gallery's pages and `GET /api/images/{id}` look the source up.
    refine_missing: bool = False
    # True when that source is in the recycle bin rather than deleted:
    # `refine_missing` is True with it, since a binned source cannot be used
    # either, but a Restore brings it back. Read, not stored, like
    # `refine_missing`.
    refine_binned: bool = False
    # An inpaint's mask, by the 64-digit hash `storage.mask_path` resolves:
    # set only beside `refine`, which stays the source the mask was painted
    # on, so every piece of source machinery reads an inpaint as a pass.
    mask: str | None = None
    # An outpaint's spec - its fill, the padding on each side, the band -
    # set only beside `refine` and `mask` (the mask the server built from
    # it): `refine` is the unpadded source and `width` x `height` the canvas.
    outpaint: OutpaintSpec | None = None
    # How many images the pipeline call that made this one held (the panel's
    # Batch). No part of the reproduction set - each image has its own
    # generator - but what More seeds runs the batch's next images at. Null
    # for rows from before it was recorded.
    batch: int | None = None
    seed: int
    model_id: str
    # The first 16 hex digits of the sha256 of the decoded pixels. What the
    # table never recorded before: anything about what came *out*. "Did the
    # mu axis change anything" is then exact rather than a squint at 96 px
    # cells, and a search for `hash:` finds every image identical to one.
    # Null for rows written before it was recorded.
    content_hash: str | None = None
    # fp8, int8, none, nunchaku: part of the reproduction set - the same seed
    # through a different quantizer is a different picture - and so part of the
    # resume key. Null for rows written before it was recorded.
    quantization: str | None = None
    # The run that made it: a grid, a sweep, or one CLI invocation. Null for
    # anything written before runs were grouped.
    job_id: str | None = None
    # The *definition* that made it - `grid.run_id`, the same across every job
    # that worked on the grid, so a night resumed twice is still one run. Null
    # for anything written before runs had one.
    run_id: str | None = None
    duration_ms: int
    created_at: str
    favorite: bool = False
    # When the image was moved to the recycle bin, as an ISO UTC string like
    # `created_at`; None while it is live. Only the bin's own listing and
    # `GET /api/images/{id}?binned=true` serve a record with one.
    binned_at: str | None = None


class RunSummary(BaseModel):
    """One run of the gallery - a grid, a sweep, a CLI invocation - as a line.

    What the morning after needs: how many images a night produced, over how
    many jobs (a resumed grid is two), between when and when, and enough of
    its definition to recognise it. `seed` is the *oldest* image's, which is
    the base seed the grid was started from - the one a resume needs.
    """

    run_id: str
    images: int
    # Distinct pixel contents among them, so a run whose axis changed nothing
    # says so in one number. Null when any image of the run predates the hash.
    distinct: int | None = None
    favorites: int
    prompts: int
    jobs: int
    prompt: str
    seed: int
    model_id: str
    quantization: str | None = None
    width: int
    height: int
    started_at: str
    finished_at: str
    # What varied across it, so the Runs picker can name a run by its axes -
    # "steps×CFG 3×2", "strength 0.05-0.15 of #37", "comparison · 3
    # versions" - where every line read "6 img · A weathered…": each setting's
    # count of distinct values, for those with more than one (steps, cfg,
    # mu, true_cfg, lora_scale, strength, reference, seed, variant).
    axes: dict[str, int] = {}
    # The gallery image every image of the run refines, where they share one,
    # and the strengths it ran at, lowest and highest.
    refine: int | None = None
    strengths: list[float] | None = None


class GalleryPage(BaseModel):
    items: list[ImageRecord]
    # Everything the filters match, not the page: the badge's number.
    total: int
    limit: int
    offset: int
    # Whether a row older than this page exists under the same filters. Exact -
    # the server fetched one row past the page to know - where `offset < total`
    # is not, once rows have landed at the top since the first page.
    has_more: bool = False
    # Rows the same filters match on other checkpoints, left out because the
    # page is scoped to the running one: the number on the All models tick,
    # and what tells "this run made nothing" from "this run was made on the
    # other model". 0 when `all_models` is on.
    other_models: int = 0
    # The gallery's highest id over every row, read under the same reservation
    # as this page (`storage.list_images_at_head`), so no image lands between
    # the two: the page's starting point for `ModelStatus.newest_image_id`,
    # which a separate status read could put past an image this page missed.
    newest_image_id: int | None = None


class UnstarredPreview(BaseModel):
    """What the gallery's Delete unstarred would remove under a filter, and keep.

    The confirmation's numbers, taken server-side like Re-render's selection:
    the browser holds one page. `unstarred` is the count the dialog asks to
    have typed back and the DELETE is then pinned to: `records`, less the
    starred ones and less `sources_kept`, the unstarred images a starred one
    was made from (a pass's source, a render's references), which the delete
    keeps with the stars.
    """
    records: int
    favorites: int
    unstarred: int
    sources_kept: int = 0
    # The first few of those, by id, for the dialog to name.
    sources: list[int] = Field(default_factory=list)
    # How many runs the delete reaches into, and its first and last creation
    # times: the sample prompts are alphabetical and say nothing of either.
    runs: int = 0
    first: str | None = None
    last: str | None = None
    # How many of `unstarred` have an id above the request's `newer_than` -
    # the newest image the asking page held - or None where it was not asked:
    # the dialog's "made since this page last loaded", by the delete's own
    # predicate, which keeps a starred image's sources the page cannot see.
    newer: int | None = None
    # A few of the prompts, and the jobs and checkpoints a substring term
    # matched: the dialog names what is about to go, as `clear --where` does.
    prompts: list[str]
    jobs: list[str]
    models: list[str]
    # Whether a job is queued or running on the server - this page's or
    # another's - which can land an image in scope before the count is typed
    # back, and the pinned delete would then refuse.
    job_active: bool = False


class IdsBody(BaseModel):
    """Gallery ids to move to the recycle bin or back out of it."""

    ids: list[int] = Field(min_length=1, max_length=settings.max_grid_images)


class FavoritesBody(IdsBody):
    """Gallery ids to star (`value` true) or unstar together: the gallery's
    picked cards, Star picked and Unstar picked."""

    value: bool = True


class FavoritesResult(BaseModel):
    """What `POST /api/images/favorite` did with each id: starred or
    unstarred now, so already, in the recycle bin (which keeps the flag it
    was binned with, as a single star's 404 says), or not in the gallery."""

    changed: list[int]
    already: list[int]
    binned: list[int]
    missing: list[int]


class BinResult(BaseModel):
    """What `POST /api/images/bin` did with each id: moved now, in the bin
    already (a second tab, a double press), or not in the gallery at all."""

    binned: list[int]
    already: list[int]
    missing: list[int]


class RestoreResult(BaseModel):
    """What `POST /api/images/restore` did with each id: back in the gallery
    now, never in the bin (live already), or not in the gallery at all."""

    restored: list[int]
    not_binned: list[int]
    missing: list[int]


class PurgeBody(BaseModel):
    """Exactly the binned picks confirmed, repeated as their identity pin."""

    ids: list[Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]] = Field(
        min_length=1, max_length=settings.max_grid_images
    )
    expected_ids: list[Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]] = Field(
        min_length=1, max_length=settings.max_grid_images
    )


class PurgeResult(BaseModel):
    """The picked rows permanently deleted; a changed set refuses as a whole."""

    deleted: list[int]
    files: int


class BinSummary(BaseModel):
    """The recycle bin's footer count and what Empty recycle bin's
    confirmation says before it deletes from disk.

    `bytes` is the PNGs and thumbnails of the binned images; `sources_in_use`
    the binned ids a live image names as its refine source or a reference,
    which cannot be made again once the bin is emptied. `job_active` is
    whether a job is queued or running, whose cell using one would fail.
    `pin` names the binned ids (`gallery_delete.bin_pin`): Empty recycle
    bin sends it back, so a bin whose count is the same but whose images
    are not is refused.
    """

    records: int
    bytes: int
    sources_in_use: list[int] = Field(default_factory=list)
    pin: str = ""
    job_active: bool = False
