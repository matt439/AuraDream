"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import settings
from .model_defaults import defaults_for
from .seeds import MAX_BATCH, MAX_GRID_SEEDS, MAX_RUNS, MAX_SEED


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    width: int = Field(default=1024, ge=256, le=settings.max_side)
    height: int = Field(default=1024, ge=256, le=settings.max_side)
    # Sampling defaults are not constants: whether 8 steps at CFG 0 or 28 at 3.5
    # is right depends on the checkpoint, and a copy of one profile frozen here
    # is wrong for every other one - Turbo's numbers used to run on FLUX for any
    # client that omitted them, /docs included. None means "not given"; the
    # validator below fills those in from `model_defaults`.
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    # Flow-matching timestep shift. Explicit null means "no shift of ours at
    # all", which is not the same as omitting it - see _profile_defaults.
    mu: float | None = Field(default=None, ge=0.0, le=10.0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    num_images: int = Field(default=1, ge=1, le=MAX_BATCH)
    # Sequential passes. Each one is a separate pipeline call whose base seed is
    # the previous run's plus `num_images`, so a long sweep costs no more VRAM
    # than a single image - which is what makes 2K sweeps possible at all.
    runs: int = Field(default=1, ge=1, le=MAX_RUNS)

    @model_validator(mode="after")
    def _profile_defaults(self) -> "GenerateRequest":
        profile = defaults_for(settings.model_id)
        if self.steps is None:
            self.steps = profile["steps"]
        if self.guidance_scale is None:
            self.guidance_scale = profile["guidance_scale"]
        # Keyed off whether the field was *set*, not off None: the UI sends
        # `mu: null` deliberately to mean "let the pipeline shift for itself",
        # and filling that in would put Turbo's 1.15 back on an image the user
        # asked to be free of it - and leave the record unable to reproduce it.
        if "mu" not in self.model_fields_set:
            self.mu = profile["mu"]
        return self

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, v: int) -> int:
        # The VAE downsamples by 16; off-grid sizes get silently rounded anyway.
        if v % 16:
            raise ValueError("width/height must be a multiple of 16")
        return v

    def check_budget(self) -> None:
        if self.width * self.height > settings.max_pixels:
            raise ValueError(
                f"{self.width}x{self.height} exceeds the {settings.max_pixels:,} pixel budget"
            )
        # The batch is held in VRAM as one forward pass, so it multiplies the
        # pixels the way the per-image budget was meant to bound.
        if self.width * self.height * self.num_images > settings.max_batch_pixels:
            raise ValueError(
                f"{self.num_images} x {self.width}x{self.height} in one batch exceeds the "
                f"{settings.max_batch_pixels:,} pixel budget; use Runs instead of Batch, "
                "which costs one image's VRAM at a time"
            )

    def applied(
        self,
        *,
        negative_prompt_supported: bool,
        cfg_threshold: float,
        mu_supported: bool = True,
    ) -> "GenerateRequest":
        """This request as the pipeline will actually run it.

        A negative prompt is dropped where it cannot apply - a pipeline that
        takes none, or a guidance scale at or below the threshold where the
        uncond pass switches on - and mu where the pipeline has no way to take
        one. What comes back is what gets *recorded*: the gallery row, the PNG
        and the resume key all describe the image that was made, not the boxes
        that were filled in, so a resume whose negative prompt the pipeline never
        saw still matches the rows it wrote, and Reuse restores what ran.
        """
        changes: dict[str, Any] = {}
        if self.negative_prompt and (
            not negative_prompt_supported or self.guidance_scale <= cfg_threshold
        ):
            changes["negative_prompt"] = None
        if self.mu is not None and not mu_supported:
            changes["mu"] = None
        return self.model_copy(update=changes) if changes else self


def error_detail(exc: ValueError) -> str:
    """One line out of an axis parser or out of pydantic, for a 422 body or a terminal.

    A cell that violates the schema - `--grid-steps 90-120` against the 100 cap -
    raises a ValidationError, whose own str() is several lines of type and url
    noise. The UI puts this straight under the field; the CLI prints it.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'request'}: {error['msg']}"
        for error in errors()
    )


class GridRequest(GenerateRequest):
    """A sweep over the sampling axes, sitting on top of a normal request.

    The inherited fields are the *base*: an axis left blank contributes the one
    value already in the panel, so a grid with no axes set is the image Generate
    would have made. That holds for the prompt too: no variables, and every cell
    carries the prompt as typed. `runs` has no meaning here - the seed axis is
    what a grid steps through - and is rejected rather than silently multiplying
    the night.
    """

    # Named values for the placeholders in the prompt, one variable per line -
    # `size = big | small`. Longer than the numeric axes because its values are
    # prose, and split on `|` because a prompt is full of commas.
    prompt_axis: str = Field(default="", max_length=2000)
    steps_axis: str = Field(default="", max_length=200)
    cfg_axis: str = Field(default="", max_length=200)
    mu_axis: str = Field(default="", max_length=200)
    seeds: int = Field(default=1, ge=1, le=MAX_GRID_SEEDS)
    # Skip the cells this prompt has already been rendered at. What makes an
    # overnight grid survivable: a crash costs the cell it died on, not the
    # night. Matched on settings, not on the job that ran them.
    resume: bool = False

    @model_validator(mode="after")
    def _runs_are_not_a_grid_axis(self) -> "GridRequest":
        if self.runs != 1:
            raise ValueError("a grid steps its own seeds; leave runs at 1")
        return self

    @model_validator(mode="after")
    def _resume_needs_a_fixed_seed(self) -> "GridRequest":
        # Without one the resumed grid draws a fresh random base, matches none
        # of the night it is meant to continue, and quietly renders the whole
        # thing again. The base seed is published when the grid is queued and
        # written into the Seed box; failing that, the *oldest* image of the run
        # carries it - the newest carries the highest seed the night reached.
        if self.resume and self.seed is None:
            raise ValueError(
                "resume needs the seed the grid was started from - the Seed box shows it "
                "after Run grid, and the oldest image of the run carries it"
            )
        return self


class GridPreview(BaseModel):
    """What a grid would do, without queueing it."""

    # Keyed by axis: the numbers each one resolved to, and under `prompt` the
    # finished prompt of every cell - the substitution is the half of a prompt
    # grid that is easy to get wrong, so the panel shows what it produced.
    values: dict[str, list[float | int | str | None]]
    cells: int
    images: int
    # Cells dropped because the gallery already holds them; always 0 unless the
    # request asked to resume.
    skipped: int = 0
    summary: str
    # Null until the gallery holds an image at this size to measure against.
    seconds: float | None = None


class ImageRecord(BaseModel):
    id: int
    filename: str
    url: str
    # A ~384 px WEBP beside the PNG, for the cards and the sheet. Null for rows
    # written before it existed; the client then shows the PNG.
    thumb_url: str | None = None
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    # Persisted since the first release but never served, which left the UI
    # unable to restore it - and on Turbo a different mu is a different image.
    mu: float | None = None
    seed: int
    model_id: str
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


class RunSummary(BaseModel):
    """One run of the gallery - a grid, a sweep, a CLI invocation - as a line.

    What the morning after needs: how many images a night produced, over how
    many jobs (a resumed grid is two), between when and when, and enough of
    its definition to recognise it. `seed` is the *oldest* image's, which is
    the base seed the grid was started from - the one a resume needs.
    """

    run_id: str
    images: int
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


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "ready", "error"]
    defaults: dict[str, float | int | bool | None] = {}
    # Prose for the same profile, keyed by control name - what steps/CFG/mu mean
    # on *this* checkpoint. Unlike `defaults`, which the UI applies once, these
    # follow the server for the whole session.
    hints: dict[str, str] = {}
    model_id: str
    quantization: str
    offload: str
    detail: str | None = None
    # The most images one grid may queue, which is also the page size the
    # contact sheet asks for: a matrix with a page boundary through it is a
    # puzzle, so the client reads the ceiling from here rather than assuming it.
    max_grid_images: int = settings.max_grid_images
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    device_name: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    # The base seed the job drew or was given. A grid with a blank Seed box is
    # otherwise unrecoverable: every image carries a *derived* seed, and resume
    # needs the base.
    seed: int
    run_id: str
