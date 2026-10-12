"""The generation request and its one normalisation, `applied()`.

The request contract (fields, profile defaults, size and budget guards,
validators), the grid sweep request and its preview, the helpers that read
recorded or raised values (`effective_true_cfg`, `error_detail`) and the
answer every submission returns. The leaf the other `schemas_*` modules build
on; `schemas.py` re-exports all of it. `default_strength` and the profile
defaults read this module's `settings` at call time, so a test that changes
the running checkpoint through `settings` patches it here.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .model_selection import current_model_id
from .config import settings
from .model_defaults import defaults_for
from .refine import DEFAULT_STRENGTH, leaves_a_step
from .seeds import MAX_BATCH, MAX_GRID_SEEDS, MAX_RUNS, MAX_SEED

# How many reference images one request may carry. klein concatenates each one
# onto the latent sequence, and attention is quadratic in that length: four 1 MP
# references is already several times the cost of the prompt alone.
MAX_REFERENCES = 4


def default_strength() -> float:
    """Where a refine pass's strength starts on the running checkpoint.

    The profile's, like the steps and the CFG: the same strength lands at a
    different noise level on SDXL's schedule and on a shifted flow-matching
    one, so one number for every checkpoint redraws faces on some and barely
    touches others. `DEFAULT_STRENGTH` where the profile names none - a
    checkpoint the table does not know, whose pipeline turned out to have a
    pass. Also what `applied()` puts a strength back to when there is no
    source for it to act on, so that it is one value per checkpoint.
    """
    return defaults_for(current_model_id(settings)).get("refine_strength") or DEFAULT_STRENGTH


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    # Bounded by the larger of the two guards, a refine pass's; a direct
    # render is held to the presets' by `_direct_renders_stay_in_the_presets`.
    width: int = Field(default=1024, ge=256, le=settings.refine_max_side)
    height: int = Field(default=1024, ge=256, le=settings.refine_max_side)
    # Sampling defaults are not constants: whether 8 steps at CFG 0 or 28 at 3.5
    # is right depends on the checkpoint, and a copy of one profile frozen here
    # is wrong for every other one - Turbo's numbers used to run on FLUX for any
    # client that omitted them, /docs included. None means "not given"; the
    # validator below fills those in from `model_defaults`.
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    # Real classifier-free guidance on a pipeline whose `guidance_scale` is a
    # distilled conditioning value rather than CFG - FLUX.1, where the model
    # takes the guidance number as an input and only `true_cfg_scale > 1` runs
    # a second, unconditioned pass against the negative prompt. The two used
    # to be welded together (both set from `guidance_scale`), so the usual
    # way to run a negative prompt on a dev-family FLUX - the trained 3.5
    # embedding with a mild real CFG of 1.5, or an embedding near 1 with real
    # CFG doing the work at 3-5 - could not be asked for, and typing a
    # negative prompt at 3.5 silently stacked guidance twice. None is off.
    # On a pipeline whose `guidance_scale` *is* real CFG (Krea 2, SD3, klein)
    # the knob does not exist, and `applied()` records the CFG the negative
    # prompt ran at here instead - so the column means one thing everywhere:
    # the scale the negative prompt was applied at, or None when it was not.
    true_cfg: float | None = Field(default=None, ge=1.0, le=20.0)
    # Flow-matching timestep shift. Explicit null means "no shift of ours at
    # all", which is not the same as omitting it - see _profile_defaults.
    mu: float | None = Field(default=None, ge=0.0, le=10.0)
    # Gallery images to condition on, by *id*. FLUX.2 [klein] takes references
    # as its first parameter, resizes each to <=1 MP, VAE-encodes it and
    # concatenates the tokens onto the latent sequence every step - native
    # multi-reference editing at no extra weights. Ids rather than paths on
    # purpose: a path from the request joined onto a directory is an
    # arbitrary-file read reachable from /api/generate, and everything this app
    # has made is already a row. Dropped by `applied()` where the pipeline has
    # no such parameter, and refused at submit so a 64-cell night does not
    # discover it at cell 1.
    reference: list[int] = Field(default_factory=list, max_length=MAX_REFERENCES)
    # A LoRA adapter by file name in `settings.loras_dir`, and the weight it is
    # set to. Unlike everything else here the capability is the environment's,
    # not the checkpoint's: peft must be installed and the base must be one an
    # adapter can attach to. `lora_scale` has no meaning without a name, and
    # `applied()` puts it back to the default when there is none, so two cells
    # that differ only in the scale of an adapter nobody selected are one image
    # and one resume key rather than two.
    lora: str | None = Field(default=None, max_length=200)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    # The refine pass: a gallery image, by id like a reference, resized to this
    # request's width and height and re-denoised under its prompt and settings
    # from the point on the schedule `strength` names - a low one keeps the
    # picture and redraws the detail, 1.0 starts from pure noise and keeps
    # nothing. It runs the image-to-image class diffusers ships beside the
    # checkpoint's pipeline, over the components already loaded, so the
    # capability is the pipeline's: dropped by `applied()` where there is no
    # such class, and refused at submit. `strength` has no meaning without a
    # source, and `applied()` puts it back to the default when there is none,
    # for the reason `lora_scale` goes back: one image, one resume key. None
    # is "not given", filled from the checkpoint's profile like the steps:
    # what a strength does depends on the schedule it indexes into.
    refine: int | None = Field(default=None, ge=1)
    # In (0, 1], checked by `_a_fraction` below rather than by gt/le, whose
    # "Input should be less than or equal to 1" the Refine dialog would show.
    strength: float | None = None
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    num_images: int = Field(default=1, ge=1, le=MAX_BATCH)
    # Sequential passes. Each one is a separate pipeline call whose base seed is
    # the previous run's plus `num_images`, so a long sweep costs no more VRAM
    # than a single image - which is what makes 2K sweeps possible at all.
    runs: int = Field(default=1, ge=1, le=MAX_RUNS)

    @model_validator(mode="after")
    def _profile_defaults(self) -> "GenerateRequest":
        profile = defaults_for(current_model_id(settings))
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
        if self.strength is None:
            self.strength = default_strength()
        return self

    @model_validator(mode="after")
    def _direct_renders_stay_in_the_presets(self) -> "GenerateRequest":
        # Only a refine pass goes past the largest preset's side (config's
        # `refine_max_side`): it takes its composition from the source.
        side = max(self.width, self.height)
        if self.refine is None and side > settings.max_side:
            raise ValueError(
                f"a {side}-pixel side is past the {settings.max_side} a direct render takes; "
                f"only a refine pass goes to {settings.refine_max_side}"
            )
        return self

    def check_refine_steps(self) -> None:
        # Check after capability validation, on actual cells: a grid's axes
        # can replace an otherwise unusable base steps/strength pair.
        if self.refine is not None and not leaves_a_step(self.steps, self.strength):
            raise ValueError(
                f"strength {self.strength:g} of {self.steps} steps is less than one step; "
                "raise the strength or the steps"
            )

    @field_validator("reference")
    @classmethod
    def _real_ids(cls, v: list[int]) -> list[int]:
        # Ids are the gallery's own, so 0 and negatives are typos, and the same
        # image twice is a doubled cost for one condition rather than a second
        # one. Order is kept: it is the order the tokens are concatenated in.
        for image_id in v:
            if image_id < 1:
                raise ValueError(f"{image_id} is not a gallery image id")
        return list(dict.fromkeys(v))

    @field_validator("lora")
    @classmethod
    def _bare_name(cls, v: str | None) -> str | None:
        # A name in `loras_dir`, never a path: `../../../etc/passwd` and
        # `C:\keys.safetensors` are the reason the field is a name at all.
        if v is None:
            return None
        name = v.strip()
        if not name:
            return None
        if "/" in name or "\\" in name or name in (".", "..") or ":" in name:
            raise ValueError(f"{name!r} is not a file name in the loras directory")
        return name

    @field_validator("strength")
    @classmethod
    def _a_fraction(cls, v: float | None) -> float | None:
        # The Refine dialog shows a refusal in the server's words, under the
        # field's name ("strength: ..."), so the words say what a strength is.
        # A custom error, not a ValueError: pydantic prefixes those with
        # "Value error, ". NaN fails the comparison too.
        if v is not None and not 0.0 < v <= 1.0:
            raise PydanticCustomError(
                "strength_fraction",
                "{value} is not a fraction between 0 and 1: a strength is more than 0, "
                "and at 1 keeps nothing of the source",
                {"value": f"{v:g}"},
            )
        return v

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, v: int) -> int:
        # The VAE downsamples by 16; off-grid sizes get silently rounded anyway.
        # Said with the value and its two neighbours, as the panel's size line
        # says it (sizes.js) - a custom error, so pydantic adds no "Value error, ".
        if v % 16:
            below = v - v % 16
            raise PydanticCustomError(
                "multiple_of_16",
                "{value} is not a multiple of 16: {below} or {above}",
                {"value": v, "below": below, "above": below + 16},
            )
        return v

    def check_budget(self, *, grid: bool | None = None, cli: bool = False) -> None:
        """Refuse a size or a batch past the pixel budgets.

        The batch refusal names what to do instead in the words of where the
        request came from: the panel's Repeat (`runs`, the CLI's `--runs`), or
        on a grid, which takes no repeat - its seeds are its runs - a lower
        Batch. `grid` defaults to whether this is a `GridRequest`; the CLI,
        whose grid is a plain request with axes, says so itself.
        """
        if self.refine is not None:
            if self.width * self.height > settings.refine_max_pixels:
                raise ValueError(
                    f"{self.width}x{self.height} exceeds the {settings.refine_max_pixels:,} "
                    "pixel budget of a refine pass"
                )
            # No batch budget: a pass is one pipeline call per image
            # (`DiffusersBackend._refine`), so a batch never holds more than
            # one image's pixels at once.
            return
        if self.width * self.height > settings.max_pixels:
            raise ValueError(
                f"{self.width}x{self.height} exceeds the {settings.max_pixels:,} pixel budget"
            )
        # The batch is held in VRAM as one forward pass, so it multiplies the
        # pixels the way the per-image budget was meant to bound.
        if self.width * self.height * self.num_images > settings.max_batch_pixels:
            if grid is None:
                grid = isinstance(self, GridRequest)
            batch, repeat = ("--batch", "--runs") if cli else ("Batch", "Repeat")
            advice = (
                f"lower {batch}: a grid runs its seeds one batch at a time" if grid
                else f"use {repeat} instead of {batch}, which costs one image's VRAM at a time"
            )
            raise ValueError(
                f"{self.num_images} x {self.width}x{self.height} in one batch exceeds the "
                f"{settings.max_batch_pixels:,} pixel budget; {advice}"
            )

    def applied(
        self,
        *,
        negative_prompt_supported: bool,
        cfg_threshold: float,
        mu_supported: bool = True,
        true_cfg_supported: bool = False,
        reference_supported: bool = False,
        lora_supported: bool = True,
        refine_supported: bool = False,
    ) -> "GenerateRequest":
        """This request as the pipeline will actually run it.

        A negative prompt is dropped where it cannot apply - a pipeline that
        takes none, or a guidance scale at or below the threshold where the
        uncond pass switches on - and mu where the pipeline has no way to take
        one. What comes back is what gets *recorded*: the gallery row, the PNG
        and the resume key all describe the image that was made, not the boxes
        that were filled in, so a resume whose negative prompt the pipeline never
        saw still matches the rows it wrote, and Reuse restores what ran.

        Which scale the threshold is compared against depends on the pipeline.
        Where `true_cfg_scale` exists (FLUX.1) it is `true_cfg`, and the
        distilled `guidance_scale` has no say in whether the negative prompt
        runs; everywhere else `guidance_scale` is the real CFG. Either way
        `true_cfg` comes back as the scale the negative prompt was applied at,
        or None when it was not - one meaning for the column whatever the
        checkpoint, which is what lets a resume compare rows across the two.
        """
        changes: dict[str, Any] = {}
        negative = self.negative_prompt or None
        scale = self.true_cfg if true_cfg_supported else self.guidance_scale
        if negative and (
            not negative_prompt_supported or scale is None or scale <= cfg_threshold
        ):
            negative = None
        true_cfg = scale if negative else None
        if negative != self.negative_prompt:
            changes["negative_prompt"] = negative
        if true_cfg != self.true_cfg:
            changes["true_cfg"] = true_cfg
        if self.mu is not None and not mu_supported:
            changes["mu"] = None
        # References the pipeline has no parameter for, and an adapter this
        # installation cannot load. Both are refused at submit; this is what
        # keeps the *row* honest on the paths that get past that - a `--from-image`
        # carrying a reference onto another checkpoint, say.
        if self.reference and not reference_supported:
            changes["reference"] = []
        lora = self.lora
        if lora and not lora_supported:
            lora = None
            changes["lora"] = None
        # A scale without an adapter is not a setting. Left alone it would split
        # the resume key on a number that changed no pixel, and a scale axis run
        # with no LoRA chosen would look like a sweep instead of the one image
        # repeated - which is what `duplicate_count` is there to say.
        if lora is None and self.lora_scale != 1.0:
            changes["lora_scale"] = 1.0
        # A refine source where the pipeline has no image-to-image sibling:
        # refused at submit like a reference, and dropped here so a row that
        # got past that is not recorded as a refinement of an image the
        # pipeline never saw. The strength goes with it, as the LoRA scale
        # goes with the adapter.
        refine = self.refine
        if refine is not None and not refine_supported:
            refine = None
            changes["refine"] = None
        if refine is None and self.strength != default_strength():
            changes["strength"] = default_strength()
        # The sibling's `image` is the source, so references have nowhere to
        # go on a refine pass even where the text-to-image call takes them.
        if refine is not None and self.reference and reference_supported:
            changes["reference"] = []
        return self.model_copy(update=changes) if changes else self


def effective_true_cfg(
    true_cfg: float | None, negative_prompt: str | None, guidance_scale: float
) -> float | None:
    """The scale a *recorded* image's negative prompt ran at.

    Rows written before the column existed carry NULL and a negative prompt
    that ran at `guidance_scale` - the two knobs were one then, on every
    pipeline - so the column is read through this rather than raw, and a
    night from before the change resumes against the same key it would be
    written under today. `search.TRUE_CFG_SQL` is the same rule in SQL for
    the paths that filter or group in the database; a test holds them equal.
    """
    if true_cfg is not None:
        return true_cfg
    return guidance_scale if negative_prompt else None


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


class PromptVariant(BaseModel):
    """One whole prompt on a comparison grid's prompt axis, under a short name."""

    label: str = Field(min_length=1, max_length=40)
    prompt: str = Field(min_length=1, max_length=4000)


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
    # Whole prompts as the prompt axis - the analyser's original, reordered
    # and rewritten versions on shared seeds. Variables cannot carry them:
    # three long prompts overrun `prompt_axis`, a `|` or a newline in one
    # splits it, and NewBie's XML is all newlines. Each cell records
    # `variables={"variant": label}`, so the sheet gives each its own row.
    prompt_variants: list[PromptVariant] | None = Field(default=None, min_length=2, max_length=8)
    steps_axis: str = Field(default="", max_length=200)
    cfg_axis: str = Field(default="", max_length=200)
    mu_axis: str = Field(default="", max_length=200)
    # LoRA weight as an axis. The comparison a LoRA is bought for - "the same
    # prompt at 0, 0.25, 0.5, 0.75, 1" - and a per-call setting, so it costs a
    # `set_adapters` between cells rather than a reload.
    lora_scale_axis: str = Field(default="", max_length=200)
    # Refine strength as an axis: `0.2-0.5x0.1` over one source is the sheet
    # that shows where the picture stops being the source's. Like the LoRA
    # weight, only an axis while there is something for it to act on.
    strength_axis: str = Field(default="", max_length=200)
    seeds: int = Field(default=1, ge=1, le=MAX_GRID_SEEDS)
    # The seed axis as a list - `4471,8823`, `100-103` - which wins over the
    # count and the base seed when given: the second night of an investigation
    # runs the two seeds that worked against a finer sweep. Long enough for
    # sixty-four 16-digit seeds.
    seed_axis: str = Field(default="", max_length=2000)
    # Skip the cells this prompt has already been rendered at. What makes an
    # overnight grid survivable: a crash costs the cell it died on, not the
    # night. Matched on settings, not on the job that ran them.
    resume: bool = False
    # Start no cell after this many minutes from submission. Seeds are the
    # outermost loop precisely so that a night which ends early has finished
    # every setting for the seeds it reached - and every way of ending early
    # used to be an accident. With a budget the grid is over-provisioned on
    # purpose and the night takes what it reaches; what it did not reach is a
    # resume. Counted from submission, since "until seven" includes the load.
    budget_minutes: float | None = Field(default=None, gt=0, le=7 * 24 * 60)

    @model_validator(mode="after")
    def _runs_are_not_a_grid_axis(self) -> "GridRequest":
        if self.runs != 1:
            raise ValueError("a grid steps its own seeds; leave runs at 1")
        return self

    @model_validator(mode="after")
    def _variants_are_the_prompt_axis(self) -> "GridRequest":
        if self.prompt_variants is None:
            return self
        if self.prompt_axis.strip():
            raise ValueError("prompt variants are the prompt axis; leave the variables empty")
        labels = [variant.label.strip() for variant in self.prompt_variants]
        if len(set(labels)) != len(labels):
            raise ValueError("each prompt variant needs its own label")
        return self

    @model_validator(mode="after")
    def _resume_needs_a_fixed_seed(self) -> "GridRequest":
        # Without one the resumed grid draws a fresh random base, matches none
        # of the night it is meant to continue, and quietly renders the whole
        # thing again. The base seed is published when the grid is queued and
        # written into the Seed box; failing that, the *oldest* image of the run
        # carries it - the newest carries the highest seed the night reached.
        # A listed seed axis is determinate on its own, and is the one case
        # where a resume is most clearly so.
        if self.resume and self.seed is None and not self.seed_axis.strip():
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
    # The same cells counted without being dropped, when the request did not
    # ask to resume and its seeds are fixed: what ticking Resume would skip.
    in_gallery: int = 0
    # Cells that will render the very same image as an earlier cell once the
    # pipeline's own drops are applied: a negative-prompt variable on a
    # checkpoint that takes none, a CFG axis that never clears the threshold
    # under a negative prompt. The preview used to count them as prompts and
    # price them into the estimate; the run spends the night on them.
    duplicates: int = 0
    summary: str
    # Null until the gallery holds an image at this size to measure against.
    seconds: float | None = None
    # With a time budget and no Resume: the seeds it completes under the
    # job's stop rule, a cell starting until the deadline has passed
    # (`grid.seeds_within_budget`). Null wherever `seconds` is.
    budget_seeds: int | None = None
    # Likewise: the median PNG size at this size on this checkpoint times the
    # image count. The weakest of the estimates - PNG size follows content -
    # but a night that fills the disk dies at the first save.
    bytes: int | None = None
    # A refine pass whose cells all start at one point: the sigma it starts
    # at, read off the loaded pipeline's schedule for this size
    # (`backend.start_sigma`) - on a flow-matching schedule, the share of
    # noise the source is mixed with. One strength means different things at
    # different sizes on FLUX.1, whose shift grows with the size. Null before
    # a load, for a schedule that is not flow matching, and for a sweep.
    start_sigma: float | None = None


class JobAccepted(BaseModel):
    job_id: str
    # The base seed the job drew or was given. A grid with a blank Seed box is
    # otherwise unrecoverable: every image carries a *derived* seed, and resume
    # needs the base.
    seed: int
    run_id: str


# Load settings from PNG's body (`api_png.py`): what the page read out of a
# PNG's tEXt, zTXt and iTXt chunks - the image itself never leaves the
# browser. A PNG keyword is at most 79 bytes; the route caps the whole body.
MAX_PNG_CHUNKS = 64


class PngSettingsRequest(BaseModel):
    """A PNG's text chunks by keyword, and the file's name for the answer's words."""

    name: str = Field(default="the PNG", min_length=1, max_length=255)
    chunks: dict[str, str] = Field(max_length=MAX_PNG_CHUNKS)

    @field_validator("chunks")
    @classmethod
    def _keywords(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if not 1 <= len(key) <= 79:
                raise ValueError(f"{key[:20]!r} is not a PNG text keyword")
        return v


class PngSettings(BaseModel):
    """A PNG's settings as the panel takes them, after `applied()`.

    `settings` is what Generate would run on the running checkpoint. `size`
    and `sampling` say which of it is the PNG's: the size where it fits a
    direct render here, and the steps, CFG, true CFG, Mu, seed and
    conditioning only from this app's own record - an A1111 block's are
    another sampler's and another checkpoint's, so those are the profile's
    and the seed is None. `negative_prompt` is the text the PNG carries,
    which the box keeps (greyed) where `applied()` leaves it out, as the
    panel keeps a typed one. `notes` are the sentences that say what did
    not carry over and why.
    """

    source: str
    name: str
    model_id: str | None = None
    settings: GenerateRequest
    negative_prompt: str | None = None
    size: bool
    sampling: bool
    notes: list[str]
