"""The generation request and its one normalisation, `applied()`.

The request contract (fields, profile defaults, size and budget guards,
validators), the submission and its Then refine pass (`GenerateSubmission`,
`ThenRefine`), an instruction edit's submission
(`InstructionEditSubmission`), the grid sweep request and its preview, the
helpers that read recorded or raised values (`effective_true_cfg`,
`error_detail`), the answer every submission returns (`JobAccepted`) and
Load settings from PNG's models (`Png*`). An outpaint's spec and the mask
and preset answers are `schemas_outpaint.py`'s, imported here for the
request. The leaf the other `schemas_*` modules build on; `schemas.py`
re-exports its public request and response models. `default_strength` and the profile
defaults read this module's `settings` at call time, so a test that changes
the running checkpoint through `settings` patches it here.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from . import instruction_edit as edit_rules
from .instruction_edit import InstructionEditSpec, StoredInstructionEdit
from .model_selection import current_model_id
from .config import settings
from .execution import (
    EXECUTION_OMITTED, ExecutionSpec, canonical_execution, require_default_execution,
    resolve_execution,
)
from .model_defaults import defaults_for, inpaint_family, inpaint_start
from . import refine
from .refine import DEFAULT_STRENGTH, Upscaler, leaves_a_step, upscaler_supported
from .schemas_outpaint import Outpaint
from .seeds import MAX_BATCH, MAX_GRID_SEEDS, MAX_RUNS, MAX_SEED

if TYPE_CHECKING:
    from .job_state import Job

# How many reference images one request may carry. klein concatenates each one
# onto the latent sequence, and attention is quadratic in that length: four 1 MP
# references is already several times the cost of the prompt alone.
MAX_REFERENCES = 4

CropAnchor = Literal[
    "center", "top", "bottom", "left", "right",
    "top-left", "top-right", "bottom-left", "bottom-right",
]


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


def default_inpaint_strength() -> float | None:
    """Where an inpaint's strength starts on the running checkpoint: the
    profile's own measured start where it has one (`model_defaults.
    inpaint_start`, FLUX.1's 0.95), else the library's own default for its
    inpaint class (`refine.library_strength`), as the Inpaint dialog starts,
    or None where it has no inpaint enabled.

    Read off the profile's family (`model_defaults.inpaint_family`) rather
    than the loaded pipeline, so the CLI's `--mask` and an API request with
    no strength record the number the dialog would before any load.
    """
    model_id = current_model_id(settings)
    family = inpaint_family(model_id)
    if family is None:
        return None
    # A profile's own measured start (FLUX.1's 0.95) comes before the class's.
    start = inpaint_start(model_id)
    if start is not None:
        return start
    import diffusers

    pipeline = getattr(diffusers, family, None)
    cls = refine.sibling(pipeline, "Inpaint") if pipeline is not None else None
    return refine.library_strength(cls) if cls is not None else None


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
    crop_anchor: CropAnchor = "center"
    # An optional GAN before an ordinary Refine pass; absent keeps the
    # existing Lanczos source fit. Never applied to a mask or an outpaint.
    upscaler: Upscaler | None = None
    # An inpaint: a refine pass at the source's own size through the inpaint
    # class beside the pipeline, repainting where this mask is white. The
    # mask is a stored picture named by the SHA-256 of its pixels, never a
    # path (`storage.mask_path` resolves it), and means nothing without the
    # source it was painted on: `applied()` drops it with the source.
    mask: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # An outpaint: an inpaint over a padded copy of the source, at the
    # source's size plus the padding, whose mask the server builds (the new
    # border and a band into the original) and stores at submit
    # (`storage.stamp_outpaints`). Never sent with a mask; dropped by
    # `applied()` with the source, as a mask is.
    outpaint: Outpaint | None = None
    # Reserved shared reproduction extension. No non-default recipe can run
    # before its storage/key carry and backend implementation are complete.
    execution: ExecutionSpec | None = None
    # New intent may carry source/recipe alone. Complete stored replay data
    # travels here too, but its proof is always checked again at admission.
    edit: InstructionEditSpec | StoredInstructionEdit | None = None
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    num_images: int = Field(default=1, ge=1, le=MAX_BATCH)
    # Sequential passes. Each one is a separate pipeline call whose base seed is
    # the previous run's plus `num_images`, so a long sweep costs no more VRAM
    # than a single image - which is what makes 2K sweeps possible at all.
    runs: int = Field(default=1, ge=1, le=MAX_RUNS)

    @field_validator("edit", mode="before")
    @classmethod
    def _strict_edit(cls, value: object) -> InstructionEditSpec | StoredInstructionEdit | None:
        if value is None:
            return None
        if isinstance(value, StoredInstructionEdit) or isinstance(value, dict) and "edit_source" in value:
            return StoredInstructionEdit.model_validate(value)
        return InstructionEditSpec.model_validate(value)

    def validate_edit(self, *, require_bound: bool = False) -> StoredInstructionEdit | InstructionEditSpec | None:
        """Recheck copied requests before normalization, keys or model work.

        Complete replay proof is untrusted until source admission compares
        it with current pixels. Requiring it here prevents source ids alone
        from claiming a durable identity; this function performs no I/O.
        """
        spec = self._strict_edit(self.edit)
        if spec is None:
            return None
        if (
            self.refine is not None or self.mask is not None or self.outpaint is not None
            or self.reference or self.lora is not None or self.upscaler is not None
            or self.crop_anchor != "center" or self.strength is not None
        ):
            raise ValueError("an instruction edit cannot include references, LoRA, Refine, strength, mask, outpaint, upscaler or crop anchor")
        if self.negative_prompt or self.true_cfg is not None or self.mu is not None:
            raise ValueError("kontext-v1 requires an empty negative prompt, no true CFG and automatic Mu")
        require_default_execution(self.execution)
        if self.num_images != 1 or self.runs != 1:
            raise ValueError("an instruction edit runs one image per call; batch and Repeat are unavailable")
        if isinstance(spec, StoredInstructionEdit):
            if spec.source is None:
                raise ValueError(edit_rules.UNRESOLVED_SOURCE)
            resolved = edit_rules.geometry(spec.edit_source.width, spec.edit_source.height, recipe=spec.recipe)
            if (self.width, self.height) != resolved.output_size:
                raise ValueError("instruction-edit dimensions must match the frozen source geometry")
        elif require_bound:
            raise ValueError("Bind and verify the instruction-edit source before computing or saving its identity.")
        return spec

    @model_validator(mode="after")
    def _edit_is_explicit_and_exclusive(self) -> "GenerateRequest":
        self.validate_edit()
        return self

    @model_validator(mode="after")
    def _execution_is_available(self) -> "GenerateRequest":
        require_default_execution(self.execution)
        return self

    def with_execution(self, selected_loader: object) -> "GenerateRequest":
        """Resolve a runtime request narrowly, without revalidating its mask."""
        recorded = self.execution if "execution" in self.model_fields_set else EXECUTION_OMITTED
        return self.model_copy(update={"execution": resolve_execution(selected_loader, recorded)})

    @model_validator(mode="after")
    def _profile_defaults(self) -> "GenerateRequest":
        if self.edit is not None:
            recipe = edit_rules.recipe_for(self.edit.recipe)
            if self.steps is None:
                self.steps = recipe.default_steps
            if self.guidance_scale is None:
                self.guidance_scale = recipe.default_guidance_scale
            return self
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
            # An inpaint starts where the inpaint class's own default does -
            # where the Inpaint dialog starts - not at the refine strength.
            self.strength = (self.inpainting and default_inpaint_strength()) or default_strength()
        return self

    @property
    def inpainting(self) -> bool:
        """Whether this request runs the inpaint class: a source with a mask,
        or with an outpaint (whose mask the server builds). What chooses the
        class, its numbers and its words; the stored mask file's own checks
        ask `mask is not None`."""
        return self.refine is not None and (self.mask is not None or self.outpaint is not None)

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

    @model_validator(mode="after")
    def _a_mask_covers_its_source(self) -> "GenerateRequest":
        # A mask is painted on a picture's pixels, as a strength indexes a
        # source's schedule; asked for alone it would be no setting at all.
        # `applied()` drops one whose source a pipeline cannot take, which
        # `model_copy` reaches without this check.
        if self.mask is not None and self.refine is None:
            raise ValueError("a mask needs the image it covers")
        return self

    @model_validator(mode="after")
    def _an_upscaler_needs_an_ordinary_pass(self) -> "GenerateRequest":
        if self.upscaler is not None:
            if self.refine is None:
                raise ValueError("an upscaler needs the image to refine")
            if self.mask is not None or self.outpaint is not None:
                raise ValueError("an upscaler is for an ordinary refine pass, without a mask or outpaint")
        return self

    @model_validator(mode="after")
    def _a_crop_anchor_needs_an_ordinary_pass(self) -> "GenerateRequest":
        if self.crop_anchor != "center":
            if self.refine is None:
                raise ValueError("a crop anchor needs the image to refine")
            if self.mask is not None or self.outpaint is not None:
                raise ValueError("a crop anchor is for an ordinary refine pass, without a mask or outpaint")
        return self

    @model_validator(mode="after")
    def _an_outpaint_extends_its_source(self) -> "GenerateRequest":
        # Exclusive with a mask in every request a client sends; the
        # server's stamp adds the derived hash by `model_copy`, which does
        # not validate.
        if self.outpaint is not None and self.refine is None:
            raise ValueError("an outpaint needs the image it extends")
        if self.outpaint is not None and self.mask is not None:
            raise ValueError("an outpaint builds its own mask; send none with it")
        return self

    def check_refine_steps(self) -> None:
        # Check after capability validation, on actual cells: a grid's axes
        # can replace an otherwise unusable base steps/strength pair.
        if self.refine is not None:
            assert self.steps is not None and self.strength is not None, "profile defaults fill steps and strength"
            if not leaves_a_step(self.steps, self.strength):
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
        if self.edit is not None:
            self.validate_edit(require_bound=True)
            return
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
        inpaint_supported: bool = False,
        instruction_edit_supported: bool = False,
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
        # Also covers internal model_copy requests without revalidating a
        # stamped outpaint (whose derived mask is deliberately post-validation).
        require_default_execution(self.execution)
        if self.edit is not None:
            bound_edit = self.validate_edit(require_bound=True)
            if not instruction_edit_supported:
                raise ValueError(edit_rules.INSTRUCTION_EDIT_UNAVAILABLE)
            # The named first recipe has no strength, Mu or negative/true-CFG
            # branch. Preserve its complete source identity instead of running
            # ordinary capability normalization that could erase the pass kind.
            return self.model_copy(update={
                "edit": bound_edit, "negative_prompt": None, "execution": canonical_execution(self.execution),
            })
        changes: dict[str, Any] = {}
        if self.execution is not None:
            # Availability is checked above. Normalization canonicalizes an
            # empty spelling; it must never erase a known admitted recipe.
            changes["execution"] = canonical_execution(self.execution)
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
        # With a mask the question is the inpaint class's, without one the
        # image-to-image class's; where the source goes its mask goes too, and
        # a mask with no source is not a setting at all.
        # An outpaint is the inpaint class's question too, and goes with its
        # source (and its mask) wherever the source goes.
        refine, mask, outpaint = self.refine, self.mask, self.outpaint
        if refine is not None and not (inpaint_supported if self.inpainting else refine_supported):
            refine = None
            changes["refine"] = None
        if mask is not None and refine is None:
            mask = None
            changes["mask"] = None
        if outpaint is not None and refine is None:
            outpaint = None
            changes["outpaint"] = None
        if self.upscaler is not None and (
            refine is None or mask is not None or outpaint is not None
            or not upscaler_supported(current_model_id(settings))
        ):
            changes["upscaler"] = None
        if refine is None and self.strength != default_strength():
            changes["strength"] = default_strength()
        if self.crop_anchor != "center" and (refine is None or mask is not None or outpaint is not None):
            changes["crop_anchor"] = "center"
        # The sibling's `image` is the source, so references have nowhere to
        # go on a refine pass (or an inpaint: SDXL's and Z-Image's classes take
        # none) even where the text-to-image call takes them.
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


class _ErrorDetail(TypedDict):
    loc: Iterable[object]
    msg: object


class _ValidationErrors(Protocol):
    def __call__(self) -> Iterable[_ErrorDetail]: ...


def error_detail(exc: ValueError) -> str:
    """One line out of an axis parser or out of pydantic, for a 422 body or a terminal.

    A cell that violates the schema - `--grid-steps 90-120` against the 100 cap -
    raises a ValidationError, whose own str() is several lines of type and url
    noise. The UI puts this straight under the field; the CLI prints it.
    """
    # Keep the duck-typed path for callers' custom ValueError subclasses too:
    # an errors() method supplies the same location/message entries we read.
    errors: _ValidationErrors | None = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'request'}: {error['msg']}"
        for error in errors()
    )


class ThenRefine(BaseModel):
    """A refine pass of every image a Generate or a grid makes, run right
    after its cell: Mass refine's three numbers. `strength` None is the
    checkpoint's own (`default_strength`), `steps` None each cell's own. No
    prompt and no upscaler: the pass keeps its base's prompt pair and
    resizes with Lanczos, as Mass refine does."""

    scale: float = Field(default=1.5, ge=1.0, le=2.0)
    strength: float | None = None
    steps: int | None = Field(default=None, ge=1, le=100)

    @field_validator("strength")
    @classmethod
    def _a_fraction(cls, value: float | None) -> float | None:
        return GenerateRequest._a_fraction.__func__(cls, value)


class GenerateSubmission(GenerateRequest):
    """What `/api/generate` takes: the request and, beside it, what is to be
    done with its images once made. `then_refine` is never part of the
    reproduction set - a cell is built from the request without it - so the
    base rows are what a plain Generate of the same settings writes."""

    then_refine: ThenRefine | None = None
    # Composer intent only. `execution`, when present, is complete replay
    # metadata even when it is null or contains only a step-cache recipe.
    step_cache: Literal["fbc-005-v1"] | None = None

    @model_validator(mode="after")
    def _one_execution_channel(self) -> "GenerateSubmission":
        if self.edit is not None and (self.then_refine is not None or self.step_cache is not None):
            raise ValueError("instruction edits cannot use Then refine or step caching")
        if "execution" in self.model_fields_set and "step_cache" in self.model_fields_set:
            raise ValueError("Choose composer step cache or a recorded execution recipe, not both.")
        return self

    def base(self, selected_loader: object = None) -> GenerateRequest:
        """The request alone, as the cells carry it."""
        return GenerateRequest.model_validate(self.with_execution(selected_loader).model_dump())

    def with_execution(self, selected_loader: object) -> GenerateRequest:
        """Consume submission-only intent at direct runtime boundaries too."""
        recorded = self.execution if "execution" in self.model_fields_set else EXECUTION_OMITTED
        cache = self.step_cache if "step_cache" in self.model_fields_set else EXECUTION_OMITTED
        effective = resolve_execution(selected_loader, recorded, step_cache=cache)
        # Copy field objects, not serialized dictionaries. A stamped outpaint
        # deliberately carries both its spec and derived mask at this point.
        fields = {name: getattr(self, name) for name in GenerateRequest.model_fields}
        fields["execution"] = effective
        return GenerateRequest.model_construct(
            _fields_set=(self.model_fields_set - {"then_refine", "step_cache"}) | {"execution"}, **fields,
        )

    def applied(self, **kwargs: Any) -> GenerateRequest:
        # Portable normalization cannot infer a loader for composer transport.
        # Require that its submission-only channel was consumed first.
        if "step_cache" in self.model_fields_set:
            raise ValueError("Resolve composer step cache before normalizing its request.")
        return super().applied(**kwargs)


class InstructionEditSubmission(BaseModel):
    """What `POST /api/instruction-edit` takes: one gallery source, the
    named recipe, the instruction and the size its preview resolved.

    Nothing else a Generate carries - no negative, references, adapter,
    Refine, mask, batch, Repeat, grid or execution channel - so an edit
    cannot borrow a composer setting. The size is the preview's answer,
    checked against the source again at admission (`bind_request`); the
    steps and guidance, left out, are the recipe's own."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: int = Field(gt=0, le=2**63 - 1)
    recipe: Literal["kontext-v1"]
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(ge=256, le=settings.refine_max_side)
    height: int = Field(ge=256, le=settings.refine_max_side)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)

    @field_validator("prompt")
    @classmethod
    def _an_instruction(cls, value: str) -> str:
        # Kept as typed, as Generate keeps its prompt; only a blank refuses.
        if not value.strip():
            raise PydanticCustomError("blank_instruction", "write an instruction for the edit")
        return value

    def request(self, selected_loader: object = None) -> GenerateRequest:
        """The one-image request the job carries, its edit still unbound.

        Execution resolves as `GenerateSubmission.base` resolves a composer
        request's; admission then binds the source (`bind_request`).
        """
        values: dict[str, Any] = {
            "prompt": self.prompt, "width": self.width, "height": self.height,
            "edit": InstructionEditSpec(source=self.source, recipe=self.recipe),
            "num_images": 1, "runs": 1, "negative_prompt": None,
        }
        for name in ("seed", "steps", "guidance_scale"):
            if getattr(self, name) is not None:
                values[name] = getattr(self, name)
        return GenerateRequest.model_validate(values).with_execution(selected_loader)


class PromptVariant(BaseModel):
    """One whole prompt on a comparison grid's prompt axis, under a short name."""

    label: str = Field(min_length=1, max_length=40)
    prompt: str = Field(min_length=1, max_length=4000)


class GridRequest(GenerateSubmission):
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
    # Cross every value, or draw one value per variable from each image's seed.
    # Random mode runs images separately so each cell has one resolved prompt.
    prompt_mode: Literal["cross", "random"] = "cross"
    # Whole prompts as the prompt axis - the analyser's original, reordered
    # and rewritten versions on shared seeds. Variables cannot carry them:
    # three long prompts overrun `prompt_axis`, a `|` or a newline in one
    # splits it, and NewBie's XML is all newlines. Each cell records
    # `variables={"variant": label}`, so the sheet gives each its own row.
    prompt_variants: list[PromptVariant] | None = Field(default=None, min_length=2, max_length=8)
    steps_axis: str = Field(default="", max_length=200)
    # Paired sizes, never independent width and height products.
    size_axis: str = Field(default="", max_length=2000)
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
        if self.edit is not None:
            raise ValueError("instruction edits cannot run as a grid")
        if self.runs != 1:
            raise ValueError("a grid steps its own seeds; leave runs at 1")
        return self

    @model_validator(mode="after")
    def _variants_are_the_prompt_axis(self) -> "GridRequest":
        if self.prompt_mode == "random":
            if self.prompt_variants is not None:
                raise ValueError("random prompt variables cannot accompany prompt variants")
            if not self.prompt_axis.strip():
                raise ValueError("random prompt mode needs prompt variables")
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
    # With no fixed base/list, random-choice text is a sample, not the next
    # submission's promise. Counts are exact; Run grid draws another base.
    prompt_sample: bool = False
    cells: int
    images: int
    # Cells dropped because the gallery already holds them; always 0 unless the
    # request asked to resume.
    skipped: int = 0
    # Cells dropped under resume because a live job already plans them: the
    # same run still queued or running. Never counted in `skipped`, which the
    # panel shows as "already done".
    queued: int = 0
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
    # Under Then refine: the passes queued after the cells (one per image)
    # and their size. Their time and disk are in `seconds` and `bytes`.
    passes: int = 0
    pass_width: int | None = None
    pass_height: int | None = None
    pass_sizes: list[tuple[int, int]] = Field(default_factory=list)


class JobAccepted(BaseModel):
    job_id: str
    # The base seed the job drew or was given. A grid with a blank Seed box is
    # otherwise unrecoverable: every image carries a *derived* seed, and resume
    # needs the base.
    seed: int
    run_id: str
    # The job's stored price from matching history (`Job.estimate_seconds`),
    # or None: the queue row's figure from its first render. Additive.
    estimate_seconds: float | None = None

    @classmethod
    def of(cls, job: Job) -> JobAccepted:
        """The answer for a queued job, so a field added here is filled at
        every route that queues one."""
        return cls.model_validate({
            "job_id": job.id, "seed": job.seed, "run_id": job.run_id, "estimate_seconds": job.estimate_seconds,
        })


# Load settings from PNG's body (`api_png.py`): what the page read out of a
# PNG's tEXt, zTXt and iTXt chunks - the image itself never leaves the
# browser. A PNG keyword is at most 79 bytes; the route caps the whole body.
MAX_PNG_CHUNKS = 64


class PngPanelSampling(BaseModel):
    """The panel's own steps, CFG, true CFG and Mu, as Generate would send
    them - kept for another tool's PNG, whose sampling does not carry, so
    `applied()` judges what will run. Unbounded here: a value Generate would
    refuse leaves the answer at the profile's rather than refusing the file.
    """

    steps: int | None = None
    guidance_scale: float | None = None
    true_cfg: float | None = None
    mu: float | None = None


class PngSettingsRequest(BaseModel):
    """A PNG's text chunks by keyword, the file's name for the answer's
    words, and the panel's sampling (`PngPanelSampling`)."""

    name: str = Field(default="the PNG", min_length=1, max_length=255)
    chunks: dict[str, str] = Field(max_length=MAX_PNG_CHUNKS)
    panel: PngPanelSampling | None = None

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
    another sampler's and another checkpoint's, so those are the panel's
    as it sent them (else the profile's) and the seed is None. `negative_prompt` is the text the PNG carries,
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
