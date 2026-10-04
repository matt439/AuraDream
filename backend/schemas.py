"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

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
    return defaults_for(settings.model_id).get("refine_strength") or DEFAULT_STRENGTH


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
        if v % 16:
            raise ValueError("width/height must be a multiple of 16")
        return v

    def check_budget(self) -> None:
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


class PromptExpansionRequest(BaseModel):
    """A visible, pre-generation rewrite of the prompt editor's contents."""

    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=settings.max_side)
    height: int = Field(default=1024, ge=256, le=settings.max_side)


class PromptExpansion(BaseModel):
    original_prompt: str
    expanded_prompt: str
    translated_prompt: str | None = None
    translation_detail: str | None = None


class PromptAnalysisRequest(GenerateRequest):
    """What Generate would send, analysed instead of rendered.

    The whole request rather than the text alone, because half of what a prompt
    checker has to say depends on the rest of it: a negative prompt at CFG 0 on
    Krea 2 Turbo, or without True CFG on FLUX.1, never runs, and that is
    `applied()`'s answer, not the text's. The critic takes the same body.
    """

    @model_validator(mode="after")
    def _refine_leaves_a_step(self) -> "PromptAnalysisRequest":
        self.check_refine_steps()
        return self


class PromptFinding(BaseModel):
    """One thing the analyser noticed, and what it rests on.

    `source` is the sourcing rule made visible: `guide` and `author` findings
    quote the checkpoint authors (`cite`, verbatim, with `url`), `encoder` ones
    describe what the pipeline's own tokenizer or template does to the text,
    and `heuristic` ones are this app's pattern matching - the rewrite checks
    on the critic's output included - and say so. `start`/`end` are character
    offsets into
    `field`'s text, so the page can select the words a finding is about -
    code points, as Python indexes a str, which the page converts to its
    UTF-16 units; every offset in the analysis and the critique counts so.
    """

    rule: str
    severity: Literal["info", "warn"]
    message: str
    field: Literal["prompt", "negative_prompt"] = "prompt"
    start: int | None = None
    end: int | None = None
    source: Literal["guide", "author", "encoder", "heuristic"]
    cite: str | None = None
    url: str | None = None
    suggestion: str | None = None


class EncoderReport(BaseModel):
    """How much of one field one text encoder reads.

    `tokens` counts the text's own tokens, the pipeline's template excluded;
    `budget` is how many of them survive the pipeline's truncation. `cut` is
    the character offset where reading stops - None when everything fits -
    because every pipeline here cuts from the right in silence.
    """

    name: str
    field: Literal["prompt", "negative_prompt"] = "prompt"
    tokens: int
    budget: int
    cut: int | None = None
    # What the encoder contributes, where that changes what the count means -
    # FLUX.1's CLIP-L feeds one pooled vector, not the token sequence.
    role: str | None = None
    # A cut here is advice, not lost text: this encoder feeds one pooled vector
    # while another reads the words on (FLUX.1's CLIP-L). Its `encoder.cut` is
    # info rather than a warning; every other role's cut is a real loss.
    pooled_only: bool = False
    # The tokenizer the count came from, repo and subfolder, for the tooltip.
    source: str | None = None


class PromptMeasureRequest(BaseModel):
    """The prompt alone, as `collect()` sends it (trimmed), for the chip
    tray's count: what the encoders read of it needs nothing else."""

    prompt: str = Field(min_length=1, max_length=4000)


class PromptMeasure(BaseModel):
    """The encoder layer of a prompt check and nothing more - the counts and
    the cut, no rules and no tag lookup - for the chip tray to mark the tags
    past the cut while the prompt is typed. `encoders_detail` says what went
    unmeasured, a tokenizer still loading included: the tray then says the
    cut is unknown, never that there is none."""

    model_id: str
    encoders: list[EncoderReport] = Field(default_factory=list)
    encoders_detail: str | None = None


class DocumentedOrder(BaseModel):
    """The order the checkpoint's guide states, when it states one."""

    steps: list[str]
    cite: str
    url: str | None = None


class AuthorOrder(BaseModel):
    """An order an author's text gives that the app does not check: the
    excerpt the critic may cite, with its source."""

    text: str
    url: str
    # Whose words these are when not this checkpoint's own authors', or that
    # `text` is this app's translation.
    note: str | None = None


class PromptAnalysis(BaseModel):
    model_id: str
    # How the text reads: sentences, a comma tag list, both, NewBie XML, JSON.
    style: Literal["prose", "tags", "hybrid", "xml", "json"]
    words: int
    encoders: list[EncoderReport] = Field(default_factory=list)
    # Why `encoders` is empty or partial - a gated tokenizer, no network on a
    # first use - so an unchecked budget is never shown as a passed one.
    encoders_detail: str | None = None
    findings: list[PromptFinding] = Field(default_factory=list)
    # None when the guide states no order for the app to check. The page then
    # says the app checks none, rather than inventing one.
    order: DocumentedOrder | None = None
    # Whether the rules held this text to `order` at all. Only a tag
    # checkpoint's comma items are placed; a sentence's "subject first", an
    # XML prompt's blocks are documented but unchecked, and the page says so
    # rather than letting "nothing to flag" read as a passed order.
    order_checked: bool = False
    # Why some comma item a rule reads was never given its Danbooru category
    # - the tag index unreachable, no usable reply from it, or items past the
    # 64 one check may ask about - as `encoders_detail` says why a budget went
    # unmeasured. Where the guide states no order, the page says not every
    # tag was checked for an artist name (Anima's @ rule reads a category).
    # None when every item was looked up, or none needed to be.
    categories_detail: str | None = None
    # Why the order was only partly checked: which items the order rule
    # reads it could not place where that decides the order - each keeps
    # every slot a category could give it, some out of order beside another
    # item and some not, so it may sit out of order unflagged - and why they
    # have no category: the lookup's own reason above, a tag the index
    # answered for with none it could use (unknown, deprecated, past the
    # cutoff), a prose prompt's clauses (prose is never looked up), an item
    # never asked about (lettering, an @ name). Set exactly where
    # `order_checked` and `prompt_rules.places_every_item` is false with
    # `categories` - the predicate the critic reads for a reorder of its own
    # - and the page then says "Partly checked"; None otherwise, a lone item
    # and one whose every slot gives the same answer included.
    order_detail: str | None = None
    # The categories themselves, as the rules placed the items with them,
    # for the critic to hold a reorder of its own to the same placement.
    # Never sent: the page has the chips' own lookup for what a tag is.
    categories: dict[str, str] | None = Field(default=None, exclude=True)
    # Where the guide states no order: orders the authors' own texts give
    # anyway (Anima's card, the rewriter instructions), shown as information,
    # never checked. Empty where `order` is set.
    author_orders: list[AuthorOrder] = Field(default_factory=list)
    # A tag checkpoint's own tags moved into its documented order, nothing
    # added or removed; None when there is nothing to move or no order.
    reordered: str | None = None


class CritiqueIssue(BaseModel):
    kind: Literal["order", "missing", "vague", "conflict", "other"]
    problem: str
    fix: str
    # The numbered source the critic cited, resolved on the server: an issue
    # whose id names nothing it was given is dropped, never shown uncited.
    cite_id: str
    cite: str
    url: str | None = None
    source: Literal["guide", "author"]
    # Whose words `cite` is when not the checkpoint's own authors' - this
    # app's translation, a family or parent model's guide - or that it is
    # this app's general advice for a checkpoint it does not know (no `url`).
    note: str | None = None
    # The prompt's own words the issue is about, when the critic quoted any
    # that really occur - located here, not trusted from the model.
    quote: str | None = None
    start: int | None = None
    end: int | None = None


class PromptCritique(BaseModel):
    model: str
    seconds: float
    # The prompt taken apart: subject, action, setting, style, lighting,
    # camera, text to render. None where the prompt has no such part.
    parts: dict[str, str | None] = Field(default_factory=dict)
    issues: list[CritiqueIssue] = Field(default_factory=list)
    rewrite: str | None = None
    # The prompt's own pieces, as it writes them, in the critic's order - the
    # variant that isolates order in a comparison grid. Where Check placed the
    # tags itself (`order_checked`), only Check's own `reordered` text - or,
    # where it could not place every tag (an item with no category, never
    # looked up or unknown to the tag index), an order Check's order rule
    # finds nothing wrong with. None when the critic's version was too long,
    # changed words or moved nothing, split or joined the items Check reads,
    # took a form Check does not read the order of, was an order Check's rule
    # flags or, every tag placed, was not Check's own, or when the checkpoint
    # documents no order.
    reordered: str | None = None
    # Which reordering `reordered` is, for the page to say: "check" - Check's
    # own text, the documented order as it placed the tags; "allowed" - the
    # critic's, where Check could not place every tag and its order rule finds
    # nothing wrong with it; "unchecked" - the critic's reading, where Check
    # reads no order. Decided with the categories this critique's own
    # analysis had, which the page's last check need not have had (the tag
    # index back online between the two). None with `reordered`.
    reordered_kind: Literal["check", "allowed", "unchecked"] | None = None
    # The analyser run over the rewrite: a critic can reintroduce what it was
    # asked to remove, so its output is checked by the same rules as input -
    # and first by the critic's own checks on it: quoted words the prompt
    # does not ask to render.
    rewrite_findings: list[PromptFinding] = Field(default_factory=list)
    # Issues discarded because their citation named nothing the critic was
    # given, or they were empty. A quote that is not in the prompt only loses
    # its `quote`.
    dropped: int = 0
    # What else validation took out of the answer, one sentence each, for the
    # page to say: issues restating a rule finding or speaking to a length the
    # check measured, order issues the documented order reverses or that
    # cannot all hold, parts the model named that are not the prompt's words
    # or lettering, a `reordered` that was too long, was not a reorder, split
    # or joined the items Check reads, was not read or was flagged by Check's
    # order rule, or, every tag placed, was not Check's own order, and a
    # rewrite that was too long, the prompt unchanged, wrapped in quotation
    # marks or lettering words the prompt lacks.
    set_aside: list[str] = Field(default_factory=list)


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


class RerenderRequest(BaseModel):
    """Render a gallery selection again at another size.

    The selection is the gallery's own filter - the search box and the
    Favorites tick - taken server-side, because the browser holds only the
    page it has scrolled to. Every record carries the complete reproduction
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

    search: str | None = Field(default=None, max_length=2000)
    favorites: bool = True
    # The gallery's third filter: off, the selection is this checkpoint's
    # images alone, as the grid shows them; on, another checkpoint's records
    # are selected too and refused below, since their steps and CFG are not
    # this checkpoint's.
    all_models: bool = False
    width: int = Field(ge=256, le=settings.max_side)
    height: int = Field(ge=256, le=settings.max_side)

    _multiple_of_16 = GenerateRequest._multiple_of_16


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
    # Likewise: the median PNG size at this size on this checkpoint times the
    # image count. The weakest of the estimates - PNG size follows content -
    # but a night that fills the disk dies at the first save.
    bytes: int | None = None


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
    # A few of the prompts, and the jobs and checkpoints a substring term
    # matched: the dialog names what is about to go, as `clear --where` does.
    prompts: list[str]
    jobs: list[str]
    models: list[str]


class PromptTagGroup(BaseModel):
    """One documented family of tags shown as buttons in the prompt helper."""

    label: str
    tags: list[str] = Field(default_factory=list)
    # Some checkpoint-authored groups are one-of choices in their prompt order.
    exclusive: bool = False


class PromptTagHelper(BaseModel):
    """Structured, checkpoint-authored vocabulary for a tag-based prompt."""

    order: list[str] = Field(default_factory=list)
    groups: list[PromptTagGroup] = Field(default_factory=list)
    year_tag: bool = False
    vocabulary: Literal["danbooru"] | None = None
    vocabulary_cutoff: str | None = None
    # Keep spellcheck and single-sentence prose behavior for checkpoints trained
    # on both natural language and comma-separated tags.
    hybrid_prompt: bool = False
    recommended_suffix: list[str] = Field(default_factory=list)
    negative_prompt: str | None = None


class TagSuggestion(BaseModel):
    text: str
    canonical: str
    category: str
    post_count: int | None = None


class TagSuggestionResponse(BaseModel):
    available: bool
    suggestions: list[TagSuggestion] = Field(default_factory=list)
    detail: str | None = None


class TagValidationRequest(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=64)

    @field_validator("tags")
    @classmethod
    def _usable_tags(cls, values: list[str]) -> list[str]:
        tags = [value.strip() for value in values]
        if any(not tag or len(tag) > 200 for tag in tags):
            raise ValueError("tags must be 1-200 characters after trimming")
        return tags


class TagValidationItem(BaseModel):
    tag: str
    valid: bool | None
    canonical: str | None = None
    source: Literal["danbooru", "model"] | None = None
    category: str | None = None
    post_count: int | None = None
    reason: str | None = None
    # Why an unknown tag is unknown, as data beside `reason`'s sentence: the
    # page marks a tag created after the checkpoint's cutoff apart from one
    # the index never had. None for a known or unchecked tag.
    code: Literal["not_found", "deprecated", "empty", "no_date", "after_cutoff"] | None = None


class TagValidationResponse(BaseModel):
    available: bool
    results: list[TagValidationItem] = Field(default_factory=list)
    detail: str | None = None


class PromptGuide(BaseModel):
    """The structured half of the checkpoint's prompting guide.

    The prose - how to write a prompt for it, and what sizes it is documented
    for - travels in `ModelStatus.hints` under `prompt` and `size`, beside the
    other tooltips. What the UI needs as data is here: the megapixel range the
    authors vouch for, so the size line can say when a request is outside it,
    an optional finite aspect-ratio list when the authors publish one,
    the word that range deserves (a model card's "supports 0.1-2 MP" and a
    README's "recommended 1024-1280" are not the same promise), and the
    authors' own example prompts for the Examples menu. Tag-based checkpoints
    may also expose their documented finite vocabulary and quick actions.
    """

    pixels: tuple[float, float] | None = None
    pixels_word: str = "documented"
    # Null means any framing inside the pixel range. A finite model-card table
    # can instead name the aspect buttons that deserve the recommended state.
    aspects: list[str] | None = None
    # Its exact dimensions too, so a hand-entered author size can be recognised
    # without a loose ratio tolerance aliasing 16:9 to nearby 7:4.
    resolutions: list[tuple[int, int]] | None = None
    examples: list[str] = Field(default_factory=list)
    # Null for prose checkpoints. When present the UI exposes only vocabulary
    # the checkpoint authors documented, while the prompt stays freely editable.
    tag_helper: PromptTagHelper | None = None


class CriticStatus(BaseModel):
    """What the page says about the prompt critic before it is asked, from
    the settings and a few file checks, as `ModelStatus.prompt_critique_ready`
    is: the model it runs and what a first use still downloads."""

    # The model's name as the answer's footer gives it: the pinned GGUF's
    # label, or the stem of the file LOCAL_DIFFUSION_CRITIC_MODEL names.
    label: str
    # Both halves the app's pins, the pairing whose time docs/verified.md
    # measured. A server or a model of the user's own gets no time and no
    # GPU promised, since nothing here measured it.
    pinned: bool
    # What a first use downloads, in order - the pinned llama-server build,
    # the pinned GGUF, both or neither - and their size together in bytes.
    # An override is never a download.
    fetch: list[str] = Field(default_factory=list)
    fetch_bytes: int = 0


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "ready", "error"]
    defaults: dict[str, float | int | bool | None] = {}
    # Prose for the same profile, keyed by control name - what steps/CFG/mu mean
    # on *this* checkpoint. Unlike `defaults`, which the UI applies once, these
    # follow the server for the whole session.
    hints: dict[str, str] = {}
    guide: PromptGuide = PromptGuide()
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
    # Free space on the volume the images land on. One syscall, so it rides on
    # the status poll; the file tally (`summarise()`) walks the directory and
    # does not.
    disk_free_mb: int | None = None
    # ERNIE's separate rewriter is an explicit editor action, never a hidden
    # part of generation. False for every pipeline without that facility.
    prompt_expansion_supported: bool = False
    # The local prompt critic: supported where it is not switched off and a
    # server can run - a platform with a pinned llama-server build, or one
    # LOCAL_DIFFUSION_CRITIC_SERVER names; ready once the binary and the model
    # file are on disk, so the page can say a first use downloads them.
    prompt_critique_supported: bool = False
    prompt_critique_ready: bool = False
    # What the line under the critic's button names, rather than a fixed
    # "Qwen3-4B" an override makes untrue.
    prompt_critic: CriticStatus | None = None


class JobAccepted(BaseModel):
    job_id: str
    # The base seed the job drew or was given. A grid with a blank Seed box is
    # otherwise unrecoverable: every image carries a *derived* seed, and resume
    # needs the base.
    seed: int
    run_id: str
