"""What the model status poll serves and the switch request: the structured
prompting guide and its tag helper, the critic's and Expand's cold status,
`ModelSwitchRequest` and `ModelStatus`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .config import settings
from .schemas_prompt import AuthorOrder

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
    # A checkpoint-specific spelling, applied only to Danbooru artist names.
    artist_prefix: str | None = None
    recommended_suffix: list[str] = Field(default_factory=list)
    negative_prompt: str | None = None
    # Served beside the guide's own keys (`model_defaults._served_tag_helper`)
    # from what Check's Order section reads, so the two say one thing:
    # `order` is `documented_order`'s steps (empty where there is none), its
    # cite and page; `author_orders` the authors' orders Check lists where no
    # order is checked; `group_steps` the step each group's tags are placed
    # in, and `year_step` the year form's.
    order_cite: str | None = None
    order_url: str | None = None
    author_orders: list[AuthorOrder] = Field(default_factory=list)
    group_steps: dict[str, str] = Field(default_factory=dict)
    year_step: str | None = None


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
    # Where `pixels_word` is "example": whose sizes those are and what, as the
    # size prose names them ("the author's reference sizes"), for the size
    # line's "above the 1.38 MP of ...". None for a range in writing.
    pixels_source: str | None = None
    # Null means any framing inside the pixel range. A finite model-card table
    # can instead name the aspect buttons that deserve the recommended state.
    aspects: list[str] | None = None
    # Its exact dimensions too, so a hand-entered author size can be recognised
    # without a loose ratio tolerance aliasing 16:9 to nearby 7:4.
    resolutions: list[tuple[int, int]] | None = None
    examples: list[str] = Field(default_factory=list)
    # Keyed by prompt so a reordered menu cannot attach an excerpt note to
    # the wrong example. Notes describe editing; they are never prompt text.
    example_notes: dict[str, str] = Field(default_factory=dict)
    # The authors' template slots in those examples - Z-Anime's "[character]"
    # and "[location]" - as the placeholder rule's patterns, matched
    # case-insensitively: the menu says "fill in" beside an example holding
    # one, and a pick selects the first, so typing replaces it.
    slots: list[str] = Field(default_factory=list)
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


class ExpanderDownload(BaseModel):
    """One thing a first Expand downloads, and where in the expansion it
    falls: the page's running line names each at its own step."""

    name: str
    bytes: int
    # "expand": fetched before the expansion runs (ERNIE's rewriter);
    # "translate": after it, for the English (the translator).
    step: Literal["expand", "translate"]


class ExpanderStatus(BaseModel):
    """What the page says about ERNIE's Expand before it is pressed, from the
    live pipeline and a few file checks, as `CriticStatus` is the critic's: a
    first use loads a 7.7 GB rewriter, and downloads it and a translator
    where they are not on disk, behind a small button in the Prompt caption."""

    # The rewriter is on the live pipeline: an Expand waits for nothing but
    # itself. False before the image model loads, and again after Unload.
    loaded: bool = False
    # What a first use downloads, in the order it does - ERNIE's rewriter,
    # the translator, both or neither - and their size together in bytes.
    fetch: list[ExpanderDownload] = Field(default_factory=list)
    fetch_bytes: int = 0


class ModelSwitchRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=512)
    quantization: Literal["fp8", "int8", "none", "nunchaku"] | None = None
    offload: Literal["model", "sequential", "none"] | None = None
    format: Literal["nvfp4", "int4"] | None = None
    load_on_startup: bool | None = None
    load: bool = True


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "ready", "error"]
    switching: bool = False
    model_revision: int = 0
    defaults: dict[str, float | int | bool | None] = {}
    # Prose for the same profile, keyed by control name - what steps/CFG/mu mean
    # on *this* checkpoint. Unlike `defaults`, which the UI applies once, these
    # follow the server for the whole session.
    hints: dict[str, str] = {}
    guide: PromptGuide = PromptGuide()
    model_id: str
    quantization: str
    offload: str
    format: str = "nvfp4"
    load_on_startup: bool = False
    # The saved choices this start's flags or environment set aside, as
    # {preference: what set it} - "--model", "LOCAL_DIFFUSION_QUANT",
    # "--defaults", "--menu" - so the page can say a saved choice will not
    # apply to a start like this one. Empty when imported rather than launched.
    startup_overrides: dict[str, str] = {}
    # Why the last switch stopped before selecting its checkpoint, with the
    # running one still selected and loaded as it was; not a load failure.
    switch_error: str | None = None
    detail: str | None = None
    # Beside a failed load's detail: its kind where it was one of the Hub's
    # ("gated", "not_found", "offline") and the model page that fixes it, for
    # the viewer's card to link - None for every other failure.
    detail_kind: str | None = None
    detail_url: str | None = None
    # When the server process started (Unix seconds, its clock). A page that
    # sees it change knows the server restarted, and that every job it was
    # following ended with it.
    started_at: float | None = None
    # The start-up catalogue's name for the running checkpoint (launch.py's
    # CHECKPOINTS, by exact id), which the page shows: the repository's own
    # name hid the pinned version - NetaYume v4 read "NetaYume-Lumina-Image-
    # 2.0", the parent model's - and None off the catalogue, where the page
    # shows the id's last part. `model_id` stays the public id everywhere it
    # is stored.
    model_label: str | None = None
    # The ids of the jobs queued or running, oldest first. A page watching
    # none of them otherwise never learned that another tab had a job on the
    # card - it offered Unload and the critic, and its gallery missed the
    # images - and one watching its own never learned of a job queued behind
    # it from elsewhere, or of a queued one cancelled there.
    active_jobs: list[str] = []
    # What holds the card - the generation gate - now: "job", or the editor
    # action holding it ("critic", "expand"), with the ticket that action was
    # sent with (`PromptExpansionRequest.ticket`); None while it is free. An
    # Expand waiting behind another tab's critique said "Expanding…" for the
    # whole of it; its running line now says it waits for the card, and for
    # whose, where the holder is not its own press.
    card_holder: str | None = None
    card_ticket: str | None = None
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
    # Beside it, where it is True: whether the expander is loaded and what a
    # first use downloads. None for every pipeline without one.
    prompt_expander: ExpanderStatus | None = None
    # The local prompt critic: supported where it is not switched off and a
    # server can run - a platform with a pinned llama-server build, or one
    # LOCAL_DIFFUSION_CRITIC_SERVER names; ready once the binary and the model
    # file are on disk, so the page can say a first use downloads them.
    prompt_critique_supported: bool = False
    prompt_critique_ready: bool = False
    # What the line under the critic's button names, rather than a fixed
    # "Qwen3-4B" an override makes untrue.
    prompt_critic: CriticStatus | None = None
