"""What the model status poll serves and the switch request: the structured
prompting guide and its tag helper, the critic's and Expand's cold status,
`ModelSwitchRequest` and `ModelStatus`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from .config import settings
from .backends.regional_compile import RegionalDiagnostics
from .execution import ExecutionSpec
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
    # A positive prefix the card recommends, offered as a button (Anima
    # Turbo's "Start with"), never checked.
    recommended_prefix: list[str] = Field(default_factory=list)
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
    # label, or the stem of the file AURADREAM_CRITIC_MODEL names.
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
    # "expand": fetched before the expansion runs (the expander's weights);
    # "translate": after it, for the English (the translator).
    step: Literal["expand", "translate"]


class ExpanderStatus(BaseModel):
    """What the page says about Expand before it is pressed, from the live
    pipeline and a few file checks, as `CriticStatus` is the critic's: a
    first use loads ERNIE's 7.7 GB rewriter (and downloads it and a
    translator), or FLUX 3's 18.8 GB prompt upsampler, where they are not on
    disk, behind a small button in the Prompt caption."""

    # The expander as the page names it: "ERNIE's prompt rewriter", "FLUX 3's
    # prompt upsampler".
    name: str = "ERNIE's prompt rewriter"
    # Whether its answer is translated after (ERNIE's, which may be Chinese);
    # FLUX 3's upsampler writes English only.
    translates: bool = True
    # The expander is on the live pipeline: an Expand waits for nothing but
    # itself. False before the image model loads, and again after Unload.
    loaded: bool = False
    # What a first use downloads, in the order it does - the expander, the
    # translator, both or neither - and their size together in bytes.
    fetch: list[ExpanderDownload] = Field(default_factory=list)
    fetch_bytes: int = 0
    # Why it cannot run under the current settings (FLUX 3's upsampler
    # without offload), said before the press; None where it can.
    held: str | None = None



class KeepAwakeStatus(BaseModel):
    """Whether the server holds the PC awake (`keep_awake`), from its last
    check: the status poll starts no process. kind is "systemd-inhibit" on
    Linux with it on the PATH, "power-request" on Windows, "none" where
    nothing can be held, "off" with AURADREAM_KEEP_AWAKE=false. A refused
    hold is wanted but not holding, with systemd-inhibit's or Windows' words
    in detail."""

    kind: Literal["systemd-inhibit", "power-request", "none", "off"]
    holding: bool = False
    wanted: bool = False
    reasons: list[str] = Field(default_factory=list)
    detail: str | None = None
    idle_minutes: int = 0


class ModelSwitchRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=512)
    quantization: Literal["fp8", "int8", "none", "nunchaku"] | None = None
    offload: Literal["model", "sequential", "none"] | None = None
    format: Literal["nvfp4", "int4"] | None = None
    load_on_startup: bool | None = None
    load: bool = True
    # Omission preserves loader intent; explicit empty selects Eager/native.
    # Only load=false may save a supported but unavailable loader recipe.
    execution: ExecutionSpec | None = None


class ModelDownloadStatus(BaseModel):
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    cache_state: Literal["complete", "partial", "absent", "unknown"] = "unknown"
    cache_free_bytes: int | None = None
    approximate: bool = True
    note: str = ""


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "ready", "error"]
    switching: bool = False
    idle_unload_minutes: int = 0
    idle_unloaded: bool = False
    model_revision: int = 0
    # Advances after every image-model load attempt, including failures that
    # downloaded only some files; lets an open picker refresh its cache facts.
    cache_revision: int = 0
    model_download: ModelDownloadStatus | None = None
    load_started_at: float | None = None
    # Numbers and flags, plus `instruction_edit_reason`: the held edit's words.
    defaults: dict[str, float | int | bool | str | None] = {}
    # Prose for the same profile, keyed by control name - what steps/CFG/mu mean
    # on *this* checkpoint. Unlike `defaults`, which the UI applies once, these
    # follow the server for the whole session.
    hints: dict[str, str] = {}
    guide: PromptGuide = PromptGuide()
    model_id: str
    quantization: str
    offload: str
    execution: ExecutionSpec | None = None
    execution_available: bool = True
    execution_unavailable_reason: str | None = None
    execution_configuration_reason: str | None = None
    execution_options: dict[str, list[dict]] = Field(default_factory=dict)
    # Observations for one actual published load, separate from chosen intent.
    # Returned does not establish warm kernels, disk-cache hits or qualification.
    execution_diagnostics: RegionalDiagnostics | None = None

    @field_serializer("execution")
    def _compact_execution(self, value: ExecutionSpec | None) -> dict | None:
        return value.model_dump(exclude_none=True) if value is not None else None

    format: str = "nvfp4"
    load_on_startup: bool = False
    # The saved choices this start's flags or environment set aside, as
    # {preference: what set it} - "--model", "AURADREAM_QUANT",
    # "--defaults", "--menu" - so the page can say a saved choice will not
    # apply to a start like this one. Empty when imported rather than launched.
    startup_overrides: dict[str, str] = {}
    # Whether a Hugging Face token is present on this machine: HF_TOKEN (or
    # HUGGING_FACE_HUB_TOKEN) set or
    # the token file existing (`hf_sign_in`), never read or checked with the Hub.
    hf_token_present: bool = False
    # The open checkpoint a fresh visit's picker preselects (an unsaved choice)
    # while the selection is the gated built-in default nothing chose and no
    # token is present (`main._open_model_suggestion`); None otherwise.
    open_model_suggestion: str | None = None
    # Snapshot before Diffusers imports, never the external variable's value.
    attention_environment_warning: str | None = None
    # Why the last switch stopped before selecting its checkpoint, with the
    # running one still selected and loaded as it was; not a load failure.
    switch_error: str | None = None
    detail: str | None = None
    # Failed-load categories: gated/not_found/offline, disk_full/connection.
    # A model-page URL accompanies actionable Hub errors only; other kinds
    # retain their plain retry advice without a URL.
    detail_kind: str | None = None
    detail_url: str | None = None
    # When the server process started (Unix seconds, its clock). A page that
    # sees it change knows the server restarted, and that every job it was
    # following ended with it.
    started_at: float | None = None
    # The checkout's short Git commit as the server process started
    # (`build_info.read_commit`, read once at import): after a pull without a
    # restart it is still the old one, which the checkout's own no longer
    # says. None when it could not be told (no Git checkout, no git).
    commit: str | None = None
    # The server's clock now (Unix seconds): a pause's `paused_at` is on it,
    # and a WSL server's clock is a VM's, which drifts from the host's over a
    # sleep - the page counts a pause's minutes against this, not its own.
    server_time: float | None = None
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
    # The gallery's highest image id, binned rows included (0 when empty;
    # `storage.newest_image_id`). A page learned of another tab's job only by
    # seeing it in `active_jobs`, and one queued, run and finished between two
    # polls never showed there: its images stayed out of the gallery until a
    # reload. The page catches up when this moves past what it has seen and
    # its own jobs do not explain it. None while the gallery is reserved
    # (a bundle operation, a clear) - the page keeps what it had.
    newest_image_id: int | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    device_name: str | None = None
    # Free space on the volume the images land on. One syscall, so it rides on
    # the status poll; the file tally (`summarise()`) walks the directory and
    # does not.
    disk_free_mb: int | None = None
    # ERNIE's separate rewriter and FLUX 3's prompt upsampler are an explicit
    # editor action, never a hidden part of generation. False for every
    # pipeline without that facility.
    prompt_expansion_supported: bool = False
    # Beside it, where it is True: whether the expander is loaded and what a
    # first use downloads. None for every pipeline without one.
    prompt_expander: ExpanderStatus | None = None
    # The local prompt critic: supported where it is not switched off and a
    # server can run - a platform with a pinned llama-server build, or one
    # AURADREAM_CRITIC_SERVER names; ready once the binary and the model
    # file are on disk, so the page can say a first use downloads them.
    prompt_critique_supported: bool = False
    prompt_critique_ready: bool = False
    # What the line under the critic's button names, rather than a fixed
    # "Qwen3-4B" an override makes untrue.
    prompt_critic: CriticStatus | None = None
    # Keeping the PC awake while busy; None outside the server's lifespan.
    keep_awake: KeepAwakeStatus | None = None
    # Whether this server asks for a passcode (`passcode.py`): true while
    # its gate is on, which is when the page offers Sign out. Never the
    # passcode, nor whether an unasked one is set.
    passcode: bool = False
