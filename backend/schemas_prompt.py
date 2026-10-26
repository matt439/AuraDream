"""Prompt-editor contracts that never render: Expand's request and answer,
Check's request and analysis, the chip count, the critic's critique, and
Danbooru tag suggestion and validation with their order slots.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from .config import settings
from .schemas_generate import GenerateRequest

class PromptExpansionRequest(BaseModel):
    """A visible, pre-generation rewrite of the prompt editor's contents."""

    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=settings.max_side)
    height: int = Field(default=1024, ge=256, le=settings.max_side)
    # The page's own name for this press, which the status answers as
    # `card_ticket` while the expansion holds the card: an expansion waiting
    # behind another tab's tells the two apart and says it waits.
    ticket: str | None = Field(default=None, min_length=1, max_length=64)


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
    cut is unknown, never that there is none. `encoders_known` is False where
    the encoder table has no entry for the checkpoint at all: nothing will
    ever count its prompt, so the line under the prompt box says nothing
    rather than "cut unknown" under every prompt (Check's Encoders section
    still says why)."""

    model_id: str
    encoders: list[EncoderReport] = Field(default_factory=list)
    encoders_detail: str | None = None
    encoders_known: bool = True


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
    # What the rules count as words (`prompt_rules.word_count`): each Han or
    # kana character one, as Chinese and Japanese space no words, beside the
    # spaced words. `characters` is how many of `words` are such characters,
    # so the page says "300 characters" for a Chinese paragraph, not "300
    # words".
    words: int
    characters: int = 0
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
    # A candidate order can be withheld when comma-separated fragments form
    # one prose passage that must not be split by the move.
    reordered_detail: str | None = None
    # The categories themselves, as the rules placed the items with them,
    # for the critic to hold a reorder of its own to the same placement.
    # Never sent: the page has the chips' own lookup for what a tag is.
    categories: dict[str, str] | None = Field(default=None, exclude=True)
    # The items the tag index was asked about, answered or not: what the
    # categories above answer for. The critique route checks the rewrite's
    # artists with them, and says whether they cover it. Never sent.
    categories_asked: list[str] = Field(default_factory=list, exclude=True)

    @computed_field
    @property
    def categories_unread(self) -> bool:
        """Whether the lookup gave no category at all - the index unreachable,
        or no usable reply - where `categories_detail` says why: no item was
        checked, not only some (the cap). The page's artist note says "No tag
        was checked" for this, "Not every tag" past the cap."""
        return self.categories is None and self.categories_detail is not None

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
    # Whether the rewrite's artist check (Anima's @ rule) read every artist
    # the rewrite may name (`prompt_rules.artists_read`): each item a name
    # could be, a clause included, was asked by this critique's own analysis,
    # whose categories the rewrite is checked with, no lookup of its own; and
    # no artist that answer named stands in the rewrite without its @ other
    # than as a whole item the rule reads. Then an artist finding of the
    # prompt's that the rewrite no longer has is fixed, not merely unseen;
    # False where any item - an artist the critic added - was never looked
    # up, the name was kept inside a clause or a sentence, or the lookup gave
    # nothing, and the page keeps its "may still apply".
    rewrite_artists_checked: bool = False
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


class TagSuggestion(BaseModel):
    text: str
    canonical: str
    category: str
    post_count: int | None = None


class TagSuggestionResponse(BaseModel):
    available: bool
    suggestions: list[TagSuggestion] = Field(default_factory=list)
    detail: str | None = None


class TagDescription(TagSuggestion):
    # The tag's Danbooru wiki, its first sentence: how the finder's reader
    # tells `ahoge` from a tag whose page only mentions the words.
    definition: str | None = None


class TagDescriptionResponse(BaseModel):
    available: bool
    results: list[TagDescription] = Field(default_factory=list)
    detail: str | None = None


# The longest prompt a Check reads (`PromptAnalysisRequest.prompt`), held
# equal to it by a test: the chips' order slots are Check's order reading.
TAG_SLOT_PROMPT_MAX = 4000


class TagValidationRequest(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=64)
    # The whole prompt the chips were split from, for their order slots
    # (`TagValidationResponse.slots`). Optional - the Refine dialog's unknown
    # tags line asks about tags alone - and never a reason to refuse the
    # verdicts: a prompt longer than a Check reads, or blank, is no prompt
    # here, and the answer has no slots.
    prompt: str | None = None

    @field_validator("tags")
    @classmethod
    def _usable_tags(cls, values: list[str]) -> list[str]:
        # Any comma item of a prompt: one longer than a tag - a hybrid
        # prompt's sentence - is answered unknown without asking the index
        # (`tags.TAG_NAME_MAX`). Bounded at 200, the whole request was refused
        # for one such item, and the tray said Danbooru was unreachable.
        tags = [value.strip() for value in values]
        if any(not tag or len(tag) > TAG_SLOT_PROMPT_MAX for tag in tags):
            raise ValueError(f"tags must be 1-{TAG_SLOT_PROMPT_MAX} characters after trimming")
        return tags

    @field_validator("prompt")
    @classmethod
    def _readable_prompt(cls, value: str | None) -> str | None:
        if value is None or not value.strip() or len(value) > TAG_SLOT_PROMPT_MAX:
            return None
        return value


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
    # the index never had. None for a known or unchecked tag, but for
    # `grid_variable`: a grid placeholder (`{hair}`), given no verdict and
    # never asked about, which the page draws as the grid's, not as a tag.
    code: Literal[
        "not_found", "deprecated", "empty", "no_date", "after_cutoff", "grid_variable"
    ] | None = None


class TagSlot(BaseModel):
    """One comma item of the prompt and the step of the guide's documented
    order the order rule places it in (`prompt_rules.order_slots`), or None:
    lettering, or an item with no one place - a tag the index gave no
    category, one never asked about."""

    tag: str
    slot: str | None = None


class PromptPart(BaseModel):
    """A contextual tag or prose passage beside strict dictionary verdicts.

    ``first``/``last`` are inclusive original comma-item indexes. ``start``
    and ``end`` slice the exact raw prompt in JavaScript UTF-16 code units.
    """

    kind: Literal["tag", "prose"]
    first: int
    last: int
    start: int
    end: int


class TagValidationResponse(BaseModel):
    available: bool
    results: list[TagValidationItem] = Field(default_factory=list)
    detail: str | None = None
    # Whole matching prompt context, on explicitly hybrid guides only.
    # Lexical results above remain strict and independently cacheable.
    parts: list[PromptPart] | None = None
    # With a `prompt` in the request, and only where the rules place its items
    # (`prompt_rules.order_slots`; a lone item too): the documented order's steps and
    # every comma item of the prompt - past the lookup's 64 too, where a guide
    # tag still has its slot - index for index with the chips. None where the
    # guide states no order, the rules place nothing in this prompt, or no prompt
    # was sent.
    order_steps: list[str] | None = None
    slots: list[TagSlot] | None = None
