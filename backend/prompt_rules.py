"""The analyser's rule checks: a prompt held against its checkpoint's guide.

Each guide in `guides.py` carries its checkable statements as `analysis`
data. This module reads the text the way those statements talk about it -
sentences or a tag list, what is quoted, which comma item belongs in which
slot of a documented order - and turns each mismatch into a `PromptFinding`
that names what it rests on. `guide` findings quote the guide's own prose
(`cite`, verbatim) with the source's URL, `author` ones an author's example,
and `heuristic` ones are this module's pattern matching and say so: a check
with no author sentence behind it is labelled, never passed off as advice the
checkpoint's authors gave.

Nothing here reads a tokenizer, loads a model or reaches the network - what
the encoders do to the text is `prompt_encoding`'s. What a comma item *is* in
Danbooru's vocabulary (character, series, artist) is the caller's to supply
as `categories`, from `tags.tag_vocabulary`; without it the order check
places only what the guide's own tag groups know, and says nothing it would
have to guess.

A finding's `suggestion` is always a whole replacement for its field's text,
built mechanically from that text - reordered, suffixed, quoted, joined -
so the page can offer it as one click. Advice that cannot be applied that
way (what to write instead of "no people") stays in the message.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, NamedTuple
from xml.etree import ElementTree

from .backends.netayume_loading import SYSTEM_PROMPT as NETAYUME_SYSTEM_PROMPT

# The profile table's own first-match lookup, so a checkpoint's rules are
# exactly the guide its hints and Examples menu come from.
from .model_defaults import _profile_for
from .schemas import DocumentedOrder, GenerateRequest, PromptFinding

Style = Literal["prose", "tags", "hybrid", "xml", "json"]

# Past this many comma items the order check stands down rather than run a
# quadratic placement over what is no longer a tag list anyone typed.
_MAX_ORDERED = 256


# ------------------------------------------------------------------ reading the text


class Item(NamedTuple):
    """One comma item: its text as typed, stripped, and where it sits."""

    text: str
    start: int
    end: int


def comma_items(prompt: str) -> list[Item]:
    """The prompt's comma items, split the way the page's tag chips split it.

    `categories` is keyed by these texts, so the caller that builds it from
    the tag lookup and the rules that read it agree on what an item is.
    """
    items = []
    for match in re.finditer(r"[^,]+", prompt):
        raw = match.group()
        text = raw.strip()
        if text:
            start = match.start() + len(raw) - len(raw.lstrip())
            items.append(Item(text, start, start + len(text)))
    return items


# Ideographs and kana count one word each, the way word processors count
# them: Chinese and Japanese put no spaces between words. Hangul does, so it
# counts like Latin.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def word_count(prompt: str) -> int:
    return len(_CJK.findall(prompt)) + len(_WORD.findall(_CJK.sub(" ", prompt)))


_XML_ELEMENT = re.compile(r"<([A-Za-z_][\w.\-]*)>.*?</\1\s*>", re.S)
_XML_START = re.compile(r"<[A-Za-z_][\w.\-]*>")
# One sentence ending and another starting inside a comma item - "embroidery.
# Impeccable makeup" - which no tag does. Only before a capital: "dr. stone"
# and "love live! school idol project" are Danbooru series. A decimal point
# and a one- or two-letter abbreviation ("Dr. Stone", "St. Ives") are not one.
_SENTENCE_BREAK = re.compile(
    r"(?<!\d)(?<!\b[^\W\d_])(?<!\b[^\W\d_]{2})[.!?]\s+(?=[A-ZÀ-ÖØ-Þ])|[。！？](?=\S)"
)
_SENTENCE_END = re.compile(r"[.!?。！？][\"'”’)\]]*\s*$")
_ARTICLE = re.compile(r"(?i)^(?:a|an|the)\b")
# Words a sentence cannot do without and a tag list almost never has.
_FUNCTION_WORDS = frozenset(
    "a an the of with and or is are was on in at to from by into onto its their".split()
)
# What makes a comma item read as a sentence. Articles, which a Danbooru
# tag drops ("holding sign") but a title can keep ("the legend of zelda"),
# and clause words - pronouns, be and have - which tags never carry: they
# write "hand on own chest" where a caption says "her". A preposition marks
# a phrase only in a longer piece, since short tags use them freely (looking
# at viewer, hand on hip); "to" is left out for the romanised particle that
# opens "to aru majutsu no index".
_ARTICLES = frozenset(("a", "an", "the"))
_CLAUSE_WORDS = frozenset(
    "he she it they him her his its their them is are was were be been being "
    "has have had".split()
)
_LINKING_WORDS = frozenset(
    "of with and or on in at from by into onto over under through toward towards "
    "across along behind beneath above below between among around near inside while".split()
)
# Words past which an item is no Danbooru name: the longest romanised
# series titles run to about ten ("ore no imouto ga konna ni kawaii wake ga nai").
_LONGEST_TAG = 10


def _lower_words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(_CJK.sub(" ", text))]


def _is_prose(text: str) -> bool:
    """A comma item that reads as a phrase or a sentence rather than a tag.

    Not by length alone: "jojo no kimyou na bouken" and "march 7th (honkai:
    star rail)" are five-word Danbooru names. It takes a sentence break, the
    article or clause word a sentence is built with, or - in a piece of five
    words or more - a preposition or conjunction linking them ("expressive
    eyes with detailed reflections"). Ideographs carry none of those signs,
    so a Chinese or Japanese piece counts by length.
    """
    if _SENTENCE_BREAK.search(text):
        return True
    count = word_count(text)
    if _CJK.search(text):
        return count >= 5
    words = _lower_words(text)
    if count >= 3 and any(word in _ARTICLES or word in _CLAUSE_WORDS for word in words):
        return True
    return count >= 5 and any(word in _LINKING_WORDS for word in words)


def _is_sentence(text: str) -> bool:
    """Prose no Danbooru name could be, so a tag lookup could not know it.

    Stricter than `_is_prose`: a title keeps its articles ("the king of
    fighters"), and only a clause word, a sentence break, a closing full stop
    after an opening article or more words than any tag runs to rule a name out.
    """
    if not _is_prose(text):
        return False
    return (
        bool(_SENTENCE_BREAK.search(text))
        or word_count(text) > _LONGEST_TAG
        or any(word in _CLAUSE_WORDS for word in _lower_words(text))
        or bool(_ARTICLE.match(text) and _SENTENCE_END.search(text))
    )


def detect_style(prompt: str, categories: dict[str, str] | None = None) -> Style:
    """How the text reads: sentences, a comma tag list, both, XML or JSON.

    By the share of its words that sit in phrase-length comma pieces: a
    caption's list of attributes ("black horns, and a long, sinuous tail")
    makes short pieces too, so a mixed text that ends like a sentence and is
    full of articles and prepositions is still prose. With `categories` -
    `check`'s own - a piece the tag index knows reads as a tag, so the style
    reported is the one the rules applied.
    """
    return _detect_style(prompt, _category_lookup(categories))


def _detect_style(prompt: str, known: dict[str, str] | None) -> Style:
    """`detect_style`, with the pieces the tag index knows read as tags
    whatever their shape: "the legend of zelda" is a series, not a phrase."""

    def listed(piece: str) -> bool:
        return bool(known) and (piece in known or _norm(piece) in known)

    def phrase(piece: str) -> bool:
        return _is_prose(piece) and not listed(piece)

    text = prompt.strip()
    if text[:1] in ("{", "["):
        # A few thousand nested brackets fit inside the 4000-character limit
        # and exhaust the parser's recursion before it can say "not JSON".
        try:
            parsed = json.loads(text)
        except (ValueError, RecursionError):
            parsed = None
        if isinstance(parsed, (dict, list)) and parsed:
            return "json"
    if _XML_ELEMENT.search(text) or _XML_START.match(text):
        return "xml"
    pieces = [piece.strip() for piece in re.split(r"[,，\n]", text) if piece.strip()]
    if len(pieces) <= 1:
        single = pieces[0] if pieces else ""
        tag = listed(single) or (
            single
            and not _is_prose(single)
            and word_count(single) <= 3
            and not _ARTICLE.match(single)
        )
        return "tags" if tag else "prose"
    phrase_words = tag_words = 0
    for piece in pieces:
        words = word_count(piece)
        if phrase(piece):
            phrase_words += words
        else:
            tag_words += words
    total = phrase_words + tag_words
    if not total:
        return "prose"
    share = phrase_words / total
    if share >= 0.75:
        return "prose"
    if share <= 0.25:
        return "tags"
    words = [word.lower() for word in _WORD.findall(_CJK.sub(" ", text))]
    function = sum(1 for word in words if word in _FUNCTION_WORDS)
    if words and function / len(words) >= 0.15 and _SENTENCE_END.search(text):
        return "prose"
    return "hybrid"


# Quoted stretches, in the marks a prompt uses for text to render. A single
# quote counts only as a pair that does not touch a word, so "animal's" is
# an apostrophe and 'Black Forest Labs' a quotation. The one definition: the
# critic's reorder and rewrite checks and the encoder layer's weighting scan
# read quoted text with it too.
QUOTED = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|„[^“”\n]*[“”]|「[^」\n]*」|『[^』\n]*』|«[^»\n]*»|‘[^’\n]*’'
    r"|(?<![\w'])'[^'\n]+'(?![\w'])"
)
_QUOTE_OPEN = "\"'“„「『«‘"


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in QUOTED.finditer(text)]


def _inside(spans: list[tuple[int, int]], position: int) -> bool:
    return any(start <= position < end for start, end in spans)


def _norm(tag: str) -> str:
    """A comma item as comparable text: case, underscores, the escaped
    parentheses the tag autocomplete writes and a sentence's closing full stop
    ("..., masterpiece.") do not make it another tag."""
    text = tag.strip().lower().replace(r"\(", "(").replace(r"\)", ")").replace("_", " ")
    return re.sub(r"(?<=\w)\.$", "", re.sub(r"\s+", " ", text).strip())


_YEAR_TAG = re.compile(r"year \d{4}")


def _separator(prompt: str) -> str:
    """The comma style the prompt already uses, for anything added to it."""
    commas = re.findall(r",([ \t]?)", prompt)
    spaced = sum(1 for gap in commas if gap)
    return ", " if not commas or spaced * 2 >= len(commas) else ","


def _rebuild(prompt: str, items: list[Item], texts: list[str]) -> str:
    """The prompt with its comma items replaced by `texts`, in that order.

    Whatever sat between the original items - ", ", ",\\n", a stray double
    space - stays where it was, position for position, so a reorder changes
    the order and nothing else. Items beyond the original count take the
    prompt's own separator; fewer items drop the trailing separators.
    """
    if not items:
        return prompt
    gaps = [prompt[a.end:b.start] for a, b in pairwise(items)]
    default = _separator(prompt)
    out = [prompt[: items[0].start]]
    for index, text in enumerate(texts):
        if index:
            out.append(gaps[index - 1] if index - 1 < len(gaps) else default)
        out.append(text)
    out.append(prompt[items[-1].end:])
    return "".join(out)


# ------------------------------------------------------------------ the guide


def _guide(model_id: str) -> dict[str, Any]:
    return _profile_for(model_id)["guide"]


def _analysis(guide: dict[str, Any]) -> dict[str, Any]:
    return guide.get("analysis") or {}


class _OrderSpec(NamedTuple):
    """A tag checkpoint's documented order, as slots a comma item can take.

    `local` is what the guide's own tag groups, suffix and year form already
    place; `categories` is what a Danbooru category places; `unknown` is
    every slot an item with no category could still belong to - the range
    within which the rules may not move it.
    """

    steps: list[str]
    local: dict[str, int]
    year: int | None
    categories: dict[str, int]
    general: int
    unknown: frozenset[int]


def _order_spec(guide: dict[str, Any]) -> _OrderSpec | None:
    order = _analysis(guide).get("order")
    if not order or "general" not in order:
        return None
    helper = guide.get("tag_helper") or {}
    steps = list(order.get("steps") or helper.get("order") or [])
    index = {label: position for position, label in enumerate(steps)}
    local: dict[str, int] = {}
    for group in helper.get("groups", []):
        slot = order.get("groups", {}).get(group["label"])
        if slot is not None:
            for tag in group["tags"]:
                local.setdefault(_norm(tag), index[slot])
    if order.get("suffix"):
        for tag in helper.get("recommended_suffix", []):
            local.setdefault(_norm(tag), index[order["suffix"]])
    categories = {name: index[slot] for name, slot in order.get("categories", {}).items()}
    general = index[order["general"]]
    return _OrderSpec(
        steps=steps,
        local=local,
        year=index[order["year"]] if order.get("year") else None,
        categories=categories,
        general=general,
        unknown=frozenset(categories.values()) | {general},
    )


def _category_lookup(categories: dict[str, str] | None) -> dict[str, str] | None:
    if categories is None:
        return None
    lookup = {_norm(text): category for text, category in categories.items() if category}
    lookup.update({text: category for text, category in categories.items() if category})
    return lookup


def _place(item: Item, spec: _OrderSpec, lookup: dict[str, str] | None) -> frozenset[int]:
    """The slots `item` may occupy: one when anything says which, else all
    the slots a Danbooru category could have put it in.

    What is known comes first - the guide's own groups, the year form, then
    the index's category - so a long name the index knows ("jojo no kimyou
    na bouken", a copyright) is never taken for prose by its shape.
    """
    key = _norm(item.text)
    if key in spec.local:
        return frozenset({spec.local[key]})
    if spec.year is not None and _YEAR_TAG.fullmatch(key):
        return frozenset({spec.year})
    if lookup is not None:
        category = lookup.get(item.text) or lookup.get(key)
        if category in spec.categories:
            return frozenset({spec.categories[category]})
    # Prose sits in the general slot on every hybrid checkpoint's order -
    # Kohaku's is literally "general tags or prose". A prose-shaped item a
    # name could still be ("the legend of zelda") counts as prose only once
    # the index has been asked and did not know it.
    if _is_sentence(item.text) or (lookup is not None and _is_prose(item.text)):
        return frozenset({spec.general})
    return spec.unknown


def _ordered(places: list[frozenset[int]]) -> list[int]:
    """The stable order in which nothing follows an item that must come after it.

    An item must precede another only when every slot it could take comes
    before every slot the other could take. Repeatedly taking the earliest
    item nothing has to precede keeps everything else where it was: known
    items move as far as the order needs, unknown ones only as far as the
    known items around them require, and a slot's items keep their order.
    """
    low = [min(place) for place in places]
    high = [max(place) for place in places]
    remaining = list(range(len(places)))
    order: list[int] = []
    while remaining:
        first = second = float("inf")
        count = 0
        for index in remaining:
            value = high[index]
            if value < first:
                first, second, count = value, first, 1
            elif value == first:
                count += 1
            elif value < second:
                second = value
        for position, index in enumerate(remaining):
            others = second if high[index] == first and count == 1 else first
            if low[index] <= others:
                order.append(index)
                del remaining[position]
                break
    return order


def _violation(places: list[frozenset[int]]) -> tuple[int, int] | None:
    """The first pair written out of the documented order, beyond doubt."""
    low = [min(place) for place in places]
    high = [max(place) for place in places]
    for a in range(len(places)):
        for b in range(a + 1, len(places)):
            if low[a] > high[b]:
                return a, b
    return None


# ------------------------------------------------------------------ public


def documented_order(model_id: str) -> DocumentedOrder | None:
    """The order the checkpoint's guide states, or None.

    None is an answer, not a gap: the guide prose of Krea 2, Qwen-Image,
    Z-Image, ERNIE, Z-Anime, Anima and NetaYume states no order, so this app
    checks none, and the page says that rather than inventing one - showing,
    as information, any order the authors' own texts give (Anima's card, the
    rewriter instructions; `prompt_sources`' `order` excerpts).
    """
    guide = _guide(model_id)
    analysis = _analysis(guide)
    order = analysis.get("order")
    if not order:
        return None
    steps = list(order.get("steps") or (guide.get("tag_helper") or {}).get("order") or [])
    if not steps:
        return None
    return DocumentedOrder(
        steps=steps, cite=order["cite"], url=order.get("url") or analysis.get("url")
    )


def order_checked(model_id: str, prompt: str) -> bool:
    """Whether the rules hold this prompt to the documented order at all.

    Only a tag checkpoint's order is checkable - its comma items take slots
    (`_OrderSpec`) - and only on a text that is not XML or JSON and not past
    `_MAX_ORDERED` items. A FLUX sentence's "subject first", klein's four
    steps and NewBie's blocks are documented but read by nothing here, and the
    page must not let an empty findings list pass for a checked order.
    """
    if _order_spec(_guide(model_id)) is None or detect_style(prompt) in ("xml", "json"):
        return False
    return len(comma_items(prompt)) <= _MAX_ORDERED


def reorder_tags(model_id: str, prompt: str, categories: dict[str, str] | None) -> str | None:
    """The same comma items, same spelling, moved into the documented order.

    None where the checkpoint documents no tag order, the text is XML or
    JSON, or nothing has to move. Prose pieces all sit in the general slot
    and keep their order, so a caption that ends in tags - Illustrious's own
    natural-language example does - moves only the tags. Nothing is added or
    removed, and the text between items stays where it was, so the result is
    a pure permutation: the variant that isolates order on a comparison grid.
    """
    spec = _order_spec(_guide(model_id))
    if spec is None or detect_style(prompt) in ("xml", "json"):
        return None
    items = comma_items(prompt)
    if not 2 <= len(items) <= _MAX_ORDERED:
        return None
    lookup = _category_lookup(categories)
    order = _ordered([_place(item, spec, lookup) for item in items])
    if order == list(range(len(items))):
        return None
    return _rebuild(prompt, items, [items[index].text for index in order])


def wants_artist_categories(model_id: str) -> bool:
    """Whether a rule reads which comma items are artists - Anima's "put @
    before an artist name" - so the caller asks the tag index even on a guide
    with no tag helper of its own."""
    return "artist_prefix" in _analysis(_guide(model_id))


def category_items(model_id: str, prompt: str) -> list[str]:
    """The comma items whose Danbooru category would change a finding.

    Empty where no rule reads a category, or the text is XML or JSON - so a
    caller asks the tag index only for what the order and artist checks need.
    Every item a Danbooru name could be is asked about, a long or dotted one
    ("dr. stone") and one with an article ("the king of fighters") included,
    since the category outranks the item's shape; only sentences no name
    could be are left out, and items the guide's own groups already place,
    which need no lookup.
    """
    guide = _guide(model_id)
    spec = _order_spec(guide)
    placing = spec is not None and any(slot != spec.general for slot in spec.categories.values())
    if not (placing or "artist_prefix" in _analysis(guide)):
        return []
    if detect_style(prompt) in ("xml", "json"):
        return []
    wanted: list[str] = []
    for item in comma_items(prompt):
        key = _norm(item.text)
        if _is_sentence(item.text) or item.text.startswith("@") or item.text in wanted:
            continue
        if spec is not None and (
            key in spec.local or (spec.year is not None and _YEAR_TAG.fullmatch(key))
        ):
            continue
        wanted.append(item.text)
    return wanted


@dataclass
class _Context:
    """One prompt, read once, for every rule."""

    guide: dict[str, Any]
    analysis: dict[str, Any]
    request: GenerateRequest
    applied: GenerateRequest
    lookup: dict[str, str] | None
    text: str
    style: Style
    items: list[Item]
    # Each item's comparable form, index for index.
    norms: list[str]
    quoted: list[tuple[int, int]]
    # Words that read as text to render without quotation marks: "NO
    # PARKING" on a sign is lettering to quote, not an absence to rephrase.
    rendered: list[tuple[int, int]]

    def entry(self, key: str) -> dict[str, Any] | None:
        entry = self.analysis.get(key)
        return entry if isinstance(entry, dict) else None

    def finding(
        self,
        entry: dict[str, Any],
        rule: str,
        severity: Literal["info", "warn"],
        message: str,
        *,
        start: int | None = None,
        end: int | None = None,
        suggestion: str | None = None,
        field: Literal["prompt", "negative_prompt"] = "prompt",
    ) -> PromptFinding:
        return PromptFinding(
            rule=rule,
            severity=severity,
            message=message,
            field=field,
            start=start,
            end=end,
            source=entry.get("source", "guide"),
            cite=entry["cite"],
            url=entry.get("url") or self.analysis.get("url"),
            suggestion=suggestion if suggestion != self._field_text(field) else None,
        )

    def _field_text(self, field: str) -> str | None:
        return self.text if field == "prompt" else self.request.negative_prompt

    @property
    def taglike(self) -> bool:
        return self.style in ("tags", "hybrid")


def check(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    categories: dict[str, str] | None = None,
) -> list[PromptFinding]:
    """Every rule the checkpoint's guide supports, run over `request.prompt`.

    `applied` is the request as the pipeline will run it: a negative prompt
    `applied()` dropped never runs, and that - not the box being filled in -
    is what decides whether a negation in the prompt has anywhere else to go.
    `categories` maps a comma item, as typed, to its Danbooru category; None
    means the lookup was unavailable, and the order checks then place only
    what they know. Warnings first, then by position.
    """
    guide = _guide(model_id)
    text = request.prompt
    items = comma_items(text)
    quoted = _quoted_spans(text)
    lookup = _category_lookup(categories)
    context = _Context(
        guide=guide,
        analysis=_analysis(guide),
        request=request,
        applied=applied,
        lookup=lookup,
        text=text,
        style=_detect_style(text, lookup),
        items=items,
        norms=[_norm(item.text) for item in items],
        quoted=quoted,
        rendered=_unquoted_text(text, quoted),
    )
    findings: list[PromptFinding] = []
    for rule in _RULES:
        findings.extend(rule(context))
    return sorted(
        findings,
        key=lambda f: (f.severity != "warn", f.start is None, f.start or 0),
    )


# ------------------------------------------------------------------ the rules

Rule = Callable[[_Context], Iterable[PromptFinding]]


def _style(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("style")
    if entry is None or ctx.style in entry.get("accepts", ()):
        return
    expects = entry["expects"]
    if expects == "prose" and ctx.style == "tags" and len(ctx.items) >= 3:
        yield ctx.finding(
            entry, "guide.style", "info",
            "This reads as a comma-separated tag list; this checkpoint's guide asks "
            "for descriptive sentences.",
        )
    elif expects == "tags" and ctx.style in ("prose", "hybrid"):
        what = "natural-language prose" if ctx.style == "prose" else "partly prose"
        yield ctx.finding(
            entry, "guide.style", "info",
            f"This reads as {what}; this checkpoint's card asks for comma-separated "
            "Danbooru tags.",
        )


def _words(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("words")
    if entry is None or ctx.style in ("xml", "json"):
        return
    low, high = entry["range"]
    count = word_count(ctx.text)
    if not low <= count <= high:
        side = "under" if count < low else "over"
        yield ctx.finding(
            entry, "guide.words", "info",
            f"{count} words, {side} the {low}-{high} the guide suggests for most images.",
        )


# BFL's replacement table ("Working Without Negative Prompts") and the FLUX.2
# guide's "no blur", for the phrases they name exactly - their idea, quoted,
# never one this module makes up.
_BFL_POSITIVE = {
    "no people": '"empty", "deserted" or "solitary"',
    "no colors": '"monochrome", "black and white" or "grayscale"',
    "no text": '"clean surfaces", "unmarked" or "blank"',
    "no modern elements": '"traditional", "historical" or "period-accurate"',
    "not dark": '"brightly lit" or "sun-drenched"',
    "not sad": '"joyful" or "content"',
    "not running": '"walking peacefully" or "standing still"',
    "not many": '"few", "single" or "minimal"',
    "no crowds": '"peaceful solitude" or "empty pathways"',
    "no blur": '"sharp focus throughout"',
}
_NEGATION = re.compile(
    r"\b(?P<head>do\s+not|does\s+not|don['’]t|doesn['’]t|without|never|avoid(?:ing)?|no|not)"
    r"(?:[ \t]+[^\s,.;:!?()\[\]\"“”]+){1,3}",
    re.IGNORECASE,
)
# Words a negated phrase does not end on: "no people in the background" is
# about the people.
_TRAILING = frozenset(
    "a an the in on at of to with and or for from by into onto over under any is are".split()
)
_IDIOMS = ("no longer", "no matter", "no doubt", "no less", "no more than", "not only", "not just")
# A lowercase "no" between two capitalised words is the Japanese particle of
# a romanised title or name - Sousou no Frieren, Kimi no Na wa - not a negation.
_WORD_BEFORE = re.compile(r"([^\W\d_][\w'’-]*)[ \t]+$")
_CAPITAL_AFTER = re.compile(r"[ \t]+[A-ZÀ-ÖØ-Þ]")


def _particle(text: str, match: re.Match[str]) -> bool:
    if match.group("head") != "no":
        return False
    before = _WORD_BEFORE.search(text, max(0, match.start() - 64), match.start())
    return bool(
        before and before.group(1)[0].isupper() and _CAPITAL_AFTER.match(text, match.end("head"))
    )


def _negation(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("negation")
    if entry is None:
        return
    runs = ctx.applied.negative_prompt is not None
    for match in _NEGATION.finditer(ctx.text):
        # Quoted or unquoted, lettering on a sign ("NO PARKING") is text to
        # render - the quote rule's business - and not an absence.
        if _inside(ctx.quoted, match.start()) or _inside(ctx.rendered, match.start()):
            continue
        if _particle(ctx.text, match):
            continue
        head = len(match.group("head").split())
        words = list(re.finditer(r"\S+", match.group()))
        while len(words) > head and words[-1].group().lower() in _TRAILING:
            words.pop()
        lowered = " ".join(word.group().lower() for word in words)
        if len(words) <= head or lowered.startswith(_IDIOMS):
            continue
        start, end = match.start(), match.start() + words[-1].end()
        phrase = ctx.text[start:end]
        if runs:
            message = (
                f'"{phrase}" asks for an absence in the prompt itself, where a named '
                "thing tends to be drawn anyway. Describe what should be there; "
                "exclusions belong in the negative prompt, which runs here."
            )
        else:
            message = (
                f'"{phrase}" asks for an absence, and no negative prompt runs with '
                "these settings: a named thing tends to be drawn anyway. Describe "
                "what should be there instead."
            )
        replacement = _BFL_POSITIVE.get(lowered) if entry.get("replacements") == "bfl" else None
        if replacement:
            message += f' BFL\'s guide writes {replacement} for "{lowered}".'
        yield ctx.finding(
            entry, "guide.negation", "info" if runs else "warn", message,
            start=start, end=end,
        )


# `verb` triggers are ordinary verbs that only now and then introduce
# lettering; "titled", "spells out" and the nouns always do.
_TEXT_TRIGGER = re.compile(
    r"\b(?:(?P<verb>reads|reading|says|saying|said|labell?ed(?:[ \t]+as)?"
    r"|spells(?![ \t]+out\b)|spelling(?![ \t]+out\b))"
    r"|titled|entitled|spells[ \t]+out|spelling[ \t]+out"
    r"|(?P<noun>(?:the|with)[ \t]+(?:words?|text|letters|lettering|phrase|slogan|caption"
    r"|headline|inscription)))\b[ \t]*:?[ \t]*",
    re.IGNORECASE,
)
_CJK_TEXT_TRIGGER = re.compile(r"(?:写着|写有|印着|印有|标着|标有)[:：]?")
# A verb trigger is text only on something that carries text, earlier in
# the same sentence: a sign reading OPEN, a mug that says MONDAY, a jar
# labelled SUGAR - not a girl reading a book, a mother saying goodbye or a
# wizard casting spells. Places a speaker stands (a door, a window, a wall)
# and clothes other than the shirt are left out: "a man in a hat says hello"
# is speech.
_TEXT_BEARER = re.compile(
    r"(?i)\b(?:signs?|signboard|signpost|board|banner|poster|flyer|leaflet|label|screen"
    r"|display|plaque|card|postcard|note|letter|envelope|message|text|tag|shirt|t-shirt"
    r"|mug|cup|bottle|jar|box|package|crate|vial|potion|cover|book|notebook|magazine"
    r"|newspaper|billboard|marquee|neon|letters|lettering|headline|caption|title|logo"
    r"|placard|chalkboard|whiteboard|blackboard|menu|sticker|badge|page|paper|scroll"
    r"|monitor|tattoo|cake|bubble|balloons?|flag|stamp|ticket|receipt|certificate|trophy"
    r"|tombstone|gravestone|headstone|doormat|graffiti)(?:e?s)?\b[^.!?\n]*$"
)
# "labeled" as an adjective - "a diagram with labeled parts", "a labeled
# map" - names no text; after a noun it names the text: a jar labeled SUGAR.
_ATTRIBUTIVE = re.compile(r"(?i)\b(?:a|an|the|with|and|or|of|its|their|some|many)[ \t]+$")
# After "the text", a word that makes it a noun rather than a quotation:
# "the text is blurry".
_NOT_TEXT = frozenset(
    "is are was were be been appears appear reads says in on at of that which should "
    "will can has have and or with from to it its below above".split()
)
# Words that end the text to render: where it is and how it is written.
_TARGET_STOP = frozenset(
    "in on at above below under over beside next while written painted printed "
    "across along near behind inside appears glowing".split()
)
# Words that may end it or belong to it: "says OPEN and a cat sleeps" against
# "says Fish and Chips", "reads Welcome to Paradise". A capital after one
# carries the text on, as a name or a title would; anything else ends it
# there, and where it ends is then a guess.
_TARGET_JOIN = frozenset("and & with as from for to".split())
_TARGET_WORDS = 6
_TARGET_WORD = re.compile(r"[ \t]*([^\s,.;:!?\"“”()]+)")
_CJK_TARGET = re.compile(r"[^\s，。、；！？,.;!?\"“”「」『』]{1,12}")


class _Target(NamedTuple):
    """Words that read as text to render, and whether their extent is sure."""

    start: int
    end: int
    # False where the words run on past a joining word or the word limit and
    # the text's end is a guess: the finding then offers no one-click quote,
    # which would letter half a name.
    exact: bool


def _capitalised(word: str) -> bool:
    return word[:1].isupper() or word[:1].isdigit()


def _unquoted_targets(text: str, quoted: list[tuple[int, int]]) -> list[_Target]:
    """Words that read as text to render but carry no quotation marks."""
    targets: list[_Target] = []
    for match in _TEXT_TRIGGER.finditer(text):
        if _inside(quoted, match.start()):
            continue
        before = text[max(0, match.start() - 40):match.start()]
        verb = match.group("verb")
        if verb and not _TEXT_BEARER.search(before):
            continue
        if verb and verb.lower().startswith("label") and _ATTRIBUTIVE.search(before):
            continue
        position = match.end()
        if position >= len(text) or text[position] in _QUOTE_OPEN:
            continue
        # To punctuation, to the preposition that starts describing where the
        # text is (says hello world | in red letters), or to a joining word
        # with no capital after it (says Fish and Chips | and a cat).
        words: list[re.Match[str]] = []
        cursor = position
        exact = True
        while True:
            word = _TARGET_WORD.match(text, cursor)
            if word is None:
                break
            token = word.group(1).lower()
            if (words and token in _TARGET_STOP) or (not words and token in _NOT_TEXT):
                break
            if len(words) == _TARGET_WORDS:
                exact = False
                break
            if words and token in _TARGET_JOIN:
                after = _TARGET_WORD.match(text, word.end())
                spaced = text[word.end():word.end() + 1] in (" ", "\t")
                if not (spaced and after and _capitalised(after.group(1))):
                    # Capitals up to here are the lettering (says OPEN and a
                    # cat); lowercase words may run on (says fish and chips).
                    exact = all(w.group(1).isupper() for w in words)
                    break
            words.append(word)
            cursor = word.end()
            if cursor < len(text) and text[cursor] not in " \t":
                break
        # A joining word never ends the text: "says Fish and" is not a name.
        while words and words[-1].group(1).lower() in _TARGET_JOIN:
            words.pop()
        if words:
            targets.append(_Target(words[0].start(1), words[-1].end(1), exact))
    for match in _CJK_TEXT_TRIGGER.finditer(text):
        if _inside(quoted, match.start()):
            continue
        target = _CJK_TARGET.match(text, match.end())
        if target and text[match.end()] not in _QUOTE_OPEN:
            targets.append(_Target(*target.span(), True))
    return sorted(targets)


def _unquoted_text(text: str, quoted: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Spans of words that read as text to render but carry no quotation marks."""
    return [(target.start, target.end) for target in _unquoted_targets(text, quoted)]


def rendered_spans(text: str) -> list[tuple[int, int]]:
    """Where `text` asks for lettering: its quoted stretches, marks included,
    and the words a rule reads as unquoted text to render. The critic's checks
    hold its `text_to_render` and the rewrite's quotation marks to these."""
    quoted = _quoted_spans(text)
    return sorted(quoted + _unquoted_text(text, quoted))


def _quote_text(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("quote_text")
    if entry is None or ctx.style in ("xml", "json"):
        return
    unquoted = ctx.rendered
    if entry.get("supported") is False:
        spans = sorted(ctx.quoted + unquoted)
        if spans:
            start, end = spans[0]
            yield ctx.finding(
                entry, "guide.quote_text", "info",
                f"{ctx.text[start:end]} reads as text to render, which this "
                "checkpoint's card says it does not do.",
                start=start, end=end,
            )
        return
    for start, end, exact in _unquoted_targets(ctx.text, ctx.quoted):
        target = ctx.text[start:end]
        message = (
            f'"{target}" reads as text to render but is not in quotation marks, so '
            "the model may treat it as description instead of lettering."
        )
        if not exact:
            # "says fish and chips for sale": where the lettering ends is a
            # guess, and quoting a guess would letter half of it.
            message += " Put the exact words to render in quotation marks."
        yield ctx.finding(
            entry, "guide.quote_text", "info", message,
            start=start, end=end,
            suggestion=f'{ctx.text[:start]}"{target}"{ctx.text[end:]}' if exact else None,
        )


def _one_paragraph(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("one_paragraph")
    if entry is None:
        return
    text = ctx.text
    blank = re.search(r"\n[ \t]*\n", text)
    bullets = re.findall(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+\S", text)
    if len(bullets) >= 2:
        yield ctx.finding(
            entry, "guide.one_paragraph", "info",
            "This is a list; the guide asks for one cohesive paragraph.",
        )
    elif blank:
        yield ctx.finding(
            entry, "guide.one_paragraph", "info",
            "A blank line splits this into paragraphs; the guide asks for one.",
            start=blank.start(), end=blank.end(),
            suggestion=re.sub(r"\s*\n\s*", " ", text.strip()),
        )


_PROMPT_START = re.compile(r"<\s*prompt\s+start\s*>", re.IGNORECASE)


def _system_prefix(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("system_prefix")
    if entry is None:
        return
    text = ctx.text
    spans = [match.span() for match in _PROMPT_START.finditer(text)]
    # The loader's own instruction, not a copy: what it sends is what doubles.
    at = text.lower().find(NETAYUME_SYSTEM_PROMPT.lower())
    if at >= 0:
        spans.append((at, at + len(NETAYUME_SYSTEM_PROMPT)))
    for start, end in sorted(spans):
        cleaned = text
        for s, e in sorted(spans, reverse=True):
            cleaned = cleaned[:s] + cleaned[e:]
        yield ctx.finding(
            entry, "guide.system_prefix", "warn",
            f'"{text[start:end]}" is already sent: the pipeline adds the author\'s '
            "system instruction and its <Prompt Start> marker itself, so here it "
            "is encoded a second time as image description.",
            start=start, end=end,
            suggestion=re.sub(r"\s{2,}", " ", cleaned).strip() or None,
        )


# "image 2", "images 1 and 2", "images 1, 2 & 3", "images 1-3", "images 1 to 3".
_IMAGE_MENTION = re.compile(
    r"(?i)\bimages?\s*#?\s*(\d+(?:\s*(?:,\s*(?:and\s+|&\s*)?|and\s+|&\s*|-|–|to\s+)#?\s*\d+)*)"
)
_IMAGE_RANGE = re.compile(r"(\d+)\s*(?:-|–|to)\s*(\d+)")


def _named_images(text: str) -> set[int]:
    named: set[int] = set()
    for match in _IMAGE_MENTION.finditer(text):
        numbers = match.group(1)
        for low, high in _IMAGE_RANGE.findall(numbers):
            # A range names every image in it; a backwards or huge one is a typo.
            if int(low) <= int(high) <= int(low) + 16:
                named.update(range(int(low), int(high) + 1))
        named.update(int(number) for number in re.findall(r"\d+", numbers))
    return named


def _references(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("references")
    count = len(ctx.applied.reference)
    # One reference needs no number: "turn this cat into a dog" is BFL's own.
    if entry is None or count < 2:
        return
    named = _named_images(ctx.text)
    missing = [str(n) for n in range(1, count + 1) if n not in named]
    if missing:
        yield ctx.finding(
            entry, "guide.references", "info",
            f"{count} reference images are attached, but the prompt never says what "
            f"to take from image {', '.join(missing)}.",
        )


def _xml(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("xml")
    if entry is None or ctx.style != "xml":
        return
    text = ctx.text
    # A bare ampersand is prose, not markup; a same-length stand-in keeps
    # the parser's offsets the prompt's own.
    wrapped = "<prompt>" + re.sub(r"&(?!#?\w+;)", "+", text) + "</prompt>"
    try:
        root = ElementTree.fromstring(wrapped)
    except ElementTree.ParseError as exc:
        line, column = exc.position
        rows = wrapped.split("\n")[: line - 1]
        offset = sum(len(row) + 1 for row in rows) + column - len("<prompt>")
        start = end = None
        if 0 <= offset < len(text):
            # Expat points at the tag's name; select from its "<" or "</".
            for opener in ("</", "<"):
                if text[max(0, offset - len(opener)):offset] == opener:
                    offset -= len(opener)
                    break
            start = offset
            newline = text.find("\n", offset)
            end = newline if newline > offset else len(text)
        yield ctx.finding(
            entry, "guide.xml", "warn",
            f"This does not parse as XML ({str(exc).split(':')[0]}), so the blocks "
            "no longer keep each character's attributes together.",
            start=start, end=end,
        )
        return
    characters = [node for node in root if re.fullmatch(r"character_\d+", node.tag)]
    if not characters:
        yield ctx.finding(
            entry, "guide.xml", "info",
            "No <character_1> block: the guide puts each character in its own.",
        )
    for node in characters:
        present = {child.tag for child in node}
        missing = [] if present & set(entry["names"]) else ["name"]
        missing += [name for name in entry["fields"] if name not in present]
        if missing:
            yield ctx.finding(
                entry, "guide.xml", "info",
                f"<{node.tag}> has no {', '.join(missing)}; the guide's character "
                "block has name, gender, appearance, clothing, expression, action "
                "and position.",
            )
    if root.find(entry["scene"]) is None:
        yield ctx.finding(
            entry, "guide.xml", "info",
            f"No <{entry['scene']}> block: the guide puts the shared scene there.",
        )


_MULTIPLE = re.compile(
    r"(?i)\b(?:[2-9]\+?[ _]?(?:girls|boys|others)|multiple[ _](?:girls|boys|others))\b"
)


def _xml_multi(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("xml_multi")
    if entry is None or ctx.style in ("xml", "json"):
        return
    match = _MULTIPLE.search(ctx.text)
    people = {norm for norm in ctx.norms if norm in ("1girl", "1boy", "1other")}
    if match or len(people) >= 2:
        yield ctx.finding(
            entry, "guide.xml", "info",
            "Several characters in plain text: the authors recommend XML, one "
            "<character_N> block each, so their attributes stay apart.",
            start=match.start() if match else None, end=match.end() if match else None,
        )


def _placeholders(ctx: _Context) -> Iterable[PromptFinding]:
    for entry in ctx.analysis.get("placeholders", []):
        for match in re.finditer(entry["pattern"], ctx.text, re.IGNORECASE):
            yield ctx.finding(
                entry, "author.placeholder", "warn",
                f"{match.group()} is the author's template slot, and the encoder "
                "reads it literally: put the character's name or description there.",
                start=match.start(), end=match.end(),
            )


def _magic(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("magic")
    if entry is None:
        return
    text = ctx.text
    flat = re.sub(r"[\W_]+", " ", text.lower())
    if "ultra hd 4k cinematic composition" in flat or ("超清" in text and "电影级构图" in text):
        return
    chinese = len(_CJK.findall(text)) > len(_WORD.findall(_CJK.sub(" ", text)))
    suffix = entry["zh" if chinese else "en"]
    base = text.rstrip()
    if base[-1:] in ".!?":
        suggestion = base + " " + suffix.lstrip(", ")
    elif base[-1:] in "。！？":
        suggestion = base + suffix.lstrip(", ")
    else:
        suggestion = base + suffix
    yield ctx.finding(
        entry, "guide.qwen_magic", "info",
        f'Without Qwen\'s positive magic, "{suffix.strip(", .")}", which its '
        "README appends to every prompt.",
        suggestion=suggestion,
    )


def _negative_prompt(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("negative_prompt")
    if entry is None or ctx.applied.negative_prompt is not None:
        return
    typed = bool((ctx.request.negative_prompt or "").strip())
    recommended = (ctx.guide.get("tag_helper") or {}).get("negative_prompt")
    if typed:
        message = (
            "The negative prompt is filled in but does not run at these guidance "
            "settings, and the authors recommend one."
        )
    else:
        message = "No negative prompt runs, and the authors recommend one."
    yield ctx.finding(
        entry, "guide.negative_prompt", "info", message,
        field="negative_prompt",
        suggestion=recommended if not typed else None,
    )


# ------------------------------------------------------------------ tag rules


def _order(ctx: _Context) -> Iterable[PromptFinding]:
    order = ctx.entry("order")
    spec = _order_spec(ctx.guide)
    if order is None or spec is None or ctx.style in ("xml", "json"):
        return
    if not 2 <= len(ctx.items) <= _MAX_ORDERED:
        return
    places = [_place(item, spec, ctx.lookup) for item in ctx.items]
    pair = _violation(places)
    if pair is None:
        return
    first, second = (ctx.items[index] for index in pair)

    def label(place: frozenset[int]) -> str:
        return f" ({spec.steps[next(iter(place))]})" if len(place) == 1 else ""

    reordered = _rebuild(ctx.text, ctx.items, [ctx.items[i].text for i in _ordered(places)])
    yield ctx.finding(
        order, "guide.order", "info",
        f'"{first.text}"{label(places[pair[0]])} comes before "{second.text}"'
        f"{label(places[pair[1]])}; the documented order is "
        f"{' → '.join(spec.steps)}.",
        start=first.start, end=first.end,
        suggestion=reordered,
    )


def _suffix(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("suffix")
    suffix = (ctx.guide.get("tag_helper") or {}).get("recommended_suffix") or []
    if entry is None or not suffix or ctx.style in ("xml", "json"):
        return
    present = set(ctx.norms)
    missing = [tag for tag in suffix if _norm(tag) not in present]
    if missing:
        yield ctx.finding(
            entry, "guide.quality", "info",
            f"The recommended finish is missing {', '.join(missing)}.",
            suggestion=_rebuild(
                ctx.text, ctx.items, [item.text for item in ctx.items] + missing
            ) if ctx.items else ", ".join(missing),
        )


def _subject(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("subject")
    helper = ctx.guide.get("tag_helper") or {}
    recipe = {_norm(tag) for tag in helper.get("recommended_suffix") or []}
    if entry is None or not ctx.norms:
        return
    if all(norm in recipe for norm in ctx.norms):
        yield ctx.finding(
            entry, "guide.subject", "info",
            "This is the quality recipe alone: the subject and scene go before it.",
        )


def _group(ctx: _Context, label: str) -> set[str]:
    for group in (ctx.guide.get("tag_helper") or {}).get("groups", []):
        if group["label"] == label:
            return {_norm(tag) for tag in group["tags"]}
    return set()


def _quality_tier(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("quality_tier")
    if entry is None or not ctx.taglike:
        return
    tiers = {_norm(tag) for tag in entry["tiers"]}
    present = _distinct(ctx, _group(ctx, entry["group"]))
    names = [ctx.norms[index] for index in present]
    if not any(name in tiers for name in names):
        yield ctx.finding(
            entry, "guide.quality", "info",
            f"No quality tier: the guide ends a tag prompt with one of "
            f"{', '.join(entry['tiers'])}.",
        )
    elif len(names) > 1:
        item = ctx.items[present[1]]
        yield ctx.finding(
            entry, "guide.quality", "info",
            f"{len(names)} quality tiers ({', '.join(names)}); the guide ends a tag "
            "prompt with one.",
            start=item.start, end=item.end,
        )
    years = _distinct(ctx, _group(ctx, entry["years"]))
    if len(years) > 1:
        item = ctx.items[years[1]]
        yield ctx.finding(
            entry, "guide.quality", "info",
            f"{len(years)} year modifiers ({', '.join(ctx.norms[i] for i in years)}); "
            "the guide allows one.",
            start=item.start, end=item.end,
        )


def _distinct(ctx: _Context, tags: set[str]) -> list[int]:
    """Where each distinct member of `tags` first appears, in prompt order."""
    first: dict[str, int] = {}
    for index, norm in enumerate(ctx.norms):
        if norm in tags:
            first.setdefault(norm, index)
    return list(first.values())


def _composition(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("composition")
    if entry is None:
        return
    seen = _distinct(ctx, {_norm(tag) for tag in entry["tags"]})
    if len(seen) >= 2:
        item = ctx.items[seen[1]]
        yield ctx.finding(
            entry, "guide.composition", "info",
            f"{len(seen)} composition tags ({', '.join(ctx.norms[i] for i in seen)}); "
            "the authors warn that conflicting ones confuse it - keep one.",
            start=item.start, end=item.end,
        )


def _rating_names(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("rating_names")
    if entry is None:
        return
    for index, norm in enumerate(ctx.norms):
        replacement = entry["replace"].get(norm)
        if replacement:
            item = ctx.items[index]
            texts = [i.text for i in ctx.items]
            texts[index] = replacement
            yield ctx.finding(
                entry, "guide.rating", "info",
                f'"{item.text}" is not a Danbooru rating name, which is what this '
                f'rating slot takes; the same rating is "{replacement}".',
                start=item.start, end=item.end,
                suggestion=_rebuild(ctx.text, ctx.items, texts),
            )


def _explicit_pair(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("explicit_pair")
    if entry is None or "explicit" not in ctx.norms or "nsfw" in ctx.norms:
        return
    index = ctx.norms.index("explicit")
    item = ctx.items[index]
    texts = [i.text for i in ctx.items]
    texts.insert(index, "nsfw")
    yield ctx.finding(
        entry, "guide.rating", "info",
        '"explicit" without "nsfw": the author\'s explicit captions carry both.',
        start=item.start, end=item.end,
        suggestion=_rebuild(ctx.text, ctx.items, texts),
    )


def _rating_negative(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("rating_negative")
    if entry is None:
        return
    ratings = [norm for norm in ctx.norms if norm in entry["ratings"]]

    def has_nsfw(text: str | None) -> bool:
        return "nsfw" in {_norm(item.text) for item in comma_items(text or "")}

    if not ratings or has_nsfw(ctx.applied.negative_prompt):
        return
    typed = ctx.request.negative_prompt or ""
    if has_nsfw(typed):
        # Written, but dropped by applied(): the CFG is at or under the threshold.
        message = (
            f'The prompt asks for a "{ratings[0]}" rating and the negative prompt '
            "carries nsfw, but the negative prompt does not run at these settings."
        )
        suggestion = None
    else:
        message = (
            f'The prompt asks for a "{ratings[0]}" rating; the author suggests nsfw '
            "in the negative prompt to filter sexual content."
        )
        base = typed.rstrip().rstrip(",").rstrip()
        suggestion = (base + _separator(typed) + "nsfw") if base else "nsfw"
    yield ctx.finding(
        entry, "guide.rating_negative", "info", message,
        field="negative_prompt", suggestion=suggestion,
    )


# An underscore joining words; Delta's card keeps underscores in short emoji
# tags (^_^, o_o), which join no words.
_JOINED = re.compile(r"[^\W_]{2,}_|_[^\W_]{2,}")


def _underscores(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("underscores")
    if entry is None or ctx.style in ("xml", "json"):
        return
    keep = re.compile(entry["keep"], re.IGNORECASE) if entry.get("keep") else None
    flagged = [
        index for index, item in enumerate(ctx.items)
        if not _is_prose(item.text)
        and _JOINED.search(item.text)
        and not (keep and keep.fullmatch(item.text))
    ]
    if flagged:
        texts = [item.text for item in ctx.items]
        for index in flagged:
            texts[index] = texts[index].replace("_", " ")
        first = ctx.items[flagged[0]]
        yield ctx.finding(
            entry, "guide.underscores", "info",
            f"Underscores in {', '.join(ctx.items[i].text for i in flagged)}; the "
            "guide writes tags with spaces.",
            start=first.start, end=first.end,
            suggestion=_rebuild(ctx.text, ctx.items, texts),
        )


def _lowercase(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("lowercase")
    # Only a tag list: in prose, names take their capitals.
    if entry is None or ctx.style != "tags":
        return
    flagged = [index for index, item in enumerate(ctx.items) if item.text != item.text.lower()]
    if flagged:
        texts = [item.text for item in ctx.items]
        for index in flagged:
            texts[index] = texts[index].lower()
        first = ctx.items[flagged[0]]
        yield ctx.finding(
            entry, "guide.lowercase", "info",
            f"Capitals in {', '.join(ctx.items[i].text for i in flagged)}; the guide "
            "writes tags in lowercase.",
            start=first.start, end=first.end,
            suggestion=_rebuild(ctx.text, ctx.items, texts),
        )


def _artist_prefix(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("artist_prefix")
    if entry is None or ctx.lookup is None or not ctx.taglike:
        return
    flagged = [
        index for index, item in enumerate(ctx.items)
        if not item.text.startswith("@")
        and (ctx.lookup.get(item.text) or ctx.lookup.get(ctx.norms[index])) == "artist"
    ]
    if flagged:
        texts = [item.text for item in ctx.items]
        for index in flagged:
            texts[index] = "@" + texts[index]
        first = ctx.items[flagged[0]]
        yield ctx.finding(
            entry, "guide.artist_prefix", "info",
            f"{', '.join(ctx.items[i].text for i in flagged)} is an artist in "
            "Danbooru's index; the guide puts @ before an artist name.",
            start=first.start, end=first.end,
            suggestion=_rebuild(ctx.text, ctx.items, texts),
        )


def _sentences(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("sentences")
    if entry is None or ctx.style != "prose":
        return
    sentences = [
        piece for piece in re.split(r"(?<=[.!?])\s+|(?<=[。！？])", ctx.text.strip())
        if word_count(piece) >= 2
    ]
    if len(sentences) < entry["min"]:
        yield ctx.finding(
            entry, "guide.sentences", "info",
            f"{len(sentences)} sentence; the guide asks for at least {entry['min']} "
            "when prompting in natural language.",
        )


# ------------------------------------------------------------------ heuristics
#
# Pattern matching with no author sentence behind it. Labelled `heuristic`,
# with no cite and no url, so the page never shows one as a guide's advice.

_BOILERPLATE = re.compile(
    r"(?i)\b(?:masterpiece|best quality|high quality|top quality|highest quality"
    r"|amazing quality|ultra[- ]?detailed|hyper[- ]?detailed|extremely detailed"
    r"|highly detailed|absurdres|highres|8k|4k|16k|uhd|ultra hd|hdr"
    r"|trending on artstation|award[- ]winning)\b"
)


def _heuristic(
    rule: str, message: str, *, start: int | None = None, end: int | None = None,
    suggestion: str | None = None,
) -> PromptFinding:
    return PromptFinding(
        rule=rule, severity="info", message=message, source="heuristic",
        start=start, end=end, suggestion=suggestion,
    )


def _quality_boilerplate(ctx: _Context) -> Iterable[PromptFinding]:
    # Prose checkpoints whose guide asks for no such words. Qwen's does -
    # its positive magic is "Ultra HD, 4K" - and the tag checkpoints' quality
    # tags are their authors' own vocabulary.
    style = ctx.entry("style") or {}
    if style.get("expects") != "prose" or "magic" in ctx.analysis:
        return
    matches = [m for m in _BOILERPLATE.finditer(ctx.text) if not _inside(ctx.quoted, m.start())]
    terms = list(dict.fromkeys(match.group().lower() for match in matches))
    if len(terms) >= 2:
        yield _heuristic(
            "heuristic.quality_boilerplate",
            f"Quality boilerplate ({', '.join(terms)}): tag-model habits this "
            "checkpoint's guide does not ask for, taking room a description could use.",
            start=matches[0].start(), end=matches[0].end(),
        )


def _repeated(ctx: _Context) -> Iterable[PromptFinding]:
    if ctx.style in ("xml", "json"):
        return
    seen: set[str] = set()
    repeats: list[int] = []
    for index, norm in enumerate(ctx.norms):
        key = norm.strip(" .!?")
        if key in seen:
            repeats.append(index)
        seen.add(key)
    if repeats:
        first = ctx.items[repeats[0]]
        kept = [item.text for index, item in enumerate(ctx.items) if index not in repeats]
        names = list(dict.fromkeys(ctx.items[index].text for index in repeats))
        yield _heuristic(
            "heuristic.repeated",
            f"Repeated: {', '.join(names)}. A second copy adds no new content "
            "and takes encoder room.",
            start=first.start, end=first.end,
            suggestion=_rebuild(ctx.text, ctx.items, kept),
        )


_RULES: tuple[Rule, ...] = (
    _style,
    _words,
    _negation,
    _quote_text,
    _one_paragraph,
    _system_prefix,
    _references,
    _xml,
    _xml_multi,
    _placeholders,
    _magic,
    _negative_prompt,
    _order,
    _suffix,
    _subject,
    _quality_tier,
    _composition,
    _rating_names,
    _explicit_pair,
    _rating_negative,
    _underscores,
    _lowercase,
    _artist_prefix,
    _sentences,
    _quality_boilerplate,
    _repeated,
)
