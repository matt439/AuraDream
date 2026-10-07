"""Deterministic prompt findings grounded in each checkpoint guide.

Text parsing lives in prompt_text and documented placement in prompt_order.
This module builds one context and dispatches the guide and heuristic checks."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal
from xml.etree import ElementTree

from . import prompt_order, prompt_text
from .backends.netayume_loading import SYSTEM_PROMPT as NETAYUME_SYSTEM_PROMPT
from .schemas import GenerateRequest, PromptFinding

# Public entry points stay available from this module.
from .prompt_order import (
    artists_read as artists_read,
    category_items as category_items,
    documented_order as documented_order,
    lookup_items as lookup_items,
    order_checked as order_checked,
    order_items as order_items,
    order_slots as order_slots,
    places_every_item as places_every_item,
    reads_order as reads_order,
    reorder_tags as reorder_tags,
    scene_ahead as scene_ahead,
    unplaced_items as unplaced_items,
    wants_artist_categories as wants_artist_categories,
)
from .prompt_text import (
    Item as Item,
    QUOTED as QUOTED,
    Style as Style,
    cjk_count as cjk_count,
    comma_items as comma_items,
    detect_style as detect_style,
    rendered_spans as rendered_spans,
    rendered_targets as rendered_targets,
    word_count as word_count,
)


@dataclass
class _Context:
    """One prompt, read once, for every rule."""

    guide: dict[str, Any]
    analysis: dict[str, Any]
    request: GenerateRequest
    applied: GenerateRequest
    lookup: dict[str, str] | None
    text: str
    style: prompt_text.Style
    # The comma items, each stretch of lettering one item (`_units`).
    items: list[prompt_text.Item]
    # Each item's comparable form, index for index; None where it holds
    # lettering, which no rule matches as a tag: "safe" in "hi, safe, bye"
    # on a sign is no rating.
    norms: list[str | None]
    quoted: list[tuple[int, int]]
    # Words that read as text to render without quotation marks: "NO
    # PARKING" on a sign is lettering to quote, not an absence to rephrase.
    rendered: list[tuple[int, int]]
    # The text with both blanked, offsets kept, for the rules that look for
    # a tag anywhere in it rather than item by item.
    bare: str

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

    def segments(self, item: prompt_text.Item) -> list[tuple[str, bool]]:
        """`item`'s text in stretches, each True where it is lettering - quoted,
        or read as text to render - which a rule leaves as written: 'Sign
        says "HELLO"' lowercases to 'sign says "HELLO"'."""
        parts: list[tuple[str, bool]] = []
        cursor = item.start
        for start, end in sorted(self.quoted + self.rendered):
            start, end = max(start, cursor), min(end, item.end)
            if start < end:
                parts += [(self.text[cursor:start], False), (self.text[start:end], True)]
                cursor = end
        parts.append((self.text[cursor:item.end], False))
        return [(text, lettering) for text, lettering in parts if text]


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
    context = _read(model_id, request, applied, categories)
    findings: list[PromptFinding] = []
    for rule in _RULES:
        findings.extend(rule(context))
    return sorted(
        findings,
        key=lambda f: (f.severity != "warn", f.start is None, f.start or 0),
    )


def _read(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    categories: dict[str, str] | None,
) -> _Context:
    """`request.prompt` read once, as every rule reads it."""
    guide = prompt_order._guide(model_id)
    text = request.prompt
    quoted = prompt_text._quoted_spans(text)
    rendered = prompt_text._unquoted_text(text, quoted)
    lettering = quoted + rendered
    items = prompt_text._units(text, prompt_text.comma_items(text), lettering)
    lookup = prompt_text._category_lookup(categories)
    return _Context(
        guide=guide,
        analysis=prompt_order._analysis(guide),
        request=request,
        applied=applied,
        lookup=lookup,
        text=text,
        style=prompt_text._detect_style(text, lookup),
        items=items,
        norms=[
            None if held else prompt_text._norm(item.text)
            for item, held in zip(items, prompt_text._held(items, lettering), strict=True)
        ],
        quoted=quoted,
        rendered=rendered,
        bare=prompt_text._blank(text, lettering),
    )


def order_finding(
    model_id: str, text: str, categories: dict[str, str] | None
) -> PromptFinding | None:
    """The order rule's finding on `text`, its items placed with `categories`
    as `check` places them, or None: `text` in the documented order as far as
    they place it - or not read at all, which `order_checked` tells apart.

    What a critic's reorder other than `reorder_tags`' is held to where the
    categories placed only some items (`places_every_item` false): the
    rules' reorder is then one order the documented one allows, not the only
    one.
    """
    # The rule reads the text alone; the request is only the shell a
    # `_Context` holds, and a reorder's text is not one the user sent.
    request = GenerateRequest.model_construct(prompt=text)
    return next(iter(_order(_read(model_id, request, request, categories))), None)


# ------------------------------------------------------------------ the rules

Rule = Callable[[_Context], Iterable[PromptFinding]]


def _style(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("style")
    if entry is None or ctx.style in entry.get("accepts", ()):
        return
    expects = entry["expects"]
    # Three of the pieces the style was read from - split at commas, "，" and
    # line breaks (`_pieces`), not at commas alone as the chips split - each
    # stretch of lettering one piece, as `_units` makes it one comma item: a
    # list written with "，" or one item a line is a tag list here too, while
    # `1girl, "hi, there"` is still two.
    pieces = prompt_text._units(ctx.text, prompt_text._split(ctx.text, r"[^,，\n]+"), ctx.quoted + ctx.rendered)
    if expects == "prose" and ctx.style == "tags" and len(pieces) >= 3:
        # Not "comma-separated": a list one item a line is one too.
        yield ctx.finding(
            entry, "guide.style", "info",
            "This reads as a tag list; this checkpoint's guide asks for descriptive "
            "sentences.",
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
    count = prompt_text.word_count(ctx.text)
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


def _negation(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("negation")
    if entry is None:
        return
    runs = ctx.applied.negative_prompt is not None
    for match in prompt_text._NEGATION.finditer(ctx.text):
        # Quoted or unquoted, lettering on a sign ("NO PARKING") is text to
        # render - the quote rule's business - and not an absence.
        if prompt_text._inside(ctx.quoted, match.start()) or prompt_text._inside(ctx.rendered, match.start()):
            continue
        absence = prompt_text._absence(ctx.text, match)
        if absence is None:
            continue
        start, end, lowered = absence
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
    for start, end, exact, _, _ in prompt_text._unquoted_targets(ctx.text, ctx.quoted):
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
# The loader's own instruction, not a copy: what it sends is what doubles -
# as a person retypes it, too: any case, any run of whitespace between its
# words, the closing full stop or not. Without the stop it was missed, and
# Apply cut only the marker after it.
_NETAYUME_INSTRUCTION = re.compile(
    r"(?<!\w)"
    + r"\s+".join(map(re.escape, NETAYUME_SYSTEM_PROMPT.rstrip(".").split()))
    + r"(?:\.|(?!\w))",
    re.IGNORECASE,
)
# How a system instruction opens: "You are an assistant designed to ...",
# diffusers' Lumina 2 one and the Neta variants - an assistant, a model, an
# AI, a generator - and not a scene a person describes ("You are looking at
# a girl in the rain"), whose words Apply would otherwise cut. The noun ends
# the opener: a stop or the end follows it, or a word an instruction goes on
# with ("designed to", "that generates"), so "a model student walking in the
# rain" and "an AI artist painting a girl" stay the user's.
_INSTRUCTION = re.compile(
    r"\s*you\s+are\s+an?\s+(?:[\w-]+\s+){0,3}(?:assistant|model|ai|generator|bot)"
    r"(?=\s*(?:[.,;:]|$)|\s+(?:designed|that|which|who|built|trained|made|"
    r"speciali[sz]ed|capable|for|to)\b)",
    re.IGNORECASE,
)
# A sentence ending with more text after it. Not `_SENTENCE_BREAK`, which
# reads "an AI. A girl" as an abbreviation's stop.
_MORE_SENTENCES = re.compile(r"[.!?](?=\s+\S)|[。！？](?=\s*\S)")


def _system_prefix(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("system_prefix")
    if entry is None:
        return
    text = ctx.text
    # Lettering is the image description the user asked for, even where it
    # spells a marker or the whole instruction. Search the same-length bare
    # text so a quoted occurrence neither gets cut nor hides a later prefix.
    markers = list(_PROMPT_START.finditer(ctx.bare))
    spans = [match.span() for match in markers]
    # Found in the text itself, not in text.lower(), whose offsets are not the
    # text's ("İ" lowers to two characters).
    at = _NETAYUME_INSTRUCTION.search(ctx.bare)
    if at:
        spans.append(at.span())
    # Lumina's template is instruction, marker, prompt: what stands before
    # the first marker, where it opens as an instruction does ("You are an
    # assistant ..."),
    # is a system instruction of some other workflow's - diffusers' own
    # Lumina 2 one, a Neta variant - and the marker its separator. Cut as one,
    # so Apply leaves the image description and nothing of the instruction;
    # a marker after tags ("1girl <Prompt Start> moon") is cut alone, and so
    # is one after lettering, which is the user's, and one after more than a
    # sentence ("You are an AI. A girl stands in the rain."), where nothing
    # says how much of it is the instruction.
    extended: tuple[int, int] | None = None
    if markers and (at is None or at.start() >= markers[0].start()):
        first = markers[0]
        head = text[: first.start()]
        if (
            _INSTRUCTION.match(ctx.bare, 0, first.start())
            and ctx.bare[: first.start()] == head
            and not _MORE_SENTENCES.search(head.strip())
        ):
            extended = (len(head) - len(head.lstrip()), first.end())
            spans[0] = extended
    # Each cut's seam becomes one space, and a comma either side of it one
    # comma; whitespace anywhere else - a blank line, the spacing inside
    # quoted lettering - is the user's.
    cleaned = text
    for s, e in sorted(spans, reverse=True):
        head, tail = cleaned[:s].rstrip(), cleaned[e:].lstrip()
        if tail[:1] in (",", "，") and (not head or head[-1] in (",", "，")):
            tail = tail[1:].lstrip()
        cleaned = head + (" " if head and tail else "") + tail
    for start, end in sorted(spans):
        message = (
            f'"{text[start:end]}" is a system instruction and its <Prompt Start> marker: the '
            "pipeline adds the author's own instruction and marker itself, so here this one "
            "is encoded as image description."
            if (start, end) == extended else
            f'"{text[start:end]}" is already sent: the pipeline adds the author\'s '
            "system instruction and its <Prompt Start> marker itself, so here it "
            "is encoded a second time as image description."
        )
        yield ctx.finding(
            entry, "guide.system_prefix", "warn", message,
            start=start, end=end,
            suggestion=cleaned.strip() or None,
        )


# "image 2", "images 1 and 2", "images 1, 2 & 3", "images 1-3", "images 1 to 3".
_IMAGE_MENTION = re.compile(
    r"(?i)\bimages?\s*#?\s*(\d+(?:\s*(?:,\s*(?:and\s+|&\s*)?|and\s+|&\s*|-|–|to\s+)#?\s*\d+)*)"
)
_IMAGE_RANGE = re.compile(r"(\d+)\s*(?:-|–|to)\s*(\d+)")
# An edit of a picture the prompt takes to be there already: "Turn this cat
# into a dog", "Make the photo brighter". The Examples menu reads the same
# pattern for the authors' examples (web/app.js's EDIT_PROMPT, held equal to
# this one by tests/test_prompt_rules.py), so both say the same thing.
EDIT_PROMPT = re.compile(
    r"^\s*(turn|change|make|replace|remove|put|give)\b"
    r".*\b(this|these|the (photo|image|picture)|image \d)\b",
    re.IGNORECASE,
)


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
    if entry is None:
        return
    count = len(ctx.applied.reference)
    # With none attached, an edit of "this" picture renders from the words
    # alone - a plain text-to-image that looks like a success. No author
    # sentence says so, so it is the app's own reading, and a note.
    if count == 0 and EDIT_PROMPT.search(ctx.text):
        yield _heuristic(
            "heuristic.no_reference",
            "This reads as an edit of an existing picture, but no reference image is "
            "attached - it will render from the prompt alone. Add one with \"Use as a "
            "reference image\" (❐) on a gallery image.",
        )
    # One reference needs no number: "turn this cat into a dog" is BFL's own.
    if count < 2:
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
    match = _MULTIPLE.search(ctx.bare)
    people = {norm for norm in ctx.norms if norm in ("1girl", "1boy", "1other")}
    if match or len(people) >= 2:
        yield ctx.finding(
            entry, "guide.xml", "info",
            "Several characters in plain text: the authors recommend XML, one "
            "<character_N> block each, so their attributes stay apart.",
            start=match.start() if match else None, end=match.end() if match else None,
        )


def _slot_cite(entry: dict[str, Any], slot: str, lines: list[str]) -> str:
    """What a slot's finding quotes: the entry's own cite where it shows this
    slot, else the shortest line of the authors' examples that does - so
    "$character_2$" quotes "<n>$character_2$</n>", not the first slot's line -
    else the entry's cite."""
    if slot.casefold() in entry["cite"].casefold():
        return entry["cite"]
    holding = [line for line in lines if slot.casefold() in line.casefold()]
    return min(holding, key=len) if holding else entry["cite"]


def _placeholders(ctx: _Context) -> Iterable[PromptFinding]:
    lines = [
        line.strip()
        for example in ctx.guide.get("examples") or []
        for line in example.splitlines()
        if line.strip()
    ]
    for entry in ctx.analysis.get("placeholders", []):
        fill = entry.get("fill", "what it stands for")
        for match in re.finditer(entry["pattern"], ctx.text, re.IGNORECASE):
            slot = match.group()
            yield ctx.finding(
                {**entry, "cite": _slot_cite(entry, slot, lines)}, "author.placeholder", "warn",
                f"{slot} is the author's template slot, and the encoder "
                f"reads it literally: put {fill} there.",
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
    chinese = len(prompt_text._CJK.findall(text)) > len(prompt_text._WORD.findall(prompt_text._CJK.sub(" ", text)))
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
    spec = prompt_order._order_spec(ctx.guide)
    if order is None or spec is None or len(ctx.items) < 2:
        return
    if not prompt_order._placeable(spec, ctx.style, ctx.text, ctx.items, ctx.quoted + ctx.rendered):
        return
    places = prompt_order._places(spec, ctx.items, [norm is None for norm in ctx.norms], ctx.lookup)
    pair = prompt_order._violation(places)
    if pair is None:
        return
    first, second = (ctx.items[index] for index in pair)

    def label(place: frozenset[int]) -> str:
        return f" ({spec.steps[next(iter(place))]})" if len(place) == 1 else ""

    # None where the move would change what is quoted: the order is still
    # out, but there is no one-click fix.
    reordered = prompt_order._reordered(ctx.text, ctx.items, places)
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
    # A tag on a line of its own or after "，" is there, though not a comma
    # item: appending it again would write it twice. One in lettering is not.
    present = set(ctx.norms) | {prompt_text._norm(piece) for piece in prompt_text._pieces(ctx.bare)}
    missing = [tag for tag in suffix if prompt_text._norm(tag) not in present]
    if missing:
        yield ctx.finding(
            entry, "guide.quality", "info",
            f"The recommended finish is missing {', '.join(missing)}.",
            suggestion=prompt_text._rebuild(
                ctx.text, ctx.items, [item.text for item in ctx.items] + missing
            ) if ctx.items else ", ".join(missing),
        )


def _subject(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("subject")
    helper = ctx.guide.get("tag_helper") or {}
    recipe = {prompt_text._norm(tag) for tag in helper.get("recommended_suffix") or []}
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
            return {prompt_text._norm(tag) for tag in group["tags"]}
    return set()


def _quality_tier(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("quality_tier")
    if entry is None or not ctx.taglike:
        return
    tiers = {prompt_text._norm(tag) for tag in entry["tiers"]}
    present = _distinct(ctx, _group(ctx, entry["group"]))
    names = [ctx.norms[index] for index in present]
    # A tier on a line of its own or after "，" is there, though not a comma
    # item (`_suffix` reads it the same way).
    pieces = {prompt_text._norm(piece) for piece in prompt_text._pieces(ctx.bare)}
    if not any(name in tiers for name in names) and not tiers & pieces:
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
    seen = _distinct(ctx, {prompt_text._norm(tag) for tag in entry["tags"]})
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
                suggestion=prompt_text._rebuild(ctx.text, ctx.items, texts),
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
        suggestion=prompt_text._rebuild(ctx.text, ctx.items, texts),
    )


def _rating_negative(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("rating_negative")
    if entry is None:
        return
    ratings = [norm for norm in ctx.norms if norm in entry["ratings"]]

    def has_nsfw(text: str | None) -> bool:
        return "nsfw" in {prompt_text._norm(item.text) for item in prompt_text.comma_items(text or "")}

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
        suggestion = (base + prompt_text._separator(typed) + "nsfw") if base else "nsfw"
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
    # Underscores in quoted lettering ("user_name" on a screen) are text, and
    # its words make no tag beside it prose.
    bare = [
        "".join(" " * len(part) if lettering else part for part, lettering in ctx.segments(item))
        for item in ctx.items
    ]
    flagged = [
        index for index, item in enumerate(ctx.items)
        if not prompt_text._is_prose(bare[index])
        and _JOINED.search(bare[index])
        and not (keep and keep.fullmatch(item.text))
    ]
    if flagged:
        texts = [item.text for item in ctx.items]
        for index in flagged:
            texts[index] = "".join(
                part if lettering else part.replace("_", " ")
                for part, lettering in ctx.segments(ctx.items[index])
            )
        first = ctx.items[flagged[0]]
        yield ctx.finding(
            entry, "guide.underscores", "info",
            f"Underscores in {', '.join(ctx.items[i].text for i in flagged)}; the "
            "guide writes tags with spaces.",
            start=first.start, end=first.end,
            suggestion=prompt_text._rebuild(ctx.text, ctx.items, texts),
        )


def _lowercase(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("lowercase")
    # Only a tag list: in prose, names take their capitals.
    if entry is None or ctx.style != "tags":
        return
    # Quoted lettering keeps its capitals: "HELLO" on a sign is not a tag.
    lowered = [
        "".join(part if lettering else part.lower() for part, lettering in ctx.segments(item))
        for item in ctx.items
    ]
    flagged = [index for index, item in enumerate(ctx.items) if lowered[index] != item.text]
    if flagged:
        texts = [item.text for item in ctx.items]
        for index in flagged:
            texts[index] = lowered[index]
        first = ctx.items[flagged[0]]
        yield ctx.finding(
            entry, "guide.lowercase", "info",
            f"Capitals in {', '.join(ctx.items[i].text for i in flagged)}; the guide "
            "writes tags in lowercase.",
            start=first.start, end=first.end,
            suggestion=prompt_text._rebuild(ctx.text, ctx.items, texts),
        )


def _listed(names: list[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _artist_prefix(ctx: _Context) -> Iterable[PromptFinding]:
    # Item by item, whatever the whole prompt reads as: Anima's card says to
    # use tags, natural language or both, and a tag list that ends in two
    # sentences reads as prose - which once skipped this rule and reported
    # nothing to flag over an artist written without its @. RouWei uses by
    # followed by a space instead, supplied by its analysis entry. A sentence is no
    # artist's name, and the lookup was never asked about one (`lookup_items`).
    entry = ctx.entry("artist_prefix")
    if entry is None or ctx.lookup is None or ctx.style in ("xml", "json"):
        return
    prefix = entry.get("prefix", "@")
    flagged = [
        index for index, (item, norm) in enumerate(zip(ctx.items, ctx.norms, strict=True))
        if norm is not None
        and not item.text.lower().startswith(prefix.lower())
        and not prompt_text._is_sentence(item.text)
        and (ctx.lookup.get(item.text) or ctx.lookup.get(norm)) == "artist"
    ]
    if flagged:
        texts = [item.text for item in ctx.items]
        for index in flagged:
            texts[index] = prefix + texts[index]
        first = ctx.items[flagged[0]]
        # Each artist once, as first written: "wlop" and "WLOP" are one.
        first_spelling: dict[str | None, str] = {}
        for index in flagged:
            first_spelling.setdefault(ctx.norms[index], ctx.items[index].text)
        names = list(first_spelling.values())
        yield ctx.finding(
            entry, "guide.artist_prefix", "info",
            f"{_listed(names)} {'is an artist' if len(names) == 1 else 'are artists'} in "
            f"Danbooru's index; the guide puts {prefix.strip()} before an artist name.",
            start=first.start, end=first.end,
            suggestion=prompt_text._rebuild(ctx.text, ctx.items, texts),
        )


def _sentences(ctx: _Context) -> Iterable[PromptFinding]:
    entry = ctx.entry("sentences")
    if entry is None or ctx.style != "prose":
        return
    sentences = [
        piece for piece in re.split(r"(?<=[.!?])\s+|(?<=[。！？])", ctx.text.strip())
        if prompt_text.word_count(piece) >= 2
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
    matches = [m for m in _BOILERPLATE.finditer(ctx.text) if not prompt_text._inside(ctx.quoted, m.start())]
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
        # "Ho, ho, ho" in quotation marks is the lettering, repeated on purpose.
        if norm is None:
            continue
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
            suggestion=prompt_text._rebuild(ctx.text, ctx.items, kept),
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
