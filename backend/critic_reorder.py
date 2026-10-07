"""Pure prompt permutations: preserve whole pieces, lettering and JSON values."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .prompt_order import order_items
from .prompt_text import QUOTED, _norm, rendered_targets


# --- Reading the answer ------------------------------------------------------

_QUOTE_TRIM = " \t\r\n\"'“”‘’„«»「」『』`.,;:!?"

# Text to render is what `prompt_rules.QUOTED` reads as quoted - straight and
# curly quotes, „…“, «…», CJK corner brackets, single quotes that do not touch
# a word - the one definition the rules use too. A reorder carries each quoted
# stretch character for character; a „…“ or «…» split across two pieces once
# passed here, when this module kept a narrower copy.
#
# Where one of a prompt's pieces ends. A tag list's items; prose's clauses and
# sentences (a stop followed by space, so FLUX.1 and 3.5 stay whole). Chinese
# punctuation counts as its English counterpart.
_TAG_BREAK = re.compile(r"[,，、\n]")
_PROSE_BREAK = re.compile(r"[,;，、；\n]|[.!?](?=\s|$)|[。！？]")
# A full stop after an initial or an abbreviation may belong to a name,
# not end its sentence. Keep it inside the piece: otherwise "Mr. Green"
# and "Ms. Blue" can trade their names while passing as a pure reorder.
# An ambiguous stop ("Main St. A car passes") stays inside too: declining
# that sentence reorder is safer than offering a changed subject as order.
_ABBREVIATION = re.compile(
    r"(?i)\b(?:[^\W\d_]|Mr|Mrs|Ms|Mx|Dr|Prof|Rev|Fr|St|Mt|Sr|Jr|"
    r"Gen|Col|Lt|Sgt|Capt|Cmdr|Adm|Sen|Rep|Gov|Pres|Hon|Esq|"
    r"Inc|Ltd|Co|Corp|Dept|No|vs|etc)\.$"
)
_PIECE_TRIM = " \t\r\n.,;:!?。，、；：！？"
# NewBie's top-level blocks, <character_1>...</character_1> and <general_tags>.
_XML_BLOCK = re.compile(r"<([A-Za-z_][\w.-]*)>.*?</\1>", re.DOTALL)
# How a prompt pasted from a word processor or a phone writes an apostrophe,
# a quotation mark or a dash, against one typed on a keyboard - and the
# critic writes either. Folded one character for one, so an offset into the
# folded text is the same offset into the original.
_FOLD = str.maketrans(
    "\u2019\u2018\u201b\u02bc\u2032\u201c\u201d\u201e\u201f\u2033"
    "\u2010\u2011\u2012\u2013\u2014\u2212",
    "'''''\"\"\"\"\"------",
)


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    """`text[start:end]` without the space and punctuation at its ends."""
    while start < end and (text[start] in _PIECE_TRIM or text[start].isspace()):
        start += 1
    while end > start and (text[end - 1] in _PIECE_TRIM or text[end - 1].isspace()):
        end -= 1
    return start, end


def _piece_spans(text: str, style: str) -> list[tuple[int, int, str]]:
    """The phrases a reorder may move, in order: where each one is in `text`,
    and its normalised form.

    A piece is a tag, a clause or sentence, an XML block - moved whole or not
    at all - so no attribute changes owner: in "a red dress and a blue hat"
    the colours stay where they are, which a count of words cannot see. Text
    to render never splits, whatever punctuation it holds, nor does a name
    split at its initials or an abbreviation. Case, runs of whitespace,
    how apostrophes, quotation marks and dashes are typed (`_FOLD`) and the
    punctuation at a piece's ends are not part of its form; the span leaves
    the ends out too, so what lies between two pieces - a break and its
    spacing - is not part of either.
    """
    blocks = [match.span() for match in _XML_BLOCK.finditer(text)] if style == "xml" else []
    # A block is held whole; blanked to line breaks, same length, no
    # quotation can run across it, as none runs across a line.
    masked = text
    for start, end in blocks:
        masked = masked[:start] + "\n" * (end - start) + masked[end:]
    held = blocks + [(start, end) for start, end, _, _ in rendered_targets(masked)]
    breaks = _TAG_BREAK if style in ("tags", "xml") else _PROSE_BREAK
    cuts = [
        match.span()
        for match in breaks.finditer(masked)
        if not any(start <= match.start() < end for start, end in held)
        and not (
            style not in ("tags", "xml") and match.group() == "."
            and _ABBREVIATION.search(masked, 0, match.end())
        )
    ]
    # A block is a piece by itself: cut at both its edges.
    cuts = sorted(cuts + [(edge, edge) for block in blocks for edge in block])
    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    for cut, after in cuts + [(len(text), len(text))]:
        start, end = _trimmed(text, cursor, max(cursor, cut))
        cursor = max(cursor, after)
        if start == end:
            continue
        form = " ".join(text[start:end].translate(_FOLD).split())
        if (start, end) in blocks:
            form = re.sub(r">\s*<", "><", form)
        pieces.append((start, end, form.casefold()))
    return pieces


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key repeats")
    return dict(pairs)


class _Written:
    """A JSON number as the text writes it: the encoder reads the text, so
    0.50, 0.5 and 5e-1 are three, as 1.0, 1 and true are."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def _write(value: Any, sort: bool) -> str:
    """`value` written back as JSON, its keys sorted or in its own order, each
    number as it was written (`_Written`) - bare, where every string is
    quoted, so no string can pass for one."""
    if isinstance(value, _Written):
        return value.text
    if isinstance(value, dict):
        items = sorted(value.items()) if sort else value.items()
        return "{" + ",".join(f"{json.dumps(key)}:{_write(item, sort)}" for key, item in items) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_write(item, sort) for item in value) + "]"
    return json.dumps(value)


def _json_forms(text: str) -> tuple[str, str] | None:
    """`text`'s JSON value written with its keys sorted - what its values as
    written alone decide: 0.50 and 0.5 are two numbers here, and 1.0, 1 and
    true three - and in its own key order. None where it is not JSON, or
    repeats a key, whose earlier value `json.loads` would drop unseen.

    Strings are compared as values here; how they are written - an escape
    for a letter - is `_quoted`'s to hold, as in any other prompt."""
    try:
        value = json.loads(
            text, object_pairs_hook=_unique,
            parse_int=_Written, parse_float=_Written, parse_constant=_Written,
        )
        return _write(value, sort=True), _write(value, sort=False)
    except (ValueError, RecursionError):
        return None


def _quoted(text: str) -> Counter[str]:
    return Counter(quote.translate(_FOLD) for quote in QUOTED.findall(text))


def _reordered(candidate: str, prompt: str, style: str) -> tuple[bool, str | None]:
    """`candidate` read as a reorder of `prompt`: whether it holds the prompt's
    pieces and quoted text and nothing else, and the candidate with each of
    its pieces written as the prompt writes it - None where nothing moved.

    The pieces are compared normalised, but the words returned are the
    prompt's own. So a candidate that also changed case - the card's critic
    upper-cased one, and T5 and Qwen3 read case - or retyped a curly
    apostrophe comes back as the user's words, moved. What lies between the
    pieces, space and punctuation only, is the candidate's: a sentence's "!"
    or "?" goes where the sentence went, which a break left where it lay
    would not do.
    """
    if _quoted(candidate) != _quoted(prompt):
        return False, None
    if style == "json":
        # An object's keys have no order as values, so equal as values -
        # each number as written - and different in key order is exactly a
        # reorder; a list's order is content. What comes back is the
        # candidate, whose values are then the prompt's own writing and
        # whose space between them is the critic's.
        theirs, ours = _json_forms(candidate), _json_forms(prompt)
        if theirs is None or ours is None or theirs[0] != ours[0]:
            return False, None
        return True, candidate if theirs[1] != ours[1] else None
    theirs, ours = _piece_spans(candidate, style), _piece_spans(prompt, style)
    forms = [form for _, _, form in theirs]
    if Counter(forms) != Counter(form for _, _, form in ours):
        return False, None
    if forms == [form for _, _, form in ours]:
        return True, None
    # The prompt's writing of each form, in its order: a form twice ("Cat",
    # "cat") is taken first to first.
    own: dict[str, list[str]] = {}
    for start, end, form in ours:
        own.setdefault(form, []).append(prompt[start:end])
    out, cursor = [], 0
    for start, end, form in theirs:
        out += [candidate[cursor:start], own[form].pop(0)]
        cursor = end
    out.append(candidate[cursor:])
    return True, "".join(out)


_WORDS = re.compile(r"\w+")


def _words(text: str) -> Counter[str]:
    """`text`'s words in any order, as a piece's form reads them (`_FOLD`,
    case aside); "blue_hair" is one word, so "hair_blue" is another."""
    return Counter(_WORDS.findall(text.translate(_FOLD).casefold()))


def _not_a_reorder(candidate: str, prompt: str, style: str) -> str:
    """Why `candidate` is no reorder of `prompt` (`_reordered` false), in the
    set-aside sentence's words.

    An XML prompt's blocks move whole, so no attribute changes owner
    (`_piece_spans`); a candidate with exactly the prompt's words that moved
    a field or a tag inside a block, or from one block to another, changed
    no word, and the sentence once said it had. One whose every piece holds
    the prompt's words in the prompt's order moved none either: it changed
    the punctuation or spacing between them ("happy, smile" to "happy
    smile", "<n>miku</n>" to "<n> miku </n>"), and says so.
    """
    if style == "xml" and _quoted(candidate) == _quoted(prompt) and (
        _words(candidate) == _words(prompt)
    ):

        def pieces(text: str, ordered: bool) -> Counter[tuple]:
            # Each piece as its block's name, or None outside a block, and
            # its words: in their order, or in any.
            found: Counter[tuple] = Counter()
            for start, end, _ in _piece_spans(text, style):
                block = _XML_BLOCK.fullmatch(text[start:end])
                words = _WORDS.findall(text[start:end].translate(_FOLD).casefold())
                found[(
                    block.group(1).casefold() if block else None,
                    tuple(words) if ordered else tuple(sorted(words)),
                )] += 1
            return found

        def blocks(text: str, ordered: bool) -> Counter[tuple]:
            return Counter({
                key: count for key, count in pieces(text, ordered).items() if key[0] is not None
            })

        def outside(text: str) -> list[str]:
            # The words outside the blocks, in their order: a break added or
            # taken out there parts or joins pieces and moves no word.
            masked = _XML_BLOCK.sub(" ", text)
            return _WORDS.findall(masked.translate(_FOLD).casefold())

        if pieces(candidate, True) == pieces(prompt, True) or (
            blocks(candidate, True) == blocks(prompt, True)
            and outside(candidate) == outside(prompt)
        ):
            return "it changed punctuation or spacing, not only the order of whole blocks"
        whole = "where only whole blocks move, so no attribute changes owner"
        if blocks(candidate, False) != blocks(prompt, False):
            return f"it moved words between XML blocks, {whole}"
        if blocks(candidate, True) != blocks(prompt, True):
            return f"it moved words inside an XML block, {whole}"
        return "it changed the text outside the XML blocks, not only the order of whole pieces"
    return "it changed words, not only their order"


def is_reorder(candidate: str, prompt: str, style: str) -> bool:
    """Whether `candidate` is `prompt` with its pieces moved and nothing else -
    the variant that isolates order, so a grid row of anything more would lie."""
    kept, moved = _reordered(candidate, prompt, style)
    return kept and moved is not None


def _order_items(text: str) -> Counter[str]:
    """The items the order rule reads in `text` (`order_items`: its comma
    items, each stretch of lettering one item), as the rules tell them apart
    (`_norm`: a closing full stop is no other tag), in any order: equal for a
    reorder that moved them whole."""
    return Counter(_norm(item) for item in order_items(text))
