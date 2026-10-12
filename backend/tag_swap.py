"""Swap tags: the rewrite "Redo with tags swapped" makes to a gallery record's prompt.

An editing tool, not an analyser rule: nothing here is a finding, so the
analyser's `source`/`cite` contract does not apply, and nothing here is a
checkpoint's profile key. Pure functions over text; the route in `api_swap.py`
reads the records and queues the cells.

A row is "find these comma items, put this in their place". The unit is the
comma item (`prompt_text.comma_items`, the chips' split), compared through
`prompt_text._norm` - case, whitespace, `_` for a space and `\\(` for `(` do not
make another tag. What the user sees in the preview is what runs:

- Every row matches the *original* prompt, at the same time, and replacements
  are never matched again, so `red hair -> blue hair` beside `blue hair -> red
  hair` swaps the two.
- Strict: a row whose finds are not all in the prompt makes the image
  `missing`, and a missing image is not queued. No partial application.
- A tag item is matched whole, never by substring (`hair` is not in `red
  hair`). An A1111 weight wrapper - `(tag:1.2)`, `((tag))`, `[tag]` - is
  looked through only where it wraps the whole item, and is put back round the
  replacement. The replacement follows the matched item's spelling: underscores
  for spaces where the item used them, escaped parentheses where it escaped
  them.
- Inside a prose item (`prompt_text._is_prose`) a one-tag row also matches as
  a phrase, bounded by lookarounds on the find's own edges (a find that starts
  or ends with a word character must not continue a word there), never inside
  lettering (`rendered_spans`). A find with Han or kana in it matches as a
  plain substring: those scripts put no spaces between words. A many-to-one
  row that would need a phrase rewritten is `by_hand`: which phrase to keep
  and where the others' words went is not a rewrite a rule can make.
- Many to one: the replacement goes where the first matched item sits in the
  prompt; the other matched items go, with the separator before each.
  A replacement already in the prompt (as an item no row matched) is not
  written twice; an empty replacement removes the matched items.
- An XML or JSON prompt (NewBie's) is `structured` and never rewritten.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .prompt_text import _CJK, Item, _is_prose, _norm, comma_items, detect_style, rendered_spans

# Per row: a guard rail, not a style rule. A row is meant to be a handful of
# tags read as one thing ("red hair, long hair").
MAX_FINDS = 20


@dataclass(frozen=True)
class Row:
    """One swap: the finds as typed, their comparable keys, the replacement."""

    finds: tuple[str, ...]
    keys: tuple[str, ...]
    replace: str

    def describe(self) -> str:
        return f"{', '.join(self.finds)} → {self.replace or '(removed)'}"


def parse_row(find: str, replace: str = "") -> Row:
    """A row from the dialog's two boxes. Raises ValueError with the reason."""
    finds: list[str] = []
    keys: list[str] = []
    for item in comma_items(find):
        key = _norm(_unwrap(item.text)[0])
        if key and key not in keys:
            finds.append(item.text)
            keys.append(key)
    if not keys:
        raise ValueError("a row needs a tag to find")
    if len(keys) > MAX_FINDS:
        raise ValueError(f"a row finds at most {MAX_FINDS} tags; this one has {len(keys)}")
    return Row(tuple(finds), tuple(keys), replace.strip())


def parse_rows(pairs: list[tuple[str, str]]) -> list[Row]:
    """Every row, refusing a tag that two rows both find: matched at the same
    time, the two replacements would fight over one item."""
    rows = [parse_row(find, replace) for find, replace in pairs]
    seen: dict[str, str] = {}
    for row in rows:
        for find, key in zip(row.finds, row.keys):
            if key in seen:
                raise ValueError(f"{seen[key]} is in two rows' Find")
            seen[key] = find
    return rows


# ------------------------------------------------------------------ one item

# `(tag:1.2)`: the weight form. `((tag))` / `[tag]`: emphasis by nesting, the
# same count of marks on both sides. The core may hold escaped parentheses
# (`red hair \(cosplay\)`) and nothing else that brackets.
_WEIGHTED = re.compile(r"^\((?P<core>.+?(?<!\\)):\s*(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$", re.S)
_ROUND = re.compile(r"^(?P<open>\(+)(?P<core>.+?(?<!\\))(?P<close>\)+)$", re.S)
_SQUARE = re.compile(r"^(?P<open>\[+)(?P<core>.+?(?<!\\))(?P<close>\]+)$", re.S)


def _bare(core: str) -> bool:
    plain = core.replace(r"\(", "").replace(r"\)", "")
    return not any(mark in plain for mark in "()[]")


def _unwrap(text: str) -> tuple[str, str, str]:
    """(core, before, after): `text` without the one weight wrapper round the
    whole of it, if there is one, and the wrapper's two halves."""
    match = _WEIGHTED.match(text)
    if match and _bare(match["core"]):
        return match["core"].strip(), "(", text[match.end("core"):]
    for pattern in (_ROUND, _SQUARE):
        match = pattern.match(text)
        if match and len(match["open"]) == len(match["close"]) and _bare(match["core"]):
            return match["core"].strip(), match["open"], match["close"]
    return text, "", ""


def _styled(replacement: str, core: str, before: str, after: str) -> str:
    """The replacement in the matched item's spelling, in its wrapper.

    Underscores where the item joined its words with them and had no space;
    escaped parentheses where it escaped its own. A replacement of several
    comma items takes the convention and the wrapper item by item.
    """
    underscores = "_" in core and not re.search(r"\s", core)
    escaped = r"\(" in core or r"\)" in core
    out = []
    for item in comma_items(replacement):
        text = item.text
        if underscores:
            text = re.sub(r"\s+", "_", text)
        if escaped:
            text = re.sub(r"(?<!\\)([()])", r"\\\1", text)
        out.append(f"{before}{text}{after}")
    return ", ".join(out)


def _phrase(find: str) -> re.Pattern[str]:
    """`find` as a phrase inside prose: its words with any run of spaces or
    underscores between them, `(` with or without its backslash, case
    ignored; bounded only on an edge that is a word character. Han or kana:
    a plain substring."""
    text = find.replace(r"\(", "(").replace(r"\)", ")").strip()
    if _CJK.search(text):
        return re.compile(re.escape(text))
    words = [re.escape(word) for word in re.split(r"[\s_]+", text) if word]
    body = r"[\s_]+".join(words).replace(r"\(", r"\\?\(").replace(r"\)", r"\\?\)")
    lead = r"(?<!\w)" if re.match(r"\w", text[0]) else ""
    tail = r"(?!\w)" if re.match(r"\w", text[-1]) else ""
    return re.compile(lead + body + tail, re.I)


# ------------------------------------------------------------------ one text

Status = Literal["changed", "unchanged", "missing", "by_hand", "structured"]


@dataclass
class Outcome:
    """What the rows did to one text.

    `spans` are (start, end) offsets into `text` of what was written in -
    the page marks them, and adds nothing of its own. `removed` is what the
    rows took out of the original, as it was written there (a weight and
    all) and in its order - items dropped or written over, phrases replaced
    - which the new text alone cannot show; an item whose tag the new text
    still has (an identity or a reorder row) is not taken out. `missing` names the
    finds a row needed and the text lacks; `by_hand`, the rows that would
    have had to rewrite a phrase many-to-one.
    """

    text: str
    status: Status
    spans: list[tuple[int, int]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    by_hand: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Read:
    item: Item
    prose: bool
    core: str
    before: str
    after: str
    key: str


def swap(text: str | None, rows: list[Row]) -> Outcome:
    """`rows` applied to `text` at once, strictly (see the module's notes)."""
    text = text or ""
    if text.strip() and detect_style(text) in ("xml", "json"):
        return Outcome(text, "structured")
    reads = []
    for item in comma_items(text):
        prose = _is_prose(item.text)
        core, before, after = (item.text, "", "") if prose else _unwrap(item.text)
        reads.append(_Read(item, prose, core, before, after, _norm(core)))
    lettering = rendered_spans(text)

    def lettered(start: int, end: int) -> bool:
        return any(a < end and start < b for a, b in lettering)

    # Per row: the whole items each key matched, and the phrase matches.
    missing: list[str] = []
    by_hand: list[str] = []
    whole: list[list[int]] = []
    phrases: list[list[tuple[int, int, int]]] = []
    for row in rows:
        hits = [[i for i, read in enumerate(reads) if read.key == key] for key in row.keys]
        found: list[tuple[int, int, int]] = []
        if len(row.keys) == 1:
            pattern = _phrase(_unwrap(row.finds[0])[0])
            cjk = bool(_CJK.search(row.finds[0]))
            for i, read in enumerate(reads):
                if i in hits[0] or not (read.prose or cjk):
                    continue
                for match in pattern.finditer(read.item.text):
                    start, end = read.item.start + match.start(), read.item.start + match.end()
                    if not lettered(start, end):
                        found.append((i, start, end))
            if not hits[0] and not found:
                missing.append(row.finds[0])
        else:
            lacking = [find for find, hit in zip(row.finds, hits) if not hit]
            if lacking:
                prose_has = [
                    find for find in lacking
                    if any(
                        not lettered(read.item.start + m.start(), read.item.start + m.end())
                        for read in reads if read.prose
                        for m in _phrase(_unwrap(find)[0]).finditer(read.item.text)
                    )
                ]
                if len(prose_has) == len(lacking):
                    by_hand.append(row.describe())
                else:
                    missing.extend(find for find in lacking if find not in prose_has)
        whole.append(sorted({i for hit in hits for i in hit}))
        phrases.append(found)
    # A phrase match inside another row's (`hair` in the `red hair` a row
    # rewrites) is not applied - `_put_phrases` keeps the earlier - so a row
    # left with no item and no phrase of its own is missing, as strictness
    # has it, rather than queued without it.
    applied = _applied_phrases(phrases)
    for r, (row, indexes, found) in enumerate(zip(rows, whole, phrases)):
        if len(row.keys) == 1 and found and not indexes and r not in applied:
            missing.append(row.finds[0])
    if missing:
        return Outcome(text, "missing", missing=missing, by_hand=by_hand)
    if by_hand:
        return Outcome(text, "by_hand", by_hand=by_hand)

    matched = {i for indexes in whole for i in indexes}
    untouched = {read.key for i, read in enumerate(reads) if i not in matched}
    # Each item's fate: None, dropped; a string, what it becomes.
    fate: dict[int, str | None] = {}
    marked: set[int] = set()
    notes: list[str] = []
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for row, indexes, found in zip(rows, whole, phrases):
        if indexes:
            single = len(comma_items(row.replace)) == 1
            if not row.replace:
                for i in indexes:
                    fate[i] = None
            elif single and _norm(row.replace) in untouched:
                for i in indexes:
                    fate[i] = None
                notes.append(f"already had {row.replace}")
            else:
                # One to one: every occurrence becomes the replacement. Many
                # to one: the first item matched, in the prompt's order, does.
                targets = indexes if len(row.keys) == 1 else indexes[:1]
                for i in indexes:
                    if i in targets:
                        read = reads[i]
                        fate[i] = _styled(row.replace, read.core, read.before, read.after)
                        marked.add(i)
                    else:
                        fate[i] = None
        for i, start, end in found:
            edits.setdefault(i, []).append((start, end, row.replace))

    out: list[str] = []
    spans: list[tuple[int, int]] = []
    # (as written, key): an item written over, or a phrase replaced, which
    # the new text may still hold (an identity row, a reorder).
    removed: list[tuple[str, str | None]] = []
    length = 0

    def put(piece: str, mark: bool = False) -> None:
        nonlocal length
        if mark and piece:
            spans.append((length, length + len(piece)))
        out.append(piece)
        length += len(piece)

    if not reads:
        return Outcome(text, "unchanged")
    put(text[: reads[0].item.start])
    first = True
    for i, read in enumerate(reads):
        if i in fate:
            removed.append((read.item.text, read.key))
            if fate[i] is None:
                continue
        if not first:
            put(text[reads[i - 1].item.end: read.item.start])
        first = False
        if i in fate:
            put(fate[i] or "", mark=i in marked)
        elif i in edits:
            _put_phrases(text, read.item, edits[i], put, removed.append)
        else:
            put(read.item.text)
    put(text[reads[-1].item.end:])
    new = "".join(out)
    if not new.strip():
        new = ""
        spans = []
    if new == text:
        return Outcome(new, "unchanged", spans=spans, notes=notes)
    # Only what is gone: an item whose tag the new text still has as an item,
    # or a phrase put back as it was, was not taken out.
    kept = {_norm(_unwrap(item.text)[0]) for item in comma_items(new)}
    gone = [written for written, key in removed if key is None or key not in kept]
    return Outcome(new, "changed", spans=spans, removed=gone, notes=notes)


def _applied_phrases(phrases: list[list[tuple[int, int, int]]]) -> set[int]:
    """The rows at least one of whose phrase matches `_put_phrases` applies:
    per item, by position, an overlapping match losing to the earlier one
    (the same order `_put_phrases` takes, ties in row order)."""
    by_item: dict[int, list[tuple[int, int, int]]] = {}
    for r, found in enumerate(phrases):
        for i, start, end in found:
            by_item.setdefault(i, []).append((start, end, r))
    applied: set[int] = set()
    for edits in by_item.values():
        position = -1
        for start, end, r in sorted(edits, key=lambda edit: (edit[0], edit[1])):
            if start < position:
                continue
            applied.add(r)
            position = end
    return applied


def _put_phrases(text: str, item: Item, edits: list[tuple[int, int, str]], put, took) -> None:
    """One prose item with its phrase matches replaced, in place, each phrase
    replaced by something else handed to `took` as (as written, None). Overlapping matches (two rows' finds in one
    phrase) keep the earlier one. A phrase removed takes one space beside it
    with it, so no double space is left."""
    position = item.start
    for start, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1])):
        if start < position:
            continue
        if _norm(text[start:end]) != _norm(replacement):
            took((text[start:end], None))
        if not replacement:
            if start > item.start and text[start - 1] == " " and (
                end >= item.end or text[end] in " ,.;:!?"
            ):
                start -= 1
            elif end < item.end and text[end] == " ":
                end += 1
        put(text[position:start])
        put(replacement, mark=True)
        position = end
    put(text[position: item.end])


# ------------------------------------------------------------------ one record


@dataclass
class RecordSwap:
    """A record's rewrite: the field's outcome and the variables to go with it."""

    outcome: Outcome
    prompt: str
    negative_prompt: str | None
    variables: dict[str, str] | None


def swap_record(
    prompt: str,
    negative_prompt: str | None,
    variables: dict[str, str] | None,
    rows: list[Row],
    field_name: Literal["prompt", "negative_prompt"] = "prompt",
) -> RecordSwap:
    """`rows` applied to one record's prompt or negative prompt.

    A variable's value is rewritten too where it is part of the text swapped,
    so a `{hair}` grid's label agrees with the prompt it now runs; a value
    that lacks a find is left as it is, and so is a comparison's `variant`,
    which is a version's name, not its text.
    """
    original = prompt if field_name == "prompt" else (negative_prompt or "")
    outcome = swap(original, rows)
    new_vars = dict(variables) if variables else None
    if outcome.status == "changed" and new_vars:
        for name, value in new_vars.items():
            if name == "variant" or not isinstance(value, str) or not value or value not in original:
                continue
            rewritten = swap(value, rows)
            if rewritten.status == "changed":
                new_vars[name] = rewritten.text
    if field_name == "prompt":
        return RecordSwap(outcome, outcome.text if outcome.status == "changed" else prompt,
                          negative_prompt, new_vars)
    negative = outcome.text if outcome.status == "changed" else negative_prompt
    return RecordSwap(outcome, prompt, negative or None, new_vars)
