"""Contextual tag/prose presentation without changing lexical tag verdicts.

Comma fragments are not independently grammatical: ``old hat`` continues
``a man wearing a red, old hat``. Only an explicitly hybrid guide and the
matching whole prompt enable this reading. The strict dictionary answers
remain separate, so a fragment's cached answer never depends on its context.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Collection

from . import prompt_text
from .tags import is_model_tag


REORDER_PROSE_DETAIL = (
    "The app withheld a reordered version because it would split a natural-language "
    "passage at its commas. Keep that passage together when changing the order."
)


def _special(helper: dict) -> set[str]:
    names = {tag.lower() for group in helper.get("groups", []) for tag in group.get("tags", [])}
    names.update(tag.lower() for tag in helper.get("recommended_suffix", []))
    return names


def _prose_ranges(
    helper: dict, prompt: str, results: list[dict], placed: Collection[str] = (),
) -> list[tuple[int, int]]:
    """Inclusive comma-item ranges whose context reads as natural language.

    An absent prose-shaped fragment anchors its adjacent absent fragments.
    A known name, any spelling ``prompt_text._norm`` maps to a name the
    guide's order places (``placed``: Anima Aesthetic's score tags, which its
    Tags menu does not offer, or ``best_quality`` for ``best quality``), a
    concrete dictionary problem, an explicit artist prefix
    or a grid variable ends that passage. Unchecked fragments do not supply
    evidence of dictionary absence, although a definite sentence needs none.
    """
    items = prompt_text.comma_items(prompt)
    special = _special(helper)
    prefix = (helper.get("artist_prefix") or "").lower()
    quoted = prompt_text._quoted_spans(prompt)
    eligible, anchors = [], []
    for index, item in enumerate(items):
        row = results[index] if index < len(results) else None
        known = is_model_tag(item.text, special, bool(helper.get("year_tag")))
        protected = (
            known or prompt_text.is_grid_variable(item.text)
            or prompt_text._norm(item.text) in placed
            or bool(prefix and item.text.lower().startswith(prefix))
            or bool(row and (row.get("valid") is True or row.get("category")))
            or bool(row and row.get("code") not in (None, "not_found"))
        )
        # A quoted comma belongs to the lettering, including a word that
        # happens to be a model tag. It is text rather than a movable chip.
        lettered = any(low < item.end and high > item.start for low, high in quoted)
        absent = bool(
            row and row.get("valid") is False and row.get("code") in (None, "not_found")
            and row.get("reason") in (None, "not found in Danbooru", "longer than any Danbooru tag")
        )
        sentence = prompt_text._is_sentence(item.text)
        eligible.append(lettered or (not protected and (absent or sentence)))
        anchors.append(lettered or (not protected and (sentence or (absent and prompt_text._is_prose(item.text)))))
    ranges = []
    start = 0
    while start < len(items):
        if not eligible[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(items) and eligible[end + 1]:
            end += 1
        if any(anchors[start:end + 1]):
            ranges.append((start, end))
        start = end + 1
    return ranges


def contextual_parts(
    helper: dict, prompt: str | None, tags: list[str], results: list[dict],
) -> list[dict] | None:
    """Whole-prompt parts alongside unchanged strict results.

    ``tags`` must match the prompt's comma-item prefix. The request may cap
    that prefix at 64 or append grid-value lookups after all prompt items.
    Returned indexes cover the whole prompt; offsets count UTF-16 code units
    so JavaScript can slice the exact raw string, including astral characters.
    """
    if not prompt or not helper.get("hybrid_prompt") or prompt_text.detect_style(prompt) in ("xml", "json"):
        return None
    items = prompt_text.comma_items(prompt)
    count = min(len(items), len(tags))
    if not count or [item.text for item in items[:count]] != tags[:count]:
        return None
    ranges = dict(_prose_ranges(helper, prompt, results[:len(items)]))
    offsets = [0]
    for char in prompt:
        offsets.append(offsets[-1] + (2 if ord(char) > 0xFFFF else 1))
    parts = []
    index = 0
    while index < len(items):
        last = ranges.get(index, index)
        parts.append({
            "kind": "prose" if index in ranges else "tag", "first": index, "last": last,
            "start": offsets[items[index].start], "end": offsets[items[last].end],
        })
        index = last + 1
    return parts


def reorder_problem(
    helper: dict, prompt: str, reordered: str | None, categories: dict[str, str] | None,
    placed: Collection[str] = (),
) -> str | None:
    """Refuse only a candidate that breaks a comma-containing prose passage.

    Check has categories rather than complete lexical verdicts. Treat its
    unknown adjacent fragments conservatively for this safety check; doing
    so can withhold a reorder, never claim that a dictionary name is invalid.
    """
    if not reordered or not helper.get("hybrid_prompt"):
        return None
    items = prompt_text.comma_items(prompt)
    known = prompt_text._category_lookup(categories) or {}
    results = [
        {"valid": True, "category": known[item.text]}
        if item.text in known else
        {"valid": True, "category": known[prompt_text._norm(item.text)]}
        if prompt_text._norm(item.text) in known else {"valid": False, "code": "not_found"}
        for item in items
    ]
    blocks = Counter(
        prompt[items[first].start:items[last].end]
        for first, last in _prose_ranges(helper, prompt, results, placed) if first != last
    )
    if any(reordered.count(block) < count for block, count in blocks.items()):
        return REORDER_PROSE_DETAIL
    return None
