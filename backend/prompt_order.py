"""Documented prompt order, tag placement and category coverage."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any, NamedTuple

from . import prompt_text
from .model_defaults import _profile_for
from .schemas import DocumentedOrder

# Past this many comma items the order check stands down rather than run a
# quadratic placement over what is no longer a tag list anyone typed.
_MAX_ORDERED = 256


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
                local.setdefault(prompt_text._norm(tag), index[slot])
    if order.get("suffix"):
        for tag in helper.get("recommended_suffix", []):
            local.setdefault(prompt_text._norm(tag), index[order["suffix"]])
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


def _place(
    item: prompt_text.Item, spec: _OrderSpec, lookup: dict[str, str] | None, *, lettering: bool = False
) -> frozenset[int]:
    """The slots `item` may occupy: one when anything says which, else all
    the slots a Danbooru category could have put it in.

    What is known comes first - the guide's own groups, the year form, then
    the index's category - so a long name the index knows ("jojo no kimyou
    na bouken", a copyright) is never taken for prose by its shape. An item
    holding `lettering` is no name, whatever its words: only its shape
    places it.
    """
    # A grid placeholder stands for values the grid fills in, so it may take
    # any slot at all: never out of order beside another item, and never
    # "not placed" either (`unplaced_items`) - it is no tag to look up.
    if not lettering and prompt_text.is_grid_variable(item.text):
        return frozenset(range(len(spec.steps)))
    key = prompt_text._norm(item.text)
    if not lettering:
        if key in spec.local:
            return frozenset({spec.local[key]})
        if spec.year is not None and prompt_text._YEAR_TAG.fullmatch(key):
            return frozenset({spec.year})
        if lookup is not None:
            category = lookup.get(item.text) or lookup.get(key)
            if category in spec.categories:
                return frozenset({spec.categories[category]})
    # Prose sits in the general slot on every hybrid checkpoint's order -
    # Kohaku's is literally "general tags or prose". A prose-shaped item a
    # name could still be ("the legend of zelda") counts as prose only once
    # the index has been asked and did not know it; lettering, which is
    # never asked about, by its shape alone.
    if prompt_text._is_sentence(item.text) or ((lettering or lookup is not None) and prompt_text._is_prose(item.text)):
        return frozenset({spec.general})
    return spec.unknown


def _places(
    spec: _OrderSpec,
    items: list[prompt_text.Item],
    lettered: list[bool],
    lookup: dict[str, str] | None,
) -> list[frozenset[int]]:
    """Each of `items`' slots, `_place`d with `lookup` (`_category_lookup`):
    the one placement the order rule, `reorder_tags`, `unplaced_items` and
    `order_slots` all read."""
    return [
        _place(item, spec, lookup, lettering=held)
        for item, held in zip(items, lettered, strict=True)
    ]


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


def _undecided(places: list[frozenset[int]], keys: list[Any]) -> list[int]:
    """The items whose missing slot decides the order: each is in a pair
    that some slot it could take puts out of the documented order and
    another does not, which `_violation`, flagging only what is out of
    order beyond doubt, therefore passes. Only an item with more than one
    slot can be in such a pair; one whose every slot sits in order beside
    the rest ("zzkx oc" between 1girl and masterpiece), or out of it beyond
    doubt, is not, and neither is a lone item. Nor is a pair of copies - two
    items with one key, which `_place` places alike under any categories,
    so they take one slot whatever it is and are never out of order beside
    each other: "1girl, zzkx oc, zzkx oc, masterpiece" is settled.

    Where none is, the rules' answer is the same whatever category each
    item has, and so is `_ordered`'s order: a pair it swaps was out of order
    beyond doubt, and one it keeps was in order beyond doubt, or is a pair
    of copies, in order whatever their slot.
    """
    low = [min(place) for place in places]
    high = [max(place) for place in places]
    undecided: set[int] = set()
    for a in range(len(places)):
        for b in range(a + 1, len(places)):
            if keys[a] != keys[b] and high[a] > low[b] and low[a] <= high[b]:
                undecided.update(index for index in (a, b) if len(places[index]) > 1)
    return sorted(undecided)


def _reordered(text: str, items: list[prompt_text.Item], places: list[frozenset[int]]) -> str | None:
    """`text` with `items` moved into the order `places` allow, or None where
    nothing moves - or where the move would change what is quoted: a stray
    mark (12" figure) moved ahead of a quotation would pair with its opening
    one, and the page would offer that as the same words reordered.

    A grid placeholder (`{hair}`) keeps its position and the other items are
    ordered around it: it takes every slot (`_place`), so `_ordered` would
    take it first, ahead of the Person count, and every value the grid fills
    in would then sit where Check of the filled-in prompt flags it. Its
    values' slots are the grid's to say, so the user's placement stands.
    """
    fixed = [prompt_text.is_grid_variable(item.text) for item in items]
    free = [index for index, held in enumerate(fixed) if not held]
    moved = iter(free[position] for position in _ordered([places[index] for index in free]))
    order = [index if held else next(moved) for index, held in enumerate(fixed)]
    if order == list(range(len(items))):
        return None
    moved = prompt_text._rebuild(text, items, [items[index].text for index in order])
    if Counter(prompt_text.QUOTED.findall(moved)) != Counter(prompt_text.QUOTED.findall(text)):
        return None
    return moved


def _placeable(
    spec: _OrderSpec | None,
    style: prompt_text.Style,
    text: str,
    items: list[prompt_text.Item],
    lettering: list[tuple[int, int]],
) -> bool:
    """Whether the order rules hold this text's comma items to the documented
    order at all - the one answer `order_checked`, `reorder_tags` and the
    order rule share, so the page never shows an order as checked that the
    rule stood down from.

    Not where the guide gives no slots, not on XML or JSON, not past
    `_MAX_ORDERED` items, and not where a line break or a fullwidth comma
    splits what the chips read as one comma item: such a list reads as tags,
    but nothing in it could be placed. `items` are `_units`, each stretch of
    lettering one item, and the lettering is one piece of its item: a "，"
    inside "你好，世界" splits nothing, while the one in '"hi"，1girl' still
    splits an item whose 1girl nothing could place.
    """
    solid = prompt_text._blank(text, lettering, "x")
    return (
        spec is not None
        and style not in ("xml", "json")
        and len(items) <= _MAX_ORDERED
        and all(len(prompt_text._pieces(solid[item.start:item.end])) <= 1 for item in items)
    )


def _tag_items(text: str) -> tuple[list[prompt_text.Item], list[bool], list[tuple[int, int]]]:
    """The order rules' reading of `text` outside `check`: its `_units`,
    which of them hold lettering, and the lettering - as `check` reads it
    (`_read`), a participle's words stopped at a negation, not as far as
    `rendered_targets` lets the lettering reach, so both place one list."""
    quoted = prompt_text._quoted_spans(text)
    lettering = quoted + prompt_text._unquoted_text(text, quoted)
    items = prompt_text._units(text, prompt_text.comma_items(text), lettering)
    return items, prompt_text._held(items, lettering), lettering


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
    (`_OrderSpec`), a stretch of lettering one item with its commas - and
    only on a text that is not XML or JSON, not past `_MAX_ORDERED` items and
    split into its items by commas (`_placeable`).
    A FLUX sentence's "subject first", klein's four steps and NewBie's blocks
    are documented but read by nothing here, and the page must not let an
    empty findings list pass for a checked order.
    """
    spec = _order_spec(_guide(model_id))
    items, _, lettering = _tag_items(prompt)
    return _placeable(spec, prompt_text.detect_style(prompt), prompt, items, lettering)


def reorder_tags(model_id: str, prompt: str, categories: dict[str, str] | None) -> str | None:
    """The same comma items, same spelling, moved into the documented order.

    None where the order is not checked (`order_checked`) or nothing has to
    move. Prose pieces all sit in the general slot and keep their order, so
    a caption that ends in tags - Illustrious's own natural-language example
    does - moves only the tags. Nothing is added or
    removed, and the text between items stays where it was, so the result is
    a pure permutation: the variant that isolates order on a comparison grid.
    A stretch of lettering moves whole, commas and all, and every quoted
    stretch comes out as it went in (`_reordered`).
    """
    reading = _order_reading(model_id, prompt)
    if reading is None:
        return None
    spec, items, lettered = reading
    return _reordered(
        prompt, items, _places(spec, items, lettered, prompt_text._category_lookup(categories))
    )


def unplaced_items(model_id: str, prompt: str, categories: dict[str, str] | None) -> list[str]:
    """The items the order rules read in `prompt` (`order_items`) that they
    could not place where it decides the order, placing with `categories`
    as `check` and `reorder_tags` do: each keeps every slot a category could
    give it, and some of those slots put it out of order beside another item
    while others do not (`_undecided`), so it may sit out of order
    unflagged.

    Such an item is in none of the guide's own groups, is not prose `_place`
    puts in the general slot, and has no category - whether the tag index
    was never asked about it (offline, a reply that did not read, past the
    lookup's cap: `PromptAnalysis.categories_detail`; a prose prompt, which
    is never looked up; lettering or an @ name, which never are) or was asked
    and gave it none it could use (an OC's name, a trigger word, a typo; a
    deprecated tag, or one past the vocabulary's cutoff). Not an item every
    slot of which gives the same answer - "zzkx oc" on Animagine between
    1girl and masterpiece, in order whatever it is, or ahead of 1girl, out
    of it whatever it is - nor a lone item, which no order can be wrong
    about: the order rule and `reorder_tags` read fewer than two items not
    at all. Nor does an item's copy, written the same, make it one: the two
    take one slot, whatever it is, so "1girl, zzkx oc, zzkx oc, masterpiece"
    is settled. Were it said of every unplaced item, one-tag prompts and
    "1girl, zzkx oc, masterpiece" would read as partly checked though their
    order was settled. Empty where the guide gives no slots.
    """
    spec = _order_spec(_guide(model_id))
    items, lettered, _ = _tag_items(prompt)
    if spec is None or len(items) < 2:
        return []
    places = _places(spec, items, lettered, prompt_text._category_lookup(categories))
    # A grid placeholder is left out: it takes every slot (`_place`), so it
    # would pair with each item as undecided and name itself and them.
    kept = [
        index for index, (item, held) in enumerate(zip(items, lettered, strict=True))
        if held or not prompt_text.is_grid_variable(item.text)
    ]
    # What `_place` reads: an item's text as typed, and whether it holds
    # lettering. Two items alike in both are placed alike by any categories.
    keys = [(items[index].text, lettered[index]) for index in kept]
    undecided = _undecided([places[index] for index in kept], keys)
    return [items[kept[index]].text for index in undecided]


def places_every_item(model_id: str, prompt: str, categories: dict[str, str] | None) -> bool:
    """Whether the order rules decide `prompt`'s order whatever category each
    item has, placing with `categories` as `check` and `reorder_tags` do
    (`unplaced_items` empty): every item has one slot, or none left open
    where it decides the order. Where they read the order (`order_checked`),
    `reorder_tags`' order - or its "nothing to move" - is then the documented
    order itself, not one order it allows (but beside a grid placeholder,
    which it leaves where it was: `holds_grid_variable`); where they read it and this is
    false, the order was only partly checked, which the analysis says and
    the page shows (`PromptAnalysis.order_detail`).
    """
    return not unplaced_items(model_id, prompt, categories)


def holds_grid_variable(model_id: str, prompt: str) -> bool:
    """Whether an item the order rules read in `prompt` is a grid placeholder
    (`{hair}`): it is never "not placed" (`unplaced_items`), but where it
    sits is the user's, which `reorder_tags` keeps - one order the
    documented order allows, not the only one, for the critic's reorder."""
    if _order_spec(_guide(model_id)) is None:
        return False
    items, lettered, _ = _tag_items(prompt)
    return any(
        not held and prompt_text.is_grid_variable(item.text)
        for item, held in zip(items, lettered, strict=True)
    )


def _order_reading(
    model_id: str, prompt: str, least: int = 2
) -> tuple[_OrderSpec, list[prompt_text.Item], list[bool]] | None:
    """The order rule's reading of `prompt` outside `check` - its slots, its
    items (`_units`) and which hold lettering - or None where it reads no
    order among them: no slots in the guide, fewer than `least` items, or not
    `_placeable` - the gate the order rule applies to its `_Context`, for
    `reorder_tags` and `order_slots`. `least` is two for an order, which a
    lone item cannot be out of; one for placement (`order_slots`), where a
    lone item still has the step it would take beside any other."""
    spec = _order_spec(_guide(model_id))
    items, lettered, lettering = _tag_items(prompt)
    if spec is None or len(items) < least:
        return None
    if not _placeable(spec, prompt_text.detect_style(prompt), prompt, items, lettering):
        return None
    return spec, items, lettered


def reads_order(model_id: str, prompt: str) -> bool:
    """Whether the order rule reads an order among `prompt`'s items at all
    (`order_checked`, and two items or more): where `reorder_tags` and
    `order_slots` answer anything. Needs no categories, so a caller asks it
    before looking any up - a prompt it is False for would cost the tag index
    a request for nothing."""
    return _order_reading(model_id, prompt) is not None


def places_items(model_id: str, prompt: str) -> bool:
    """Whether `order_slots` answers anything for `prompt`: `reads_order`,
    but a lone item too - placement needs no second item, only an order
    does. Needs no categories, asked before any are looked up."""
    return _order_reading(model_id, prompt, least=1) is not None


def order_slots(
    model_id: str, prompt: str, categories: dict[str, str] | None
) -> list[tuple[str, str | None]] | None:
    """Each comma item of `prompt` - `comma_items`, the chips' split - with
    the step of the documented order the order rule places it in, or None.

    The rule's own reading, placing with `categories` as `check`,
    `reorder_tags` and `unplaced_items` do (`_places`), never a second one:
    None for the whole prompt where the rules place nothing in it
    (`places_items` false: no slots in the guide, no items, or
    `order_checked` false). A lone item is placed - its step is the one it
    would take beside any other, which the Tags helper's caret line reads to
    say where the next tag goes - though no order rule reads it. And None for an item that holds lettering,
    which the rules read as a piece of a longer item, commas and all, or that
    keeps more than one slot: an item the guide's groups do not know and the
    tag index gave no category (never asked - offline, past the lookup's
    cap, an @ name, a prose prompt's clause - or asked and given none it
    could use). Such an item has no one place, and the chip shows none
    rather than a guess.
    """
    reading = _order_reading(model_id, prompt, least=1)
    if reading is None:
        return None
    spec, items, lettered = reading
    places = _places(spec, items, lettered, prompt_text._category_lookup(categories))
    slots: list[tuple[str, str | None]] = []
    unit = 0
    for item in prompt_text.comma_items(prompt):
        # The rules' item holding this comma item: itself, or the stretch of
        # lettering it is a piece of (`_units`).
        while items[unit].end < item.end:
            unit += 1
        place = places[unit]
        alone = len(place) == 1 and not lettered[unit]
        slots.append((item.text, spec.steps[next(iter(place))] if alone else None))
    return slots


def order_items(text: str) -> list[str]:
    """The items the order rules read in `text`, as typed: its comma items,
    each stretch of lettering one item, commas and all (`_units`). A reorder
    that moves these whole is the same list to the rules; one that parts
    lettering at a comma inside it - "《你好, 世界》", which no quotation
    mark holds together - is another list, whose halves the rules would
    place apart."""
    return [item.text for item in _tag_items(text)[0]]


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
    could be are left out, items the guide's own groups already place, which
    need no lookup, lettering, which no rule reads as a name, and a grid
    placeholder (`{hair}`), which stands for the grid's values.
    """
    guide = _guide(model_id)
    spec = _order_spec(guide)
    placing = spec is not None and any(slot != spec.general for slot in spec.categories.values())
    if not (placing or "artist_prefix" in _analysis(guide)):
        return []
    if prompt_text.detect_style(prompt) in ("xml", "json"):
        return []
    wanted: list[str] = []
    artist_prefix = _analysis(guide).get("artist_prefix", {}).get("prefix", "@")
    items, lettered, _ = _tag_items(prompt)
    for item, held in zip(items, lettered, strict=True):
        key = prompt_text._norm(item.text)
        if (
            held or prompt_text._is_sentence(item.text) or prompt_text.is_grid_variable(item.text)
            or item.text.lower().startswith(artist_prefix.lower()) or item.text in wanted
        ):
            continue
        if spec is not None and (
            key in spec.local or (spec.year is not None and prompt_text._YEAR_TAG.fullmatch(key))
        ):
            continue
        wanted.append(item.text)
    return wanted


def lookup_items(model_id: str, prompt: str) -> list[str]:
    """What a check of `prompt` asks the tag index about: `category_items`
    where the prompt reads as tags, or tags and prose. Where it reads as
    prose, only on a guide whose rules read artists (Anima, whose card says
    "Use tags, natural language, or both": its prefix and tags ahead of two
    sentences read as prose, and an artist among them went unchecked while
    the panel said nothing to flag) - and there only the tag-shaped items,
    since the artist rule reads a whole comma item and a clause is no
    artist's name; asking about every clause would cost the index a request
    for each one Check. Prose is never looked up for the order rules: their
    "partly checked" names a prose prompt's clauses as never asked
    (`prompt_check.order_detail`), and no guide that states an order reads artists
    (`tests/test_prompt_rules.py` holds that). Nothing for XML or JSON."""
    style = prompt_text.detect_style(prompt)
    if style in ("tags", "hybrid"):
        return category_items(model_id, prompt)
    if style != "prose" or not wants_artist_categories(model_id):
        return []
    return [item for item in category_items(model_id, prompt) if not prompt_text._is_prose(item)]


def _named(name: str, prefix: str = "@") -> re.Pattern[str]:
    """`name` - an item the tag index answered for - written anywhere in a
    text as words of their own, exempting @ only for that prefix: any case, a space or
    an underscore between its words, its parentheses escaped or not."""
    words = r"[\s_]+".join(re.escape(word) for word in prompt_text._norm(name).split(" "))
    words = words.replace(r"\(", r"\\?\(").replace(r"\)", r"\\?\)")
    # Only Anima's prefix exempts an inline name. A foreign @ on RouWei
    # must not hide an artist the 'by ' rule never read.
    ahead = r"(?<![@\w])(?<!@\s)" if prefix == "@" else r"(?<!\w)"
    return re.compile(rf"{ahead}{words}(?!\w)", re.IGNORECASE)


def artists_read(
    model_id: str, text: str, asked: Iterable[str], artists: Iterable[str]
) -> bool:
    """Whether the artist rule, reading `text` with the categories a lookup
    of `asked` answered with - `artists` the items it called artists - reads
    every artist `text` may name: the critic's rewrite, checked with its
    analysis's own categories and no lookup of its own (`prompt_check.lint`).

    Every item a Danbooru name could be (`category_items`, a prose text's
    clauses included, which a check of it never asks about) must be among
    `asked`, read as the rules match an item (`_norm`); and no artist the
    lookup named may stand in `text` other than as a whole item the rule
    reads or after its @ - "a painting in the style of wlop", "A girl
    painted by wlop smiles." keep the name with no @ and no finding, and the
    page would otherwise call the prompt's finding on it fixed.
    """
    known = {prompt_text._norm(item) for item in asked}
    if not all(prompt_text._norm(item) in known for item in category_items(model_id, text)):
        return False
    named = {prompt_text._norm(name) for name in artists}
    prefix = _analysis(_guide(model_id)).get("artist_prefix", {}).get("prefix", "@")
    patterns = [_named(name, prefix) for name in named]
    if prompt_text.detect_style(text) in ("xml", "json"):
        # The rule reads no item of either.
        return not any(pattern.search(text) for pattern in patterns)
    items, lettered, _ = _tag_items(text)
    for item, held in zip(items, lettered, strict=True):
        if (
            not held and item.text.lower().startswith(prefix.lower())
            and prompt_text._norm(item.text[len(prefix):]) in named
        ):
            continue
        if not held and not prompt_text._is_sentence(item.text) and prompt_text._norm(item.text) in named:
            continue  # The rule reads it, and flags it unless it has its @.
        if any(pattern.search(item.text) for pattern in patterns):
            return False
    return True


def scene_ahead(model_id: str, text: str) -> str | None:
    """The guide's scene block - NewBie's <general_tags> - where `text` puts
    it ahead of a <character_N> block at the top level, against the order
    the guide states (each character's block, then the shared scene); None
    where it does not, or the guide states no such order. The order rule
    never reads XML (`order_checked`); this is what a critic's reorder of
    the blocks is held to."""
    analysis = _analysis(_guide(model_id))
    entry = analysis.get("xml")
    if not isinstance(entry, dict) or not analysis.get("order"):
        return None
    names = [match.group(1) for match in prompt_text._XML_ELEMENT.finditer(text)]
    scene = entry["scene"]
    if scene not in names:
        return None
    later = names[names.index(scene) + 1:]
    return scene if any(re.fullmatch(r"character_\d+", name) for name in later) else None
