"""Chip order slots agree with the rule checker and reorder helper."""
from __future__ import annotations

from collections import Counter
import re

import pytest

from backend import guides, prompt_rules

from tests.prompt_rules_helpers import (
    ANIMA,
    ANIMAGINE,
    CATEGORIES,
    FLUX_DEV,
    ILLUSTRIOUS,
    KOHAKU,
    KREA2,
    NEWBIE,
    OVER_CAP,
    WAI,
)


# ------------------------------------------------------------------ the chips' order slots


def _slots(model_id, prompt, categories=None):
    answer = prompt_rules.order_slots(model_id, prompt, categories)
    if answer is not None:
        # One per chip: the comma items, as the chips split them.
        assert [tag for tag, _ in answer] == [
            item.text for item in prompt_rules.comma_items(prompt)
        ]
    return None if answer is None else [slot for _, slot in answer]


def test_each_chip_gets_the_slot_the_order_rule_places_it_in():
    prompt = "masterpiece, smile, safe, honkai: star rail, firefly, 1girl"
    assert _slots(ANIMAGINE, prompt, CATEGORIES) == [
        "Quality", "Details", "Rating", "Series", "Character", "Person count",
    ]
    # Without the index only the guide's own groups place anything; an item
    # a category would place keeps every slot it could take, and no mark.
    assert _slots(ANIMAGINE, prompt) == [
        "Quality", None, "Rating", None, None, "Person count",
    ]
    # An item the index was asked about and gave no category is no different.
    assert _slots(ANIMAGINE, "1girl, zzkx oc, masterpiece", {}) == [
        "Person count", None, "Quality",
    ]
    # The year form and the score group are the guide's too.
    assert _slots(ANIMAGINE, "1girl, year 2024, high score") == [
        "Person count", "Details", "Quality",
    ]


@pytest.mark.parametrize(
    "model_id, prompt, categories",
    [
        (ANIMAGINE, "masterpiece, smile, safe, honkai: star rail, firefly, 1girl", CATEGORIES),
        (ANIMAGINE, "safe, masterpiece, 1girl, year 2024, outdoors, firefly", CATEGORIES),
        (KOHAKU, guides.KOHAKU_GUIDE["examples"][0], {"absurdres": "meta", "ciloranko": "artist"}),
        (KOHAKU, "sensitive, explicit, nsfw, newest, 1boy,  solo ,\nsky", None),
        (ILLUSTRIOUS, "newest, masterpiece, general, 1girl, looking at viewer",
         {"looking at viewer": "general"}),
        (ILLUSTRIOUS, "A girl standing in a field at dusk, masterpiece, 1girl", None),
        (WAI, "masterpiece, best quality, amazing quality, 1girl, beach", None),
    ],
)
def test_the_slots_are_the_order_the_rules_reorder_to(model_id, prompt, categories):
    # Not a second placement: Check's reorder puts every placed item in
    # the order of the steps its slot names, and the order rule's finding
    # names the same slots.
    steps = prompt_rules.documented_order(model_id).steps
    assert prompt_rules.reads_order(model_id, prompt)
    reordered = prompt_rules.reorder_tags(model_id, prompt, categories)
    after = _slots(model_id, reordered, categories)
    placed = [steps.index(slot) for slot in after if slot is not None]
    assert placed == sorted(placed)
    assert Counter(_slots(model_id, prompt, categories)) == Counter(after)
    finding = prompt_rules.order_finding(model_id, prompt, categories)
    named = re.findall(r'"([^"]+)" \(([^)]+)\)', finding.message)
    slots = dict(prompt_rules.order_slots(model_id, prompt, categories))
    assert named and all(slots[tag] == slot for tag, slot in named)


def test_wai_places_every_item_without_asking():
    # Its categories all share one slot, so an item no group knows has one
    # place whatever it is - and the chips say so with no lookup at all.
    assert prompt_rules.category_items(WAI, "1girl, beach, masterpiece") == []
    assert _slots(WAI, "1girl, beach, masterpiece") == [
        "Subject and scene", "Subject and scene", "Quality",
    ]


def test_prose_takes_the_general_slot_as_the_rule_places_it():
    # A sentence no name could be is prose with or without the index; a
    # phrase a name could still be, only once the index was asked.
    assert _slots(KOHAKU, "1girl, she stands in the rain, masterpiece") == [
        "Person count", "General tags / prose", "Quality",
    ]
    phrase = "1girl, a girl standing in the rain, safe"
    assert _slots(ANIMAGINE, phrase) == ["Person count", None, "Rating"]
    assert _slots(ANIMAGINE, phrase, {}) == ["Person count", "Details", "Rating"]


def test_lettering_and_names_never_looked_up_have_no_slot():
    # Lettering is one item to the rules, commas and all; its pieces are
    # chips of their own, and none of them is a tag with a place.
    assert _slots(ANIMAGINE, '1girl, "hello, world", masterpiece') == [
        "Person count", None, None, "Quality",
    ]
    assert _slots(ANIMAGINE, '1girl, sign reading "OPEN", masterpiece', {}) == [
        "Person count", None, "Quality",
    ]
    # Lettering that reads as prose is placed by its shape - but a chip of
    # it is still no tag.
    assert _slots(ANIMAGINE, '1girl, "the cat and the hat", safe') == [
        "Person count", None, "Rating",
    ]
    # An @ name is never asked about; with categories for everything else,
    # it still has no one slot.
    prompt = "1girl, @ciloranko, smile, masterpiece"
    assert "@ciloranko" not in prompt_rules.category_items(ANIMAGINE, prompt)
    assert _slots(ANIMAGINE, prompt, {"smile": "general"}) == [
        "Person count", None, "Details", "Quality",
    ]


def test_a_lone_item_is_placed_though_no_order_reads_it():
    # One item cannot be out of order, so the order rule and the reorder read
    # none - but it still has the step it would take beside any other, which
    # the Tags helper's caret line reads to say where the next tag goes.
    for prompt, slot in (("1girl", "Person count"), ("masterpiece", "Quality"),
                         ("year 2024", "Details")):
        assert _slots(ANIMAGINE, prompt) == [slot]
        assert prompt_rules.places_items(ANIMAGINE, prompt)
        assert not prompt_rules.reads_order(ANIMAGINE, prompt)
        assert prompt_rules.reorder_tags(ANIMAGINE, prompt, CATEGORIES) is None
    # The same step it takes beside another item.
    assert _slots(ANIMAGINE, "firefly", CATEGORIES) == ["Character"]
    assert _slots(ANIMAGINE, "firefly, masterpiece", CATEGORIES)[0] == "Character"
    # An item no group knows and no category places keeps no one step.
    assert _slots(ANIMAGINE, "zzkx oc") == [None]
    # Anima's @ name takes the Artist step by its spelling, never asked
    # about; a sentence stands anywhere, so it keeps no one step.
    assert _slots(ANIMA, "@nnn yryr") == ["Artist"]
    assert _slots(ANIMA, "1girl, A girl stands in the rain. She smiles.", {}) == [
        "Person count", None,
    ]
    # Where the guide gives no slots, a lone item has none either.
    assert prompt_rules.order_slots("Minthy/RouWei-0.8", "1girl", None) is None


@pytest.mark.parametrize(
    "model_id, prompt",
    [
        # No slots in the guide.
        (FLUX_DEV, "masterpiece, 1girl"),
        (KREA2, "masterpiece, 1girl"),
        ("Minthy/RouWei-0.8", "1girl, smile, masterpiece"),
        (NEWBIE, guides.NEWBIE_GUIDE["examples"][0]),
        # No items at all.
        (ANIMAGINE, ""),
        # Not placeable: what `order_checked` stands down from.
        (ANIMAGINE, "masterpiece\nsafe\n1girl"),
        (ANIMAGINE, "masterpiece，safe，1girl，smile"),
        (ANIMAGINE, '{"scene": "a cafe", "tags": "1girl, masterpiece"}'),
        (ANIMAGINE, "<x>, masterpiece, 1girl"),
        (ANIMAGINE, OVER_CAP),
        (ANIMAGINE, 'masterpiece\n"hello, world", 1girl'),
    ],
)
def test_no_slots_where_the_rules_read_no_order(model_id, prompt):
    assert prompt_rules.order_slots(model_id, prompt, CATEGORIES) is None
    assert prompt_rules.order_slots(model_id, prompt, None) is None
    # What the validate route asks before it looks any category up.
    assert not prompt_rules.places_items(model_id, prompt)
    assert not prompt_rules.reads_order(model_id, prompt)
    assert prompt_rules.reorder_tags(model_id, prompt, CATEGORIES) is None
    if prompt_rules.documented_order(model_id) is not None and len(
        prompt_rules.order_items(prompt)
    ) > 1:
        assert not prompt_rules.order_checked(model_id, prompt)
