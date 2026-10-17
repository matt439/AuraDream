"""Tag reordering preserves every item and follows documented placement."""
from __future__ import annotations

from collections import Counter
from itertools import pairwise
import itertools

import pytest

from backend import guides, launch, prompt_rules
from backend.model_defaults import guide_for

from tests.prompt_rules_helpers import (
    ANIMA,
    ANIMAGINE,
    CATEGORIES,
    FLUX_DEV,
    ILLUSTRIOUS,
    KOHAKU,
    NEWBIE,
    OVER_CAP,
    WAI,
    only,
    run,
    spanned,
)


# ------------------------------------------------------------------ tag order


def _gaps(prompt: str) -> list[str]:
    items = prompt_rules.comma_items(prompt)
    return [prompt[a.end:b.start] for a, b in pairwise(items)]


def _assert_permutation(original: str, reordered: str) -> None:
    # Same items, same spelling, nothing added or removed, and the text
    # between items where it was.
    before = [item.text for item in prompt_rules.comma_items(original)]
    after = [item.text for item in prompt_rules.comma_items(reordered)]
    assert Counter(before) == Counter(after)
    assert _gaps(original) == _gaps(reordered)


def test_reorder_with_categories_follows_the_card():
    prompt = "masterpiece, smile, safe, honkai: star rail, firefly, 1girl"
    reordered = prompt_rules.reorder_tags(ANIMAGINE, prompt, CATEGORIES)
    assert reordered == "1girl, firefly, honkai: star rail, safe, smile, masterpiece"
    _assert_permutation(prompt, reordered)
    [finding] = only(run(ANIMAGINE, prompt, CATEGORIES), "guide.order")
    assert finding.suggestion == reordered and finding.severity == "info"
    assert spanned(prompt, finding) == "masterpiece"
    assert "(Quality)" in finding.message and "Person count → Character" in finding.message


def test_reorder_without_categories_never_guesses():
    # Unknown items could be characters, series or details; the rating is
    # not moved ahead of one it might belong after. Only the certain moves
    # are made: the person count first, the quality finish last.
    prompt = "masterpiece, smile, safe, honkai: star rail, firefly, 1girl"
    reordered = prompt_rules.reorder_tags(ANIMAGINE, prompt, None)
    assert reordered == "1girl, smile, safe, honkai: star rail, firefly, masterpiece"
    _assert_permutation(prompt, reordered)
    # Nothing definite to move: no finding, no variant.
    assert prompt_rules.reorder_tags(ANIMAGINE, "1girl, smile, safe, firefly", None) is None
    assert only(run(ANIMAGINE, "1girl, smile, safe, firefly"), "guide.order") == []


@pytest.mark.parametrize("model_id", [ANIMAGINE, KOHAKU, ILLUSTRIOUS])
@pytest.mark.parametrize(
    "prompt",
    [
        "zzkx oc",
        "1girl, zzkx oc",
        "zzkx oc, 1girl",
        "1girl, zzkx oc, masterpiece",
        "zzkx oc, masterpiece",
        "masterpiece, zzkx oc",
        "zzkx oc, smile",
        "smile, zzkx oc",
        "safe, zzkx oc, 1girl",
        "1girl, safe, zzkx oc",
        "masterpiece, 1girl, zzkx oc",
        "zzkx oc, newest, masterpiece",
        "1girl, smile, zzkx oc, yyqv oc, masterpiece",
        # A tag written twice takes one slot, whatever it is: its copies are
        # never out of order beside each other, only beside another item.
        "1girl, zzkx oc, zzkx oc, masterpiece",
        "zzkx oc, zzkx oc",
        "smile, zzkx oc, zzkx oc",
        "zzkx oc, smile, zzkx oc",
    ],
)
def test_an_item_is_left_unplaced_exactly_where_its_category_decides(model_id, prompt):
    # "Partly checked" (`unplaced_items`; `places_every_item`, the critic's
    # `placed_partly`) exactly where the order rule's answer hangs on a
    # category it lacks: some category of the items it could not place makes
    # it flag the prompt, or Check's own order, where the categories it has
    # do not. Where nothing is named, every category gives the same answer
    # and Check's order is in the documented order under each - so a lone
    # item, or one whose every slot sits in order beside the rest, is not
    # "partly checked"; said of every unplaced item, both were.
    for partial in (None, {"smile": "general"}):
        unknown = [
            item for item in prompt_rules.category_items(model_id, prompt)
            if partial is None or item not in partial
        ]
        flagged = prompt_rules.order_finding(model_id, prompt, partial) is not None
        ours = prompt_rules.reorder_tags(model_id, prompt, partial) or prompt
        changes = []
        for assignment in itertools.product(_DANBOORU_CATEGORIES, repeat=len(unknown)):
            full = {**(partial or {}), **dict(zip(unknown, assignment, strict=True))}
            changes.append(
                (prompt_rules.order_finding(model_id, prompt, full) is not None) != flagged
                or prompt_rules.order_finding(model_id, ours, full) is not None
            )
        decided = prompt_rules.places_every_item(model_id, prompt, partial)
        assert decided == (prompt_rules.unplaced_items(model_id, prompt, partial) == [])
        assert decided == (not any(changes)), (partial, prompt_rules.unplaced_items(
            model_id, prompt, partial
        ))


_DANBOORU_CATEGORIES = ("character", "copyright", "general", "artist", "meta")


MARCH = r"march 7th \(honkai: star rail\)"


SENKU = r"senku \(dr. stone\)"


@pytest.mark.parametrize(
    "prompt, categories, expected",
    [
        # A five-word character, placed by its category, not by its length.
        (
            f"1girl, honkai: star rail, safe, {MARCH}, smile",
            {MARCH: "character", "honkai: star rail": "copyright", "smile": "general"},
            f"1girl, {MARCH}, honkai: star rail, safe, smile",
        ),
        # A five-word series, and a dotted one with its dotted character.
        (
            "1girl, safe, jojo no kimyou na bouken, smile",
            {"jojo no kimyou na bouken": "copyright", "smile": "general"},
            "1girl, jojo no kimyou na bouken, safe, smile",
        ),
        (
            f"1boy, smile, dr. stone, {SENKU}, safe",
            {"dr. stone": "copyright", SENKU: "character", "smile": "general"},
            f"1boy, {SENKU}, dr. stone, safe, smile",
        ),
        # An article does not make a title prose once the index knows it.
        (
            "1girl, smile, safe, the legend of zelda, link",
            {"the legend of zelda": "copyright", "link": "character", "smile": "general"},
            "1girl, link, the legend of zelda, safe, smile",
        ),
    ],
)
def test_long_and_dotted_names_take_their_category(prompt, categories, expected):
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, categories) == expected
    _assert_permutation(prompt, expected)
    [finding] = only(run(ANIMAGINE, prompt, categories), "guide.order")
    assert finding.suggestion == expected
    assert only(run(ANIMAGINE, expected, categories), "guide.order") == []
    # Each of them is asked about, and nothing the guide's groups place.
    asked = prompt_rules.category_items(ANIMAGINE, prompt)
    assert set(asked) == set(categories)


@pytest.mark.parametrize(
    "prompt",
    [
        f"1girl, {MARCH}, honkai: star rail, safe, smile, masterpiece",
        "1girl, jojo no kimyou na bouken, safe, smile, masterpiece",
        "1girl, love live! school idol project, safe, smile, masterpiece",
        f"1boy, {SENKU}, dr. stone, safe, smile, masterpiece",
        "1girl, link, the legend of zelda, safe, smile, masterpiece",
    ],
)
def test_long_and_dotted_names_are_not_moved_without_categories(prompt):
    # Offline, a name the guide's groups do not know could be any category:
    # none is pinned to the Details slot and moved after the rating.
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, None) is None
    assert only(run(ANIMAGINE, prompt), "guide.order") == []


def test_prose_the_index_did_not_know_is_general():
    # A phrase a name could still be counts as prose once the lookup ran
    # and did not know it ({} is an answer; None is no lookup), and a
    # sentence no name could be is prose either way.
    phrase = "1girl, a girl standing in the rain, safe"
    assert only(run(ANIMAGINE, phrase), "guide.order") == []
    [finding] = only(run(ANIMAGINE, phrase, {}), "guide.order")
    assert finding.suggestion == "1girl, safe, a girl standing in the rain"
    sentence = "1girl, she stands in the rain, safe"
    [finding] = only(run(ANIMAGINE, sentence), "guide.order")
    assert finding.suggestion == "1girl, safe, she stands in the rain"


@pytest.mark.parametrize(
    "model_id, prompt, categories",
    [
        (ANIMAGINE, "safe, masterpiece, 1girl, year 2024, outdoors, firefly", CATEGORIES),
        (ANIMAGINE, "absurdres, great score, 1other, smile, smile, outdoors", None),
        (KOHAKU, guides.KOHAKU_GUIDE["examples"][0], {"absurdres": "meta", "ciloranko": "artist"}),
        (KOHAKU, "sensitive, explicit, nsfw, newest, 1boy,  solo ,\nsky", None),
        (
            ILLUSTRIOUS,
            "newest, masterpiece, general, 1girl, looking at viewer",
            {"looking at viewer": "general"},
        ),
        (ILLUSTRIOUS, "A girl standing in a field at dusk, masterpiece, 1girl", None),
        (WAI, "masterpiece, best quality, amazing quality, 1girl, beach", None),
    ],
)
def test_reordering_is_a_pure_permutation(model_id, prompt, categories):
    reordered = prompt_rules.reorder_tags(model_id, prompt, categories)
    assert reordered is not None
    _assert_permutation(prompt, reordered)
    # And it is already in order: a second pass has nothing to move.
    assert prompt_rules.reorder_tags(model_id, reordered, categories) is None


def test_reordering_keeps_a_slots_order_and_the_prose():
    # Stable within a slot; prose pieces in a hybrid prompt keep their order.
    prompt = "masterpiece, b tag, a tag, 1girl"
    assert prompt_rules.reorder_tags(
        ANIMAGINE, prompt, {"b tag": "general", "a tag": "general"}
    ) == "1girl, b tag, a tag, masterpiece"
    hybrid = "A girl standing in a field at dusk, the wind in her hair, masterpiece, 1girl"
    assert prompt_rules.reorder_tags(ILLUSTRIOUS, hybrid, None) == (
        "1girl, A girl standing in a field at dusk, the wind in her hair, masterpiece"
    )
    # The Kohaku example's line breaks stay where they were.
    example = guides.KOHAKU_GUIDE["examples"][0]
    reordered = prompt_rules.reorder_tags(KOHAKU, example, {"absurdres": "meta"})
    assert reordered.endswith("masterpiece, newest, absurdres, sensitive")
    assert reordered.count("\n") == example.count("\n")


def _quoted(text: str) -> Counter:
    return Counter(prompt_rules.QUOTED.findall(text))


@pytest.mark.parametrize(
    "model_id, prompt, categories, expected",
    [
        # Categories that read the halves of a quotation as tags placed
        # "world" as a character, ahead of the sign it is on.
        (
            ANIMAGINE,
            '1girl, holding sign, "hello, world", smile',
            {"holding sign": "general", '"hello': "general", 'world"': "character"},
            None,
        ),
        # Quoted, a name is lettering: the index's answer for it places nothing.
        (
            ANIMAGINE,
            '1girl, smile, "firefly"',
            {"smile": "general", '"firefly"': "character"},
            None,
        ),
        # Read piece by piece, "she is here" is a sentence and "Hello" could
        # be a name: a character went in between them.
        (
            ANIMAGINE,
            '"Hello, she is here", firefly',
            {"firefly": "character"},
            'firefly, "Hello, she is here"',
        ),
        # A tag inside the lettering is the lettering's; the rest moves round it.
        (
            ANIMAGINE,
            'masterpiece, sign says "hi, 1girl, bye", smile',
            None,
            'sign says "hi, 1girl, bye", smile, masterpiece',
        ),
        # The spacing inside it is its own, and a "，" in it splits nothing.
        (
            ILLUSTRIOUS,
            'masterpiece, sign says "hi,there", 1girl',
            None,
            '1girl, sign says "hi,there", masterpiece',
        ),
        (
            ANIMAGINE,
            'masterpiece, sign says "你好，世界", 1girl',
            None,
            '1girl, sign says "你好，世界", masterpiece',
        ),
        # Unquoted lettering too: a title in its marks.
        (
            ANIMAGINE,
            "masterpiece, 封面上写着《hello, world》, 1girl",
            None,
            "1girl, 封面上写着《hello, world》, masterpiece",
        ),
        (
            ANIMAGINE,
            "masterpiece, 1girl, 封面上写着《hello, world》, smile",
            {"world》": "character"},
            "1girl, 封面上写着《hello, world》, smile, masterpiece",
        ),
        # Lettering that starts an item is that item's, not the one before.
        (ANIMAGINE, 'masterpiece, "hi", 1girl', None, '1girl, "hi", masterpiece'),
    ],
)
def test_lettering_is_moved_whole(model_id, prompt, categories, expected):
    # The chips' split cuts at every comma, lettering's included; the order
    # rules read a stretch of lettering as one item, placed by its shape.
    assert prompt_rules.order_checked(model_id, prompt)
    reordered = prompt_rules.reorder_tags(model_id, prompt, categories)
    assert reordered == expected
    findings = only(run(model_id, prompt, categories), "guide.order")
    assert [finding.suggestion for finding in findings] == ([expected] if expected else [])
    if reordered:
        assert _quoted(reordered) == _quoted(prompt)
        assert prompt_rules.reorder_tags(model_id, reordered, categories) is None


def test_a_move_that_would_change_what_is_quoted_is_not_offered():
    # A stray inch mark moved ahead of the quotation would open it: the
    # "reordered" prompt would quote ' figure, a sign that says '.
    prompt = 'a sign that says "hi, there", 12" figure, 1girl'
    categories = {'12" figure': "character"}
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, categories) is None
    # The order is still out, with no one-click fix.
    [finding] = only(run(ANIMAGINE, prompt, categories), "guide.order")
    assert finding.suggestion is None


@pytest.mark.parametrize(
    "prompt",
    [
        # Lettering splits into no pieces of its own, and must not make up
        # for an item a line break splits.
        '"a", "b", masterpiece\n1girl',
        # Nor may it hide a tag that a line break or a "，" joins to it in
        # one item, where nothing would place the tag.
        'firefly, "hi"，1girl',
        'firefly, "hi"\n1girl',
        'masterpiece\n"hello, world", 1girl',
        'masterpiece, 1girl，"hi"',
    ],
)
def test_a_line_break_beside_lettering_still_stands_the_order_down(prompt):
    categories = {"firefly": "character"}
    assert not prompt_rules.order_checked(ANIMAGINE, prompt)
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, categories) is None
    assert only(run(ANIMAGINE, prompt, categories), "guide.order") == []


def test_lettering_is_placed_by_its_shape_asked_or_not():
    # The index is never asked about lettering, so whether it was asked
    # about anything else must not move it: prose-shaped, it is prose.
    prompt = '1girl, "the cat and the hat", safe'
    expected = '1girl, safe, "the cat and the hat"'
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, None) == expected
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, {}) == expected


@pytest.mark.parametrize(
    "model_id, prompt",
    [
        (FLUX_DEV, "masterpiece, 1girl"),
        (NEWBIE, "masterpiece, 1girl"),
        (ANIMAGINE, "A girl with green hair stands outside at night, masterpiece."),
        (ANIMAGINE, "1girl, green hair, masterpiece, high score"),
        (ANIMAGINE, "1girl"),
    ],
)
def test_nothing_to_reorder(model_id, prompt):
    assert prompt_rules.reorder_tags(model_id, prompt, None) is None


# Comma items the categories would place out of order inside a JSON list,
# were its quoted strings read as tags.
JSON_CATEGORIES = {'"smile"': "general", '"firefly"': "character"}


@pytest.mark.parametrize(
    "prompt, twin, categories",
    [
        # Past the item cap: no 1,700-character Apply, no quadratic placement.
        (OVER_CAP, "masterpiece, t0, 1girl", None),
        # Quoted, the strings are lettering in a tag list too: the twin's are bare.
        ('[0, "smile", "firefly", 0]', "x, smile, firefly, y", {**JSON_CATEGORIES, **CATEGORIES}),
        ("<x>, masterpiece, 1girl", "x, masterpiece, 1girl", None),
        ("masterpiece\nsafe\n1girl", "masterpiece, safe, 1girl", None),
    ],
)
def test_the_order_rule_stands_down_where_the_order_is_not_checked(prompt, twin, categories):
    # The same pair written as a plain tag list is flagged; here it is not.
    assert only(run(ANIMAGINE, twin, categories), "guide.order")
    assert not prompt_rules.order_checked(ANIMAGINE, prompt)
    assert only(run(ANIMAGINE, prompt, categories), "guide.order") == []
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, categories) is None


@pytest.mark.parametrize("model_id", [c.repo for c in launch.CHECKPOINTS])
def test_an_order_not_checked_is_never_flagged(model_id):
    # The page says "Not checked" wherever order_checked is False, so no
    # order finding or reordered version may exist there.
    prompts = [
        OVER_CAP, '[0, "smile", "firefly", 0]', "<x>, masterpiece, 1girl",
        "masterpiece\nsafe\n1girl", "masterpiece，safe，1girl，smile",
        "masterpiece, smile, safe, honkai: star rail, firefly, 1girl",
        "A girl standing in a field at dusk, masterpiece, 1girl",
        *guide_for(model_id)["examples"],
    ]
    for prompt in prompts:
        if prompt_rules.order_checked(model_id, prompt):
            continue
        for categories in (None, {**CATEGORIES, **JSON_CATEGORIES}):
            assert only(run(model_id, prompt, categories), "guide.order") == [], prompt
            assert prompt_rules.reorder_tags(model_id, prompt, categories) is None, prompt


def test_the_kohaku_example_departs_from_the_epsilon_order():
    # The TIPO demo's DEFAULT_TAGS ends "masterpiece, absurdres, sensitive,
    # newest", while the caption order Zeta refers to puts the date before
    # meta and the rating last. Advice, and the variant to test it with.
    [finding] = only(run(KOHAKU, guides.KOHAKU_GUIDE["examples"][0]), "guide.order")
    assert finding.severity == "info" and '"sensitive" (Rating)' in finding.message


def test_category_items_asks_only_for_what_a_rule_reads():
    prompt = "1girl, firefly, masterpiece, smile, year 2024"
    assert prompt_rules.category_items(ANIMAGINE, prompt) == ["firefly", "smile"]
    # WAI's categories all land in one slot, so a lookup would change nothing.
    assert prompt_rules.category_items(WAI, "1girl, firefly, masterpiece") == []
    assert prompt_rules.category_items(FLUX_DEV, "1girl, firefly") == []
    assert prompt_rules.category_items(ANIMAGINE, "A girl stands in the rain at night.") == []
    # Anima reads artists; an item already prefixed needs no lookup, and
    # neither does one its Tags menu places.
    assert prompt_rules.category_items(ANIMA, "1girl, @nnn yryr, oomuro sakurako") == [
        "oomuro sakurako",
    ]
