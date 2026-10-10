"""Swap tags' rewrite, as text: what a row matches and what it writes."""
from __future__ import annotations

import pytest

from backend.tag_swap import parse_rows, swap, swap_record


def run(text, *pairs):
    outcome = swap(text, parse_rows(list(pairs)))
    return outcome, [outcome.text[a:b] for a, b in outcome.spans]


def test_many_to_one_lands_where_the_first_match_sits():
    outcome, marked = run("1girl, red hair, long hair, smile", ("red hair, long hair", "silver hair"))
    assert (outcome.status, outcome.text, marked) == ("changed", "1girl, silver hair, smile", ["silver hair"])
    # The first item matched in the prompt's order, not the row's.
    outcome, _ = run("1girl, long hair, smile, red hair", ("red hair, long hair", "silver hair"))
    assert outcome.text == "1girl, silver hair, smile"


def test_case_spacing_underscores_and_escapes_are_one_tag():
    for written in ("Red Hair", "red_hair", "red   hair", "RED_HAIR"):
        outcome, _ = run(f"1girl, {written}", ("red hair", "silver hair"))
        assert outcome.status == "changed", written
    outcome, _ = run(r"1girl, red hair \(cosplay\)", ("red hair (cosplay)", "blue hair (cosplay)"))
    # The item escaped its parentheses, and so does what replaces it.
    assert outcome.text == r"1girl, blue hair \(cosplay\)"


def test_a_weight_wrapping_the_whole_item_is_kept_round_the_replacement():
    outcome, marked = run("1girl, (red_hair:1.2), long_hair", ("red hair, long hair", "silver hair"))
    # Underscores, as the matched item wrote its words.
    assert outcome.text == "1girl, (silver_hair:1.2)" and marked == ["(silver_hair:1.2)"]
    outcome, _ = run("((red hair)), [smile]", ("red hair", "blue hair"), ("smile", "grin"))
    assert outcome.text == "((blue hair)), [grin]"
    # Only the whole item: a bracket inside one is part of its text.
    outcome, _ = run("1girl, (red hair) and more", ("red hair", "blue hair"))
    assert outcome.status == "missing"


def test_strict_a_row_missing_one_of_its_tags_leaves_the_image_out():
    outcome, _ = run("1girl, red hair, smile", ("red hair, long hair", "silver hair"), ("smile", "grin"))
    assert (outcome.status, outcome.missing, outcome.text) == (
        "missing", ["long hair"], "1girl, red hair, smile",
    )


def test_rows_match_the_original_at_once():
    outcome, marked = run("red hair, blue hair", ("red hair", "blue hair"), ("blue hair", "red hair"))
    assert outcome.text == "blue hair, red hair" and marked == ["blue hair", "red hair"]


def test_a_tag_two_rows_find_is_refused():
    with pytest.raises(ValueError, match="red hair is in two rows' Find"):
        parse_rows([("red hair, smile", "x"), ("Red_Hair", "y")])
    with pytest.raises(ValueError, match="needs a tag"):
        parse_rows([(" , ", "x")])


def test_a_replacement_already_there_is_not_written_twice():
    outcome, marked = run("1girl, red hair, silver hair", ("red hair", "silver hair"))
    assert outcome.text == "1girl, silver hair" and marked == []
    assert outcome.notes == ["already had silver hair"]


def test_an_empty_replacement_removes_with_the_separator_before_it():
    outcome, _ = run("1girl, red hair,\nsmile", ("red hair", ""))
    assert outcome.text == "1girl,\nsmile"
    outcome, _ = run("red hair, smile", ("red hair", ""))
    assert outcome.text == "smile"
    # Every item removed is an empty prompt, which the route calls invalid.
    outcome, _ = run("red hair", ("red hair", ""))
    assert (outcome.status, outcome.text) == ("changed", "")


def test_a_tag_never_matches_inside_another_tag():
    outcome, marked = run("hair, red hair", ("hair", "x"))
    assert outcome.text == "x, red hair" and marked == ["x"]
    outcome, _ = run("1girl, red hair", ("hair", "x"))
    assert outcome.status == "missing"


def test_a_one_tag_row_matches_a_phrase_in_prose_but_not_its_lettering():
    text = 'a girl with red hair holding a sign that reads "red hair", masterpiece'
    outcome, marked = run(text, ("red hair", "silver hair"))
    assert outcome.text == 'a girl with silver hair holding a sign that reads "red hair", masterpiece'
    assert marked == ["silver hair"]
    # Bounded on the find's own word edges.
    outcome, _ = run("a girl with tired hairstyles standing in the rain", ("red hair", "x"))
    assert outcome.status == "missing"
    # Removing a phrase leaves no double space.
    outcome, _ = run("a girl with red hair standing in the rain", ("red hair", ""))
    assert outcome.text == "a girl with standing in the rain"


def test_many_to_one_in_prose_is_left_to_the_user():
    outcome, _ = run("a girl with red hair and long hair walking home", ("red hair, long hair", "x"))
    assert outcome.status == "by_hand" and outcome.by_hand == ["red hair, long hair → x"]


def test_han_and_kana_match_as_a_substring():
    outcome, marked = run("一个女孩，红色头发", ("红色", "蓝色"))
    assert outcome.text == "一个女孩，蓝色头发" and marked == ["蓝色"]


def test_an_xml_prompt_is_not_rewritten():
    outcome, _ = run("<character_1><n>miku</n></character_1>", ("miku", "x"))
    assert outcome.status == "structured"


def test_variables_follow_the_text_and_a_variant_label_does_not():
    rows = parse_rows([("red hair", "silver hair")])
    swapped = swap_record(
        "1girl, red hair, smile", None, {"hair": "red hair", "variant": "red hair", "pose": "sitting"}, rows,
    )
    assert swapped.prompt == "1girl, silver hair, smile"
    assert swapped.variables == {"hair": "silver hair", "variant": "red hair", "pose": "sitting"}
    # The negative prompt only when asked, and then not the prompt.
    swapped = swap_record("red hair", "red hair, blurry", None, rows, "negative_prompt")
    assert (swapped.prompt, swapped.negative_prompt) == ("red hair", "silver hair, blurry")
