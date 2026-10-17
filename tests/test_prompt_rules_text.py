"""Prompt text parsing, styles, word counts, and quoted comma items."""
from __future__ import annotations

import pytest

from backend import guides, prompt_rules

from tests.prompt_rules_helpers import (
    ANIMA,
    ANIMAGINE,
    ERNIE,
    FLUX_DEV,
    ILLUSTRIOUS,
    KLEIN,
    KREA_DEV,
    NEWBIE,
    only,
    run,
    silent,
)


# ------------------------------------------------------------------ reading the text


@pytest.mark.parametrize(
    "prompt, style",
    [
        ("A cat holding a sign that says hello world", "prose"),
        ("1girl, green hair, sweater, looking at viewer, upper body, beanie", "tags"),
        ("1girl", "tags"),
        ("white owl", "tags"),
        ("a cat", "prose"),
        ("a red fox sitting in fresh snow, golden hour, photorealistic", "hybrid"),
        (guides.ANIMA_TURBO_GUIDE["examples"][0], "prose"),
        (guides.KREA2_TURBO_GUIDE["examples"][3], "prose"),
        (guides.NEWBIE_GUIDE["examples"][0], "xml"),
        ("<character_1><n>miku", "xml"),
        ('{"scene": "a quiet cafe", "subjects": []}', "json"),
        ("[character] in a garden", "prose"),
        ("一只黑白相间的中华田园犬", "prose"),
        ("You are an assistant. <Prompt Start> 1girl", "prose"),
        # Long, dotted and exclaimed Danbooru names are tags, not phrases:
        # length and a full stop alone do not make a sentence.
        ("1girl, jojo no kimyou na bouken, safe, smile", "tags"),
        (r"1girl, march 7th \(honkai: star rail\), honkai: star rail, smile", "tags"),
        ("1girl, love live! school idol project, smile", "tags"),
        (r"1boy, senku \(dr. stone\), dr. stone, smile", "tags"),
        # A preposition in a longer piece still makes a phrase.
        ("woman running through tall grass, stormy skies, dramatic lighting", "hybrid"),
    ],
)
def test_detect_style(prompt, style):
    assert prompt_rules.detect_style(prompt) == style


def test_the_style_reported_is_the_one_the_rules_applied():
    # "the legend of zelda" reads as a phrase until the tag index calls it a
    # series; with the categories the rules had, the reported style is theirs.
    prompt = "1girl, solo, the legend of zelda, princess zelda, smile"
    known = {"the legend of zelda": "copyright", "princess zelda": "character"}
    assert prompt_rules.detect_style(prompt) == "hybrid"
    assert prompt_rules.detect_style(prompt, known) == "tags"
    assert silent(ANIMAGINE, prompt, "guide.style", known)


def test_order_is_checked_only_where_the_rules_can_place_it():
    # A tag checkpoint's comma items take slots; a FLUX sentence's "subject
    # first", klein's steps and NewBie's blocks are documented, not read.
    assert prompt_rules.order_checked(ANIMAGINE, "1girl, smile, masterpiece")
    assert prompt_rules.order_checked(ILLUSTRIOUS, "A girl stands in the rain at night.")
    assert prompt_rules.order_checked(ANIMA, "1girl, smile")
    for model_id, prompt in (
        (KREA_DEV, "Golden hour light over misty mountains, a lone hiker on a ridge"),
        (FLUX_DEV, "1girl, smile"),
        (KLEIN, "A red fox in fresh snow"),
        (NEWBIE, guides.NEWBIE_GUIDE["examples"][0]),
        (ANIMAGINE, '{"scene": "a cafe"}'),
        ("Minthy/RouWei-0.8", "1girl, smile"),
    ):
        assert not prompt_rules.order_checked(model_id, prompt), model_id
    assert not prompt_rules.order_checked(ANIMAGINE, ", ".join(f"t{i}" for i in range(300)))


@pytest.mark.parametrize(
    "prompt", ["masterpiece\nsafe\n1girl", "masterpiece，safe，1girl，smile"]
)
def test_a_list_not_split_by_commas_is_not_shown_as_ordered(prompt):
    # One tag per line, or "，" between them: tags to read, one comma item to
    # the chips and the order rule, so nothing in it was placed.
    assert prompt_rules.detect_style(prompt) == "tags"
    assert not prompt_rules.order_checked(ANIMAGINE, prompt)
    assert prompt_rules.reorder_tags(ANIMAGINE, prompt, None) is None
    assert only(run(ANIMAGINE, prompt), "guide.order") == []
    # masterpiece is there: not reported missing, not appended again.
    [finish] = only(run(ANIMAGINE, prompt), "guide.quality")
    assert "masterpiece" not in finish.message
    assert finish.suggestion.count("masterpiece") == 1
    # Nor is Illustrious's quality tier, on a line or after "，" of its own.
    separator = "\n" if "\n" in prompt else "，"
    assert silent(ILLUSTRIOUS, separator.join(["1girl", "solo", "best quality"]), "guide.quality")
    assert only(run(ILLUSTRIOUS, separator.join(["1girl", "solo", "smile"])), "guide.quality")
    # Line breaks after commas split nothing the commas did not.
    assert prompt_rules.order_checked(ANIMAGINE, "1girl,\nsmile,\nmasterpiece")


@pytest.mark.parametrize("prompt", ["[" * 3999, "{" * 3999, '{"a":' * 799])
def test_nesting_deeper_than_the_parser_is_not_json(prompt):
    # Inside the 4000-character limit, and deep enough that json.loads runs
    # out of recursion rather than answering "not JSON".
    assert prompt_rules.detect_style(prompt) != "json"


@pytest.mark.parametrize(
    "prompt, count",
    [
        ("A cat holding a sign", 5),
        ("russet-brown harvest mouse's nose", 4),
        ("long_hair, blue_eyes", 4),
        ("一只猫", 3),
        ("a 猫 and 狗", 4),
        ("", 0),
    ],
)
def test_word_count(prompt, count):
    assert prompt_rules.word_count(prompt) == count


def test_comma_items_carry_their_place():
    prompt = " 1girl ,  long hair,\nsmile,, "
    items = prompt_rules.comma_items(prompt)
    assert [item.text for item in items] == ["1girl", "long hair", "smile"]
    assert all(prompt[item.start:item.end] == item.text for item in items)


@pytest.mark.parametrize(
    "text, quoted",
    [
        # Chinese, Japanese and Korean put no space around a quotation, and
        # ERNIE's rewriter wraps lettering in English single quotes.
        ("写着'营业中'的招牌", ["'营业中'"]),
        ("一块木招牌上写着'营业中'，傍晚的街道", ["'营业中'"]),
        ("'营业中'的字样", ["'营业中'"]),
        ("看板に'OPEN'と書いてある", ["'OPEN'"]),
        ("'영업중'이라고 쓰인 간판", ["'영업중'"]),
        ("A sign that says 'Black Forest Labs'", ["'Black Forest Labs'"]),
        # A Latin letter or a digit on either side is still an apostrophe.
        ("an animal's toy, rock'n'roll, the '90s", []),
        ("the cat's 'toy'", ["'toy'"]),
        ("猫's toy isn't here", []),
    ],
)
def test_single_quotes_may_touch_ideographs(text, quoted):
    assert prompt_rules.QUOTED.findall(text) == quoted


def test_single_quoted_chinese_lettering_is_quoted_text():
    # Lettering in quotation marks: nothing to quote, and the span the
    # critic holds text_to_render and a rewrite's lettering to.
    prompt = "一家小店门口，一块木招牌上写着'营业中'，傍晚的街道"
    assert only(run(ERNIE, prompt), "guide.quote_text") == []
    assert [prompt[s:e] for s, e in prompt_rules.rendered_spans(prompt)] == ["'营业中'"]
