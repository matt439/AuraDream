"""Rendered lettering boundaries, quotations, and negation near misses."""
from __future__ import annotations

import pytest

from backend import prompt_rules

from tests.prompt_rules_helpers import (
    ANIMAGINE,
    ERNIE,
    FLUX_DEV,
    KLEIN,
    KREA_DEV,
    QWEN,
    only,
    run,
    spanned,
)


def test_sign_text_is_quoted_rather_than_rephrased():
    # One finding for "NO PARKING", and it is the quote rule's info.
    prompt = "A red sign reading NO PARKING on a brick wall at dusk"
    findings = [f for f in run(FLUX_DEV, prompt) if f.rule.startswith("guide.")]
    assert [(f.rule, f.severity, spanned(prompt, f)) for f in findings if f.start is not None] == [
        ("guide.quote_text", "info", "NO PARKING"),
    ]
    # A capital "No" or a lowercase neighbour is still a negation.
    for prompt, phrase in (
        ("No people on the beach at dawn", "No people"),
        ("Tokyo street at night with no people", "no people"),
    ):
        assert [spanned(prompt, f) for f in only(run(FLUX_DEV, prompt), "guide.negation")] == [
            phrase
        ]


def test_unquoted_text_to_render_is_found_and_quoted():
    prompt = "A neon sign reading OPEN above the door"
    [finding] = only(run(FLUX_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == "OPEN" and finding.severity == "info"
    assert finding.suggestion == 'A neon sign reading "OPEN" above the door'
    prompt = "A cat holding a sign that says hello world in red letters"
    [finding] = only(run(KLEIN, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == "hello world"
    prompt = "街边的招牌上写着欢迎光临，夜晚"
    [finding] = only(run(QWEN, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == "欢迎光临"


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("A jar labeled SUGAR on a kitchen shelf", "SUGAR"),
        ("A bottle labelled as POISON", "POISON"),
        ("A coffee mug that says MONDAY", "MONDAY"),
        ("A speech bubble saying hi", "hi"),
        ("Colorful balloons spelling out HAPPY BIRTHDAY", "HAPPY BIRTHDAY"),
        ("A shop sign in the window says CLOSED", "CLOSED"),
    ],
)
def test_text_on_something_that_carries_it(prompt, target):
    [finding] = only(run(FLUX_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == target


@pytest.mark.parametrize(
    "prompt, target",
    [
        # A joining word with a capital after it carries the lettering on.
        ("A seaside shack with a hand-painted sign that says Fish and Chips, gulls on the roof",
         "Fish and Chips"),
        ("A classroom chalkboard that reads Welcome to Paradise in looping cursive",
         "Welcome to Paradise"),
        ("A diner neon sign that says Rock and Roll Forever", "Rock and Roll Forever"),
        ("A shop sign reading Books and Tea", "Books and Tea"),
        # Capitals before a lowercase joining word are the whole of it.
        ("A sign that says OPEN and a cat asleep under it", "OPEN"),
        ("A coffee mug that says MONDAY with coffee inside", "MONDAY"),
    ],
)
def test_multi_word_lettering_is_quoted_whole(prompt, target):
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == target
    start = prompt.index(target)
    assert finding.suggestion == f'{prompt[:start]}"{target}"{prompt[start + len(target):]}'
    # Applied, it is quoted, and the check finds nothing more to quote.
    assert only(run(KREA_DEV, finding.suggestion), "guide.quote_text") == []


def test_lettering_whose_end_is_a_guess_gets_no_one_click_quote():
    # "fish" alone, quoted, would letter half the sign; the finding says what
    # to do instead of doing it wrong.
    prompt = "A market stall with a sign that says fish and chips for sale"
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == "fish" and finding.suggestion is None
    assert "Put the exact words to render in quotation marks." in finding.message
    long = "A banner that says one two three four five six seven eight"
    [finding] = only(run(KREA_DEV, long), "guide.quote_text")
    assert finding.suggestion is None


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("A sign says MADE IN JAPAN.", "MADE"),
        ("A sign says Made In Japan.", "Made"),
        ("A sign reading OPEN AT NOON.", "OPEN"),
        ("A poster with the words LOVE IS IN THE AIR.", "LOVE IS"),
        ("A sign says 9:00.", "9"),
        ("A price tag says $9.99.", "$9"),
        ("A poster says www.example.com.", "www"),
        ("A poster reading www.example.com.", "www"),
        ("A sign says HELLO, WORLD!", "HELLO"),
        ("A sign says OPEN: 9 AM.", "OPEN"),
    ],
)
def test_english_lettering_continuing_past_a_stop_gets_no_one_click_quote(prompt, target):
    # A place word in capitals can belong to the inscription; marks can
    # belong to a time, price, address or list. Quoting the opening alone
    # silently changes what the image is asked to letter.
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == target and finding.suggestion is None
    assert "Put the exact words to render in quotation marks." in finding.message
    start = prompt.index(target)
    end = start + len(target)
    assert prompt_rules.rendered_targets(prompt) == [(start, end, False, end)]


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("A sign says OPEN.", "OPEN"),
        ("A sign says OPEN. A cat sleeps.", "OPEN"),
        ("A sign says OPEN! A cat sleeps.", "OPEN"),
        ("A sign says OPEN, a cat sleeps nearby.", "OPEN"),
        ("A sign says OPEN in red letters.", "OPEN"),
        ("A sign says open above the door.", "open"),
        ("A price tag says $9.", "$9"),
    ],
)
def test_english_lettering_with_a_certain_end_keeps_its_one_click_quote(prompt, target):
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    start = prompt.index(target)
    end = start + len(target)
    assert finding.suggestion == f'{prompt[:start]}"{target}"{prompt[end:]}'
    assert prompt_rules.rendered_targets(prompt) == [(start, end, True, end)]
    assert only(run(KREA_DEV, finding.suggestion), "guide.quote_text") == []


@pytest.mark.parametrize(
    "model_id, prompt, target, ends",
    [
        # 的 links the lettering to what it is on: a wooden sign reading
        # 欢迎光临. Whether 的 is the lettering's own is the guess; what
        # follows it is description, so nothing may read on past it.
        (QWEN, "一块写着欢迎光临的木质招牌挂在门口", "欢迎光临", True),
        (ERNIE, "一件印着我爱北京的白色T恤", "我爱北京", True),
        # A verb that puts the sign somewhere starts describing it, with no
        # 的 before it and punctuation only after the description.
        (QWEN, "门上写着营业中挂在门口，夜晚", "营业中", True),
        (QWEN, "一块牌子写着营业中挂在门口", "营业中", True),
        (QWEN, "横幅上写着欢迎光临悬挂在大门上方", "欢迎光临", True),
        (ERNIE, "T恤上印着我爱北京站在街头", "我爱北京", True),
        (QWEN, "招牌上写着营业中位于街角", "营业中", True),
        # Only a request or a prohibition right before the verb makes the
        # verb the sign's own; a slogan's (我们站在一起) loses its one click
        # rather than a description getting one.
        (QWEN, "墙上写着请勿吸烟挂在墙上，夜晚", "请勿吸烟", True),
        (QWEN, "门上写着请把门关好挂在门上，夜晚", "请把门关好", True),
        (QWEN, "墙上写着禁止停放在此贴在墙上，夜晚", "禁止停放在此", True),
        (QWEN, "横幅上写着我们站在一起，夜晚", "我们", True),
        # 等 before 字样 is "and so on", or the end of a word (平等).
        (QWEN, "横幅上写有平等字样，夜晚", "平", True),
        (QWEN, "写着欢迎光临等字样", "欢迎光临", True),
        # A verb that starts right at the cap still ends it.
        (QWEN, "写着一二三四五六七八九十一挂在门口，", "一二三四五六七八九十一", True),
        # Longer than the target's cap, with more text after it.
        (QWEN, "招牌上写着欢迎光临本店祝您用餐愉快万事如意", "欢迎光临本店祝您用餐愉快", False),
        # Latin lettering may run on past a space, or past an ASCII mark
        # with a letter or a digit after it: hours, a price, an address.
        (QWEN, "招牌上写着OPEN NOW的字样", "OPEN", False),
        (QWEN, "门上写着营业时间9:00-21:00", "营业时间9", False),
        (QWEN, "招牌上写着3.14的数字", "3", False),
        (QWEN, "招牌上写着www.shop.com", "www", False),
    ],
)
def test_chinese_lettering_whose_end_is_a_guess_gets_no_one_click_quote(
    model_id, prompt, target, ends
):
    # With no spaces to end Chinese lettering, twelve characters of whatever
    # followed were quoted - the sign and the door with the words - and the
    # re-check then passed.
    [finding] = only(run(model_id, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == target and finding.suggestion is None
    assert "Put the exact words to render in quotation marks." in finding.message
    assert [(prompt[s:e], x, h) for s, e, x, h in prompt_rules.rendered_targets(prompt)] == [
        (target, ends, prompt.index(target) + len(target))
    ]


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("街边的招牌上写着欢迎光临，夜晚", "欢迎光临"),
        ("门上写着：营业中", "营业中"),
        ("招牌上写着OPEN，夜晚", "OPEN"),
        # Latin lettering ends where the script changes, 的 included.
        ("写着OPEN的招牌", "OPEN"),
        ("一件印有LOGO的T恤", "LOGO"),
        ("写着OPEN挂在门口", "OPEN"),
        # A title is its marks and what they hold.
        ("封面上写着《营业中》的招牌", "《营业中》"),
        # 字样 names the lettering before it: "bearing the word 欢迎".
        ("门上写有欢迎字样，夜晚", "欢迎"),
        ("一块写有欢迎字样的牌子", "欢迎"),
        ("写着OPEN等字样", "OPEN"),
        # So does a count of its characters, one that starts at the cap too.
        ("门上写着福两个大字，夜晚", "福"),
        ("门上写着欢迎两个字，夜晚", "欢迎"),
        ("写着一二三四五六七八九十一两个大字，", "一二三四五六七八九十一"),
        # Signs that hold 在 or a verb of their own keep their one click:
        # only a verb placing the sign ends one, and not where the sign asks
        # for it.
        ("写着正在营业，夜晚", "正在营业"),
        ("写着欢迎光临本店，夜晚", "欢迎光临本店"),
        ("地上写着请勿站在黄线内，夜晚", "请勿站在黄线内"),
        ("门口写着禁止停放在此，夜晚", "禁止停放在此"),
        ("地铁站的牌子上写着请站在黄线外，夜晚", "请站在黄线外"),
        ("停车场的牌子上写着请停在白线内，夜晚", "请停在白线内"),
        ("牌子上写着请勿随意停放在此，夜晚", "请勿随意停放在此"),
        ("牌子上写着请将车辆停放在指定区域，夜晚", "请将车辆停放在指定区域"),
    ],
)
def test_chinese_lettering_ended_by_punctuation_is_quoted(prompt, target):
    [finding] = only(run(QWEN, prompt), "guide.quote_text")
    start = prompt.index(target)
    assert finding.suggestion == f'{prompt[:start]}"{target}"{prompt[start + len(target):]}'
    assert only(run(QWEN, finding.suggestion), "guide.quote_text") == []


def test_lowercase_words_after_a_participle_get_no_one_click_quote():
    # "reading" belongs to whichever noun it follows: the boy reads comics.
    prompt = "A boy sitting on a box reading comics"
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == "comics" and finding.suggestion is None
    [finding] = only(run(KREA_DEV, "A speech bubble saying hi"), "guide.quote_text")
    assert finding.suggestion is None
    # Capitals after one, and lowercase words after "says" whose end a stop
    # word marks, are still quoted in one click.
    for prompt, target in (
        ("A red sign reading NO PARKING on a brick wall", "NO PARKING"),
        ("A wooden sign that says open above the bakery door", "open"),
    ):
        [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
        start = prompt.index(target)
        assert finding.suggestion == f'{prompt[:start]}"{target}"{prompt[start + len(target):]}'


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("A sign above her reads OPEN", "OPEN"),
        ("A sign behind her reads NO ENTRY", "NO ENTRY"),
        ("A banner behind her said WELCOME HOME", "WELCOME HOME"),
        ("A girl at a cafe, a sign above her reads OPEN", "OPEN"),
        ("A poster of her says HELLO", "HELLO"),
        ("A sign above her says welcome", "welcome"),
        # "her" is an object here too, and a capital is lettering.
        ("A sign above her reading OPEN", "OPEN"),
    ],
)
def test_a_trigger_after_her_is_still_a_verb(prompt, target):
    # Only a form that can be a noun - "a book of spells", "her reading
    # glasses" - is one after a possessive: "her" is also an object, and
    # read as a noun, "NO ENTRY" lost its quote and warned as an absence.
    findings = run(KREA_DEV, prompt)
    [finding] = only(findings, "guide.quote_text")
    start = prompt.index(target)
    assert finding.suggestion == f'{prompt[:start]}"{target}"{prompt[start + len(target):]}'
    assert only(findings, "guide.negation") == []


def test_rendered_targets_say_which_ends_are_sure():
    # The critic reads past a lettering end only where the rules stopped at
    # a guess and the lettering may go on ("fish and chips for sale").
    prompt = (
        'A sign that says OPEN in red, a banner reading "SALE", '
        "a stall with a sign that says fish and chips for sale"
    )
    targets = prompt_rules.rendered_targets(prompt)
    assert [(prompt[start:end], ends) for start, end, ends, _ in targets] == [
        ("OPEN", True), ('"SALE"', True), ("fish", False),
    ]
    # Lettering starts anywhere in each: `head` is its end.
    assert [head for _, end, _, head in targets] == [end for _, end, _, _ in targets]
    assert prompt_rules.rendered_spans(prompt) == [(start, end) for start, end, _, _ in targets]
    # No one-click quote on these, but what follows is description, not
    # more lettering: the sign itself after 的, "about space" after a
    # participle's lowercase words, even where a joining word stopped them.
    chinese = "一块写着欢迎光临的木质招牌"
    assert prompt_rules.rendered_targets(chinese) == [(4, 8, True, 8)]
    assert prompt_rules.rendered_spans(chinese) == [(4, 8)]
    comics = "A boy sitting on a box reading comics about space and a cat asleep"
    assert only(run(KREA_DEV, comics), "guide.quote_text")[0].suggestion is None
    assert [(comics[s:e], x, s, h) for s, e, x, h in prompt_rules.rendered_targets(comics)] == [
        ("comics about space", True, comics.index("comics"), comics.index(" and"))
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        'A cat holding a sign that says "hello world"',
        "A shop sign that says 'Black Forest Labs'",
        "A girl reading a book in the park",
        "The text is blurry and the letters are faded",
        'A poster with the words: "Grand Opening"',
        # A verb of saying with nothing to carry text is speech, and quoting
        # its words would ask for lettering.
        "A mother says goodbye to her son at the station",
        "Two friends saying hello in a busy cafe",
        "A man in a hat says hello",
        "A wizard casting spells over a cauldron",
        # "labeled" as an adjective names no text.
        "A diagram with labeled parts of a flower",
        "A labeled map of the island",
        # What follows a trigger may describe the text rather than be it: how
        # it is made, where it sits, whose it is.
        "A t-shirt with the words printed across the chest",
        "A wooden plaque with the text carved into the wood",
        "A leaflet with the text about climate change",
        "A poster with the words neatly printed below the photo",
        "An old man with a newspaper reading without glasses by the fire",
        "A girl on a crate reading a book",
        # After a noun trigger, a participle with a preposition of place after
        # it places the words, listed describing word or not.
        "A t-shirt with the words shimmering across the chest",
        "A sign with the words blinking in the rain",
        "A banner with the slogan hanging above the stage",
        "A wall with the letters dripping down the bricks",
        "A door with the words scratched into the paint",
        "A cake with the text piped in blue icing",
        "A cake with the words spelled out in candles",
        # An irregular one too, and an -ing form the clause ends on or an
        # adverb follows.
        "A poster with the headline set in bold type",
        "A sign with the words lit up at night",
        "A rug with the text woven into it",
        "A banner with the slogan strung up between two poles",
        "A neon sign with the words flickering",
        "A neon sign with the words flickering softly above the bar",
        # A trigger that can be a noun is one after "of", an article or a
        # possessive.
        "An ancient book of spells glowing on the altar",
        "A book of spells bound in leather",
        "A poster of a girl with reading glasses",
        "A sign on the wall behind her reading glasses",
    ],
)
def test_quoted_or_non_text_words_are_left_alone(prompt):
    assert only(run(FLUX_DEV, prompt), "guide.quote_text") == []
    # Nor does the critic take them for lettering.
    if not prompt_rules.QUOTED.search(prompt):
        assert prompt_rules.rendered_spans(prompt) == []


@pytest.mark.parametrize(
    "prompt, target",
    [
        # A participle not first, an -ed one with no preposition after (a
        # sign says "closed" as often), or an -ing one before a word other
        # than an adverb, or before an adverb of when.
        ("A t-shirt with the words OPEN shimmering across the chest", "OPEN shimmering"),
        ("A shop door with the words closed", "closed"),
        ("A shop door with the words closed temporarily", "closed temporarily"),
        ("A shop door with the words opening shortly", "opening shortly"),
        ("A shop door with the words closing early today", "closing early today"),
        ("A cake with the words happy birthday piped in icing", "happy birthday piped"),
        # Lettering too, but not in capitals: it keeps its finding.
        ("A billboard with the words coming soon above the highway", "coming soon"),
        ("A cake with the words Happy Birthday in blue icing", "Happy Birthday"),
    ],
)
def test_words_not_in_capitals_after_a_noun_trigger_get_no_one_click_quote(prompt, target):
    # "With the words" may be followed by the words or by how they look, and
    # a quote of "set" letters the description: no Apply unless in capitals.
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == target and finding.suggestion is None
    assert "Put the exact words to render in quotation marks." in finding.message
    # Where the words end is as sure as it was: what follows is description.
    start, end = prompt.index(target), prompt.index(target) + len(target)
    assert prompt_rules.rendered_targets(prompt) == [(start, end, True, end)]


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("A cake with the words HAPPY BIRTHDAY in blue icing", "HAPPY BIRTHDAY"),
        ("A sign with the words FISH and CHIPS above the door", "FISH and CHIPS"),
        ("A cake with the text 2024 in gold", "2024"),
        ("A poster with the headline BREAKING NEWS", "BREAKING NEWS"),
    ],
)
def test_capitals_after_a_noun_trigger_keep_their_one_click_quote(prompt, target):
    [finding] = only(run(KREA_DEV, prompt), "guide.quote_text")
    start = prompt.index(target)
    assert finding.suggestion == f'{prompt[:start]}"{target}"{prompt[start + len(target):]}'


def test_a_negation_after_a_trigger_is_still_one():
    # Read as lettering, "without glasses by the fire" hid its negation.
    prompt = "An old man with a newspaper reading without glasses by the fire"
    [finding] = only(run(KREA_DEV, prompt), "guide.negation")
    assert spanned(prompt, finding) == "without glasses"
    # So did one further on in a participle's lowercase words: they stop at
    # it, and it is the negation rule's. The lettering may still hold it
    # ("love without limits"), so `rendered_targets` reads on through it to
    # where the words stop without it, and what follows is description: left
    # at the negation with `ends` False, the critic read on to the clause's
    # end and took "above the stage" for lettering. Its `head` keeps the
    # rules' stop, where lettering starts before.
    for prompt, target, phrase, reach in (
        ("A boy on a box reading comics without pictures", "comics", "without pictures",
         "comics without pictures"),
        ("A girl on a crate reading letters never sent", "letters", "never sent",
         "letters never sent"),
        ("A banner reading love without limits", "love", "without limits",
         "love without limits"),
        # The negation rule's window is its own: three words, a trailing
        # "the" or "in" dropped.
        ("A banner reading love without limits above the stage", "love",
         "without limits above", "love without limits"),
        ("A sign reading dogs not allowed in red letters on the gate", "dogs",
         "not allowed in red", "dogs not allowed"),
        ("A poster reading books without pictures in the attic", "books",
         "without pictures", "books without pictures"),
        # A place word after the negation is no phrase it closes: "comics
        # not in color" are black and white. The reach runs to "in".
        ("A boy on a box reading comics not in color", "comics", "not in color",
         "comics not"),
    ):
        findings = run(KREA_DEV, prompt)
        [quote] = only(findings, "guide.quote_text")
        assert spanned(prompt, quote) == target and quote.suggestion is None
        assert [spanned(prompt, f) for f in only(findings, "guide.negation")] == [phrase]
        start = prompt.index(reach)
        assert prompt_rules.rendered_targets(prompt) == [
            (start, start + len(reach), True, start + len(target))
        ]
        assert prompt_rules.rendered_spans(prompt) == [(start, start + len(reach))]
    # The same end as the words have with no negation among them.
    prompt = "A banner reading love forever above the stage"
    start, end = prompt.index("love forever"), prompt.index(" above")
    assert prompt_rules.rendered_targets(prompt) == [(start, end, True, end)]
    # Only where the negation rule reads one from there: not inside a
    # negation that starts a word earlier or opens the words, nor at one with
    # nothing after it, nor right after "or" or "than", whose phrase it
    # closes. Those are lettering whole, as before the stop; the last two
    # were cut at "or" and "than" wherever the words went on, and the
    # placement read as the absence.
    for prompt, target in (
        ("A red sign reading do not enter above the door", "do not enter"),
        ("A sign reading please do not touch", "please do not touch"),
        ("A hotel sign saying does not compute", "does not compute"),
        ("A t-shirt reading no pain no gain", "no pain no gain"),
        ("A t-shirt reading never say never", "never say never"),
        ("A t-shirt reading now or never", "now or never"),
        ("A poster reading now or never in bold type", "now or never"),
        ("A sign reading better late than never on the door", "better late than never"),
        ("A sign reading ready or not on the door", "ready or not"),
        ("A sign reading yes or no in red letters", "yes or no"),
        ("A t-shirt reading now or never again", "now or never again"),
        # Nor where the words before make it their own: a verb it negates,
        # "but", a verb of saying, or a negation among them. A joining word
        # right after it goes on with it.
        ("A shop sign reading we are not open", "we are not open"),
        ("A shop sign reading we're not open on the door", "we're not open"),
        ("A sign reading we will never forget above the memorial", "we will never forget"),
        ("A sign reading we are not for sale", "we are not for sale"),
        ("A tombstone reading gone but not forgotten", "gone but not forgotten"),
        ("A poster reading say no to war above the crowd", "say no to war"),
        ("A diner sign reading no shoes no shirt no service", "no shoes no shirt no service"),
        ("A sign reading no shoes no shirt no service on the door",
         "no shoes no shirt no service"),
    ):
        findings = run(KREA_DEV, prompt)
        assert [spanned(prompt, f) for f in only(findings, "guide.quote_text")] == [target]
        assert only(findings, "guide.negation") == []
        start, end = prompt.index(target), prompt.index(target) + len(target)
        assert prompt_rules.rendered_targets(prompt) == [(start, end, True, end)]
    # Capitals after a participle are lettering, negation and all.
    prompt = "A red sign reading OPEN NOT CLOSED above the door"
    findings = run(KREA_DEV, prompt)
    assert [spanned(prompt, f) for f in only(findings, "guide.quote_text")] == ["OPEN NOT CLOSED"]
    assert only(findings, "guide.negation") == []


def test_animagine_says_text_is_not_rendered():
    prompt = '1girl, holding sign, sign says "hi", masterpiece'
    [finding] = only(run(ANIMAGINE, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == '"hi"' and finding.suggestion is None
    assert "does not" in finding.message
