"""The analyser's rule checks, held against each checkpoint's own guide.

Pure and fast: no tokenizer, no network, no weights. The `analysis` data is
held to its guide's prose - every cite verbatim, every entry sourced - the
authors' own examples to their own guide (no warning), and each rule is
driven from both sides: the text it exists for, and the near miss it must
leave alone.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise

import pytest

from backend import guides, launch, prompt_rules
from backend.backends.netayume_loading import SYSTEM_PROMPT
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for, guide_for
from backend.schemas import GenerateRequest

FLUX_DEV = "black-forest-labs/FLUX.1-dev"
KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"
SCHNELL = "black-forest-labs/FLUX.1-schnell"
KLEIN = "black-forest-labs/FLUX.2-klein-4B"
KREA2 = "krea/Krea-2-Turbo"
QWEN = "Qwen/Qwen-Image"
ERNIE = "baidu/ERNIE-Image-Turbo"
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
KOHAKU = "KBlueLeaf/Kohaku-XL-Zeta"
ILLUSTRIOUS = "OnomaAIResearch/Illustrious-XL-v2.0"
WAI = "LyliaEngine/waiIllustriousSDXL_v170"
Z_ANIME = "SeeSee21/Z-Anime"
ANIMA = "circlestone-labs/Anima"
NETAYUME = "duongve/NetaYume-Lumina-Image-2.0"
NEWBIE = "NewBie-AI/NewBie-image-Exp0.1"


def _catalogue_repos() -> list[str]:
    return [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]


def _all_guides() -> dict[str, dict]:
    # Every guide constant, the two undistilled ones outside the catalogue too.
    return {name: getattr(guides, name) for name in dir(guides) if name.endswith("_GUIDE")}


def _entries(analysis: dict):
    """Each rule entry with its key; the list-valued ones entry by entry."""
    for key, value in analysis.items():
        if isinstance(value, dict):
            yield key, value
        elif isinstance(value, list):
            for item in value:
                yield key, item


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def run(model_id: str, prompt: str, categories: dict[str, str] | None = None, **fields):
    """`check` as the route will call it: the request, and applied() on the
    checkpoint's own profile answers."""
    request = GenerateRequest(prompt=prompt, **fields)
    profile = defaults_for(model_id)
    applied = request.applied(
        negative_prompt_supported=profile["negative_prompt_supported"],
        cfg_threshold=profile["cfg_threshold"],
        mu_supported=profile["mu_supported"],
        true_cfg_supported=profile["true_cfg_supported"],
        reference_supported=profile["reference_supported"],
    )
    findings = prompt_rules.check(model_id, request, applied, categories)
    for finding in findings:
        text = prompt if finding.field == "prompt" else (request.negative_prompt or "")
        if finding.start is not None:
            assert 0 <= finding.start <= finding.end <= len(text), finding
        if finding.source in ("guide", "author"):
            assert finding.cite and finding.url and finding.url.startswith("https://"), finding
        else:
            assert finding.source == "heuristic" and finding.cite is None, finding
    return findings


def only(findings, rule):
    return [finding for finding in findings if finding.rule == rule]


def silent(model_id: str, prompt: str, rule: str, categories=None, **fields) -> bool:
    """`rule` has nothing to say about this prompt on this checkpoint."""
    return only(run(model_id, prompt, categories, **fields), rule) == []


def spanned(prompt: str, finding) -> str:
    return prompt[finding.start:finding.end]


# ------------------------------------------------------------------ the data


def test_every_catalogue_checkpoint_has_an_analysis():
    # Resolved the way every other guide lookup is, official and lite-infer
    # ids alike; the fallback's app-written prose has nothing to cite.
    for repo in _catalogue_repos():
        analysis = _profile_for(repo)["guide"].get("analysis")
        assert analysis, repo
    assert _FALLBACK["guide"]["analysis"] == {}
    # So an unrecognised checkpoint gets the app's own pattern matching, labelled.
    findings = run("nobody/whatever", "a cat, a cat, no people, 8k, masterpiece")
    assert findings and all(finding.source == "heuristic" for finding in findings)


@pytest.mark.parametrize("name", sorted(_all_guides()))
def test_every_cite_is_the_guides_own_words(name):
    # An entry is a claim the prose already makes. A cite that is not a
    # verbatim piece of the prompt hint - or, for an author placeholder, of
    # an example - is an invented rule wearing a quotation.
    guide = _all_guides()[name]
    analysis = guide["analysis"]
    for key, entry in _entries(analysis):
        if key == "url":
            continue
        assert entry.get("cite"), (name, key)
        if entry.get("source") == "author":
            assert any(entry["cite"] in example for example in guide["examples"]), (name, key)
            assert any(re.search(entry["pattern"], example) for example in guide["examples"])
        else:
            assert entry["cite"] in guide["prompt"], (name, key, entry["cite"])
        url = entry.get("url") or analysis.get("url")
        assert url and url.startswith("https://"), (name, key)


@pytest.mark.parametrize("name", sorted(_all_guides()))
def test_the_analysis_holds_no_braces(name):
    # The same rule as the prose it cites: a brace in a cite that ever
    # reaches a format string fails the way a brace in a hint does.
    assert not any("{" in text or "}" in text for text in _strings(_all_guides()[name]["analysis"]))


def test_the_analysis_stays_out_of_the_status_payload():
    # Rules are the server's; the page gets findings. A top-level profile key
    # would leak into ModelStatus.defaults and fail its validation.
    from backend.schemas import PromptGuide

    for repo in _catalogue_repos():
        assert "analysis" not in guide_for(repo), repo
        assert "analysis" not in defaults_for(repo), repo
        assert "analysis" not in PromptGuide(**guide_for(repo)).model_dump()


def test_tag_order_slots_are_the_helpers_own():
    # Every slot a group, category, suffix or year form places an item in is a
    # step of `tag_helper.order`, and every group named is one the helper has.
    for name, guide in _all_guides().items():
        order = guide["analysis"].get("order") or {}
        if "general" not in order:
            continue
        helper = guide["tag_helper"]
        steps = set(helper["order"])
        labels = {group["label"] for group in helper["groups"]}
        assert set(order["groups"]) <= labels, name
        slots = set(order["groups"].values()) | set(order["categories"].values())
        slots |= {order["general"]} | {order[k] for k in ("suffix", "year") if order.get(k)}
        assert slots <= steps, name
        assert set(order["categories"]) == {"general", "artist", "copyright", "character", "meta"}


@pytest.mark.parametrize(
    "model_id",
    [KREA2, "krea/Krea-2-Raw", QWEN, "Tongyi-MAI/Z-Image-Turbo", ERNIE, Z_ANIME, ANIMA, NETAYUME,
     "nobody/whatever"],
)
def test_no_order_is_invented_where_the_guide_states_none(model_id):
    assert prompt_rules.documented_order(model_id) is None


def test_documented_orders_are_the_guides():
    flux = prompt_rules.documented_order(FLUX_DEV)
    assert flux.steps == ["subject", "setting", "style", "camera", "lighting"]
    assert flux.cite in guides.FLUX1_GUIDE["prompt"] and flux.url.startswith("https://docs.bfl.ai/")
    for repo in (KREA_DEV, SCHNELL):
        assert prompt_rules.documented_order(repo).steps == ["subject first"]
    kleins = [c.repo for c in launch.CHECKPOINTS if "klein" in c.repo]
    assert len(kleins) == 4
    for repo in kleins:
        steps = prompt_rules.documented_order(repo).steps
        assert steps == ["subject", "action", "style", "context"]
    for repo in (ANIMAGINE, KOHAKU, ILLUSTRIOUS, WAI):
        assert prompt_rules.documented_order(repo).steps == guide_for(repo)["tag_helper"]["order"]
    assert "<general_tags>" in prompt_rules.documented_order(NEWBIE).steps
    # The lite-infer re-uploads read the same guide.
    lite = next(c for c in launch.CHECKPOINTS if c.repo == KREA_DEV).lite_repo("nvfp4")
    assert prompt_rules.documented_order(lite) == prompt_rules.documented_order(KREA_DEV)


# ------------------------------------------------------------------ the authors' own examples


def _examples():
    seen = set()
    for repo in _catalogue_repos():
        for example in guide_for(repo)["examples"]:
            if (repo.split("/")[0] != "lite-infer") and (repo, example) not in seen:
                seen.add((repo, example))
                yield repo, example


@pytest.mark.parametrize("repo, example", list(_examples()))
def test_the_authors_examples_raise_no_warning(repo, example):
    # An author's own prompt should pass its own guide. Two are templates:
    # Z-Anime's [character] and NewBie's $character_N$ are slots the author
    # left for a name, and a slot left unfilled reaches the encoder as
    # literal text - the one warning those examples are meant to raise.
    findings = run(repo, example)
    warnings = [finding for finding in findings if finding.severity == "warn"]
    if repo in (Z_ANIME, NEWBIE) and ("[character]" in example or "$character_" in example):
        assert warnings and all(finding.rule == "author.placeholder" for finding in warnings)
        assert all(finding.source == "author" for finding in warnings)
    else:
        assert warnings == [], (repo, example, warnings)


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
    for model_id, prompt in (
        (KREA_DEV, "Golden hour light over misty mountains, a lone hiker on a ridge"),
        (FLUX_DEV, "1girl, smile"),
        (KLEIN, "A red fox in fresh snow"),
        (NEWBIE, guides.NEWBIE_GUIDE["examples"][0]),
        (ANIMAGINE, '{"scene": "a cafe"}'),
        (ANIMA, "1girl, smile"),
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


# ------------------------------------------------------------------ prose rules


def test_negation_warns_where_no_negative_prompt_runs():
    prompt = "A quiet street with no people in the background, without cars"
    findings = only(run(FLUX_DEV, prompt), "guide.negation")
    assert [spanned(prompt, f) for f in findings] == ["no people", "without cars"]
    assert all(f.severity == "warn" for f in findings)
    # BFL's own replacement for a phrase its table names, quoted as theirs.
    assert '"empty", "deserted" or "solitary"' in findings[0].message
    assert findings[0].url == guides._BFL_TECHNICAL
    # With True CFG on, the negative prompt runs: advice, not a warning.
    running = only(run(FLUX_DEV, prompt, negative_prompt="cars", true_cfg=4.0), "guide.negation")
    assert [f.severity for f in running] == ["info", "info"]
    # klein has no negative prompt at all.
    klein = only(run(KLEIN, "A desk, do not show any text"), "guide.negation")
    assert [spanned("A desk, do not show any text", f) for f in klein] == ["do not show any text"]
    assert klein[0].severity == "warn"


@pytest.mark.parametrize(
    "model_id, prompt",
    [
        (FLUX_DEV, 'A shop sign that says "no parking" on a brick wall'),
        (FLUX_DEV, "An old man, no longer young, on a bench"),
        (FLUX_DEV, "A notable nothing, nonetheless"),
        (KREA2, "a street with no people"),
        (ILLUSTRIOUS, "analogous colors, scenery, no lighting, night"),
        # Unquoted lettering is text to render - the quote rule's - not an absence.
        (FLUX_DEV, "A red sign reading NO PARKING on a brick wall at dusk"),
        (FLUX_DEV, "A neon sign that says No Vacancy above a motel"),
        # The particle of a romanised title or name, between capitals.
        (FLUX_DEV, "Fern from Sousou no Frieren standing in a meadow"),
        (FLUX_DEV, "A poster of Kimi no Na wa and Shingeki no Kyojin on a bedroom wall"),
        # Avoiding something is an action to draw; a capital "No" in
        # mid-sentence or before a title-case word is a name.
        (KREA_DEV, "A cat trying to avoid the rain under a porch"),
        (KREA_DEV, "A cat, trying to avoid the rain"),
        (KREA_DEV, "A skier avoiding trees on a steep slope"),
        (KREA_DEV, "The film Dr. No playing on a TV"),
        (KREA_DEV, "A poster for No Country for Old Men on a wall"),
        (KREA_DEV, "A No Smoking sign on a wall"),
    ],
)
def test_negation_leaves_quotes_idioms_and_other_guides_alone(model_id, prompt):
    assert only(run(model_id, prompt), "guide.negation") == []


@pytest.mark.parametrize(
    "prompt, phrases",
    [
        # "avoid" as an instruction still asks for an absence.
        ("A clean desk, avoid clutter", ["avoid clutter"]),
        ("Avoid text. A quiet beach at dawn", ["Avoid text"]),
        ("Please avoid text on the walls", ["avoid text"]),
        ("A beach at dawn. No people anywhere.", ["No people anywhere"]),
        ("A quiet beach at dawn — No people in sight", ["No people in sight"]),
        # Only a word or a title's abbreviation right before a head keeps it
        # inside a clause: a year's or a short word's full stop, a spaced
        # dash, a bullet or a bar opens one.
        ("A street photo taken in 1998. No people in sight.", ["No people in sight"]),
        ("A dog looking up. No people.", ["No people"]),
        ("A quiet beach at 6 a.m. No people.", ["No people"]),
        ("A tidy desk with a lamp on it. Avoid clutter.", ["Avoid clutter"]),
        ("A beach at dawn - No people in sight", ["No people in sight"]),
        ("A beach at dawn - avoid people", ["avoid people"]),
        ("A beach at dawn\n- No people", ["No people"]),
        ("A beach at dawn\n* Avoid people", ["Avoid people"]),
        ("A beach at dawn | No people", ["No people"]),
        # An acronym after "No" is no title.
        ("Photorealistic portrait. No CGI look, no plastic skin.", ["No CGI look", "no plastic skin"]),
        ("A living room. No TV in the room.", ["No TV"]),
        # A name is not cut at the window's end.
        ("A climber without ropes on El Capitan at dawn", ["without ropes on El Capitan"]),
    ],
)
def test_negation_near_misses_are_still_negations(prompt, phrases):
    assert [spanned(prompt, f) for f in only(run(KREA_DEV, prompt), "guide.negation")] == phrases


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
    "model_id, prompt, target, ends",
    [
        # 的 links the lettering to what it is on: a wooden sign reading
        # 欢迎光临. Whether 的 is the lettering's own is the guess; what
        # follows it is description, so nothing may read on past it.
        (QWEN, "一块写着欢迎光临的木质招牌挂在门口", "欢迎光临", True),
        (ERNIE, "一件印着我爱北京的白色T恤", "我爱北京", True),
        (ERNIE, "一块写有欢迎字样的牌子", "欢迎字样", True),
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
    assert [(prompt[s:e], x) for s, e, x in prompt_rules.rendered_targets(prompt)] == [
        (target, ends)
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
        # A title is its marks and what they hold.
        ("封面上写着《营业中》的招牌", "《营业中》"),
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
    assert [(prompt[start:end], ends) for start, end, ends in targets] == [
        ("OPEN", True), ('"SALE"', True), ("fish", False),
    ]
    assert prompt_rules.rendered_spans(prompt) == [(start, end) for start, end, _ in targets]
    # No one-click quote on these, but what follows is description, not
    # more lettering: the sign itself after 的, "about space" after a
    # participle's lowercase words, even where a joining word stopped them.
    chinese = "一块写着欢迎光临的木质招牌"
    assert prompt_rules.rendered_targets(chinese) == [(4, 8, True)]
    assert prompt_rules.rendered_spans(chinese) == [(4, 8)]
    comics = "A boy sitting on a box reading comics about space and a cat asleep"
    assert only(run(KREA_DEV, comics), "guide.quote_text")[0].suggestion is None
    assert [(comics[s:e], x) for s, e, x in prompt_rules.rendered_targets(comics)] == [
        ("comics about space", True)
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


def test_a_negation_after_a_trigger_is_still_one():
    # Read as lettering, "without glasses by the fire" hid its negation.
    prompt = "An old man with a newspaper reading without glasses by the fire"
    [finding] = only(run(KREA_DEV, prompt), "guide.negation")
    assert spanned(prompt, finding) == "without glasses"


def test_animagine_says_text_is_not_rendered():
    prompt = '1girl, holding sign, sign says "hi", masterpiece'
    [finding] = only(run(ANIMAGINE, prompt), "guide.quote_text")
    assert spanned(prompt, finding) == '"hi"' and finding.suggestion is None
    assert "does not" in finding.message


def test_word_range():
    short = only(run(FLUX_DEV, "A red fox in the snow"), "guide.words")
    assert short and "under" in short[0].message
    enough = "A red fox " + "sits quietly in fresh snow " * 7
    assert only(run(FLUX_DEV, enough), "guide.words") == []
    long = "A red fox " + "sits quietly in fresh snow " * 20
    assert "over" in only(run(FLUX_DEV, long), "guide.words")[0].message
    # schnell's guide gives no range this app can stand behind.
    assert only(run(SCHNELL, "A red fox"), "guide.words") == []


def test_style_against_the_guide():
    assert only(run(FLUX_DEV, "red hair, blue eyes, smile, outdoors"), "guide.style")
    # Two items is a phrase with a comma, not yet a tag list.
    assert only(run(FLUX_DEV, "red hair, blue eyes"), "guide.style") == []
    assert only(run(ANIMAGINE, "A girl with green hair stands outside at night."), "guide.style")
    # JSON is klein's documented alternative to sentences.
    assert only(run(KLEIN, '{"scene": "a cafe", "subjects": ["a barista"]}'), "guide.style") == []
    assert only(run(KOHAKU, "A girl with green hair stands outside at night."), "guide.style") == []
    # The author's phrase list is not called a tag list.
    assert silent(Z_ANIME, guides.Z_ANIME_GUIDE["examples"][0], "guide.style")
    # A pure tag list with a long series name is not "partly prose"; a name
    # with an article is, until the tag index says it is a series.
    long_name = "1girl, jojo no kimyou na bouken, safe, smile, masterpiece"
    assert silent(ANIMAGINE, long_name, "guide.style")
    titled = "1girl, link, the legend of zelda, safe, smile"
    assert only(run(ANIMAGINE, titled), "guide.style")
    known = {"link": "character", "the legend of zelda": "copyright", "smile": "general"}
    assert silent(ANIMAGINE, titled, "guide.style", known)


def test_one_paragraph():
    prompt = "A fox in fresh snow.\n\nGolden hour light from the left."
    [finding] = only(run(KREA2, prompt), "guide.one_paragraph")
    assert finding.suggestion == "A fox in fresh snow. Golden hour light from the left."
    [listed] = only(run(ERNIE, "A poster:\n- a title\n- a subtitle"), "guide.one_paragraph")
    assert listed.suggestion is None
    assert only(run(FLUX_DEV, prompt), "guide.one_paragraph") == []


def test_qwen_positive_magic():
    [finding] = only(run(QWEN, "A cat on a sofa"), "guide.qwen_magic")
    assert finding.suggestion == "A cat on a sofa, Ultra HD, 4K, cinematic composition."
    [finding] = only(run(QWEN, "A cat on a sofa."), "guide.qwen_magic")
    assert finding.suggestion == "A cat on a sofa. Ultra HD, 4K, cinematic composition."
    [finding] = only(run(QWEN, "一只猫在沙发上"), "guide.qwen_magic")
    assert finding.suggestion == "一只猫在沙发上, 超清，4K，电影级构图."
    assert silent(QWEN, "A cat on a sofa, ultra HD, 4k, cinematic composition", "guide.qwen_magic")
    assert silent(FLUX_DEV, "A cat on a sofa", "guide.qwen_magic")


def test_a_recommended_negative_prompt():
    [finding] = only(run(ILLUSTRIOUS, "1girl, masterpiece"), "guide.negative_prompt")
    assert finding.field == "negative_prompt"
    assert finding.suggestion == guides.ILLUSTRIOUS_GUIDE["tag_helper"]["negative_prompt"]
    assert only(
        run(ILLUSTRIOUS, "1girl, masterpiece", negative_prompt="worst quality", guidance_scale=5),
        "guide.negative_prompt",
    ) == []
    # Filled in but dropped by applied(): Qwen's runs only above True CFG 1.
    [dropped] = only(run(QWEN, "A cat", negative_prompt="blur"), "guide.negative_prompt")
    assert "does not run" in dropped.message and dropped.suggestion is None
    assert silent(QWEN, "A cat", "guide.negative_prompt", negative_prompt=" ", true_cfg=4.0)
    # Kohaku's author publishes no universal negative prompt.
    assert only(run(KOHAKU, "1girl"), "guide.negative_prompt") == []


def test_references_are_named_when_there_are_several():
    prompt = "Put the jacket on the man"
    [finding] = only(run(KLEIN, prompt, reference=[1, 2]), "guide.references")
    assert "image 1, 2" in finding.message
    named = "Put the jacket from image 1 on the man from image 2"
    assert only(run(KLEIN, named, reference=[1, 2]), "guide.references") == []
    # A list or a range names every image in it.
    for listed, references in (
        ("Combine images 1 and 2: the cat from one in the room of the other", [1, 2]),
        ("Blend images 1 & 2 into one scene", [1, 2]),
        ("The people from images 1, 2 and 3 at one table", [1, 2, 3]),
        ("Use images 1-3 as one set", [1, 2, 3]),
        ("Use images 1 to 3 as one set", [1, 2, 3]),
    ):
        assert only(run(KLEIN, listed, reference=references), "guide.references") == [], listed
    [partial] = only(run(KLEIN, "Combine images 1 and 2", reference=[1, 2, 3]), "guide.references")
    assert partial.message.endswith("from image 3.")
    # One reference needs no number: "Turn this cat into a dog" is BFL's own.
    assert only(run(KLEIN, prompt, reference=[1]), "guide.references") == []


def test_netayume_system_prefix_is_not_typed_twice():
    prompt = f"{SYSTEM_PROMPT} <Prompt Start> 1girl, solo"
    findings = only(run(NETAYUME, prompt), "guide.system_prefix")
    assert [spanned(prompt, f) for f in findings] == [SYSTEM_PROMPT, "<Prompt Start>"]
    assert all(f.severity == "warn" and f.suggestion == "1girl, solo" for f in findings)
    assert only(run(NETAYUME, "1girl, solo"), "guide.system_prefix") == []


def test_netayume_system_prefix_is_found_in_the_text_itself():
    # "İ" lowers to two characters, so a span found in text.lower() selected
    # the wrong words and the suggestion cut into the tags.
    prompt = f"İİ {SYSTEM_PROMPT.upper()} 1girl"
    [finding] = only(run(NETAYUME, prompt), "guide.system_prefix")
    assert spanned(prompt, finding) == SYSTEM_PROMPT.upper()
    assert finding.suggestion == "İİ 1girl"
    # Only the cut's seam is tidied: a blank line and the spacing inside
    # quoted lettering are the user's.
    prompt = f'{SYSTEM_PROMPT}   1girl, sign reading "OPEN    NOW"\n\nsmile'
    [finding] = only(run(NETAYUME, prompt), "guide.system_prefix")
    assert finding.suggestion == '1girl, sign reading "OPEN    NOW"\n\nsmile'
    # A comma either side of the cut is one comma, and none opens the text.
    for prompt, suggestion in (
        (f"1girl,{SYSTEM_PROMPT},smile", "1girl, smile"),
        (f"1girl, {SYSTEM_PROMPT}, smile", "1girl, smile"),
        (f"{SYSTEM_PROMPT}, 1girl", "1girl"),
    ):
        [finding] = only(run(NETAYUME, prompt), "guide.system_prefix")
        assert finding.suggestion == suggestion


def test_newbie_xml():
    broken = "<character_1>\n<n>miku</n>\n<gender>1girl</gender>\n</character_2>"
    [finding] = only(run(NEWBIE, broken), "guide.xml")
    assert finding.severity == "warn" and "does not parse" in finding.message
    assert spanned(broken, finding).startswith("</character_2>")
    partial = "<character_1><n>miku</n><gender>1girl</gender></character_1>"
    messages = [f.message for f in only(run(NEWBIE, partial), "guide.xml")]
    assert any("appearance, clothing, expression, action, position" in m for m in messages)
    assert any("<general_tags>" in m for m in messages)
    assert all(f.severity == "info" for f in only(run(NEWBIE, partial), "guide.xml"))
    # A bare ampersand is text, not broken markup.
    amp = partial.replace("miku", "black & white") + "<general_tags>x</general_tags>"
    assert not any(f.severity == "warn" for f in run(NEWBIE, amp))
    # Several characters in plain tags: the authors' case for XML.
    for plain in ("2girls, school uniform, holding hands", "1girl, 1boy, holding hands"):
        [hint] = only(run(NEWBIE, plain), "guide.xml")
        assert hint.severity == "info"
    assert only(run(NEWBIE, "1girl, school uniform"), "guide.xml") == []


def test_placeholders_are_the_authors_slots():
    example = guides.NEWBIE_GUIDE["examples"][0]
    findings = only(run(NEWBIE, example), "author.placeholder")
    assert [spanned(example, f) for f in findings] == ["$character_1$", "$character_2$"]
    assert all(f.source == "author" and f.cite in example for f in findings)
    filled = example.replace("$character_1$", "miku").replace("$character_2$", "teto")
    assert silent(NEWBIE, filled, "author.placeholder")
    assert only(run(Z_ANIME, "Portrait of [character], soft light"), "author.placeholder")


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


CATEGORIES = {
    "smile": "general",
    "honkai: star rail": "copyright",
    "firefly": "character",
    "outdoors": "general",
}


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


OVER_CAP = "masterpiece, " + ", ".join(f"t{i}" for i in range(300)) + ", 1girl"
# Comma items the categories place out of order, inside a JSON list.
JSON_CATEGORIES = {'"smile"': "general", '"firefly"': "character"}


@pytest.mark.parametrize(
    "prompt, twin, categories",
    [
        # Past the item cap: no 1,700-character Apply, no quadratic placement.
        (OVER_CAP, "masterpiece, t0, 1girl", None),
        ('[0, "smile", "firefly", 0]', 'x, "smile", "firefly", y', JSON_CATEGORIES),
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
    # Anima reads artists; an item already prefixed needs no lookup.
    assert prompt_rules.category_items(ANIMA, "1girl, @nnn yryr, oomuro sakurako") == [
        "1girl", "oomuro sakurako",
    ]


# ------------------------------------------------------------------ tag checkpoints' specifics


def test_quality_suffix_and_recipe():
    [finding] = only(run(ANIMAGINE, "1girl, solo, masterpiece"), "guide.quality")
    assert finding.suggestion == "1girl, solo, masterpiece, high score, great score, absurdres"
    [recipe] = only(run(WAI, "1girl,solo"), "guide.quality")
    assert recipe.suggestion == "1girl,solo,masterpiece,best quality,amazing quality"
    finished = "1girl, beach, masterpiece, best quality, amazing quality"
    assert silent(WAI, finished, "guide.quality")
    # The example is the recipe alone; the author says to add a subject first.
    assert only(run(WAI, guides.WAI_GUIDE["examples"][0]), "guide.subject")
    assert silent(WAI, finished, "guide.subject")


def test_illustrious_quality_tier_and_year():
    assert "No quality tier" in only(run(ILLUSTRIOUS, "1girl, solo"), "guide.quality")[0].message
    prompt = "1girl, masterpiece, best quality"
    [tiers] = only(run(ILLUSTRIOUS, prompt), "guide.quality")
    assert spanned(prompt, tiers) == "best quality"
    prompt = "1girl, masterpiece, recent, newest"
    [years] = only(run(ILLUSTRIOUS, prompt), "guide.quality")
    assert spanned(prompt, years) == "newest"
    assert only(run(ILLUSTRIOUS, "1girl, solo, masterpiece, newest"), "guide.quality") == []


def test_illustrious_composition_and_rating_names():
    prompt = "1girl, upper body, full body, masterpiece"
    [finding] = only(run(ILLUSTRIOUS, prompt), "guide.composition")
    assert spanned(prompt, finding) == "full body"
    assert only(run(ILLUSTRIOUS, "1girl, upper body, masterpiece"), "guide.composition") == []
    [rating] = only(run(ILLUSTRIOUS, "1girl, safe, masterpiece"), "guide.rating")
    assert rating.suggestion == "1girl, general, masterpiece"
    # The release post accepts these; the guide does not repeat them, and
    # nothing here calls them wrong.
    for accepted in ("nsfw", "sfw", "r-18", "general"):
        assert only(run(ILLUSTRIOUS, f"1girl, {accepted}, masterpiece"), "guide.rating") == []


def test_kohaku_explicit_and_underscores():
    [pair] = only(run(KOHAKU, "1girl, explicit"), "guide.rating")
    assert pair.suggestion == "1girl, nsfw, explicit"
    assert only(run(KOHAKU, "1girl, nsfw, explicit"), "guide.rating") == []
    prompt = "1girl, long_hair, ^_^, o_o"
    [finding] = only(run(KOHAKU, prompt), "guide.underscores")
    assert spanned(prompt, finding) == "long_hair"
    assert finding.suggestion == "1girl, long hair, ^_^, o_o"


def test_anima_tags_and_sentences():
    [score] = only(run(ANIMA, "score_7, long_hair, score_8_up"), "guide.underscores")
    assert score.suggestion == "score_7, long hair, score_8_up"
    [caps] = only(run(ANIMA, "1girl, Hatsune Miku, smile"), "guide.lowercase")
    assert caps.suggestion == "1girl, hatsune miku, smile"
    # In prose, names take their capitals.
    assert silent(ANIMA, "Hatsune Miku sings on a stage under bright lights.", "guide.lowercase")
    categories = {"nnn yryr": "artist"}
    [artist] = only(run(ANIMA, "1girl, nnn yryr, smile", categories), "guide.artist_prefix")
    assert artist.suggestion == "1girl, @nnn yryr, smile"
    assert only(run(ANIMA, "1girl, nnn yryr, smile"), "guide.artist_prefix") == []
    one = "A girl with long blonde hair stands in a field of flowers."
    assert only(run(ANIMA, one), "guide.sentences")
    two = one + " Soft evening light falls across her face."
    assert only(run(ANIMA, two), "guide.sentences") == []


def test_wai_rating_and_the_nsfw_negative():
    prompt = "1girl, general, beach, masterpiece, best quality, amazing quality"
    [bare] = only(run(WAI, prompt), "guide.rating_negative")
    assert bare.field == "negative_prompt" and bare.suggestion == "nsfw"
    negative = guides.WAI_GUIDE["tag_helper"]["negative_prompt"]
    rule = "guide.rating_negative"
    [card] = only(run(WAI, prompt, negative_prompt=negative, guidance_scale=6), rule)
    assert card.suggestion == "bad quality,worst quality,worst detail,sketch,censor,nsfw"
    assert silent(WAI, prompt, rule, negative_prompt="lowres, nsfw", guidance_scale=6)
    # Written, but dropped by applied() at CFG 1.
    [idle] = only(run(WAI, prompt, negative_prompt="lowres, nsfw", guidance_scale=1), rule)
    assert "does not run" in idle.message and idle.suggestion is None
    assert only(run(WAI, "1girl, explicit, masterpiece"), "guide.rating_negative") == []


# ------------------------------------------------------------------ heuristics


def test_quality_boilerplate_is_labelled_a_heuristic():
    prompt = "A portrait of a woman in a garden, masterpiece, best quality, 8k"
    [finding] = only(run(FLUX_DEV, prompt), "heuristic.quality_boilerplate")
    assert finding.source == "heuristic" and finding.cite is None and finding.url is None
    assert spanned(prompt, finding) == "masterpiece"
    # One detail word is a description; Qwen asks for "Ultra HD, 4K"; tag
    # checkpoints' quality tags are their authors' own.
    rule = "heuristic.quality_boilerplate"
    assert silent(FLUX_DEV, "A highly detailed map of an island", rule)
    assert silent(QWEN, "A cat, Ultra HD, 4K, masterpiece", rule)
    assert silent(ANIMAGINE, "1girl, masterpiece, best quality, 8k", rule)


def test_repeated_items():
    prompt = "1girl, smile, long hair, Smile, long_hair"
    [finding] = only(run(KOHAKU, prompt), "heuristic.repeated")
    assert finding.source == "heuristic" and spanned(prompt, finding) == "Smile"
    assert finding.suggestion == "1girl, smile, long hair"
    # NewBie characters share tags on purpose.
    assert silent(NEWBIE, guides.NEWBIE_GUIDE["examples"][0], "heuristic.repeated")


def test_rules_leave_quoted_lettering_as_written():
    # A comma inside quotation marks splits comma items, not the words to
    # render: repeated, lowercased or de-underscored, the sign's text changed
    # in one click and the re-check found nothing wrong.
    rule = "heuristic.repeated"
    card = 'A Christmas card with the text "Ho, ho, ho, Merry Christmas" in red'
    assert silent(KREA_DEV, card, rule)
    assert silent(QWEN, 'A banner that reads "Go, team, go, team, go!"', rule)
    [repeat] = only(run(KOHAKU, '1girl, sign says "hi, hi", smile, smile'), rule)
    assert repeat.suggestion == '1girl, sign says "hi, hi", smile'
    assert silent(ANIMA, '1girl, sign says "HELLO", smile', "guide.lowercase")
    assert silent(ANIMA, '1girl, "I LOVE NY" shirt, smile', "guide.lowercase")
    [caps] = only(run(ANIMA, 'Sign says "HELLO", 1girl, smile'), "guide.lowercase")
    assert caps.suggestion == 'sign says "HELLO", 1girl, smile'
    assert silent(KOHAKU, '1girl, holding sign, "user_name", smile', "guide.underscores")
    [joined] = only(run(KOHAKU, '1girl, name_tag "user_name", smile'), "guide.underscores")
    assert joined.suggestion == '1girl, name tag "user_name", smile'


# ------------------------------------------------------------------ robustness


@pytest.mark.parametrize(
    "prompt",
    [
        ",,,", "<", "{", "[x]", "\n\n", "&", "no", "says", "写着", "<a><b></a>",
        "zero" + chr(0x200B) + "width", "reading", "the text",
        # Past the order check's item cap, and at the prompt's length limit.
        "1girl, " * 570, "no " * 1300, "a," * 1999,
    ],
)
def test_nothing_raises_on_odd_text(prompt):
    for repo in [c.repo for c in launch.CHECKPOINTS] + ["nobody/whatever"]:
        run(repo, prompt)
        prompt_rules.reorder_tags(repo, prompt, None)
        prompt_rules.category_items(repo, prompt)


def test_warnings_come_first():
    prompt = "Portrait of [character], soft light, soft light"
    severities = [finding.severity for finding in run(Z_ANIME, prompt)]
    assert severities[0] == "warn" and "info" in severities
    assert severities == sorted(severities, key=lambda severity: severity != "warn")
