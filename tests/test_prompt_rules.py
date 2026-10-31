"""Guide analysis data, author examples, heuristics, and rule robustness."""
from __future__ import annotations

import re

import pytest

from backend import guides, launch, prompt_rules
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for, guide_for

from tests.prompt_rules_helpers import (
    ANIMA,
    ANIMAGINE,
    ERNIE,
    FLUX_DEV,
    ILLUSTRIOUS,
    KOHAKU,
    KREA2,
    KREA_DEV,
    NETAYUME,
    NEWBIE,
    QWEN,
    SCHNELL,
    WAI,
    Z_ANIME,
    _all_guides,
    only,
    run,
    silent,
    spanned,
)
from tests.backend_helpers import catalogue_repos


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


# ------------------------------------------------------------------ the data


def test_every_catalogue_checkpoint_has_an_analysis():
    # Resolved the way every other guide lookup is, official and lite-infer
    # ids alike; the fallback's app-written prose has nothing to cite.
    for repo in catalogue_repos():
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

    for repo in catalogue_repos():
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
        slots |= set(order.get("tags", {}))
        assert slots <= steps, name
        assert set(order["categories"]) == {"general", "artist", "copyright", "character", "meta"}


@pytest.mark.parametrize(
    "model_id",
    [KREA2, "krea/Krea-2-Raw", QWEN, "Tongyi-MAI/Z-Image-Turbo", ERNIE, Z_ANIME, NETAYUME,
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
    for repo in (ANIMAGINE, KOHAKU, ILLUSTRIOUS, WAI, ANIMA, "circlestone-labs/Anima@aesthetic-v1.1"):
        assert prompt_rules.documented_order(repo).steps == guide_for(repo)["tag_helper"]["order"]
    assert "<general_tags>" in prompt_rules.documented_order(NEWBIE).steps
    # The lite-infer re-uploads read the same guide.
    lite = next(c for c in launch.CHECKPOINTS if c.repo == KREA_DEV).lite_repo("nvfp4")
    assert prompt_rules.documented_order(lite) == prompt_rules.documented_order(KREA_DEV)


# ------------------------------------------------------------------ the authors' own examples


def _examples():
    seen = set()
    for repo in catalogue_repos():
        for example in guide_for(repo)["examples"]:
            if (repo.split("/")[0] != "lite-infer") and (repo, example) not in seen:
                seen.add((repo, example))
                yield repo, example


@pytest.mark.parametrize("repo, example", list(_examples()))
def test_the_authors_examples_raise_no_warning(repo, example):
    # An author's own prompt should pass its own guide. Some are templates:
    # Z-Anime's [character] and [location] are slots the author left to fill,
    # and a slot left unfilled reaches the encoder as
    # literal text - the one warning those examples are meant to raise. The
    # slots are the guide's own (`guide_for`'s `slots`, what the page reads).
    findings = run(repo, example)
    warnings = [finding for finding in findings if finding.severity == "warn"]
    slots = guide_for(repo)["slots"]
    if any(re.search(pattern, example, re.IGNORECASE) for pattern in slots):
        assert repo == Z_ANIME
        assert warnings and all(finding.rule == "author.placeholder" for finding in warnings)
        assert all(finding.source == "author" for finding in warnings)
    else:
        assert warnings == [], (repo, example, warnings)


# Findings about something other than the example's own words: the negative
# box, the references, and a template's slots - each said beside the example
# by the menu itself ("needs a reference image", "fill in [character]").
_NOT_THE_EXAMPLES_WORDS = {"guide.negative_prompt", "heuristic.no_reference", "author.placeholder"}


@pytest.mark.parametrize("repo, example", list(_examples()))
def test_the_authors_examples_draw_no_finding_unnoted(repo, example):
    # Every rule, notes included, over every example: BFL's hello-world sign
    # drew "not in quotation marks" on eight checkpoints while the tip above
    # the menu said to quote text, and Illustrious's own prompts an order
    # note. The examples stay as written; where one parts from its guide the
    # menu says so beside it (`example_notes`), so Check is not the first news.
    rules = sorted({f.rule for f in run(repo, example)} - _NOT_THE_EXAMPLES_WORDS)
    assert rules == [] or example in guide_for(repo)["example_notes"], (repo, example, rules)


def _sentence_examples():
    # Only the guides with a sentence rule: a guide without one has nothing
    # to hold its examples to, and a skip per example hid the suite's real
    # skips among them.
    for repo, example in _examples():
        if "sentences" in _profile_for(repo)["guide"].get("analysis", {}):
            yield repo, example


def test_some_guides_ask_for_a_number_of_sentences():
    assert list(_sentence_examples())


@pytest.mark.parametrize("repo, example", list(_sentence_examples()))
def test_the_authors_examples_meet_their_guides_sentence_count(repo, example):
    # A note, not a warning, so the test above let it by: Anima's Example was
    # the card's caption cut by the app to its first sentence, and Check's
    # first word on the authors' own prompt was that it had too few. Where a
    # guide asks for a number of sentences its examples are quoted whole.
    assert only(run(repo, example), "guide.sentences") == [], (repo, example)


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
    # Nor do its words make the tag beside it prose.
    said = 'name_tag "hello, my name is bob", 1girl'
    [joined] = only(run(KOHAKU, said), "guide.underscores")
    assert joined.suggestion == 'name tag "hello, my name is bob", 1girl'


def test_a_tag_inside_lettering_is_no_tag():
    # A piece of a quotation that happens to equal a tag was rewritten in it,
    # or counted as the tag the prompt has.
    assert silent(ILLUSTRIOUS, '1girl, sign says "hi, safe, bye", masterpiece', "guide.rating")
    assert silent(ILLUSTRIOUS, "1girl, 封面上写着《hi, safe, bye》, masterpiece", "guide.rating")
    assert silent(KOHAKU, 'scenery, sign says "no, explicit, content"', "guide.rating")
    artist = '1girl, sign says "hi, nnn yryr, bye", smile'
    assert silent(ANIMA, artist, "guide.artist_prefix", {"nnn yryr": "artist"})
    rule = "guide.artist_prefix"
    assert silent(ANIMA, '1girl, "nnn yryr", smile', rule, {'"nnn yryr"': "artist"})
    finished = ", masterpiece, best quality, amazing quality"
    assert silent(WAI, '1girl, sign says "hi, general, bye"' + finished, "guide.rating_negative")
    composition = '1girl, upper body, sign says "hi, full body, bye", masterpiece'
    assert silent(ILLUSTRIOUS, composition, "guide.composition")
    assert silent(NEWBIE, '1girl, sign says "2girls", smile', "guide.xml")
    # Nor is a quality tag in it there: the finish is still missing.
    [tier] = only(run(ILLUSTRIOUS, '1girl, sign says "hi, masterpiece, bye"'), "guide.quality")
    assert "No quality tier" in tier.message
    sign = '1girl, sign says "masterpiece, high score, great score, absurdres"'
    [finish] = only(run(ANIMAGINE, sign), "guide.quality")
    assert finish.suggestion == sign + ", masterpiece, high score, great score, absurdres"
    # And the tag index is not asked about it (nor about 1girl, which
    # Anima's own Person count group places).
    assert prompt_rules.category_items(ANIMA, artist) == ["smile"]


def test_a_rule_that_adds_or_drops_an_item_keeps_the_lettering_as_typed():
    # The items after an inserted or dropped one shift into the gaps before
    # them; a comma with no space inside the lettering once took ", ".
    [pair] = only(run(KOHAKU, 'scenery, explicit, sign says "hi,there"'), "guide.rating")
    assert pair.suggestion == 'scenery, nsfw, explicit, sign says "hi,there"'
    [repeat] = only(run(KOHAKU, 'smile, smile, sign says "hi,there"'), "heuristic.repeated")
    assert repeat.suggestion == 'smile, sign says "hi,there"'


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
