"""Checkpoint-specific tag, artist, quality, rating, and XML rules."""
from __future__ import annotations

import re

import pytest

from backend import guides, prompt_rules

from tests.prompt_rules_helpers import (
    ANIMA,
    ANIMAGINE,
    FLUX_DEV,
    ILLUSTRIOUS,
    KOHAKU,
    NEWBIE,
    WAI,
    _all_guides,
    only,
    run,
    silent,
    spanned,
)


# ------------------------------------------------------------------ tag checkpoints' specifics


@pytest.mark.parametrize(
    "model_id, finish",
    [
        (ANIMAGINE, "masterpiece, high score, great score, absurdres"),
        (KOHAKU, "masterpiece"),
        (ILLUSTRIOUS, "masterpiece"),
        (WAI, "masterpiece, best quality, amazing quality"),
        (ANIMA, "masterpiece"),
    ],
)
@pytest.mark.parametrize("identity", ["", "original, "])
@pytest.mark.parametrize("lookup_available", [False, True])
def test_generic_and_original_characters_need_no_name_or_series(
    model_id, finish, identity, lookup_available,
):
    # The order describes positions for supplied tags, not mandatory slots.
    # Animagine's first two author examples are unnamed people; adding a
    # character or franchise would change the subject the user asked for.
    prompt = f"1girl, {identity}green hair, sweater, looking at viewer, {finish}"
    categories = {
        "original": "copyright",
        "green hair": "general",
        "sweater": "general",
        "looking at viewer": "general",
    } if lookup_available else None
    assert run(model_id, prompt, categories, negative_prompt="low quality") == []
    assert prompt_rules.reorder_tags(model_id, prompt, categories) is None


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


def test_anima_reads_artists_whatever_the_whole_prompt_reads_as():
    # The card's own shape - its prefix and tags, then sentences - reads as
    # prose; the rule reads each item, and once read none in it.
    prompt = (
        "masterpiece, best quality, score_7, safe, 1girl, wlop, A girl stands in the rain "
        "under a red umbrella. She looks up at the grey sky."
    )
    assert prompt_rules.detect_style(prompt) == "prose"
    [artist] = only(run(ANIMA, prompt, {"wlop": "artist"}), "guide.artist_prefix")
    assert spanned(prompt, artist) == "wlop"
    assert artist.suggestion == prompt.replace(", wlop,", ", @wlop,")
    # What a check asks the index: the tag-shaped items, never a clause.
    assert prompt_rules.lookup_items(ANIMA, prompt) == [
        "masterpiece", "best quality", "score_7", "safe", "1girl", "wlop"
    ]
    clauses = "A girl painted by wlop stands in the rain, smiling at the viewer, soft light"
    assert prompt_rules.lookup_items(ANIMA, clauses) == ["soft light"]
    # A sentence is no artist's name, whatever a lookup said of it.
    sentence = "She looks up at the grey sky."
    assert silent(ANIMA, f"1girl, {sentence}", "guide.artist_prefix", {sentence: "artist"})
    # Nor XML or JSON, which no lookup asks about.
    assert prompt_rules.lookup_items(ANIMA, '{"artist": "wlop"}') == []
    # Several are named together, each once.
    [both] = only(
        run(ANIMA, "1girl, wlop, kantoku, wlop", {"wlop": "artist", "kantoku": "artist"}),
        "guide.artist_prefix",
    )
    assert both.message.startswith("wlop and kantoku are artists in Danbooru's index")
    assert both.suggestion == "1girl, @wlop, @kantoku, @wlop"


def test_prose_is_looked_up_only_where_the_rules_read_artists():
    # Elsewhere a prose prompt's clauses are never asked about, which the
    # order's "partly checked" names; no guide that states an order reads
    # artists, so the two never meet (`lookup_items`).
    prose = "A girl with green hair stands in the rain, soft light, masterpiece"
    assert prompt_rules.detect_style(prose) == "prose"
    assert prompt_rules.lookup_items(ANIMAGINE, prose) == []
    # A tag list is, less what the guide's own groups place (1girl).
    assert prompt_rules.lookup_items(ANIMAGINE, "1girl, green hair, smile") == [
        "green hair", "smile"
    ]
    for name, guide in _all_guides().items():
        analysis = guide["analysis"]
        assert not ("artist_prefix" in analysis and "order" in analysis), name


def test_a_rewrite_is_covered_by_what_was_asked():
    asked, artists = ["1girl", "wlop", "smile"], ["wlop"]

    def read(text: str, asked: list[str] = asked) -> bool:
        return prompt_rules.artists_read(ANIMA, text, asked, artists)

    assert read("1girl, @wlop, smile")
    assert read("1girl, @ wlop, smile")
    # A whole item the rule reads, and flags: read.
    assert read("Smile, WLOP, 1girl")
    assert not read("1girl, kantoku, smile")
    sentences = "A girl stands in the rain. She looks up at the grey sky."
    assert read(sentences, [])
    # The artist kept with no @ and no finding: in a clause no check asks
    # about, or in a sentence the rule never reads as a name.
    assert not read("1girl, smile, a painting in the style of wlop with soft light")
    assert not read("A girl painted by wlop smiles in the rain.")
    assert not read("1girl, smile. A girl painted by WLOP smiles in the rain.", asked)
    # A clause no lookup asked about may itself be a name.
    assert not read("1girl, @wlop, smile, a girl standing in the rain")
    # Written as the tag autocomplete writes it, or with an underscore.
    escaped = ["1girl", "mika pikazo", "smile"]
    for text in ("1girl, smile. Drawn by mika_pikazo.", "1girl. Drawn by Mika Pikazo."):
        assert not prompt_rules.artists_read(ANIMA, text, escaped, ["mika pikazo"])
    assert prompt_rules.artists_read(ANIMA, "1girl, @mika_pikazo, smile", escaped, ["mika pikazo"])
    # Part of a longer word is not the name.
    assert read("1girl, smile. A wlopish glow fills the room.", asked)


def test_the_scene_block_ahead_of_a_character_is_against_newbies_order():
    example = guides.NEWBIE_GUIDE["examples"][0]
    blocks = re.findall(r"<(\w+)>.*?</\1>", example, re.S)
    assert blocks == ["character_1", "character_2", "general_tags"]
    assert prompt_rules.scene_ahead(NEWBIE, example) is None
    first, second, scene = [m.group() for m in re.finditer(r"<(\w+)>.*?</\1>", example, re.S)]
    assert prompt_rules.scene_ahead(NEWBIE, "\n\n".join([scene, first, second])) == "general_tags"
    assert prompt_rules.scene_ahead(NEWBIE, "\n\n".join([first, scene, second])) == "general_tags"
    assert prompt_rules.scene_ahead(NEWBIE, "\n\n".join([second, first, scene])) is None
    # No order stated, no scene block: nothing to hold it to.
    assert prompt_rules.scene_ahead(FLUX_DEV, "\n\n".join([scene, first])) is None


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
