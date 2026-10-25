"""Checkpoint-specific tag, artist, quality, rating, and XML rules."""
from __future__ import annotations

import re

import pytest

from backend import guides, prompt_order, prompt_rules, prompt_text

from tests.prompt_rules_helpers import (
    ANIMA,
    ANIMA_AESTHETIC,
    ANIMAGINE,
    FLUX_DEV,
    ILLUSTRIOUS,
    KOHAKU,
    NEWBIE,
    WAI,
    _all_guides,
    _catalogue_repos,
    only,
    run,
    silent,
    spanned,
)


# ------------------------------------------------------------------ tag checkpoints' specifics


@pytest.mark.parametrize(
    "model_id, lead, finish",
    [
        (ANIMAGINE, "", ", masterpiece, high score, great score, absurdres"),
        (KOHAKU, "", ", masterpiece"),
        (ILLUSTRIOUS, "", ", masterpiece"),
        (WAI, "", ", masterpiece, best quality, amazing quality"),
        # Anima's card puts its quality tags first.
        (ANIMA, "masterpiece, ", ""),
    ],
)
@pytest.mark.parametrize("identity", ["", "original, "])
@pytest.mark.parametrize("lookup_available", [False, True])
def test_generic_and_original_characters_need_no_name_or_series(
    model_id, lead, finish, identity, lookup_available,
):
    # The order describes positions for supplied tags, not mandatory slots.
    # Animagine's first two author examples are unnamed people; adding a
    # character or franchise would change the subject the user asked for.
    prompt = f"{lead}1girl, {identity}green hair, sweater, looking at viewer{finish}"
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
    # A recipe alone still needs a subject; the Examples menu supplies scenes.
    assert only(run(WAI, "masterpiece, best quality, amazing quality"), "guide.subject")
    assert all(silent(WAI, prompt, "guide.subject") for prompt in guides.WAI_GUIDE["examples"])
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
        assert only(run(ILLUSTRIOUS, f"scenery, {accepted}, masterpiece"), "guide.rating") == []


def test_kohaku_explicit_and_underscores():
    [pair] = only(run(KOHAKU, "scenery, explicit"), "guide.rating")
    assert pair.suggestion == "scenery, nsfw, explicit"
    assert only(run(KOHAKU, "scenery, nsfw, explicit"), "guide.rating") == []
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
    # What a check asks the index: what its Tags menu does not place, never
    # a sentence - but a clause a name could be, which the order reads too.
    assert prompt_rules.lookup_items(ANIMA, prompt) == ["wlop"]
    clauses = "A girl painted by wlop stands in the rain, smiling at the viewer, soft light"
    assert prompt_rules.lookup_items(ANIMA, clauses) == [
        "A girl painted by wlop stands in the rain", "smiling at the viewer", "soft light"
    ]
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
    # order's "partly checked" names (`lookup_items`).
    prose = "A girl with green hair stands in the rain, soft light, masterpiece"
    assert prompt_rules.detect_style(prose) == "prose"
    assert prompt_rules.lookup_items(ANIMAGINE, prose) == []
    # A tag list is, less what the guide's own groups place (1girl).
    assert prompt_rules.lookup_items(ANIMAGINE, "1girl, green hair, smile") == [
        "green hair", "smile"
    ]
    # Where a guide both reads artists and places by category (Anima's), the
    # order places every clause the lookup answered nothing for as prose, so
    # a prose prompt's lookup must have asked about every one: the
    # categories the order places with are None, or cover every item a
    # category could place - never only the tag-shaped ones.
    for repo in _catalogue_repos():
        items = prompt_rules.category_items(repo, prose)
        asked = prompt_rules.lookup_items(repo, prose)
        if prompt_rules.wants_artist_categories(repo) and prompt_order.places_by_category(repo):
            assert asked == items and "stands in the rain" in " ".join(asked), repo
        elif prompt_order.places_by_category(repo):
            assert asked == [], repo
        else:
            # The artist rule alone reads only whole tag-shaped items.
            assert all(not prompt_text._is_prose(item) for item in asked), repo
    assert prompt_rules.lookup_items(ANIMA, prose) == [
        "A girl with green hair stands in the rain", "soft light"
    ]


ANIMA_CARD_EXAMPLE = (
    "year 2025, newest, normal quality, score_5, highres, safe, 1girl, oomuro sakurako, "
    "yuru yuri, @nnn yryr, smile, brown hair, hat, solo, fur-trimmed gloves, open mouth, "
    "long hair, gift box, fang, skirt, red gloves, blunt bangs, gloves, one eye closed, "
    "shirt, brown eyes, santa costume, red hat, skin fang, twitter username, white "
    "background, holding bag, fur trim, simple background, brown skirt, bag, gift bag, "
    "looking at viewer, santa hat, ;d, red shirt, box, gift, fur-trimmed headwear, "
    "holding, red capelet, holding box, capelet"
)


@pytest.mark.parametrize("model_id", [ANIMA, ANIMA_AESTHETIC])
def test_anima_cards_full_tag_example_passes_check_clean(model_id):
    # The card's "Full tag example", verbatim, against its own order: the
    # year, period, quality, score, meta and safety tags, 1girl, the
    # character, the series, the @ artist, then the general tags - among them
    # twitter username, a Danbooru meta tag, which is why the index's meta
    # category goes with the general tags.
    asked = prompt_rules.lookup_items(model_id, ANIMA_CARD_EXAMPLE)
    assert "@nnn yryr" not in asked and "safe" not in asked and "year 2025" not in asked
    categories = dict.fromkeys(asked, "general") | {
        "oomuro sakurako": "character", "yuru yuri": "copyright", "twitter username": "meta",
    }
    # score_5 is no tag of Aesthetic's menu, but the order places it still.
    assert "score_5" not in asked
    findings = run(model_id, ANIMA_CARD_EXAMPLE, categories)
    assert prompt_rules.places_every_item(model_id, ANIMA_CARD_EXAMPLE, categories)
    assert findings == []
    assert prompt_rules.reorder_tags(model_id, ANIMA_CARD_EXAMPLE, categories) is None
    # Offline the order is still the card's as far as the menu places it.
    assert silent(model_id, ANIMA_CARD_EXAMPLE, "guide.order")


def test_anima_checks_its_cards_order_with_sentences_anywhere():
    categories = {"wlop": "artist", "smile": "general", "hatsune miku": "character"}
    # A tag out of the card's order is flagged, and moved.
    prompt = "1girl, masterpiece, smile"
    [order] = only(run(ANIMA, prompt, categories), "guide.order")
    assert order.message.startswith('"1girl" (Person count) comes before "masterpiece"')
    assert order.suggestion == "masterpiece, 1girl, smile"
    # An @ name is an artist by its spelling: the Artist slot, before the
    # general tags and after the character, never asked about.
    prompt = "masterpiece, 1girl, smile, @wlop"
    [order] = only(run(ANIMA, prompt, categories), "guide.order")
    assert order.suggestion == "masterpiece, 1girl, @wlop, smile"
    assert silent(ANIMA, "masterpiece, 1girl, hatsune miku, @wlop, smile", "guide.order", categories)
    assert prompt_rules.order_slots(ANIMA, "@wlop", None) == [("@wlop", "Artist")]
    # Sentences stand anywhere ("You can mix tags and natural language in
    # arbitrary order"), before the tags, after them or between, and are
    # never "not placed"; the tags around them still keep the order.
    sentence = "A girl stands in the rain under a red umbrella. She looks up at the grey sky."
    for prompt in (
        f"{sentence}, masterpiece, 1girl, smile",
        f"masterpiece, 1girl, smile, {sentence}",
        f"masterpiece, {sentence}, 1girl, smile",
        f"masterpiece, best quality, @wlop. {sentence}",
    ):
        assert silent(ANIMA, prompt, "guide.order", categories), prompt
        assert prompt_rules.unplaced_items(ANIMA, prompt, categories) == [], prompt
        assert prompt_rules.reorder_tags(ANIMA, prompt, categories) is None, prompt
    # Moved around a sentence, which keeps its place.
    prompt = f"1girl, {sentence}, masterpiece"
    [order] = only(run(ANIMA, prompt, categories), "guide.order")
    assert order.suggestion == f"masterpiece, {sentence}, 1girl"
    assert prompt_order.holds_grid_variable(ANIMA, prompt)
    # Kohaku's sentence still takes its "general tags or prose" slot: the
    # switch is Anima's alone.
    assert not guides.KOHAKU_GUIDE["analysis"]["order"].get("free_sentences")
    kohaku = f"1girl, {sentence}, masterpiece"
    assert prompt_rules.order_slots(KOHAKU, kohaku, {})[1] == (sentence, "General tags / prose")
    assert silent(KOHAKU, kohaku, "guide.order", {})
    [late] = only(run(KOHAKU, f"masterpiece, {sentence}", {}), "guide.order")
    assert late.suggestion == f"{sentence}, masterpiece"
    assert not prompt_order.holds_grid_variable(KOHAKU, kohaku)
    switched = sorted(
        name for name, guide in _all_guides().items()
        if (guide["analysis"].get("order") or {}).get("free_sentences")
    )
    assert switched == ["ANIMA_AESTHETIC_GUIDE", "ANIMA_TURBO_GUIDE"]


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
    assert only(run(WAI, "scenery, explicit, masterpiece"), "guide.rating_negative") == []
