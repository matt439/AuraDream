"""Prose rules, negative prompts, references, and checkpoint instructions."""
from __future__ import annotations

from pathlib import Path
import re

import pytest

from backend import guides, prompt_rules
from backend.backends.netayume_loading import SYSTEM_PROMPT
from backend.model_defaults import _profile_for, guide_for
from backend.schemas import GenerateRequest

from tests.prompt_rules_helpers import (
    ANIMAGINE,
    ERNIE,
    FLUX_DEV,
    ILLUSTRIOUS,
    KLEIN,
    KOHAKU,
    KREA2,
    KREA_DEV,
    NETAYUME,
    NEWBIE,
    QWEN,
    SCHNELL,
    Z_ANIME,
    _all_guides,
    _catalogue_repos,
    only,
    run,
    silent,
    spanned,
)


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
        # mid-clause is a name, after an abbreviation that goes before a
        # name, spaced or not.
        (KREA_DEV, "A cat trying to avoid the rain under a porch"),
        (KREA_DEV, "A cat, trying to avoid the rain"),
        (KREA_DEV, "A skier avoiding trees on a steep slope"),
        (KREA_DEV, "The film Dr. No playing on a TV"),
        (KREA_DEV, "The film Dr.No playing on a TV"),
        (KREA_DEV, "A poster for No Country for Old Men on a wall"),
        (KREA_DEV, "A No Smoking sign on a wall"),
        # The price of "Dr. No": "Dr." for a street reads as the title too,
        # so the absence after it is silent.
        (KREA_DEV, "A house on Elm Dr. No cars."),
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
        # Only an abbreviation that goes before a name ("Dr. No") keeps it
        # inside a clause; one that comes after a name ends a sentence as
        # often, and the absence after it was silent.
        ("A photo taken on Main St. No cars.", ["No cars"]),
        ("Martin Luther King Jr. No crowds.", ["No crowds"]),
        ("A portrait of John Smith Sr. No background.", ["No background"]),
        # An acronym after "No" is no title.
        ("Photorealistic portrait. No CGI look, no plastic skin.", ["No CGI look", "no plastic skin"]),
        ("A living room. No TV in the room.", ["No TV"]),
        # Nor is a capitalised word: a clause's own "No" is capitalised by
        # grammar, so what follows it says nothing about a name.
        ("A vintage travel poster of Kyoto. No English text.", ["No English text"]),
        ("A street market at dusk. No Chinese characters on the signs.", ["No Chinese characters"]),
        ("A quiet street. No Christmas decorations.", ["No Christmas decorations"]),
        ("A product photo. No Apple logo.", ["No Apple logo"]),
        ("No People on the beach", ["No People"]),
        # The price: a title that opens a clause reads as an absence, whole.
        ("No Country for Old Men poster on a wall", ["No Country for Old Men"]),
        # A name is not cut at the window's end.
        ("A climber without ropes on El Capitan at dawn", ["without ropes on El Capitan"]),
        # Nor at a lowercase particle between two of its capitalised words,
        # nor at the full stop of an abbreviation that opens it, spaced or not.
        ("A beach without tourists in Rio de Janeiro at dusk", ["without tourists in Rio de Janeiro"]),
        ("A portrait, no posters of Ludwig van Beethoven", ["no posters of Ludwig van Beethoven"]),
        ("A harbour at noon, no boats in St. Ives today", ["no boats in St. Ives"]),
        ("A waiting room. Avoid Dr. Smith in the frame.", ["Avoid Dr. Smith"]),
        ("A waiting room. Avoid Dr.Smith in the frame.", ["Avoid Dr.Smith"]),
        ("A lake, no crowds near Mt. Fuji", ["no crowds near Mt. Fuji"]),
        # After a capitalised word the abbreviation is a street, whose full
        # stop ends a sentence as often: the span stops there.
        ("A road with no cars on Elm Dr. Lovely day.", ["no cars on Elm Dr"]),
        ("A road with no cars on Main St. Lovely day.", ["no cars on Main St"]),
        ("A harbour, no people in St. Ives. Sunny.", ["no people in St. Ives"]),
        # Nor into a word that opens an absence of its own; the film's "No"
        # after a title is none.
        ("A harbour, no boats in St. No cars.", ["no boats in St", "No cars"]),
        ("A waiting room. Avoid Dr. No in the frame.", ["Avoid Dr. No"]),
    ],
)
def test_negation_near_misses_are_still_negations(prompt, phrases):
    assert [spanned(prompt, f) for f in only(run(KREA_DEV, prompt), "guide.negation")] == phrases


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


@pytest.mark.parametrize(
    "prompt",
    [
        "小狗，雪地，松树林，阴天，冷色调",
        "一只狗，没有项圈，不要文字",
        "puppy\nsnow\npine forest\novercast",
        "puppy, snow\npine forest, overcast",
    ],
)
def test_a_tag_list_is_one_however_its_items_are_parted(prompt):
    # The style is read from pieces split at commas, "，" and line breaks; the
    # note once counted the chips' comma items alone, so a Chinese list or a
    # one-per-line one read as "a tag list" with nothing to say about it.
    assert prompt_rules.detect_style(prompt) == "tags"
    for model_id in (ERNIE, FLUX_DEV):
        assert only(run(model_id, prompt), "guide.style"), (model_id, prompt)


def test_lettering_is_one_piece_to_the_style_rule():
    # A quotation's commas part no piece: two items are a phrase with a
    # comma, not yet a tag list, as for the chips.
    for prompt in ('1girl, "hi, there"', '1girl，"hi, there"', '1girl\n"hi, there"'):
        assert prompt_rules.detect_style(prompt) == "tags"
        assert only(run(ERNIE, prompt), "guide.style") == [], prompt
    assert only(run(ERNIE, '1girl, "hi, there", smile'), "guide.style")


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


def test_z_anime_recommends_a_negative_prompt():
    # Its guide: "Use a negative prompt for unwanted features; Base supports
    # it fully." The card names no text to use, so nothing to Apply.
    prompt = "A girl with silver hair stands under cherry blossoms. Soft light falls on her."
    [finding] = only(run(Z_ANIME, prompt, guidance_scale=4), "guide.negative_prompt")
    assert finding.field == "negative_prompt" and finding.severity == "info"
    assert finding.suggestion is None and finding.message.startswith("No negative prompt runs")
    assert silent(Z_ANIME, prompt, "guide.negative_prompt", negative_prompt="blurry", guidance_scale=4)
    # At CFG 0 a typed one does not run, and the note says so.
    [idle] = only(run(Z_ANIME, prompt, negative_prompt="blurry", guidance_scale=0), "guide.negative_prompt")
    assert "does not run" in idle.message
    # The undistilled Z-Image's card calls one optional: no note there.
    assert silent("Tongyi-MAI/Z-Image", prompt, "guide.negative_prompt", guidance_scale=4)


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


def test_an_edit_with_no_reference_attached_is_noted_as_the_apps_reading():
    # BFL's own "Turn this cat into a dog" renders from the words alone with
    # no reference: a plain text-to-image that looks like a success.
    [finding] = only(run(KLEIN, "Turn this cat into a dog"), "heuristic.no_reference")
    assert finding.severity == "info" and finding.source == "heuristic"
    assert finding.cite is None and finding.url is None and finding.suggestion is None
    assert "no reference image is attached" in finding.message
    for prompt in ("Make the photo black and white", "replace the sky in image 1 with stars"):
        assert only(run(KLEIN, prompt), "heuristic.no_reference"), prompt
    # Attached, or not an edit, or a checkpoint that takes no references.
    assert silent(KLEIN, "Turn this cat into a dog", "heuristic.no_reference", reference=[1])
    assert silent(KLEIN, "A cat holding a sign that says hello world", "heuristic.no_reference")
    assert silent(KLEIN, "Put the jacket on the man", "heuristic.no_reference")
    assert silent(FLUX_DEV, "Turn this cat into a dog", "heuristic.no_reference")


def test_the_examples_menu_reads_edits_as_the_rule_does():
    # web/prompt-examples.js marks an author's example "needs a reference image" by the
    # same pattern: one copy to drift would have the menu and Check disagree.
    source = (Path(__file__).resolve().parents[1] / "web" / "prompt-examples.js").read_text(encoding="utf-8")
    match = re.search(r"^const EDIT_PROMPT = /(.+)/i;$", source, re.M)
    assert match and match.group(1) == prompt_rules.EDIT_PROMPT.pattern
    assert prompt_rules.EDIT_PROMPT.flags & re.IGNORECASE


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


@pytest.mark.parametrize(
    "prompt",
    [
        'A sign reading "<Prompt Start>".',
        f'A poster saying "{SYSTEM_PROMPT}".',
        "A sign reading '<Prompt Start>'.",
        "A sign says <Prompt Start>.",
    ],
)
def test_netayume_system_prefix_inside_lettering_is_left_alone(prompt):
    assert only(run(NETAYUME, prompt), "guide.system_prefix") == []


# diffusers' own Lumina 2 instruction, and the anime one as a person retypes it.
_LUMINA = (
    "You are an assistant designed to generate superior images with the superior degree of "
    "image-text alignment based on textual prompts or user prompts."
)


_NETA_LIKE = (
    "You are an assistant designed to generate high-quality anime images with the highest "
    "degree of image-text alignment based on textual prompts."
)


@pytest.mark.parametrize(
    "prompt, spans, suggestion",
    [
        # The loader's own words without their full stop, or spaced oddly.
        (f"{SYSTEM_PROMPT[:-1]} <Prompt Start> 1girl, solo, moon",
         [SYSTEM_PROMPT[:-1], "<Prompt Start>"], "1girl, solo, moon"),
        (SYSTEM_PROMPT.replace(" ", "  \n", 2).lower() + " 1girl, solo",
         [SYSTEM_PROMPT.replace(" ", "  \n", 2).lower()], "1girl, solo"),
        # Another workflow's instruction ahead of the marker: one span, whole.
        (f"{_LUMINA} <Prompt Start> 1girl, solo", [f"{_LUMINA} <Prompt Start>"], "1girl, solo"),
        (f"  {_NETA_LIKE}\n<Prompt Start> 1girl, solo",
         [f"{_NETA_LIKE}\n<Prompt Start>"], "1girl, solo"),
        # A marker after tags is cut alone.
        ("1girl, solo <Prompt Start> moon", ["<Prompt Start>"], "1girl, solo moon"),
        ("1girl <Prompt Start> moon", ["<Prompt Start>"], "1girl moon"),
        # Lettering before the marker is the user's: the marker alone goes.
        ('You are "my sunshine" <Prompt Start> 1girl', ["<Prompt Start>"],
         'You are "my sunshine" 1girl'),
        # So is a scene that opens "You are", which no instruction is.
        ("You are looking at a girl standing in the rain <Prompt Start> 1girl, moon",
         ["<Prompt Start>"], "You are looking at a girl standing in the rain 1girl, moon"),
        ("You are an assistant <Prompt Start> 1girl", ["You are an assistant <Prompt Start>"],
         "1girl"),
        ("you are a helpful AI <Prompt Start> 1girl", ["you are a helpful AI <Prompt Start>"],
         "1girl"),
        ("You are an AI assistant that generates anime images. <Prompt Start> 1girl",
         ["You are an AI assistant that generates anime images. <Prompt Start>"], "1girl"),
        ("You are an expert image generation model capable of anything <Prompt Start> 1girl",
         ["You are an expert image generation model capable of anything <Prompt Start>"],
         "1girl"),
        # Only a noun that ends the opener makes one: "a model student" and
        # "an AI artist" are the scene.
        ("You are a model student walking in the rain <Prompt Start> 1girl",
         ["<Prompt Start>"], "You are a model student walking in the rain 1girl"),
        ("You are an AI artist painting a girl in the rain <Prompt Start> 1girl",
         ["<Prompt Start>"], "You are an AI artist painting a girl in the rain 1girl"),
        # More than a sentence ahead of the marker: nothing says how much of
        # it is an instruction, so the marker alone goes.
        ("You are an AI. A girl stands in the rain with a red umbrella. <Prompt Start> 1girl",
         ["<Prompt Start>"],
         "You are an AI. A girl stands in the rain with a red umbrella. 1girl"),
    ],
)
def test_netayume_system_prefix_is_found_as_retyped(prompt, spans, suggestion):
    findings = only(run(NETAYUME, prompt), "guide.system_prefix")
    assert [spanned(prompt, finding) for finding in findings] == spans
    assert all(finding.suggestion == suggestion for finding in findings)
    # What Apply leaves is clean: no instruction kept as description.
    assert only(run(NETAYUME, suggestion), "guide.system_prefix") == []


def test_another_instruction_is_named_for_what_it_is():
    prompt = f"{_LUMINA} <Prompt Start> 1girl, solo"
    [finding] = only(run(NETAYUME, prompt), "guide.system_prefix")
    assert "is a system instruction and its <Prompt Start> marker" in finding.message
    assert "already sent" not in finding.message


def test_netayume_quoted_prefix_does_not_hide_an_actual_duplicate():
    lettering = f'A poster says "{SYSTEM_PROMPT}" and a card reads "<Prompt Start>".'
    prompt = f"{lettering} {SYSTEM_PROMPT} <Prompt Start> 1girl"
    findings = only(run(NETAYUME, prompt), "guide.system_prefix")
    assert [spanned(prompt, finding) for finding in findings] == [SYSTEM_PROMPT, "<Prompt Start>"]
    assert all(finding.start >= len(lettering) for finding in findings)
    assert all(finding.suggestion == f"{lettering} 1girl" for finding in findings)
    assert only(run(NETAYUME, findings[0].suggestion), "guide.system_prefix") == []


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


def test_newbie_anonymous_character_labels_are_ready_to_use():
    example = guides.NEWBIE_GUIDE["examples"][0]
    assert silent(NEWBIE, example, "author.placeholder")
    assert guide_for(NEWBIE)["slots"] == []
    filled = example.replace("$character_1$", "miku").replace("$character_2$", "teto")
    assert silent(NEWBIE, filled, "author.placeholder")
    assert silent(NEWBIE, "<character_3><n>$character_3$</n></character_3>", "author.placeholder")


def test_placeholders_are_the_authors_slots():
    [character] = only(run(Z_ANIME, "Portrait of [character], soft light"), "author.placeholder")
    assert "put the character's name or description there" in character.message
    # The card's landscape template's four slots, each cited from it.
    template = next(p for p in guides.Z_ANIME_GUIDE["examples"] if "[location]" in p)
    slots = only(run(Z_ANIME, template), "author.placeholder")
    assert [spanned(template, f) for f in slots] == [
        "[location]", "[time of day]", "[lighting]", "[atmosphere]"
    ]
    assert all(f.cite == "Anime [location] at [time of day], [lighting], [atmosphere]" for f in slots)
    assert all("put what it stands for there" in f.message for f in slots)


@pytest.mark.parametrize("name", sorted(_all_guides()))
def test_each_slot_cites_the_authors_own_words(name):
    # A slot's finding quotes a line of the examples holding that slot: the
    # per-slot cite is held to the examples as the entry's own is.
    guide = _all_guides()[name]
    if not guide["analysis"].get("placeholders"):
        return
    model_id = next(repo for repo in _catalogue_repos() if _profile_for(repo)["guide"] is guide)
    for example in guide["examples"]:
        request = GenerateRequest(prompt=example)
        for finding in only(prompt_rules.check(model_id, request, request), "author.placeholder"):
            assert any(finding.cite in other for other in guide["examples"]), (name, finding.cite)
            assert spanned(example, finding).casefold() in finding.cite.casefold()


def test_the_page_reads_each_slot_pattern_as_the_rules_do():
    # `guide_for` sends the placeholder patterns for the Examples menu's
    # RegExp: only syntax both engines read alike - plain text, escapes,
    # classes, non-capturing groups and alternation - with no inline flags,
    # lookbehind or named groups, case left to each side's flag.
    seen = 0
    for repo in _catalogue_repos():
        analysis = _profile_for(repo)["guide"]["analysis"]
        patterns = [entry["pattern"] for entry in analysis.get("placeholders", [])]
        assert guide_for(repo)["slots"] == patterns, repo
        for pattern in patterns:
            seen += 1
            assert all(pattern[i + 2] == ":" for i in range(len(pattern)) if pattern.startswith("(?", i))
            assert not re.search(r"\\[AZbBz]|\(\?P", pattern), pattern
    assert seen
    assert guide_for(FLUX_DEV)["slots"] == []
