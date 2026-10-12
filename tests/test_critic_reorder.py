"""Critic reorders preserve whole pieces and the documented order."""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pytest

from backend import (
    critic_context,
    critic_reorder,
    critic_validation,
    guides,
    prompt_rules,
    prompt_weighting,
)
from backend.schemas import GenerateRequest, PromptAnalysis

from tests.critic_helpers import (
    KLEIN,
    _answer,
    _validate,
)


def test_reordered_is_kept_only_as_a_permutation_under_a_documented_order():
    prompt = "neon city at night, a woman with a red umbrella, shot on film"
    moved = "a woman with a red umbrella, neon city at night, shot on film"
    assert _validate(_answer(reordered=moved), prompt=prompt).reordered == moved
    # Case and punctuation are not words, so they do not make a reorder
    # something else; but the words kept are the prompt's own, moved - T5
    # and Qwen3 read case, and a row credited to order must differ in order.
    assert (
        _validate(_answer(reordered=moved.upper() + "."), prompt=prompt).reordered == moved + "."
    )
    # Unquoted sign text included: the sign still says Open Late.
    sign = "Neon city at night, a sign that says Open Late, a woman in a Burberry trench coat"
    shouted = "A WOMAN IN A BURBERRY TRENCH COAT, NEON CITY AT NIGHT, A SIGN THAT SAYS OPEN LATE"
    assert _validate(_answer(reordered=shouted), prompt=sign).reordered == (
        "a woman in a Burberry trench coat, Neon city at night, a sign that says Open Late"
    )
    # What lies between the pieces is the critic's, space and punctuation
    # only: a sentence's "!" or "?" goes where the sentence went, and so does
    # the stop that closes the prompt.
    for prompt_, reordered, kept in (
        ("A diner at night. A neon sign says EAT! A waitress smiles.",
         "A NEON SIGN SAYS EAT! A DINER AT NIGHT. A WAITRESS SMILES.",
         "A neon sign says EAT! A diner at night. A waitress smiles."),
        ("Is it raining? A woman walks past a shop.",
         "a woman walks past a shop. is it raining?",
         "A woman walks past a shop. Is it raining?"),
        ("A neon city at night.  A woman walks past;\nshot on film",
         "Shot on film. A woman walks past. A neon city at night.",
         "shot on film. A woman walks past. A neon city at night."),
    ):
        answer = _answer(reordered=reordered, rewrite=None)
        assert _validate(answer, prompt=prompt_).reordered == kept, reordered
    # Nor is how a quotation mark or an apostrophe is typed: the version kept
    # has the prompt's own.
    curly = "neon city at night, a sign that says “OPEN”, Joe’s old truck"
    straight = "Joe's old truck, neon city at night, a sign that says \"OPEN\""
    assert _validate(_answer(reordered=straight, rewrite=None), prompt=curly).reordered == (
        "Joe’s old truck, neon city at night, a sign that says “OPEN”"
    )
    for bad in (
        moved + ", cinematic",  # a word added
        moved.replace("red ", ""),  # a word dropped
        moved.replace("woman", "girl"),  # a word changed
        prompt + ".",  # nothing moved
        "",
    ):
        assert _validate(_answer(reordered=bad), prompt=prompt).reordered is None, bad
    tags = "1girl, long_hair, smile"
    assert _validate(_answer(reordered="long hair, 1girl, smile"), prompt=tags).reordered is None
    # No documented order, no reordered variant, whatever the model says.
    assert _validate(_answer(reordered=moved), prompt=prompt, ordered=False).reordered is None


def _kept(reordered: str, prompt: str, style: str = "prose") -> bool:
    answer = _answer(reordered=reordered)
    return _validate(answer, prompt=prompt, style=style).reordered is not None


def test_a_reorder_moves_whole_phrases_and_keeps_rendered_text():
    city, woman, sign = "neon city at night", "a woman in a red dress and a blue hat", '"OPEN"'
    prompt = f"{city}, {woman}, sign that says {sign}"
    assert _kept(f"{woman}, {city}, sign that says {sign}", prompt)
    # Every word is still there, but the colours changed owner or the sign
    # lost its quotation marks: a different picture, not a different order.
    swapped = "a woman in a blue dress and a red hat"
    assert not _kept(f"{swapped}, {city}, sign that says OPEN", prompt)
    assert not _kept(f"{swapped}, {city}, sign that says {sign}", prompt)
    assert not _kept(f"{woman}, {city}, sign that says OPEN", prompt)
    assert not _kept(f'{woman}, {city}, sign that says "open"', prompt)
    # A tag list's items are its pieces.
    tags = "1girl, blue hair, red eyes"
    assert _kept("1girl, red eyes, blue hair", tags, "tags")
    assert not _kept("1girl, red hair, blue eyes", tags, "tags")
    # Sentences move whole; quoted text never splits, whatever it holds.
    sentences = "A neon city at night. A woman walks past."
    assert _kept("A woman walks past. A neon city at night.", sentences)
    quoted = 'sign that says "OPEN, LATE", a woman'
    assert _kept('a woman, sign that says "OPEN, LATE"', quoted)
    assert not _kept('LATE", a woman, sign that says "OPEN', quoted)
    # An apostrophe is not a quotation mark.
    possessive = "the animal's face, a dog's tail, 'OPEN' sign"
    assert _kept("a dog's tail, the animal's face, 'OPEN' sign", possessive)
    # Guillemets and „…“ are quotation marks as the rules read them: text in
    # them moves whole or not at all (a narrower copy of the pattern here once
    # let «Fresh bread, hot coffee» split across two pieces).
    for opening, closing in (("«", "»"), ("„", "“")):
        sign = f"a chalkboard sign reading {opening}Fresh bread, hot coffee{closing}"
        prompt = f"A bakery at dawn, {sign}, a cat on the step"
        assert _kept(f"{sign}, A bakery at dawn, a cat on the step", prompt)
        split = (f"A bakery at dawn, hot coffee{closing}, a chalkboard sign reading "
                 f"{opening}Fresh bread, a cat on the step")
        assert not _kept(split, prompt), opening
    assert critic_reorder.QUOTED is prompt_rules.QUOTED is prompt_weighting.QUOTED


@pytest.mark.parametrize("prompt, changed, moved", [
    (
        "Mr. Green in a red coat. Ms. Blue in a blue coat.",
        "Mr. Blue in a blue coat. Ms. Green in a red coat.",
        "Ms. Blue in a blue coat. Mr. Green in a red coat.",
    ),
    (
        "A painting by J. M. W. Turner, a stormy sea",
        "A painting by J. W. M. Turner, a stormy sea",
        "a stormy sea, A painting by J. M. W. Turner",
    ),
    (
        "A St. Bernard sitting, a Mt. Fuji painting",
        "A St. Fuji painting, a Mt. Bernard sitting",
        "a Mt. Fuji painting, A St. Bernard sitting",
    ),
    (
        "A portrait of Prof. Green, a sketch of Dr. Blue",
        "A portrait of Prof. Blue, a sketch of Dr. Green",
        "a sketch of Dr. Blue, A portrait of Prof. Green",
    ),
    (
        "招牌写着《你好, 世界》, a woman walking, cinematic lighting",
        "a woman walking, 世界》, 招牌写着《你好, cinematic lighting",
        "a woman walking, 招牌写着《你好, 世界》, cinematic lighting",
    ),
    (
        "The shop has a sign 招牌写着《你好, 世界》. A woman walks past.",
        "A woman walks past. 世界》. The shop has a sign 招牌写着《你好.",
        "A woman walks past. The shop has a sign 招牌写着《你好, 世界》.",
    ),
])
def test_prose_reorders_keep_names_and_unquoted_lettering_whole(prompt, changed, moved):
    # Actual prose/hybrid placement: unlike a tag checkpoint, klein has no
    # deterministic reorder to reject a permutation that splits a name or
    # lettering. The critic's pieces must protect them before that decision.
    request = GenerateRequest(prompt=prompt)
    analysis = PromptAnalysis(
        model_id=KLEIN, style=prompt_rules.detect_style(prompt),
        words=prompt_rules.word_count(prompt),
        order=prompt_rules.documented_order(KLEIN),
        order_checked=prompt_rules.order_checked(KLEIN, prompt),
        reordered=prompt_rules.reorder_tags(KLEIN, prompt, None),
    )
    assert analysis.style in ("prose", "hybrid") and not analysis.order_checked
    context = critic_context.build_context(KLEIN, request, request, analysis)

    def validate(candidate):
        return critic_validation.validate(
            _answer(parts={}, issues=[], rewrite=None, reordered=candidate),
            context, prompt, model="test", seconds=0,
        )

    rejected = validate(changed)
    assert rejected.reordered is None and rejected.reordered_kind is None
    assert rejected.set_aside == [
        "The critic's reordered version was left out: it changed words, not only "
        "their order; a reordering of your own can go under Compare."
    ]
    kept = validate(moved)
    assert kept.reordered == moved and kept.reordered_kind == "unchecked"
    assert kept.set_aside == []


def test_a_reorder_of_xml_moves_blocks_and_of_json_moves_keys():
    xml = guides.NEWBIE_GUIDE["examples"][0]
    first = xml[: xml.index("</character_1>") + len("</character_1>")]
    second = xml[xml.index("<character_2>") : xml.index("</character_2>") + len("</character_2>")]
    scene = xml[xml.index("<general_tags>") :]
    assert _kept(f"{scene}\n{first}\n  {second}", xml, "xml")
    # Kept as the prompt writes its blocks, whatever the critic did to the
    # lines inside them; the space between them is the critic's.
    answer = _answer(reordered=f"{scene}\n{first}\n  {second}".replace("\n<", "<"))
    assert _validate(answer, prompt=xml, style="xml").reordered == f"{scene}{first}\n  {second}"
    # A tag moved from one character to the other is not an order.
    leaked = xml.replace("blue_hair, long_hair", "long_hair")
    leaked = leaked.replace("pink_hair,", "pink_hair, blue_hair,")
    assert not _kept(leaked, xml, "xml")
    scene, subjects = '"scene": "a street"', '"subjects": [{"type": "woman", "coat": "red"}]'
    prompt = f"{{{scene}, {subjects}}}"
    moved = '{"subjects": [{"coat": "red", "type": "woman"}], "scene": "a street"}'
    assert _kept(moved, prompt, "json")
    assert not _kept(prompt.replace("red", "blue"), prompt, "json")
    assert not _kept(f"{{{scene},  {subjects}}}", prompt, "json")  # nothing moved
    # Python's 1 == 1.0 == True is no JSON equality: a literal written as
    # another type is a changed value, with a key moved or without.
    typed = '{"subject": "a red fox", "count": 1.0, "style": "photo", "sharp": true}'
    assert not _kept(typed.replace("1.0", "1").replace("true", "1"), typed, "json")
    retyped = '{"count": 1.0, "subject": "a red fox", "style": "photo", "sharp": 1}'
    assert not _kept(retyped, typed, "json")
    assert _kept(retyped.replace(": 1}", ": true}"), typed, "json")
    # Nor is a number written otherwise at the same value: the kept version
    # is the critic's text, and the encoder reads 0.5 where the user wrote
    # 0.50. A move alone keeps each number as written.
    weight = '{"subject": "a red fox", "weight": 0.50, "style": "photo", "sharp": true}'
    moved = '{"subject": "a red fox", "style": "photo", "weight": 0.50, "sharp": true}'
    assert _kept(moved, weight, "json")
    for respelt in ("0.5", "5e-1", "0.500"):
        assert not _kept(moved.replace("0.50", respelt), weight, "json"), respelt
    style_first = '{"style": "photo", "subject": "a red fox", "count": 1.0, "sharp": true}'
    assert _kept(style_first, typed, "json")
    for respelt in ("1.00", "1e0", "1E+0"):
        assert not _kept(style_first.replace("1.0", respelt), typed, "json"), respelt
    signed = '{"scene": "a street", "tilt": -0}'
    assert _kept('{"tilt": -0, "scene": "a street"}', signed, "json")
    assert not _kept('{"tilt": 0, "scene": "a street"}', signed, "json")
    pose = '{"subject": "a fox", "pose": {"ears": 0.25, "tail": false}}'
    assert _kept('{"pose": {"tail": false, "ears": 0.25}, "subject": "a fox"}', pose, "json")
    assert not _kept('{"pose": {"ears": 0.250, "tail": false}, "subject": "a fox"}', pose, "json")
    # Through validate: set aside as changed words, not offered verbatim.
    respelt = _validate(
        _answer(reordered=moved.replace("0.50", "0.5"), issues=[], rewrite=None, parts={}),
        prompt=weight, style="json",
    )
    assert respelt.reordered is None and respelt.set_aside == [
        "The critic's reordered version was left out: it changed words, not only their "
        "order; a reordering of your own can go under Compare."
    ]
    # A number is never a string that reads the same.
    assert not _kept('{"count": "1", "scene": "a street"}', '{"scene": "a street", "count": 1}',
                     "json")
    # A repeated key hides the value json.loads drops.
    assert critic_reorder._json_forms('{"a": 1, "a": 2}') is None
    assert critic_reorder._json_forms('{"a": {"b": 1, "b": 1}}') is None
    assert critic_reorder._json_forms(typed) is not None


_NEWBIE = "NewBie-AI/NewBie-image-Exp0.1"


_FILLED = (
    guides.NEWBIE_GUIDE["examples"][0]
    .replace("$character_1$", "miku").replace("$character_2$", "teto")
)


def _newbie_reorder(candidate: str, prompt: str = _FILLED):
    """`candidate` as the critic's reordered version of `prompt` - NewBie's
    filled example - through the real analysis and validate."""
    request = GenerateRequest(prompt=prompt)
    analysis = PromptAnalysis(
        model_id=_NEWBIE,
        style=prompt_rules.detect_style(prompt),
        words=prompt_rules.word_count(prompt),
        order=prompt_rules.documented_order(_NEWBIE),
        order_checked=prompt_rules.order_checked(_NEWBIE, prompt),
    )
    assert analysis.style == "xml" and analysis.order and not analysis.order_checked
    context = critic_context.build_context(_NEWBIE, request, request, analysis)
    return critic_validation.validate(
        _answer(parts={}, issues=[], rewrite=None, reordered=candidate),
        context, prompt, model="test", seconds=0,
    )


def test_newbies_scene_block_is_not_moved_ahead_of_its_characters():
    # Check reads no XML order, but the guide states one - each character's
    # block, then <general_tags> - and the critic is told to move pieces into
    # it: the other way round was kept and offered under Use reordered.
    first, second, scene = [
        match.group() for match in re.finditer(r"<(\w+)>.*?</\1>", _FILLED, re.S)
    ]
    swapped = _newbie_reorder(f"{second}\n\n{first}\n\n{scene}")
    assert swapped.reordered == f"{second}\n\n{first}\n\n{scene}"
    assert swapped.reordered_kind == "unchecked" and swapped.set_aside == []
    against = (
        "The critic's reordered version was left out: it puts <general_tags> ahead of a "
        "character block, against the documented order."
    )
    for candidate in (f"{scene}\n\n{first}\n\n{second}", f"{first}\n\n{scene}\n\n{second}"):
        result = _newbie_reorder(candidate)
        assert result.reordered is None and result.reordered_kind is None
        assert result.set_aside == [against]
    # The prompt's own scene-first order, the characters swapped: still no
    # reading of the documented order, but the critic only left it there.
    own = f"{scene}\n\n{first}\n\n{second}"
    result = _newbie_reorder(f"{scene}\n\n{second}\n\n{first}", own)
    assert result.reordered is None and result.set_aside == [
        "The critic's reordered version was left out: it keeps <general_tags> ahead of a "
        "character block, against the documented order."
    ]


def test_words_moved_inside_an_xml_block_are_not_called_changed():
    # Blocks move whole, so no attribute changes owner: a field or a tag
    # moved inside one is still refused, but no word changed, and the
    # sentence once said one had.
    def reason(candidate, prompt=_FILLED):
        result = _newbie_reorder(candidate, prompt)
        assert result.reordered is None
        (line,) = result.set_aside
        return line.removeprefix("The critic's reordered version was left out: ").removesuffix(
            "; a reordering of your own can go under Compare."
        )

    gender = "<gender>1girl</gender>\n"
    appearance = _FILLED[_FILLED.index("<appearance>"):_FILLED.index("</appearance>") + 13] + "\n"
    fields = _FILLED.replace(gender + appearance, appearance + gender, 1)
    assert fields != _FILLED
    inside = "it moved words inside an XML block, where only whole blocks move, so no attribute changes owner"
    assert reason(fields) == inside
    assert reason(_FILLED.replace("happy, smile", "smile, happy", 1)) == inside
    # One tag moved to the other character changed no word either.
    leaked = _FILLED.replace("blue_hair, long_hair", "long_hair").replace(
        "pink_hair,", "pink_hair, blue_hair,"
    )
    assert reason(leaked) == (
        "it moved words between XML blocks, where only whole blocks move, so no attribute "
        "changes owner"
    )
    # Text outside the blocks is compared piece by piece, as ever.
    loose = f"masterpiece, best quality\n{_FILLED}"
    assert reason(f"masterpiece, best quality\n{fields}", loose) == inside
    # The fake critic's reorder, its last comma item first, parts the blocks.
    items = [item.strip() for item in loose.split(",")]
    rotated = ", ".join(items[-1:] + items[:-1])
    assert reason(rotated, loose) == (
        "it moved words between XML blocks, where only whole blocks move, so no attribute "
        "changes owner"
    )
    # Words moved in the text outside the blocks, the blocks as they were.
    assert reason("masterpiece, quality best\n" + _FILLED, loose) == (
        "it changed the text outside the XML blocks, not only the order of whole pieces"
    )
    # Every piece's words in their own order: punctuation or spacing changed,
    # and no word moved.
    spacing = "it changed punctuation or spacing, not only the order of whole blocks"
    assert "<n>miku</n>" in _FILLED
    for retyped in (
        _FILLED.replace("happy, smile", "happy smile", 1),
        _FILLED.replace("<n>miku</n>", "<n> miku </n>", 1),
        _FILLED.replace("happy, smile", "happy; smile", 1),
        f"masterpiece best quality\n{_FILLED}",
    ):
        assert retyped != _FILLED
        prompt = loose if retyped.startswith("masterpiece") else _FILLED
        assert reason(retyped, prompt) == spacing, retyped
    # A changed word is still called one.
    assert reason(_FILLED.replace("blue_hair", "green_hair")) == (
        "it changed words, not only their order"
    )
    assert reason(loose.replace("best quality", "top quality"), loose) == (
        "it changed words, not only their order"
    )


def test_the_sheet_finds_xml_blocks_as_the_critic_does():
    # web/sheet.js names a comparison's moved XML blocks by their tags
    # (blockMove), finding them with a copy of _XML_BLOCK - and web/ has no
    # test suite. JavaScript's \w is ASCII alone, so the copy writes Python's
    # Unicode \w out as letters, numbers and the underscore; [\s\S] is DOTALL's
    # dot. A block one side reads and the other does not would name a move
    # the critic never made.
    sheet = (Path(__file__).resolve().parents[1] / "web" / "sheet.js").read_text(encoding="utf-8")
    match = re.search(r"^const XML_BLOCK = /(.+)/gu;$", sheet, re.M)
    assert match
    written = match.group(1).replace(r"\p{L}\p{N}_", r"\w").replace(r"[\s\S]", ".").replace(r"<\/", "</")
    assert written == critic_reorder._XML_BLOCK.pattern
    assert critic_reorder._XML_BLOCK.flags & re.DOTALL
    # Python's \w is exactly the letters, the numbers and the underscore.
    word = re.compile(r"\w")
    for code in (*range(0x80, 0x3100), *range(0x4E00, 0x4F00), *range(0xAC00, 0xAD00)):
        char = chr(code)
        assert bool(word.fullmatch(char)) == (unicodedata.category(char)[0] in "LN" or char == "_"), hex(code)
    assert [m.group(1) for m in critic_reorder._XML_BLOCK.finditer("<人物>a</人物> <b>c</b>")] == ["b"]
    assert [m.group(1) for m in critic_reorder._XML_BLOCK.finditer("<a人物>a</a人物> <b>c</b>")] == ["a人物", "b"]


def test_a_reordered_version_that_changed_words_is_said_so():
    prompt = "neon city at night, a woman with a red umbrella, shot on film"
    words_changed = [
        "The critic's reordered version was left out: it changed words, not only their "
        "order; a reordering of your own can go under Compare."
    ]

    def reordered(text, prompt=prompt, style="prose"):
        answer = _answer(reordered=text, rewrite=None, parts={})
        return _validate(answer, prompt=prompt, style=style)

    changed = reordered("a woman with an umbrella, neon city at night, shot on film")
    assert changed.reordered is None and changed.set_aside == words_changed
    # Nothing moved is not a change of words, read as the reorder check
    # reads words - case, a closing stop, a break's spacing aside: nothing
    # to say.
    for echo in (prompt, prompt.upper(), prompt + ".", prompt.replace(", ", " ,  ")):
        unmoved = reordered(echo)
        assert unmoved.reordered is None and unmoved.set_aside == [], echo
    # Or how a quotation mark or an apostrophe is typed: the critic writes
    # straight ones back for the curly ones a phone put in.
    curly = "neon city at night, a sign that says “OPEN”, Joe’s old truck"
    unmoved = reordered("neon city at night, a sign that says \"OPEN\", Joe's old truck", curly)
    assert unmoved.reordered is None and unmoved.set_aside == []
    # JSON the same, key for key, in other spacing; and a value retyped.
    scene = '{"scene": "a street", "count": 1}'
    unmoved = reordered('{"scene":"a street","count":1}', scene, "json")
    assert unmoved.reordered is None and unmoved.set_aside == []
    retyped = reordered('{"count": 1.0, "scene": "a street"}', scene, "json")
    assert retyped.reordered is None and retyped.set_aside == words_changed


def test_where_check_placed_the_tags_the_critic_reorders_only_as_it_does():
    # Check places a tag checkpoint's comma items in the documented order
    # itself (`order_checked`), and its reorder is the documented order: a
    # critic's permutation other than that one - the whole list reversed,
    # which Check would flag - is some other order, left out and said so.
    animagine = "cagliostrolab/animagine-xl-4.0"
    firefly = r"firefly \(honkai: star rail\)"
    categories = {"green hair": "general", firefly: "character"}
    prompt = f"1girl, green hair, {firefly}, masterpiece, high score"

    def critique(prompt: str, reordered: str):
        request = GenerateRequest(prompt=prompt)
        # As `prompt_check.analyse` builds it: the categories the rules placed with
        # travel with the analysis, for the critic to place with the same.
        analysis = PromptAnalysis(
            model_id=animagine, style=prompt_rules.detect_style(prompt, categories), words=8,
            order=prompt_rules.documented_order(animagine),
            order_checked=prompt_rules.order_checked(animagine, prompt),
            categories=categories,
            reordered=prompt_rules.reorder_tags(animagine, prompt, categories),
        )
        assert analysis.order_checked
        context = critic_context.build_context(animagine, request, request, analysis)
        answer = _answer(issues=[], parts={}, rewrite=None, reordered=reordered)
        return critic_validation.validate(answer, context, prompt, model="test", seconds=0.0)

    other_order = [
        "The critic's reordered version was left out: Check places these tags in the "
        "documented order itself, and the critic's order is not that one."
    ]
    in_order = f"1girl, {firefly}, green hair, masterpiece, high score"
    reversed_ = ", ".join(reversed(prompt.split(", ")))
    same = critique(prompt, in_order.upper())
    assert same.reordered == in_order and same.set_aside == []
    assert same.reordered_kind == "check"
    # Check's own text, whatever the critic's punctuation: on a hybrid
    # prompt the sentences inside a comma item stay as the prompt set them.
    hybrid = "masterpiece, 1girl, a girl stands in the rain. she smiles softly, green hair"
    checks = "1girl, a girl stands in the rain. she smiles softly, green hair, masterpiece"
    assert prompt_rules.reorder_tags(animagine, hybrid, categories) == checks
    for answer in (checks, checks.upper().replace(". ", ", ") + "."):
        same = critique(hybrid, answer)
        assert same.reordered == checks and same.set_aside == [], answer
    other = critique(prompt, reversed_)
    assert other.reordered is None and other.set_aside == other_order
    assert other.reordered_kind is None
    # Already in order, nothing moves: any reorder is another order.
    moved = critique(in_order, ", ".join(reversed(in_order.split(", "))))
    assert moved.reordered is None and moved.set_aside == other_order
    # A prose prompt on the same checkpoint is one comma item Check placed,
    # and what the critic moved were its sentences: the note speaks of the
    # prompt, not of tags it does not have.
    rain = "A girl stands in the rain. She holds a red umbrella."
    swapped = critique(rain, "She holds a red umbrella. A girl stands in the rain.")
    assert swapped.reordered is None and swapped.set_aside == [
        "The critic's reordered version was left out: Check places this prompt in the "
        "documented order itself, and the critic's order is not that one."
    ]
    # Where Check places nothing - klein's prose - the critic's reading stands.
    klein = _validate(
        _answer(reordered=reversed_, rewrite=None, parts={}), prompt=prompt, style="tags"
    )
    assert klein.reordered == reversed_ and klein.reordered_kind == "unchecked"
    # Nothing moved, nothing kept, and no kind.
    still = _validate(_answer(reordered=prompt, rewrite=None, parts={}), prompt=prompt, style="tags")
    assert still.reordered is None and still.reordered_kind is None


def test_where_check_placed_the_tags_only_partly_its_order_rule_decides_the_critics():
    # A tag with no category keeps every slot one could give it - offline,
    # past the 64 tags one check may look up, or unknown to the tag index -
    # so where that slot decides the order, Check's own order is one the
    # documented order allows, not the only one. A reorder of the critic's
    # then stands where the order rule, placing with the same categories,
    # flags nothing in it - and is set aside, as before, once every tag is
    # placed, or none left unplaced decides anything.
    animagine = "cagliostrolab/animagine-xl-4.0"
    offline = "the tag index was unreachable, so no tag has its Danbooru category"
    prompt = "1girl, firefly, green hair, masterpiece"
    placed = {"firefly": "character", "green hair": "general"}

    def critique(reordered, categories=None, detail=offline, prompt=prompt):
        request = GenerateRequest(prompt=prompt)
        analysis = PromptAnalysis(
            model_id=animagine, style=prompt_rules.detect_style(prompt, categories), words=6,
            order=prompt_rules.documented_order(animagine),
            order_checked=prompt_rules.order_checked(animagine, prompt),
            categories_detail=detail,
            categories=categories,
            reordered=prompt_rules.reorder_tags(animagine, prompt, categories),
        )
        assert analysis.order_checked and analysis.reordered is None
        context = critic_context.build_context(animagine, request, request, analysis)
        answer = _answer(issues=[], parts={}, rewrite=None, reordered=reordered)
        return critic_validation.validate(answer, context, prompt, model="test", seconds=0.0)

    # Neither firefly nor green hair placed: either may come first. The
    # server says which kind of reordering it kept, for the page to say.
    swapped = "1girl, green hair, firefly, masterpiece"
    kept = critique(swapped)
    assert kept.reordered == swapped and kept.set_aside == []
    assert kept.reordered_kind == "allowed"
    # In the prompt's own spelling, and past a closing full stop, which the
    # rules read as no other tag.
    kept = critique(swapped.upper() + ".")
    assert kept.reordered == swapped + "." and kept.set_aside == []
    # Green hair placed and firefly not. After green hair, firefly's slot
    # decides the order - out of it were firefly a character - so a swap
    # putting it first stands. Ahead of green hair it is in order whatever it
    # is: Check's order is then the documented one, not one it allows, and a
    # swap putting firefly after green hair is another order, set aside. The
    # same where every tag was looked up and the index knew no firefly - an
    # OC's name, a trigger word: no lookup failed (`categories_detail` None),
    # and the tag is placed no better than one never asked about.
    another = [
        "The critic's reordered version was left out: Check places these tags in the "
        "documented order itself, and the critic's order is not that one."
    ]
    for detail in (offline, None):
        known = {"green hair": "general"}
        kept = critique(prompt, categories=known, detail=detail, prompt=swapped)
        assert (kept.reordered, kept.reordered_kind, kept.set_aside) == (prompt, "allowed", [])
        other = critique(swapped, categories=known, detail=detail)
        assert (other.reordered, other.reordered_kind, other.set_aside) == (None, None, another)
    kept = critique(swapped, categories={}, detail=None)
    assert (kept.reordered, kept.reordered_kind, kept.set_aside) == (swapped, "allowed", [])
    # Lettering with a comma no quotation mark holds is one item to the
    # rules and the critic alike: a reorder moving the title whole stands,
    # one parting it at the comma fails the critic's pieces before the
    # deterministic order check. The sign is a clause beside the tags, so
    # the prompt reads as hybrid.
    title = "1girl, 招牌写着《你好, 世界》, green hair, masterpiece"
    assert prompt_rules.detect_style(title) == "hybrid"
    whole = "1girl, green hair, 招牌写着《你好, 世界》, masterpiece"
    kept = critique(whole, prompt=title)
    assert (kept.reordered, kept.set_aside) == (whole, [])
    for parted in (
        "1girl, 世界》, green hair, 招牌写着《你好, masterpiece",
        "1girl, 世界》, 招牌写着《你好, green hair, masterpiece",
    ):
        other = critique(parted, prompt=title)
        assert other.reordered is None and other.reordered_kind is None, parted
        assert other.set_aside == [
            "The critic's reordered version was left out: it changed words, not only "
            "their order; a reordering of your own can go under Compare."
        ], parted
    # What the rule flags stays out, and the note says what it found.
    flagged = critique("masterpiece, 1girl, firefly, green hair")
    assert flagged.reordered is None
    chain = "the documented order is Person count → Character → Series → Rating → Details → Quality."
    assert flagged.set_aside == [
        "The critic's reordered version was left out: in it, \"masterpiece\" (Quality) comes "
        f"before \"1girl\" (Person count); {chain}"
    ]
    # With the categories Check had: a tag it placed keeps its slot, though
    # another went unplaced past the lookup's cap.
    capped = (
        "the tag index was asked about the first 2 of 3 tags only, so the rest have no "
        "Danbooru category"
    )
    held = critique(
        "1girl, green hair, firefly, mystery, masterpiece", categories=placed, detail=capped,
        prompt="1girl, firefly, green hair, mystery, masterpiece",
    )
    assert held.reordered is None and held.set_aside == [
        "The critic's reordered version was left out: in it, \"green hair\" (Details) comes "
        f"before \"firefly\" (Character); {chain}"
    ]
    # The critic's pieces are not always the rules' items: joined by "、" or
    # set on lines of their own, the four tags are one item to the rules,
    # which no order can be wrong about.
    regrouped = [
        "The critic's reordered version was left out: it does not keep the prompt's comma "
        "items whole, so Check would read it as other tags."
    ]
    for joined in ("1girl、green hair、firefly、masterpiece", "1girl\ngreen hair\nfirefly\nmasterpiece"):
        other = critique(joined)
        assert other.reordered is None and other.set_aside == regrouped, joined
    # The note names tags only where the prompt is tags: a prose prompt's
    # comma items are clauses (as `order_detail` names them), a hybrid's
    # both - once said to be tags on either.
    rain = "a girl standing in the rain. smiling softly, holding a red umbrella"
    for split in (
        "smiling softly, a girl standing in the rain, holding a red umbrella",
        "holding a red umbrella. a girl standing in the rain. smiling softly",
    ):
        other = critique(split, detail=None, prompt=rain)
        assert other.reordered is None and other.set_aside == [
            "The critic's reordered version was left out: it does not keep the prompt's comma "
            "items whole, so Check would read it as other clauses."
        ], split
    hybrid = "1girl, a girl stands in the rain. she smiles softly, green hair, masterpiece"
    assert prompt_rules.detect_style(hybrid) == "hybrid"
    other = critique(
        "1girl, she smiles softly, a girl stands in the rain, green hair, masterpiece",
        prompt=hybrid,
    )
    assert other.reordered is None and other.set_aside == [
        "The critic's reordered version was left out: it does not keep the prompt's comma "
        "items whole, so Check would read it as other tags and clauses."
    ]
    # The same items, read as XML once an opening tag leads: the rule stands
    # down, and an order it did not read is not one it passed.
    tagged = "1girl, <character_1>, green hair, masterpiece"
    unread = critique("<character_1>, 1girl, green hair, masterpiece", prompt=tagged)
    assert unread.reordered is None and unread.set_aside == [
        "The critic's reordered version was left out: Check does not read the order of it, "
        "so nothing has checked that order."
    ]
    # Lettering is placed by its shape and never looked up: two captions read
    # as prose, in the general slot, and with the rest in the guide's own
    # groups every item is placed with no categories at all - so a swap of
    # the two is some other order, as where every tag has its category. The
    # captions make it prose, and the note speaks of the prompt, not tags.
    signs = '1girl, a sign that says "open", a banner that reads "sale", masterpiece'
    other = critique(
        '1girl, a banner that reads "sale", a sign that says "open", masterpiece',
        detail=None, prompt=signs,
    )
    assert other.reordered is None and other.set_aside == [
        "The critic's reordered version was left out: Check places this prompt in the "
        "documented order itself, and the critic's order is not that one."
    ]
    # Every tag placed, the rule finds the same swap out of order - and
    # Check's order is the documented one: any other is set aside, as ever.
    # So too where some other item went unasked: what it placed decides.
    for detail in (None, capped):
        full = critique(swapped, categories=placed, detail=detail)
        assert full.reordered is None and full.reordered_kind is None
        assert full.set_aside == [
            "The critic's reordered version was left out: Check places these tags in the "
            "documented order itself, and the critic's order is not that one."
        ]


def test_a_grid_variable_is_where_the_user_put_it_and_the_critic_may_put_it_elsewhere():
    # `{hair}` takes every slot, so Check never flags it and never calls it
    # unplaced - but where it sits is the user's, not the documented order's:
    # Check's reorder leaves it there, and a critic's that leaves it beside
    # smile instead is another order the documented one allows, not one set
    # aside as "not that one".
    animagine = "cagliostrolab/animagine-xl-4.0"
    prompt = "smile, {hair}, 1girl, masterpiece"
    categories = {"smile": "general"}
    request = GenerateRequest(prompt=prompt)
    analysis = PromptAnalysis(
        model_id=animagine, style=prompt_rules.detect_style(prompt, categories), words=5,
        order=prompt_rules.documented_order(animagine),
        order_checked=prompt_rules.order_checked(animagine, prompt),
        categories=categories,
        reordered=prompt_rules.reorder_tags(animagine, prompt, categories),
    )
    assert analysis.reordered == "1girl, {hair}, smile, masterpiece"
    assert prompt_rules.places_every_item(animagine, prompt, categories)
    context = critic_context.build_context(animagine, request, request, analysis)
    assert context.placed_partly

    def critique(reordered):
        answer = _answer(issues=[], parts={}, rewrite=None, reordered=reordered)
        return critic_validation.validate(answer, context, prompt, model="test", seconds=0.0)

    beside = critique("1girl, smile, {hair}, masterpiece")
    assert (beside.reordered, beside.reordered_kind, beside.set_aside) == (
        "1girl, smile, {hair}, masterpiece", "allowed", []
    )
    ours = critique("1girl, {hair}, smile, masterpiece")
    assert (ours.reordered, ours.reordered_kind) == ("1girl, {hair}, smile, masterpiece", "check")
    # What the rule flags among the tags still stays out.
    flagged = critique("smile, {hair}, masterpiece, 1girl")
    assert flagged.reordered is None and flagged.set_aside
