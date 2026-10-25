"""Critic rewrites preserve rendered lettering and expose added and dropped content."""
from __future__ import annotations

from backend import critic_validation

from tests.critic_helpers import (
    ANIMAGINE,
    KLEIN,
    KREA_DEV,
    _answer,
    _validate,
)


def test_the_rewrites_quotation_marks_are_held_to_the_prompts_lettering():
    prompt = "A quiet street after the rain, a small shop with a wooden sign that says Open Late."

    def check(rewrite, prompt=prompt, model_id=KLEIN):
        return _validate(_answer(rewrite=rewrite, parts={}), prompt=prompt, model_id=model_id)

    # Wrapped whole: unwrapped, and said so.
    wrapped = check(f'"{prompt[:-1]}, empty and glistening."')
    assert wrapped.rewrite == f"{prompt[:-1]}, empty and glistening."
    assert wrapped.set_aside[0].startswith("The rewrite came back wrapped in quotation marks")
    # A stop after the closing mark still leaves a wrap, in straight or curly
    # marks, and the rewrite keeps it; a wrapped copy of the prompt is none.
    cat = "a cat on a sofa"
    wrap = ["The rewrite came back wrapped in quotation marks, which ask for lettering; "
            "they were taken off."]
    ginger = "A ginger cat curled up asleep on a worn velvet sofa"
    for rewrite in (f'"{ginger}".', f"“{ginger}”.", f'"{ginger}" .', f"“{ginger}” ."):
        result = check(rewrite, cat)
        assert result.rewrite == f"{ginger}.", rewrite
        assert result.set_aside == wrap and result.rewrite_findings == [], rewrite
    # Any sentence punctuation after it, spaced or not: a stop is kept,
    # closed up; a comma, semicolon or colon was the critic's sentence going
    # on, and goes. None of it is lettering the prompt does not ask for.
    for tail, kept in (("!", "!"), (" ?", "?"), (";", ""), (",", ""), (":", ""), (" ;", ""),
                       ("，", ""), ("。", "。")):
        for rewrite in (f'"{ginger}"{tail}', f"“{ginger}”{tail}"):
            result = check(rewrite, cat)
            assert result.rewrite == ginger + kept, rewrite
            assert result.set_aside == wrap and result.rewrite_findings == [], rewrite
    for copy in (f'"{cat}".', f"“{cat}”", f'"{cat}."', f'"{cat}" .', f'"{cat}";', f'"{cat}",'):
        result = check(copy, cat)
        assert result.rewrite is None and result.rewrite_findings == [], copy
        assert result.set_aside == wrap + [
            "The critic's rewrite was your prompt as it stands, so there is no rewrite to show."
        ], copy
    # Wrapped around a quotation of its own: the inner marks still pair up.
    inner = 'A quiet street after the rain, a wooden sign that says "Open Late"'
    assert check(f"“{inner}”").rewrite == inner
    # Two quotations are not one wrap, nor is an apostrophe a quotation.
    assert critic_validation._unwrapped('"Open Late" on the sign, "Closed" on the door') is None
    assert critic_validation._unwrapped('"Open Late" on the sign, "Closed".') is None
    assert critic_validation._unwrapped('"Open Late" on the sign, "Closed";') is None
    assert critic_validation._unwrapped('"A cat on a sofa." ,') == "A cat on a sofa."
    assert critic_validation._unwrapped("'Open' on the sign, 'Closed' on the door") is None
    assert critic_validation._unwrapped("'A dog's tail'") == "A dog's tail"
    assert critic_validation._unwrapped("'A dog's tail'!") == "A dog's tail!"
    assert critic_validation._unwrapped('"A sign that says "Open Late""') == 'A sign that says "Open Late"'
    assert critic_validation._unwrapped('"A cat on a sofa.".') == "A cat on a sofa."
    assert critic_validation._unwrapped("A plain rewrite.") is None
    # The sign's own words quoted: what the guide asks for.
    good = check(inner)
    assert good.rewrite == inner and good.rewrite_findings == [] and good.set_aside == []
    # Lettering the prompt does not have - copied from an authors' example -
    # drops the rewrite.
    invented = check(inner + ', a neon sign reading "Black Forest Labs"')
    assert invented.rewrite is None
    assert invented.set_aside == [
        'The rewrite was left out: it puts "Black Forest Labs" in quotation marks - '
        "lettering the prompt does not ask for."
    ]
    # So does lettering changed in Chinese, quoted with no space, as ERNIE's
    # own rewriter writes it: '营业中' becomes '欢迎光临'.
    shop = "一家小店门口，一块木招牌上写着'营业中'，傍晚的街道"
    changed = shop.replace("写着'营业中'", "写着'欢迎光临'")
    ernie = _validate(
        _answer(parts={"text_to_render": "营业中"}, rewrite=changed),
        prompt=shop, model_id="baidu/ERNIE-Image-Turbo", ordered=False,
    )
    assert ernie.parts["text_to_render"] == "营业中" and ernie.rewrite is None
    assert ernie.set_aside == [
        'The rewrite was left out: it puts "欢迎光临" in quotation marks - '
        "lettering the prompt does not ask for."
    ]
    # The prompt's own description in quotes is flagged where it sits.
    described = 'A "quiet street after the rain", a wooden sign that says "Open Late"'
    result = check(described)
    assert result.rewrite == described
    (flag,) = result.rewrite_findings
    assert flag.rule == "heuristic.rewrite_quote" and flag.severity == "warn"
    assert flag.source == "heuristic" and flag.cite is None
    assert described[flag.start:flag.end] == '"quiet street after the rain"'
    # And so is description past lettering the rules ended for certain: the
    # sign says OPEN, and the rest is where and how.
    diner = "A diner at night, a neon sign that says OPEN in red letters above the door"
    run_on = 'A diner at night; its neon sign reads "OPEN in red letters above the door".'
    result = check(run_on, diner, "black-forest-labs/FLUX.1-Krea-dev")
    assert result.rewrite == run_on
    (flag,) = result.rewrite_findings
    assert flag.rule == "heuristic.rewrite_quote"
    assert run_on[flag.start:flag.end] == '"OPEN in red letters above the door"'
    # A name in capitals goes on past the "On" the rules stopped at.
    cafe = "A cafe front at dusk, a sign that says Cafe On Main"
    titled = check('A cafe front at dusk, its sign reading "Cafe On Main".', cafe)
    assert titled.rewrite is not None and titled.rewrite_findings == []
    assert titled.set_aside == []


def test_words_placing_the_lettering_are_no_lettering_to_the_critic():
    # "With the words shimmering across the chest" names no words to render:
    # the critic may not call "shimmering" text to render, and a rewrite that
    # quotes it would have it lettered, so it is flagged.
    for prompt, words in (
        ("A t-shirt with the words shimmering across the chest", "shimmering"),
        ("A door with the words scratched into the paint", "scratched into the paint"),
        ("A poster with the headline set in bold type", "set"),
        ("A sign with the words lit up at night", "lit up"),
        ("A rug with the text woven into it", "woven into it"),
        ("A neon sign with the words flickering softly", "flickering softly"),
    ):
        rewrite = prompt.replace(words, f'"{words}"')
        result = _validate(
            _answer(rewrite=rewrite, parts={"text_to_render": words}), prompt=prompt
        )
        assert result.parts["text_to_render"] is None, prompt
        assert result.rewrite == rewrite
        (flag,) = result.rewrite_findings
        assert flag.rule == "heuristic.rewrite_quote"
        assert rewrite[flag.start:flag.end] == f'"{words}"'
    # Words in capitals there are the lettering.
    prompt = "A t-shirt with the words GOOD VIBES across the chest"
    rewrite = prompt.replace("GOOD VIBES", '"GOOD VIBES"')
    result = _validate(
        _answer(rewrite=rewrite, parts={"text_to_render": "GOOD VIBES"}), prompt=prompt
    )
    assert result.parts["text_to_render"] == "GOOD VIBES" and result.rewrite_findings == []


def test_lettering_holding_a_negation_is_lettering_to_the_critic_whole():
    # A participle's lowercase words stop at a negation the negation rule
    # reads, but the lettering may run on through it; one that starts inside
    # the words ("do not enter") stops nothing. Either way a rewrite that
    # quotes the whole lettering, as the quote rule asks, is not flagged.
    for prompt, words in (
        ("A red sign reading do not enter above the door", "do not enter"),
        ("A sign reading please do not touch", "please do not touch"),
        ("A t-shirt reading no pain no gain", "no pain no gain"),
        ("A banner reading love without limits", "love without limits"),
        ("A sign reading dogs not allowed", "dogs not allowed"),
        ("A poster reading now or never in bold type", "now or never"),
        ("A sign reading better late than never on the door", "better late than never"),
        ("A shop sign reading we are not open", "we are not open"),
        ("A tombstone reading gone but not forgotten", "gone but not forgotten"),
        ("A poster reading say no to war above the crowd", "say no to war"),
        ("A diner sign reading no shoes no shirt no service", "no shoes no shirt no service"),
    ):
        rewrite = prompt.replace(words, f'"{words}"')
        result = _validate(
            _answer(rewrite=rewrite, parts={"text_to_render": words}), prompt=prompt
        )
        assert result.parts["text_to_render"] == words, prompt
        assert result.rewrite == rewrite and result.rewrite_findings == [], prompt


def test_description_past_lettering_holding_a_negation_is_no_lettering_to_the_critic():
    # Past the negation, the lettering ends where the words would without
    # it, and what follows is description. Left at the negation as a guess,
    # the critic read on to the clause's end and kept "love without limits
    # above the stage" as text to render, and a rewrite quoting it passed.
    for prompt, words, described in (
        ("A banner reading love without limits above the stage", "love without limits",
         "love without limits above the stage"),
        ("A sign reading dogs not allowed in red letters on the gate", "dogs not allowed",
         "dogs not allowed in red letters"),
        ("A poster reading books without pictures in the attic", "books without pictures",
         "books without pictures in the attic"),
    ):
        rewrite = prompt.replace(words, f'"{words}"')
        kept = _validate(_answer(rewrite=rewrite, parts={"text_to_render": words}), prompt=prompt)
        assert kept.parts["text_to_render"] == words, prompt
        assert kept.rewrite == rewrite and kept.rewrite_findings == [], prompt
        rewrite = prompt.replace(described, f'"{described}"')
        result = _validate(
            _answer(rewrite=rewrite, parts={"text_to_render": described}), prompt=prompt
        )
        assert result.parts["text_to_render"] is None, prompt
        assert result.rewrite == rewrite, prompt
        (flag,) = result.rewrite_findings
        assert flag.rule == "heuristic.rewrite_quote", prompt
        assert rewrite[flag.start:flag.end] == f'"{described}"', prompt


def test_a_negation_the_rules_read_is_no_lettering_on_its_own():
    # The lettering may hold the negation, but it starts where the rules'
    # own reading of it does: the words from the negation on are the
    # absence the negation rule warns on, and quoting them letters it. Read
    # on through the negation as a piece of "comics without pictures", a
    # rewrite quoting "without pictures" alone passed, which it did not
    # while the lettering stopped there.
    for prompt, pieces, kept in (
        ("A boy on a box reading comics without pictures",
         ("without pictures", "without", "pictures"), ("comics", "comics without")),
        ("A girl on a crate reading letters never sent", ("never sent", "sent"), ("letters",)),
        ("A sign reading dogs not allowed", ("not allowed", "allowed"), ("dogs", "dogs not")),
        ("A banner reading big love without limits above the stage",
         ("without limits", "limits"), ("big love", "love without limits")),
    ):
        for piece in pieces:
            rewrite = prompt.replace(piece, f'"{piece}"', 1)
            result = _validate(
                _answer(rewrite=rewrite, parts={"text_to_render": piece}),
                prompt=prompt, model_id=KREA_DEV,
            )
            assert result.parts["text_to_render"] is None, (prompt, piece)
            assert result.rewrite == rewrite, (prompt, piece)
            (flag,) = result.rewrite_findings
            assert flag.rule == "heuristic.rewrite_quote", (prompt, piece)
            assert rewrite[flag.start:flag.end] == f'"{piece}"', (prompt, piece)
        for piece in kept:
            rewrite = prompt.replace(piece, f'"{piece}"', 1)
            result = _validate(
                _answer(rewrite=rewrite, parts={"text_to_render": piece}),
                prompt=prompt, model_id=KREA_DEV,
            )
            assert result.parts["text_to_render"] == piece, (prompt, piece)
            assert result.rewrite == rewrite and result.rewrite_findings == [], (prompt, piece)


def test_a_rewrite_that_adds_content_says_so():
    # Seen on the card, told not to: Krea's tea party example came back with
    # a lens, light, greenery and sunlight it never asked for.
    prompt = "Photo of two people having a tea party in a lush garden"
    padded = (
        "Photo of two people having a tea party in a lush garden, wide-angle lens, shallow "
        "depth of field, soft natural lighting, cinematic composition, vibrant greenery"
    )
    (adds,) = _validate(_answer(rewrite=padded), prompt=prompt).rewrite_findings
    assert adds.rule == "heuristic.rewrite_adds" and adds.severity == "info"
    assert adds.source == "heuristic" and adds.start is None
    assert "adds 12 words the prompt does not have (wide-angle, lens, shallow, depth" in adds.message
    # Rewording is not new content: "no people" -> "empty", an inflection, a
    # quotation mark closing on a word.
    reworded = "A photo of two people who have a tea party in a lush, empty garden"
    assert _validate(_answer(rewrite=reworded), prompt=prompt).rewrite_findings == []
    sign = "A quiet street, no people, a sign that says Open Late."
    quoted = "A quiet street, empty, a sign that says 'Open Late'."
    assert _validate(_answer(rewrite=quoted), prompt=sign).rewrite_findings == []
    # Words run together in Chinese; nothing is counted there.
    chinese = "一只猫在沙发上"
    longer = "一只橘色的猫在阳光下的旧沙发上睡觉，背景是木地板和绿植，电影感光线"
    assert _validate(_answer(rewrite=longer), prompt=chinese).rewrite_findings == []


def _dropping(rewrite, prompt, style="tags", model_id=ANIMAGINE):
    found = _validate(_answer(rewrite=rewrite), prompt=prompt, model_id=model_id, style=style)
    return [flag for flag in found.rewrite_findings if flag.rule == "heuristic.rewrite_drops"]


def test_a_rewrite_that_drops_the_users_tags_says_so():
    # The card's SDXL session, 2026-10-25: 8 of 19 rewrites lost or changed
    # a tag the user wrote, unreported. These are its recorded prompt and
    # rewrite pairs (Animagine XL 4.0); a CPU replay of all 19 flags the 8.
    doc = "1girl, green hair, zzkx oc, smile, outdoors, masterpiece"
    (drops,) = _dropping(
        "1girl, zzkx oc, (series), safe, smile, outdoors, masterpiece, high score, great score, "
        "absurdres", doc,
    )
    assert drops.severity == "info" and drops.source == "heuristic" and drops.start is None
    assert drops.cite is None and drops.url is None
    assert "drops 1 of your tags (green hair) in any form" in drops.message
    # A changed tag is a lost one.
    (drops,) = _dropping(
        "1girl, zzkx (series), safe, green hair, smile, outdoors, masterpiece, high score, "
        "great score, absurdres", doc,
    )
    assert "drops 1 of your tags (zzkx oc)" in drops.message
    quality_first = (
        "masterpiece, high score, 1girl, twintails, mirelle valcour, smile, cherry blossoms, "
        "outdoors, upper body"
    )
    (drops,) = _dropping(
        "1girl, mirelle valcour, [series], [rating], smile, cherry blossoms, outdoors, upper body, "
        "masterpiece, high score, great score, absurdres", quality_first,
    )
    assert "drops 1 of your tags (twintails)" in drops.message
    # Spacing aside: `twin tails` is Danbooru's alias of `twintails`.
    assert _dropping("1girl, mirelle valcour, long twin tails, smile, cherry blossoms, outdoors, "
                     "upper body, masterpiece, high score", quality_first) == []
    # The rule set reports it beside the rewrite: it is not set aside.
    settled = "zzkx oc, 1girl, green hair, masterpiece, smile, outdoors"
    rewrite = (
        "1girl, zzkx (series), safe, green hair, smile, outdoors, masterpiece, high score, "
        "great score, absurdres"
    )
    result = _validate(_answer(rewrite=rewrite), prompt=settled, model_id=ANIMAGINE, style="tags")
    assert [flag.rule for flag in result.rewrite_findings if flag.rule.endswith("drops")] == [
        "heuristic.rewrite_drops"
    ]
    assert result.rewrite is not None and "zzkx (series)" in result.rewrite


def test_a_rewrite_that_keeps_every_tag_in_some_form_drops_nothing():
    # Recorded on the card, and not among the 8: the tags moved, gained
    # brackets or sit inside a longer tag.
    out_of_order = "green hair, zzkx oc, 1girl, smile, outdoors, masterpiece, high score"
    for rewrite in (
        "1girl, zzkx oc, smile, outdoors, green hair, safe, masterpiece, high score, great score, "
        "absurdres",
        "1girl, zzkx (oc), safe, green hair, smile, outdoors, masterpiece, high score, great score, "
        "absurdres",
    ):
        assert _dropping(rewrite, out_of_order) == [], rewrite
    miku = (
        "1girl, solo, hatsune miku, smile, zzkx oc, vocaloid, twintails, outdoors, masterpiece, "
        "absurdres"
    )
    assert _dropping(
        "1girl, hatsune miku, vocaloid, safe, solo, smile, zzkx oc, twintails, outdoors, "
        "masterpiece, high score, great score, absurdres", miku,
    ) == []
    two = (
        "1girl, aqua eyes, zzkx oc, long hair, qqvr series, school uniform, smile, classroom, "
        "masterpiece, great score"
    )
    assert _dropping(
        "1girl, zzkx oc (qqvr series), safe, aqua eyes, long hair, school uniform, smile, "
        "classroom, masterpiece, high score, great score, absurdres", two,
    ) == []
    prompt = r"1girl, zzkx oc, green hair, twintails, hatsune miku \(vocaloid\), masterpiece"
    for rewrite in (
        # A pure reorder.
        r"masterpiece, 1girl, hatsune miku \(vocaloid\), twintails, green hair, zzkx oc",
        # Underscores, case, escapes and a weight only.
        "1girl, ZZKX_OC, green_hair, (twintails:1.2), hatsune_miku_(vocaloid), Masterpiece",
        # A longer tag holding one, and added tags beside them.
        r"1girl, zzkx oc, long green hair, twintails, hatsune miku \(vocaloid\), masterpiece, smile",
    ):
        assert _dropping(rewrite, prompt) == [], rewrite


def test_a_prose_rewrite_drops_words_only_past_rewording():
    # Z-Image Turbo on the card lost "a dramatic and melancholic mood": the
    # recorded prompt, and its recorded rewrite, the same less that clause.
    prompt = (
        "A weathered lighthouse keeper in his seventies stands on the rocky shore of a stormy "
        "North Atlantic coast, his yellow oilskin coat whipping in the wind as he holds a brass "
        "lantern up against the dusk. Behind him the red and white striped lighthouse rises from "
        "black basalt cliffs, its beam cutting a pale arc through sheets of rain and low, bruised "
        "clouds. Waves explode against the rocks in the foreground, throwing spray that catches the "
        "last orange light from a break in the western sky. Shot on a medium format film camera "
        "with an 80mm lens at f/4, shallow depth of field, cinematic colour grading with teal "
        "shadows and warm highlights, fine film grain, a dramatic and melancholic mood."
    )
    rewrite = prompt.replace(", a dramatic and melancholic mood.", ".")
    (drops,) = _dropping(rewrite, prompt, style="prose", model_id="Tongyi-MAI/Z-Image-Turbo")
    assert "drops 3 of the prompt's words (dramatic, melancholic, mood)" in drops.message
    # Rewording one word is not taken out content.
    sign = "A quiet street, no people, a sign that says Open Late."
    assert _dropping("A quiet street, empty, a sign that says Open Late.", sign, style="prose",
                     model_id=KREA_DEV) == []
