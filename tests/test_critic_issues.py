"""Critic issues are held to deterministic findings and documented claims."""
from __future__ import annotations

import pytest

from backend import critic_context, critic_validation, prompt_rules
from backend.schemas import GenerateRequest, PromptAnalysis, PromptFinding

from tests.critic_helpers import (
    ANIMAGINE,
    KLEIN,
    KREA_DEV,
    _answer,
    _validate,
)


def test_issues_restating_a_rule_finding_are_left_out_and_said_so():
    def issue(problem, fix="Fix it.", kind="other", cite="G1"):
        return {"kind": kind, "problem": problem, "quote": None, "cite_id": cite, "fix": fix}

    finding = PromptFinding(rule="guide.negation", severity="warn", message="m", source="guide")
    # The prompt is under klein's 30-80, so the rules reported its count.
    words = PromptFinding(rule="guide.words", severity="info", message="m", source="guide")
    issues = [
        issue("Prompt is too short (7 words)", "Expand it to 30-80 words"),
        issue("Uses negation", "Rephrase the negative phrasing"),
        issue("The sign text is not in quotation marks"),
    ]
    # A length claim restates the count the rules measured; the negation is
    # repeated only once it was reported. Neither is a dropped issue.
    unreported = _validate(_answer(issues=issues), findings=[words])
    assert [i.problem for i in unreported.issues] == [
        "Uses negation", "The sign text is not in quotation marks",
    ]
    assert unreported.set_aside == ["1 issue left out: it repeated the rule findings above (word count)."]
    assert unreported.dropped == 0
    reported = _validate(_answer(issues=issues), findings=[words, finding])
    assert [i.problem for i in reported.issues] == ["The sign text is not in quotation marks"]
    assert reported.set_aside == [
        "2 issues left out: they repeated the rule findings above (word count, negation)."
    ]
    assert reported.dropped == 0
    # Krea 2 states no word range: its critic may speak to length.
    krea2 = _validate(_answer(issues=issues[:1]), model_id="krea/Krea-2-Turbo", ordered=False)
    assert len(krea2.issues) == 1 and krea2.set_aside == []
    # An issue quoting the very words a finding points at restates it, in
    # whatever words it puts it ("no people" / "Replace with 'empty'").
    prompt = "A quiet street after the rain, no people, a shop sign that says Open Late."
    start = prompt.index("no people")
    negation = finding.model_copy(update={"start": start, "end": start + len("no people")})
    cut = PromptFinding(rule="encoder.cut", severity="info", message="m", source="encoder",
                        start=prompt.index("a shop"), end=len(prompt))
    quoting = [
        {"kind": "missing", "problem": "no people", "quote": "no people,", "cite_id": "A9",
         "fix": "Replace with 'empty', 'deserted' or 'solitary'"},
        # A cut's span is where reading stops, not what the words say.
        {"kind": "vague", "problem": "The shop is not described", "quote": "a shop sign",
         "cite_id": "A1", "fix": "Say what the shop sells"},
    ]
    result = _validate(
        _answer(issues=quoting, rewrite=None, parts={}), prompt=prompt, findings=[negation, cut]
    )
    assert [i.problem for i in result.issues] == ["The shop is not described"]
    assert result.set_aside == ["1 issue left out: it repeated the rule findings above (negation)."]
    assert result.dropped == 0


FLUX1_DEV = "black-forest-labs/FLUX.1-dev"


def _checked(model_id, prompt, issues, *, findings=(), words=None, negative=None, style="prose"):
    """`validate` on `model_id` over the analysis Check would give: its
    documented order, the prompt's word count unless `words` says, and
    `findings`."""
    request = GenerateRequest(prompt=prompt, negative_prompt=negative)
    analysis = PromptAnalysis(
        model_id=model_id, style=style, findings=list(findings),
        words=prompt_rules.word_count(prompt) if words is None else words,
        order=prompt_rules.documented_order(model_id),
    )
    context = critic_context.build_context(model_id, request, request, analysis)
    answer = _answer(issues=issues, rewrite=None, parts={})
    return critic_validation.validate(answer, context, prompt, model="test", seconds=0)


def _quoting(problem, quote, fix="Fix it.", cite="G1", kind="other"):
    return {"kind": kind, "problem": problem, "quote": quote, "cite_id": cite, "fix": fix}


def _at(prompt, words, rule="guide.negation", source="guide", field="prompt"):
    start = prompt.index(words)
    return PromptFinding(rule=rule, severity="warn", message="m", source=source, field=field,
                         start=start, end=start + len(words))


def test_only_a_quote_taking_in_a_findings_words_restates_it():
    # Short phrases beside a finding are about themselves: "a shop" and
    # "rain" sit within a few characters of "no people", and were once left
    # out as restating the negation.
    prompt = "A quiet street after the rain, no people, a shop sign that says Open Late."
    issues = [
        _quoting("What kind of shop?", "a shop"),
        _quoting("Rain is vague", "rain"),
        # Over the finding's words and a word or two past them: about them,
        # in words none of the patterns read.
        _quoting("Says what is missing", "rain, no people,", fix="Name what is there"),
        # Further past them, either side, it is about more than the negation.
        _quoting("The street's mood is unclear", "street after the rain, no people"),
        _quoting("The sign is vague", "no people, a shop sign", fix="Say what the shop sells"),
    ]
    result = _checked(KREA_DEV, prompt, issues, findings=[_at(prompt, "no people")])
    assert [i.problem for i in result.issues] == [
        "What kind of shop?", "Rain is vague", "The street's mood is unclear", "The sign is vague",
    ]
    assert result.set_aside == ["1 issue left out: it repeated the rule findings above (negation)."]
    assert result.dropped == 0
    # A finding on one word takes a quote of that word alone: "says OPEN" is
    # about the sign, the bare "OPEN" about its missing quotation marks.
    sign = "A cafe window at night, a sign that says OPEN, rain on the glass"
    quote = _at(sign, "OPEN", rule="guide.quote_text")
    issues = [
        _quoting("The sign's colour is not given", "says OPEN", fix="Say what colour it is"),
        _quoting("OPEN", "OPEN,", fix="Write it as the sign's lettering"),
    ]
    result = _checked(KREA_DEV, sign, issues, findings=[quote])
    assert [i.problem for i in result.issues] == ["The sign's colour is not given"]
    assert result.set_aside == [
        "1 issue left out: it repeated the rule findings above (quotation marks)."
    ]


def test_a_finding_about_characters_is_not_one_about_the_words():
    # T5 reads <unk> for nearly all of a Chinese prompt on FLUX.1, so its
    # unknown-character spans cover the text; every quoted issue once went
    # as restating them, whatever it was about.
    prompt = "夜晚的霓虹城市，一个撑红伞的女人走过一家商店，没有行人，胶片摄影"
    unknown = [
        PromptFinding(rule="encoder.unknown_chars", severity="info", message="m", source="encoder",
                      start=start, end=end)
        for start, end in ((0, 7), (8, 22), (23, 27), (28, 32))
    ]
    issues = [
        _quoting("An absence rather than what is there", "没有行人", fix="Say the street is empty"),
        _quoting("The woman is not described", "一个撑红伞的女人"),
    ]
    result = _checked(KREA_DEV, prompt, issues, findings=unknown)
    assert len(result.issues) == 2 and result.set_aside == [] and result.dropped == 0
    # A brace T5 cannot read, or a control token, next to the words quoted.
    cat = "a cat {sitting} on a mat, </s> by the door"
    marks = [
        _at(cat, "{", rule="encoder.unknown_chars", source="encoder"),
        _at(cat, "}", rule="encoder.unknown_chars", source="encoder"),
        _at(cat, "</s>", rule="encoder.control_token", source="encoder"),
    ]
    issues = [
        _quoting("Which cat?", "a cat"), _quoting("Which door?", "by the door"),
        # Or the marks themselves: what they are, not what they mean.
        _quoting("Sitting how?", "{sitting}"), _quoting("What is this?", "</s>"),
    ]
    result = _checked(KREA_DEV, cat, issues, findings=marks)
    assert len(result.issues) == 4 and result.set_aside == []
    # An author's template slot is words the prompt was to replace.
    anime = "Detailed anime portrait of [character], soft light"
    slot = _at(anime, "[character]", rule="author.placeholder", source="author")
    issues = [_quoting("Who is it?", "[character]"), _quoting("Which style?", "Detailed anime")]
    result = _checked("SeeSee21/Z-Anime", anime, issues, findings=[slot])
    assert [i.problem for i in result.issues] == ["Which style?"]
    assert result.set_aside == ["1 issue left out: it repeated the rule findings above (placeholder)."]
    # Literal weighting syntax is about the words it weights.
    weighted = "a cat (sitting:1.3) on a mat"
    weighting = _at(weighted, "(sitting:1.3)", rule="encoder.weighting", source="encoder")
    result = _checked(KREA_DEV, weighted, [_quoting("Sitting how?", "(sitting:1.3)")],
                      findings=[weighting])
    assert result.issues == [] and result.set_aside == [
        "1 issue left out: it repeated the rule findings above (weighting)."
    ]


def test_a_length_or_a_number_of_words_is_not_always_the_prompts_length():
    # Krea [dev]'s 30-80, a prompt inside it and no rule finding at all: a
    # dress length, a focal length and three words more on a glow are issues
    # of their own, once left out as "word count".
    prompt = (
        "A tall woman in a long red silk dress stands on a wet cobblestone street at night, "
        "neon signs glowing behind her, light rain falling, reflections of pink and blue "
        "light on the stones, a black umbrella in her hand, cinematic mood, shot on film"
    )
    words = prompt_rules.word_count(prompt)
    assert 30 <= words <= 80
    issues = [
        _quoting("The dress length is not specified", "long red silk dress"),
        _quoting("No lens is named", None, fix="Give a focal length of 85mm", cite="G2"),
        _quoting("The glow is thin", "neon signs glowing", fix="Add 3 words on its glow"),
        # Nor were these the prompt's length, said to be one the check had
        # measured after the first fix: a count for, to or describing a part.
        _quoting("The lighting is thin", None, fix="Add at least 5 words for the lighting"),
        _quoting("Only 2 words describe the setting", None, fix="Describe the street more"),
        _quoting("The umbrella is vague", "a black umbrella",
                 fix="Describe the umbrella in more than 2 words"),
    ]
    result = _checked(KREA_DEV, prompt, issues)
    assert len(result.issues) == 6 and result.set_aside == [] and result.dropped == 0
    # Nor, after the second, a count of the prompt's words it gives to a
    # part, or one a part is said to have.
    issues = [
        _quoting("Only 3 words of the prompt are about the lighting", None, fix="Add more"),
        _quoting("Just 2 words of the prompt cover the umbrella", "a black umbrella",
                 fix="Say more about it"),
        _quoting("The lighting has only 3 words", None, fix="Describe the light source"),
    ]
    result = _checked(KREA_DEV, prompt, issues)
    assert len(result.issues) == 3 and result.set_aside == [] and result.dropped == 0
    # Nor, after the third, a count given to a part named with its
    # description: the description alone is the prompt's, not the lighting's.
    issues = [
        _quoting("The lighting is long", None, fix="Keep the lighting description under 5 words"),
        _quoting("The background is thin", None,
                 fix="Give the background description at least 5 words"),
        _quoting("The lighting description has only 3 words", None,
                 fix="Describe the light source"),
    ]
    result = _checked(KREA_DEV, prompt, issues)
    assert len(result.issues) == 3 and result.set_aside == [] and result.dropped == 0
    # Nor, after its review, one named "your" or "a" lighting description,
    # or a part named in three words ahead of it.
    issues = [
        _quoting("The lighting is long", None, fix="Keep your lighting description under 5 words"),
        _quoting("The umbrella is thin", "a black umbrella",
                 fix="Give an umbrella description at least 5 words"),
        _quoting("The lighting is long", None,
                 fix="Keep the soft pink neon description under 5 words"),
        _quoting("The lighting is thin", None,
                 fix="Give the description of your lighting at least 5 words"),
    ]
    result = _checked(KREA_DEV, prompt, issues)
    assert len(result.issues) == 4 and result.set_aside == [] and result.dropped == 0
    # A claim about the prompt's own length is still left out, and said to
    # be what it is: the check measured the count, and there is no finding.
    # "Reads as long" is one, as "reads long" is; so is a count given to the
    # critic's own version of the description.
    measured = [
        f"1 issue left out: it spoke to the prompt's length, which the check measured "
        f"({words} words, inside the 30-80 the guide suggests)."
    ]
    long = [_quoting("The prompt is too long for the guide", None, fix="Cut it to 30-80 words")]
    reads = [_quoting("The prompt reads as long", None, fix="Shorten it")]
    revised = [_quoting("Thin", None, fix="Keep the revised description under 80 words")]
    result = _checked(KREA_DEV, prompt, revised)
    assert result.issues == [] and result.set_aside == measured
    for claim in (long, reads):
        result = _checked(KREA_DEV, prompt, claim)
        assert result.issues == [] and result.set_aside == measured
        assert result.dropped == 0
    # Reported, it repeats the finding; in XML or JSON, which the rules do
    # not hold to the range however many words it has, it is the critic's
    # to raise.
    under = PromptFinding(rule="guide.words", severity="info", message="m", source="guide")
    for claim in (long, reads):
        result = _checked(KREA_DEV, "A cat.", claim, findings=[under])
        assert result.set_aside == [
            "1 issue left out: it repeated the rule findings above (word count)."
        ]
    structured = (
        '{"subject": "a tall woman in a long red silk dress", "setting": "a wet cobblestone '
        'street at night, neon signs glowing behind her", "style": "cinematic film photograph", '
        '"lighting": "pink and blue neon reflected on the stones", "camera": "shot on 35mm film"}'
    )
    assert 30 <= prompt_rules.word_count(structured) <= 80
    result = _checked(KLEIN, structured, long, style="json")
    assert len(result.issues) == 1 and result.set_aside == []
    assert len(_checked(KREA_DEV, "<cat/>", long, style="xml", words=45).issues) == 1


# What reads as the prompt's own length, and what does not: a number of
# words is often the sign's, the glow's or a lens's.
@pytest.mark.parametrize(
    ("problem", "fix", "claim"),
    [
        ("Prompt is too short (7 words)", "Expand it to 30-80 words", True),
        ("It is too short for the guide.", "Add more words.", True),
        ("Only 7 words.", "Expand to about 40.", True),
        ("It is 95 words, above the range.", "Trim to 80.", True),
        ("Too long", "Cut it down to 60 words.", True),
        ("At 7 words, the prompt falls short of the guide's 30-80.", "Expand it.", True),
        ("The word count is below the recommended range", "Add detail", True),
        ("Out of range", "Keep the prompt between 30 and 80 words", True),
        ("12 words is below the recommended 30", "Add detail", True),
        ("The prompt is short", "Describe more", True),
        ("Thin", "Add details to reach at least 30 words", True),
        ("The prompt, at 12 words, is short", "Add detail", True),
        ("The prompt runs long", "Shorten it", True),
        # A count that names the prompt, "it", a total or the range is still
        # its length, whatever preposition carries it.
        ("Only 12 words in the prompt", "Add detail", True),
        ("The prompt has fewer than 30 words in it", "Add detail", True),
        ("Thin", "Add at least 30 words to the prompt", True),
        ("Thin", "Add at least 30 words to reach the range", True),
        ("Thin", "Write at least 30 words for the guide", True),
        ("Thin", "Add at least 10 words in total", True),
        ("Thin", "Give the prompt at least 30 words", True),
        ("Thin", "Give it at least 30 words", True),
        ("Too long a prompt", "Cut it", True),
        ("It is below the ideal length of the prompt", "Add detail", True),
        ("The dress length is not specified", "Say how long", False),
        ("No lens is named", "Give a focal length of 85mm", False),
        ("The glow is thin", "Add 3 words on its glow", False),
        ("The glow is thin", "In the prompt, add 3 words on the glow.", False),
        ("The lighting is vague", "Expand the prompt with 5 words about the lighting.", False),
        ("The lighting is vague", "Expand the prompt with 5 words for the lighting.", False),
        ("The lighting is thin", "Add at least 5 words about the lighting", False),
        # Some part's count, whichever words carry it: once left out as the
        # prompt's length the check had measured.
        ("The lighting is thin", "Add at least 5 words for the lighting", False),
        ("The lighting is thin", "Add more than 3 words to the lighting", False),
        ("The setting is thin", "Add at least 3 words to describe the setting", False),
        ("The setting is thin", "Add at least 3 words in the setting clause", False),
        ("The lighting is thin", "Add at least 5 words covering the lighting", False),
        ("The umbrella is vague", "Describe the umbrella in more than 2 words", False),
        ("Only 2 words describe the setting", "Describe the street more", False),
        ("The fog is described in just 4 words", "Say how thick", False),
        ("The ideal length of the raincoat is unclear", "Say how long", False),
        ("The lighting is thin", "Give the lighting at least 5 words", False),
        ("The lighting gets only 3 words", "Describe the light source", False),
        ("Too long a shadow is implied", "Shorten the shadow", False),
        ("The sign is long", "Keep the sign text under 5 words.", False),
        ("The lettering is 3 words long.", "Shorten it", False),
        ("The text on the mug is long", "Cut it to 2 words", False),
        ("The shot is too long", "Crop it", False),
        # Phase-4 review: the prompt's words given to a part are the part's,
        # as they were before "of the prompt" came to name the prompt.
        ("Only 3 words of the prompt are about the lighting", "Add more", False),
        ("Just 2 words of the prompt cover the umbrella", "Say more", False),
        ("Only 12 words of the prompt describe the subject", "Add detail", False),
        ("Only 2 words in the prompt go to the lighting", "Add more", False),
        # A fit, "all", a range or a guide is the prompt's only where it is
        # the prompt's: "to match the lighting" and "the dynamic range" are not.
        ("The lighting is thin", "Add at least 5 words to match the lighting", False),
        ("The mood is thin", "Add at least 5 words to fit the mood", False),
        ("The subject is thin", "Add at least 5 words to follow the subject", False),
        ("The figures are thin", "Add at least 3 words for all three figures", False),
        ("The lighting is thin", "Add at least 5 words for the dynamic range", False),
        ("The sky is thin", "Add at least 5 words to the colour range", False),
        ("The backdrop is thin", "Add at least 5 words to the mountain range", False),
        ("The style is thin", "Add at least 5 words to the style guide", False),
        ("The lighting has only 3 words", "Describe the light source", False),
        ("The lighting is only 3 words", "Describe the light source", False),
        ("The setting has just 2 words", "Describe the street", False),
        ("The lighting has fewer than 3 words", "Describe the light source", False),
        ("The lighting is thin", "Expand the lighting to at least 5 words", False),
        ("Thin", "Add at least 10 words in all", True),
        ("Thin", "Use fewer than 80 words to stay in range", True),
        ("Thin", "Add at least 30 words to fit the guide's 30-80", True),
        ("Thin", "Add at least 10 words for a total of 40", True),
        ("It has only 12 words", "Add detail", True),
        ("The whole prompt has only 12 words", "Add detail", True),
        # ... and what the first fix kept that is the prompt's own length:
        # "it" in so many words, a length of so many words, too long a
        # description.
        ("Thin", "Keep it in under 80 words", True),
        ("Thin", "Say it in under 80 words", True),
        ("It is below the recommended length of 30 words", "Add detail", True),
        ("The ideal length of prompts is longer", "Add detail", True),
        ("Too long a description", "Cut it", True),
        ("Too long a list of tags", "Cut it", True),
        ("Too short a description for the guide", "Add detail", True),
        # Last fix round: a part named with its description is that part,
        # once left out as the prompt's measured length; the description
        # alone, or the full one, is still the prompt's. So is how it reads.
        ("The lighting is long", "Keep the lighting description under 5 words", False),
        ("The background is thin", "Give the background description at least 5 words", False),
        ("The lighting description has only 3 words", "Describe the light source", False),
        ("The lighting is long", "Keep the lighting description in under 5 words", False),
        ("The lighting is thin", "Give the description of the lighting at least 5 words", False),
        ("Thin", "Keep the description under 80 words", True),
        ("Thin", "Keep your description under 80 words", True),
        ("Thin", "Keep your description in under 80 words", True),
        ("The prompt reads as long", "Shorten it", True),
        ("The prompt reads as short", "Add detail", True),
        # Its review: a word for the critic's own version or for the whole
        # ahead of "description" read as a part, and "your", "a" or three
        # words ahead of a part's description as the prompt's; "descriptions"
        # was a part's in a count, the prompt's in a manner; and "as long as",
        # a comparison or a condition, read as a claim whatever the verb.
        ("Thin", "Keep the revised description under 80 words", True),
        ("Thin", "Keep the rewritten description in under 80 words", True),
        ("Thin", "Give the short description at least 30 words", True),
        ("Thin", "Expand the short description to at least 30 words", True),
        ("The new description has only 12 words", "Add detail", True),
        ("The lighting is long", "Keep your lighting description under 5 words", False),
        ("The lighting is thin", "Give your lighting description at least 5 words", False),
        ("Your lighting description has only 3 words", "Describe the light source", False),
        ("The lighting is long", "Keep a lighting description under 5 words", False),
        ("The lighting is thin", "Give the description of your lighting at least 5 words", False),
        ("The lighting is long", "Keep the soft warm lighting description under 5 words", False),
        ("The descriptions have only 12 words", "Add detail", True),
        ("Thin", "Keep the descriptions under 80 words", True),
        ("Thin", "Keep the descriptions in under 80 words", True),
        ("The prompt is as long as it needs to be, but the lighting is vague",
         "Describe the light source", False),
        ("The prompt runs as long as the scene needs, but no lens is named", "Name a lens", False),
        ("The prompt seems as long as it should be, but the mood is unclear", "Name the mood",
         False),
        ("The prompt reads as long as it needs to, but the lighting is vague",
         "Describe the light source", False),
        ("The prompt reads as too long", "Shorten it", True),
    ],
)
def test_a_length_claim_is_one_about_the_prompts_own_length(problem, fix, claim):
    # Krea [dev] with a prompt counted inside its 30-80: a claim about the
    # length is left out, anything else is the critic's.
    result = _checked(KREA_DEV, "A cat on a mat.", [_quoting(problem, None, fix=fix)], words=45)
    assert (result.issues == []) is claim
    assert result.set_aside == ([
        "1 issue left out: it spoke to the prompt's length, which the check measured "
        "(45 words, inside the 30-80 the guide suggests)."
    ] if claim else [])


# Every word `guide.words` takes as tying a count to a part, or to the
# prompt, its guide or its range, one row apiece - so that each is pinned.
@pytest.mark.parametrize(
    ("problem", "fix", "claim"),
    [
        # A count given to a part after it (`_GIVEN`), partitive or not.
        ("The lighting is thin", "Spend at least 5 words on the lighting", False),
        ("The lighting is thin", "Use at least 5 words of detail on the lighting", False),
        ("The setting is thin", "Add at least 3 words describing the setting", False),
        ("Just 1 word describes the setting", "Describe the pier", False),
        ("Only 2 words described the setting", "Describe the pier", False),
        ("Only 2 words cover the fog", "Say how thick", False),
        ("Just 1 word covers the fog", "Say how thick", False),
        ("Only 3 words are about the lighting", "Add more", False),
        ("Just 1 word is for the lighting", "Add more", False),
        ("Only 2 words go to the fog", "Say how thick", False),
        ("Just 1 word goes on the fog", "Say how thick", False),
        # How a part is to be described (`_MANNER`), unless it is the prompt.
        ("The fog is described with only 4 words", "Say how thick", False),
        ("The lighting is long", "Rewrite the lighting description in fewer than 10 words",
         False),
        ("The lighting is long", "Say the soft lighting description with at most 5 words", False),
        ("Thin", "Say this in under 80 words", True),
        ("Thin", "Keep the description in under 80 words", True),
        ("Thin", "Pad the prompt with at least 10 words", True),
        ("Thin", "Expand it with at least 10 words", True),
        # A fit to the guide, its range or a bound (`_OWN`, `_FIT`).
        ("Thin", "Add at least 30 words to meet the minimum", True),
        ("Thin", "Add at least 30 words to hit the guide's range", True),
        ("Thin", "Write at least 30 words to fit the model's range", True),
        ("Thin", "Add at least 30 words to match the recommended range", True),
        ("Thin", "Use fewer than 80 words to fall within the suggested range", True),
        ("Thin", "Add at least 30 words to satisfy the guide", True),
        ("Thin", "Add at least 30 words to follow the guide", True),
        ("Thin", "Add at least 30 words for the 30-80 range", True),
        ("The lighting is thin", "Add at least 5 words to describe it", False),
        ("Thin", "Add at least 30 words to get into range", True),
        ("Thin", "Add at least 30 words to get it into range", True),
        ("Thin", "Use fewer than 80 words to stay inside the 30-80 range", True),
        ("Thin", "Use fewer than 80 words to stay under the maximum", True),
        ("Thin", "Use fewer than 80 words to stay below 80", True),
        ("Thin", "Add at least 30 words to reach about 40", True),
        ("Thin", "Add at least 30 words to reach around 40", True),
        ("Thin", "Add at least 10 words to reach at least 30", True),
        ("Thin", "Use fewer than 80 words to fall within a suggested range", True),
        ("Thin", "Add at least 30 words to reach its range", True),
        ("Thin", "Add at least 30 words to meet this minimum", True),
        ("Thin", "Add at least 30 words to fit your limit", True),
        # A part named ahead of its count (`_PART_COUNT`), each determiner,
        # verb and shaping verb - and the prompt, the whole, the count or the
        # problem named there instead.
        ("The lighting is thin", "Devote the lighting at least 5 words", False),
        ("The lighting is thin", "Give its lighting at least 5 words", False),
        ("The coat is thin", "Give his coat at least 5 words", False),
        ("The dress is thin", "Give her dress at least 5 words", False),
        ("The faces are thin", "Give their faces at least 5 words", False),
        ("The detail is thin", "Give this detail at least 5 words", False),
        ("The detail is thin", "Give that detail at least 5 words", False),
        ("The figures are thin", "Give each figure at least 3 words", False),
        ("The lighting receives only 3 words", "Describe the light source", False),
        ("The eyes have only 2 words", "Describe them", False),
        ("The eyes are only 2 words", "Describe them", False),
        ("The lighting uses only 3 words", "Describe the light source", False),
        ("The lighting is long", "Cut the lighting to at most 5 words", False),
        ("The lighting is long", "Trim the lighting to at most 5 words", False),
        ("The lighting is long", "Shorten the lighting to at most 5 words", False),
        ("The lighting is long", "Keep the lighting under 5 words", False),
        ("The lighting is long", "Limit the lighting to at most 5 words", False),
        ("The lighting is long", "Keep the lighting below 5 words", False),
        ("The full description has only 12 words", "Add detail", True),
        ("The whole thing is only 12 words", "Add detail", True),
        ("The entire thing is only 12 words", "Add detail", True),
        ("The word count is only 12 words", "Add detail", True),
        ("Its length is only 12 words", "Add detail", True),
        ("The total is only 12 words", "Add detail", True),
        ("The problem is only 12 words", "Add detail", True),
        ("The issue is only 12 words", "Add detail", True),
        # A part named with its description, before or after it, however
        # many words name the part and whatever the verb - and the prompt's
        # own description named the same way.
        ("The lighting descriptions have only 3 words", "Describe the light source", False),
        ("The lighting is long", "Keep the full lighting description under 5 words", False),
        ("The lighting is long", "Keep the description of the lighting under 5 words", False),
        ("The lighting is thin", "Give the description of its lighting at least 5 words", False),
        ("Thin", "Give the description at least 30 words", True),
        ("Thin", "Give the description of the prompt at least 30 words", True),
        ("Thin", "Give the description of the whole scene at least 30 words", True),
        ("Thin", "Give the description of the image at least 30 words", True),
        ("Thin", "Give the description of the full picture at least 30 words", True),
        ("The moon is thin", "Give the description of the full moon at least 5 words", False),
        ("Thin", "Keep the full image description under 80 words", True),
        # A length of so many words, or of the prompt's description.
        ("It is below the recommended length of about 30 words", "Add detail", True),
        ("It is below the recommended length of around 30 words", "Add detail", True),
        ("It is below the ideal length of the description", "Add detail", True),
        # How the prompt reads.
        ("The prompt reads long", "Shorten it", True),
        ("The prompt seems short", "Add detail", True),
        ("The prompt runs a little long", "Shorten it", True),
        ("The prompt is a bit short", "Add detail", True),
        ("The prompt is slightly long", "Shorten it", True),
    ],
)
def test_each_word_tying_a_count_to_a_part_or_to_the_prompt_is_read(problem, fix, claim):
    result = _checked(KREA_DEV, "A cat on a mat.", [_quoting(problem, None, fix=fix)], words=45)
    assert (result.issues == []) is claim


def _claims_length(fix):
    result = _checked(KREA_DEV, "A cat on a mat.", [_quoting("Thin", None, fix=fix)], words=45)
    return result.issues == []


# Every word just ahead of "description" that makes it the prompt's - a word
# for the whole or for the critic's own version of it (`_WHOLE_WORDS`) - one
# row apiece, in each form a count or a manner takes, singular and plural (a
# noun for the whole as a possessive too); beside a part's word after it,
# which names a part again.
@pytest.mark.parametrize(
    "word",
    ["full", "whole", "entire", "complete", "overall", "total", "general", "current",
     "existing", "original", "main", "new", "revised", "rewritten", "updated", "edited",
     "improved", "expanded", "suggested", "proposed", "final", "detailed", "short", "shorter",
     "brief", "long", "longer", "image", "picture", "prompt"],
)
def test_each_word_making_a_description_the_prompts_is_read(word):
    for noun, has in (("description", "has"), ("descriptions", "have")):
        fixes = [
            f"Give the {word} {noun} at least 30 words",
            f"Keep the {word} {noun} under 80 words",
            f"Expand the {word} {noun} to at least 30 words",
            f"The {word} {noun} {has} only 12 words",
            f"Keep the {word} {noun} in under 80 words",
            f"Keep the {word} {noun} with under 80 words",
        ]
        if word in ("image", "picture", "prompt"):
            fixes += [f"Keep the {word}'s {noun} in under 80 words",
                      f"The {word}’s {noun} {has} only 12 words"]
        for fix in fixes:
            assert _claims_length(fix), fix
        for fix in (
            f"Keep the {word} lighting {noun} under 5 words",
            f"The {word} lighting {noun} {has} only 3 words",
            f"Keep the {word} lighting {noun} in under 5 words",
        ):
            assert not _claims_length(fix), fix


# A determiner alone ahead of "description" makes it the prompt's, and one
# ahead of a part's word names that part - in a count and in a manner alike,
# every determiner (`_DETERMINERS`), however many words name the part.
@pytest.mark.parametrize(
    "determiner",
    ["the", "this", "that", "your", "my", "our", "its", "his", "her", "their", "each", "a", "an"],
)
def test_a_description_after_a_determiner_alone_is_the_prompts(determiner):
    # "An" is tried on an umbrella, and before a word for the whole alone.
    part = "umbrella" if determiner == "an" else "lighting"
    alone = "an overall" if determiner == "an" else determiner
    lead = alone[:1].upper() + alone[1:]
    named = determiner[:1].upper() + determiner[1:]
    singular = determiner in ("this", "that", "each", "a", "an")
    for noun, has in (("description", "has"),) + (() if singular else (("descriptions", "have"),)):
        for fix in (
            f"Keep {alone} {noun} under 80 words",
            f"{lead} {noun} {has} only 12 words",
            f"Say {alone} {noun} in under 80 words",
            f"Say {alone} {noun} with under 80 words",
        ):
            assert _claims_length(fix), fix
        for fix in (
            f"Keep {determiner} {part} {noun} under 5 words",
            f"Give {determiner} {part} {noun} at least 5 words",
            f"{named} {part} {noun} {has} only 3 words",
            f"Keep {determiner} soft warm {part} {noun} under 5 words",
            f"Say {determiner} {part} {noun} in under 5 words",
            f"Say {determiner} soft warm {part} {noun} with at most 5 words",
            f"Keep {alone} {noun} of the {part} under 5 words",
        ):
            assert not _claims_length(fix), fix
    # A part named ahead of its count without its description takes a
    # determiner other than "a" or "an", which read as a quantity there.
    if determiner in ("a", "an"):
        assert _claims_length("Keep a single paragraph under 80 words")
    else:
        assert not _claims_length(f"Give {determiner} lighting at least 5 words")
        assert not _claims_length(f"Keep the description of {determiner} lighting under 5 words")


def test_a_negative_prompts_findings_leave_the_prompts_issues_alone():
    # Weighting found in Animagine's negative prompt is no reason to leave
    # out an issue asking the prompt for emphasis.
    prompt, negative = "1girl, smile, looking at viewer", "(worst quality:1.4), lowres"
    weighting = _at(negative, "(worst quality:1.4)", rule="encoder.weighting", source="encoder",
                    field="negative_prompt")
    issues = [
        _quoting("The expression is flat", None, fix="Give the expression more emphasis"),
        _quoting("Where is she looking?", "looking at viewer"),
    ]
    result = _checked(ANIMAGINE, prompt, issues, findings=[weighting], negative=negative,
                      style="tags")
    assert len(result.issues) == 2 and result.set_aside == [] and result.dropped == 0
    # One that says it is about the negative prompt restates that finding.
    about = _quoting("The negative prompt uses weighting syntax, read as literal text", None,
                     fix="Drop the weights from the negative prompt")
    result = _checked(ANIMAGINE, prompt, [*issues, about], findings=[weighting],
                      negative=negative, style="tags")
    assert len(result.issues) == 2 and result.dropped == 0
    assert result.set_aside == ["1 issue left out: it repeated the rule findings above (weighting)."]
    # Weighting found in the prompt is what the first one restates.
    weighted = "1girl, (smile:1.2), looking at viewer"
    on_prompt = _at(weighted, "(smile:1.2)", rule="encoder.weighting", source="encoder")
    result = _checked(ANIMAGINE, weighted, issues, findings=[on_prompt], style="tags")
    assert [i.problem for i in result.issues] == ["Where is she looking?"]
    assert result.set_aside == ["1 issue left out: it repeated the rule findings above (weighting)."]


def test_order_issues_that_ask_for_different_things_first_are_left_out():
    def order(problem, fix):
        return {"kind": "order", "problem": problem, "quote": None, "cite_id": "A1", "fix": fix}

    # Both from one answer about Krea's own "Moody night photography, ...":
    # with no documented order to settle them, neither can stand.
    both = [
        order("Subject not first", "Move subject to the beginning"),
        order("Style not front-loaded", "Place style description first"),
        {**_answer()["issues"][0]},
    ]
    result = _validate(_answer(issues=both), ordered=False)
    assert [i.kind for i in result.issues] == ["missing"]
    assert result.set_aside == [
        "2 order issues left out: they asked for orders that cannot both hold "
        "(subject first, style first)."
    ]
    assert result.dropped == 0
    # Under one - klein's subject, action, style, context - the style first
    # is what it reverses, and the subject first stands.
    result = _validate(_answer(issues=both))
    assert [i.fix for i in result.issues] == ["Move subject to the beginning", both[2]["fix"]]
    assert result.set_aside == [
        "1 order issue left out: it asked for style first, against the documented order."
    ]
    assert result.dropped == 0
    for fixes, wants in (
        (("Start with the subject.", "Lead with the lighting"), "subject first, lighting first"),
        (("Lead with the lighting", "Put the camera before the lighting."),
         "lighting first, camera before lighting"),
        (("Put the lighting before the camera.", "Put the camera before the lighting."),
         "lighting before camera, camera before lighting"),
    ):
        pair = [order("Out of order", fix) for fix in fixes]
        result = _validate(_answer(issues=pair), ordered=False)
        assert result.issues == [] and result.set_aside == [
            f"2 order issues left out: they asked for orders that cannot both hold ({wants})."
        ], fixes
        assert result.dropped == 0
    # Ones that agree stand, whatever the prompt's words quoted in a fix say
    # and in whatever order a problem names the parts it describes.
    agreeing = [
        order("Subject is not first", "Start with 'a lone hiker in a red jacket'"),
        order("Environmental details precede subject", "Move 'misty light' after the subject"),
        order("Technical specs not prioritized", "Focus on image description, not camera details"),
        # Both from one answer on the card, which once read as subject against style.
        order("The subject is not placed first as required", "Move the subject to the beginning."),
        order("The style description is placed before the subject",
              "Place the subject first, then the style description."),
        # BFL's first step is "the image type and subject".
        order("Out of order", "Start with the image type and subject, then add the rest"),
    ]
    kept = _validate(_answer(issues=agreeing))
    assert len(kept.issues) == 6 and kept.set_aside == []


def test_one_part_before_another_is_held_to_the_documented_order_not_read_as_first():
    def order(fix):
        return {"kind": "order", "problem": "Out of order", "quote": None, "cite_id": "A1",
                "fix": fix}

    # FLUX.1 [dev]: subject, setting, style, camera, lighting. Camera before
    # lighting says nothing of what comes first, and both follow the order:
    # the pair was once read as "camera first" and both left out.
    prompt = "Moody night photography, a small store with warm light, 35mm lens"
    agreeing = [order("Start with the subject."), order("Put the camera settings before the lighting.")]
    result = _checked(FLUX1_DEV, prompt, agreeing)
    assert [i.fix for i in result.issues] == [issue["fix"] for issue in agreeing]
    assert result.set_aside == [] and result.dropped == 0
    # One the documented order reverses is left out; the one it agrees with
    # stands - on klein too, where it once took the subject-first issue with it.
    reversed_ = [order("Start with the subject."), order("Put the lighting before the camera.")]
    result = _checked(FLUX1_DEV, prompt, reversed_)
    assert [i.fix for i in result.issues] == ["Start with the subject."]
    assert result.set_aside == [
        "1 order issue left out: it asked for lighting before camera, against the documented order."
    ]
    klein = [order("Move subject to first position"), order("Place style description before subject")]
    result = _validate(_answer(issues=klein))
    assert [i.fix for i in result.issues] == ["Move subject to first position"]
    assert result.set_aside == [
        "1 order issue left out: it asked for style before subject, against the documented order."
    ]
    # Krea [dev]'s one step is "subject first": nothing goes ahead of it.
    result = _checked(KREA_DEV, prompt, [order("Place the style before the subject.")])
    assert result.issues == [] and result.set_aside == [
        "1 order issue left out: it asked for style before subject, against the documented order."
    ]
    # Where the order says nothing of the parts, a lone pair is the critic's.
    assert len(_checked(KREA_DEV, prompt, [order("Put the camera before the lighting.")]).issues) == 1
    # A part first is held to the order as a pair is: lighting first is
    # what it reverses, not the subject before the lighting.
    lighting = [order("Start with the lighting."), order("Put the subject before the lighting.")]
    result = _checked(FLUX1_DEV, prompt, lighting)
    assert [i.fix for i in result.issues] == ["Put the subject before the lighting."]
    assert result.set_aside == [
        "1 order issue left out: it asked for lighting first, against the documented order."
    ]
    assert result.dropped == 0
    # WAI's first step is "Subject and scene": the scene first is in it.
    wai = "LyliaEngine/waiIllustriousSDXL_v170"
    scene = [order("Start with the scene."), order("Put the scene before the subject.")]
    assert len(_checked(wai, "1girl, city street, night", scene, style="tags").issues) == 2


def test_an_order_instruction_that_is_negated_asks_for_nothing_it_names():
    def order(fix):
        return {"kind": "order", "problem": "Out of order", "quote": None, "cite_id": "A1",
                "fix": fix}

    # Each follows FLUX.1 [dev]'s order - read without its "not", each once
    # asked for what the order reverses and went as against it.
    prompt = "Moody night photography, a small store with warm light, 35mm lens"
    for fix in (
        "Avoid placing the style before the subject.",
        "Do not put the lighting before the camera.",
        "The style should not come before the subject.",
        "Never put the style before the subject.",
        "Don't start with the lighting.",
        "Rather than leading with the style, lead with the subject.",
    ):
        result = _checked(FLUX1_DEV, prompt, [order(fix)])
        assert len(result.issues) == 1 and result.set_aside == [], fix
    # Nor does one clash with another: only the subject is asked for first.
    issues = [order("Don't start with the style."), order("Start with the subject.")]
    result = _checked(FLUX1_DEV, prompt, issues)
    assert len(result.issues) == 2 and result.set_aside == []
    # A negation in another clause is another instruction's.
    result = _checked(FLUX1_DEV, prompt, [order("Do not crop it; put the lighting before the camera.")])
    assert result.issues == [] and result.set_aside == [
        "1 order issue left out: it asked for lighting before camera, against the documented order."
    ]


def test_each_issue_left_out_is_counted_once():
    # One of each, in the six an answer may hold: uncited, restating a
    # finding, speaking to a length the check measured, in an order clash,
    # kept. `dropped` is the uncited alone - the page says those cited
    # nothing. (Against the documented order: the test above.)
    def order(fix):
        return {"kind": "order", "problem": "Out of order", "quote": None, "cite_id": "A1",
                "fix": fix}

    prompt = "A quiet street after the rain, no people, a shop sign that says Open Late."
    issues = [
        _quoting("Uncited", "a shop", cite="G99"),
        _quoting("An absence", "no people", fix="Say the street is empty"),
        _quoting("The prompt is too long", None, fix="Cut it to 30-80 words"),
        order("Put the lighting before the camera."),
        order("Put the camera before the lighting."),
        _quoting("What kind of shop?", "a shop"),
    ]
    assert len(issues) == critic_context.MAX_ISSUES
    # Counted as Check would count a prompt in Krea [dev]'s range.
    result = _checked(KREA_DEV, prompt, issues, findings=[_at(prompt, "no people")], words=45)
    assert [i.problem for i in result.issues] == ["What kind of shop?"]
    assert result.dropped == 1
    assert result.set_aside == [
        "1 issue left out: it repeated the rule findings above (negation).",
        "1 issue left out: it spoke to the prompt's length, which the check measured "
        "(45 words, inside the 30-80 the guide suggests).",
        "2 order issues left out: they asked for orders that cannot both hold "
        "(lighting before camera, camera before lighting).",
    ]
