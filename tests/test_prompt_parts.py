"""Mixed prompts retain comma-containing prose beside factual tag lookups."""
from __future__ import annotations

import pytest

from backend import critic_context, critic_validation, prompt_parts, prompt_rules, prompt_text, tag_lookup, tags
from backend.model_defaults import guide_for
from backend.schemas import GenerateRequest, PromptAnalysis
from tests.api_analysis_helpers import analyse, claim
from tests.critic_helpers import _answer
from tests.prompt_rules_helpers import only, run


KOHAKU = "KBlueLeaf/Kohaku-XL-Zeta"
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
EXAMPLE = "1boy, masterpiece, a man wearing a red, old hat"


def helper(model=KOHAKU):
    return guide_for(model)["tag_helper"]


def absent(text):
    return {"tag": text, "valid": False, "code": "not_found", "reason": "not found in Danbooru"}


def read_parts(prompt, parts):
    encoded = prompt.encode("utf-16-le")
    return [
        (part["kind"], encoded[part["start"] * 2:part["end"] * 2].decode("utf-16-le"))
        for part in parts
    ]


def classify(prompt, verdicts=None, model=KOHAKU, limit=None):
    names = [item.text for item in prompt_text.comma_items(prompt)]
    selected = names if limit is None else names[:limit]
    rows = [dict((verdicts or {}).get(name, absent(name)), tag=name) for name in selected]
    return prompt_parts.contextual_parts(helper(model), prompt, selected, rows)


def test_reported_example_is_two_tags_and_one_exact_prose_passage():
    parts = classify(EXAMPLE)
    assert read_parts(EXAMPLE, parts) == [
        ("tag", "1boy"), ("tag", "masterpiece"), ("prose", "a man wearing a red, old hat"),
    ]
    assert [(part["first"], part["last"]) for part in parts] == [(0, 0), (1, 1), (2, 3)]


def test_prose_does_not_depend_on_the_whole_prompts_majority_style():
    prompt = ", ".join(["blue eyes"] * 40 + ["a man wearing a red", "old hat"])
    assert prompt_text.detect_style(prompt) == "tags"
    parts = classify(prompt, {"blue eyes": {"valid": True, "category": "general"}})
    assert parts[-1]["kind"] == "prose" and parts[-1]["first"] == 40 and parts[-1]["last"] == 41
    assert read_parts(prompt, parts)[-1][1] == "a man wearing a red, old hat"


def test_pure_tags_and_short_misspellings_remain_tag_units():
    prompt = "1boy, masterpiece, bluu eyes, old hat"
    assert all(part["kind"] == "tag" for part in classify(prompt))


def test_pure_prose_with_commas_is_one_passage():
    prompt = "A man wearing a red, old hat, walks through the evening rain."
    assert read_parts(prompt, classify(prompt)) == [("prose", prompt)]


@pytest.mark.parametrize("protected", [
    {"valid": True, "category": "copyright"},
    {"valid": False, "code": "deprecated"},
    {"valid": False, "code": "empty"},
    {"valid": False, "code": "no_date"},
    {"valid": False, "code": "after_cutoff"},
])
def test_known_names_and_concrete_dictionary_problems_end_prose(protected):
    title = "the legend of zelda"
    prompt = f"1boy, a man wearing a red, old hat, {title}, strange words"
    parts = classify(prompt, {title: protected})
    assert read_parts(prompt, parts) == [
        ("tag", "1boy"), ("prose", "a man wearing a red, old hat"),
        ("tag", title), ("tag", "strange words"),
    ]


def test_explicit_artist_prefixes_and_grid_variables_are_not_swallowed():
    prompt = "1boy, a man wearing a red, old hat, @, @no such artist, {lighting}, blue eyes"
    parts = classify(prompt, model="circlestone-labs/Anima")
    assert read_parts(prompt, parts)[2:] == [
        ("tag", "@"), ("tag", "@no such artist"), ("tag", "{lighting}"), ("tag", "blue eyes"),
    ]


def test_raw_spacing_punctuation_and_utf16_offsets_are_preserved():
    prompt = '  1boy,\tmasterpiece,  a man wearing a red 🧢,\n old hat.  '
    assert read_parts(prompt, classify(prompt)) == [
        ("tag", "1boy"), ("tag", "masterpiece"), ("prose", "a man wearing a red 🧢,\n old hat."),
    ]


def test_quoted_commas_and_model_words_are_one_text_passage():
    prompt = '1boy, a sign reading "masterpiece, old hat", blue eyes'
    parts = classify(prompt, {"blue eyes": {"valid": True, "category": "general"}})
    assert read_parts(prompt, parts)[1] == ("prose", 'a sign reading "masterpiece, old hat"')


def test_full_matching_context_is_required_but_grid_value_tail_is_allowed():
    names = [item.text for item in prompt_text.comma_items(EXAMPLE)]
    rows = [absent(name) for name in names]
    assert prompt_parts.contextual_parts(helper(), None, names, rows) is None
    assert prompt_parts.contextual_parts(helper(), EXAMPLE, names[1:], rows[1:]) is None
    assert prompt_parts.contextual_parts(helper(), EXAMPLE, list(reversed(names)), list(reversed(rows))) is None
    tail = ["blue eyes", "green eyes"]
    assert prompt_parts.contextual_parts(helper(), EXAMPLE, names + tail, rows + [absent(n) for n in tail]) == classify(EXAMPLE)


def test_cap_tail_is_represented_without_claiming_unchecked_fragments_are_absent():
    prompt = ", ".join(["blue eyes"] * 63 + ["a man wearing a red", "old hat"])
    parts = classify(prompt, {"blue eyes": {"valid": True, "category": "general"}}, limit=64)
    assert parts[-2]["kind"] == "prose" and parts[-2]["first"] == parts[-2]["last"] == 63
    assert parts[-1]["kind"] == "tag" and parts[-1]["first"] == 64
    assert read_parts(prompt, parts)[-1] == ("tag", "old hat")


def test_unchecked_dictionary_absence_is_not_invented_offline():
    names = [item.text for item in prompt_text.comma_items(EXAMPLE)]
    rows = [{"tag": name, "valid": None} for name in names]
    assert all(part["kind"] == "tag" for part in prompt_parts.contextual_parts(helper(), EXAMPLE, names, rows))
    sentence = "A man is wearing a hat, old hat"
    names = [item.text for item in prompt_text.comma_items(sentence)]
    parts = prompt_parts.contextual_parts(helper(), sentence, names, [{"valid": None}] * 2)
    assert read_parts(sentence, parts) == [("prose", "A man is wearing a hat"), ("tag", "old hat")]


@pytest.mark.parametrize("model", [
    KOHAKU, "OnomaAIResearch/Illustrious-XL-v2.0", "LyliaEngine/waiIllustriousSDXL_v170",
    "Laxhar/noobai-XL-Vpred-1.0", "Minthy/RouWei-0.8", "circlestone-labs/Anima",
    "circlestone-labs/Anima@aesthetic-v1.1", "Gazingstars123/Anima-2.9B",
])
def test_only_explicit_hybrid_metadata_enables_contextual_parts(model):
    assert helper(model)["hybrid_prompt"]
    assert read_parts(EXAMPLE, classify(EXAMPLE, model=model))[-1][0] == "prose"
    assert classify(EXAMPLE, model=ANIMAGINE) is None


def test_parts_do_not_change_raw_endpoint_or_cached_dictionary_verdicts(client, fake, monkeypatch):
    claim(monkeypatch, fake, KOHAKU)
    vocabulary = tags.DanbooruTagVocabulary(lambda path, params: [])
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
    names = [item.text for item in prompt_text.comma_items(EXAMPLE)]
    raw = client.post("/api/tags/validate", json={"tags": names}).json()
    contextual = client.post("/api/tags/validate", json={"tags": names, "prompt": EXAMPLE}).json()
    again = client.post("/api/tags/validate", json={"tags": names}).json()
    assert raw["parts"] is None and again["parts"] is None
    assert raw["results"] == contextual["results"] == again["results"]
    assert all(result["valid"] is False for result in contextual["results"][2:])
    assert read_parts(EXAMPLE, contextual["parts"])[-1] == ("prose", "a man wearing a red, old hat")
    claim(monkeypatch, fake, ANIMAGINE)
    strict = client.post("/api/tags/validate", json={"tags": names, "prompt": EXAMPLE}).json()
    assert strict["parts"] is None and strict["results"][3]["valid"] is False


def test_reordering_withholds_only_a_candidate_that_splits_the_prose():
    unsafe = "1boy, a man wearing a red, masterpiece, old hat"
    safe = "1boy, a man wearing a red, old hat, masterpiece"
    assert prompt_parts.reorder_problem(helper(), EXAMPLE, unsafe, {}) == prompt_parts.REORDER_PROSE_DETAIL
    assert prompt_parts.reorder_problem(helper(), EXAMPLE, safe, {}) is None
    assert prompt_parts.reorder_problem(helper(ANIMAGINE), EXAMPLE, unsafe, {}) is None
    assert prompt_rules.reorder_tags(KOHAKU, EXAMPLE, {}) is None
    assert prompt_rules.reordered_detail(KOHAKU, EXAMPLE, {}) == prompt_parts.REORDER_PROSE_DETAIL
    (finding,) = only(run(KOHAKU, EXAMPLE, {}), "guide.order")
    assert finding.suggestion is None and prompt_parts.REORDER_PROSE_DETAIL in finding.message


def test_a_name_only_the_order_places_is_not_swallowed_into_prose():
    # Anima Aesthetic's Tags menu leaves out the score tags its order still
    # puts first; beside a sentence, score_5 is a placed name, not prose, so
    # moving it splits nothing (smoke-anima-aesthetic).
    model = "circlestone-labs/Anima@aesthetic-v1.1"
    prompt = "1girl, score_5, A girl stands in the rain under a red umbrella. She looks up at the grey sky."
    moved = "score_5, 1girl, A girl stands in the rain under a red umbrella. She looks up at the grey sky."
    assert not any(tag == "score_5" for group in helper(model)["groups"] for tag in group["tags"])
    assert prompt_rules.reorder_tags(model, prompt, {}) == moved
    assert prompt_rules.reordered_detail(model, prompt, {}) is None
    (finding,) = only(run(model, prompt, {}), "guide.order")
    assert finding.suggestion == moved and prompt_parts.REORDER_PROSE_DETAIL not in finding.message


def test_an_underscore_spelling_of_a_placed_name_ends_a_prose_passage():
    # Kohaku's order places "best quality" last; its underscore spelling
    # normalises to the same name, so beside a sentence it is moved past the
    # passage, not read as part of it.
    sentence = "A girl stands in the rain under a red umbrella. She looks up at the grey sky."
    prompt = f"1girl, best_quality, {sentence}"
    assert prompt_rules.reorder_tags(KOHAKU, prompt, {}) == f"1girl, {sentence}, best_quality"
    assert prompt_rules.reordered_detail(KOHAKU, prompt, {}) is None


def test_analysis_explains_why_use_reordered_is_not_offered(client, fake, monkeypatch):
    claim(monkeypatch, fake, KOHAKU)
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", tags.DanbooruTagVocabulary(lambda path, params: []))
    checked = analyse(client, prompt=EXAMPLE)
    assert checked["reordered"] is None
    assert checked["reordered_detail"] == prompt_parts.REORDER_PROSE_DETAIL
    assert all(finding["suggestion"] is None for finding in checked["findings"] if finding["rule"] == "guide.order")


def test_critic_cannot_restore_a_reorder_that_splits_the_same_prose_passage():
    request = GenerateRequest(prompt=EXAMPLE)
    analysis = PromptAnalysis(
        model_id=KOHAKU, style=prompt_rules.detect_style(EXAMPLE), words=10,
        order=prompt_rules.documented_order(KOHAKU), order_checked=True,
        categories={}, reordered=None,
    )
    context = critic_context.build_context(KOHAKU, request, request, analysis)
    answer = _answer(
        issues=[], parts={}, rewrite=None,
        reordered="1boy, a man wearing a red, masterpiece, old hat",
    )
    result = critic_validation.validate(answer, context, EXAMPLE, model="test", seconds=0)
    assert result.reordered is None
    assert any(prompt_parts.REORDER_PROSE_DETAIL in message for message in result.set_aside)
