"""The expanded preview keeps its own defaults and author advice."""
from __future__ import annotations

from backend import guides, prompt_rules, prompt_sources
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for, guide_for, hints_for
from backend.schemas import GenerateRequest, PromptGuide

from tests.prompt_rules_helpers import run


MODEL_ID = "Gazingstars123/Anima-2.9B"


def test_preview_profile_has_guided_defaults_and_final_custom_pipeline_limits():
    defaults = defaults_for(MODEL_ID)
    assert (defaults["steps"], defaults["guidance_scale"], defaults["mu"]) == (30, 4.0, None)
    assert defaults["negative_prompt_supported"] and defaults["cfg_threshold"] == 1
    assert defaults["refine_settled"] and defaults["inpaint_settled"]
    assert defaults["refine_strength"] is None
    assert not any(defaults[key] for key in (
        "mu_supported", "reference_supported", "refine_supported", "inpaint_supported",
        "true_cfg_supported",
    ))
    assert _profile_for(MODEL_ID)["guide"] is guides.ANIMA_PREVIEW_GUIDE
    assert _profile_for(MODEL_ID.upper()) is _profile_for(MODEL_ID)
    for unrelated in ("someone/Anima-2.9B", "Gazingstars123/Anima", "Gazingstars123/Anima-3.8B"):
        assert _profile_for(unrelated) is _FALLBACK
    hints = hints_for(MODEL_ID)
    assert "28-50" in hints["steps"] and "3.5-5" in hints["guidance_scale"]
    assert "Unavailable" in hints["mu"]


def test_preview_guide_separates_recommended_sizes_and_parent_example():
    guide = PromptGuide(**guide_for(MODEL_ID))
    assert guide.pixels == (812 * 1216 / 1e6, 1152 * 1536 / 1e6)
    assert guide.pixels_word == "recommended"
    assert "1536x1536 (2.36 MP) iffy" in hints_for(MODEL_ID)["size"]
    assert guide.examples == guides.ANIMA_TURBO_GUIDE["examples"]
    assert "Parent Anima author example" in guide.example_notes[guide.examples[0]]
    helper = guide.tag_helper
    assert helper.hybrid_prompt and helper.vocabulary == "danbooru"
    assert helper.artist_prefix == "@" and helper.vocabulary_cutoff is None
    assert helper.order == [] and not helper.recommended_prefix
    [parent_order] = helper.author_orders
    assert parent_order.url == "https://huggingface.co/circlestone-labs/Anima"
    assert "inherited parent Anima" in parent_order.note
    assert not any(tag.startswith("score_") for group in helper.groups for tag in group.tags)
    assert prompt_rules.documented_order(MODEL_ID) is None


def test_preview_checks_spelling_without_inventing_order_or_score_prohibition():
    findings = run(
        MODEL_ID, "blue_hair, @wlop, 1girl, score_7, masterpiece", guidance_scale=4,
        categories={"wlop": "artist", "blue hair": "general"},
    )
    spelling = next(f for f in findings if f.rule == "guide.underscores")
    assert spelling.url == "https://huggingface.co/circlestone-labs/Anima"
    assert not any(f.rule in ("guide.order", "guide.sentences") for f in findings)
    score_findings = run(MODEL_ID, "score_7, masterpiece, 1girl", guidance_scale=4)
    assert not any(f.source == "guide" for f in score_findings)
    # The custom pipeline's guided and unguided normalization remains the
    # same before load, while Turbo and Aesthetic retain their own defaults.
    defaults = defaults_for(MODEL_ID)
    request = GenerateRequest(prompt="a dragon", negative_prompt="blurry", guidance_scale=4)
    applied = request.applied(
        cfg_threshold=defaults["cfg_threshold"],
        negative_prompt_supported=defaults["negative_prompt_supported"],
    )
    assert applied.negative_prompt == "blurry"
    assert defaults_for("circlestone-labs/Anima")["steps"] == 8
    assert defaults_for("circlestone-labs/Anima@aesthetic-v1.1")["steps"] == 30


def test_preview_critic_gets_own_score_advice_and_labelled_inherited_sources():
    sources = prompt_sources.sources_for(MODEL_ID)
    assert sources.guide is guides.ANIMA_PREVIEW_GUIDE
    assert sources.url == "https://huggingface.co/Gazingstars123/Anima-2.9B"
    excerpts = {excerpt.id: excerpt for excerpt in sources.excerpts}
    assert "anima.preview_scores" in excerpts
    assert "anima.prefix" not in excerpts and "anima.aesthetic_scores" not in excerpts
    assert "inherited parent Anima" in excerpts["anima.preview_spelling"].note
    assert excerpts["anima.preview_parent_order"].order
    assert "does not enforce" in excerpts["anima.preview_parent_order"].note
    spelling = next(sentence for sentence in sources.sentences if "write tags in lowercase" in sentence.text)
    assert spelling.url == "https://huggingface.co/circlestone-labs/Anima"
    assert not any("This app uses" in sentence.text for sentence in sources.sentences)
