"""Each catalogue checkpoint reaches the writer with its complete prompt contract."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from backend import critic_context, critic_runtime, prompt_assistant, prompt_order
from backend.launch import CHECKPOINTS
from backend.prompt_sources import Excerpt, GuideSentence
from backend.schemas_assistant import PromptAssistantRequest


MODELS = [entry.repo for entry in CHECKPOINTS]
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
ANIMA = "circlestone-labs/Anima"
AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"
PREVIEW = "Gazingstars123/Anima-2.9B"


def context(model_id: str):
    return prompt_assistant.build_context(model_id, PromptAssistantRequest(
        ticket="guidance-review", mode="create", instruction="a fox in a forest",
    ))


@pytest.mark.parametrize("model_id", MODELS)
def test_every_catalogue_checkpoint_keeps_all_authored_guidance(model_id):
    sources = prompt_assistant.sources_for(model_id)
    system = critic_context._restored(context(model_id).system)
    assert "CHECKPOINT PROMPT CONTRACT" in system
    assert sources.guide_name != "FALLBACK_GUIDE"
    if sources.caveat:
        assert sources.caveat in system
    for sentence in sources.sentences:
        assert sentence.text in system
    for excerpt in sources.excerpts:
        assert excerpt.text in system
        if excerpt.note:
            assert excerpt.note in system


@pytest.mark.parametrize("model_id", MODELS)
def test_documented_order_is_explicit_without_inventing_a_total_sequence(model_id):
    system = critic_context._restored(context(model_id).system)
    order = prompt_order.documented_order(model_id)
    if order is None:
        assert "This app checks no fixed positive order" in system
        assert "Documented positive order:" not in system
    else:
        assert "Documented positive order: " + " > ".join(order.steps) in system
        assert order.cite in system


@pytest.mark.parametrize("model_id", MODELS)
def test_exact_checkpoint_recipes_reach_the_writer_as_recommendations(model_id):
    sources = prompt_assistant.sources_for(model_id)
    helper = sources.guide.get("tag_helper") or {}
    system = critic_context._restored(context(model_id).system)
    for key, label in (
        ("recommended_prefix", "Recommended positive prefix"),
        ("recommended_suffix", "Recommended positive suffix"),
        ("negative_prompt", "Recommended negative recipe"),
    ):
        value = helper.get(key)
        if value:
            text = ", ".join(value) if isinstance(value, list) else value
            assert f"{label}: {text}" in system


@pytest.mark.parametrize("model_id", [
    "KBlueLeaf/Kohaku-XL-Zeta",
    "OnomaAIResearch/Illustrious-XL-v2.0",
    "LyliaEngine/waiIllustriousSDXL_v170",
    "Minthy/RouWei-0.8",
    PREVIEW,
])
def test_examples_keep_their_exception_omission_and_inheritance_notes(model_id):
    sources = prompt_assistant.sources_for(model_id)
    system = critic_context._restored(context(model_id).system)
    notes = sources.guide.get("example_notes", {})
    assert notes
    for example, note in notes.items():
        if example in system:
            assert note in system
    assert any(example in system for example in notes)


def test_anima_variants_do_not_share_incompatible_quality_recipes():
    turbo = critic_context._restored(context(ANIMA).system)
    aesthetic = critic_context._restored(context(AESTHETIC).system)
    preview = critic_context._restored(context(PREVIEW).system)
    assert "Recommended positive prefix: masterpiece, best quality, score_7, safe" in turbo
    assert "At Turbo's recommended CFG 1, negative prompts have no effect" in turbo
    assert "Aesthetic does not need positive quality tags" in aesthetic
    assert "avoiding score_* tags in both the positive and negative prompts" in aesthetic
    assert "Recommended positive prefix: masterpiece, best quality, score_7, safe" not in aesthetic
    assert "Recommended positive prefix: masterpiece, best quality, score_7, safe" not in preview
    assert "reference sequence is informational" in preview
    assert "The training captions have no score tags" in preview
    assert "This app checks no fixed positive order" in preview


@pytest.mark.parametrize("model_id,expected", [
    ("black-forest-labs/FLUX.1-Krea-dev", "True CFG > 1"),
    ("black-forest-labs/FLUX.1-dev", "True CFG > 1"),
    ("black-forest-labs/FLUX.1-schnell", "True CFG > 1"),
    ("Qwen/Qwen-Image", "True CFG > 1"),
    ("krea/Krea-2-Turbo", "CFG > 0"),
    ("krea/Krea-2-Raw", "CFG > 0"),
    (ANIMAGINE, "CFG > 1"),
    (ANIMA, "CFG > 1"),
    (AESTHETIC, "CFG > 1"),
    (PREVIEW, "CFG > 1"),
    ("black-forest-labs/flux-3-image-klein-base", "CFG > 1"),
])
def test_negative_prompt_effect_is_conditional_on_the_correct_guidance_control(model_id, expected):
    system = critic_context._restored(context(model_id).system)
    contract = system.split("CHECKPOINT PROMPT CONTRACT", 1)[1].split("SUPPLIED CHECKPOINT GUIDANCE", 1)[0]
    assert expected in contract


@pytest.mark.parametrize("model_id", [
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.2-klein-base-4B",
    "black-forest-labs/FLUX.2-klein-base-9B",
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/FLUX.1-Kontext-dev",
])
def test_unsupported_negative_prompt_has_an_explicit_contract(model_id):
    system = critic_context._restored(context(model_id).system)
    contract = system.split("CHECKPOINT PROMPT CONTRACT", 1)[1].split("SUPPLIED CHECKPOINT GUIDANCE", 1)[0]
    assert "negative" in contract.lower()
    assert "no negative prompt" in contract.lower() or "does not support" in contract.lower()
    assert "CFG >" not in contract


def test_negative_input_stays_context_with_no_automatic_positive_migration():
    request = PromptAssistantRequest(
        ticket="guidance-review", mode="improve", instruction="sharpen the scene",
        prompt="1girl, forest", negative_prompt="extra people, watermark",
    )
    result = prompt_assistant.build_context(ANIMAGINE, request)
    data = json.loads(critic_context._restored(result.user))
    assert data["current_prompt"] == request.prompt
    assert data["negative_prompt_context"] == request.negative_prompt
    assert "never silently move negative text into a positive draft" in result.system
    assert "drafts" in result.schema["properties"]


def test_grown_guidance_cannot_silently_drop_a_later_rule(monkeypatch):
    sources = prompt_assistant.sources_for(ANIMAGINE)
    early = "An intentionally expanded author sentence " + "detail " * 700 + "."
    later = "Do not transplant another checkpoint's required tag sequence."
    final_excerpt = "Keep this final author-specific negative recipe intact."
    monkeypatch.setattr(prompt_assistant, "sources_for", lambda _: replace(
        sources,
        sentences=(GuideSentence(early, sources.url), GuideSentence(later, sources.url)),
        excerpts=(Excerpt("late.author-rule", final_excerpt, sources.url or "https://example.invalid"),),
    ))
    result = context(ANIMAGINE)
    system = critic_context._restored(result.system)
    assert early in system
    assert later in system
    assert final_excerpt in system
    assert system.index(early) < system.index(later) < system.index(final_excerpt)
    assert result.input_tokens >= critic_context.estimate_tokens(result.system)


def test_oversized_guidance_is_refused_by_context_budget_without_pruning_rules(monkeypatch):
    sources = prompt_assistant.sources_for(ANIMAGINE)
    enormous = "complete source clause " * 12000
    last_rule = "The last required clause must remain part of the question."
    monkeypatch.setattr(prompt_assistant, "sources_for", lambda _: replace(
        sources,
        sentences=(GuideSentence(enormous, sources.url), GuideSentence(last_rule, sources.url)),
    ))
    result = context(ANIMAGINE)
    assert last_rule in result.system and enormous in result.system
    with pytest.raises(critic_runtime.CriticError, match="too long"):
        prompt_assistant.check_fits(result)
