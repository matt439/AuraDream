"""Critic questions carry applied requests, sources, and output contracts."""
from __future__ import annotations

import pytest

from backend import critic_context, critic_runtime, prompt_sources
from backend.schemas import DocumentedOrder, GenerateRequest, PromptAnalysis, PromptFinding

from tests.critic_helpers import (
    ANIMAGINE,
    KLEIN,
    KREA_DEV,
    PROMPT,
    Server,
    _analysis,
    _catalogue_repos,
    _context,
    prepared,
)


ANIMA = "circlestone-labs/Anima"


# --- The question ------------------------------------------------------------


def test_schema_enumerates_exactly_the_numbered_sources():
    context = _context()
    ids = context.schema["properties"]["issues"]["items"]["properties"]["cite_id"]["enum"]
    assert ids == list(context.citations)
    sources = prompt_sources.sources_for(KLEIN)
    assert ids == [f"G{i}" for i in range(1, len(sources.sentences) + 1)] + [
        f"A{i}" for i in range(1, len(sources.excerpts) + 1)
    ]
    for cite_id in ids:
        assert f"\n{cite_id}: " in context.system
    assert context.schema["required"] == ["parts", "issues", "rewrite", "reordered"]
    assert context.schema["additionalProperties"] is False
    assert context.schema["properties"]["issues"]["maxItems"] == critic_context.MAX_ISSUES
    assert context.schema["properties"]["parts"]["required"] == list(critic_context.PARTS)
    assert context.schema["properties"]["reordered"] == {"type": ["string", "null"]}
    assert _context(ordered=False).schema["properties"]["reordered"] == {"type": "null"}


def test_the_messages_carry_the_request_as_it_runs():
    runs = _context(ANIMA, "1girl, smile", ordered=False, negative="lowres")
    assert "Negative prompt (it runs):\nlowres" in runs.user
    assert "No negative prompt runs" not in runs.system
    assert "reordered: null" in runs.system and "DOCUMENTED ORDER" not in runs.system
    # Anima's card does give a tag order - as A1 - and nothing may say it
    # does not: the app has no check of its own, the authors have an order.
    assert "document no order" not in runs.system
    assert "An order a source below states may still be an issue" in runs.system
    assert "\nA1: [quality/meta/year/safety tags] [1girl/1boy/1other etc]" in runs.system
    silent = _context(KLEIN)
    assert silent.user == f"Prompt:\n{PROMPT}"
    assert "No negative prompt runs" in silent.system
    assert "DOCUMENTED ORDER: subject > action > style > context" in silent.system
    # Quotation marks are asked for where the guide asks for them, not on a
    # tag checkpoint whose guide never mentions rendered text - and only for
    # lettering: the card's critic wrapped whole prompts on "text in quotes".
    lettering = "Put quotation marks only around words the image must show as lettering"
    assert lettering in silent.system and "never around the prompt" in silent.system
    assert lettering not in runs.system
    # The rewrite adds nothing and pads nothing; the examples lend their form.
    for context in (runs, silent):
        assert "Add no object, scenery, mood or text the prompt lacks" in context.system
        assert "not a minimum" in context.system
    assert "AUTHORS' EXAMPLE PROMPTS (their form only; never copy their subjects or text)" in (
        silent.system
    )


def test_the_critic_accepts_generic_subjects_and_can_leave_a_good_prompt_alone():
    for model_id, prompt, style in (
        (ANIMAGINE, "1girl, green hair, smile", "tags"),
        ("OnomaAIResearch/Illustrious-XL-v2.0", "1boy, doctor, masterpiece", "tags"),
        (ANIMA, "1girl, green hair, A girl waves from a balcony.", "hybrid"),
        (KLEIN, "A girl with green hair smiles.", "prose"),
    ):
        context = _context(model_id, prompt, style=style)
        assert "Order lists place present parts" in context.system
        assert "Generic/original subjects need no character/series names" in context.system
        assert "never invent them" in context.system
        assert "Return it unchanged if no edit is needed" in context.system
        assert ("Keep tags' backslash escapes" in context.system) == (
            style in ("tags", "hybrid")
        )


def test_the_critic_is_told_the_negative_prompt_applied_runs_not_the_one_typed():
    # FLUX.1 without True CFG: applied() drops the negative the user typed,
    # and the critic must hear that none runs - in both messages and in what
    # its window is sized for.
    model_id, prompt = "black-forest-labs/FLUX.1-Krea-dev", "A quiet street with no people"
    typed = GenerateRequest(prompt=prompt, negative_prompt="people, crowds")
    bare = GenerateRequest(prompt=prompt)
    analysis = _analysis(model_id, ordered=False)
    dropped = critic_context.build_context(model_id, typed, bare, analysis)
    assert "(it runs)" not in dropped.user and "people, crowds" not in dropped.user
    assert "No negative prompt runs" in dropped.system
    assert dropped.input_tokens == critic_context.build_context(model_id, bare, bare, analysis).input_tokens
    # And the other way: what applied() runs is what the critic is told.
    runs = critic_context.build_context(model_id, bare, typed, analysis)
    assert "Negative prompt (it runs):\npeople, crowds" in runs.user
    assert "No negative prompt runs" not in runs.system
    # A long negative that does not run costs a long prompt no room.
    prompt = "a lighthouse on a cliff at dusk, waves below, " * 80
    long = GenerateRequest(prompt=prompt, negative_prompt="人群，模糊，" * 650)
    critic_context.check_fits(critic_context.build_context(model_id, long, GenerateRequest(prompt=prompt), analysis))
    with pytest.raises(critic_runtime.CriticError, match="too long for the critic"):
        critic_context.check_fits(critic_context.build_context(model_id, long, long, analysis))


def test_the_real_critic_asks_about_the_negative_prompt_applied_runs(monkeypatch, prepared):
    # PromptCritic.critique hands on what applied() runs, not what was typed:
    # passing the request twice told llama-server "Negative prompt (it
    # runs)" on Krea [dev] without True CFG, and no test above noticed.
    typed = GenerateRequest(prompt="A quiet street with no people", negative_prompt="people, crowds")
    analysis = _analysis(KREA_DEV, ordered=False)
    for true_cfg, runs in ((None, False), (3.0, True)):
        request = typed.model_copy(update={"true_cfg": true_cfg})
        applied = request.applied(
            negative_prompt_supported=True, cfg_threshold=1.0, true_cfg_supported=True
        )
        assert (applied.negative_prompt is not None) is runs
        server = Server(monkeypatch)
        prepared.critique(KREA_DEV, request, applied, analysis)
        [body] = [body for _, path, _, body in server.calls if path == "/v1/chat/completions"]
        system, user = (message["content"] for message in body["messages"])
        assert ("Negative prompt (it runs):\npeople, crowds" in user) is runs, true_cfg
        assert ("people, crowds" in user) is runs, true_cfg
        assert ("No negative prompt runs" in system) is not runs, true_cfg


def test_the_flux1_family_is_told_how_bfl_reads_its_order():
    # On the card a critic held to a bare "subject first" called Krea's own
    # "Moody night photography, a small store..." out of order, told the same
    # prompt to put its style first, and cut camera details as a fault.
    for model_id in ("black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-dev",
                     "black-forest-labs/FLUX.1-schnell"):
        context = _context(model_id)
        assert "'Moody night photography') still puts the subject first" in context.system
        assert "Camera settings are part of BFL's template, not a fault" in context.system
        texts = {citation.text for citation in context.citations.values()}
        assert "Start with the image type and subject." in texts
        assert any("not a strict formula" in text and "[CAMERA SETTINGS]" in text for text in texts)


def test_krea_dev_is_told_its_sources_are_bfls():
    context = _context("black-forest-labs/FLUX.1-Krea-dev")
    assert "Krea published no prompting rules" in context.system
    # A shared note is written once and pointed back to.
    note = "(BFL's guide for the whole FLUX family; Krea published no prompting rules of its own)"
    assert context.system.count(note) == 1
    assert "(as A1)" in context.system


@pytest.mark.parametrize("model_id", sorted(set(_catalogue_repos())))
@pytest.mark.parametrize("style", ["prose", "tags"])
def test_the_system_message_fits_its_budget(model_id, style):
    findings = [
        PromptFinding(rule="encoder.cut", severity="warn", message="m" * 400, source="encoder")
        for _ in range(20)
    ]
    request = GenerateRequest(prompt="a cat", negative_prompt="blurry")
    analysis = PromptAnalysis(
        model_id=model_id,
        style=style,
        words=2,
        findings=findings,
        order=DocumentedOrder(steps=["a", "b", "c", "d", "e"], cite="c" * 400),
    )
    context = critic_context.build_context(model_id, request, request, analysis)
    assert len(context.system) <= critic_context.SYSTEM_BUDGET_CHARS, len(context.system)
    assert context.system.count("\n- encoder.cut") == critic_context._MAX_FINDINGS
