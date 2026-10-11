"""Literal weighting-syntax findings, the same for any checkpoint.

Their spans, messages and suggestions; a deeply nested prompt is measured,
not raised; a failing syntax pass costs only its own findings; and an
unknown checkpoint still gets them without a tokenizer load.
"""
from __future__ import annotations

import re

import pytest

from backend import prompt_encoding as pe
from backend import prompt_weighting
from tests.prompt_encoding_helpers import (  # noqa: F401  (fixtures by name)
    FAILURES,
    _kind,
    clip,
    fresh_cache,
    rules,
    tokenizers,
)


@pytest.mark.parametrize("prompt, flagged", [
    ("(masterpiece:1.3), 1girl", ["(masterpiece:1.3)"]),
    ("((best quality)), 1girl", ["((best quality))"]),
    ("((masterpiece:1.2)), solo", ["((masterpiece:1.2))"]),
    ("1girl, [blurry], solo", ["[blurry]"]),
    ("a [cat:dog:0.5] on a mat", ["[cat:dog:0.5]"]),
    ("a cat BREAK a dog", ["BREAK"]),
    ("a cat, <lora:foo:0.8>", ["<lora:foo:0.8>"]),
    ("a cat, embedding:bad_hands", ["embedding:bad_hands"]),
    ("a {red|blue} dress", ["{red|blue}"]),
    ("(1girl:1.1), (score_9:1.2)", ["(1girl:1.1)", "(score_9:1.2)"]),
    # Escapes are the autocomplete's own output and the authors' spelling.
    (r"firefly \(honkai: star rail\), 1girl", []),
    (r"\(masterpiece:1.2\)", []),
    # A ratio, a time, a placeholder in prose, a break in lower case.
    ("widescreen (16:9), at dusk (at 3:30)", []),
    ("Detailed anime portrait of [character], soft rim lighting", []),
    ("take a break", []),
    # Text meant to be rendered, and a JSON prompt's own structure.
    ('a neon sign that says "[OPEN] BREAK (24:7)"', []),
    ('{"subject": "a fox", "tags": ["snow", "dusk"], "mood": {"a|b": 1}}', []),
    ("a {name} in the snow", []),
])
def test_weighting_syntax_is_spanned_where_it_is_syntax(tokenizers, prompt, flagged):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    weighting = rules(findings, "encoder.weighting")
    assert [prompt[f.start:f.end] for f in weighting] == flagged
    assert all(f.severity == "warn" and f.source == "encoder" for f in weighting)


def test_weighting_messages_say_what_the_encoder_reads(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    negative = r"(worst quality:1.4), (bad \(anatomy\):1.2)"
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", "red hair", negative)
    first, second = rules(findings, "encoder.weighting")
    assert first.field == second.field == "negative_prompt"
    assert "reads the brackets, the colon and the number as text" in first.message
    assert first.suggestion == 'Write "worst quality" plainly; no pipeline here reads the number.'
    assert second.suggestion == 'Write "bad (anatomy)" plainly; no pipeline here reads the number.'
    assert first.cite.startswith("diffusers has no prompt-weighting parser")
    assert "pipeline_stable_diffusion_xl.py" in first.cite


@pytest.mark.parametrize("prompt, suggestion", [
    # Nested emphasis around a weight: the words, never the weight.
    ("((masterpiece:1.2)), solo",
     'Write "masterpiece" plainly; no pipeline here reads the number.'),
    ("((best quality)), 1girl", 'Write "best quality" plainly.'),
    # Prompt editing and alternation: the wordings, never the colons or pipes.
    ("a [cat:dog:0.5] on a mat",
     "Pick \"cat\" or \"dog\", or compare them on a grid's prompt axis."),
    ("a [dog:0.5] on a mat", 'Write "dog" plainly, or leave it out.'),
    ("a [cat::10] on a mat", 'Write "cat" plainly, or leave it out.'),
    ("a [cat|dog] on a mat",
     "Pick \"cat\" or \"dog\", or compare them on a grid's prompt axis."),
    ("a {red|blue|green} dress",
     "Pick \"red\", \"blue\" or \"green\", or compare them on a grid's prompt axis."),
    ("1girl, [[blurry:0.8]]", 'Write "blurry" plainly, or leave it out.'),
    ("1girl, [[blurry]], solo", 'Write "blurry" plainly.'),
    # One bracket around a whole item may be a template slot (Z-Anime's
    # `[character]`, which prompt_rules names as the author's): the advice
    # must not tell the user to write the slot's name out.
    ("[character], 1girl",
     'Write "character" plainly - or, if it is a template slot, put what it stands for there.'),
])
def test_weighting_suggestions_name_the_words_not_the_syntax(tokenizers, prompt, suggestion):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    (finding,) = rules(findings, "encoder.weighting")
    assert finding.suggestion == suggestion
    quoted = re.findall(r'"([^"]*)"', finding.suggestion)
    assert quoted and not any(re.search(r"[\[\]{}()|]|:\s*[\d.]", q) for q in quoted)


@pytest.mark.parametrize("prompt, message", [
    ("((masterpiece:1.2))", "the brackets, the colon and the number as text"),
    ("a [cat:dog:0.5]", "prompt-editing syntax"),
    ("a [dog:0.5]", "the brackets, the colon and the number as text"),
    ("a [cat|dog]", "alternation syntax"),
    ("a {red|blue|green} dress", "the braces and the pipes as text"),
    ("1girl, [blurry]", "de-emphasis syntax"),
])
def test_weighting_messages_name_the_construct(tokenizers, prompt, message):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    (finding,) = rules(findings, "encoder.weighting")
    assert message in finding.message


@pytest.mark.parametrize("model_id", ["KBlueLeaf/Kohaku-XL-Zeta", "someone/unknown-model"])
@pytest.mark.parametrize("prompt", [
    # The schema's 4000-character ceiling, nested deeper than json recurses.
    "[" * 3994 + " 1girl",
    "[" * 4000,
    "{" * 4000,
    "(" * 4000,
])
def test_a_deeply_nested_prompt_is_measured_not_raised(tokenizers, model_id, prompt):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    assert not prompt_weighting._is_json(prompt)
    reports, findings, detail = pe.measure(model_id, prompt, prompt)
    assert all(f.rule.startswith("encoder.") for f in findings)
    if model_id == "KBlueLeaf/Kohaku-XL-Zeta":
        assert reports and detail is None


@pytest.mark.parametrize("error", [RuntimeError("boom"), *FAILURES], ids=_kind)
def test_a_failing_syntax_pass_costs_its_findings_not_the_analysis(
    tokenizers, monkeypatch, error
):
    def broken(text):
        raise error

    monkeypatch.setattr(prompt_weighting, "_weighting_spans", broken)
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    reports, findings, detail = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", "(a:1.2), cat", None)
    assert [r.name for r in reports] == ["CLIP-L", "CLIP-G"]
    assert rules(findings, "encoder.weighting") == []
    assert detail == f"Weighting syntax unchecked: {_kind(error)}"
    _, _, detail = pe.measure("someone/unknown-model", "(a:1.2), cat", None)
    assert detail == (
        "Token budgets unchecked: no encoder table entry for this checkpoint; "
        f"Weighting syntax unchecked: {_kind(error)}."
    )


def test_an_unknown_checkpoint_is_unchecked_but_still_gets_the_syntax_findings(tokenizers):
    reports, findings, detail = pe.measure("someone/unknown-model", "a cat BREAK a dog", None)
    assert reports == []
    assert detail == "Token budgets unchecked: no encoder table entry for this checkpoint."
    assert [f.rule for f in findings] == ["encoder.weighting"]
    assert tokenizers["loads"] == []
