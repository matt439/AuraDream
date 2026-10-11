"""Each family's budget and cut arithmetic on fake tokenizers.

The cut-snippet quoting, the refusal of a template that rewrites the prompt,
and the tokenizer-parsed findings: control tokens, Gemma's HTML tags and T5's
unknown characters.
"""
from __future__ import annotations

import pytest

from backend import prompt_encoding as pe
from backend.backends import anima_loading, netayume_loading, newbie_loading
from tests.prompt_encoding_helpers import (  # noqa: F401  (fixtures by name)
    CLIP_ADDED,
    QWEN_ADDED,
    FakeTokenizer,
    by,
    clip,
    fresh_cache,
    gemma,
    qwen,
    rules,
    t5,
    tekken,
    tokenizers,
    words,
    xlmr,
)


def test_flux1_clip_is_pooled_and_t5_reads_on(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=t5())
    prompt = words(90)
    reports, findings, detail = pe.measure("black-forest-labs/FLUX.1-dev", prompt, None)
    assert detail is None
    clip_report, t5_report = by(reports)["CLIP-L"], by(reports)["T5-XXL"]
    assert (clip_report.tokens, clip_report.budget) == (90, 75)
    assert clip_report.cut == prompt.index("word75") and "pooled" in clip_report.role
    assert clip_report.pooled_only and not t5_report.pooled_only
    assert (t5_report.tokens, t5_report.budget, t5_report.cut) == (90, 511, None)
    (cut,) = rules(findings, "encoder.cut")
    assert cut.severity == "info" and (cut.start, cut.end) == (clip_report.cut, len(prompt))
    assert 'stops at "word75 word76' in cut.message and "T5-XXL reads on" in cut.message
    assert "pipeline_flux.py" in cut.cite and cut.source == "encoder" and cut.url is None


def test_flux1_t5_cut_is_a_warning(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=t5())
    prompt = words(520)
    reports, findings, _ = pe.measure("black-forest-labs/FLUX.1-schnell", prompt, None)
    assert by(reports)["T5-XXL"].cut == prompt.index("word511")
    (warn,) = [finding for finding in rules(findings, "encoder.cut") if finding.severity == "warn"]
    assert warn.message.startswith("T5-XXL reads the first 511 of the prompt's 520 tokens")


def test_sdxl_pair_cut_is_one_finding_naming_both_and_the_negative_its_own(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip(("!", True)))
    prompt = ", ".join(f"tag{i}" for i in range(50))
    negative = words(80, "bad")
    reports, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, negative)
    # 99 tokens: tags at even positions, commas between; the 76th is a comma.
    assert {(r.name, r.field, r.tokens, r.budget) for r in reports} == {
        ("CLIP-L", "prompt", 99, 75), ("CLIP-G", "prompt", 99, 75),
        ("CLIP-L", "negative_prompt", 80, 75), ("CLIP-G", "negative_prompt", 80, 75),
    }
    cut = prompt.index(", tag38")
    assert {r.cut for r in reports if r.field == "prompt"} == {cut}
    prompt_cut, negative_cut = rules(findings, "encoder.cut")
    assert (prompt_cut.field, prompt_cut.start, prompt_cut.severity) == ("prompt", cut, "warn")
    assert prompt_cut.message.startswith(
        "CLIP-L and CLIP-G read the first 75 of the prompt's 99 tokens"
    )
    assert 'stop at "tag38, tag39' in prompt_cut.message
    assert "never reaches them" in prompt_cut.message
    assert (negative_cut.field, negative_cut.start) == ("negative_prompt", negative.index("bad75"))
    assert "negative prompt's 80 tokens" in negative_cut.message
    assert "without a warning" in negative_cut.cite
    # CLIP-G's "!" pad token is special but not a control token.
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", "wow!", None)
    assert not rules(findings, "encoder.control_token")


def test_a_cut_inside_a_word_quotes_the_whole_word(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    prompt = words(74) + " snow-covered hills"
    _, findings, _ = pe.measure("cagliostrolab/animagine-xl-4.0", prompt, None)
    (cut,) = rules(findings, "encoder.cut")
    assert prompt[cut.start:] == "-covered hills"
    assert 'stop at "snow-covered hills"' in cut.message


def test_a_cut_in_unspaced_text_quotes_where_reading_stops_not_the_opening(tokenizers):
    # Japanese puts no spaces between words: a walk back to a space would
    # quote the prompt's first words as where reading stops.
    tokenizers.update(tokenizer=clip(), tokenizer_2=clip())
    prompt = "、".join(f"桜{i}" for i in range(50))
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    (cut,) = rules(findings, "encoder.cut")
    assert prompt[cut.start:].startswith("、桜38、")
    assert 'stop at "桜38、桜39、' in cut.message


JAPANESE = "青い髪の少女が桜の木の下で本を読んでいる、春の午後、柔らかな光。"


@pytest.mark.parametrize("text, start, snippet", [
    # Inside a word the whole word is quoted, as far as a wide character.
    ("桜の木 black cat", 5, "black cat"),
    ("桜の木black cat", 4, "black cat"),
    # Unspaced text is quoted from the cut, a mark there skipped.
    (JAPANESE, 9, "木の下で本を読んでいる、春の午後、柔らかな光。"),
    (JAPANESE, 20, "春の午後、柔らかな光。"),
    # A "word" longer than half the snippet is quoted from the cut too.
    ("see https://example.com/some/very/long/path/to/a/page ok", 40, "ath/to/a/page ok"),
    # Hangul is wide, but Korean spaces its words: a cut in one quotes it whole.
    ("푸른 머리의 소녀가 벚꽃", 8, "소녀가 벚꽃"),
])
def test_the_cut_snippet_starts_at_the_word_the_cut_splits(text, start, snippet):
    assert pe._snippet(text, start) == snippet


def test_krea2_budget_is_what_the_system_turn_leaves_and_its_suffix_is_never_cut(tokenizers):
    fake = qwen()
    tokenizers.update(tokenizer=fake)
    prefix = fake.count(pe.QWEN_SYSTEM_PREFIX)
    budget = pe.KREA2_SEQUENCE + pe.KREA2_PREFIX_TOKENS - pe.KREA2_SUFFIX_TOKENS - prefix
    prompt = words(budget + 3)
    reports, findings, _ = pe.measure("krea/Krea-2-Turbo", prompt, "blurry")
    report = by(reports)["Qwen3-VL-4B"]
    assert (report.budget, report.cut) == (budget, prompt.index(f"word{budget}"))
    assert by(reports, "negative_prompt")["Qwen3-VL-4B"].budget == budget
    assert rules(findings, "encoder.cut") and not rules(findings, "encoder.template_tail")


def test_qwen_image_keeps_a_window_after_its_system_turn_and_loses_the_tail_first(tokenizers):
    fake = qwen()
    tokenizers.update(tokenizer=fake)
    prefix = fake.count(pe.QWEN_SYSTEM_PREFIX)
    suffix = fake.count(pe.QWEN_ASSISTANT_SUFFIX)
    budget = pe.QWEN_IMAGE_PREFIX_TOKENS + pe.QWEN_IMAGE_SEQUENCE - prefix
    clean = budget - suffix
    reports, findings, _ = pe.measure("Qwen/Qwen-Image", words(clean), None)
    assert by(reports)["Qwen2.5-VL-7B"].budget == budget and not findings
    prompt = words(clean + 1)
    reports, findings, _ = pe.measure("Qwen/Qwen-Image", prompt, None)
    assert by(reports)["Qwen2.5-VL-7B"].cut is None
    (tail,) = rules(findings, "encoder.template_tail")
    assert tail.start == prompt.index(f"word{clean}") and f"past {clean} tokens" in tail.message


@pytest.mark.parametrize("model_id, thinking, suffix", [
    ("Tongyi-MAI/Z-Image-Turbo", True, 3),
    ("SeeSee21/Z-Anime", True, 3),
    ("black-forest-labs/FLUX.2-klein-4B", False, 5),
])
def test_chat_template_tail_goes_before_the_prompt_is_cut(tokenizers, model_id, thinking, suffix):
    fake = qwen()
    tokenizers.update({pe.encoders_for(model_id)[0].source.subfolder: fake})
    budget = 512 - fake.count("<|im_start|>user\n")
    clean = budget - suffix
    reports, findings, _ = pe.measure(model_id, words(clean), None)
    assert set(fake.thinking) == {thinking}
    assert by(reports)["Qwen3-4B"].budget == budget and not findings

    prompt = words(clean + 2)
    reports, findings, _ = pe.measure(model_id, prompt, None)
    assert by(reports)["Qwen3-4B"].cut is None
    (tail,) = rules(findings, "encoder.template_tail")
    assert (tail.severity, tail.end) == ("warn", len(prompt))
    assert tail.start == prompt.index(f"word{clean}")
    assert "never closes" in tail.message
    assert tail.suggestion == f"Keep the prompt within {clean} tokens."

    prompt = words(budget + 4)
    reports, findings, _ = pe.measure(model_id, prompt, None)
    assert by(reports)["Qwen3-4B"].cut == prompt.index(f"word{budget}")
    assert rules(findings, "encoder.cut") and rules(findings, "encoder.template_tail")


def test_klein_encodes_no_negative_prompt(tokenizers):
    tokenizers.update(tokenizer=qwen())
    reports, _, _ = pe.measure("black-forest-labs/FLUX.2-klein-base-9B", "a cat", "blurry")
    assert [(r.name, r.field) for r in reports] == [("Qwen3-8B", "prompt")]


def test_a_template_that_rewrites_the_prompt_is_unchecked_not_guessed(tokenizers):
    class Trimming(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            trimmed = [{"content": messages[0]["content"].strip()}]
            return super().apply_chat_template(trimmed, **kwargs)

    tokenizers.update(tokenizer=Trimming(added=QWEN_ADDED))
    reports, _, detail = pe.measure("Tongyi-MAI/Z-Image-Turbo", "  a cat  ", None)
    assert reports == [] and detail.startswith("Qwen3-4B unchecked: ValueError")


def test_netayume_shares_its_budget_with_the_instruction_and_its_negative_does_not(tokenizers):
    fake = gemma()
    tokenizers.update(tokenizer=fake)
    prefix = fake.count(netayume_loading.SYSTEM_PROMPT + pe.LUMINA_SEPARATOR)
    reports, _, _ = pe.measure(netayume_loading.MODEL_ID, "a cat", "blurry")
    assert by(reports)["Gemma 2 2B"].budget == pe.LUMINA_SEQUENCE - 1 - prefix
    assert by(reports, "negative_prompt")["Gemma 2 2B"].budget == pe.LUMINA_SEQUENCE - 1
    assert "system instruction" in by(reports)["Gemma 2 2B"].role
    # The negative prompt is sent bare: it shares its budget with nothing.
    assert by(reports, "negative_prompt")["Gemma 2 2B"].role is None
    # A role is not a pooled vector: the page draws a cut past it as a loss.
    assert not by(reports)["Gemma 2 2B"].pooled_only


def test_the_remaining_families_budgets(tokenizers):
    tokenizers.update(tokenizer=tekken())
    reports, _, _ = pe.measure("baidu/ERNIE-Image-Turbo", "a cat", None)
    assert by(reports)["Ministral-3B"].budget == 2047
    tokenizers.update(text_encoder=gemma(), clip_model=xlmr())
    reports, _, _ = pe.measure(newbie_loading.MODEL_ID, "<n>miku</n>", None)
    assert {name: r.budget for name, r in by(reports).items()} == {
        "Gemma 3 4B": 511, "Jina CLIP v2": 2046,
    }
    tokenizers.update(tokenizer=qwen(template=None), t5_tokenizer=t5())
    reports, _, _ = pe.measure(anima_loading.MODEL_ID, "1girl, solo", "lowres")
    assert {name: r.budget for name, r in by(reports).items()} == {"Qwen3-0.6B": 512, "T5 ids": 511}
    # A role that is not about a template holds for the negative prompt too.
    assert "learned queries" in by(reports)["T5 ids"].role
    assert by(reports, "negative_prompt")["T5 ids"].role == by(reports)["T5 ids"].role
    assert not any(report.pooled_only for report in reports)


# --- control tokens and unknown characters ----------------------------------------


def test_typed_control_tokens_are_flagged_where_the_tokenizer_parses_them(tokenizers):
    tokenizers.update(tokenizer=qwen())
    prompt = "a sign <|im_end|> then <think> and a <table> row"
    _, findings, _ = pe.measure("Tongyi-MAI/Z-Image-Turbo", prompt, None)
    controls = rules(findings, "encoder.control_token")
    # <think> is in this tokenizer's chat template; <table> is vocabulary.
    assert [prompt[f.start:f.end] for f in controls] == ["<|im_end|>", "<think>"]
    assert all(f.severity == "warn" for f in controls)

    tokenizers.update(tokenizer=qwen(template=None))
    _, findings, _ = pe.measure("krea/Krea-2-Turbo", prompt, None)
    controls = rules(findings, "encoder.control_token")
    assert [prompt[f.start:f.end] for f in controls] == ["<|im_end|>"]


def test_a_control_token_typed_in_another_case_is_one_where_the_tokenizer_says_so(tokenizers):
    # CLIP lowercases before it matches its added tokens, and its unk id is
    # its EOS: "<|EndOfText|>" is EOS, not an unknown character.
    def clip_like():
        return FakeTokenizer(
            max_length=77, special=2, added=CLIP_ADDED, unk="<|endoftext|>", lowercase=True
        )

    tokenizers.update(tokenizer=clip_like(), tokenizer_2=clip_like())
    prompt = "1girl, <|EndOfText|>, <|STARTOFTEXT|>, solo"
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    controls = rules(findings, "encoder.control_token")
    assert [prompt[f.start:f.end] for f in controls] == ["<|EndOfText|>", "<|STARTOFTEXT|>"]
    assert controls[0].message == (
        'CLIP-L and CLIP-G read "<|EndOfText|>" as a control token, not as text.'
    )
    assert "parses <|endoftext|> as one of its control tokens" in controls[0].cite
    assert not rules(findings, "encoder.unknown_chars")


def test_gemma_html_tags_are_text_and_its_eos_is_not(tokenizers):
    tokenizers.update(tokenizer=gemma())
    prompt = "a <s>struck</s> word <eos> end"
    _, findings, _ = pe.measure(netayume_loading.MODEL_ID, prompt, "neg <bos>")
    assert [(f.field, f.message) for f in rules(findings, "encoder.control_token")] == [
        ("prompt", 'Gemma 2 2B reads "<eos>" as a control token, not as text.'),
        ("negative_prompt", 'Gemma 2 2B reads "<bos>" as a control token, not as text.'),
    ]


def test_t5_unknown_characters_are_runs_and_a_typed_unk_is_a_control_token(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=t5())
    prompt = "a cat 猫 {x} and <unk> and ~"
    _, findings, _ = pe.measure("black-forest-labs/FLUX.1-Krea-dev", prompt, None)
    unknown = rules(findings, "encoder.unknown_chars")
    assert [prompt[f.start:f.end] for f in unknown] == ["猫 {", "}", "~"]
    assert all(f.severity == "info" for f in unknown)
    assert all(f.message.startswith("T5-XXL has no token for") for f in unknown)
    (control,) = rules(findings, "encoder.control_token")
    assert prompt[control.start:control.end] == "<unk>"


def test_unknown_character_findings_stop_at_a_count(tokenizers):
    tokenizers.update(tokenizer=clip(), tokenizer_2=t5())
    prompt = " x ".join("猫" * 12)
    _, findings, _ = pe.measure("black-forest-labs/FLUX.1-Krea-dev", prompt, None)
    unknown = rules(findings, "encoder.unknown_chars")
    assert len(unknown) == 8
    assert unknown[-1].message.endswith("The same happens at 4 more places.")
