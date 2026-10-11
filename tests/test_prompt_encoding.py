"""The encoder budgets, cut points and literal-syntax findings, on fake tokenizers.

Nothing here downloads or loads a real tokenizer unless one is already in the
Hub cache (the last tests, which skip otherwise). The fakes split words and
punctuation marks with offsets, which is enough to check the arithmetic: what
a template costs, what the pipeline's truncation keeps, and where in the
user's own text reading stops. A stub load that never returns stands for a
stalled first-use fetch: what it may hold up - nothing but its own count -
is checked here, and through the app's Unload and Check routes in
tests/test_api_analysis.py.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib.util
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from jinja2 import TemplateError

from backend import launch, model_defaults
from backend import prompt_encoding as pe
from backend.backends import anima_loading, loading, netayume_loading, newbie_loading, single_file
from tests import fakes

ROOT = Path(__file__).resolve().parents[1]


class Added(NamedTuple):
    content: str
    special: bool


class FakeTokenizer:
    """Words and single punctuation marks as tokens, with offsets.

    Added tokens are split out of typed text whole, as the real tokenizers
    do; characters in `unknown` map to the unk id. `lowercase` is CLIP's
    normalizer, which runs before the added tokens are matched.
    """

    def __init__(
        self, *, max_length=77, special=0, added=(), unknown="", template=None, unk="<unk>",
        lowercase=False,
    ):
        self.model_max_length = max_length
        self.special = special
        self.chat_template = template
        self.added_tokens_decoder = {100 + i: Added(*item) for i, item in enumerate(added)}
        self._ids = {added.content: i for i, added in self.added_tokens_decoder.items()}
        self.all_special_ids = [
            i for i, added in self.added_tokens_decoder.items() if added.special
        ]
        self.unk_token = unk
        self.unk_token_id = self._ids.get(unk, 3)
        self._lowercase = lowercase
        if lowercase:
            normalizer = SimpleNamespace(normalize_str=str.lower)
            self.backend_tokenizer = SimpleNamespace(normalizer=normalizer)
        self._unknown = set(unknown)
        self._vocab: dict[str, int] = {}
        parts = [re.escape(content) for content in sorted(self._ids, key=len, reverse=True)]
        if unknown:
            parts.append(f"[{re.escape(unknown)}]")
        self._pattern = re.compile("|".join([*parts, r"\w+", r"[^\w\s]"]))
        self.thinking: list[bool] = []

    def num_special_tokens_to_add(self):
        return self.special

    def convert_ids_to_tokens(self, ids):
        names = {i: added.content for i, added in self.added_tokens_decoder.items()}
        return [names.get(i, f"t{i}") for i in ids]

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        assert add_special_tokens is False and return_offsets_mapping
        ids, offsets = [], []
        for match in self._pattern.finditer(text.lower() if self._lowercase else text):
            piece = match.group()
            if piece in self._ids:
                ids.append(self._ids[piece])
            elif piece in self._unknown:
                ids.append(self.unk_token_id)
            else:
                ids.append(self._vocab.setdefault(piece, 1000 + len(self._vocab)))
            offsets.append(match.span())
        return {"input_ids": ids, "offset_mapping": offsets}

    def count(self, text):
        return len(self(text, add_special_tokens=False, return_offsets_mapping=True)["input_ids"])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        # Qwen3's template for one user turn, with and without the empty
        # thinking block klein's `enable_thinking=False` renders.
        assert tokenize is False and add_generation_prompt is True
        self.thinking.append(enable_thinking)
        (message,) = messages
        text = f"<|im_start|>user\n{message['content']}<|im_end|>\n<|im_start|>assistant\n"
        return text + ("<think>\n\n</think>\n\n" if enable_thinking is False else "")


QWEN_ADDED = (
    ("<|endoftext|>", True), ("<|im_start|>", True), ("<|im_end|>", True),
    ("<think>", False), ("</think>", False), ("<table>", False),
)
QWEN_TEMPLATE = "{{ '<|im_start|>' }} ... {{ '<think>' }} ... {{ '</think>' }}"
CLIP_ADDED = (("<|startoftext|>", True), ("<|endoftext|>", True))


def clip(*extra):
    return FakeTokenizer(max_length=77, special=2, added=(*CLIP_ADDED, *extra))


def t5(unknown="猫{}~"):
    added = (("<pad>", True), ("</s>", True), ("<unk>", True), ("<extra_id_0>", True))
    return FakeTokenizer(max_length=512, special=1, added=added, unknown=unknown)


def qwen(template=QWEN_TEMPLATE):
    return FakeTokenizer(max_length=131072, added=QWEN_ADDED, template=template)


def gemma():
    added = (("<pad>", True), ("<eos>", True), ("<bos>", True), ("<s>", False), ("</s>", False))
    return FakeTokenizer(max_length=10**30, special=1, added=added)


def xlmr():
    added = (("<s>", True), ("</s>", True), ("<mask>", True))
    return FakeTokenizer(max_length=8194, special=2, added=added)


def tekken():
    added = (("<s>", True), ("</s>", True), ("[INST]", True))
    return FakeTokenizer(max_length=2048, special=1, added=added)


@pytest.fixture(autouse=True)
def fresh_cache():
    pe.unload()
    yield
    pe.unload()


@pytest.fixture
def tokenizers(monkeypatch):
    """Install fakes by tokenizer subfolder; a value that is an exception is
    raised by the load instead. `loads` records every load."""
    table: dict[str, object] = {}
    loads: list[pe.TokenizerSource] = []

    def load(source):
        loads.append(source)
        value = table[source.subfolder]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(pe, "_load_tokenizer", load)
    table["loads"] = loads
    return table


def words(count, word="word"):
    return " ".join(f"{word}{i}" for i in range(count))


def by(reports, field="prompt"):
    return {report.name: report for report in reports if report.field == field}


def rules(findings, rule):
    return [finding for finding in findings if finding.rule == rule]


def names(model_id):
    return [spec.name for spec in pe.encoders_for(model_id)]


# --- the table ------------------------------------------------------------------


def _catalogue_ids():
    for checkpoint in launch.CHECKPOINTS:
        yield checkpoint.repo
        for fmt, _ in launch.FORMATS:
            if checkpoint.lite_repo(fmt):
                yield checkpoint.lite_repo(fmt)


@pytest.mark.parametrize("model_id", list(_catalogue_ids()))
def test_every_catalogue_checkpoint_resolves_to_its_encoders(model_id):
    specs = pe.encoders_for(model_id)
    assert specs, f"{model_id} has no encoder entry"
    for spec in specs:
        assert spec.name and spec.source.repo and spec.source.subfolder
        assert spec.cite.endswith(f"({spec.file})")
    assert len({spec.name for spec in specs}) == len(specs)


def test_ordinary_repos_read_their_own_tokenizers_and_pinned_releases_their_pins():
    # A lite-infer re-upload carries its own tokenizer folders.
    lite = "lite-infer/flux.1-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"
    assert [(s.name, s.source.repo, s.source.subfolder) for s in pe.encoders_for(lite)] == [
        ("CLIP-L", lite, "tokenizer"), ("T5-XXL", lite, "tokenizer_2"),
    ]
    sdxl = single_file.SDXL_CONFIG
    for model_id in (single_file.ILLUSTRIOUS_ID, single_file.WAI_ID, "Minthy/RouWei-0.8"):
        sources = [spec.source for spec in pe.encoders_for(model_id)]
        assert [(s.repo, s.subfolder, s.revision, s.tree) for s in sources] == [
            (sdxl["repo_id"], "tokenizer", sdxl["revision"], model_id),
            (sdxl["repo_id"], "tokenizer_2", sdxl["revision"], model_id),
        ]
    (z_anime,) = pe.encoders_for("SeeSee21/Z-Anime")
    revision, folder = loading.NESTED_MODELS["SeeSee21/Z-Anime"]
    assert (z_anime.source.subfolder, z_anime.source.revision) == (f"{folder}/tokenizer", revision)
    (neta,) = pe.encoders_for(netayume_loading.MODEL_ID)
    shared = netayume_loading.SHARED
    assert (neta.source.repo, neta.source.revision) == (shared["repo_id"], shared["revision"])
    assert neta.prefix == netayume_loading.SYSTEM_PROMPT + " <Prompt Start> "
    gemma_spec, jina_spec = pe.encoders_for(newbie_loading.MODEL_ID)
    assert gemma_spec.source.subfolder == "text_encoder"
    assert jina_spec.source.subfolder == "clip_model"
    assert gemma_spec.source.revision == newbie_loading.SOURCE["revision"]
    assert (gemma_spec.max_length, jina_spec.max_length) == (
        newbie_loading.RECIPE["gemma_max_length"], newbie_loading.RECIPE["jina_max_length"]
    )
    qwen_spec, t5_spec = pe.encoders_for(anima_loading.MODEL_ID)
    assert qwen_spec.source.repo == anima_loading.SHARED["repo_id"]
    assert t5_spec.source.subfolder == "t5_tokenizer"


def test_anima_is_matched_exactly_and_never_by_animagines_token():
    assert names("cagliostrolab/animagine-xl-4.0") == ["CLIP-L", "CLIP-G"]
    assert names("circlestone-labs/Anima") == ["Qwen3-0.6B", "T5 ids"]
    assert names(anima_loading.AESTHETIC_MODEL_ID) == ["Qwen3-0.6B", "T5 ids"]
    assert names(anima_loading.AESTHETIC_MODEL_ID.upper()) == ["Qwen3-0.6B", "T5 ids"]
    assert pe.encoders_for("circlestone-labs/Anima@aesthetic-v1.0") is None
    assert pe.encoders_for("someone/anima-remix") is None
    assert pe.encoders_for("someone/unknown-model") is None
    # klein's encoder follows its size; FLUX.1's CLIP and T5 never reach it.
    assert names("black-forest-labs/FLUX.2-klein-9B") == ["Qwen3-8B"]
    assert names("black-forest-labs/FLUX.2-klein-base-4B") == ["Qwen3-4B"]


def test_anima_variants_share_tokenizers_but_read_their_own_converted_tree(monkeypatch, tmp_path):
    import huggingface_hub

    turbo_specs = pe.encoders_for(anima_loading.MODEL_ID)
    aesthetic_specs = pe.encoders_for(anima_loading.AESTHETIC_MODEL_ID)
    for turbo, aesthetic in zip(turbo_specs, aesthetic_specs):
        assert (aesthetic.source.repo, aesthetic.source.revision, aesthetic.source.subfolder) == (
            turbo.source.repo, turbo.source.revision, turbo.source.subfolder,
        )
        assert aesthetic.source.tree == anima_loading.AESTHETIC_MODEL_ID
        assert turbo.source.tree == anima_loading.MODEL_ID
        assert aesthetic.max_length == turbo.max_length == pe.ANIMA_SEQUENCE
    roots = {}
    for model, specs in (
        (anima_loading.MODEL_ID, turbo_specs),
        (anima_loading.AESTHETIC_MODEL_ID, aesthetic_specs),
    ):
        root = tmp_path / ("aesthetic" if "@" in model else "turbo")
        roots[model] = str(root)
        for spec in specs:
            (root / spec.source.tree_subfolder).mkdir(parents=True)
    monkeypatch.setattr(pe, "_converted_tree", roots.get)
    monkeypatch.setattr(pe, "_from_pretrained", lambda directory: directory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("cached tree ignored"))
    for specs in (turbo_specs, aesthetic_specs):
        for spec in specs:
            assert pe._load_tokenizer(spec.source) == Path(roots[spec.source.tree]) / spec.source.tree_subfolder


# --- budgets per family ---------------------------------------------------------


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


# --- weighting syntax -------------------------------------------------------------


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
    assert not pe._is_json(prompt)
    reports, findings, detail = pe.measure(model_id, prompt, prompt)
    assert all(f.rule.startswith("encoder.") for f in findings)
    if model_id == "KBlueLeaf/Kohaku-XL-Zeta":
        assert reports and detail is None


# What the libraries under `measure` really raise beside the usual three: the
# tokenizers library's bare Exception for a tokenizer.json it cannot read,
# jinja2's TemplateError from a chat template, and plain Python errors.
FAILURES = [
    Exception("data did not match any variant of untagged enum ModelWrapper"),
    TemplateError("unexpected end of template"),
    KeyError("chat_template"),
    TypeError("expected str, got NoneType"),
]


def _kind(error):
    return type(error).__name__


@pytest.mark.parametrize("error", [RuntimeError("boom"), *FAILURES], ids=_kind)
def test_a_failing_syntax_pass_costs_its_findings_not_the_analysis(
    tokenizers, monkeypatch, error
):
    def broken(text):
        raise error

    monkeypatch.setattr(pe, "_weighting_spans", broken)
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


# --- failure, the cache and the lock ----------------------------------------------


def test_a_tokenizer_that_cannot_load_leaves_a_detail_not_an_exception(tokenizers):
    gated = RuntimeError("401 Client Error: gated repo\nmore")
    tokenizers.update(tokenizer=clip(), tokenizer_2=gated)
    reports, findings, detail = pe.measure("black-forest-labs/FLUX.1-dev", words(80), "blurry")
    assert [(r.name, r.field) for r in reports] == [
        ("CLIP-L", "prompt"), ("CLIP-L", "negative_prompt"),
    ]
    assert detail == "T5-XXL unchecked: RuntimeError: 401 Client Error: gated repo"
    # The pooled cut stands without claiming T5 reads on: T5 was not read.
    (cut,) = rules(findings, "encoder.cut")
    assert "reads on" not in cut.message
    # One failed load per analysis, not one per field.
    assert [s.subfolder for s in tokenizers["loads"]].count("tokenizer_2") == 1


@pytest.mark.parametrize("error", FAILURES, ids=_kind)
def test_any_error_a_load_raises_leaves_its_encoder_unchecked(tokenizers, error):
    tokenizers.update(tokenizer=clip(), tokenizer_2=error)
    reports, _, detail = pe.measure("black-forest-labs/FLUX.1-dev", "a cat", "blurry")
    assert [(r.name, r.field) for r in reports] == [
        ("CLIP-L", "prompt"), ("CLIP-L", "negative_prompt"),
    ]
    assert detail.startswith(f"T5-XXL unchecked: {_kind(error)}")


def test_hub_errors_are_said_in_words(tokenizers):
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    gated = GatedRepoError.__new__(GatedRepoError)
    tokenizers.update(tokenizer=clip(), tokenizer_2=gated)
    detail = pe.measure("black-forest-labs/FLUX.1-dev", "a cat", None)[2]
    assert detail == (
        "T5-XXL unchecked: the tokenizer's repo is gated - accept its terms on "
        "huggingface.co and log in"
    )
    pe.unload()
    tokenizers.update(tokenizer=LocalEntryNotFoundError("Cannot find the requested files"))
    detail = pe.measure("baidu/ERNIE-Image-Turbo", "a cat", None)[2]
    assert detail == (
        "Ministral-3B unchecked: the tokenizer is not cached and the Hub cannot be reached"
    )


@pytest.mark.parametrize(
    "error", [NotImplementedError("return_offset_mapping is not available"), *FAILURES], ids=_kind
)
def test_a_tokenizer_that_breaks_while_counting_is_unchecked_too(tokenizers, error):
    class Broken(FakeTokenizer):
        def __call__(self, *args, **kwargs):
            raise error

    tokenizers.update(tokenizer=Broken(max_length=2048, special=1))
    reports, _, detail = pe.measure("baidu/ERNIE-Image-Turbo", "a cat", None)
    assert reports == [] and detail.startswith(f"Ministral-3B unchecked: {_kind(error)}")

    # A chat template that breaks is the same: Z-Image renders one per field.
    class Unrendered(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise error

    tokenizers.update(tokenizer=Unrendered(added=QWEN_ADDED))
    reports, _, detail = pe.measure("Tongyi-MAI/Z-Image-Turbo", "a cat", None)
    assert reports == [] and detail.startswith(f"Qwen3-4B unchecked: {_kind(error)}")


def test_a_failed_load_is_retried_and_a_good_one_is_kept_until_unload(tokenizers):
    ernie = "baidu/ERNIE-Image-Turbo"
    tokenizers.update(tokenizer=OSError("offline"))
    assert pe.measure(ernie, "a cat", None)[2] == "Ministral-3B unchecked: OSError: offline"
    tokenizers.update(tokenizer=tekken())
    assert pe.measure(ernie, "a cat", None)[2] is None
    assert pe.measure(ernie, "a dog", None)[2] is None
    assert len(tokenizers["loads"]) == 2
    pe.unload()
    assert pe._tokenizers == {}
    pe.measure(ernie, "a cat", None)
    assert len(tokenizers["loads"]) == 3


def test_concurrent_analyses_load_each_tokenizer_once(monkeypatch):
    loads = []

    def load(source):
        loads.append(source.subfolder)
        time.sleep(0.05)
        return clip()

    monkeypatch.setattr(pe, "_load_tokenizer", load)
    results = []

    def analyse():
        results.append(pe.measure("KBlueLeaf/Kohaku-XL-Zeta", words(80), None))

    threads = [threading.Thread(target=analyse) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(loads) == ["tokenizer", "tokenizer_2"]
    assert len(results) == 4
    assert all(detail is None and len(reports) == 2 for reports, _, detail in results)


class Stalled:
    """A `_load_tokenizer` that blocks until released, as a first-use fetch
    does on a stalled connection (the Hub's client has no timeout)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.entered = threading.Event()
        self.release = threading.Event()
        self.loads = 0

    def __call__(self, source):
        self.loads += 1
        self.entered.set()
        self.release.wait(30)
        return self.tokenizer


@pytest.fixture
def stalled(monkeypatch):
    load = Stalled(tekken())
    monkeypatch.setattr(pe, "_load_tokenizer", load)
    yield load
    load.release.set()


def _in_thread(fn):
    """`fn` started on a daemon thread; the list gets its result."""
    result: list = []
    thread = threading.Thread(target=lambda: result.append(fn()), daemon=True)
    thread.start()
    return thread, result


def test_unload_never_waits_for_a_load_and_a_late_one_is_not_kept(stalled):
    ernie = "baidu/ERNIE-Image-Turbo"
    check, answers = _in_thread(lambda: pe.measure(ernie, "a cat", None))
    assert stalled.entered.wait(5)
    unloading, _ = _in_thread(pe.unload)
    unloading.join(5)
    assert not unloading.is_alive(), "unload() waited on the tokenizer fetch"
    stalled.release.set()
    check.join(5)
    # The check that asked still reads what the load brought back ...
    ((reports, _, detail),) = answers
    assert detail is None and by(reports)["Ministral-3B"].budget == 2047
    # ... but a load that lands after unload() is not cached.
    assert pe._tokenizers == {} and pe._loading == {}
    pe.measure(ernie, "a cat", None)
    assert stalled.loads == 2 and len(pe._tokenizers) == 1


def test_a_stalled_load_costs_the_check_its_count_and_lands_for_the_next(
    stalled, monkeypatch, caplog
):
    ernie = "baidu/ERNIE-Image-Turbo"
    monkeypatch.setattr(pe, "LOAD_WAIT", 0.2)
    started = time.monotonic()
    with caplog.at_level("INFO", logger=pe.log.name):
        reports, findings, detail = pe.measure(ernie, "a cat BREAK a dog", None)
    assert time.monotonic() - started < 5
    assert reports == [] and detail == (
        "Ministral-3B unchecked: the tokenizer is still loading (a first use downloads it) - "
        "check again in a moment"
    )
    # The log says what timed out, not just which files.
    assert "tokenizer still loading when the check's 0.2 s wait ran out" in caplog.text
    # The rest of the analysis stands.
    assert [f.rule for f in findings] == ["encoder.weighting"]
    # A second check waits on the same load rather than starting its own ...
    assert pe.measure(ernie, "a cat", None)[2] == detail
    assert stalled.loads == 1
    # ... and once it lands the next one reads it.
    stalled.release.set()
    monkeypatch.setattr(pe, "LOAD_WAIT", 5.0)
    assert pe.measure(ernie, "a cat", None)[2] is None
    assert stalled.loads == 1


def test_a_load_still_in_flight_does_not_speak_for_one_that_failed(monkeypatch):
    # "Check again in a moment" is true of the load still going only: a
    # gated tokenizer beside it keeps its own reason, and a shared reason
    # names both.
    gemma, jina = pe.encoders_for(newbie_loading.MODEL_ID)
    release = threading.Event()

    def load(source):
        if source == jina.source:
            raise PermissionError("403 Client Error: gated")
        release.wait(30)
        return fakes.fake_tokenizer(source)

    monkeypatch.setattr(pe, "_load_tokenizer", load)
    monkeypatch.setattr(pe, "LOAD_WAIT", 0.2)
    try:
        detail = pe.measure(newbie_loading.MODEL_ID, "a cat", None)[2]
    finally:
        release.set()
    assert detail == (
        "Gemma 3 4B unchecked: the tokenizer is still loading (a first use downloads it) - "
        "check again in a moment; Jina CLIP v2 unchecked: PermissionError: 403 Client Error: gated"
    )
    pe.unload()
    monkeypatch.setattr(pe, "_load_tokenizer", lambda source: _raise(OSError("offline")))
    assert pe.measure(newbie_loading.MODEL_ID, "a cat", None)[2] == (
        "Gemma 3 4B and Jina CLIP v2 unchecked: OSError: offline"
    )


def _raise(error):
    raise error


def test_loading_prefers_a_converted_tree_then_the_cache_then_the_hub(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    snapshot = tmp_path / "snapshot"
    (snapshot / "text_encoder").mkdir(parents=True)

    def snapshot_download(repo, *, revision, allow_patterns, local_files_only):
        calls.append((repo, revision, tuple(allow_patterns), local_files_only))
        if local_files_only:
            raise FileNotFoundError("not in the cache")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(pe, "_from_pretrained", lambda directory: directory)
    monkeypatch.setattr(pe, "_converted_tree", lambda model_id: None)
    source = pe.encoders_for(newbie_loading.MODEL_ID)[0].source
    assert pe._load_tokenizer(source) == snapshot / "text_encoder"
    assert [call[3] for call in calls] == [True, False]
    repo, revision, patterns, _ = calls[0]
    assert (repo, revision) == (newbie_loading.MODEL_ID, newbie_loading.SOURCE["revision"])
    # Tokenizer files only: never the encoder's weights beside them.
    for wanted in ("tokenizer.json", "tokenizer.model", "merges.txt", "chat_template.jinja"):
        assert any(fnmatch.fnmatch(f"text_encoder/{wanted}", pattern) for pattern in patterns)
    for weights in ("gemma3-4b-it.safetensors", "model.bin"):
        assert not any(fnmatch.fnmatch(f"text_encoder/{weights}", pattern) for pattern in patterns)

    # A converted tree answers without touching the Hub.
    tree = tmp_path / "tree"
    (tree / "tokenizer").mkdir(parents=True)
    trees = {newbie_loading.MODEL_ID: str(tree)}
    monkeypatch.setattr(pe, "_converted_tree", trees.get)
    calls.clear()
    assert pe._load_tokenizer(source) == tree / "tokenizer"
    assert calls == []

    # A local checkpoint directory is read in place; a missing folder is a
    # plain not-found, never handed on as a repo id.
    local = pe.TokenizerSource(str(tmp_path), "tokenizer_2")
    with pytest.raises(FileNotFoundError, match="no tokenizer at"):
        pe._load_tokenizer(local)
    (tmp_path / "tokenizer_2").mkdir()
    assert pe._load_tokenizer(local) == tmp_path / "tokenizer_2"
    assert calls == []


# --- the pipelines' own templates -------------------------------------------------


def _diffusers(relative: str) -> ast.Module:
    locations = importlib.util.find_spec("diffusers").submodule_search_locations
    return ast.parse((Path(next(iter(locations))) / relative).read_text(encoding="utf-8"))


def _repo(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _self_values(tree: ast.Module) -> dict[str, object]:
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == "self":
                try:
                    values[target.attr] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return values


def _default(tree: ast.Module, cls: str, function: str, arg: str) -> object:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == function:
                args = item.args
                positional = [a.arg for a in args.posonlyargs + args.args]
                named = positional[len(positional) - len(args.defaults):]
                defaults = dict(zip(named, args.defaults))
                defaults.update((a.arg, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d)
                return ast.literal_eval(defaults[arg])
    raise AssertionError(f"{cls}.{function} not found")


def _keywords(tree: ast.Module, callee: str) -> list[dict[str, str]]:
    """The keywords of every call to `callee(...)` or `<anything>.callee(...)`, unparsed."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if callee in (getattr(node.func, "attr", None), getattr(node.func, "id", None)):
            calls.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg})
    return calls


def _call_default(tree: ast.Module, cls: str) -> object:
    return _default(tree, cls, "__call__", "max_sequence_length")


def test_krea2_and_qwen_image_templates_match_the_installed_pipelines():
    krea = _diffusers("pipelines/krea2/pipeline_krea2.py")
    values = _self_values(krea)
    assert values["prompt_template_encode_prefix"] == pe.QWEN_SYSTEM_PREFIX
    assert values["prompt_template_encode_suffix"] == pe.QWEN_ASSISTANT_SUFFIX
    assert values["prompt_template_encode_start_idx"] == pe.KREA2_PREFIX_TOKENS
    assert values["prompt_template_encode_num_suffix_tokens"] == pe.KREA2_SUFFIX_TOKENS
    assert _call_default(krea, "Krea2Pipeline") == pe.KREA2_SEQUENCE
    assert {
        "max_length": (
            "max_sequence_length + prefix_idx - self.prompt_template_encode_num_suffix_tokens"
        ),
        "truncation": "True", "padding": "'max_length'", "return_tensors": "'pt'",
    } in _keywords(krea, "tokenizer")

    qwen_image = _diffusers("pipelines/qwenimage/pipeline_qwenimage.py")
    values = _self_values(qwen_image)
    template = pe.QWEN_SYSTEM_PREFIX + "{}" + pe.QWEN_ASSISTANT_SUFFIX
    assert values["prompt_template_encode"] == template
    assert values["prompt_template_encode_start_idx"] == pe.QWEN_IMAGE_PREFIX_TOKENS
    assert values["tokenizer_max_length"] == pe.QWEN_IMAGE_TOKENIZER_MAX
    assert _call_default(qwen_image, "QwenImagePipeline") == pe.QWEN_IMAGE_SEQUENCE
    assert "self.tokenizer_max_length + drop_idx" in [
        kw.get("max_length") for kw in _keywords(qwen_image, "tokenizer")
    ]


def test_chat_template_switches_and_lengths_match_the_installed_pipelines():
    z_image = _diffusers("pipelines/z_image/pipeline_z_image.py")
    assert [kw["enable_thinking"] for kw in _keywords(z_image, "apply_chat_template")] == ["True"]
    assert _call_default(z_image, "ZImagePipeline") == pe.Z_IMAGE_SEQUENCE
    klein = _diffusers("pipelines/flux2/pipeline_flux2_klein.py")
    assert [kw["enable_thinking"] for kw in _keywords(klein, "apply_chat_template")] == ["False"]
    assert _call_default(klein, "Flux2KleinPipeline") == pe.KLEIN_SEQUENCE
    assert pe.encoders_for("Tongyi-MAI/Z-Image-Turbo")[0].chat_thinking is True
    assert pe.encoders_for("black-forest-labs/FLUX.2-klein-4B")[0].chat_thinking is False


def test_lumina_flux_ernie_and_sdxl_lengths_match_the_installed_pipelines():
    lumina = _diffusers("pipelines/lumina2/pipeline_lumina2.py")
    constants = {node.value for node in ast.walk(lumina) if isinstance(node, ast.Constant)}
    assert pe.LUMINA_SEPARATOR in constants
    assert _call_default(lumina, "Lumina2Pipeline") == pe.LUMINA_SEQUENCE
    flux = _diffusers("pipelines/flux/pipeline_flux.py")
    assert _call_default(flux, "FluxPipeline") == pe.FLUX_T5_SEQUENCE
    lengths = [kw.get("max_length") for kw in _keywords(flux, "tokenizer")]
    assert "self.tokenizer_max_length" in lengths
    # ERNIE truncates with no max_length: the tokenizer's model_max_length.
    ernie = _diffusers("pipelines/ernie_image/pipeline_ernie_image.py")
    (call,) = [kw for kw in _keywords(ernie, "tokenizer") if "truncation" in kw]
    assert call["truncation"] == "True" and "max_length" not in call
    assert pe.encoders_for("baidu/ERNIE-Image-Turbo")[0].max_length is None
    sdxl = _diffusers("pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py")
    lengths = [kw.get("max_length") for kw in _keywords(sdxl, "tokenizer")]
    assert "tokenizer.model_max_length" in lengths


def test_the_repos_own_pipelines_lengths_match():
    anima = _repo("backend/backends/anima_pipeline.py")
    lengths = [
        ast.literal_eval(value)
        for node in ast.walk(anima) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "max_sequence_length"
    ]
    assert lengths == [pe.ANIMA_SEQUENCE]
    newbie = _repo("backend/backends/newbie_pipeline.py")
    calls = _keywords(newbie, "tokenizer") + _keywords(newbie, "tokenizer_2")
    recipe = newbie_loading.RECIPE
    assert sorted(int(kw["max_length"]) for kw in calls if "max_length" in kw) == sorted(
        [recipe["gemma_max_length"], recipe["jina_max_length"]]
    )


# --- the real tokenizers, where this machine has them ----------------------------


def _cached(source: pe.TokenizerSource) -> Path | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    snapshots = Path(HF_HUB_CACHE) / f"models--{source.repo.replace('/', '--')}" / "snapshots"
    candidates = [snapshots / source.revision] if source.revision else sorted(snapshots.glob("*"))
    for snapshot in candidates:
        folder = snapshot / source.subfolder
        if (folder / "tokenizer_config.json").is_file():
            return folder
    return None


# Measured budgets for a one-line prompt. NetaYume's instruction costs 18
# tokens and the space after "<Prompt Start>" joins the first word.
REAL = [
    ("black-forest-labs/FLUX.1-Krea-dev", {"CLIP-L": 75, "T5-XXL": 511}),
    ("krea/Krea-2-Turbo", {"Qwen3-VL-4B": 507}),
    ("black-forest-labs/FLUX.2-klein-4B", {"Qwen3-4B": 509}),
    ("SeeSee21/Z-Anime", {"Qwen3-4B": 509}),
    ("KBlueLeaf/Kohaku-XL-Zeta", {"CLIP-L": 75, "CLIP-G": 75}),
    ("duongve/NetaYume-Lumina-Image-2.0", {"Gemma 2 2B": 237}),
    ("NewBie-AI/NewBie-image-Exp0.1", {"Gemma 3 4B": 511, "Jina CLIP v2": 2046}),
    ("circlestone-labs/Anima", {"Qwen3-0.6B": 512, "T5 ids": 511}),
]
# Where the chat template's closing turn starts being lost, measured the same
# way: klein's empty thinking block costs it four tokens more than Z-Anime's.
REAL_CLEAN = {"black-forest-labs/FLUX.2-klein-4B": 500, "SeeSee21/Z-Anime": 504}
FOX = "Cinematic photo of a red fox in fresh snow, golden hour"


def _clean(model_id):
    (spec,) = pe.encoders_for(model_id)
    return pe._read(spec, pe._tokenizers[spec.source], "prompt", FOX).clean


@pytest.mark.parametrize("model_id, budgets", REAL)
def test_real_tokenizers_agree_with_the_pipelines_own_truncation(monkeypatch, model_id, budgets):
    folders = {}
    for spec in pe.encoders_for(model_id):
        folder = _cached(spec.source)
        if folder is None:
            pytest.skip(f"{spec.source.label} is not in the Hub cache")
        folders[spec.source] = folder
    monkeypatch.setattr(pe, "_load_tokenizer", lambda source: pe._from_pretrained(folders[source]))
    reports, findings, detail = pe.measure(model_id, FOX, None)
    assert detail is None and findings == []
    assert {report.name: report.budget for report in reports} == budgets
    if model_id in REAL_CLEAN:
        assert _clean(model_id) == REAL_CLEAN[model_id]

    # Past every budget, the text before the cut is exactly what the
    # pipeline's own truncating tokenizer call keeps after its template.
    long = " ".join(f"forest{i % 7}" for i in range(700))
    for spec in pe.encoders_for(model_id):
        loaded = pe._tokenizers[spec.source]
        tokenizer = loaded.tokenizer
        reading = pe._read(spec, loaded, "prompt", long)
        prefix, suffix = pe._frame(spec, tokenizer, long, negative=False)
        limit = spec.max_length or tokenizer.model_max_length
        kept = tokenizer(prefix + long + suffix, truncation=True, max_length=limit)["input_ids"]
        content = len(kept) - tokenizer.num_special_tokens_to_add()
        assert reading.cut is not None
        read = tokenizer(prefix + long[: reading.cut], add_special_tokens=False)["input_ids"]
        assert len(read) == content

    # The authors' own examples fit their encoders. A template's slot that
    # stands as a whole comma item ("[lighting]") reads as A1111's brackets
    # here; the assembled check reports that span once, as the author's slot
    # (`prompt_check.checked`), so it is no warning of the encoder layer's.
    slots = model_defaults.guide_for(model_id)["slots"]
    for example in model_defaults._profile_for(model_id)["guide"]["examples"]:
        _, findings, _ = pe.measure(model_id, example, None)
        spans = {m.span() for pattern in slots for m in re.finditer(pattern, example, re.I)}
        assert not [
            finding for finding in findings
            if finding.severity == "warn"
            and not (finding.rule == "encoder.weighting" and (finding.start, finding.end) in spans)
        ], example[:60]


def test_real_clip_reads_its_control_tokens_in_any_case(monkeypatch):
    specs = pe.encoders_for("KBlueLeaf/Kohaku-XL-Zeta")
    folders = {spec.source: _cached(spec.source) for spec in specs}
    if None in folders.values():
        pytest.skip("Kohaku XL Zeta's tokenizers are not in the Hub cache")
    monkeypatch.setattr(pe, "_load_tokenizer", lambda source: pe._from_pretrained(folders[source]))
    prompt = "1girl, <|EndOfText|>, <|STARTOFTEXT|>, solo"
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    controls = rules(findings, "encoder.control_token")
    assert [prompt[f.start:f.end] for f in controls] == ["<|EndOfText|>", "<|STARTOFTEXT|>"]
    assert not rules(findings, "encoder.unknown_chars")


# Qwen-Image is not measured above, but its budget is the pipeline's own
# arithmetic: it drops its 34-token system turn and keeps 512, and the
# template's closing turn goes past 507.
FAKE_ONLY = [("Qwen/Qwen-Image", {"Qwen2.5-VL-7B": pe.QWEN_IMAGE_SEQUENCE})]
FAKE_CLEAN = {**REAL_CLEAN, "Qwen/Qwen-Image": 507}


@pytest.mark.parametrize("model_id, budgets", REAL + FAKE_ONLY)
def test_the_fake_tokenizers_budgets_are_the_real_ones(monkeypatch, model_id, budgets):
    # The API tests and the fake server take the fakes' budgets as the real.
    monkeypatch.setattr(pe, "_load_tokenizer", fakes.fake_tokenizer)
    reports, _, detail = pe.measure(model_id, FOX, None)
    assert detail is None
    assert {report.name: report.budget for report in reports} == budgets
    if model_id in FAKE_CLEAN:
        assert _clean(model_id) == FAKE_CLEAN[model_id]
