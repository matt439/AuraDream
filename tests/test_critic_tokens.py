"""Critic token estimates, measured counts, and context-window admission."""
from __future__ import annotations

from pathlib import Path
import dataclasses
import logging
import unicodedata

import pytest

from backend import critic, critic_context, critic_runtime, prompt_encoding

from tests.critic_helpers import (
    KLEIN,
    Server,
    _context,
    _critique,
    _words,
    prepared,
)


# Qwen3-4B's tokenizer on these texts, measured 2026-09-27 (text, repeats, tokens).
MEASURED = [
    (
        (
            "A tall sharp-featured man in an oversized charcoal wool coat, standing on a "
            "wet cobblestone street at night, fashion editorial, moody street lighting. "
        ),
        10, 321,
    ),
    (
        (
            "1girl, long_hair, blue_eyes, school_uniform, serafuku, looking_at_viewer, "
            "(masterpiece:1.2), absurdres, "
        ),
        10, 340,
    ),
    ("一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。", 10, 290),
    ("桜の木の下で微笑む少女、夕暮れの光、柔らかい風。", 10, 220),
    ("0123456789 3.1415926-53589793 ", 10, 300),
    ("😊🎉✨ ", 10, 40),
    # Measured 2026-09-28 with klein's tokenizer.json, whose ids equal the
    # pinned GGUF's under b11206's llama-tokenize: the kinds of text the
    # first count fell short on, by up to half.
    (
        (
            "Ένας ψηλός άντρας με ένα μεγάλο γκρίζο παλτό στέκεται σε έναν βρεγμένο "
            "πλακόστρωτο δρόμο τη νύχτα, φωτογραφία μόδας, σκοτεινός φωτισμός δρόμου. "
        ),
        10, 1260,
    ),
    (
        "비 오는 밤, 젖은 자갈길 위에 서 있는 긴 회색 코트를 입은 키 큰 남자, 패션 화보, 어두운 거리 조명. ",
        10, 471,
    ),
    (
        (
            "रात में गीली पत्थर की सड़क पर खड़ा एक लंबा आदमी, बड़े आकार का ग्रे ऊनी कोट पहने, "
            "फैशन संपादकीय, उदास सड़क की रोशनी। "
        ),
        10, 1090,
    ),
    ("61997f1c2ba68c75a684ba42c5800248a818d827044d2ae33eb50e069434b1a1 ", 10, 610),
    (
        (
            "lXeIC8Uvfa4teA8RcdWstppnTx961ANZVM7BDDA5WWZuuHk5fGJhCCA1STEFS9r5ASTHuUeBkJkQBxFm"
            "/q4CZQ== "
        ),
        10, 631,
    ),
    ("👨‍👩‍👧‍👦 👩🏽‍🚀 🏳️‍🌈 🧑‍🤝‍🧑 ", 10, 330),
    (
        (
            "Ein großer Mann in einem übergroßen grauen Wollmantel steht nachts auf einer "
            "nassen Kopfsteinpflasterstraße, Modefotografie, düstere Straßenbeleuchtung. "
        ),
        10, 461,
    ),
    # And two the estimate once counted far too long, which would refuse
    # what fits before a server starts: brands, whole words in the
    # vocabulary, and accents decomposed (the tokenizer composes them).
    (
        (
            "A photo of a MacBook and an iPhone beside a PlayStation controller on a desk, "
            "YouTube on the screen. "
        ),
        10, 211,
    ),
    (
        unicodedata.normalize(
            "NFD",
            "Một người đàn ông cao lớn mặc áo khoác len xám đứng trên con đường lát đá ướt "
            "vào ban đêm. ",
        ),
        10, 282,
    ),
]


def _qwen3_tokenizer():
    """Qwen3-4B's tokenizer as FLUX.2 [klein] ships it, if the Hub cache has
    it: its ids equal the pinned GGUF's under b11206's llama-tokenize."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from tokenizers import Tokenizer

    (spec,) = prompt_encoding.encoders_for(KLEIN)
    source = spec.source
    snapshots = Path(HF_HUB_CACHE) / f"models--{source.repo.replace('/', '--')}" / "snapshots"
    candidates = [snapshots / source.revision] if source.revision else sorted(snapshots.glob("*"))
    for snapshot in candidates:
        path = snapshot / source.subfolder / "tokenizer.json"
        if path.is_file():
            return Tokenizer.from_file(str(path))
    pytest.skip("Qwen3-4B's tokenizer is not in the Hub cache")


def test_measured_counts_and_pieces_are_the_tokenizers():
    tokenizer = _qwen3_tokenizer()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    for text, repeats, tokens in MEASURED:
        text *= repeats
        assert count(text) == tokens
        # `_PIECES` is the pre-tokenizer, after its NFC: no merge crosses
        # its pieces.
        text = unicodedata.normalize("NFC", text)
        pieces = [match.group(0) for match in critic_context._PIECES.finditer(text)]
        assert "".join(pieces) == text
        assert sum(count(piece) for piece in pieces) == tokens


@pytest.mark.parametrize(("text", "repeats", "tokens"), MEASURED)
def test_the_token_estimate_is_near_the_count(text, repeats, tokens):
    estimate = critic_context.estimate_tokens(text * repeats)
    # Never so long that, scaled by the floor, it refuses what fits (English
    # prose is the most, at 1.39); and not far short, since it sizes the
    # answer for a server that will not count (base64 is the least, 0.78).
    assert 0.75 * tokens <= estimate <= tokens / critic_context._ESTIMATE_FLOOR


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize(("text", "repeats", "tokens"), MEASURED)
def test_no_measured_text_that_fits_is_refused_unasked(
    monkeypatch, text, repeats, tokens, ordered
):
    # The text as the whole input, in a window its answer fills exactly: the
    # estimate before a server starts must not refuse it, and the count
    # after must admit it, with one token less refused.
    prompt = text * repeats
    estimated = critic_context.estimate_tokens(prompt)
    context = dataclasses.replace(
        _context(KLEIN, "a cat", ordered=ordered), prompt_tokens=estimated, input_tokens=estimated
    )
    counted = dataclasses.replace(context, prompt_tokens=tokens, input_tokens=tokens)
    monkeypatch.setattr(critic_runtime, "CONTEXT", tokens + context.copies * tokens + critic_context.ANSWER_OVERHEAD)
    critic_context.check_fits(context, estimated=True)
    critic_context.check_fits(counted)
    monkeypatch.setattr(critic_runtime, "CONTEXT", critic_runtime.CONTEXT - 1)
    with pytest.raises(critic_runtime.CriticError, match="too long"):
        critic_context.check_fits(counted)


def test_the_estimate_reads_text_as_the_tokenizer_normalizes_it():
    # Qwen3's normalizer is NFC, so decomposed accents count as composed ones.
    text = "Crème brûlée à la française, élégant café, naïve façade. " * 10
    assert critic_context.estimate_tokens(unicodedata.normalize("NFD", text)) == critic_context.estimate_tokens(text)


@pytest.mark.parametrize("ordered", [True, False])
def test_the_servers_count_admits_what_fits_and_refuses_the_rest(monkeypatch, prepared, ordered):
    server = Server(monkeypatch)
    # The question as the server counts it: 3000 tokens in, and a prompt of
    # 1000 that the answer repeats `copies` times.
    server.count = lambda text: 3000 if text.startswith("<|im_start|>") else 1000
    copies = 3 if ordered else 2
    need = 3000 + copies * 1000 + critic_context.ANSWER_OVERHEAD
    monkeypatch.setattr(critic_runtime, "CONTEXT", need)
    _critique(prepared, ordered=ordered)
    assert server.calls[-1][1] == "/v1/chat/completions"
    budget = critic_runtime.REASONING_BUDGET + copies * 1000 + critic_context.ANSWER_OVERHEAD
    assert server.calls[-1][3]["max_tokens"] == min(need, max(critic_context.MIN_TOKENS, budget))

    monkeypatch.setattr(critic_runtime, "CONTEXT", need - 1)
    server.calls.clear()
    with pytest.raises(critic_runtime.CriticError, match="too long"):
        _critique(prepared, ordered=ordered)
    # Started, counted and stopped; never asked.
    assert [path for _, path, _, _ in server.calls][-1] == "/tokenize"
    assert server.process.terminated and prepared._process is None


def test_a_long_prompt_is_given_room_for_its_answer(monkeypatch, prepared):
    server = Server(monkeypatch)
    # Near the request's 4000-character limit: 848 tokens by Qwen3's own
    # tokenizer, which the answer repeats three times after up to
    # REASONING_BUDGET of thought.
    prompt = (MEASURED[0][0] * 30)[:4000]
    server.count = lambda text: _words(text, {prompt: 848})
    _critique(prepared, prompt=prompt)
    budget = server.calls[-1][3]["max_tokens"]
    assert budget == critic_runtime.REASONING_BUDGET + 3 * 848 + critic_context.ANSWER_OVERHEAD
    # Unordered, the answer has no reordered copy, and a short prompt is
    # given the measured floor.
    _critique(prepared, prompt=prompt, ordered=False)
    assert server.calls[-1][3]["max_tokens"] == budget - 848
    _critique(prepared, prompt="a cat")
    assert server.calls[-1][3]["max_tokens"] == critic_context.MIN_TOKENS


def test_a_server_that_cannot_count_is_sized_by_the_estimate(monkeypatch, prepared, caplog):
    server = Server(monkeypatch)
    server.counts = False
    prompt = (MEASURED[0][0] * 30)[:4000]
    with caplog.at_level(logging.WARNING, logger=critic.log.name):
        result = _critique(prepared, prompt=prompt)
    assert result.issues and server.calls[-1][1] == "/v1/chat/completions"
    # The estimate is long on English prose, so the answer keeps its room.
    budget = server.calls[-1][3]["max_tokens"]
    assert critic_runtime.REASONING_BUDGET + 3 * 848 + critic_context.ANSWER_OVERHEAD < budget <= critic_runtime.CONTEXT
    assert "sized by the estimate" in caplog.text


def test_a_prompt_whose_answer_cannot_fit_is_refused_before_a_server_starts(
    monkeypatch, prepared
):
    server = Server(monkeypatch)
    # 4000 characters of Chinese measured 2698 tokens: repeated twice with the
    # sources, and again as the negative prompt, it overflows 8192 tokens
    # before the critic has thought at all, by more than the estimate's
    # margin - so no server is started to count it.
    chinese = (MEASURED[2][0] * 100)[:4000]
    with pytest.raises(critic_runtime.CriticError, match="too long"):
        _critique(prepared, "baidu/ERNIE-Image-Turbo", chinese, negative=chinese, ordered=False)
    assert server.commands == []
    # A shorter one is asked.
    _critique(prepared, "baidu/ERNIE-Image-Turbo", chinese[:1500], ordered=False)
    assert len(server.commands) == 1 and server.calls[-1][1] == "/v1/chat/completions"


def test_a_greek_prompt_past_the_window_is_refused_before_it_is_asked(monkeypatch, prepared):
    server = Server(monkeypatch)
    # 2500 characters of Greek are 2189 tokens by Qwen3's tokenizer, and with
    # the sources and two copies in the answer need over 8400, which the
    # first estimate put well inside the window: the server ran to its end
    # and answered 502 then. Within the estimate's margin, a server starts;
    # its own count refuses the question before it is sent.
    greek = (MEASURED[6][0] * 20)[:2500]
    server.count = lambda text: _words(text, {greek: 2189})
    with pytest.raises(critic_runtime.CriticError, match="too long"):
        _critique(prepared, "black-forest-labs/FLUX.1-Krea-dev", greek, ordered=False)
    assert len(server.commands) == 1
    assert not any(path == "/v1/chat/completions" for _, path, _, _ in server.calls)
    assert server.process.terminated and prepared._process is None
