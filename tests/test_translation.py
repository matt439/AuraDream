from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import pytest

from backend.translation import MarianTranslator
from backend.backends import Cancelled


class StubTokenizer:
    model_max_length = 8

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, chunks, **kwargs):
        self.batches.append(chunks)
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 8,
        }
        return {"input_ids": torch.tensor([[index] for index, _ in enumerate(chunks)])}

    def batch_decode(self, output_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [f"translated-{int(row[0])}" for row in output_ids]


class StubModel:
    def __init__(self) -> None:
        self.device = None
        self.evaluated = False
        self.generations = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def generate(self, input_ids, **kwargs):
        self.generations += 1
        assert kwargs == {
            "do_sample": False,
            "max_new_tokens": 511,
            "renormalize_logits": True,
        }
        return input_ids + 10


class TruncatingTokenizer(StubTokenizer):
    """Character source ids, two special tokens, real truncation and padding."""

    def __call__(self, chunks, *, return_tensors, padding, truncation, max_length):
        assert return_tensors == "pt" and padding and truncation
        self.batches.append(chunks)
        rows = []
        for chunk in chunks:
            # Like a tokenizer that preserves its special tokens while cutting
            # the source to fit the requested total input length.
            source_ids = self.encode(chunk)
            rows.append([1, *source_ids[:max_length - 2], 2])
        longest = max(map(len, rows))
        return {"input_ids": torch.tensor([row + [0] * (longest - len(row)) for row in rows])}


@pytest.mark.parametrize("length", [5, 6, 7, 12, 13, 19])
def test_full_chunks_keep_every_source_token_within_the_total_input_limit(monkeypatch, length):
    tokenizer = TruncatingTokenizer()
    received = []

    class RecordingModel(StubModel):
        def generate(self, input_ids, **kwargs):
            received.extend(input_ids.tolist())
            return super().generate(input_ids, **kwargs)

    translator = MarianTranslator("local/marian")
    monkeypatch.setattr(translator, "_load", lambda: (tokenizer, RecordingModel()))
    source = "abcdefghijklmnopqrs"[:length]
    translator.translate(source)

    assert all(len(row) <= tokenizer.model_max_length for row in received)
    source_ids = [token_id for row in received for token_id in row if token_id > 2]
    assert source_ids == tokenizer.encode(source)
    assert len(received) == (length + 5) // 6


@pytest.mark.parametrize("advertised", [512, 100000, None, "unknown", float("inf")])
def test_total_input_limit_stays_bounded_when_tokenizer_limit_is_missing_or_unusable(advertised):
    tokenizer = SimpleNamespace(model_max_length=advertised)
    assert MarianTranslator._input_limit(tokenizer) == 512
    assert MarianTranslator._source_limit(tokenizer) == 510


def test_stop_after_the_translators_download_skips_translation(monkeypatch):
    translator = MarianTranslator("local/marian")
    model = StubModel()
    stopped = False

    def load():
        nonlocal stopped
        stopped = True
        return StubTokenizer(), model

    monkeypatch.setattr(translator, "_load", load)
    with pytest.raises(Cancelled):
        translator.translate("source", lambda: stopped)
    assert model.generations == 0


def test_marian_loads_once_on_cpu_and_cuts_only_an_over_long_sentence(monkeypatch):
    # Each sentence is its own sequence; one past the limit is cut at it,
    # every token kept.
    tokenizer = StubTokenizer()
    model = StubModel()
    tokenizer_load = SimpleNamespace(calls=0)
    model_load = SimpleNamespace(calls=0)

    def load_tokenizer(model_id):
        tokenizer_load.calls += 1
        assert model_id == "local/marian"
        return tokenizer

    def load_model(model_id, **kwargs):
        model_load.calls += 1
        assert model_id == "local/marian" and kwargs == {"low_cpu_mem_usage": True}
        return model

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    monkeypatch.setattr("transformers.AutoModelForSeq2SeqLM.from_pretrained", load_model)
    translator = MarianTranslator("local/marian")

    assert translator.translate("第一句。第二句真的很长！") == "translated-10 translated-11 translated-12"
    assert translator.translate("短句。") == "translated-10"
    assert tokenizer_load.calls == model_load.calls == 1
    assert tokenizer.batches[0] == ["第一句。", "第二句真的很", "长！"]
    assert model.device == "cpu" and model.evaluated and model.generations == 2

    translator.unload()
    assert translator._tokenizer is None and translator._model is None


def test_a_multi_sentence_expansion_sends_each_sentence_separately(monkeypatch):
    # Marian keeps a sentence or two of a packed sequence and drops the rest
    # with no error (GEN#21, measured on the card): sentences that would fit
    # the limit together still go one to a sequence, in one generate() call.
    tokenizer = StubTokenizer()
    model = StubModel()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda model_id: tokenizer)
    monkeypatch.setattr(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained", lambda model_id, **kwargs: model
    )
    translator = MarianTranslator("local/marian")

    assert translator.translate("一。二！三？\n四；五. 六\n\n七") == " ".join(
        f"translated-{10 + index}" for index in range(7)
    )
    assert tokenizer.batches == [["一。", "二！", "三？", "四；", "五.", "六", "七"]]
    assert model.generations == 1


def test_a_western_full_stop_ends_a_sentence_only_before_whitespace():
    tokenizer = StubTokenizer()
    tokenizer.model_max_length = 64
    assert MarianTranslator._chunks("It is 1.5 m tall. It is red.", tokenizer) == [
        "It is 1.5 m tall.",
        "It is red.",
    ]


def test_the_translators_load_fetches_no_second_copy_of_its_weights(monkeypatch):
    # Transformers, having loaded the repository's `.bin`, starts a thread that
    # downloads the conversion bot's safetensors copy too - a second 312 MB the
    # status does not name. The load turns that off, for itself alone.
    import inspect

    import transformers.modeling_utils

    from backend import translation

    name = translation._NO_CONVERSION
    assert f'"{name}"' in inspect.getsource(transformers.modeling_utils)
    monkeypatch.delenv(name, raising=False)
    seen = {}

    def load_model(model_id, **kwargs):
        seen["during"] = os.environ.get(name)
        return StubModel()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda model_id: StubTokenizer())
    monkeypatch.setattr("transformers.AutoModelForSeq2SeqLM.from_pretrained", load_model)
    MarianTranslator("local/marian")._load()
    assert seen["during"] == "1"
    assert name not in os.environ
    # A value the environment already had is put back after the load, and
    # one that a failed load interrupted too.
    monkeypatch.setenv(name, "0")

    def failing(model_id, **kwargs):
        seen["failed"] = os.environ.get(name)
        raise OSError("no network")

    monkeypatch.setattr("transformers.AutoModelForSeq2SeqLM.from_pretrained", failing)
    try:
        MarianTranslator("local/marian")._load()
    except OSError:
        pass
    else:
        raise AssertionError("the failed load was not raised")
    assert seen["failed"] == "1" and os.environ[name] == "0"


def test_marian_refuses_empty_text_without_loading():
    translator = MarianTranslator("local/marian")
    try:
        translator.translate("  ")
    except ValueError as exc:
        assert str(exc) == "there is no expanded prompt to translate"
    else:
        raise AssertionError("empty text was accepted")
    assert translator._model is None


def test_whether_a_first_translation_downloads_is_a_file_check(monkeypatch, tmp_path):
    import huggingface_hub

    # A local directory: the config and the weights in either form.
    tree = tmp_path / "marian"
    tree.mkdir()
    translator = MarianTranslator(str(tree))
    assert not translator.cached()
    (tree / "config.json").write_text("{}", encoding="utf-8")
    assert not translator.cached()
    (tree / "pytorch_model.bin").write_bytes(b"weights")
    assert translator.cached()
    # A Hub id: the cache answers, file by file, and the network is never asked.
    held = {"config.json": tree / "config.json", "pytorch_model.bin": tree / "pytorch_model.bin"}
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename, **kwargs: str(held[filename]) if filename in held else None,
    )
    hub = MarianTranslator("Helsinki-NLP/opus-mt-zh-en")
    assert hub.cached()
    del held["pytorch_model.bin"]
    assert not hub.cached()
    held["model.safetensors"] = tree / "pytorch_model.bin"
    assert hub.cached()
    # Loaded is cached, whatever the disk says.
    held.clear()
    hub._model = object()
    assert hub.cached()
