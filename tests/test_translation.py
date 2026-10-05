from __future__ import annotations

from types import SimpleNamespace

import torch

from backend.translation import MarianTranslator


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
            "max_length": 6,
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


def test_marian_loads_once_on_cpu_and_keeps_long_prompt_chunks(monkeypatch):
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
