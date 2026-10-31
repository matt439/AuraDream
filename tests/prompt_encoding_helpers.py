"""Shared fakes and fixtures for the prompt encoding tests.

The fakes split words and punctuation marks with offsets, which is enough to
check the arithmetic: what a template costs, what the pipeline's truncation
keeps, and where in the user's own text reading stops. `fresh_cache` is
autouse: every test module imports it by name so pe.unload() runs around each
of its tests.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import NamedTuple

import pytest
from jinja2 import TemplateError

from backend import prompt_encoding as pe


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


# What the libraries under `measure` really raise beside the usual three: the
# tokenizers library's bare Exception for a tokenizer.json it cannot read,
# jinja2's TemplateError from a chat template, and plain Python errors.
FAILURES = [
    Exception("data did not match any variant of untagged enum ModelWrapper"),
    TemplateError("unexpected end of template"),
    KeyError("chat_template"),
    TypeError("expected str, got NoneType"),
]


def error_kind(error):
    return type(error).__name__
