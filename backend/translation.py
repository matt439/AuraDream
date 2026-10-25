"""Small, CPU-only Chinese-to-English translation for expanded prompts."""
from __future__ import annotations

import gc
import os
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .backends.base import Cancelled

MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
# What a first translation downloads: the repository's `pytorch_model.bin`,
# the weights Transformers loads from it (it publishes no safetensors on its
# main branch). The page names it beside ERNIE's rewriter before a first
# Expand (`ModelStatus.prompt_expander`); the tokenizer files are a few MB.
# Only the `.bin`: the load turns off Transformers' safetensors conversion
# (`_without_conversion`), which would fetch the same weights a second time.
MODEL_BYTES = 312_087_009
# Having loaded a `.bin`, Transformers starts a thread ("auto_conversion")
# that asks the Hub's conversion bot for the same weights as safetensors -
# on this repository an open pull request - and downloads them beside the
# `.bin`: a second 312 MB a translation never reads, behind network calls
# the status could not name. This variable, which Transformers reads at each
# load, turns that off; it is set for the translator's load alone.
_NO_CONVERSION = "DISABLE_SAFETENSORS_CONVERSION"
# The files whose presence in the Hub cache means a first translation
# downloads nothing: the config, and the weights in either form.
_CONFIG = "config.json"
_WEIGHTS = ("pytorch_model.bin", "model.safetensors")
_MODEL_TOKEN_LIMIT = 512
_SPECIAL_TOKEN_ROOM = 2
# A sentence ends at Chinese or Western sentence punctuation or a newline;
# a Western full stop only before whitespace, so "1.5" stays whole.
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])|(?<=\.)(?=\s)")


@contextmanager
def _without_conversion() -> Iterator[None]:
    """Transformers' safetensors conversion off for one load, and the variable
    put back as it was after it: the image pipeline's loads are left alone."""
    previous = os.environ.get(_NO_CONVERSION)
    os.environ[_NO_CONVERSION] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_NO_CONVERSION, None)
        else:
            os.environ[_NO_CONVERSION] = previous


class MarianTranslator:
    """Lazily load Marian and translate one expanded prompt at a time.

    The image pipeline owns the accelerator. A single short translation is
    quick enough on the CPU, where it cannot disturb the pipeline's offload
    chain or consume the VRAM an image job is about to need.
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()

    def cached(self) -> bool:
        """Whether a first translation would load without downloading.

        File checks only - a local directory, or the Hub cache - with no
        network and no import of the model code, so `/api/status` can ask it
        on every poll, as it asks the critic's `ready()`. Loaded already is
        cached too.
        """
        if self._model is not None:
            return True
        if os.path.isdir(self.model_id):
            return os.path.isfile(os.path.join(self.model_id, _CONFIG)) and any(
                os.path.isfile(os.path.join(self.model_id, name)) for name in _WEIGHTS
            )
        from huggingface_hub import try_to_load_from_cache

        def present(filename: str) -> bool:
            try:
                found = try_to_load_from_cache(self.model_id, filename)
            except (OSError, ValueError):
                return False
            return isinstance(found, str) and os.path.isfile(found)

        return present(_CONFIG) and any(present(name) for name in _WEIGHTS)

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            with _without_conversion():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id,
                    low_cpu_mem_usage=True,
                )
            model.to("cpu")
            model.eval()
            # Assign only once both loads succeeded. A failed first download is
            # retryable instead of leaving a half-ready translator behind.
            self._tokenizer, self._model = tokenizer, model
        return self._tokenizer, self._model

    @staticmethod
    def _source_limit(tokenizer: Any) -> int:
        advertised = getattr(tokenizer, "model_max_length", _MODEL_TOKEN_LIMIT)
        try:
            advertised = int(advertised)
        except (TypeError, ValueError, OverflowError):
            advertised = _MODEL_TOKEN_LIMIT
        return max(1, min(advertised, _MODEL_TOKEN_LIMIT) - _SPECIAL_TOKEN_ROOM)

    @classmethod
    def _chunks(cls, text: str, tokenizer: Any) -> list[str]:
        """One sequence per sentence, every source token kept.

        Marian translates a sentence or two of a longer sequence and drops
        the rest with no error, so sentences are never packed together: an
        expansion sent as one chunk came back as a fraction of its English
        (measured on the card 2026-10-25). Only a sentence longer than the
        token limit is cut, at the limit.
        """
        limit = cls._source_limit(tokenizer)
        chunks: list[str] = []
        for sentence in _SENTENCE_END.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            token_ids = list(tokenizer.encode(sentence, add_special_tokens=False))
            if len(token_ids) <= limit:
                chunks.append(sentence)
                continue
            for start in range(0, len(token_ids), limit):
                piece = tokenizer.decode(
                    token_ids[start:start + limit], skip_special_tokens=True
                ).strip()
                if piece:
                    chunks.append(piece)
        return chunks

    def translate(self, text: str, should_cancel: Callable[[], bool] | None = None) -> str:
        def check() -> None:
            if should_cancel is not None and should_cancel():
                raise Cancelled("prompt expansion stopped")

        check()
        source = text.strip()
        if not source:
            raise ValueError("there is no expanded prompt to translate")
        with self._lock:
            check()
            import torch

            tokenizer, model = self._load()
            check()
            chunks = self._chunks(source, tokenizer)
            inputs = tokenizer(
                chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._source_limit(tokenizer),
            )
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=_MODEL_TOKEN_LIMIT - 1,
                    renormalize_logits=True,
                )
            check()
            translated = " ".join(
                part.strip()
                for part in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                if part.strip()
            )
            if not translated:
                raise RuntimeError("Marian returned no English translation")
            return translated

    def unload(self) -> None:
        """Drop the CPU model along with the image pipeline on Unload."""
        with self._lock:
            loaded = self._model is not None or self._tokenizer is not None
            self._model = None
            self._tokenizer = None
        if loaded:
            gc.collect()


prompt_translator = MarianTranslator()
