"""Small, CPU-only Chinese-to-English translation for expanded prompts."""
from __future__ import annotations

import gc
import re
import threading
from typing import Any

MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
_MODEL_TOKEN_LIMIT = 512
_SPECIAL_TOKEN_ROOM = 2
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])")


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

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
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
        """Keep every source token while preferring Chinese sentence breaks."""
        limit = cls._source_limit(tokenizer)
        chunks: list[str] = []
        current: list[int] = []

        def flush() -> None:
            if current:
                chunk = tokenizer.decode(current, skip_special_tokens=True).strip()
                if chunk:
                    chunks.append(chunk)
                current.clear()

        for sentence in filter(None, _SENTENCE_END.split(text)):
            token_ids = list(tokenizer.encode(sentence, add_special_tokens=False))
            if len(token_ids) <= limit:
                if current and len(current) + len(token_ids) > limit:
                    flush()
                current.extend(token_ids)
                continue
            flush()
            while token_ids:
                current.extend(token_ids[:limit])
                del token_ids[:limit]
                flush()
        flush()
        return chunks

    def translate(self, text: str) -> str:
        source = text.strip()
        if not source:
            raise ValueError("there is no expanded prompt to translate")
        with self._lock:
            import torch

            tokenizer, model = self._load()
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
