"""The tests' stand-ins for the GPU and the downloads: a backend that draws
nothing, a translator that translates nothing, a prompt critic with no
llama.cpp and tokenizers with no vocabulary file, running the whole job and
analysis machinery at a few milliseconds a call. tools/fake_server.py builds
the browser smoke server on the same fakes."""
from __future__ import annotations

import http.client
import json
import re
import threading
import time
from typing import NamedTuple

from PIL import Image

from backend import critic, prompt_encoding
from backend.backends.base import GenerationBackend
from backend.config import settings
from backend.critic import CriticError, CriticUnavailable
from backend.schemas import GenerateRequest, PromptCritique


class FakeBackend(GenerationBackend):
    """Runs the whole job machinery at a few milliseconds per step."""

    name = "fake"

    def __init__(self, step_seconds: float = 0.0, cfg_threshold: float | None = None, loaded: bool = True):
        self.step_seconds = step_seconds
        self.requests: list[GenerateRequest] = []
        self.seeds: list[int] = []
        self.loaded = loaded
        self.loads = 0
        # None: apply everything as asked, like the base class. A number: behave
        # like a pipeline whose negative prompt switches on above it - on the
        # CFG value, or on a separate true CFG when `true_cfg_supported`.
        self.cfg_threshold = cfg_threshold
        self.true_cfg_supported = False
        # A capable stand-in by default: the pipeline takes reference images
        # and an adapter can be applied, so a test that cares about either
        # turns it off rather than every other test having to turn it on.
        self.reference_supported = True
        self.lora_reason: str | None = None
        # Adapters that do not fit the claimed checkpoint, by name, with the
        # reason the real backend would give: hidden by the page, refused at
        # submit. Everything else fits.
        self.lora_mismatches: dict[str, str] = {}
        self.prompt_expander_supported = False
        self.expansions: list[tuple[str, int, int]] = []
        # Calls (1-based) that raise instead of rendering, and a hook run after
        # each render - for a cancel that lands after the last step.
        self.fail_on: set[int] = set()
        self.after_generate = None

    def lora_supported(self) -> bool:
        return self.lora_reason is None

    def lora_unavailable_reason(self) -> str | None:
        return self.lora_reason

    def lora_mismatch(self, name: str) -> str | None:
        return self.lora_mismatches.get(name)

    def prompt_expansion_supported(self) -> bool:
        return self.prompt_expander_supported

    def expand_prompt(self, prompt: str, width: int, height: int) -> str:
        self.expansions.append((prompt, width, height))
        return f"Expanded: {prompt} ({width}x{height})"

    def applied(self, request):
        # What this fake would drop, the way the real backend drops it: an
        # input it cannot use must not reach the row.
        changes = {}
        if request.reference and not self.reference_supported:
            changes["reference"] = []
        if request.lora and not self.lora_supported():
            changes["lora"] = None
        if changes:
            request = request.model_copy(update=changes)
        if self.cfg_threshold is None:
            return request
        return request.applied(
            negative_prompt_supported=True,
            cfg_threshold=self.cfg_threshold,
            true_cfg_supported=self.true_cfg_supported,
            reference_supported=self.reference_supported,
            lora_supported=self.lora_supported(),
        )

    def load(self) -> None:
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def is_ready(self) -> bool:
        return self.loaded

    def generate(self, request, seed, on_progress=None, should_cancel=None, on_preview=None):
        from backend.backends import Cancelled

        if not self.loaded:
            # The real backend's contract: loading happens in main.ensure_loaded
            # and nowhere else, so a job that reaches here unloaded is a bug.
            raise RuntimeError("model is not loaded")
        self.requests.append(request)
        self.seeds.append(seed)
        if len(self.requests) in self.fail_on:
            raise RuntimeError(f"cell {len(self.requests)} blew up")
        for step in range(request.steps):
            if self.step_seconds:
                time.sleep(self.step_seconds)
            if on_progress is not None:
                on_progress(step + 1, request.steps)
            # The contract every backend signs up to: poll between steps.
            if should_cancel is not None and should_cancel():
                raise Cancelled
        if self.after_generate is not None:
            self.after_generate()
        # At the size that was asked for: the row records the image's own
        # dimensions, so anything matching on them - resume, the duration
        # estimate - needs a double that does not quietly render 16x16.
        return [
            Image.new("RGB", (request.width, request.height)) for _ in range(request.num_images)
        ]

    def describe(self) -> dict:
        return {
            "model_id": settings.model_id,
            "quantization": settings.quantization,
            "offload": settings.offload,
            "cfg_threshold": None,
            "reference_supported": self.reference_supported,
            "lora_supported": self.lora_supported(),
        }

    def free_cache(self) -> None:
        self.cache_freed = getattr(self, "cache_freed", 0) + 1


class FakeTranslator:
    def __init__(self) -> None:
        self.translations: list[str] = []
        self.unloads = 0
        self.fail = False

    def translate(self, text: str) -> str:
        self.translations.append(text)
        if self.fail:
            raise RuntimeError("translation broke")
        return f"English: {text}"

    def unload(self) -> None:
        self.unloads += 1


class CutReply:
    """`urlopen`'s reply to a tag lookup, stopping mid-body: a chunked reply
    cut off raises http.client's error, not an OSError."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        raise http.client.IncompleteRead(b"[{", 100)


class FakeCritic:
    """`critic.PromptCritic` without llama.cpp or a model: the same five
    methods, and a deterministic answer written as the model writes one - raw
    JSON - then put through the real `critic.validate`, so what reaches the
    page is only what the real pipeline could return.

    `available` is what `supported()` answers and `fetched` what `ready()`
    does; `prepare()` sets `fetched`, as a first real prepare puts both files
    on disk. `fail` makes `critique()` raise - True for a `CriticError`, or an
    exception to raise instead - and `prepare_fail` the same for `prepare()`.
    `delay` holds `critique()` for that many seconds, so a caller can act
    while it runs; `started` is set as each one begins.

    Its `reordered` moves the last comma clause to the front, which
    `critic.is_reorder` accepts; a prompt of one clause has nothing to move,
    so an ordered checkpoint gets no reordered version, as it often does from
    the real critic.
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.available = True
        self.fetched = False
        self.delay = delay
        self.fail: bool | Exception = False
        self.prepare_fail: Exception | None = None
        self.prepares = 0
        self.unloads = 0
        # (model_id, request, applied, analysis) per critique, in order, and
        # a running log of what happened when - prepare, critique, done.
        self.critiques: list[tuple] = []
        self.events: list[str] = []
        self.started = threading.Event()

    def supported(self) -> bool:
        return self.available

    def ready(self) -> bool:
        return self.available and self.fetched

    def prepare(self) -> None:
        self.prepares += 1
        self.events.append("prepare")
        if self.prepare_fail is not None:
            raise self.prepare_fail
        self.fetched = True

    def critique(self, model_id, request, applied, analysis) -> PromptCritique:
        # The real critic's contract: prepare() fetches, critique() runs.
        if not self.prepares or not self.fetched:
            raise CriticUnavailable("the prompt critic is not prepared; prepare() fetches it first")
        self.critiques.append((model_id, request, applied, analysis))
        self.events.append("critique")
        self.started.set()
        started = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        try:
            if self.fail is True:
                raise CriticError("llama-server exited with code 1 before it was ready")
            if self.fail:
                raise self.fail
            context = critic.build_context(model_id, request, applied, analysis)
            return critic.validate(
                self.answer(request.prompt, context.ordered),
                context,
                request.prompt,
                model="fake critic",
                seconds=time.perf_counter() - started,
            )
        finally:
            self.events.append("critique done")

    @staticmethod
    def answer(prompt: str, ordered: bool) -> str:
        """The model's side: the answer's JSON, before any validation."""
        items = [item.strip() for item in prompt.split(",") if item.strip()] or [prompt]
        quoted = re.search(r'"([^"\n]+)"', prompt)
        # The guide's first sentence, cited as G1 for validation to resolve,
        # quoting the prompt's first word - so the page's cite tooltip and
        # quote selection have something real to show.
        word = re.search(r"\w+", prompt)
        rotated = ", ".join(items[-1:] + items[:-1])
        return json.dumps(
            {
                "parts": {
                    "subject": items[0][:120],
                    "action": None,
                    "setting": items[-1][:120] if len(items) > 1 else None,
                    "style": None,
                    "lighting": None,
                    "camera": None,
                    "text_to_render": quoted.group(1) if quoted else None,
                },
                "issues": [
                    {
                        "kind": "other",
                        "problem": "The fake critic holds every prompt to its guide's opening sentence.",
                        "quote": word.group() if word else None,
                        "cite_id": "G1",
                        "fix": "Compare the prompt with the cited sentence.",
                    }
                ],
                "rewrite": f"Rewritten: {prompt}",
                "reordered": rotated if ordered else None,
            }
        )

    def unload(self) -> None:
        self.unloads += 1


class _Added(NamedTuple):
    content: str
    special: bool


class WordTokenizer:
    """A tokenizer as `prompt_encoding` reads one, with no vocabulary file.

    Words and single punctuation marks are tokens, with real offsets, and an
    added token typed into the text is split out whole - roughly how the real
    ones count English, which is enough for budgets, cut points and control
    tokens to come out where a reader expects them. Characters in `unknown`
    map to the unk id, as T5's missing braces do. With `newlines`, a run of
    newlines is one token too, as Qwen's byte-level BPE merges it: its chat
    templates put one between every turn, so the templated budgets come out
    as the real ones only when they count.
    """

    def __init__(
        self, *, max_length: int, special: int, added=(), unknown: str = "", template=None,
        newlines: bool = False,
    ) -> None:
        self.model_max_length = max_length
        self.special = special
        self.chat_template = template
        self.added_tokens_decoder = {100 + i: _Added(*item) for i, item in enumerate(added)}
        self._ids = {entry.content: i for i, entry in self.added_tokens_decoder.items()}
        self.all_special_ids = [i for i, entry in self.added_tokens_decoder.items() if entry.special]
        self.unk_token = "<unk>"
        self.unk_token_id = self._ids.get("<unk>", 3)
        self._unknown = set(unknown)
        self._vocab: dict[str, int] = {}
        parts = [re.escape(content) for content in sorted(self._ids, key=len, reverse=True)]
        if unknown:
            parts.append(f"[{re.escape(unknown)}]")
        if newlines:
            parts.append(r"\n+")
        self._pattern = re.compile("|".join([*parts, r"\w+", r"[^\w\s]"]))
        self._lock = threading.Lock()

    def num_special_tokens_to_add(self) -> int:
        return self.special

    def convert_ids_to_tokens(self, ids):
        names = {i: entry.content for i, entry in self.added_tokens_decoder.items()}
        return [names.get(i, f"t{i}") for i in ids]

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids, offsets = [], []
        for match in self._pattern.finditer(text):
            piece = match.group()
            if piece in self._ids:
                ids.append(self._ids[piece])
            elif piece in self._unknown:
                ids.append(self.unk_token_id)
            else:
                with self._lock:
                    ids.append(self._vocab.setdefault(piece, 1000 + len(self._vocab)))
            offsets.append(match.span())
        encoding = {"input_ids": ids}
        if return_offsets_mapping:
            encoding["offset_mapping"] = offsets
        return encoding

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        # Qwen3's template for one user turn, with the empty thinking block
        # that klein's `enable_thinking=False` renders.
        (message,) = messages
        text = f"<|im_start|>user\n{message['content']}<|im_end|>\n<|im_start|>assistant\n"
        return text + ("<think>\n\n</think>\n\n" if enable_thinking is False else "")


_QWEN_ADDED = (
    ("<|endoftext|>", True), ("<|im_start|>", True), ("<|im_end|>", True),
    ("<think>", False), ("</think>", False),
)


def _word_tokenizer(spec: prompt_encoding.EncoderSpec) -> WordTokenizer:
    """One per encoder family, with the length and special tokens the real
    tokenizer has, so every budget is the one part A's table measured: CLIP
    75, T5 511, ERNIE 2047, Jina 2046, and the templated ones what their
    templates leave - Krea 2 507, klein, Z-Image and Z-Anime 509, Qwen-Image
    512 (`test_prompt_encoding.py` holds them to its REAL table)."""
    name = spec.name
    if name.startswith("CLIP"):
        return WordTokenizer(
            max_length=77, special=2, added=(("<|startoftext|>", True), ("<|endoftext|>", True))
        )
    if name.startswith("T5"):
        added = (("<pad>", True), ("</s>", True), ("<unk>", True), ("<extra_id_0>", True))
        return WordTokenizer(max_length=512, special=1, added=added, unknown="{}")
    if name.startswith("Qwen"):
        # `<think>` is a control token only where the template uses it:
        # Z-Image's and klein's, which render the chat template.
        template = "<|im_start|> <think> </think>" if spec.chat_thinking is not None else None
        return WordTokenizer(
            max_length=131072, special=0, added=_QWEN_ADDED, template=template, newlines=True
        )
    if name.startswith("Gemma"):
        added = (("<pad>", True), ("<eos>", True), ("<bos>", True))
        return WordTokenizer(max_length=8192, special=1, added=added)
    if name.startswith("Ministral"):
        added = (("<s>", True), ("</s>", True), ("[INST]", True))
        return WordTokenizer(max_length=2048, special=1, added=added)
    if name.startswith("Jina"):
        added = (("<s>", True), ("</s>", True), ("<mask>", True))
        return WordTokenizer(max_length=8194, special=2, added=added)
    raise FileNotFoundError(f"no fake tokenizer for {name}")


def fake_tokenizer(source: prompt_encoding.TokenizerSource) -> WordTokenizer:
    """`prompt_encoding._load_tokenizer`, answered offline.

    A source belongs to the checkpoint whose id is its repo (an ordinary
    Diffusers repo, Z-Anime's nested tree) or its converted tree (a pinned
    release); that checkpoint's table entry says which encoder it feeds.
    """
    for model_id in (source.tree, source.repo):
        for spec in prompt_encoding.encoders_for(model_id or "") or []:
            if spec.source == source:
                return _word_tokenizer(spec)
    raise FileNotFoundError(f"no fake tokenizer for {source.label}")
