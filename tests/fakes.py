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
from backend.backends.base import Cancelled, GenerationBackend
from backend.config import settings
from backend.critic import CriticError, CriticUnavailable
from backend.grid import progress_steps
from backend.model_selection import current_model_id, current_settings
from backend.schemas import CriticStatus, GenerateRequest, PromptCritique, default_strength


class FakeBackend(GenerationBackend):
    """Runs the whole job machinery at a few milliseconds per step."""

    name = "fake"

    def __init__(self, step_seconds: float = 0.0, cfg_threshold: float | None = None, loaded: bool = True):
        self._profile_model = current_model_id(settings)
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
        self.negative_prompt_supported = True
        # A capable stand-in by default: the pipeline takes reference images
        # and an adapter can be applied, so a test that cares about either
        # turns it off rather than every other test having to turn it on.
        self.reference_supported = True
        # And it has an image-to-image sibling, so a refine pass runs.
        self.refine_supported = True
        # And an inpaint class: an inpaint repaints only under its mask.
        self.inpaint_supported = True
        # And a flow schedule a requested mu can reach. Off, the mu goes
        # from the row, as the real backend's `applied()` drops it.
        self.mu_supported = True
        self.lora_reason: str | None = None
        # Adapters that do not fit the claimed checkpoint, by name, with the
        # reason the real backend would give: hidden by the page, refused at
        # submit. Everything else fits.
        self.lora_mismatches: dict[str, str] = {}
        self.prompt_expander_supported = False
        self.expansions: list[tuple[str, int, int]] = []
        # The rewriter as the real backend reports it: not on disk until a
        # first expansion has fetched it, in memory from then until Unload.
        self.expander_cached = False
        self.expander_loaded = False
        # An exception the next expansions raise, for the route's failures.
        self.expand_error: Exception | None = None
        # Calls (1-based) that raise instead of rendering, and a hook run after
        # each render - for a cancel that lands after the last step.
        self.fail_on: set[int] = set()
        # The (allocated, reserved) bytes `last_peak()` answers after a call
        # that returned: None, as on a CPU, unless a test sets it.
        self.peak: tuple[int, int] | None = None
        self._last_peak: tuple[int, int] | None = None
        self.after_generate = None

    def _sync_model_profile(self) -> None:
        """Switch the fake's capabilities with the server's selected model.

        Initial capabilities stay deliberately generous so individual tests
        can set just the flag they exercise. A runtime switch, unlike those
        test overrides, takes the new checkpoint's complete profile, as the
        browser fake's startup already does.
        """
        model_id = current_model_id(settings)
        if model_id == self._profile_model:
            return
        from backend.backends.loading import lora_refusal
        from backend.model_defaults import defaults_for

        profile = defaults_for(model_id)
        self._profile_model = model_id
        for key in (
            "cfg_threshold", "true_cfg_supported", "negative_prompt_supported",
            "reference_supported", "refine_supported", "inpaint_supported", "mu_supported",
        ):
            setattr(self, key, profile[key])
        self.lora_reason = lora_refusal(model_id)
        self.prompt_expander_supported = "ernie-image" in model_id.lower().replace("_", "-")

    def lora_supported(self) -> bool:
        self._sync_model_profile()
        return self.lora_reason is None

    def lora_unavailable_reason(self) -> str | None:
        self._sync_model_profile()
        return self.lora_reason

    def lora_mismatch(self, name: str) -> str | None:
        return self.lora_mismatches.get(name)

    def prompt_expansion_supported(self) -> bool:
        self._sync_model_profile()
        return self.prompt_expander_supported

    def expand_prompt(self, prompt: str, width: int, height: int, should_cancel=None) -> str:
        if should_cancel is not None and should_cancel():
            raise Cancelled("prompt expansion stopped")
        self.expansions.append((prompt, width, height))
        if self.expand_error is not None:
            raise self.expand_error
        self.expander_cached = self.expander_loaded = True
        return f"Expanded: {prompt} ({width}x{height})"

    def prompt_expander_status(self) -> dict | None:
        if not self.prompt_expander_supported:
            return None
        fetch = [] if self.expander_cached else [("ERNIE's prompt rewriter", 7_663_346_520)]
        return {"loaded": self.loaded and self.expander_loaded, "fetch": fetch}

    def applied(self, request):
        from backend.execution import require_configured_loader_available, require_default_execution

        require_default_execution(request.execution)
        require_configured_loader_available(current_settings(settings))
        self._sync_model_profile()
        # What this fake would drop, the way the real backend drops it: an
        # input it cannot use must not reach the row.
        changes = {}
        if request.reference and not self.reference_supported:
            changes["reference"] = []
        if request.mu is not None and not self.mu_supported:
            changes["mu"] = None
        if request.lora and not self.lora_supported():
            changes["lora"] = None
        inpaint = request.inpainting
        if request.refine is not None and not (
            self.inpaint_supported if inpaint else self.refine_supported
        ):
            changes["refine"] = None
            changes["strength"] = default_strength()
        if request.mask is not None and (request.refine is None or "refine" in changes):
            changes["mask"] = None
        if request.outpaint is not None and (request.refine is None or "refine" in changes):
            changes["outpaint"] = None
        if changes:
            request = request.model_copy(update=changes)
        if self.cfg_threshold is None:
            return request
        return request.applied(
            negative_prompt_supported=self.negative_prompt_supported,
            cfg_threshold=self.cfg_threshold,
            mu_supported=self.mu_supported,
            true_cfg_supported=self.true_cfg_supported,
            reference_supported=self.reference_supported,
            lora_supported=self.lora_supported(),
            refine_supported=self.refine_supported,
            inpaint_supported=self.inpaint_supported,
        )

    def load(self) -> None:
        from backend.execution import require_configured_loader_available

        require_configured_loader_available(current_settings(settings))
        self._sync_model_profile()
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False
        self.expander_loaded = False

    def is_ready(self) -> bool:
        return self.loaded

    def generate(self, request, seed, on_progress=None, should_cancel=None, on_preview=None):
        from backend.backends import Cancelled
        from backend.execution import require_configured_loader_available, require_default_execution

        require_default_execution(request.execution)
        require_configured_loader_available(current_settings(settings))
        if not self.loaded:
            # The real backend's contract: loading happens in main.ensure_loaded
            # and nowhere else, so a job that reaches here unloaded is a bug.
            raise RuntimeError("model is not loaded")
        self._last_peak = None
        self.requests.append(request)
        self.seeds.append(seed)
        if len(self.requests) in self.fail_on:
            raise RuntimeError(f"cell {len(self.requests)} blew up")
        # A refine pass runs the part of the schedule its strength leaves,
        # once per image, as the real backend's does.
        steps = progress_steps(request)
        for step in range(steps):
            if self.step_seconds:
                time.sleep(self.step_seconds)
            if on_progress is not None:
                on_progress(step + 1, steps)
            # The contract every backend signs up to: poll between steps.
            if should_cancel is not None and should_cancel():
                raise Cancelled
        if self.after_generate is not None:
            self.after_generate()
        self._last_peak = self.peak
        if request.refine is not None and request.mask is not None:
            return self._inpaint(request, seed)
        # At the size that was asked for: the row records the image's own
        # dimensions, so anything matching on them - resume, the duration
        # estimate - needs a double that does not quietly render 16x16.
        return [
            Image.new("RGB", (request.width, request.height)) for _ in range(request.num_images)
        ]

    def last_peak(self):
        return self._last_peak

    @staticmethod
    def _inpaint(request, seed):
        """The source with only its masked pixels changed: a flat fill by each
        image's seed, through the real composite, so a test can read the
        result against the source."""
        from backend import refine

        # An outpaint's picture is its padded source, so the kept rectangle
        # reads against the source and the new area against the seed's fill.
        if request.outpaint is not None:
            source = refine.outpaint_source(request.refine, request.outpaint, request.width, request.height)
        else:
            source = refine.inpaint_source(request.refine, request.width, request.height)
        mask = refine.mask_image(request.mask, request.width, request.height)
        return [
            refine.composite(mask, source, Image.new("RGB", source.size, fake_fill(seed + offset)))
            for offset in range(request.num_images)
        ]

    def describe(self) -> dict:
        self._sync_model_profile()
        selected = current_settings(settings)
        return {
            "model_id": selected.model_id,
            "quantization": selected.quantization,
            "offload": selected.offload,
            "cfg_threshold": None,
            "reference_supported": self.reference_supported,
            "refine_supported": self.refine_supported,
            "inpaint_supported": self.inpaint_supported,
            "lora_supported": self.lora_supported(),
        }

    def free_cache(self) -> None:
        self.cache_freed = getattr(self, "cache_freed", 0) + 1


def fake_fill(seed: int) -> tuple[int, int, int]:
    """The colour the fake inpaint paints a masked region for this seed."""
    return (seed * 37 % 256, seed * 91 % 256, seed * 53 % 256)


class FakeTranslator:
    def __init__(self) -> None:
        self.translations: list[str] = []
        self.unloads = 0
        self.fail = False
        # Not on disk until a first translation has fetched it.
        self.downloaded = False

    def translate(self, text: str, should_cancel=None) -> str:
        if should_cancel is not None and should_cancel():
            raise Cancelled("prompt expansion stopped")
        self.translations.append(text)
        if self.fail:
            raise RuntimeError("translation broke")
        self.downloaded = True
        return f"English: {text}"

    def cached(self) -> bool:
        return self.downloaded

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
    """`critic.PromptCritic` without llama.cpp or a model: the same six
    methods, and a deterministic answer written as the model writes one - raw
    JSON - then put through the real `critic.validate`, so what reaches the
    page is only what the real pipeline could return. A prompt too long for
    the real critic's window is refused by the real `critic.check_fits`,
    before any answer - on the unscaled estimate, standing in for the
    server's count, so near the window's edge the two can disagree.

    `available` is what `supported()` answers and `fetched` what `ready()`
    does; `prepare()` sets `fetched`, as a first real prepare puts both files
    on disk. `describe()` gives `label` - the name its answers carry too -
    and, while `pinned`, the pinned pairing's two downloads until then;
    `pinned` False stands for an override, which is never a download.
    `fail` makes `critique()` raise - True for a `CriticError`, or an
    exception to raise instead - and `prepare_fail` the same for `prepare()`.
    `delay` holds `critique()` for that many seconds, so a caller can act
    while it runs; `started` is set as each one begins.

    Its `reordered` moves the last comma clause to the front, which
    `critic.is_reorder` accepts; a prompt of one clause has nothing to move,
    so an ordered checkpoint gets no reordered version, as it often does from
    the real critic. Where Check placed the tags itself (`order_checked`),
    `critic.validate` keeps it only as Check's own order - or, where Check
    could not place every tag (one never looked up or unknown to the tag
    index), as an order Check's order rule finds nothing wrong with - and
    says which it kept (`reordered_kind`).
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.available = True
        self.fetched = False
        self.label = "fake critic"
        self.pinned = True
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

    def describe(self) -> CriticStatus:
        build = critic.BUILDS["win-cuda-13.4-x64"]
        fetch = [] if self.fetched or not self.pinned else [
            (f"llama.cpp {critic.RELEASE}", sum(archive.size for archive in build.archives)),
            (self.label, critic.MODEL_SIZE),
        ]
        return CriticStatus(
            label=self.label,
            pinned=self.pinned,
            fetch=[name for name, _ in fetch],
            fetch_bytes=sum(size for _, size in fetch),
        )

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
        try:
            # Built and sized before the wait. The real critic refuses before
            # a server starts only what its estimate, scaled down, says cannot
            # fit, and the rest on the server's count; here the estimate,
            # unscaled, stands in for that count. A question whose answer
            # cannot fit the window is refused at once, a 502, never answered.
            context = critic.build_context(model_id, request, applied, analysis)
            critic.check_fits(context)
            if self.delay:
                time.sleep(self.delay)
            if self.fail is True:
                raise CriticError("llama-server exited with code 1 before it was ready")
            if self.fail:
                raise self.fail
            return critic.validate(
                self.answer(request.prompt, context.ordered),
                context,
                request.prompt,
                model=self.label,
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
