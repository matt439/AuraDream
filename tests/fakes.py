"""The tests' stand-ins for the GPU: a backend that draws nothing and a
translator that translates nothing, running the whole job machinery at a
few milliseconds per step. tools/fake_server.py builds the browser smoke
server on the same backend."""
from __future__ import annotations

import time

from PIL import Image

from backend.backends.base import GenerationBackend
from backend.config import settings
from backend.schemas import GenerateRequest


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
