"""ERNIE-Image Turbo: prompt expansion is visible, and mu has no way in.

`ErnieImagePipeline` lists a 7.7 GB Ministral prompt rewriter (`pe`) among its
components and, unless `use_pe` is off, samples a longer prompt through it at
temperature 0.6 before every image - so the prompt the row records would not
be the prompt the model saw, and no seed would bring the image back. The
loader leaves the component out initially (`pe=None`, which diffusers allows
for an optional component and which keeps its folder out of the download), the
explicit expansion action loads it, and generate() turns the hidden switch off
regardless. The pipeline also builds its own
linspace schedule and takes no `sigmas`, over a scheduler shipped static at
shift 4, so a requested mu is dropped and recorded as dropped; and its latents
are FLUX.2's, for which no preview map has been fitted.
"""
from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from backend.backends import diffusers_backend as module
from backend.backends import loading
from backend.backends.diffusers_backend import DiffusersBackend
from backend.schemas import GenerateRequest

STEPS = 4

ERNIE_INDEX = {
    "_class_name": "ErnieImagePipeline",
    "pe": ["transformers", "Ministral3ForCausalLM"],
    "pe_tokenizer": ["transformers", "TokenizersBackend"],
    "text_encoder": ["transformers", "Mistral3Model"],
    "transformer": ["diffusers", "ErnieImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


class ErnieLikePipeline:
    """ErnieImagePipeline as the backend sees it: a static shift-4 scheduler
    fed a plain linspace, no `sigmas`, a `use_pe` switch, FLUX.2's VAE, and
    4-D patchified latents in the callback."""

    def __init__(self, shift: float = 4.0):
        self.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False, shift=shift)
        config = type("Config", (), {"latent_channels": 32})()
        self.vae = type("AutoencoderKLFlux2", (), {"config": config})()
        self.calls: list[dict] = []
        self.expansions: list[dict] = []
        self.pe = None
        self.pe_tokenizer = None
        self.hooks_freed = 0
        self.steps_run = 0

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def register_modules(self, **components):
        for name, component in components.items():
            setattr(self, name, component)

    def _enhance_prompt_with_pe(self, prompt, device, width=1024, height=1024):
        assert self.pe is not None and self.pe_tokenizer is not None
        self.expansions.append(
            {"prompt": prompt, "device": device, "width": width, "height": height}
        )
        return f"Expanded {prompt} for {width}x{height}"

    def maybe_free_model_hooks(self):
        self.hooks_freed += 1

    def to(self, device):
        return self

    def __call__(self, prompt, negative_prompt="", height=1024, width=1024, num_inference_steps=50,
                 guidance_scale=4.0, num_images_per_prompt=1, generator=None,
                 callback_on_step_end=None, use_pe=True):
        self.calls.append({"use_pe": use_pe, "negative_prompt": negative_prompt})
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1])
        latents = torch.randn(num_images_per_prompt, 128, height // 16, width // 16)
        for step in range(num_inference_steps):
            self.steps_run += 1
            callback_on_step_end(self, step, None, {"latents": latents})
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def _index(monkeypatch, tmp_path, index: dict) -> None:
    path = tmp_path / "model_index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: str(path))


def _backend(pipe) -> DiffusersBackend:
    backend = DiffusersBackend()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend


def _request(**kwargs) -> GenerateRequest:
    base = dict(prompt="a lighthouse in fog", width=256, height=256, steps=STEPS)
    return GenerateRequest(**{**base, **kwargs})


def test_the_rewriter_is_named_from_the_index(monkeypatch, tmp_path):
    _index(monkeypatch, tmp_path, ERNIE_INDEX)
    class_name, unused = loading.unused_components("baidu/ERNIE-Image-Turbo")
    assert class_name == "ErnieImagePipeline"
    assert unused == {"pe": None, "pe_tokenizer": None}
    # Only what the repo carries: a re-upload without the rewriter has nothing
    # to leave out, and another family's index names nothing at all.
    _index(monkeypatch, tmp_path, {k: v for k, v in ERNIE_INDEX.items() if not k.startswith("pe")})
    assert loading.unused_components("x")[1] == {}
    _index(monkeypatch, tmp_path, {"_class_name": "FluxPipeline", "transformer": ["diffusers", "F"]})
    assert loading.unused_components("x") == ("FluxPipeline", {})


def test_an_unreadable_index_leaves_the_load_alone():
    # conftest's stub refuses the Hub; the lookup is an optimisation, not a gate.
    assert loading.unused_components("nobody/whatever") == (None, {})


def test_only_the_concrete_class_takes_a_none_component():
    # Why load() switches class: the base class filters passed components
    # against its own empty `_optional_components`, so through it `pe=None`
    # is a pipeline kwarg and the folder is downloaded and loaded (seen on
    # the card, diffusers 0.40). The concrete class counts them as passable.
    import diffusers

    own, _ = diffusers.ErnieImagePipeline._get_signature_keys(diffusers.ErnieImagePipeline)
    assert {"pe", "pe_tokenizer"} <= set(own)


def test_load_hands_the_rewriter_to_diffusers_as_none(monkeypatch, tmp_path):
    import diffusers

    _index(monkeypatch, tmp_path, ERNIE_INDEX)
    seen: dict = {}
    pipe = ErnieLikePipeline()

    def from_pretrained(model_id, **kwargs):
        seen.update(kwargs)
        return pipe

    def never(*args, **kwargs):
        raise AssertionError("the base class would download and load the rewriter")

    monkeypatch.setattr(diffusers.ErnieImagePipeline, "from_pretrained", from_pretrained)
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", never)
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(
            module.settings, quantization="none", offload="none", device="cpu",
            model_id="baidu/ERNIE-Image-Turbo",
        ),
    )
    backend = DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert "pe" in seen and seen["pe"] is None
    assert "pe_tokenizer" in seen and seen["pe_tokenizer"] is None


def test_prompt_expander_components_follow_the_manifest(monkeypatch, tmp_path):
    import transformers

    _index(monkeypatch, tmp_path, ERNIE_INDEX)
    calls: list[tuple[str, dict]] = []

    class Component:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return (model_id, kwargs["subfolder"])

    monkeypatch.setattr(transformers, "Ministral3ForCausalLM", Component)
    monkeypatch.setattr(transformers, "TokenizersBackend", Component)
    components = loading.prompt_expander_components("baidu/ERNIE-Image-Turbo")
    assert components == {
        "pe": ("baidu/ERNIE-Image-Turbo", "pe"),
        "pe_tokenizer": ("baidu/ERNIE-Image-Turbo", "pe_tokenizer"),
    }
    by_folder = {kwargs["subfolder"]: kwargs for _, kwargs in calls}
    assert by_folder["pe"]["dtype"] is torch.bfloat16
    assert "dtype" not in by_folder["pe_tokenizer"]


def test_expand_loads_once_then_returns_visible_size_specific_text(monkeypatch):
    pipe = ErnieLikePipeline()
    backend = _backend(pipe)
    loads = 0

    def components(model_id):
        nonlocal loads
        loads += 1
        return {"pe": object(), "pe_tokenizer": object()}

    monkeypatch.setattr(loading, "prompt_expander_components", components)
    monkeypatch.setattr(loading, "place_pipeline", lambda current, settings: None)
    assert backend.prompt_expansion_supported()
    assert backend.expand_prompt("a fox", 1264, 848) == "Expanded a fox for 1264x848"
    assert backend.expand_prompt("a fox", 1024, 1024) == "Expanded a fox for 1024x1024"
    assert loads == 1
    assert pipe.expansions[0] == {
        "prompt": "a fox", "device": torch.device("cpu"), "width": 1264, "height": 848
    }
    assert pipe.hooks_freed == 2


def test_the_status_names_the_rewriters_download_until_it_is_loaded(monkeypatch):
    # The page names a first Expand's 7.7 GB before the press that starts it:
    # off disk, the rewriter is what a first use fetches; registered on the
    # pipeline, an Expand waits for nothing but itself.
    pipe = ErnieLikePipeline()
    backend = _backend(pipe)
    cached = {"value": False}
    monkeypatch.setattr(loading, "prompt_expander_cached", lambda model_id: cached["value"])
    assert backend.prompt_expander_status() == {
        "loaded": False, "fetch": [("ERNIE's prompt rewriter", loading.PROMPT_EXPANDER_BYTES)],
    }
    cached["value"] = True
    assert backend.prompt_expander_status() == {"loaded": False, "fetch": []}
    pipe.register_modules(pe=object(), pe_tokenizer=object())
    cached["value"] = False
    assert backend.prompt_expander_status() == {"loaded": True, "fetch": []}
    # A pipeline with no rewriter has no status to give.
    backend._call_params = set()
    assert backend.prompt_expander_status() is None


def test_the_rewriters_cache_check_reads_files_only(monkeypatch, tmp_path):
    import huggingface_hub

    tree = tmp_path / "ernie"
    (tree / "pe").mkdir(parents=True)
    assert not loading.prompt_expander_cached(str(tree))
    (tree / "pe" / "model.safetensors").write_bytes(b"weights")
    assert loading.prompt_expander_cached(str(tree))
    # A Hub id is the cache's to answer, never the network's.
    asked: list[tuple[str, str]] = []

    def cache(repo_id, filename, **kwargs):
        asked.append((repo_id, filename))
        return str(tree / "pe" / "model.safetensors") if repo_id == "baidu/ERNIE-Image-Turbo" else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cache)
    assert loading.prompt_expander_cached("baidu/ERNIE-Image-Turbo")
    assert not loading.prompt_expander_cached("lite-infer/elsewhere")
    assert asked == [
        ("baidu/ERNIE-Image-Turbo", "pe/model.safetensors"),
        ("lite-infer/elsewhere", "pe/model.safetensors"),
    ]


def test_cold_prompt_expansion_support_follows_the_model_id(monkeypatch):
    backend = DiffusersBackend()
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, model_id="baidu/ERNIE-Image-Turbo"),
    )
    assert backend.prompt_expansion_supported()
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, model_id="black-forest-labs/FLUX.1-dev"),
    )
    assert not backend.prompt_expansion_supported()


def test_generate_turns_the_rewriter_off_where_the_switch_exists():
    pipe = ErnieLikePipeline()
    backend = _backend(pipe)
    backend.generate(_request(), seed=1)
    assert pipe.calls[-1]["use_pe"] is False


def test_a_pipeline_without_the_switch_is_not_handed_one():
    # The stub's signature has no `use_pe` and no **kwargs, so a stray one
    # would be a TypeError rather than a silent no-op.
    class Switchless(ErnieLikePipeline):
        def __call__(self, prompt, height, width, num_inference_steps, guidance_scale,
                     num_images_per_prompt, generator, callback_on_step_end):
            return super().__call__(prompt, "", height, width, num_inference_steps, guidance_scale,
                                    num_images_per_prompt, generator, callback_on_step_end)

    pipe = Switchless()
    backend = _backend(pipe)
    assert len(backend.generate(_request(), seed=1)) == 1


def test_a_requested_mu_has_no_way_in_and_is_recorded_as_dropped():
    pipe = ErnieLikePipeline(shift=4.0)
    backend = _backend(pipe)
    assert backend.applied(_request(mu=1.15)).mu is None
    backend.generate(_request(mu=1.15), seed=1)
    # What lands is the pipeline's own linspace under the shipped static shift
    # - read off the scheduler, and untouched by the request.
    ramp = np.linspace(1.0, 0.0, STEPS + 1)[:-1]
    expected = 4.0 * ramp / (1 + 3.0 * ramp)
    assert [float(s) for s in pipe.scheduler.sigmas[:-1]] == pytest.approx(list(expected), abs=1e-6)


def test_the_negative_prompt_applies_above_one():
    pipe = ErnieLikePipeline()
    backend = _backend(pipe)
    assert backend.cfg_threshold() == 1.0
    backend.generate(_request(negative_prompt="rain", guidance_scale=1.0), seed=1)
    assert pipe.calls[-1]["negative_prompt"] == "", "dropped at the distilled 1.0"
    backend.generate(_request(negative_prompt="rain", guidance_scale=2.0), seed=1)
    assert pipe.calls[-1]["negative_prompt"] == "rain"


def test_no_preview_without_a_map_for_the_flux2_vae_and_the_cell_completes(monkeypatch):
    from backend import latent_preview

    monkeypatch.setattr(latent_preview, "INTERVAL_SECONDS", 0.0)
    pipe = ErnieLikePipeline()
    backend = _backend(pipe)
    frames: list = []
    images = backend.generate(_request(), seed=1, on_preview=lambda *a: frames.append(a))
    assert frames == [] and len(images) == 1 and pipe.steps_run == STEPS


def test_the_pages_prompt_length_for_expand_is_the_requests():
    # web/app.js refuses a prompt past this in the panel's words, at the box,
    # before sending it; one copy to drift and the page would send what the
    # server refuses in pydantic's, or refuse what it would take.
    import re
    from pathlib import Path

    from backend.schemas import PromptExpansionRequest

    source = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")
    match = re.search(r"^const EXPAND_MAX_CHARS = (\d+);$", source, re.M)
    field = PromptExpansionRequest.model_fields["prompt"]
    limits = [m.max_length for m in field.metadata if hasattr(m, "max_length")]
    assert match and limits == [int(match.group(1))]
