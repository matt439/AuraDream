"""FLUX.2 [klein] through the same loader: the profile, the negative prompt, the VAE.

klein's ids contain "flux", so without their own entries the profile matcher
would seed 28 steps at CFG 3.5 for a model distilled to 4 steps that ignores
guidance. Its pipeline also takes no `negative_prompt` and puts the VAE tiling
switches on the VAE rather than on the pipeline.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest
from PIL import Image

from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import defaults_for, hints_for
from backend.schemas import GenerateRequest

KLEIN = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_BASE = "black-forest-labs/FLUX.2-klein-base-4B"
KLEIN_LITE = "lite-infer/flux.2-klein-9b-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"


@pytest.mark.parametrize(
    "model_id, steps, cfg",
    [(KLEIN, 4, 1.0), (KLEIN_LITE, 4, 1.0), (KLEIN_BASE, 50, 4.0)],
)
def test_klein_profiles_win_over_flux(model_id, steps, cfg):
    profile = defaults_for(model_id)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (steps, cfg, None)
    assert profile["negative_prompt_supported"] is False
    # klein is the one family here whose pipeline takes reference images.
    assert profile["reference_supported"] is True
    hints = hints_for(model_id)
    assert set(hints) == {"steps", "guidance_scale", "mu", "true_cfg", "reference", "prompt", "size"}
    assert all(text and "{" not in text for text in hints.values())


def test_flux_one_is_unchanged():
    profile = defaults_for("black-forest-labs/FLUX.1-Krea-dev")
    assert (profile["steps"], profile["guidance_scale"]) == (28, 3.5)
    assert profile["negative_prompt_supported"] is True
    # FLUX.1's pipeline has no `image` parameter: the ids contain "flux", and
    # the two klein entries sit ahead of it precisely so this stays False.
    assert profile["reference_supported"] is False


class KleinLikePipeline:
    """A `__call__` with no `negative_prompt`, and a VAE that owns the tiling switches."""

    def __init__(self):
        self.calls: list[dict] = []
        self.vae = type("Vae", (), {"tiled": False, "sliced": False})()
        self.vae.enable_tiling = lambda: setattr(self.vae, "tiled", True)
        self.vae.enable_slicing = lambda: setattr(self.vae, "sliced", True)

    def to(self, device):
        return self

    def __call__(
        self,
        prompt,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        callback_on_step_end,
        sigmas=None,
    ):
        self.calls.append(dict(prompt=prompt, guidance_scale=guidance_scale))
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def test_a_negative_prompt_is_never_sent_to_a_pipeline_without_one():
    backend = DiffusersBackend()
    pipe = KleinLikePipeline()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    described = backend.describe()
    assert described["negative_prompt_supported"] is False
    # No true_cfg_scale and no negative_prompt: the classic `guidance_scale > 1`
    # gate, which is also where the distilled model starts warning.
    assert described["cfg_threshold"] == 1.0

    request = GenerateRequest(
        prompt="a lighthouse in fog", negative_prompt="blurry", width=256, height=256,
        steps=4, guidance_scale=4.0,
    )
    images = backend.generate(request, seed=1)
    assert len(images) == 1
    assert "negative_prompt" not in pipe.calls[0]


class SD3LikePipeline(KleinLikePipeline):
    """Takes a negative prompt, no true_cfg_scale, and gates guidance at > 1 -
    like 122 of the 131 diffusers pipelines with that shape."""

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, sigmas=None,
                 negative_prompt=None):
        return super().__call__(prompt, width, height, num_inference_steps, guidance_scale,
                                num_images_per_prompt, generator, callback_on_step_end, sigmas)


class Krea2Pipeline(SD3LikePipeline):
    """Same signature; the one family that guides from 0."""


def _loaded(pipe):
    backend = DiffusersBackend()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend


class FluxLikePipeline(KleinLikePipeline):
    """`guidance_scale` is a distilled input and `true_cfg_scale` the real CFG -
    FluxPipeline's shape, the one place the two are separate knobs."""

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, sigmas=None,
                 negative_prompt=None, true_cfg_scale=1.0):
        self.calls.append(dict(
            prompt=prompt, guidance_scale=guidance_scale, negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
        ))
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


def test_real_cfg_is_its_own_knob_where_the_pipeline_keeps_it_apart():
    # The two used to be set from one number, so the only reachable states on
    # FLUX.1 were (3.5, off) and (g, g): a negative prompt typed at the
    # recommended 3.5 stacked guidance twice at double the cost, and the usual
    # "embedding at 3.5, mild real CFG of 1.5" could not be asked for.
    backend = _loaded(FluxLikePipeline())
    pipe = backend._pipe
    assert backend.describe()["true_cfg_supported"] is True
    assert backend.cfg_threshold() == 1.0

    base = dict(prompt="a lighthouse", negative_prompt="blurry", width=256, height=256, steps=4)
    # Real CFG set: the negative prompt runs at *that* scale, the guidance
    # value goes through untouched, and the record says both.
    ran = backend.applied(GenerateRequest(**base, guidance_scale=3.5, true_cfg=1.5))
    assert (ran.guidance_scale, ran.true_cfg, ran.negative_prompt) == (3.5, 1.5, "blurry")
    backend.generate(GenerateRequest(**base, guidance_scale=3.5, true_cfg=1.5), seed=1)
    assert pipe.calls[-1]["true_cfg_scale"] == 1.5 and pipe.calls[-1]["guidance_scale"] == 3.5
    assert pipe.calls[-1]["negative_prompt"] == "blurry"
    # Not set: the guidance value no longer decides. The negative prompt is
    # dropped - and recorded as dropped - however high the guidance value.
    ran = backend.applied(GenerateRequest(**base, guidance_scale=3.5))
    assert (ran.negative_prompt, ran.true_cfg) == (None, None)
    backend.generate(GenerateRequest(**base, guidance_scale=3.5), seed=1)
    assert pipe.calls[-1]["negative_prompt"] is None and pipe.calls[-1]["true_cfg_scale"] == 1.0
    # At the threshold it is off too, and without a negative prompt there is
    # no uncond pass for the scale to describe.
    assert backend.applied(GenerateRequest(**base, true_cfg=1.0)).true_cfg is None
    assert backend.applied(GenerateRequest(prompt="x", width=256, height=256, true_cfg=3.0)).true_cfg is None


def test_where_cfg_is_real_already_the_record_says_which_scale_ran():
    # Krea 2 has no separate knob: `guidance_scale` is the uncond pass. The
    # column then carries that scale whenever the negative prompt applied, so
    # it means one thing on every checkpoint - and a `true_cfg` typed at a
    # pipeline that cannot take one is not recorded as if it had.
    backend = _loaded(Krea2Pipeline())
    assert backend.describe()["true_cfg_supported"] is False
    base = dict(prompt="a lighthouse", negative_prompt="blurry", width=256, height=256, steps=4)
    ran = backend.applied(GenerateRequest(**base, guidance_scale=2.0, true_cfg=5.0))
    assert (ran.negative_prompt, ran.true_cfg) == ("blurry", 2.0)
    ran = backend.applied(GenerateRequest(**base, guidance_scale=0.0, true_cfg=5.0))
    assert (ran.negative_prompt, ran.true_cfg) == (None, None)


def test_having_a_negative_prompt_does_not_mean_guiding_from_zero():
    # "Has negative_prompt -> 0.0" was a Krea2-only rule applied to every
    # pipeline: SD3 answered 0.0 while its own gate is > 1, so loading it made
    # /api/status less accurate than the profile's guess.
    assert _loaded(SD3LikePipeline()).cfg_threshold() == 1.0
    assert _loaded(Krea2Pipeline()).cfg_threshold() == 0.0


def test_the_cfg_hint_follows_the_pipelines_negative_prompt_answer():
    # A FLUX-profile id whose pipeline takes no negative prompt: the field is
    # greyed "because this checkpoint takes no negative prompt" and the badge
    # must not say the negative prompt applies above 1.
    flux = "black-forest-labs/FLUX.1-Krea-dev"
    assert "negative prompt applies" not in hints_for(flux, 1.0, False)["guidance_scale"]
    assert "takes no negative prompt" in hints_for(flux, 1.0, False)["guidance_scale"]
    # And the profile's own text when the pipeline agrees with it.
    assert hints_for(flux, 1.0, True) == hints_for(flux, 1.0)


class OldCallbackPipeline(KleinLikePipeline):
    """The pre-0.25 diffusers API: `callback`/`callback_steps`, and no generator list."""

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, callback=None, callback_steps=1):
        return super().__call__(prompt, width, height, num_inference_steps, guidance_scale,
                                num_images_per_prompt, None, None)


def test_load_refuses_a_pipeline_without_the_callback_or_generator(monkeypatch):
    # Dropped silently, the first meant a progress bar stuck at 0 and a Cancel
    # that never landed; the second, unseeded noise under a recorded seed.
    import diffusers

    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", lambda *a, **k: OldCallbackPipeline())
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, quantization="none", offload="none", device="cpu"),
    )
    with pytest.raises(RuntimeError, match="callback_on_step_end"):
        DiffusersBackend().load()


class OffloadingPipeline(KleinLikePipeline):
    """Records the device the offload hooks were told to use."""

    def __init__(self):
        super().__init__()
        self.offload_device = "unset"

    def enable_model_cpu_offload(self, gpu_id=None, device=None):
        self.offload_device = device


def test_load_offloads_onto_the_configured_device(monkeypatch):
    # Given no device, diffusers resolves its own accelerator - always GPU 0 -
    # so `LOCAL_DIFFUSION_DEVICE=cuda:1` loaded onto cuda:0 while describe() reported
    # that card, and nothing on screen said the setting was dropped.
    import diffusers
    import torch

    pipe = OffloadingPipeline()
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", lambda *a, **k: pipe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, quantization="none", offload="model", device="cuda:1"),
    )
    DiffusersBackend().load()
    assert pipe.offload_device == torch.device("cuda:1")

    # And the status line reads that card's name and memory, not GPU 0's.
    asked = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (asked.append(index), (1 << 30, 2 << 30))[1])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"card {index}")
    info = DiffusersBackend().describe()
    assert info["device_name"] == "card 1" and asked == [1]
    assert (info["vram_total_mb"], info["vram_used_mb"]) == (2048, 1024)


def test_load_reaches_for_the_vaes_own_tiling_switches(monkeypatch):
    import diffusers

    pipe = KleinLikePipeline()
    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", lambda *a, **k: pipe)
    monkeypatch.setattr(
        module,
        "settings",
        dataclasses.replace(module.settings, quantization="none", offload="none", device="cpu"),
    )
    backend = DiffusersBackend()
    backend.load()
    assert backend.is_ready()
    assert pipe.vae.tiled and pipe.vae.sliced
    assert backend.describe()["negative_prompt_supported"] is False
