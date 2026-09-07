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
    hints = hints_for(model_id)
    assert set(hints) == {"steps", "guidance_scale", "mu"}
    assert all(text and "{" not in text for text in hints.values())


def test_flux_one_is_unchanged():
    profile = defaults_for("black-forest-labs/FLUX.1-Krea-dev")
    assert (profile["steps"], profile["guidance_scale"]) == (28, 3.5)
    assert profile["negative_prompt_supported"] is True


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
