"""Inpainting: a refine pass with a mask, through the inpaint class beside the
pipeline, composited back over the source.

What is covered here needs no GPU: which checkpoints have such a class (asked
of the installed diffusers) and which of those this app enables, that the
class is built over the loaded components and run one call per image with
the source at its own size and the stored mask, that the composite keeps the
source's pixels exactly where the mask is black, and the status the page
reads. What an inpaint does to pixels on a real checkpoint is the card's.
"""
from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import diffusers
import pytest
import torch
from PIL import Image

from backend import launch, refine
from backend.backends import Cancelled
from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import INPAINT_DEFERRED, _profile_for, defaults_for
from backend.schemas import GenerateRequest, default_strength

# `tests/` has no __init__.py; pytest puts this directory on sys.path.
from fakes import fake_fill
from test_profiles import CASES
from test_refine import RefinableMain, _loaded

# The five families diffusers 0.40 ships an inpaint class for, of the
# pipelines this app's catalogue loads.
INPAINTS = {
    "FluxPipeline": "FluxInpaintPipeline",
    "StableDiffusionXLPipeline": "StableDiffusionXLInpaintPipeline",
    "ZImagePipeline": "ZImageInpaintPipeline",
    "QwenImagePipeline": "QwenImageInpaintPipeline",
    "Flux2KleinPipeline": "Flux2KleinInpaintPipeline",
}
# And the two this pass enables.
ENABLED = {"StableDiffusionXLPipeline", "ZImagePipeline"}

ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


# ---------------------------------------------------------------- the class


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_the_inpaint_sibling_is_the_installed_diffusers_answer(model_id, class_name):
    found = refine.sibling(getattr(diffusers, class_name), "Inpaint")
    assert (found.__name__ if found else None) == INPAINTS.get(class_name), model_id
    if found is not None:
        own = set(inspect.signature(getattr(diffusers, class_name).__call__).parameters)
        params = set(inspect.signature(found.__call__).parameters)
        assert refine.INPAINT_REQUIRED <= params and "mask_image" in refine.INPAINT_REQUIRED
        assert (refine.CARRIED & own) <= params


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_the_profile_enables_exactly_the_undeferred_classes(model_id, class_name):
    # A True needs an installed exact-export class and a family not deferred;
    # a class the profile leaves off has to be a deliberate deferral, never
    # an oversight - or a family diffusers gained a class for goes unnoticed.
    found = refine.sibling(getattr(diffusers, class_name), "Inpaint")
    enabled = defaults_for(model_id)["inpaint_supported"]
    if enabled:
        assert found is not None and class_name not in INPAINT_DEFERRED, model_id
    elif found is not None:
        assert class_name in INPAINT_DEFERRED, model_id
    assert enabled is (class_name in ENABLED), model_id
    # A False is final before a load (a load can only take one away).
    assert defaults_for(model_id)["inpaint_settled"] is (not enabled)


def test_every_catalogue_checkpoint_is_one_of_the_cases():
    # So the two tests above speak for the whole catalogue: each entry (and
    # each Nunchaku repo) resolves to a profile some case resolves to, or is
    # one of this repo's own pipelines, beside which diffusers has nothing.
    from backend.model_defaults import _OWN_PIPELINES

    profiles = {id(_profile_for(model_id)) for model_id, _ in CASES}
    repos = [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]
    for repo in repos:
        if repo.lower() in _OWN_PIPELINES:
            assert not defaults_for(repo)["inpaint_supported"], repo
            continue
        assert id(_profile_for(repo)) in profiles, repo
    # The deferred list names real pipeline classes with real inpaint classes.
    for name in INPAINT_DEFERRED:
        assert refine.sibling(getattr(diffusers, name), "Inpaint") is not None, name
    # And an unrecognised checkpoint is promised none.
    assert not defaults_for("nobody/whatever")["inpaint_supported"]


def test_the_local_pipelines_and_lookalikes_have_no_inpaint_class():
    from backend.backends.anima_pipeline import AnimaAestheticPipeline, AnimaTurboPipeline
    from backend.backends.newbie_pipeline import NewbiePipeline

    for cls in (AnimaTurboPipeline, AnimaAestheticPipeline, NewbiePipeline):
        assert refine.sibling(cls, "Inpaint") is None
    assert refine.sibling(type("StableDiffusionXLPipeline", (), {}), "Inpaint") is None
    # Never beside a class that is already one, of either kind.
    assert refine.sibling(diffusers.StableDiffusionXLInpaintPipeline, "Inpaint") is None
    assert refine.sibling(diffusers.StableDiffusionXLInpaintPipeline) is None
    # The image-to-image answer is unchanged by the second kind.
    assert refine.sibling(diffusers.StableDiffusionXLPipeline) is diffusers.StableDiffusionXLImg2ImgPipeline
    with pytest.raises(ValueError, match="no sibling kind 'Outpaint'"):
        refine.sibling(diffusers.StableDiffusionXLPipeline, "Outpaint")


@pytest.mark.parametrize("class_name", sorted(INPAINTS))
def test_build_constructs_each_inpaint_class_over_stand_in_components(class_name):
    # Every inpaint __init__ takes the text-to-image class's components; the
    # loaded pipeline's config fills a flag its constructor also takes.
    own = getattr(diffusers, class_name)
    cls = refine.sibling(own, "Inpaint")
    components = {
        name: None for name, param in inspect.signature(own.__init__).parameters.items()
        if name != "self" and param.default is param.empty
    }
    pipe = SimpleNamespace(components=components, config={"force_zeros_for_empty_prompt": False})
    built = refine.build(pipe, cls)
    assert type(built) is cls
    if "force_zeros_for_empty_prompt" in inspect.signature(cls.__init__).parameters:
        assert built.config["force_zeros_for_empty_prompt"] is False
    assert callable(built.image_processor.apply_overlay)


@pytest.mark.parametrize("class_name", sorted(INPAINTS))
def test_the_starting_strength_is_read_from_the_class(class_name):
    cls = getattr(diffusers, INPAINTS[class_name])
    default = inspect.signature(cls.__call__).parameters["strength"].default
    assert refine.library_strength(cls) == float(default)

    # Read, not copied: another default is another answer, and none is None.
    def call(self, image, mask_image, strength=0.37):
        pass

    assert refine.library_strength(type("X", (), {"__call__": call})) == 0.37
    assert refine.library_strength(type("Y", (), {"__call__": lambda self, image: None})) is None


# ---------------------------------------------------------------- the composite


def _halves(width=64, height=32) -> Image.Image:
    """Black on the left, white on the right, a grey column between."""
    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (width // 2, 0, width, height))
    mask.paste(128, (width // 2 - 1, 0, width // 2, height))
    return mask


def test_the_composite_keeps_the_source_where_the_mask_is_black():
    source = Image.effect_noise((64, 32), 64).convert("RGB")
    result = Image.new("RGB", (64, 32), (10, 200, 30))
    mask = _halves()
    out = refine.composite(mask, source, result)
    assert out.mode == "RGB" and out.size == source.size
    for y in range(0, 32, 5):
        for x in range(0, 31):
            assert out.getpixel((x, y)) == source.getpixel((x, y)), (x, y)
        for x in range(32, 64):
            assert out.getpixel((x, y)) == (10, 200, 30), (x, y)
    # The feathered column blends the two rather than picking one.
    edge, src = out.getpixel((31, 0)), source.getpixel((31, 0))
    assert all(min(a, b) - 1 <= c <= max(a, b) + 1 for c, a, b in zip(edge, src, (10, 200, 30)))
    # And it refuses pictures that do not line up rather than misplacing one.
    with pytest.raises(ValueError, match="cannot composite"):
        refine.composite(mask, source, Image.new("RGB", (32, 32)))


# ---------------------------------------------------------------- the request


def _applied(request, **flags):
    return request.applied(negative_prompt_supported=True, cfg_threshold=1.0, **flags)


MASK = "ab" * 32


def test_a_mask_goes_where_its_source_goes():
    request = GenerateRequest(prompt="a fox", steps=10, refine=3, strength=0.6, mask=MASK)
    kept = _applied(request, refine_supported=False, inpaint_supported=True)
    assert (kept.refine, kept.mask, kept.strength) == (3, MASK, 0.6)
    # An inpaint asks the inpaint class: a refine pass alone is not enough.
    dropped = _applied(request, refine_supported=True, inpaint_supported=False)
    assert (dropped.refine, dropped.mask, dropped.strength) == (None, None, default_strength())
    # A mask with no source covers nothing, and is not recorded.
    alone = request.model_copy(update={"refine": None})
    assert _applied(alone, refine_supported=True, inpaint_supported=True).mask is None
    # A refine pass without a mask still asks the image-to-image class only.
    plain = request.model_copy(update={"mask": None})
    assert _applied(plain, refine_supported=True, inpaint_supported=False).refine == 3
    # References have nowhere to go on SDXL's or Z-Image's inpaint either.
    referenced = request.model_copy(update={"reference": [5]})
    assert _applied(referenced, reference_supported=True, inpaint_supported=True).reference == []


@pytest.mark.parametrize("bad", ["AB" * 32, "ab" * 31, "../" + "a" * 61, "g" * 64])
def test_a_mask_is_a_hash_never_a_path(bad):
    with pytest.raises(ValueError):
        GenerateRequest(prompt="a fox", refine=3, mask=bad)


# ---------------------------------------------------------------- the backend


class InpaintStub:
    """The inpaint class's shape: the source, the mask and a strength beside
    the size, and the steps a strength leaves read off the installed class."""

    def __init__(self, scheduler, denoiser, vae, flag=False):
        self.scheduler, self.denoiser, self.vae, self.flag = scheduler, denoiser, vae, flag
        self.calls: list[dict] = []
        self.finished = 0

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt, image, mask_image, strength, width, height, num_inference_steps,
                 guidance_scale, num_images_per_prompt, generator, callback_on_step_end,
                 sigmas=None, negative_prompt=None):
        self.calls.append(dict(
            image=image, mask_image=mask_image, strength=strength, size=(width, height),
            generator=generator, num_images_per_prompt=num_images_per_prompt,
            modes=(torch.is_grad_enabled(), torch.is_inference_mode_enabled()),
        ))
        view = SimpleNamespace(scheduler=SimpleNamespace(timesteps=range(num_inference_steps), order=1))
        _, self.num_timesteps = diffusers.ZImageInpaintPipeline.get_timesteps(
            view, num_inference_steps, strength, "cpu"
        )
        for step in range(self.num_timesteps):
            callback_on_step_end(self, step, None, {})
        self.finished += 1
        # The whole picture repainted, as the class's decode is: the kept
        # region is only kept by the composite.
        fill = fake_fill(generator.initial_seed())
        return SimpleNamespace(images=[Image.new("RGB", image.size, fill)])


@pytest.fixture
def inpaintable(store, monkeypatch, tmp_path):
    """A loaded backend on an enabled checkpoint whose pipeline has an
    inpaint class, a 256x256 source and a stored mask over its right half."""
    monkeypatch.setattr(
        refine, "sibling",
        lambda cls, kind="Img2Img": InpaintStub if cls is RefinableMain and kind == "Inpaint" else None,
    )
    monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, model_id=ANIMAGINE))
    backend = _loaded(RefinableMain())
    source = store.save_image(
        Image.effect_noise((256, 256), 64).convert("RGB"),
        GenerateRequest(prompt="a fox", width=256, height=256),
        seed=1, duration_ms=10, model_id=ANIMAGINE,
    )
    mask = Image.new("L", (256, 256), 0)
    mask.paste(255, (128, 0, 256, 256))
    path = tmp_path / f"{MASK}.png"
    mask.save(path)
    asked: list[str] = []
    monkeypatch.setattr(store, "mask_path", lambda h: asked.append(h) or path)
    return backend, source, asked


def test_an_inpaint_runs_the_inpaint_class_one_call_per_image(inpaintable):
    backend, record, asked = inpaintable
    pipe = backend._pipe
    assert backend.describe()["inpaint_supported"] is True
    assert backend.describe()["refine_supported"] is False  # no Img2Img stub here
    request = GenerateRequest(
        prompt="a fox, red scarf", width=256, height=256, steps=10,
        refine=record.id, strength=0.6, mask=MASK, num_images=2,
    )
    assert backend.applied(request).mask == MASK
    progress: list[tuple[int, int]] = []
    images = backend.generate(request, seed=7, on_progress=lambda s, t: progress.append((s, t)))

    runner = backend._inpaint_pipe
    assert pipe.calls == [] and isinstance(runner, InpaintStub) and backend._refine_pipe is None
    assert runner.scheduler is pipe.scheduler and runner.denoiser is pipe.denoiser and runner.flag
    # One call per image with its own generator; the source at its own size
    # and the stored mask, resolved by hash.
    assert [c["num_images_per_prompt"] for c in runner.calls] == [1, 1]
    assert [c["generator"].initial_seed() for c in runner.calls] == [7, 8]
    assert all(c["image"].size == (256, 256) and c["size"] == (256, 256) for c in runner.calls)
    assert all(c["mask_image"].mode == "L" and c["strength"] == 0.6 for c in runner.calls)
    assert asked == [MASK]
    assert [c["modes"] for c in runner.calls] == [(False, False), (False, False)]
    total = 6 * 2
    assert progress == [(n, total) for n in range(1, total + 1)]
    assert pipe.freed == 1
    # The composite: the source's own pixels on the left, the result's on the right.
    with Image.open(backend_source_path(record)) as handle:
        source = handle.convert("RGB")
    for image, seed in zip(images, (7, 8)):
        assert image.size == (256, 256)
        assert image.crop((0, 0, 128, 256)).tobytes() == source.crop((0, 0, 128, 256)).tobytes()
        assert image.getpixel((200, 100)) == fake_fill(seed)

    # It holds the pipeline's modules, so it goes when the pipeline goes.
    backend.unload()
    assert backend._inpaint_pipe is None and backend._inpaint_params == set()


def backend_source_path(record):
    from backend import storage

    return storage.settings.outputs_dir / record.filename


def test_an_inpaint_runs_at_its_sources_size(inpaintable):
    backend, record, _ = inpaintable
    request = GenerateRequest(prompt="x", width=384, height=384, steps=10,
                              refine=record.id, strength=0.6, mask=MASK)
    with pytest.raises(ValueError, match="an inpaint runs at its source's size"):
        backend.generate(request, seed=1)


def test_a_cancelled_inpaint_still_hands_the_card_back(inpaintable):
    backend, record, _ = inpaintable
    request = GenerateRequest(prompt="x", width=256, height=256, steps=10,
                              refine=record.id, strength=0.6, mask=MASK, num_images=2)
    steps: list[int] = []
    with pytest.raises(Cancelled):
        backend.generate(request, seed=1, on_progress=lambda s, t: steps.append(s),
                         should_cancel=lambda: len(steps) >= 2)
    assert backend._pipe.freed == 1


def test_a_deferred_family_is_refused_although_its_class_exists(inpaintable, monkeypatch):
    backend, record, _ = inpaintable
    monkeypatch.setattr(
        module, "settings",
        dataclasses.replace(module.settings, model_id="black-forest-labs/FLUX.1-Krea-dev"),
    )
    assert backend.describe()["inpaint_supported"] is False
    request = GenerateRequest(prompt="x", width=256, height=256, steps=10,
                              refine=record.id, strength=0.6, mask=MASK)
    ran = backend.applied(request)
    assert (ran.refine, ran.mask) == (None, None)
    with pytest.raises(ValueError, match="has no inpaint in this app"):
        backend.check_refine_supported(request)
    with pytest.raises(RuntimeError, match="has no inpaint pass in this app"):
        backend._inpainter(backend._pipe)


def _bare(class_name: str, model_id: str, monkeypatch) -> DiffusersBackend:
    """A backend 'loaded' with the real class and no weights: its class is
    what names the inpaint class, and the profile is the id's."""
    monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, model_id=model_id))
    backend = DiffusersBackend()
    cls = getattr(diffusers, class_name)
    backend._pipe = cls.__new__(cls)
    backend._call_params = set(inspect.signature(cls.__call__).parameters)
    return backend


@pytest.mark.parametrize("model_id, class_name, enabled", [
    (ANIMAGINE, "StableDiffusionXLPipeline", True),
    ("Tongyi-MAI/Z-Image-Turbo", "ZImagePipeline", True),
    ("SeeSee21/Z-Anime", "ZImagePipeline", True),
    ("black-forest-labs/FLUX.1-Krea-dev", "FluxPipeline", False),
    ("black-forest-labs/FLUX.2-klein-4B", "Flux2KleinPipeline", False),
    ("Qwen/Qwen-Image", "QwenImagePipeline", False),
    # An unprofiled id on an enabled class: the profile says no, so no.
    ("someone/unprofiled-sdxl", "StableDiffusionXLPipeline", False),
])
def test_the_loaded_answer_needs_the_class_and_the_profile(model_id, class_name, enabled, monkeypatch):
    backend = _bare(class_name, model_id, monkeypatch)
    info = backend.describe()
    assert info["inpaint_supported"] is enabled
    expected = refine.library_strength(getattr(diffusers, INPAINTS[class_name])) if enabled else None
    assert info["inpaint_strength"] == expected
    # Before a load: the profile's guess, and no strength yet.
    backend._pipe = None
    assert backend.describe()["inpaint_supported"] is None
    assert backend.describe()["inpaint_strength"] is None


# ---------------------------------------------------------------- the status


def test_the_status_publishes_the_capability(client, fake, monkeypatch):
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["inpaint_supported"] is True and defaults["inpaint_settled"] is True
    fake.inpaint_supported = False
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["inpaint_supported"] is False and defaults["inpaint_strength"] is None
    # Before a load the profile guesses; only its False is already final.
    for model_id, guess in ((ANIMAGINE, True), ("black-forest-labs/FLUX.1-Krea-dev", False)):
        described = dict(fake.describe(), inpaint_supported=None, model_id=model_id)
        monkeypatch.setattr(fake, "describe", lambda described=described: described)
        defaults = client.get("/api/status").json()["defaults"]
        assert defaults["inpaint_supported"] is guess
        assert defaults["inpaint_settled"] is (not guess)
    # A loaded class's own strength default reaches the page as given.
    described = dict(fake.describe(), inpaint_supported=True, inpaint_strength=0.9999, model_id=ANIMAGINE)
    monkeypatch.setattr(fake, "describe", lambda: described)
    assert client.get("/api/status").json()["defaults"]["inpaint_strength"] == 0.9999


def test_the_fake_repaints_only_under_the_mask(store, monkeypatch, tmp_path):
    from fakes import FakeBackend

    source = store.save_image(
        Image.effect_noise((256, 256), 64).convert("RGB"),
        GenerateRequest(prompt="a fox", width=256, height=256),
        seed=1, duration_ms=10, model_id="test/model",
    )
    mask = Image.new("L", (256, 256), 0)
    mask.paste(255, (0, 0, 256, 64))
    path = tmp_path / "mask.png"
    mask.save(path)
    monkeypatch.setattr(store, "mask_path", lambda h: path)
    fake = FakeBackend()
    request = GenerateRequest(prompt="x", width=256, height=256, steps=4,
                              refine=source.id, strength=0.6, mask=MASK, num_images=2)
    images = fake.generate(fake.applied(request), seed=3)
    with Image.open(store.settings.outputs_dir / source.filename) as handle:
        original = handle.convert("RGB")
    for image, seed in zip(images, (3, 4)):
        assert image.crop((0, 64, 256, 256)).tobytes() == original.crop((0, 64, 256, 256)).tobytes()
        assert image.getpixel((10, 10)) == fake_fill(seed)
    # Off, the mask goes with the source.
    fake.inpaint_supported = False
    assert fake.applied(request).mask is None and fake.applied(request).refine is None


def test_a_hash_the_server_holds_no_mask_under_is_named(store):
    with pytest.raises(ValueError, match="no stored mask abababababababab"):
        refine.mask_image(MASK, 256, 256)


def test_a_stored_mask_is_read_as_one_channel_at_its_size(store):
    mask = Image.new("L", (256, 128), 0)
    mask.paste(200, (0, 0, 64, 64))
    digest, _ = store.store_mask(mask)
    read = refine.mask_image(digest, 256, 128)
    assert read.mode == "L" and read.tobytes() == mask.tobytes()
    with pytest.raises(ValueError, match="not the 256x256 it covers"):
        refine.mask_image(digest, 256, 256)
