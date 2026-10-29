"""The diffusers instruction-edit branch, against a Kontext-shaped stub.

The real backend's gate stays the literal False; a local subclass opens it so
the call it would make can be inspected: the exact kwargs, the reproved and
prepared source, one generator, progress, preview, cancel and cleanup. No
weights, downloads or GPU - and the real backend still refuses before a call.
"""
from __future__ import annotations

import inspect

import diffusers
import pytest
import torch
from PIL import Image, PngImagePlugin

from backend import instruction_edit_admission, latent_preview, storage
from backend.backends import Cancelled
from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import _EDIT_PARAMS, DiffusersBackend
from backend.instruction_edit import (
    INSTRUCTION_EDIT_LOADER_UNQUALIFIED, INSTRUCTION_EDIT_NEEDS_MODEL, INSTRUCTION_EDIT_UNAVAILABLE, KONTEXT_V1,
    QUALIFIED_ENVELOPE, LoaderChoice, StoredInstructionEdit,
)
from backend.schemas import GenerateRequest
from backend.seeds import wrap_seed

# `tests/` has no __init__.py; pytest puts this directory on sys.path.
from instruction_edit_helpers import stored_edit
from test_instruction_edit_admission import select_editor
from test_pipeline_ownership import synthetic_loader  # noqa: F401 - fixture
from test_refine import RefinableMain

INSTALLED = inspect.signature(diffusers.FluxKontextPipeline.__call__)
STEPS = 4


def _signature_without(*names: str) -> inspect.Signature:
    return INSTALLED.replace(parameters=[p for p in INSTALLED.parameters.values() if p.name not in names])


class FluxKontextPipeline(RefinableMain):
    """The installed Kontext call's parameter names over a recording body."""

    def __init__(self, *, result=None, after=None):
        super().__init__()
        self.kwargs: list[dict] = []
        self.modes: list[tuple[bool, bool]] = []
        self.seen: list[Image.Image] = []
        self.pixels: list[tuple[str, tuple[int, int], bytes]] = []
        self.result = result
        self.after = after

    def __call__(self, **kwargs):
        self.kwargs.append(dict(kwargs))
        self.modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        image = kwargs["image"]
        self.seen.append(image)
        self.pixels.append((image.mode, image.size, image.tobytes()))
        for step in range(kwargs["num_inference_steps"]):
            kwargs["callback_on_step_end"](self, step, None, {"latents": torch.zeros(1, 4, 8)})
        if self.after is not None:
            self.after()
        images = (self.result(kwargs) if self.result is not None
                  else [Image.new("RGB", (kwargs["width"], kwargs["height"]), (200, 10, 10))])
        return type("Result", (), {"images": images})()


FluxKontextPipeline.__call__.__signature__ = INSTALLED  # type: ignore[attr-defined]
KontextStub = FluxKontextPipeline


class NoMaxArea(FluxKontextPipeline):
    pass


NoMaxArea.__name__ = "FluxKontextPipeline"
NoMaxArea.__call__ = lambda self, **kwargs: FluxKontextPipeline.__call__(self, **kwargs)  # type: ignore[method-assign]
NoMaxArea.__call__.__signature__ = _signature_without("max_area")  # type: ignore[attr-defined]


class NotKontext(FluxKontextPipeline):
    """Kontext's call under another class name."""


class Qualified(DiffusersBackend):
    """A stand-in whose gate is open; the real class's answer is untouched."""

    def instruction_edit_qualified(self) -> bool:
        return True

    def instruction_edit_host(self) -> tuple[str, str | None]:
        # The measured host, as the fake reports it, so the loader envelope
        # every load and call now asks answers the same on any test machine.
        return QUALIFIED_ENVELOPE.platform, QUALIFIED_ENVELOPE.diffusers_version


def _loaded(pipe, cls=Qualified) -> DiffusersBackend:
    backend = cls()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    # Stay off the card: the peak measurement is not what is tested here.
    backend._peak_device = lambda: None  # type: ignore[method-assign]
    return backend


def _source(store, size=(768, 512)):
    width, height = size
    image = Image.new("RGB", size)
    image.putdata([((x * 7) % 256, (y * 5) % 256, (x + y) % 256) for y in range(height) for x in range(width)])
    return store.save_image(
        image, GenerateRequest(prompt="original", width=width, height=height, mu=None),
        seed=3, duration_ms=1, model_id="source/checkpoint", quantization="fp8",
    )


def source_path(source):
    return storage.settings.outputs_dir / source.filename


def _request(bound: StoredInstructionEdit, **changes) -> GenerateRequest:
    return GenerateRequest(**{
        "prompt": "make the sky green", "width": 576, "height": 384, "steps": STEPS,
        "guidance_scale": 2.5, "mu": None, "edit": bound, **changes,
    })


@pytest.fixture
def editor(store, monkeypatch):
    """A qualified backend over the Kontext stub, the editor selected and a 3:2 source."""
    select_editor(monkeypatch)
    source = _source(store)
    prepared: list[Image.Image] = []
    real = instruction_edit_admission.load_bound_source

    def recording(value):
        answer = real(value)
        prepared.append(answer[2])
        return answer

    monkeypatch.setattr(instruction_edit_admission, "load_bound_source", recording)
    return source, stored_edit(source), prepared


def _closed(image: Image.Image) -> bool:
    try:
        image.getpixel((0, 0))
    except ValueError:
        return True
    return False


def test_the_stub_exposes_the_installed_call():
    assert set(inspect.signature(KontextStub().__call__).parameters) == set(INSTALLED.parameters) - {"self"}
    assert _EDIT_PARAMS <= set(INSTALLED.parameters)


def test_an_edit_hands_kontext_exactly_its_recipe(editor):
    source, bound, prepared = editor
    pipe = KontextStub()
    backend = _loaded(pipe)
    progress: list[tuple[int, int]] = []
    seed = 2**63 + 4
    images = backend.generate(_request(bound), seed=seed, on_progress=lambda s, t: progress.append((s, t)))

    assert len(pipe.kwargs) == 1
    kwargs = pipe.kwargs[0]
    assert set(kwargs) == _EDIT_PARAMS
    assert kwargs["prompt"] == "make the sky green"
    assert (kwargs["num_inference_steps"], kwargs["guidance_scale"]) == (STEPS, 2.5)
    assert (kwargs["width"], kwargs["height"]) == (576, 384)
    assert kwargs["max_area"] == 576 * 384 and kwargs["_auto_resize"] is False
    assert kwargs["num_images_per_prompt"] == 1 and kwargs["max_sequence_length"] == 512
    # The source: RGB at the working size, an owned copy rather than the file.
    mode, size, pixels = pipe.pixels[0]
    assert (mode, size) == ("RGB", (576, 384))
    image = pipe.seen[0]
    assert image is prepared[0]
    assert not isinstance(image, PngImagePlugin.PngImageFile) and getattr(image, "fp", None) is None
    with Image.open(source_path(source)) as stored:
        expected = stored.convert("RGB").resize((576, 384), resample=Image.Resampling.LANCZOS)
    assert pixels == expected.tobytes()
    # One generator, seeded where storage records the seed.
    generators = kwargs["generator"]
    assert len(generators) == 1 and generators[0].initial_seed() == wrap_seed(seed)
    assert pipe.modes == [(False, False)]
    assert progress == [(n, STEPS) for n in range(1, STEPS + 1)]
    assert [im.size for im in images] == [(576, 384)]
    # Hooks handed back and the prepared source closed after the call.
    assert pipe.freed == 1 and _closed(image)
    # What the row records is the bound edit with no negative.
    ran = backend.applied(_request(bound))
    assert ran.edit == bound and ran.negative_prompt is None


def test_the_preview_gets_the_latents_and_a_failing_one_costs_no_cell(editor, monkeypatch):
    _, bound, _ = editor
    monkeypatch.setattr(latent_preview, "INTERVAL_SECONDS", 0)
    backend = _loaded(KontextStub())
    seen = []

    def preview(runner, latents, request, step):
        seen.append((runner, tuple(latents.shape), step))
        return b"jpeg"

    monkeypatch.setattr(backend, "_preview", preview)
    shown = []
    backend.generate(_request(bound), seed=1, on_preview=lambda jpeg, done, total: shown.append((done, total)))
    assert [entry[0] for entry in seen] == [backend._pipe] * (STEPS - 1)
    assert all(entry[1] == (1, 4, 8) for entry in seen)
    assert shown == [(n, STEPS) for n in range(1, STEPS)]

    def broken(*args):
        raise RuntimeError("wrong-shaped latents")

    monkeypatch.setattr(backend, "_preview", broken)
    shown.clear()
    images = backend.generate(_request(bound), seed=1, on_preview=lambda *args: shown.append(args))
    assert shown == [] and len(images) == 1


def test_cancel_mid_edit_hands_the_card_back_and_closes_the_source(editor):
    _, bound, prepared = editor
    pipe = KontextStub()
    backend = _loaded(pipe)
    steps: list[int] = []
    with pytest.raises(Cancelled):
        backend.generate(
            _request(bound), seed=1, on_progress=lambda s, t: steps.append(s), should_cancel=lambda: len(steps) >= 2,
        )
    assert steps == [1, 2] and pipe.freed == 1
    assert len(prepared) == 1 and _closed(prepared[0])


def test_a_failure_after_the_loop_propagates_with_cleanup(editor):
    _, bound, prepared = editor

    def decode():
        raise RuntimeError("synthetic VAE decode failure")

    pipe = KontextStub(after=decode)
    backend = _loaded(pipe)
    with pytest.raises(RuntimeError, match="synthetic VAE decode failure"):
        backend.generate(_request(bound), seed=1)
    assert pipe.freed == 1 and _closed(prepared[0])


@pytest.mark.parametrize("pipe_cls, error, match", [
    (NoMaxArea, RuntimeError, "max_area"),
    (NotKontext, ValueError, "FluxKontextPipeline"),
])
def test_a_pipeline_that_cannot_take_the_recipe_refuses_before_any_call(editor, pipe_cls, error, match):
    _, bound, prepared = editor
    pipe = pipe_cls()
    backend = _loaded(pipe)
    with pytest.raises(error, match=match):
        backend.generate(_request(bound), seed=1)
    assert pipe.kwargs == [] and prepared == [] and pipe.freed == 0


@pytest.mark.parametrize("result", [
    lambda kwargs: [],
    lambda kwargs: [Image.new("RGB", (512, 512))],
    lambda kwargs: [Image.new("RGB", (576, 384))] * 2,
])
def test_a_result_that_is_not_one_image_at_the_output_size_is_refused(editor, result):
    _, bound, prepared = editor
    pipe = KontextStub(result=result)
    backend = _loaded(pipe)
    with pytest.raises(RuntimeError, match="one 576x384 image"):
        backend.generate(_request(bound), seed=1)
    assert len(pipe.kwargs) == 1 and pipe.freed == 1 and _closed(prepared[0])


def test_pixels_changed_after_binding_refuse_before_the_call(editor):
    source, bound, prepared = editor
    Image.new("RGB", (768, 512), (1, 2, 3)).save(source_path(source))
    pipe = KontextStub()
    backend = _loaded(pipe)
    with pytest.raises(ValueError, match="recorded pixels"):
        backend.generate(_request(bound), seed=1)
    assert pipe.kwargs == [] and prepared == [] and pipe.freed == 0


def test_a_binned_source_refuses_before_the_call(editor, store):
    source, bound, _ = editor
    store.bin_images([source.id])
    pipe = KontextStub()
    with pytest.raises(ValueError, match="recycle bin"):
        _loaded(pipe).generate(_request(bound), seed=1)
    assert pipe.kwargs == []


def test_a_size_that_is_not_the_sources_geometry_is_refused(editor):
    _, bound, prepared = editor
    pipe = KontextStub()
    backend = _loaded(pipe)
    # Validation refuses it first on the public path ...
    with pytest.raises(ValueError, match="frozen source geometry"):
        backend.generate(_request(bound, width=512, height=512), seed=1)
    # ... and the branch itself refuses a copy that skipped validation.
    request = backend.applied(_request(bound)).model_copy(update={"width": 512, "height": 512})
    generator = [torch.Generator(device="cpu").manual_seed(1)]
    with pytest.raises(ValueError, match="requires 576x384"):
        backend._edit_images(pipe, request, generator, lambda *args: args[-1], None, None)
    assert pipe.kwargs == [] and len(prepared) == 1 and _closed(prepared[0])


def test_an_unbound_edit_never_reaches_the_branch(editor):
    source, _, _ = editor
    pipe = KontextStub()
    backend = _loaded(pipe)
    raw = _request(stored_edit(source)).model_copy(update={"edit": {"source": source.id, "recipe": "kontext-v1"}})
    with pytest.raises(ValueError):
        backend.generate(raw, seed=1)
    generator = [torch.Generator(device="cpu").manual_seed(1)]
    unbound = raw.model_copy(update={"edit": raw.validate_edit()})
    with pytest.raises(ValueError, match="Bind and verify"):
        backend._edit_images(pipe, unbound, generator, lambda *args: args[-1], None, None)
    assert pipe.kwargs == []


def test_the_real_backend_refuses_before_any_call(editor):
    _, bound, prepared = editor
    pipe = KontextStub()
    backend = _loaded(pipe, cls=DiffusersBackend)
    assert backend.instruction_edit_qualified() is False
    with pytest.raises(ValueError, match="not available") as caught:
        backend.generate(_request(bound), seed=1)
    assert str(caught.value) == INSTRUCTION_EDIT_UNAVAILABLE
    with pytest.raises(ValueError, match="not available"):
        backend.applied(_request(bound))
    with pytest.raises(ValueError, match="not available"):
        backend._require_execution(_request(bound))
    assert pipe.kwargs == [] and prepared == [] and pipe.freed == 0


def test_describe_follows_the_gate_and_the_selection(store, monkeypatch):
    from dataclasses import replace

    # Off the card: describe() reads device memory where the settings say CUDA.
    real = module.current_settings
    monkeypatch.setattr(module, "current_settings", lambda startup: replace(real(startup), device="cpu", offload="none"))
    held = DiffusersBackend().describe()
    assert held["instruction_edit_supported"] is False
    assert held["instruction_edit_reason"] == INSTRUCTION_EDIT_UNAVAILABLE
    needs = Qualified().describe()
    assert needs["instruction_edit_supported"] is False
    assert needs["instruction_edit_reason"] == INSTRUCTION_EDIT_NEEDS_MODEL
    select_editor(monkeypatch)
    # Off the card, the selection's own loader is outside the envelope, and
    # describe says the first unmeasured choice (offload none, before cpu).
    unmeasured = Qualified().describe()
    assert unmeasured["instruction_edit_supported"] is False
    assert unmeasured["instruction_edit_reason"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="no offload")

    class Measured(Qualified):
        """The loader facts the card measured, whatever this machine is."""

        def instruction_edit_loader(self, selected):
            return LoaderChoice(
                quantization="fp8", offload="model", device="cuda", platform="win32",
                loader_execution=None, diffusers_version="0.40.0",
            )

    opened = Measured().describe()
    assert opened["instruction_edit_supported"] is True and opened["instruction_edit_reason"] is None
    still = DiffusersBackend().describe()
    assert still["instruction_edit_supported"] is False
    assert still["instruction_edit_reason"] == INSTRUCTION_EDIT_UNAVAILABLE


def test_a_qualified_backend_still_refuses_to_load_kontext(synthetic_loader, monkeypatch):  # noqa: F811
    from dataclasses import replace

    _, _, events = synthetic_loader
    select_editor(monkeypatch)
    selected = replace(module.settings, model_id=KONTEXT_V1.model_id, device="cpu", offload="none", quantization="none")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    backend = Qualified()
    # Both entries now ask the instance's gate and the selection's loader
    # facts: outside the envelope (no offload here), its words come first.
    unmeasured = INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="BF16 (no quantization)")
    for call in (backend.load, lambda: backend._load_prepared(None)):
        with pytest.raises(ValueError) as caught:
            call()
        assert str(caught.value) == unmeasured
    # The closed real backend still says the gate, whatever the loader.
    for call in (DiffusersBackend().load, lambda: DiffusersBackend()._load_prepared(None)):
        with pytest.raises(ValueError) as caught:
            call()
        assert str(caught.value) == INSTRUCTION_EDIT_UNAVAILABLE
    assert not backend.is_ready() and events == []


def test_a_qualified_backend_still_refuses_an_editor_class_index(synthetic_loader, monkeypatch, tmp_path):  # noqa: F811
    import json
    from dataclasses import replace

    from backend.backends import loading

    _, _, events = synthetic_loader
    selected = replace(module.settings, model_id="owned/custom-kontext-alias", device="cpu", offload="none",
                       quantization="none")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    path = tmp_path / "model_index.json"
    path.write_text(json.dumps({"_class_name": KONTEXT_V1.pipeline_class}), encoding="utf-8")
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: str(path))
    monkeypatch.setattr(loading, "unused_components", lambda model_id: (None, {}))
    backend = Qualified()
    with pytest.raises(ValueError, match="Instruction editing is not available"):
        backend.load()
    assert not backend.is_ready() and not {"overrides", "pretrained", "place"} & set(events)


class FluxPipeline(RefinableMain):
    """FLUX.1's text-to-image names, recording what it was handed."""

    def __init__(self):
        super().__init__()
        self.kwargs: list[dict] = []

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale, num_images_per_prompt,
                 generator, callback_on_step_end, sigmas=None, negative_prompt=None, true_cfg_scale=1.0):
        self.kwargs.append(dict(
            prompt=prompt, width=width, height=height, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt,
            seeds=[g.initial_seed() for g in generator], callback=callable(callback_on_step_end),
            sigmas=sigmas, negative_prompt=negative_prompt, true_cfg_scale=true_cfg_scale,
        ))
        return type("Result", (), {"images": [Image.new("RGB", (width, height))] * num_images_per_prompt})()


def test_a_text_to_image_call_is_unchanged_by_the_edit_branch(store):
    request = GenerateRequest(
        prompt="a lighthouse", negative_prompt="blurry", true_cfg=3.0, guidance_scale=3.5,
        width=256, height=256, steps=4, mu=None, num_images=2,
    )
    calls = []
    for cls in (DiffusersBackend, Qualified):
        pipe = FluxPipeline()
        _loaded(pipe, cls=cls).generate(request, seed=11)
        calls.append(pipe.kwargs)
    assert calls[0] == calls[1] == [dict(
        prompt="a lighthouse", width=256, height=256, num_inference_steps=4, guidance_scale=3.5,
        num_images_per_prompt=2, seeds=[11, 12], callback=True, sigmas=None,
        negative_prompt="blurry", true_cfg_scale=3.0,
    )]


def test_load_bound_source_needs_complete_bound_proof(store):
    from backend.instruction_edit import UNRESOLVED_SOURCE, InstructionEditSpec

    source = _source(store)
    bound = stored_edit(source)
    stored, resolved, image = instruction_edit_admission.load_bound_source(bound)
    try:
        assert stored == bound and resolved.working_size == (576, 384)
        assert (image.mode, image.size) == ("RGB", (576, 384))
    finally:
        image.close()
    with pytest.raises(ValueError, match="Bind and verify"):
        instruction_edit_admission.load_bound_source(InstructionEditSpec(source=source.id, recipe="kontext-v1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Bind and verify"):
        instruction_edit_admission.load_bound_source(bound.model_dump())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=UNRESOLVED_SOURCE):
        instruction_edit_admission.load_bound_source(bound.model_copy(update={"source": None}))
    forged = bound.model_copy(update={"edit_source": bound.edit_source.model_copy(update={"hash": "ab" * 32})})
    with pytest.raises(ValueError, match="recorded pixels"):
        instruction_edit_admission.load_bound_source(forged)


def test_load_bound_source_refuses_an_aspect_outside_the_recipe(store):
    source = _source(store, size=(512, 256))
    # A 2:1 source cannot even be bound; its proof alone carries no geometry.
    with pytest.raises(ValueError, match="exact source aspect"):
        instruction_edit_admission.bind_source({"source": source.id, "recipe": "kontext-v1"})


def test_edit_only_checkpoint_refuses_plain_generate_at_the_backend(store, monkeypatch):
    # Inside the envelope with the gate open, the backend itself still runs
    # nothing but an edit on Kontext: the command line, Mass refine and Then
    # refine reach this call without the submit check.
    from backend.instruction_edit import INSTRUCTION_EDIT_TEXT_TO_IMAGE

    select_editor(monkeypatch)
    source = _source(store, size=(320, 320))
    pipe = KontextStub()
    backend = _loaded(pipe)
    plain = GenerateRequest(prompt="a lighthouse", width=512, height=512, mu=None)
    refine = plain.model_copy(update={"refine": source.id})
    for request in (plain, refine):
        for call, args in ((backend.applied, ()), (backend.generate, (1,))):
            with pytest.raises(ValueError) as caught:
                call(request, *args)
            assert str(caught.value) == INSTRUCTION_EDIT_TEXT_TO_IMAGE
    # An edit carrying anything else is refused by its own validation first.
    bound = stored_edit(source)
    with pytest.raises(ValueError, match="cannot include"):
        backend.generate(_request(bound, width=512, height=512).model_copy(update={"lora": "style"}), seed=1)
    assert pipe.kwargs == []
    # And the closed real backend says its gate, never the pointer.
    closed = _loaded(KontextStub(), cls=DiffusersBackend)
    with pytest.raises(ValueError) as caught:
        closed.generate(plain, seed=1)
    assert str(caught.value) == INSTRUCTION_EDIT_UNAVAILABLE
