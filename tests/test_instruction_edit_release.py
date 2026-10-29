"""INP#11 stage 4's release: the real backend's gate is the literal True.

The real `DiffusersBackend` class itself, never a subclass that opens its
gate: only the host (or, for the synthetic CPU load, the loader facts) is
reported as the card's, so the answers are the same on any test machine. A
Kontext-shaped `from_pretrained` stub stands in for the weights. No weights,
downloads, model loads or GPU.
"""
from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from backend import grid, instruction_edit, main
from backend.backends import diffusers_backend as module
from backend.backends import loading
from backend.backends.diffusers_backend import DiffusersBackend
from backend.instruction_edit import (
    INSTRUCTION_EDIT_ASPECT_UNQUALIFIED, INSTRUCTION_EDIT_LOADER_UNQUALIFIED, INSTRUCTION_EDIT_TEXT_TO_IMAGE,
    KONTEXT_V1, QUALIFIED_ENVELOPE,
)
from backend.schemas import GenerateRequest

# `tests/` has no __init__.py; pytest puts this directory on sys.path.
from instruction_edit_helpers import stored_edit
from test_instruction_edit_call import KontextStub, _source
from test_instruction_edit_identity import request as identity_request
from test_instruction_edit_loader import MEASURED, kontext_hub  # noqa: F401 - fixture
from test_pipeline_ownership import synthetic_loader  # noqa: F401 - fixture

HOST = (QUALIFIED_ENVELOPE.platform, QUALIFIED_ENVELOPE.diffusers_version)


def _real(monkeypatch, *, host=HOST) -> DiffusersBackend:
    """The real class on the measured host; its gate is its own."""
    backend = DiffusersBackend()
    assert type(backend) is DiffusersBackend and backend.instruction_edit_qualified() is True
    monkeypatch.setattr(backend, "instruction_edit_host", lambda: host)
    return backend


def _on_card(monkeypatch, **changes):
    """The editor selected with the envelope's loader (fp8, model offload, cuda)."""
    selected = replace(module.settings, **{
        "model_id": KONTEXT_V1.model_id, "quantization": "fp8", "offload": "model", "device": "cuda", **changes,
    })
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    return selected


def _loaded(monkeypatch, pipe) -> DiffusersBackend:
    backend = _real(monkeypatch)
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    # Stay off the card: the peak measurement is not what is tested here.
    backend._peak_device = lambda: None  # type: ignore[method-assign]
    return backend


def test_the_real_backend_loads_the_pinned_editor_on_fp8_with_model_offload(kontext_hub, monkeypatch):  # noqa: F811
    fixture, _, events, hub = kontext_hub
    backend = DiffusersBackend()
    # The synthetic load runs on the CPU, so only the loader facts are the
    # card's (fp8, model offload, cuda, Windows, diffusers 0.40.0); the gate
    # is the real class's.
    monkeypatch.setattr(backend, "instruction_edit_loader", lambda selected: MEASURED)
    monkeypatch.setattr(backend, "_release", fixture._release)
    assert (MEASURED.quantization, MEASURED.offload) == ("fp8", "model")
    backend.load()
    revision = KONTEXT_V1.revision
    assert [call[:3] if call[0] == "configs" else call for call in hub.calls] == [
        ("configs", KONTEXT_V1.model_id, revision),
        ("weights", KONTEXT_V1.model_id, {"revision": revision}),
    ]
    assert events.count("pretrained") == 1 and events.index("download") < events.index("pretrained")
    assert backend.is_ready() and type(backend._pipe).__name__ == KONTEXT_V1.pipeline_class
    assert module._EDIT_PARAMS <= backend._call_params


@pytest.mark.parametrize(("change", "words"), [
    ({"quantization": "int8"}, "int8 quantization"),
    ({"quantization": "none"}, "BF16 (no quantization)"),
    ({"offload": "sequential"}, "sequential offload"),
])
def test_an_unmeasured_loader_is_refused_before_resolve_model(monkeypatch, change, words):
    _on_card(monkeypatch, **change)
    resolved = []
    monkeypatch.setattr(loading, "resolve_model", lambda *args, **kwargs: resolved.append(args) or "")
    backend = _real(monkeypatch)
    for call in (backend.load, lambda: backend._load_prepared(None)):
        with pytest.raises(ValueError) as caught:
            call()
        assert str(caught.value) == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice=words)
    assert resolved == [] and not backend.is_ready()


def test_a_three_two_preview_is_held_and_a_square_one_is_available(client, store, monkeypatch):
    from backend import model_selection

    # The server's own selection: the editor on the default fp8/model/cuda.
    monkeypatch.setattr(model_selection, "_selection", (KONTEXT_V1.model_id, 1))
    monkeypatch.setattr(main, "backend", _real(monkeypatch))
    square, wide = _source(store, size=(320, 320)), _source(store, size=(768, 512))
    answer = client.post("/api/instruction-edit/preview", json={"source": square.id, "recipe": "kontext-v1"}).json()
    assert (answer["available"], answer["reason"], answer["width"], answer["height"]) == (True, None, 512, 512)
    three_two = INSTRUCTION_EDIT_ASPECT_UNQUALIFIED.format(a=3, b=2, w=576, h=384)
    answer = client.post("/api/instruction-edit/preview", json={"source": wide.id, "recipe": "kontext-v1"}).json()
    assert (answer["available"], answer["reason"], answer["width"], answer["height"]) == (False, three_two, 576, 384)
    response = client.post("/api/instruction-edit", json={
        "source": wide.id, "recipe": "kontext-v1", "prompt": "make it night", "width": 576, "height": 384, "seed": 7,
    })
    assert response.status_code == 422 and response.json()["detail"] == three_two
    assert main.jobs.active() == []


def test_a_square_edit_keys_as_the_pinned_fixture(monkeypatch):
    # The run id and CellKey pinned before stage 4
    # (test_instruction_edit_envelope), through the real backend's applied().
    _on_card(monkeypatch)
    applied = _real(monkeypatch).applied(identity_request())
    cell = grid.Cell(applied, 7)
    assert grid.cell_key(cell) == grid.cell_key(grid.Cell(identity_request(), 7))
    assert grid.run_id([cell], KONTEXT_V1.model_id, "fp8") == "a83cde9e6cb2"
    assert applied.negative_prompt is None and applied.edit == identity_request().edit


def test_the_real_backend_runs_a_square_edit_at_its_frozen_size(store, monkeypatch):
    _on_card(monkeypatch)
    source = _source(store, size=(320, 320))
    pipe = KontextStub()
    backend = _loaded(monkeypatch, pipe)
    edit = GenerateRequest(
        prompt="make the sky green", width=512, height=512, steps=2, guidance_scale=2.5, mu=None,
        edit=stored_edit(source),
    )
    images = backend.generate(edit, seed=5)
    assert [image.size for image in images] == [(512, 512)]
    assert len(pipe.kwargs) == 1 and (pipe.kwargs[0]["width"], pipe.kwargs[0]["height"]) == (512, 512)


def test_plain_generate_on_the_loaded_editor_raises_before_the_pipe(store, monkeypatch):
    _on_card(monkeypatch)
    source = _source(store, size=(320, 320))
    pipe = KontextStub()
    backend = _loaded(monkeypatch, pipe)
    plain = GenerateRequest(prompt="a lighthouse", width=512, height=512, mu=None)
    for request in (plain, plain.model_copy(update={"refine": source.id})):
        for call, args in ((backend.applied, ()), (backend.generate, (1,))):
            with pytest.raises(ValueError) as caught:
                call(request, *args)
            assert str(caught.value) == INSTRUCTION_EDIT_TEXT_TO_IMAGE
    assert pipe.kwargs == []


def test_describe_on_the_loaded_editor_offers_edits_and_nothing_else(monkeypatch):
    _on_card(monkeypatch)
    # describe() reads device memory where CUDA is there: stay off the card.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = _loaded(monkeypatch, KontextStub())
    info = backend.describe()
    assert (info["instruction_edit_supported"], info["instruction_edit_reason"]) == (True, None)
    assert info["text_to_image_reason"] == INSTRUCTION_EDIT_TEXT_TO_IMAGE
    assert (info["refine_supported"], info["inpaint_supported"], info["reference_supported"]) == (False, False, False)
    assert info["inpaint_strength"] is None
    # Off the measured host the same backend holds edits in the envelope's words.
    monkeypatch.setattr(backend, "instruction_edit_host", lambda: ("linux", HOST[1]))
    held = backend.describe()
    assert held["instruction_edit_supported"] is False
    assert held["instruction_edit_reason"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="Linux (WSL)")


def test_qwen_stays_refused_on_the_real_backend(monkeypatch):
    _on_card(monkeypatch, model_id="Qwen/Qwen-Image-Edit")
    backend = _real(monkeypatch)
    with pytest.raises(ValueError) as caught:
        backend.load()
    assert str(caught.value) == instruction_edit.INSTRUCTION_EDITOR_UNSUPPORTED

