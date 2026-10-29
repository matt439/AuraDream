"""Custom editor aliases refuse on their index before any synthetic weight load;
the exact Kontext id loads only through the pinned configs-then-weights path."""
from __future__ import annotations

from dataclasses import replace
import functools
import gc
import inspect
import json
import re
import sys
from types import MappingProxyType, SimpleNamespace
import weakref

import diffusers
import pytest

from backend import instruction_edit
from backend import instruction_edit_preflight as preflight
from backend.backends import diffusers_backend as module, loading
from backend.launch import CHECKPOINTS, FORMATS
from tests.test_instruction_edit_preflight import _configs as _preflight_configs
from tests.test_pipeline_ownership import synthetic_loader


@pytest.mark.parametrize("class_name", sorted(loading.INSTRUCTION_EDITOR_CLASSES))
def test_custom_editor_class_refuses_before_any_pipeline_weight_preparation(synthetic_loader, monkeypatch, tmp_path, class_name):
    backend, _, events = synthetic_loader
    selected = replace(module.settings, model_id="owned/custom-editor-alias", device="cpu", offload="none", quantization="none")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    path = tmp_path / "model_index.json"
    path.write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
    reads = []

    def supplied(model_id, filename):
        reads.append((model_id, filename))
        assert model_id == selected.model_id and filename == "model_index.json"
        return str(path)

    monkeypatch.setattr(loading, "hub_file", supplied)
    # The existing optional optimization is deliberately inconclusive; a
    # separate strict decision must refuse before reaching this fallback.
    monkeypatch.setattr(loading, "unused_components", lambda model_id: (None, {}))
    with pytest.raises(ValueError, match="Instruction editing is not available"):
        backend.load()
    assert reads == [(selected.model_id, "model_index.json")]
    assert not backend.is_ready() and backend._call_params == set()
    assert not {"overrides", "pretrained", "place"} & set(events)


@pytest.mark.parametrize("broken", ["unreadable", "json", "list", "missing", "invalid-class"])
def test_uninspectable_custom_alias_cannot_fall_through_to_from_pretrained(synthetic_loader, monkeypatch, tmp_path, broken):
    backend, _, events = synthetic_loader
    selected = replace(module.settings, model_id="owned/custom-alias", device="cpu", offload="none", quantization="none")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    path = tmp_path / "model_index.json"
    value = {"list": [], "missing": {}, "invalid-class": {"_class_name": ["custom", "Pipeline"]}}.get(broken)
    path.write_text("{" if broken == "json" else json.dumps(value), encoding="utf-8")

    def supplied(model_id, filename):
        assert filename == "model_index.json"
        if broken == "unreadable":
            raise TimeoutError("synthetic index timeout")
        return str(path)

    monkeypatch.setattr(loading, "hub_file", supplied)
    with pytest.raises(ValueError, match="custom checkpoint"):
        backend.load()
    assert not backend.is_ready() and not {"overrides", "pretrained", "place"} & set(events)


def test_custom_noneditor_class_is_a_checked_identity_not_an_execution_grant(monkeypatch, tmp_path):
    path = tmp_path / "model_index.json"
    path.write_text('{"_class_name":"FluxPipeline"}', encoding="utf-8")
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: str(path))
    assert loading.require_custom_pipeline_class("owned/ordinary-custom-id") == "FluxPipeline"


def test_catalogue_and_enumerated_nunchaku_ids_keep_their_existing_index_behavior(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("known catalogue identity reached custom-id index inspection")

    monkeypatch.setattr(loading, "hub_file", forbidden)
    ids = {checkpoint.repo for checkpoint in CHECKPOINTS}
    ids.update(repo for checkpoint in CHECKPOINTS for fmt, _ in FORMATS
               if (repo := checkpoint.lite_repo(fmt)) is not None)
    assert all(loading.require_custom_pipeline_class(model_id) is None for model_id in ids)


@pytest.mark.parametrize("model_id", [instruction_edit.KONTEXT_V1.model_id.upper(), "Qwen/Qwen-Image-Edit", "QWEN/QWEN-IMAGE-EDIT"])
def test_named_edit_only_checkpoints_refuse_without_any_index_or_framework_access(model_id):
    with pytest.raises(ValueError, match="not available"):
        instruction_edit.require_model_available(model_id)


def test_every_require_model_available_caller_passes_qualified_and_selection():
    # Every load entry asks the instance's gate and the selection's loader
    # facts (slice 3); a call that left either out would treat the gate as
    # closed, or an open gate as outside any envelope, without saying why.
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "backend").rglob("*.py")), *sorted((root / "tools").rglob("*.py")),
               root / "tests" / "fakes.py"]
    calls = []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name != "require_model_available":
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            calls.append((path.relative_to(root).as_posix(), node.lineno, {"qualified", "loader"} <= keywords))
    callers = {path for path, _, _ in calls}
    # The load entries the plan names, and the stand-in's own copies.
    assert {
        "backend/backends/diffusers_backend.py", "backend/main.py", "backend/model_selection.py", "tests/fakes.py",
    } <= callers, calls
    assert [call for call in calls if not call[2]] == []


# ---------------------------------------------------------------- the pinned Kontext loader (stage 4, slice 4)
#
# The configs pass, then the weights pass, both at KONTEXT_V1.revision and
# both checked against the config pin, with the explicit class and the loaded
# pipeline's own checks after. Every Hub call is a stub: the synthetic configs
# are written to a temporary tree and the pin is monkeypatched to their
# hashes, as test_instruction_edit_gate does. Nothing downloads or loads.

RESOLVE = loading.resolve_model  # before the synthetic loader replaces it
INSTALLED_EDIT_CALL = inspect.signature(diffusers.FluxKontextPipeline.__call__)
MEASURED = instruction_edit.LoaderChoice(
    quantization=instruction_edit.QUALIFIED_ENVELOPE.quantization,
    offload=instruction_edit.QUALIFIED_ENVELOPE.offload, device="cuda",
    platform=instruction_edit.QUALIFIED_ENVELOPE.platform, loader_execution=None,
    diffusers_version=instruction_edit.QUALIFIED_ENVELOPE.diffusers_version,
)


class Measured(module.DiffusersBackend):
    """An open gate reporting the loader facts the card measured.

    The synthetic selection itself stays on the CPU, so the load never asks
    for a device; only the envelope's answer is the card's.
    """

    loader_facts = MEASURED

    def instruction_edit_qualified(self) -> bool:
        return True

    def instruction_edit_loader(self, selected):
        return self.loader_facts


class PinnedEditor:
    """The installed Kontext call's names, the VAE scale and a load hook."""

    vae_scale_factor = 8

    def __call__(self, **kwargs):
        raise AssertionError("a loader test invoked generation")

    def enable_vae_slicing(self):
        pass


PinnedEditor.__name__ = PinnedEditor.__qualname__ = "FluxKontextPipeline"
PinnedEditor.__call__.__signature__ = INSTALLED_EDIT_CALL  # type: ignore[attr-defined]


def _write(tree, configs):
    for path, value in configs.items():
        target = tree.joinpath(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _pin(monkeypatch, configs):
    hashes = preflight.validate_configs(
        instruction_edit.KONTEXT_V1.model_id, instruction_edit.KONTEXT_V1.revision, configs,
    ).config_hashes
    monkeypatch.setattr(preflight, "QUALIFIED_CONFIG_HASHES", MappingProxyType(dict(hashes)))


@pytest.fixture
def kontext_hub(synthetic_loader, monkeypatch, tmp_path):
    """The pinned tree on a stubbed Hub, an editor selected on the CPU, and
    a measured stand-in sharing the synthetic loader's release hook."""
    import huggingface_hub

    fixture_backend, control, events = synthetic_loader
    selected = replace(module.settings, model_id=instruction_edit.KONTEXT_V1.model_id, device="cpu",
                       offload="none", quantization="fp8")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    monkeypatch.setattr(loading, "resolve_model", RESOLVE)
    tree = tmp_path / "snapshots" / instruction_edit.KONTEXT_V1.revision
    configs = _preflight_configs()
    _pin(monkeypatch, configs)
    hub = SimpleNamespace(configs=configs, served=configs, downloaded=None, tree=tree, calls=[], pipe=PinnedEditor)

    def snapshot(model_id, *, revision, allow_patterns):
        hub.calls.append(("configs", model_id, revision, tuple(allow_patterns)))
        events.append("configs")
        _write(tree, hub.served)
        return str(tree)

    class Editor:
        @classmethod
        def download(cls, model_id, **kwargs):
            hub.calls.append(("weights", model_id, kwargs))
            events.append("download")
            if hub.downloaded is not None:
                _write(tree, hub.downloaded)
            (tree / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"synthetic")
            return str(tree)

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == str(tree)
            assert kwargs["quantization_config"] is not None
            kwargs.clear()
            events.append("pretrained")
            pipe = hub.pipe()
            control.pipe_ref = weakref.ref(pipe)
            return pipe

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(sys.modules["diffusers"], "FluxKontextPipeline", Editor, raising=False)
    backend = Measured()
    monkeypatch.setattr(backend, "_release", fixture_backend._release)
    return backend, control, events, hub


def test_kontext_fetches_only_pinned_configs_before_weights(kontext_hub):
    backend, _, events, hub = kontext_hub
    # The conftest's refusing `hub_file` stays in place: the pinned editor
    # never reads an index at the Hub's head (`require_custom_pipeline_class`).
    backend.load()
    revision = instruction_edit.KONTEXT_V1.revision
    assert hub.calls == [
        ("configs", instruction_edit.KONTEXT_V1.model_id, revision, preflight.CONFIG_PATHS),
        ("weights", instruction_edit.KONTEXT_V1.model_id, {"revision": revision}),
    ]
    assert events[:4] == ["weights-enter", "configs", "download", "overrides"]
    assert events.index("download") < events.index("pretrained") < events.index("place")
    assert backend.is_ready() and type(backend._pipe).__name__ == "FluxKontextPipeline"
    assert module._EDIT_PARAMS <= backend._call_params


@pytest.mark.parametrize("stage", ["configs", "weights"])
def test_a_changed_config_refuses_before_weights(kontext_hub, stage):
    # A config that differs in the configs pass stops before the weights are
    # asked for; one the weights pass returned changed stops before they load.
    backend, _, events, hub = kontext_hub
    changed = json.loads(json.dumps(hub.configs))
    changed["transformer/config.json"]["secret_marker"] = "do-not-echo"
    if stage == "configs":
        hub.served = changed
    else:
        hub.downloaded = changed
    with pytest.raises(ValueError) as caught:
        backend.load()
    assert str(caught.value) == "kontext-v1 config preflight refused transformer/config.json"
    assert "do-not-echo" not in str(caught.value) and not re.search(r"[0-9a-f]{64}", str(caught.value))
    assert [call[0] for call in hub.calls] == (["configs"] if stage == "configs" else ["configs", "weights"])
    assert not {"overrides", "pretrained", "place"} & set(events)
    assert ("download" in events) is (stage == "weights")
    assert not backend.is_ready()


@pytest.mark.parametrize("broken", ["missing", "oversized", "not-json"])
def test_an_unreadable_pinned_config_is_refused_by_its_path_only(tmp_path, monkeypatch, broken):
    configs = _preflight_configs()
    _pin(monkeypatch, configs)
    _write(tmp_path, configs)
    target = tmp_path / "vae" / "config.json"
    if broken == "missing":
        target.unlink()
    elif broken == "oversized":
        target.write_text(" " * (preflight.MAX_CONFIG_BYTES + 1), encoding="utf-8")
    else:
        target.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        loading.require_kontext_pin(str(tmp_path))
    assert str(caught.value) == "kontext-v1 config preflight refused vae/config.json"


def test_unmeasured_loader_refuses_before_any_download(kontext_hub):
    backend, _, events, hub = kontext_hub
    backend.loader_facts = replace(MEASURED, quantization="int8")
    unmeasured = instruction_edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="int8 quantization")
    for call in (backend.load, lambda: backend._load_prepared(None)):
        with pytest.raises(ValueError) as caught:
            call()
        assert str(caught.value) == unmeasured
    assert hub.calls == [] and events == [] and not backend.is_ready()


@pytest.mark.parametrize("miss", ["class", "vae", "call"])
def test_loaded_class_and_vae_scale_factor_are_required(kontext_hub, miss):
    backend, control, events, hub = kontext_hub

    class Wrong(PinnedEditor):
        pass

    Wrong.__name__ = "FluxPipeline" if miss == "class" else "FluxKontextPipeline"
    if miss == "vae":
        Wrong.vae_scale_factor = 16
    elif miss == "call":
        def call(self, **kwargs):
            raise AssertionError("a loader test invoked generation")

        call.__signature__ = INSTALLED_EDIT_CALL.replace(  # type: ignore[attr-defined]
            parameters=[p for p in INSTALLED_EDIT_CALL.parameters.values() if p.name != "max_area"],
        )
        Wrong.__call__ = call  # type: ignore[method-assign]
    hub.pipe = Wrong
    words = {
        "class": "resolved FluxPipeline, not FluxKontextPipeline",
        "vae": "no VAE scale factor of 8",
        "call": "takes no max_area",
    }[miss]
    with pytest.raises(RuntimeError, match=words):
        backend.load()
    # Loaded, refused and released before publication.
    assert "pretrained" in events and control.releases == 1
    assert not backend.is_ready() and backend._call_params == set()
    gc.collect()
    assert control.pipe_ref() is None


def test_closed_gate_still_refuses_kontext_load(kontext_hub, synthetic_loader):
    _, _, events, hub = kontext_hub

    class Closed(Measured):
        def instruction_edit_qualified(self) -> bool:
            return False

    for backend in (Closed(), module.DiffusersBackend()):
        for call in (backend.load, functools.partial(backend._load_prepared, None)):
            with pytest.raises(ValueError) as caught:
                call()
            assert str(caught.value) == instruction_edit.INSTRUCTION_EDIT_UNAVAILABLE
    assert hub.calls == [] and events == []


@pytest.mark.parametrize("model_id", [
    "owned/custom-kontext-alias", instruction_edit.KONTEXT_V1.model_id.upper(), "Qwen/Qwen-Image-Edit",
])
def test_kontext_class_under_another_id_and_qwen_stay_refused(kontext_hub, monkeypatch, tmp_path, model_id):
    backend, _, events, hub = kontext_hub
    selected = replace(module.settings, model_id=model_id, device="cpu", offload="none", quantization="fp8")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    index = tmp_path / "alias" / "model_index.json"
    index.parent.mkdir()
    index.write_text(json.dumps({"_class_name": instruction_edit.KONTEXT_V1.pipeline_class}), encoding="utf-8")
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: str(index))
    expected = {
        "owned/custom-kontext-alias": instruction_edit.INSTRUCTION_EDIT_UNAVAILABLE,
        "Qwen/Qwen-Image-Edit": instruction_edit.INSTRUCTION_EDITOR_UNSUPPORTED,
    }.get(model_id, instruction_edit.INSTRUCTION_EDIT_EXACT_ID)
    with pytest.raises(ValueError) as caught:
        backend.load()
    assert str(caught.value) == expected
    # Neither the pinned tree nor the explicit class: the holds are unchanged.
    assert hub.calls == [] and not {"overrides", "pretrained", "place"} & set(events)
    assert not backend.is_ready()


@pytest.mark.parametrize("stage", ["configs", "weights"])
def test_offline_maps_to_load_failure(kontext_hub, monkeypatch, stage):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    from backend import main

    backend, _, events, _ = kontext_hub

    def unreachable(*args, **kwargs):
        raise LocalEntryNotFoundError("synthetic: the Hub cannot be reached")

    if stage == "configs":
        monkeypatch.setattr(huggingface_hub, "snapshot_download", unreachable)
    else:
        monkeypatch.setattr(sys.modules["diffusers"].FluxKontextPipeline, "download", unreachable)
    with pytest.raises(LocalEntryNotFoundError) as caught:
        backend.load()
    monkeypatch.setattr(main, "current_model_id", lambda startup: instruction_edit.KONTEXT_V1.model_id)
    message, kind, page = main._load_failure(caught.value)
    assert kind == "offline" and page is None
    assert message.startswith(f"{instruction_edit.KONTEXT_V1.model_id} is not all downloaded yet")
    assert "pretrained" not in events and not backend.is_ready()
