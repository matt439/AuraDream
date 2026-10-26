"""Actual backend load/publication paths over strictly synthetic dependencies."""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.backends import diffusers_backend as module
from backend.execution import ExecutionSpec


class Registration:
    def __init__(self, failure=None):
        self.closed = 0
        self.failure = failure

    def close(self):
        self.closed += 1
        if self.failure is not None:
            raise self.failure


@pytest.fixture
def synthetic_loader(monkeypatch):
    backend = module.DiffusersBackend()
    events = []
    control = SimpleNamespace(stage=None, error=RuntimeError("synthetic preparation failed"),
                              release_error=None, releases=0, pipe_ref=None, component_ref=None,
                              collect_at_release=False, overrides=False)
    selected = replace(module.settings, model_id="black-forest-labs/FLUX.1-Krea-dev",
                       device="cpu", offload="none", quantization="fp8")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)

    def stage(name):
        events.append(name)
        assert backend._pipe is None and backend._call_params == set()
        if control.stage == name:
            raise control.error

    @contextmanager
    def weights():
        stage("weights-enter")
        try:
            yield
        finally:
            stage("weights-exit")

    @contextmanager
    def shards(enabled):
        assert enabled is True
        stage("shards-enter")
        try:
            yield
        finally:
            stage("shards-exit")

    class Pipeline:
        def __call__(self, generator=None, callback_on_step_end=None):
            raise AssertionError("a synthetic ownership test invoked generation")

        def enable_vae_slicing(self):
            stage("slice")

        def enable_vae_tiling(self):
            stage("tile")

        def set_progress_bar_config(self, **kwargs):
            assert kwargs == {"disable": False}
            stage("console")

    class MissingCallback:
        def __call__(self, generator=None):
            raise AssertionError("an invalid synthetic pipeline was invoked")

    class Component:
        pass

    class Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == selected.model_id
            assert kwargs["dtype"] is fake_torch.bfloat16
            assert kwargs["quantization_config"] is quant
            # Clear an injected override before the deliberate failure: this
            # stub must not retain it in its own exception frame.
            kwargs.clear()
            stage("pretrained")
            pipe = MissingCallback() if control.stage == "signature" else Pipeline()
            control.pipe_ref = weakref.ref(pipe)
            return pipe

    def overrides(*args, **kwargs):
        stage("overrides")
        if not control.overrides:
            return {}
        component = Component()
        control.component_ref = weakref.ref(component)
        return {"text_encoder": component}

    def release():
        assert not backend._lock.locked(), "release must run outside the backend lock"
        assert backend._pipe is None and backend._regional_registration is None
        assert backend._refine_pipe is None and backend._inpaint_pipe is None
        control.releases += 1
        events.append("release")
        if control.collect_at_release:
            gc.collect()
            assert control.pipe_ref is None or control.pipe_ref() is None
            assert control.component_ref is None or control.component_ref() is None
        if control.release_error is not None:
            raise control.release_error

    quant = object()
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.device = lambda name: SimpleNamespace(type="cpu")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: pytest.fail("synthetic load observed a device"))
    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.pipeline_utils"):
        stub = ModuleType(name)
        stub.__path__ = []
        stub.DiffusionPipeline = Loader
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(module.loading, "weights_loading", weights)
    monkeypatch.setattr(module.loading, "checkpoint_files_released", shards)
    monkeypatch.setattr(module.loading, "component_overrides", overrides)
    monkeypatch.setattr(module.loading, "resolve_model", lambda model_id, **kwargs: model_id)
    monkeypatch.setattr(module.loading, "quantization_config", lambda settings: quant)
    monkeypatch.setattr(module.loading, "unused_components", lambda model_id: (None, {}))
    monkeypatch.setattr(module.loading, "place_pipeline", lambda *args: stage("place"))
    for resolver in (module.anima_loading, module.netayume_loading, module.newbie_loading, module.single_file):
        monkeypatch.setattr(resolver, "resolve_model", lambda model_id, quantization: model_id)
    monkeypatch.setattr(module.anima_loading, "matches", lambda model_id: False)
    monkeypatch.setattr(module.schedule, "warn_about_post_processing", lambda *args: stage("post"))
    monkeypatch.setattr(backend, "_release", release)
    info = module.log.info

    def log_info(message, *args):
        if message.startswith("model ready:"):
            stage("ready-log")
        info(message, *args)

    monkeypatch.setattr(module.log, "info", log_info)
    return backend, control, events


def test_actual_eager_load_publishes_only_after_every_preparation_and_context_exit(synthetic_loader):
    backend, control, events = synthetic_loader
    backend.load()
    assert backend.is_ready() and control.pipe_ref() is backend._pipe
    assert backend._call_params == {"generator", "callback_on_step_end"}
    assert backend._loaded_execution is None and backend._regional_registration is None
    assert events == ["weights-enter", "overrides", "shards-enter", "pretrained", "shards-exit",
                      "place", "slice", "tile", "console", "post", "weights-exit", "ready-log"]
    backend.load()
    assert events[-1] == "ready-log" and events.count("pretrained") == 1
    backend.unload()
    assert control.releases == 1 and control.pipe_ref() is None


@pytest.mark.parametrize("stage", ["pretrained", "shards-exit", "place", "signature", "console", "post",
                                   "weights-exit", "ready-log"])
def test_failed_actual_load_never_publishes_and_discards_owned_candidate(synthetic_loader, stage):
    backend, control, events = synthetic_loader
    control.stage = stage
    # A signature refusal and context exit fail within application frames
    # containing no library-owned pipeline argument. Retain the exception and
    # prove those frames no longer own the failed pipeline at release.
    control.collect_at_release = stage in {"signature", "shards-exit", "weights-exit", "ready-log"}
    with pytest.raises(RuntimeError) as raised:
        backend.load()
    if stage == "signature":
        assert "takes no callback_on_step_end" in str(raised.value)
    else:
        assert raised.value is control.error
    # __exit__ attaches cleanup failures to the primary exception. Assertions
    # inside release must therefore be checked again outside that context.
    assert not getattr(raised.value, "__notes__", [])
    if control.collect_at_release:
        assert control.pipe_ref is None or control.pipe_ref() is None
    assert not backend.is_ready() and backend._call_params == set()
    assert backend._loaded_execution is None and backend._regional_registration is None
    assert control.releases == int(stage != "pretrained")
    if control.releases:
        assert events[-1] == "release"
    backend.unload()
    assert control.releases == int(stage != "pretrained")


@pytest.mark.parametrize("stage", ["shards-enter", "pretrained"])
def test_overrides_are_owned_even_when_from_pretrained_never_returns_a_pipeline(synthetic_loader, stage):
    backend, control, events = synthetic_loader
    control.overrides = True
    control.stage = stage
    control.collect_at_release = True
    with pytest.raises(RuntimeError) as raised:
        backend.load()
    assert not getattr(raised.value, "__notes__", [])
    assert raised.value is control.error and control.releases == 1
    assert control.component_ref() is None and control.pipe_ref is None
    assert events[-1] == "release" and not backend.is_ready()


def test_cleanup_failure_preserves_original_load_exception(synthetic_loader):
    backend, control, events = synthetic_loader
    control.stage = "weights-exit"
    control.release_error = OSError("synthetic release failed")
    with pytest.raises(RuntimeError) as raised:
        backend.load()
    assert raised.value is control.error
    assert any("Discarding the pipeline candidate also failed: OSError: synthetic release failed" in note
               for note in raised.value.__notes__)
    assert not backend.is_ready() and control.releases == 1


@pytest.mark.parametrize("method", ["unload", "try_unload"])
def test_both_unload_paths_drop_all_owned_references_once_and_keep_compatibility_caches(synthetic_loader, method):
    backend, control, events = synthetic_loader
    backend.load()
    registration = Registration()
    backend._regional_registration = registration
    backend._loaded_execution = ExecutionSpec(compilation="regional-v1")
    backend._lora = ("adapter.safetensors", 0.0)
    backend._refine_pipe, backend._refine_params = object(), {"image"}
    backend._inpaint_pipe, backend._inpaint_params = object(), {"mask_image"}
    sentinel = object()
    backend._refine_class = backend._inpaint_class = sentinel
    backend._denoiser_cache = sentinel
    backend._denoiser_model_id = "retained-model"
    backend._console_progress = False
    result = getattr(backend, method)()
    assert result is (True if method == "try_unload" else None)
    assert not backend.is_ready() and backend._call_params == set()
    assert backend._lora is None and backend._refine_params == backend._inpaint_params == set()
    assert backend._loaded_execution is None and backend._regional_registration is None
    assert backend._refine_class is backend._inpaint_class is backend._denoiser_cache is sentinel
    assert backend._denoiser_model_id == "retained-model" and backend._console_progress is False
    assert registration.closed == 1 and control.releases == 1
    getattr(backend, method)()
    assert registration.closed == 1 and control.releases == 1


def test_try_unload_contention_preserves_state_without_cleanup(synthetic_loader):
    backend, control, events = synthetic_loader
    backend.load()
    before = backend._pipe, backend._call_params
    with backend._lock:
        assert backend.try_unload() is False
        assert (backend._pipe, backend._call_params) == before
    assert control.releases == 0


def test_private_regional_ownership_transfers_once_but_all_eager_entrypoints_stay_closed():
    backend = module.DiffusersBackend()
    released = []
    registration = Registration()
    with module._PipelineCandidate(lambda: released.append(True), pipe=object(),
                                   params={"generator", "callback_on_step_end"}, registration=registration,
                                   execution=ExecutionSpec(compilation="regional-v1")) as candidate:
        with backend._lock:
            backend._publish_candidate_locked(candidate)
        assert candidate.pipe is None and candidate.registration is None
    assert released == [] and registration.closed == 0
    request = SimpleNamespace(execution=None)
    backend._peak_device = lambda: pytest.fail("private regional state reached device work")
    for call in (lambda: backend.applied(request), lambda: backend.generate(request, 1),
                 lambda: backend._generate_images(request, 1)):
        with pytest.raises(ValueError, match="not available"):
            call()
    backend._release = lambda: released.append(True)
    backend.unload()
    assert released == [True] and registration.closed == 1


def test_abandoned_registration_cleanup_attempts_release_and_preserves_primary_failure():
    original = RuntimeError("preparation failed")
    registration = Registration(OSError("handle close failed"))
    released = []

    def release():
        released.append(True)
        raise OSError("release failed")

    with pytest.raises(RuntimeError) as raised:
        with module._PipelineCandidate(release, pipe=object(), registration=registration):
            raise original
    assert raised.value is original and registration.closed == 1 and released == [True]
    assert "handle close failed" in original.__notes__[0]
    assert "release failed" in registration.failure.__notes__[0]


@pytest.mark.parametrize("change", [{"unknown": None}, {"compilation": "future"}])
def test_copied_candidate_and_loaded_recipes_are_revalidated_before_publication_or_work(change):
    backend = module.DiffusersBackend()
    copied = ExecutionSpec().model_copy(update=change)
    released = []
    with pytest.raises(ValidationError):
        with module._PipelineCandidate(lambda: released.append(True), pipe=object(), execution=copied) as candidate:
            backend._publish_candidate_locked(candidate)
    assert not backend.is_ready() and released == [True]
    backend._pipe, backend._loaded_execution = object(), copied
    backend._peak_device = lambda: pytest.fail("malformed loaded state reached device work")
    for call in (lambda: backend.applied(SimpleNamespace(execution=None)),
                 lambda: backend.generate(SimpleNamespace(execution=None), 1),
                 lambda: backend._generate_images(SimpleNamespace(execution=None), 1)):
        with pytest.raises(ValidationError):
            call()


def test_candidate_cannot_replace_working_weights_or_orphan_a_registration():
    backend = module.DiffusersBackend()
    working = backend._pipe = object()
    released = []
    with pytest.raises(RuntimeError, match="working weights"):
        with module._PipelineCandidate(lambda: released.append(True), pipe=object()) as candidate:
            backend._publish_candidate_locked(candidate)
    assert backend._pipe is working and released == [True]
    backend._pipe = None
    registration = backend._regional_registration = Registration()
    with pytest.raises(RuntimeError, match="working weights"):
        with module._PipelineCandidate(lambda: released.append(True), pipe=object()) as candidate:
            backend._publish_candidate_locked(candidate)
    assert backend._regional_registration is registration and registration.closed == 0


@pytest.mark.parametrize("recipe, handle", [(None, True), ("regional-v1", False)])
def test_incomplete_regional_ownership_is_not_published(recipe, handle):
    backend = module.DiffusersBackend()
    registration = Registration() if handle else None
    released = []
    with pytest.raises(ValueError, match="must transfer together"):
        with module._PipelineCandidate(lambda: released.append(True), pipe=object(), registration=registration,
                                       execution=ExecutionSpec(compilation=recipe)) as candidate:
            backend._publish_candidate_locked(candidate)
    assert not backend.is_ready() and released == [True]
    assert registration is None or registration.closed == 1


def test_backend_ownership_import_and_refusal_are_cold_in_a_fresh_process(tmp_path):
    child = r'''
import importlib.abc
import json
import sys
from types import SimpleNamespace
blocked = {"torch", "torchao", "diffusers", "huggingface_hub"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("ownership imported a runtime dependency: " + fullname)
        return None
sys.meta_path.insert(0, ColdOnly())
from backend.backends.diffusers_backend import DiffusersBackend
from backend.execution import ExecutionSpec
backend = DiffusersBackend()
backend._pipe = object()
backend._loaded_execution = ExecutionSpec(compilation="regional-v1")
request = SimpleNamespace(execution=None)
for call in (lambda: backend.applied(request), lambda: backend.generate(request, 1),
             lambda: backend._generate_images(request, 1)):
    try:
        call()
    except ValueError as error:
        assert "not available" in str(error)
    else:
        raise AssertionError("nonempty loaded state executed")
assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
print(json.dumps({"cold": True, "refused": 3}))
'''
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    done = subprocess.run([sys.executable, "-B", "-c", child], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    (tmp_path / "ownership.stdout").write_text(done.stdout, encoding="utf-8")
    (tmp_path / "ownership.stderr").write_text(done.stderr, encoding="utf-8")
    assert done.returncode == 0, done.stdout + "\n" + done.stderr
    assert json.loads(done.stdout) == {"cold": True, "refused": 3}
