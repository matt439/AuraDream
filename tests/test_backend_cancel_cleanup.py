"""Exceptional base-call offload and ownership contracts in a strict CPU child.

The ordinary suite may already import real Torch. A fresh child loads the actual
production module with fake Torch and an import blocker; no model or device runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_CPU_WORKER = getattr(sys, "_auradream_cancel_cleanup_cpu_worker", False)

if _CPU_WORKER:
    from contextlib import contextmanager
    import importlib.abc
    import sys
    from types import ModuleType, SimpleNamespace

    import pytest

    class NoRuntime(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in {"torch", "torchao", "diffusers", "accelerate", "huggingface_hub"}:
                raise AssertionError("cleanup CPU test imported a real runtime: " + fullname)
            return None

    @pytest.fixture
    def runtime(monkeypatch):
        blocker = NoRuntime()
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
        control = SimpleNamespace(
            events=[],
            grad=False,
            sigma=False,
            failure=None,
            free_failure=None,
            release_failure=None,
            free_calls=0,
            release_calls=0,
            resident=False,
        )

        @contextmanager
        def context(name):
            control.events.append(name + "-enter")
            setattr(control, name, True)
            try:
                yield
            finally:
                setattr(control, name, False)
                control.events.append(name + "-exit")

        class Generator:
            def __init__(self, *, device):
                assert device == "cpu"

            def manual_seed(self, seed):
                self.seed = seed
                return self

        torch = ModuleType("torch")
        torch.Generator = Generator
        torch.no_grad = lambda: context("grad")
        torch.cuda = SimpleNamespace(is_available=lambda: pytest.fail("unexpected device query"))
        monkeypatch.setitem(sys.modules, "torch", torch)
        from backend.backends import diffusers_backend as module

        backend = module.DiffusersBackend()
        monkeypatch.setattr(backend, "_peak_device", lambda: None)
        monkeypatch.setattr(backend, "_require_execution", lambda request: request)
        monkeypatch.setattr(backend, "applied", lambda request: request)
        monkeypatch.setattr(backend, "_apply_lora", lambda *_: None)
        monkeypatch.setattr(module, "current_model_id", lambda _: "black-forest-labs/FLUX.1-Krea-dev")
        monkeypatch.setattr(module.schedule, "requested_shift", lambda *_: (None, None))
        monkeypatch.setattr(module.schedule, "forced_mu", lambda *_: context("sigma"))

        class Pipeline:
            def __call__(self, generator=None, callback_on_step_end=None, **kwargs):
                assert control.grad and control.sigma
                assert not control.resident, "next call began while failed transformer stayed resident"
                control.resident = True
                control.events.append("forward")
                if control.failure is not None:
                    raise control.failure
                for step in range(kwargs["num_inference_steps"]):
                    callback_on_step_end(self, step, None, {})
                self.maybe_free_model_hooks()  # Installed Diffusers' successful tail.
                return SimpleNamespace(images=["synthetic image"])

            def maybe_free_model_hooks(self):
                control.free_calls += 1
                control.events.append("free-hooks")
                if control.free_failure is not None:
                    raise control.free_failure
                control.resident = False

        pipe = Pipeline()
        backend._pipe = pipe
        backend._call_params = {
            "prompt",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "num_images_per_prompt",
            "generator",
            "callback_on_step_end",
        }

        def release():
            assert not backend._lock.locked()
            assert backend._pipe is None and backend._refine_pipe is None and backend._inpaint_pipe is None
            control.release_calls += 1
            control.events.append("release")
            if control.release_failure is not None:
                raise control.release_failure

        monkeypatch.setattr(backend, "_release", release)
        request = SimpleNamespace(
            execution=None,
            model_fields_set=set(),
            prompt="synthetic",
            negative_prompt="",
            steps=2,
            width=256,
            height=256,
            guidance_scale=3.5,
            true_cfg=1.0,
            mu=None,
            num_images=1,
            runs=1,
            reference=None,
            refine=None,
            edit=None,
            mask=None,
            outpaint=None,
            mask_crop=None,
            inpainting=False,
            crop_anchor="center",
            upscaler=None,
            strength=0.1,
            lora=None,
            denoise_size=lambda: (256, 256),
        )
        return SimpleNamespace(module=module, backend=backend, pipe=pipe, control=control, request=request)

    @pytest.mark.parametrize("failure_kind", ["cancel", "error"])
    def test_actual_generate_failed_call_offloads_then_next_call_succeeds(runtime, failure_kind):
        r = runtime
        progress = []
        primary = RuntimeError("synthetic denoise failure")
        if failure_kind == "error":
            r.control.failure = primary
        with pytest.raises(r.module.Cancelled if failure_kind == "cancel" else RuntimeError) as caught:
            r.backend.generate(
                r.request,
                seed=41,
                on_progress=lambda step, total: progress.append(step),
                should_cancel=lambda: failure_kind == "cancel" and bool(progress),
            )
        if failure_kind == "error":
            assert caught.value is primary
        assert r.control.free_calls == 1 and not r.control.resident and r.backend.is_ready()
        assert r.control.events.index("forward") < r.control.events.index("free-hooks")
        r.control.failure = None
        assert r.backend.generate(r.request, seed=42) == ["synthetic image"]
        assert r.control.free_calls == 2  # Failed-call cleanup, then one ordinary successful tail.
        assert r.control.events.count("forward") == 2 and r.control.release_calls == 0
        assert not r.control.grad and not r.control.sigma

    def test_success_uses_only_existing_pipeline_tail(runtime):
        r = runtime
        assert r.backend.generate(r.request, seed=41) == ["synthetic image"]
        assert r.control.free_calls == 1 and r.control.release_calls == 0

    @pytest.mark.parametrize("release_failure", [False, True])
    @pytest.mark.parametrize("primary_kind", ["cancel", "error"])
    def test_cleanup_failure_preserves_primary_and_detaches_before_next_call(runtime, release_failure, primary_kind):
        r = runtime
        primary = (
            r.module.Cancelled("synthetic cancellation")
            if primary_kind == "cancel"
            else RuntimeError("synthetic primary")
        )
        r.control.failure = primary
        r.control.free_failure = OSError("synthetic offload failure")
        if release_failure:
            r.control.release_failure = RuntimeError("synthetic release failure")
        r.backend._refine_pipe = object()
        r.backend._inpaint_pipe = object()
        with pytest.raises(type(primary)) as caught:
            r.backend.generate(r.request, seed=41)
        assert caught.value is primary and not r.backend.is_ready()
        assert r.control.free_calls == 1 and r.control.release_calls == 1
        notes = " ".join(primary.__notes__)
        assert "synthetic offload failure" in notes
        assert ("synthetic release failure" in notes) == release_failure
        with pytest.raises(RuntimeError, match="model is not loaded"):
            r.backend.generate(r.request, seed=42)
        assert r.control.events.count("forward") == 1

    @pytest.mark.parametrize("ownership", ["owner", "disposal", "pending", "replacement"])
    def test_helper_never_touches_uncertain_or_detached_cache_ownership(runtime, ownership):
        r = runtime
        if ownership == "owner":
            r.backend._cache_owner = object()
        elif ownership == "disposal":
            r.backend._cache_disposal = object()
        elif ownership == "pending":
            r.backend._pending_cache_disposal = object()
        else:
            r.backend._pipe = object()
        r.backend._cleanup_failed_text_image(r.pipe, RuntimeError("primary"))
        assert r.control.free_calls == r.control.release_calls == 0

    def test_actual_disposal_closes_and_drains_before_offload_outside_no_grad_and_sigma(runtime, monkeypatch):
        r = runtime
        r.request.model_fields_set = {"execution"}  # Existing explicit-Off disposal scope.
        primary = RuntimeError("private cache call failure")

        class Owner:
            def close(self):
                assert not r.control.grad and not r.control.sigma
                r.control.events.append("owner-close")

            def discard(self):
                pytest.fail("a clean owner was discarded")

        class Disposal:
            pending = None

            def drain(self):
                assert not r.control.grad and not r.control.sigma
                assert r.backend._cache_owner is None and r.backend._cache_disposal is None
                r.control.events.append("drain")

        owner = Owner()
        disposer = Disposal()

        def forward(pipe, kwargs, *, disposal):
            assert r.control.grad and r.control.sigma
            disposal.cache = owner
            disposal.disposal = disposer
            r.backend._cache_owner = owner
            r.backend._cache_disposal = disposer
            r.control.events.append("private-forward")
            r.control.resident = True
            raise primary

        monkeypatch.setattr(r.backend, "_invoke_pipeline", forward)
        with pytest.raises(RuntimeError) as caught:
            r.backend.generate(r.request, seed=41)
        assert caught.value is primary
        assert r.control.free_calls == 1 and r.control.release_calls == 0 and not r.control.resident
        assert r.control.events == [
            "grad-enter",
            "sigma-enter",
            "private-forward",
            "sigma-exit",
            "grad-exit",
            "owner-close",
            "drain",
            "free-hooks",
        ]
        assert r.backend.is_ready() and r.backend._cache_owner is None

    def test_cold_cancel_after_admission_may_safely_offload_without_a_forward(runtime):
        r = runtime
        with pytest.raises(r.module.Cancelled):
            r.backend.generate(r.request, seed=41, should_cancel=lambda: True)
        assert r.control.events == ["free-hooks"] and r.control.free_calls == 1
        assert r.backend.is_ready() and not r.control.resident

    def test_no_real_framework_or_driver_was_imported(runtime):
        assert not hasattr(sys.modules["torch"], "__file__")
        assert isinstance(sys.modules["torch"], ModuleType)
        assert "diffusers" not in sys.modules and "torchao" not in sys.modules and "accelerate" not in sys.modules

    def test_failed_cleanup_cannot_detach_a_replacement_pipeline(runtime, monkeypatch):
        r = runtime
        replacement = object()
        primary = RuntimeError("primary")

        def fail_and_replace():
            r.backend._pipe = replacement
            raise OSError("cleanup failed after replacement")

        monkeypatch.setattr(r.pipe, "maybe_free_model_hooks", fail_and_replace)
        r.backend._cleanup_failed_text_image(r.pipe, primary)
        assert r.backend._pipe is replacement and r.control.release_calls == 0
        assert "cleanup failed after replacement" in " ".join(primary.__notes__)

    @pytest.mark.parametrize("gate", ["execution", "cache-candidate"])
    def test_admission_refusal_still_precedes_any_cleanup_effect(runtime, monkeypatch, gate):
        r = runtime

        def refuse(*args, **kwargs):
            raise ValueError("closed admission")

        if gate == "execution":
            monkeypatch.setattr(r.backend, "_require_execution", refuse)
        else:
            r.request.execution = r.module.ExecutionSpec(step_cache="fbc-005-v1")
            monkeypatch.setattr(r.backend, "_require_request_cache_candidate", refuse)
        with pytest.raises(ValueError, match="closed admission"):
            r.backend.generate(r.request, seed=41)
        assert r.control.events == [] and r.control.free_calls == 0

    @pytest.mark.parametrize("kind", ["cancel", "failure"])
    def test_failed_owner_close_detach_and_disposal_are_not_touched_by_offload(runtime, monkeypatch, kind):
        r = runtime
        r.request.model_fields_set = {"execution"}
        primary = r.module.Cancelled("primary cancellation") if kind == "cancel" else RuntimeError("primary failure")

        class Owner:
            def close(self):
                assert not r.control.grad and not r.control.sigma
                r.control.events.append("owner-close-failed")
                raise RuntimeError("synthetic cache close failure")

            def discard(self):
                r.control.events.append("owner-discard")

        class Disposal:
            pending = None

            def detach(self):
                assert not r.control.grad and not r.control.sigma
                r.control.events.append("detach")
                r.backend._pipe = None
                r.backend._cache_owner = None
                r.backend._cache_disposal = None
                self.pending = object()
                r.backend._pending_cache_disposal = self

            def drain(self):
                assert not r.control.grad and not r.control.sigma
                r.control.events.append("drain")
                self.pending = None
                r.backend._pending_cache_disposal = None

        owner = Owner()
        disposer = Disposal()

        def forward(pipe, kwargs, *, disposal):
            disposal.cache = owner
            disposal.disposal = disposer
            r.backend._cache_owner = owner
            r.backend._cache_disposal = disposer
            raise primary

        monkeypatch.setattr(r.backend, "_invoke_pipeline", forward)
        with pytest.raises(type(primary)) as caught:
            r.backend.generate(r.request, seed=41)
        assert caught.value is primary and "synthetic cache close failure" in " ".join(primary.__notes__)
        assert not r.backend.is_ready() and r.backend._pending_cache_disposal is None
        assert r.control.free_calls == r.control.release_calls == 0
        assert r.control.events == [
            "grad-enter",
            "sigma-enter",
            "sigma-exit",
            "grad-exit",
            "owner-close-failed",
            "owner-discard",
            "detach",
            "drain",
        ]

    @pytest.mark.parametrize("pass_kind", ["refine", "edit"])
    def test_other_passes_keep_their_existing_exception_cleanup_owner(runtime, monkeypatch, pass_kind):
        r = runtime
        setattr(r.request, pass_kind, object())
        primary = RuntimeError("other pass failure")

        def failed(*args, **kwargs):
            raise primary

        monkeypatch.setattr(r.backend, "_generate_images_prepared", failed)
        with pytest.raises(RuntimeError) as caught:
            r.backend.generate(r.request, seed=41)
        assert caught.value is primary and r.control.free_calls == 0

    @pytest.mark.parametrize("after_partial_detach", [False, True])
    def test_detach_failure_preserves_primary_and_uses_one_fail_closed_snapshot(
        runtime, monkeypatch, after_partial_detach, caplog
    ):
        r = runtime
        primary = r.module.Cancelled("original cancellation")
        r.control.failure = primary
        r.control.free_failure = RuntimeError("offload failure")
        r.backend._refine_pipe = object()
        r.backend._inpaint_pipe = object()
        original = r.backend._detach_pipeline_locked

        def fail_detach():
            if after_partial_detach:
                original()
            raise RuntimeError("synthetic detach failure")

        monkeypatch.setattr(r.backend, "_detach_pipeline_locked", fail_detach)
        caplog.set_level("DEBUG", logger=r.module.log.name)
        with pytest.raises(r.module.Cancelled) as caught:
            r.backend.generate(r.request, seed=41)
        assert caught.value is primary and not r.backend.is_ready()
        assert r.backend._refine_pipe is None and r.backend._inpaint_pipe is None
        assert r.control.release_calls == 1 and r.control.free_calls == 1
        assert "synthetic detach failure" in " ".join(primary.__notes__)
        errors = [
            record for record in caplog.records if record.levelname == "ERROR" and record.name == r.module.log.name
        ]
        assert len(errors) == 1 and "offload failure" in errors[0].getMessage()
        traces = [record for record in caplog.records if record.levelname == "DEBUG" and record.exc_info]
        assert any(record.exc_info[1] is r.control.free_failure for record in traces)
        assert any("synthetic detach failure" in str(record.exc_info[1]) for record in traces)

    @pytest.mark.parametrize("detach_failure", [False, True])
    def test_cleanup_snapshot_closes_attention_owner_exactly_once(runtime, monkeypatch, detach_failure):
        r = runtime
        closed = []

        class Attention:
            def restore(self):
                closed.append("restored")

        r.backend._attention_selection = Attention()
        r.control.free_failure = RuntimeError("offload failed")
        if detach_failure:

            def fail():
                raise RuntimeError("detach failed")

            monkeypatch.setattr(r.backend, "_detach_pipeline_locked", fail)
        r.backend._cleanup_failed_text_image(r.pipe, r.module.Cancelled("primary"))
        assert closed == ["restored"] and r.control.release_calls == 1 and not r.backend.is_ready()

    def test_actual_diagnostic_retirement_failure_fails_closed_preserving_primary(runtime, monkeypatch, caplog):
        r = runtime
        calls = []

        class Registration:
            def retire_diagnostics(self):
                calls.append("retire")
                raise OSError("actual retirement failure")

            def close(self):
                calls.append("close")

        monkeypatch.setattr(r.module.regional_compile, "RegionalRegistration", Registration)
        r.backend._regional_registration = Registration()
        r.backend._refine_pipe = object()
        r.backend._inpaint_pipe = object()
        primary = r.module.Cancelled("original cancellation")

        def fail_forward(*args, **kwargs):
            raise primary

        monkeypatch.setattr(r.backend, "_invoke_pipeline", fail_forward)
        r.control.free_failure = RuntimeError("offload failed")
        caplog.set_level("DEBUG", logger=r.module.log.name)
        with pytest.raises(r.module.Cancelled) as caught:
            r.backend.generate(r.request, seed=41)
        assert caught.value is primary and "actual retirement failure" in " ".join(primary.__notes__)
        assert calls == ["retire", "close"] and r.control.release_calls == 1
        assert not r.backend.is_ready() and r.backend._refine_pipe is None and r.backend._inpaint_pipe is None
        assert r.backend._regional_registration is None
        assert (
            len(
                [
                    record
                    for record in caplog.records
                    if record.levelname == "ERROR" and record.name == r.module.log.name
                ]
            )
            == 1
        )

else:

    def test_backend_cancel_cleanup_in_strict_cpu_worker(tmp_path):
        root = Path(__file__).resolve().parents[1]
        env = {name: value for name, value in os.environ.items() if not name.startswith("AURADREAM_")}
        env.update(
            {
                "AURADREAM_DB_PATH": str(tmp_path / "scratch.db"),
                "AURADREAM_OUTPUTS_DIR": str(tmp_path / "outputs"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONUTF8": "1",
            }
        )
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--cpu-worker"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "26 passed" in result.stdout, result.stdout + result.stderr


if __name__ == "__main__":
    if sys.argv[1:] != ["--cpu-worker"]:
        raise SystemExit("run with pytest, or the internal --cpu-worker option")
    sys._auradream_cancel_cleanup_cpu_worker = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(pytest.main(["--noconftest", "-q", "-o", "addopts=", str(Path(__file__).resolve())]))
