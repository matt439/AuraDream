"""Owned scratch startup contract only; no compiler/framework/device probes."""
from __future__ import annotations

import ast
from dataclasses import replace
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from backend import compiler_cache as module
from backend.execution import CompilerMetadata


FACTS = CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, "3.7.1.post27")


@pytest.fixture
def world(tmp_path, monkeypatch):
    # Inject observations, not real consumer imports or the process environment.
    runtime = SimpleNamespace(**{name: getattr(os, name) for name in (
        "name", "path", "fstat", "write", "fsync", "close", "open", "O_RDWR", "O_CREAT", "O_EXCL",
        "O_BINARY", "O_TEMPORARY", "O_TMPFILE",
    ) if hasattr(os, name)}, environ={})
    monkeypatch.setattr(module, "os", runtime)
    monkeypatch.setattr(module, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(module, "_attempt", None)
    monkeypatch.setattr(module, "_sealed", False)
    monkeypatch.setattr(module, "_startup_thread", None)
    plan = module.cache_plan(tmp_path / "gallery.db", tmp_path / "outputs", None, FACTS, web=tmp_path / "web")
    return plan, runtime


def start(plan):
    return module._owned_startup(lambda: plan)


def require(plan):
    return module.require_binding(plan.db, plan.outputs, plan.bundles, plan.facts, web=plan.web)


def test_none_is_cold_and_seals_even_the_unused_window(world, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("None reached cache I/O, environment or consumer checks")
    monkeypatch.setattr(module, "_initialize", forbidden)
    monkeypatch.setattr(module, "_no_consumers", forbidden)
    monkeypatch.setattr(module, "_environment", forbidden)
    assert module._owned_startup() is None
    assert module._attempt is None and module._sealed
    with pytest.raises(RuntimeError, match="window is closed"):
        start(world[0])
    assert not world[0].root.exists()


def test_plan_is_pure_deterministic_contained_and_runtime_specific(world, monkeypatch):
    plan, _ = world
    def forbidden(*args, **kwargs):
        pytest.fail("planning touched filesystem")
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "lstat", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    assert module.cache_plan(plan.db, plan.outputs, None, FACTS, web=plan.web) == plan
    assert plan.inductor.parent == plan.directory == plan.triton.parent
    assert plan.directory.parent == plan.root and plan.root.parent == plan.db.parent
    variants = [replace(FACTS, **{field: value}) for field, value in (
        ("platform", "linux"), ("python_version", (3, 12)), ("torch", "2.13.1"),
        ("diffusers", "0.40.1"), ("torchao", "0.18.1"), ("triton", "3.7"),
        ("triton_windows", None),
    )]
    assert all(module.cache_plan(plan.db, plan.outputs, None, facts, web=plan.web).namespace != plan.namespace for facts in variants)
    assert module.cache_plan(plan.db.with_name("other.db"), plan.outputs, None, FACTS, web=plan.web).namespace != plan.namespace


@pytest.mark.parametrize("which", ["outputs", "bundles", "web", "db-parent", "inside-outputs", "inside-bundles", "inside-web"])
def test_overlapping_override_refused_without_writes(world, which):
    plan, runtime = world
    root = {"outputs": plan.outputs, "bundles": plan.bundles, "db-parent": plan.db.parent,
            "web": plan.web, "inside-web": plan.web / "cache",
            "inside-outputs": plan.outputs / "cache", "inside-bundles": plan.bundles / "cache"}[which]
    with pytest.raises(ValueError, match="separate directory"):
        module.cache_plan(plan.db, plan.outputs, plan.bundles, FACTS, root=root, web=plan.web)
    assert not plan.root.exists() and not runtime.environ


def test_owned_binding_survives_repeated_verification_and_has_no_probe_files(world, monkeypatch):
    plan, runtime = world
    binding = start(plan)
    assert binding is not None and binding.plan == plan
    assert runtime.environ == dict(binding.environment)
    assert all(path.is_dir() for path in (plan.root, plan.directory, plan.inductor, plan.triton))
    assert list(plan.inductor.iterdir()) == list(plan.triton.iterdir()) == []
    assert module._attempt.status == "bound" and module._attempt.operation is None
    monkeypatch.setattr(module, "_probe", lambda *args: pytest.fail("repeat probed"))
    monkeypatch.setattr(module, "_mkdir", lambda *args: pytest.fail("repeat created directory"))
    assert require(plan) is require(plan) is binding
    # Compiler consumers may import after startup. This is verification, not rebinding.
    module.sys.modules["torch._inductor"] = object()
    assert require(plan) is binding


def test_failed_eager_setup_and_unload_preserve_process_binding_and_cache(world, monkeypatch):
    from tests.test_pipeline_ownership import synthetic_loader
    plan, runtime = world
    binding = start(plan)
    backend, control, _events = synthetic_loader.__wrapped__(monkeypatch)
    before = dict(runtime.environ)
    control.stage = "pretrained"
    with pytest.raises(RuntimeError, match="synthetic preparation failed"):
        backend.load()
    assert require(plan) is binding and runtime.environ == before
    control.stage = None
    backend.load()
    backend.unload()
    assert require(plan) is binding and runtime.environ == before and plan.inductor.is_dir()


def test_private_override_is_retained_by_verification(world):
    original, runtime = world
    plan = module.cache_plan(original.db, original.outputs, None, FACTS, root=original.db.parent / "private-cache", web=original.web)
    assert start(plan).plan == plan
    assert require(plan).plan == plan
    assert module._attempt.assigned == list(module._KEYS)


@pytest.mark.parametrize("key", module._KEYS)
def test_even_matching_initial_environment_is_unowned_and_preserved(world, key):
    plan, runtime = world
    value = str(plan.inductor if key == "TORCHINDUCTOR_CACHE_DIR" else plan.triton)
    runtime.environ[key] = value
    with pytest.raises(RuntimeError, match="already present"):
        start(plan)
    assert runtime.environ == {key: value} and not plan.root.exists()


@pytest.mark.parametrize("key,value", [("TORCHINDUCTOR_CACHE_DIR", ""), ("TRITON_CACHE_DIR", "foreign-secret"),
                                       ("TORCHINDUCTOR_CACHE_DIR", "relative")])
def test_conflicting_external_environment_is_preserved_before_fs(world, key, value):
    plan, runtime = world
    runtime.environ[key] = value
    with pytest.raises(RuntimeError, match="external compiler cache") as failure:
        start(plan)
    assert value not in str(failure.value) if value else True
    assert runtime.environ == {key: value} and not plan.root.exists()
    assert module._sealed and module._attempt.status == "failed"


@pytest.mark.parametrize("consumer", ["torch._inductor", "torch._dynamo.config", "triton.runtime"])
def test_preimported_consumer_refuses_before_fs_or_env(world, consumer):
    plan, runtime = world
    module.sys.modules[consumer] = object()
    with pytest.raises(RuntimeError, match="already imported"):
        start(plan)
    assert not plan.root.exists() and not runtime.environ


def test_module_observation_is_rechecked_before_environment_publication(world, monkeypatch):
    plan, runtime = world
    real = module._probe
    def probe(path, attempt):
        real(path, attempt)
        module.sys.modules["torch._dynamo"] = object()
    monkeypatch.setattr(module, "_probe", probe)
    with pytest.raises(RuntimeError, match="already imported"):
        start(plan)
    assert plan.inductor.is_dir() and not runtime.environ


@pytest.mark.parametrize("change", ["db", "outputs", "bundles", "web", "facts", "root"])
def test_changed_context_refuses_instead_of_rebinding(world, change):
    plan, runtime = world
    start(plan)
    before = dict(runtime.environ)
    changed = replace(plan, **{change: replace(FACTS, torch="changed") if change == "facts"
                              else getattr(plan, change).with_name("changed")})
    with pytest.raises(RuntimeError, match="binding changed"):
        if change == "root":
            module._initialize(changed)
        else:
            require(changed)
    assert runtime.environ == before


@pytest.mark.parametrize("key", module._KEYS)
def test_changed_environment_refuses_without_repair(world, key):
    plan, runtime = world
    start(plan)
    runtime.environ[key] = "intervening-owner"
    with pytest.raises(RuntimeError, match="environment changed"):
        require(plan)
    assert runtime.environ[key] == "intervening-owner"


def test_changed_directory_identity_refuses_without_deleting_either_tree(world):
    plan, _ = world
    start(plan)
    old = plan.inductor.with_name("old-inductor")
    plan.inductor.rename(old)
    plan.inductor.mkdir()
    with pytest.raises(RuntimeError, match="identity changed"):
        require(plan)
    assert old.is_dir() and plan.inductor.is_dir()


@pytest.mark.parametrize("protected", ["db", "outputs", "bundles", "web", "root"])
def test_protected_reparse_path_drift_refuses_without_repair(world, monkeypatch, protected):
    plan, runtime = world
    start(plan)
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        if path == getattr(plan, protected):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    before = dict(runtime.environ)
    with pytest.raises(ValueError, match="must not use links"):
        require(plan)
    assert runtime.environ == before and module._attempt.status == "bound"


def test_canonical_overlap_drift_is_rechecked_without_repair(world, monkeypatch):
    plan, runtime = world
    start(plan)
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        return plan.root if path == plan.outputs else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    before = dict(runtime.environ)
    with pytest.raises(ValueError, match="separate directory"):
        require(plan)
    assert runtime.environ == before


def test_file_or_reparse_identity_is_not_adopted(world, monkeypatch):
    plan, runtime = world
    plan.root.write_text("user file", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        start(plan)
    assert plan.root.read_text(encoding="utf-8") == "user file" and not runtime.environ
    assert module._is_link(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT))


def test_malformed_plan_cannot_escape_namespace(world):
    plan, runtime = world
    with pytest.raises(ValueError, match="malformed"):
        start(replace(plan, namespace="../outputs"))
    assert not plan.root.exists() and not runtime.environ


def test_failed_mkdir_retains_receipt_and_refuses_retry(world, monkeypatch):
    plan, runtime = world
    real = Path.mkdir
    def mkdir(path, *args, **kwargs):
        if path == plan.inductor:
            raise OSError("injected mkdir failure")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(OSError, match="injected"):
        start(plan)
    receipt = module._attempt
    assert receipt.status == "failed" and receipt.target == plan.inductor
    assert receipt.operation == "mkdir" and plan.directory in receipt.created
    with pytest.raises(RuntimeError, match="did not complete"):
        module._initialize(plan)
    assert module._attempt is receipt and not runtime.environ


def test_write_failure_removes_only_known_probe_preserving_primary(world, monkeypatch):
    plan, runtime = world
    primary = OSError("injected write failure")
    def write(*args):
        raise primary
    monkeypatch.setattr(runtime, "write", write)
    with pytest.raises(OSError) as failure:
        start(plan)
    assert failure.value is primary
    assert module._attempt.descriptor is None and module._attempt.probe is None
    assert list(plan.inductor.iterdir()) == [] and not runtime.environ


def test_unknown_close_keeps_receipt_and_never_retries_or_unlinks(world, monkeypatch):
    plan, runtime = world
    calls = []
    def close(descriptor):
        calls.append(descriptor)
        os.close(descriptor)  # Outcome deliberately unknown to the owner.
        raise OSError("injected uncertain close")
    monkeypatch.setattr(runtime, "close", close)
    with pytest.raises(OSError, match="uncertain close"):
        start(plan)
    receipt = module._attempt
    assert receipt.operation == "close-probe" and receipt.target == plan.inductor
    assert receipt.descriptor == calls[0] and len(calls) == 1
    with pytest.raises(RuntimeError, match="did not complete"):
        module._initialize(plan)
    assert len(calls) == 1 and receipt.target == plan.inductor


def test_zero_probe_identity_refuses_and_closes_owned_handle_once(world, monkeypatch):
    plan, runtime = world
    calls = []
    close = runtime.close
    def closed(descriptor):
        calls.append(descriptor)
        close(descriptor)
    monkeypatch.setattr(runtime, "close", closed)
    monkeypatch.setattr(runtime, "fstat", lambda _: SimpleNamespace(st_dev=0, st_ino=0))
    with pytest.raises(RuntimeError, match="stable compiler cache file identity"):
        start(plan)
    assert len(calls) == 1 and module._attempt.descriptor is None
    assert module._attempt.probe is None and module._attempt.status == "failed"
    assert list(plan.inductor.iterdir()) == [] and not runtime.environ


def test_zero_directory_identity_refuses_before_environment_publication(world, monkeypatch):
    plan, runtime = world
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == plan.root:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0, st_dev=result.st_dev, st_ino=0)
        return result
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(RuntimeError, match="stable compiler cache file identity"):
        start(plan)
    assert not runtime.environ and module._attempt.binding is None
    assert module._attempt.status == "failed" and plan.root.is_dir()


def test_partial_environment_write_retains_values_receipt_and_failed_state(world, monkeypatch):
    plan, runtime = world
    class FailingEnvironment(dict):
        def __setitem__(self, key, value):
            if key == "TRITON_CACHE_DIR":
                raise OSError("injected environment failure")
            super().__setitem__(key, value)
    runtime.environ = FailingEnvironment()
    with pytest.raises(OSError, match="environment failure"):
        start(plan)
    assert runtime.environ == {"TORCHINDUCTOR_CACHE_DIR": str(plan.inductor)}
    assert module._attempt.assigned == ["TORCHINDUCTOR_CACHE_DIR"]
    assert module._attempt.operation == "set-environment:TRITON_CACHE_DIR"
    with pytest.raises(RuntimeError, match="fresh startup"):
        require(plan)


def test_matching_value_appearing_during_publication_is_not_adopted(world):
    plan, runtime = world
    class InterveningEnvironment(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "TORCHINDUCTOR_CACHE_DIR":
                super().__setitem__("TRITON_CACHE_DIR", str(plan.triton))
    runtime.environ = InterveningEnvironment()
    with pytest.raises(RuntimeError, match="appeared during startup"):
        start(plan)
    assert runtime.environ == {"TORCHINDUCTOR_CACHE_DIR": str(plan.inductor), "TRITON_CACHE_DIR": str(plan.triton)}
    assert module._attempt.assigned == ["TORCHINDUCTOR_CACHE_DIR"]
    assert module._attempt.operation == "check-environment:TRITON_CACHE_DIR"
    assert module._attempt.status == "failed"


def test_sealed_imported_host_cannot_run_a_factory(world):
    plan, _ = world
    module._seal_startup()
    with pytest.raises(RuntimeError, match="window is closed"):
        module._owned_startup(lambda: pytest.fail("late factory ran"))
    assert not plan.root.exists()


def test_internal_initializer_cannot_be_used_as_a_hot_bootstrap(world):
    plan, runtime = world
    with pytest.raises(RuntimeError, match="requires the owned startup"):
        module._initialize(plan)
    assert module._attempt is None and not plan.root.exists() and not runtime.environ


def test_factory_failure_seals_without_cache_io_and_preserves_primary(world):
    plan, runtime = world
    primary = RuntimeError("injected plan failure")
    def factory():
        raise primary
    with pytest.raises(RuntimeError) as failure:
        module._owned_startup(factory)
    assert failure.value is primary and module._sealed and module._startup_thread is None
    assert module._attempt is None and not plan.root.exists() and not runtime.environ


def test_probe_replacement_is_not_unlinked_or_adopted(world, monkeypatch):
    plan, runtime = world
    original = runtime.close
    def close(descriptor):
        original(descriptor)
        probe = module._attempt.probe
        assert probe is not None  # Windows delete-on-close probe, no pathname cleanup.
        probe.write_text("foreign replacement", encoding="utf-8")
    monkeypatch.setattr(runtime, "close", close)
    if runtime.name != "nt":
        pytest.skip("Windows handle-deletion replacement case; Linux probes are anonymous")
    assert start(plan) is not None
    replacements = list(plan.inductor.iterdir()) + list(plan.triton.iterdir())
    assert len(replacements) == 2
    assert all(path.read_text(encoding="utf-8") == "foreign replacement" for path in replacements)
    assert module._attempt.descriptor is None and module._attempt.probe is None


def test_actual_entrypoint_orders_bootstrap_before_project_workers_and_lifespan():
    source = Path("backend/main.py").read_text(encoding="utf-8")
    ast.parse(source)
    startup = source.index("compiler_cache._owned_startup()")
    sealed = source.index("compiler_cache._seal_startup()")
    imports = source.index("from . import api_backup")
    assert startup < imports and sealed < imports
    assert startup < source.index('backend = get_backend("diffusers")') < source.index("jobs = JobManager(")
    assert startup < source.index("await asyncio.to_thread(backup_service.prepare)") < source.index("backup_service.start_once()")
    assert startup < source.index("async def lifespan(") < source.index("    yield\n")
    assert startup < source.index("asyncio.create_task(_finish_startup(")


@pytest.mark.parametrize("owned", [True, False])
def test_actual_entrypoint_dispatches_owned_bootstrap_or_imported_seal(owned):
    tree = ast.parse(Path("backend/main.py").read_text(encoding="utf-8"))
    branch = next(node for node in tree.body if isinstance(node, ast.If)
                  and any(isinstance(call, ast.Attribute) and call.attr == "_owned_startup"
                          for call in ast.walk(node)))
    events = []
    runtime = SimpleNamespace(_owned_startup=lambda: events.append("owned"),
                              _seal_startup=lambda: events.append("sealed"))
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "main-bootstrap", "exec"),
         {"__name__": "__main__" if owned else "backend.main", "compiler_cache": runtime})
    assert events == ["owned" if owned else "sealed"]
