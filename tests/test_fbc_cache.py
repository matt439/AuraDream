"""FBC ownership stubs/source contracts, never real hooks or arithmetic."""
from __future__ import annotations

import ast
import dataclasses
import gc
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import distribution
from threading import Barrier
from types import SimpleNamespace

import pytest

from backend.backends import Cancelled
from backend.backends import fbc_cache as fbc


HEAD, BLOCK = "fbc_leader_block_hook", "fbc_block_hook"


@pytest.fixture(autouse=True)
def no_new_runtime_work(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("FBC ownership tests never compile, allocate, query a device or run forward")

    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr(torch.nn.Module, "compile", forbidden)
    for name in ("tensor", "empty", "zeros", "ones", "randn"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("is_available", "device_count", "get_device_capability", "current_device", "synchronize", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


class Module:
    def __init__(self, world):
        self.world = world
        self._compiled_call_impl = None

    def forward(self, *args, **kwargs):
        raise AssertionError("class-defined forward must never execute in ownership tests")

    def named_modules(self):
        yield "", self


class ModuleList(list, Module):
    def __init__(self, world, blocks):
        list.__init__(self, blocks)
        Module.__init__(self, world)

    def named_modules(self):
        yield "", self
        for index, block in enumerate(self):
            yield str(index), block


class FluxTransformerBlock(Module):
    pass


class FluxSingleTransformerBlock(Module):
    pass


class Manager:
    def __init__(self):
        self._current_context = None
        self._state_cache = {}
        self.fail_reset = False

    def set_context(self, name):
        self._current_context = name

    def get_state(self):
        if self._current_context is None:
            raise ValueError("no context")
        return self._state_cache.setdefault(self._current_context, {})

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset failed")
        self._state_cache.clear()
        self._current_context = None


class Hook:
    _is_stateful = False

    def __init__(self, manager=None):
        self.state_manager = manager


class HeadHook(Hook):
    _is_stateful = True

    def __init__(self, manager, threshold):
        super().__init__(manager)
        self.threshold = threshold


class BlockHook(Hook):
    def __init__(self, manager, is_tail=False):
        super().__init__(manager)
        self.is_tail = is_tail


class Registry:
    def __init__(self, module):
        self._module_ref = module
        self.hooks, self._hook_order, self._fn_refs = {}, [], []

    @classmethod
    def check_if_exists_or_initialize(cls, module):
        if not hasattr(module, "_diffusers_hook"):
            module._diffusers_hook = cls(module)
        return module._diffusers_hook

    def register_hook(self, hook, name):
        if name in self.hooks:
            raise ValueError("hook conflict")
        prior = self._module_ref.forward
        self._module_ref.forward = object()  # Opaque wrapper token, never invoked.
        if self._module_ref.world.break_bookkeeping:
            raise RuntimeError("registration failed after forward rewrite")
        self.hooks[name] = hook
        self._hook_order.append(name)
        self._fn_refs.append(SimpleNamespace(prior=prior))

    def get_hook(self, name):
        return self.hooks.get(name)

    def remove_hook(self, name, recurse=True):
        if name in self.hooks:
            index = self._hook_order.index(name)
            if self._module_ref.world.remove_failure:
                raise self._module_ref.world.remove_failure
            if index == len(self._hook_order) - 1:
                self._module_ref.forward = self._fn_refs[index].prior
            else:
                self._fn_refs[index + 1].prior = self._fn_refs[index].prior
            self._module_ref.world.removed.append(self.hooks.pop(name))
            self._hook_order.pop(index)
            self._fn_refs.pop(index)
        if recurse:
            for path, module in self._module_ref.named_modules():
                if path and hasattr(module, "_diffusers_hook"):
                    module._diffusers_hook.remove_hook(name, recurse=False)

    def _get_child_registries(self):
        if not hasattr(self, "_child_registries_cache"):
            self._child_registries_cache = None
        if self._child_registries_cache is None:
            self._child_registries_cache = [module._diffusers_hook for path, module in self._module_ref.named_modules()
                                            if path and hasattr(module, "_diffusers_hook")]
        return self._child_registries_cache

    def _set_context(self, name=None):
        for hook in self.hooks.values():
            if hook._is_stateful:
                hook.state_manager.set_context(name)
        for registry in self._get_child_registries():
            registry._set_context(name)


@dataclasses.dataclass
class Config:
    threshold: float = 0.05


class FluxTransformer2DModel(Module):
    _cache_config = None

    def __init__(self, world):
        super().__init__(world)
        self.transformer_blocks = ModuleList(world, [FluxTransformerBlock(world), FluxTransformerBlock(world)])
        self.single_transformer_blocks = ModuleList(world, [FluxSingleTransformerBlock(world)])
        self.config = SimpleNamespace(num_layers=2, num_single_layers=1)
        self.extra = None

    def named_children(self):
        yield "transformer_blocks", self.transformer_blocks
        yield "single_transformer_blocks", self.single_transformer_blocks
        if self.extra is not None:
            yield "layers", self.extra

    def named_modules(self):
        yield "", self
        for name, child in self.named_children():
            for path, module in child.named_modules():
                yield f"{name}.{path}".rstrip("."), module

    @property
    def is_cache_enabled(self):
        return self._cache_config is not None

    def enable_cache(self, config):
        self.world.configs.append(config)
        manager = Manager()
        self.world.managers.append(manager)
        blocks = [*self.transformer_blocks, *self.single_transformer_blocks]
        for index, block in enumerate(blocks):
            if self.world.fail_before == index:
                raise self.world.setup_failure
            hook = HeadHook(manager, config.threshold) if index == 0 else BlockHook(manager, index == len(blocks) - 1)
            if index == 0 and self.world.unknown_head:
                hook = Hook(manager)
            Registry.check_if_exists_or_initialize(block).register_hook(hook, HEAD if index == 0 else BLOCK)
            if self.world.fail_after == index:
                raise self.world.setup_failure
        self._cache_config = config  # Deliberately after sequential installation.

    def disable_cache(self):
        self.world.disable_calls += 1
        if self.world.disable_failure:
            raise self.world.disable_failure
        if self._cache_config is not None:
            registry = Registry.check_if_exists_or_initialize(self)
            registry.remove_hook(HEAD, recurse=True)
            registry.remove_hook(BLOCK, recurse=True)
            self._cache_config = None

    @contextmanager
    def cache_context(self, name):
        registry = Registry.check_if_exists_or_initialize(self)
        registry._set_context(name)
        yield  # Match the installed generator's missing exceptional-exit reset.
        registry._set_context(None)


@pytest.fixture
def world(monkeypatch):
    state = SimpleNamespace(configs=[], managers=[], removed=[], fail_before=None, fail_after=None,
                            setup_failure=RuntimeError("partial installation"), break_bookkeeping=False,
                            remove_failure=None, disable_failure=None, disable_calls=0, bad_metadata=False,
                            unknown_head=False)

    def metadata(cls):
        return SimpleNamespace(return_hidden_states_index=0 if state.bad_metadata else 1,
                               return_encoder_hidden_states_index=0, hidden_states_argument_name="hidden_states", _cls=cls)

    monkeypatch.setattr(fbc, "_api", lambda: fbc._Api(
        FluxTransformer2DModel, (FluxTransformerBlock, FluxSingleTransformerBlock), ModuleList, Registry,
        Manager, HeadHook, BlockHook, Config, metadata,
        ("transformer_blocks", "single_transformer_blocks", "layers", "blocks"), (HEAD, BLOCK),
    ))
    return state


def make_owner(world, discarded=None):
    model = FluxTransformer2DModel(world)
    discarded = [] if discarded is None else discarded
    owner = fbc.FirstBlockCacheOwner(model, discard_shared=lambda: discarded.append(True))
    return model, owner, discarded


def assert_clean(model, owner):
    owner.assert_off()
    assert model._cache_config is None and not owner.active and not owner.invalidated
    for _, module in model.named_modules():
        assert not hasattr(module, "_diffusers_hook")
        assert "forward" not in vars(module)  # Restore class-defined method ownership.


def test_import_stays_cold_without_model_hook_or_compiler_libraries():
    done = subprocess.run([sys.executable, "-c", (
        "import sys; import backend.backends.fbc_cache; "
        "assert not {'torch', 'diffusers', 'triton'} & sys.modules.keys()"
    )], capture_output=True, text=True, timeout=15)
    assert done.returncode == 0, done.stdout + done.stderr


def test_layout_is_exact_ordered_read_only_and_frozen(world):
    model = FluxTransformer2DModel(world)
    facts = fbc.inspect_fbc_layout(model)
    assert facts.recipe == "fbc-005-v1" and facts.threshold == 0.05
    assert facts.block_paths == ("transformer_blocks.0", "transformer_blocks.1", "single_transformer_blocks.0")
    assert world.configs == [] and world.managers == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.threshold = 0.2


@pytest.mark.parametrize("problem", ["root-type", "root-compiled", "block-type", "block-compiled", "count",
                                    "empty", "alias", "extra-list", "metadata", "missing-api"])
def test_invalid_layout_refuses_before_mutation(world, problem):
    model = FluxTransformer2DModel(world)
    if problem == "root-type":
        model = type("FluxTransformer2DModel", (), {})()
    elif problem == "root-compiled":
        model._compiled_call_impl = object()
    elif problem == "block-type":
        model.transformer_blocks[0] = type("FluxTransformerBlock", (Module,), {})(world)
    elif problem == "block-compiled":
        model.single_transformer_blocks[0]._compiled_call_impl = object()
    elif problem == "count":
        model.config.num_layers = 1
    elif problem == "empty":
        model.single_transformer_blocks.clear()
        model.config.num_single_layers = 0
    elif problem == "alias":
        model.transformer_blocks[1] = model.transformer_blocks[0]
    elif problem == "extra-list":
        model.extra = ModuleList(world, [])
    elif problem == "metadata":
        world.bad_metadata = True
    else:
        model.cache_context = None
    with pytest.raises(ValueError):
        fbc.FirstBlockCacheOwner(model, discard_shared=lambda: pytest.fail("nothing was mutated"))
    assert world.configs == [] and world.managers == []


@pytest.mark.parametrize("problem", ["config", "conflict", "stateful", "registry-type"])
def test_foreign_cache_or_hook_conflicts_are_untouched(world, problem):
    model = FluxTransformer2DModel(world)
    foreign = Hook()
    if problem == "config":
        model._cache_config = object()
    elif problem == "registry-type":
        model._diffusers_hook = object()
    else:
        if problem == "stateful":
            foreign._is_stateful = True
        Registry.check_if_exists_or_initialize(model).register_hook(foreign, HEAD if problem == "conflict" else "foreign")
    with pytest.raises(ValueError):
        fbc.FirstBlockCacheOwner(model, discard_shared=lambda: pytest.fail("foreign work is preserved"))
    assert world.configs == [] and world.removed == []


def test_fixed_config_and_fresh_cond_uncond_state_for_consecutive_calls(world):
    model, owner, discarded = make_owner(world)
    managers = []
    for _ in range(3):
        with owner.cached_call() as facts:
            assert facts.threshold == 0.05 and world.configs[-1].threshold == 0.05
            manager = world.managers[-1]
            managers.append(manager)
            assert manager._state_cache == {} and manager._current_context is None
            with model.cache_context("cond"):
                manager.get_state()["value"] = object()
            with model.cache_context("uncond"):
                assert manager.get_state() == {}
            assert set(manager._state_cache) == {"cond", "uncond"}
        assert manager._state_cache == {} and manager._current_context is None
        assert_clean(model, owner)
    assert len({id(manager) for manager in managers}) == len({id(config) for config in world.configs}) == 3
    assert discarded == []


def test_eager_cached_child_list_is_invalidated_and_restored_by_identity(world):
    model, owner, discarded = make_owner(world)
    with model.cache_context("cond"):
        pass
    registry = model._diffusers_hook
    old_cache = registry._child_registries_cache
    assert old_cache == []
    for _ in range(2):
        with owner.cached_call():
            with model.cache_context("cond"):
                assert world.managers[-1]._current_context == "cond"
                world.managers[-1].get_state()["fresh"] = True
        assert registry._child_registries_cache is old_cache
        owner.assert_off()
    assert discarded == []


def test_unrelated_foreign_hook_and_instance_forward_are_preserved(world):
    model, owner, discarded = make_owner(world)
    block = model.transformer_blocks[0]
    registry = Registry.check_if_exists_or_initialize(block)
    foreign = Hook()
    registry.register_hook(foreign, "foreign")
    forward, refs = block.forward, tuple(registry._fn_refs)
    registry._child_registries_cache = []
    cache = registry._child_registries_cache
    with owner.cached_call():
        pass
    assert block.forward is forward and registry.hooks == {"foreign": foreign}
    assert tuple(registry._fn_refs) == refs and registry._child_registries_cache is cache
    assert foreign not in world.removed and discarded == []
    owner.assert_off()


def test_a_preexisting_instance_bound_method_keeps_its_attribute_ownership(world):
    model, owner, _ = make_owner(world)
    block = model.transformer_blocks[0]
    original = block.forward
    block.forward = original
    with owner.cached_call():
        pass
    assert vars(block)["forward"] is original
    owner.assert_off()


@pytest.mark.parametrize("bad_cache", [object(), [object()]])
def test_unknown_child_cache_shapes_refuse_before_mutation(world, bad_cache):
    model = FluxTransformer2DModel(world)
    registry = Registry.check_if_exists_or_initialize(model)
    registry._child_registries_cache = bad_cache
    with pytest.raises(ValueError, match="child cache"):
        fbc.FirstBlockCacheOwner(model, discard_shared=lambda: pytest.fail("no mutation"))
    assert registry._child_registries_cache is bad_cache and world.configs == []


@pytest.mark.parametrize("failure", [RuntimeError("call failed"), Cancelled(), KeyboardInterrupt()])
def test_exceptional_pipeline_context_is_reset_and_original_error_preserved(world, failure):
    model, owner, discarded = make_owner(world)
    with pytest.raises(type(failure)) as raised:
        with owner.cached_call():
            with model.cache_context("cond"):
                world.managers[-1].get_state()["residual"] = object()
                raise failure
    assert raised.value is failure
    assert world.managers[-1]._current_context is None and world.managers[-1]._state_cache == {}
    assert_clean(model, owner)
    assert discarded == []


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("when", ["before", "after"])
def test_partial_installation_unwinds_even_before_cache_config_assignment(world, position, when):
    model, owner, discarded = make_owner(world)
    setattr(world, "fail_" + when, position)
    with pytest.raises(RuntimeError) as raised:
        with owner.cached_call():
            pytest.fail("partial install was published")
    assert raised.value is world.setup_failure and model._cache_config is None
    assert_clean(model, owner)
    assert discarded == [] and world.disable_calls == 0


def test_untracked_forward_rewrite_before_hook_bookkeeping_discards_shared_owner(world):
    model, owner, discarded = make_owner(world)
    world.break_bookkeeping = True
    with pytest.raises(RuntimeError, match="after forward rewrite") as raised:
        with owner.cached_call():
            pytest.fail("failed install was published")
    assert owner.invalidated and discarded == [True]
    assert any("forward" in note for note in raised.value.__notes__)
    with pytest.raises(fbc.CacheCleanupFailed):
        owner.assert_off()


def test_partial_discovery_problems_do_not_replace_the_original_setup_error(world):
    _, owner, discarded = make_owner(world)
    world.unknown_head, world.fail_after = True, 0
    with pytest.raises(RuntimeError) as raised:
        with owner.cached_call():
            pytest.fail("unknown partial install was published")
    assert raised.value is world.setup_failure
    assert owner.invalidated and discarded == [True]
    assert any("unexpected hook" in note for note in raised.value.__notes__)


def test_replaced_state_manager_is_not_reset_as_though_it_were_owned(world):
    model, owner, discarded = make_owner(world)
    foreign = Manager()
    foreign.set_context("foreign")
    foreign.get_state()["keep"] = True
    with pytest.raises(fbc.CacheCleanupFailed, match="state manager was replaced"):
        with owner.cached_call():
            original = world.managers[-1]
            model.transformer_blocks[0]._diffusers_hook.get_hook(HEAD).state_manager = foreign
    assert original._state_cache == {} and original._current_context is None
    assert foreign._current_context == "foreign" and foreign._state_cache == {"foreign": {"keep": True}}
    assert owner.invalidated and discarded == [True]


@pytest.mark.parametrize("problem", ["reset", "remove", "disable", "forward", "hook", "new-module"])
def test_cleanup_uncertainty_preserves_foreign_work_and_discards_once(world, problem):
    model, owner, discarded = make_owner(world)
    foreign = None
    with pytest.raises(fbc.CacheCleanupFailed):
        with owner.cached_call():
            block = model.transformer_blocks[0]
            if problem == "reset":
                world.managers[-1].fail_reset = True
            elif problem == "remove":
                world.remove_failure = RuntimeError("remove failed")
            elif problem == "disable":
                world.disable_failure = RuntimeError("disable failed")
            elif problem == "forward":
                foreign = block.forward = object()
            elif problem == "hook":
                registry = block._diffusers_hook
                registry.remove_hook(HEAD, recurse=False)
                foreign = Hook()
                registry.register_hook(foreign, HEAD)
            else:
                model.extra = ModuleList(world, [Module(world)])
                foreign = Hook()
                Registry.check_if_exists_or_initialize(model.extra[0]).register_hook(foreign, HEAD)
    assert owner.invalidated and not owner.active and discarded == [True] and owner.diagnostics
    if problem == "forward":
        assert model.transformer_blocks[0].forward is foreign
    if problem in {"hook", "new-module"}:
        assert foreign not in world.removed
    with pytest.raises(fbc.CacheCleanupFailed):
        with owner.cached_call():
            pytest.fail("unsafe owner was reused")
    assert discarded == [True]


def test_cleanup_and_discard_failure_keep_original_cancellation_and_diagnostics(world):
    model = FluxTransformer2DModel(world)

    def discard():
        raise OSError("discard failed")

    owner = fbc.FirstBlockCacheOwner(model, discard_shared=discard)
    cancelled = Cancelled()
    with pytest.raises(Cancelled) as raised:
        with owner.cached_call():
            world.disable_failure = RuntimeError("disable failed")
            raise cancelled
    assert raised.value is cancelled and owner.invalidated
    assert any("discard failed" in note for note in cancelled.__notes__)
    assert any("disable failed" in diagnostic for diagnostic in owner.diagnostics)


def test_shared_sibling_uses_one_owner_before_and_during_cached_calls(world):
    model, owner, discarded = make_owner(world)
    sibling = SimpleNamespace(transformer=model)
    with pytest.raises(ValueError, match="already has an FBC owner"):
        fbc.FirstBlockCacheOwner(sibling.transformer, discard_shared=lambda: pytest.fail("foreign lease"))
    with owner.cached_call():
        with pytest.raises(RuntimeError, match="active"):
            owner.assert_off()
        with pytest.raises(RuntimeError, match="active"):
            owner.close()
        with pytest.raises(ValueError, match="already has an FBC owner"):
            fbc.FirstBlockCacheOwner(sibling.transformer, discard_shared=lambda: pytest.fail("foreign lease"))
    assert_clean(model, owner)
    assert discarded == []


def test_clean_looking_failed_cleanup_stays_invalid_after_close_and_fresh_model_recovers(world):
    model, owner, discarded = make_owner(world)
    with pytest.raises(fbc.CacheCleanupFailed, match="reset failed"):
        with owner.cached_call():
            world.managers[-1].fail_reset = True
    # This failure removed every wrapper/config; structural cleanliness must
    # not erase the decision to discard the exact model shared by siblings.
    assert model._cache_config is None and owner.invalidated and discarded == [True]
    assert all(not hasattr(module, "_diffusers_hook") and "forward" not in vars(module)
               for _, module in model.named_modules())
    with pytest.raises(fbc.CacheCleanupFailed, match="discarded"):
        fbc.FirstBlockCacheOwner(model, discard_shared=lambda: pytest.fail("cannot reclaim discarded model"))
    owner.close()
    assert owner.closed
    with pytest.raises(fbc.CacheCleanupFailed, match="discarded"):
        fbc.FirstBlockCacheOwner(model, discard_shared=lambda: pytest.fail("close cannot unpoison the model"))
    assert fbc._CLAIMS[id(model)].invalidated and fbc._CLAIMS[id(model)].owner is None
    fresh, replacement, fresh_discarded = make_owner(world)
    assert fresh is not model
    with replacement.cached_call():
        pass
    assert_clean(fresh, replacement)
    replacement.close()
    assert fresh_discarded == []


def test_closed_valid_owner_releases_its_claim_while_model_is_alive(world):
    model, owner, _ = make_owner(world)
    owner.close()
    assert id(model) not in fbc._CLAIMS
    replacement = fbc.FirstBlockCacheOwner(model, discard_shared=lambda: None)
    replacement.assert_off()
    replacement.close()


def test_dead_valid_owner_is_not_kept_alive_by_identity_bookkeeping(world):
    model, owner, _ = make_owner(world)
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None and id(model) not in fbc._CLAIMS
    replacement = fbc.FirstBlockCacheOwner(model, discard_shared=lambda: None)
    replacement.close()


def test_invalidation_tombstone_does_not_keep_a_dead_model_alive(world):
    def discarded_model():
        model, owner, _ = make_owner(world)
        with pytest.raises(fbc.CacheCleanupFailed):
            with owner.cached_call():
                world.managers[-1].fail_reset = True
        reference, key = weakref.ref(model), id(model)
        owner.close()
        assert key in fbc._CLAIMS
        return reference, key

    reference, key = discarded_model()
    gc.collect()
    assert reference() is None and key not in fbc._CLAIMS


def test_claims_use_identity_without_hash_or_equality(world, monkeypatch):
    def forbidden_equality(self, other):
        raise AssertionError("ownership must use identity")

    monkeypatch.setattr(FluxTransformer2DModel, "__hash__", None)
    monkeypatch.setattr(FluxTransformer2DModel, "__eq__", forbidden_equality)
    model, owner, _ = make_owner(world)
    with pytest.raises(ValueError, match="already has an FBC owner"):
        fbc.FirstBlockCacheOwner(model, discard_shared=lambda: None)
    owner.close()


def test_duplicate_constructor_claim_is_atomic_across_threads(world):
    model = FluxTransformer2DModel(world)
    ready = Barrier(2)

    def claim():
        ready.wait(timeout=5)
        try:
            return fbc.FirstBlockCacheOwner(model, discard_shared=lambda: None)
        except ValueError as exc:
            assert "already has an FBC owner" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim), pool.submit(claim)]
        owners = [result for future in futures if (result := future.result(timeout=5)) is not None]
    assert len(owners) == 1
    owners[0].close()


def test_layout_replacement_after_owner_creation_refuses_without_install(world):
    model, owner, discarded = make_owner(world)
    model.transformer_blocks[0] = FluxTransformerBlock(world)
    with pytest.raises(ValueError, match="layout changed"):
        with owner.cached_call():
            pytest.fail("changed layout was used")
    assert world.configs == [] and discarded == []


def test_close_is_idempotent_and_drops_owned_model_callback_references(world):
    model = FluxTransformer2DModel(world)
    reference = weakref.ref(model)
    owner = fbc.FirstBlockCacheOwner(model, discard_shared=lambda: None)
    del model
    gc.collect()
    assert reference() is not None
    owner.close()
    owner.close()
    gc.collect()
    assert reference() is None and owner.closed
    with pytest.raises(RuntimeError, match="closed"):
        owner.assert_off()


def source(path):
    return ast.parse(distribution("diffusers").locate_file("diffusers/" + path).read_text(encoding="utf-8"))


def definition(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


def test_installed_source_contract_fixed_config_names_and_install_order():
    tree = source("hooks/first_block_cache.py")
    config = definition(tree, "FirstBlockCacheConfig")
    threshold = next(node for node in config.body if isinstance(node, ast.AnnAssign) and node.target.id == "threshold")
    assert ast.literal_eval(threshold.value) == 0.05
    strings = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id in {"_FBC_LEADER_BLOCK_HOOK", "_FBC_BLOCK_HOOK"}}
    assert strings == {"_FBC_LEADER_BLOCK_HOOK": HEAD, "_FBC_BLOCK_HOOK": BLOCK}
    installer = ast.unparse(definition(tree, "apply_first_block_cache"))
    assert "module.named_children()" in installer and "torch.nn.ModuleList" in installer
    assert "remaining_blocks.pop(0)" in installer and "remaining_blocks.pop(-1)" in installer
    mixin = definition(source("models/cache_utils.py"), "enable_cache")
    assignments = [node for node in ast.walk(mixin) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr == "_cache_config" for target in node.targets)]
    apply = next(node for node in ast.walk(mixin) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "apply_first_block_cache")
    assert assignments[0].lineno > apply.lineno


def test_installed_source_contract_registry_cache_and_flux_output_metadata():
    hooks = source("hooks/hooks.py")
    assert "_child_registries_cache" in ast.unparse(definition(hooks, "_get_child_registries"))
    remove = ast.unparse(definition(hooks, "remove_hook"))
    assert "recurse" in remove and "_fn_refs" in remove and "old_forward" in remove
    manager = ast.unparse(definition(definition(hooks, "StateManager"), "reset"))
    assert "self._state_cache.pop(name)" in manager and "self._current_context = None" in manager
    registry = source("hooks/_helpers.py")
    found = {}
    for node in ast.walk(registry):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "register":
            continue
        kwargs = {item.arg: item.value for item in node.keywords}
        cls = kwargs.get("model_class")
        if isinstance(cls, ast.Name) and cls.id in {"FluxTransformerBlock", "FluxSingleTransformerBlock"}:
            found[cls.id] = {item.arg: ast.literal_eval(item.value) for item in kwargs["metadata"].keywords}
    assert found == {name: {"return_hidden_states_index": 1, "return_encoder_hidden_states_index": 0}
                     for name in ("FluxTransformerBlock", "FluxSingleTransformerBlock")}
