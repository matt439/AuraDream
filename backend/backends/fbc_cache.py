"""Private per-denoiser First Block Cache ownership; public admission stays closed.

All library imports are lazy. Real model/device behavior remains unqualified;
this module never changes runtime admission.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import TYPE_CHECKING, Literal
from weakref import ReferenceType, ref

if TYPE_CHECKING:
    from torch.nn import Module, ModuleList
    from diffusers.hooks._helpers import TransformerBlockMetadata
    from diffusers.hooks.first_block_cache import FBCBlockHook, FBCHeadBlockHook, FirstBlockCacheConfig
    from diffusers.hooks.hooks import HookRegistry, ModelHook, StateManager
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock, FluxTransformer2DModel, FluxTransformerBlock,
    )


class _Absent:
    pass


_MISSING = _Absent()
_LIST_NAMES = ("transformer_blocks", "single_transformer_blocks")


@dataclass(frozen=True)
class _Api:
    model_type: type[FluxTransformer2DModel]
    block_types: tuple[type[FluxTransformerBlock], type[FluxSingleTransformerBlock]]
    list_type: type[ModuleList]
    registry_type: type[HookRegistry]
    manager_type: type[StateManager]
    head_type: type[FBCHeadBlockHook]
    tail_type: type[FBCBlockHook]
    config_type: type[FirstBlockCacheConfig]
    metadata: Callable[[type[Module]], TransformerBlockMetadata]
    identifiers: tuple[str, ...]
    hook_names: tuple[str, str]


def _api() -> _Api:
    # These imports can probe optional devices/execute compiler decorators.
    # Tests use strict stubs; this adapter is for later runtime qualification.
    import_module("..config", __package__)
    from torch.nn import ModuleList
    from diffusers.hooks._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
    from diffusers.hooks._helpers import TransformerBlockRegistry
    from diffusers.hooks.first_block_cache import (
        _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK, FBCBlockHook, FBCHeadBlockHook, FirstBlockCacheConfig,
    )
    from diffusers.hooks.hooks import HookRegistry, StateManager
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock, FluxTransformer2DModel, FluxTransformerBlock,
    )

    return _Api(
        FluxTransformer2DModel, (FluxTransformerBlock, FluxSingleTransformerBlock), ModuleList,
        HookRegistry, StateManager, FBCHeadBlockHook, FBCBlockHook, FirstBlockCacheConfig,
        TransformerBlockRegistry.get, _ALL_TRANSFORMER_BLOCK_IDENTIFIERS,
        (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK),
    )


@dataclass(frozen=True)
class FbcLayout:
    """Validated layout facts, not cache hits, arithmetic or quality evidence."""

    recipe: Literal["fbc-005-v1"]
    threshold: float
    block_paths: tuple[str, ...]


def _inspect(value: object, api: _Api) -> tuple[FluxTransformer2DModel, tuple[Module, ...], FbcLayout]:
    if not isinstance(value, api.model_type) or type(value) is not api.model_type:
        raise ValueError("fbc-005-v1 needs the installed exact FluxTransformer2DModel class")
    if getattr(value, "_compiled_call_impl", None) is not None:
        raise ValueError("FBC needs an eager transformer root")
    for method in ("enable_cache", "disable_cache", "cache_context"):
        if not callable(getattr(value, method, None)):
            raise ValueError(f"the FLUX transformer has no {method} API")
    lists = [(name, child) for name, child in value.named_children()
             if name in api.identifiers and isinstance(child, api.list_type)]
    if tuple(name for name, _ in lists) != _LIST_NAMES:
        raise ValueError("FBC needs exactly the joint then single FLUX block lists in installer order")
    blocks: list[Module] = []
    paths: list[str] = []
    for (name, children), expected_type, count_name in zip(
        lists, api.block_types, ("num_layers", "num_single_layers"), strict=True,
    ):
        count = getattr(value.config, count_name, None)
        if type(count) is not int or count < 1 or len(children) != count:
            raise ValueError(f"FBC block-list length disagrees with {count_name}")
        for index, block in enumerate(children):
            if type(block) is not expected_type or getattr(block, "_compiled_call_impl", None) is not None:
                raise ValueError("FBC needs the exact eager FLUX block classes")
            metadata = api.metadata(type(block))
            if (metadata.return_hidden_states_index != 1 or metadata.return_encoder_hidden_states_index != 0
                    or metadata.hidden_states_argument_name != "hidden_states" or metadata._cls is not type(block)):
                raise ValueError("FBC FLUX block output metadata differs from the qualified layout")
            if any(block is previous for previous in blocks):
                raise ValueError("FBC cannot own an aliased FLUX block twice")
            blocks.append(block)
            paths.append(f"{name}.{index}")
    return value, tuple(blocks), FbcLayout("fbc-005-v1", 0.05, tuple(paths))


def inspect_fbc_layout(transformer: object) -> FbcLayout:
    """Read the first supported layout without installing hooks or contexts."""
    return _inspect(transformer, _api())[2]


def require_native_attention(transformer: object) -> None:
    """Use the common read-only guard; retain the private cache call seam."""
    from . import attention_selection

    attention_selection.require_native_attention(transformer)


def _registry(module: Module, api: _Api) -> HookRegistry | None:
    found = getattr(module, "_diffusers_hook", _MISSING)
    if found is _MISSING:
        return None
    if not isinstance(found, api.registry_type) or type(found) is not api.registry_type:
        raise ValueError("an unknown registry owns this FLUX module")
    return found


def _same_forward(first: object, second: object) -> bool:
    return first is second or (
        getattr(first, "__self__", _MISSING) is getattr(second, "__self__", None)
        and getattr(first, "__func__", _MISSING) is getattr(second, "__func__", None)
    )


@dataclass
class _Snapshot:
    path: str
    module: Module
    registry: HookRegistry | None
    hooks: dict[str, ModelHook]
    order: tuple[str, ...]
    refs: tuple[object, ...]
    child_cache: list[HookRegistry] | None | _Absent
    forward: object
    forward_owned: bool
    is_block: bool


@dataclass
class _OwnedHook:
    module: Module
    registry: HookRegistry
    name: str
    hook: FBCHeadBlockHook | FBCBlockHook
    manager: StateManager
    installed_forward: object


class CacheCleanupFailed(RuntimeError):
    """Cache ownership could not be restored; shared pipelines must be discarded."""


@dataclass
class _IdentityClaim:
    model: ReferenceType[object]
    owner: ReferenceType[FirstBlockCacheOwner] | None
    invalidated: bool = False


# Integer keys avoid invoking a module's hash/equality. Both references are
# weak; an invalidation tombstone lives only as long as that exact model.
_CLAIMS: dict[int, _IdentityClaim] = {}
_CLAIM_LOCK = RLock()


def _model_reference(model: object, key: int) -> ReferenceType[object]:
    def gone(reference: ReferenceType[object]) -> None:
        with _CLAIM_LOCK:
            claim = _CLAIMS.get(key)
            if claim is not None and claim.model is reference:
                _CLAIMS.pop(key)
    return ref(model, gone)


def _claim_identity(owner: FirstBlockCacheOwner, model: object) -> None:
    key = id(model)
    with _CLAIM_LOCK:
        claim = _CLAIMS.get(key)
        if claim is not None and claim.model() is model:
            if claim.invalidated:
                raise CacheCleanupFailed("this denoiser was discarded; load a fresh model identity")
            if claim.owner is not None and claim.owner() is not None:
                raise ValueError("this shared denoiser already has an FBC owner")
        model_ref = _model_reference(model, key)

        def owner_gone(reference: ReferenceType[FirstBlockCacheOwner]) -> None:
            with _CLAIM_LOCK:
                current = _CLAIMS.get(key)
                if current is not None and current.owner is reference:
                    if current.invalidated:
                        current.owner = None
                    else:
                        _CLAIMS.pop(key)

        _CLAIMS[key] = _IdentityClaim(model_ref, ref(owner, owner_gone))


def _release_identity(owner: FirstBlockCacheOwner, model: object) -> None:
    with _CLAIM_LOCK:
        claim = _CLAIMS.get(id(model))
        if claim is not None and claim.model() is model and claim.owner is not None and claim.owner() is owner:
            if claim.invalidated:
                claim.owner = None
            else:
                _CLAIMS.pop(id(model))


def _verify_identity(owner: FirstBlockCacheOwner, model: object) -> None:
    with _CLAIM_LOCK:
        claim = _CLAIMS.get(id(model))
        if (claim is None or claim.model() is not model or claim.invalidated
                or claim.owner is None or claim.owner() is not owner):
            raise CacheCleanupFailed("this FBC owner no longer owns a safe denoiser identity")


def _invalidate_identity(model: object) -> None:
    with _CLAIM_LOCK:
        claim = _CLAIMS.get(id(model))
        if claim is not None and claim.model() is model:
            claim.invalidated = True
        else:
            _CLAIMS[id(model)] = _IdentityClaim(_model_reference(model, id(model)), None, True)


class FirstBlockCacheOwner:
    """One owner for a loaded denoiser shared by base and sibling pipelines.

    The private backend must retain this owner, use it under its existing worker
    gate, and discard all shared pipeline references when discard_shared runs.
    """

    def __init__(self, transformer: object, *, discard_shared: Callable[[], None]) -> None:
        api = _api()
        model, blocks, layout = _inspect(transformer, api)
        self._api: _Api | None = api
        self._model: FluxTransformer2DModel | None = model
        self._blocks: tuple[Module, ...] = blocks
        self._layout: FbcLayout | None = layout
        self._discard: Callable[[], None] | None = discard_shared
        self._active = False
        self._invalidated = False
        self._diagnostics: list[str] = []
        _claim_identity(self, model)
        try:
            self.assert_off()
        except BaseException:
            _release_identity(self, model)
            raise

    @property
    def active(self) -> bool:
        return self._active

    @property
    def closed(self) -> bool:
        return self._api is None

    @property
    def invalidated(self) -> bool:
        return self._invalidated

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def _live(self) -> tuple[FluxTransformer2DModel, _Api]:
        if self._api is None or self._model is None:
            raise RuntimeError("the FBC owner is closed")
        if self._invalidated:
            raise CacheCleanupFailed("the FBC owner is unsafe; shared pipelines must be discarded")
        _verify_identity(self, self._model)
        return self._model, self._api

    def _snapshot(self, model: FluxTransformer2DModel, api: _Api) -> list[_Snapshot]:
        if model.is_cache_enabled or getattr(model, "_cache_config", None) is not None:
            raise ValueError("a foreign cache is already enabled on the shared denoiser")
        result = []
        for path, module in model.named_modules():
            registry = _registry(module, api)
            hooks = {} if registry is None else dict(registry.hooks)
            if any(name in hooks for name in api.hook_names):
                raise ValueError("a conflicting FBC hook already exists on the shared denoiser")
            if any(getattr(hook, "_is_stateful", False) for hook in hooks.values()):
                raise ValueError("a foreign stateful hook cannot be combined with FBC")
            cache = _MISSING if registry is None else getattr(registry, "_child_registries_cache", _MISSING)
            if cache is not None and not isinstance(cache, (_Absent, list)):
                raise ValueError("the registry's child cache has an unknown shape")
            if isinstance(cache, list) and any(type(child) is not api.registry_type for child in cache):
                raise ValueError("the registry's child cache contains an unknown registry")
            result.append(_Snapshot(
                path, module, registry, hooks, () if registry is None else tuple(registry._hook_order),
                () if registry is None else tuple(registry._fn_refs),
                cache,
                module.forward, "forward" in vars(module), any(module is block for block in self._blocks),
            ))
        return result

    def assert_off(self) -> None:
        model, api = self._live()
        if self._active:
            raise RuntimeError("the shared denoiser has an active FBC call")
        _, blocks, layout = _inspect(model, api)
        if layout != self._layout or any(a is not b for a, b in zip(blocks, self._blocks, strict=True)):
            raise ValueError("the shared FLUX block layout changed")
        self._snapshot(model, api)

    def require(self, transformer: object) -> FbcLayout:
        """Require this exact safe root with no installed call hooks/state."""
        model, _ = self._live()
        if model is not transformer:
            raise ValueError("the FBC owner does not own this live transformer")
        self.assert_off()
        assert self._layout is not None
        return self._layout

    def discard(self) -> None:
        """Abandon an idle owner without touching uncertain/foreign hooks."""
        if self.closed:
            return
        if self._active:
            raise RuntimeError("cannot discard an active FBC call")
        self._invalidated = True
        if self._model is not None:
            _invalidate_identity(self._model)
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self._active:
            raise RuntimeError("cannot close an active FBC call")
        if not self._invalidated:
            self.assert_off()
        if self._model is not None:
            _release_identity(self, self._model)
        self._model = None
        self._blocks = ()
        self._layout = None
        self._api = None
        self._discard = None

    def _discover(self, snapshots: list[_Snapshot], api: _Api) -> tuple[list[_OwnedHook], list[str]]:
        found = []
        errors = []
        for snapshot in snapshots:
            try:
                registry = _registry(snapshot.module, api)
            except BaseException as exc:
                errors.append(f"inspect installed registry: {type(exc).__name__}: {exc}")
                continue
            if registry is None:
                continue
            for name in api.hook_names:
                hook = registry.get_hook(name)
                if hook is None:
                    continue
                expected = api.head_type if name == api.hook_names[0] else api.tail_type
                if type(hook) is not expected or not isinstance(hook, (api.head_type, api.tail_type)):
                    errors.append("an unexpected hook appeared during FBC installation")
                    continue
                manager = hook.state_manager
                if not isinstance(manager, api.manager_type) or type(manager) is not api.manager_type:
                    errors.append("an unexpected state manager appeared during FBC installation")
                    continue
                found.append(_OwnedHook(snapshot.module, registry, name, hook, manager, snapshot.module.forward))
        return found, errors

    def _check_installed(self, owned: list[_OwnedHook], api: _Api, config: FirstBlockCacheConfig) -> None:
        assert self._model is not None
        if self._model._cache_config is not config or len(owned) != len(self._blocks):
            raise CacheCleanupFailed("FBC installation did not produce the expected owned config/hooks")
        manager = owned[0].hook.state_manager
        if not isinstance(manager, api.manager_type) or type(manager) is not api.manager_type:
            raise CacheCleanupFailed("FBC installed an unknown state manager")
        for index, block in enumerate(self._blocks):
            matches = [entry for entry in owned if entry.module is block]
            if len(matches) != 1:
                raise CacheCleanupFailed("FBC hook coverage differs from the inspected block layout")
            entry = matches[0]
            expected = api.hook_names[0] if index == 0 else api.hook_names[1]
            if entry.name != expected or entry.hook.state_manager is not manager:
                raise CacheCleanupFailed("FBC hook roles or shared state differ from the inspected layout")
            if index == 0:
                if getattr(entry.hook, "threshold", None) != 0.05:
                    raise CacheCleanupFailed("FBC installed a different threshold")
            elif getattr(entry.hook, "is_tail", None) is not (index == len(self._blocks) - 1):
                raise CacheCleanupFailed("FBC installed a different tail block")
        manager.reset()  # Explicit fresh state/context, before any pipeline call.

    @staticmethod
    def _invalidate_caches(snapshots: list[_Snapshot], api: _Api) -> None:
        for snapshot in snapshots:
            registry = _registry(snapshot.module, api)
            if registry is not None:
                registry._child_registries_cache = None

    def _cleanup(
        self, snapshots: list[_Snapshot], owned: list[_OwnedHook], config: FirstBlockCacheConfig, api: _Api,
    ) -> list[str]:
        assert self._model is not None
        errors: list[str] = []
        tree = list(self._model.named_modules())
        tree_unchanged = len(tree) == len(snapshots) and all(
            path == snapshot.path and module is snapshot.module
            for (path, module), snapshot in zip(tree, snapshots)
        )
        if not tree_unchanged:
            errors.append("the shared module tree changed during the FBC call")

        def attempt(label: str, operation: Callable[[], object]) -> None:
            try:
                operation()
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

        managers: list[StateManager] = []
        for entry in owned:
            manager = entry.manager
            if entry.hook.state_manager is not manager:
                errors.append("an owned FBC state manager was replaced")
            if not any(manager is prior for prior in managers):
                managers.append(manager)
                attempt("reset FBC state/context", manager.reset)
        for entry in reversed(owned):
            def remove(entry: _OwnedHook = entry) -> None:
                if _registry(entry.module, api) is not entry.registry or entry.registry.get_hook(entry.name) is not entry.hook:
                    raise CacheCleanupFailed("an owned FBC hook or registry was replaced or removed")
                if not _same_forward(entry.module.forward, entry.installed_forward):
                    raise CacheCleanupFailed("an owned FBC forward was replaced")
                entry.registry.remove_hook(entry.name, recurse=False)
            attempt("remove owned FBC hook", remove)
        # Only call the recursive library disable once no FBC name remains,
        # so it cannot remove a replacement or foreign hook with that name.
        try:
            remaining = any(
                registry.get_hook(name) is not None
                for snapshot in snapshots if (registry := _registry(snapshot.module, api)) is not None
                for name in api.hook_names
            )
            current = self._model._cache_config
            if remaining:
                errors.append("FBC hooks remain on the shared denoiser")
            elif current is config and tree_unchanged:
                attempt("disable owned cache config", self._model.disable_cache)
            elif current is not None:
                errors.append("the owned cache config was replaced")
            if self._model._cache_config is not None:
                errors.append("the shared denoiser still has a cache config")
        except BaseException as exc:
            errors.append(f"verify cache config: {type(exc).__name__}: {exc}")
        for snapshot in snapshots:
            def restore(snapshot: _Snapshot = snapshot) -> None:
                registry = _registry(snapshot.module, api)
                if snapshot.is_block and not _same_forward(snapshot.module.forward, snapshot.forward):
                    raise CacheCleanupFailed("a cached block forward was not restored")
                if snapshot.is_block and not snapshot.forward_owned and "forward" in vars(snapshot.module):
                    delattr(snapshot.module, "forward")
                    if not _same_forward(snapshot.module.forward, snapshot.forward):
                        raise CacheCleanupFailed("a block's original class forward changed")
                if snapshot.registry is None:
                    if registry is None:
                        return
                    if registry.hooks or registry._hook_order or registry._fn_refs:
                        raise CacheCleanupFailed("a new registry still owns hooks")
                    delattr(snapshot.module, "_diffusers_hook")
                    return
                if registry is None or registry is not snapshot.registry:
                    raise CacheCleanupFailed("a pre-existing registry was replaced")
                if (tuple(registry._hook_order) != snapshot.order or len(registry._fn_refs) != len(snapshot.refs)
                        or any(a is not b for a, b in zip(registry._fn_refs, snapshot.refs, strict=True))
                        or set(registry.hooks) != set(snapshot.hooks)
                        or any(registry.hooks[name] is not hook for name, hook in snapshot.hooks.items())):
                    raise CacheCleanupFailed("pre-existing hook ownership changed")
                if isinstance(snapshot.child_cache, _Absent):
                    if hasattr(registry, "_child_registries_cache"):
                        delattr(registry, "_child_registries_cache")
                else:
                    registry._child_registries_cache = snapshot.child_cache
            attempt("restore hook registry/cache", restore)
        return errors

    def _unsafe(self, errors: list[str]) -> CacheCleanupFailed:
        self._invalidated = True
        if self._model is not None:
            _invalidate_identity(self._model)
        self._diagnostics.extend(errors)
        if self._discard is not None:
            try:
                self._discard()
            except BaseException as exc:
                self._diagnostics.append(f"discard shared pipelines: {type(exc).__name__}: {exc}")
        return CacheCleanupFailed("; ".join(self._diagnostics))

    @contextmanager
    def cached_call(self) -> Iterator[FbcLayout]:
        self.assert_off()
        model, api = self._live()
        snapshots = self._snapshot(model, api)
        config = api.config_type(threshold=0.05)
        owned: list[_OwnedHook] = []
        discovery_errors: list[str] = []
        failure: BaseException | None = None
        self._active = True
        try:
            self._invalidate_caches(snapshots, api)
            try:
                model.enable_cache(config)
            except BaseException:
                try:
                    owned, discovery_errors = self._discover(snapshots, api)
                except BaseException as discovery:
                    discovery_errors.append(f"inspect partial FBC installation: {type(discovery).__name__}: {discovery}")
                raise
            owned, discovery_errors = self._discover(snapshots, api)
            if discovery_errors:
                raise CacheCleanupFailed("; ".join(discovery_errors))
            self._check_installed(owned, api, config)
            api.registry_type.check_if_exists_or_initialize(model)
            self._invalidate_caches(snapshots, api)
            assert self._layout is not None
            yield self._layout
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                errors = [*discovery_errors, *self._cleanup(snapshots, owned, config, api)]
            except BaseException as cleanup:
                errors = [*discovery_errors, f"FBC cleanup aborted: {type(cleanup).__name__}: {cleanup}"]
            finally:
                self._active = False
            try:
                if errors:
                    unsafe = self._unsafe(errors)
                    if failure is not None:
                        failure.add_note(str(unsafe))
                    else:
                        raise unsafe
            finally:
                # Deferred backend disposal runs after this generator exits.
                # Do not keep app-owned module snapshots in its error frame.
                snapshots.clear()
                owned.clear()
                del model, api, config
