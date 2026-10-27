"""Regional registration for private preparation; public availability stays closed.

Registration installs lazy wrappers, not kernels or a warm image. The private
backend caller owns an unpublished, already-placed candidate and discards it
on failure. Its compiled calls also enter strict_compiler_policy.
Importing this module imports no tensor, model or compiler library.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from importlib import import_module
import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock, FluxTransformer2DModel, FluxTransformerBlock,
    )


_BLOCK_NAMES = ("FluxTransformerBlock", "FluxSingleTransformerBlock")
FirstCallState = Literal["pending", "running", "returned", "failed", "cancelled"]
FailureStage = Literal["forward", "verification", "policy_exit"]
_CALL_STATES = ("pending", "running", "returned", "failed", "cancelled")
_FAILURE_STAGES = ("forward", "verification", "policy_exit")


@dataclass(frozen=True)
class RegionalDiagnostics:
    """A per-load observation, never proof of compiled/warm kernels or speed.

    A pending first call has not attempted a forward. Returned means the
    observed invocation, ownership checks and policy exit all returned; it
    does not establish pixels, cache hits, fallback behavior or qualification.
    No timing, error, model reference or reproduction identity belongs here.
    """

    load_id: int
    first_call: FirstCallState = "pending"
    failure_stage: FailureStage | None = None
    recipe: Literal["regional-v1"] = "regional-v1"
    registration: Literal["registered"] = "registered"

    def __post_init__(self) -> None:
        if type(self.load_id) is not int or self.load_id < 1:
            raise ValueError("regional diagnostics need a positive per-load identity")
        if (type(self.recipe) is not str or self.recipe != "regional-v1"
                or type(self.registration) is not str or self.registration != "registered"):
            raise ValueError("regional diagnostics describe only registered regional-v1")
        if type(self.first_call) is not str or self.first_call not in _CALL_STATES:
            raise ValueError("regional diagnostics have an invalid first-call state")
        failed = self.first_call in ("failed", "cancelled")
        if ((failed and (type(self.failure_stage) is not str or self.failure_stage not in _FAILURE_STAGES))
                or (not failed and self.failure_stage is not None)):
            raise ValueError("regional diagnostics have an invalid failure stage")


class RegionalDiagnosticOwner:
    """Weight-free mutable observation behind immutable snapshots.

    Only successful publication assigns its load identity. The caller begins
    observation immediately before the first forward, after preflight and
    policy entry; either refusal leaves pending unchanged. Each owner has one
    opaque active token. Retired/foreign/repeated completions are ignored, so
    an old call cannot change a replacement owner's first-call observation.
    Readers may retain this owner without retaining a registration or model.
    """

    __slots__ = ("_lock", "_load_id", "_state", "_stage", "_active", "_retired")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._load_id: int | None = None
        self._state: FirstCallState = "pending"
        self._stage: FailureStage | None = None
        self._active: object | None = None
        self._retired = False

    def publish(self, load_id: int) -> None:
        """Bind once at publication; the same identity never resets a call."""
        if type(load_id) is not int or load_id < 1:
            raise ValueError("regional diagnostics need a positive per-load identity")
        with self._lock:
            if self._retired:
                raise RuntimeError("cannot publish retired regional diagnostics")
            if self._load_id is not None and self._load_id != load_id:
                raise RuntimeError("regional diagnostics already belong to another load")
            self._load_id = load_id

    def snapshot(self) -> RegionalDiagnostics | None:
        """Cold value read only, before publication/after retirement is None."""
        with self._lock:
            if self._retired or self._load_id is None:
                return None
            return RegionalDiagnostics(self._load_id, self._state, self._stage)

    def begin_first_call(self) -> object | None:
        """Call only after policy entry, immediately before attempting forward.

        A terminal first outcome remains terminal across later calls. Refuse
        overlapping first calls rather than misreporting their ownership.
        """
        with self._lock:
            if self._retired or self._load_id is None:
                raise RuntimeError("the first regional call needs published live diagnostics")
            if type(self._state) is not str or self._state not in _CALL_STATES:
                raise RuntimeError("the regional diagnostic owner has an invalid state")
            if self._state == "running":
                raise RuntimeError("the first regional call is already running")
            if self._state != "pending":
                return None
            token = object()
            self._active, self._state = token, "running"
            return token

    def finish_first_call(
        self, token: object | None, outcome: Literal["returned", "failed", "cancelled"],
        *, failure_stage: FailureStage | None = None,
    ) -> bool:
        """Freeze the first outcome once; stale tokens never mutate state.

        Returned is recorded by the caller only after post-call verification
        and policy cleanup. A failed/cancelled attempt records its primary
        failure stage, not an exception or traceback. Later job cancellation
        after this boundary returned must not relabel the call.
        """
        with self._lock:
            if (self._retired or self._load_id is None or token is None
                    or token is not self._active or self._state != "running"):
                return False
            if type(outcome) is not str or outcome not in ("returned", "failed", "cancelled"):
                raise ValueError("invalid first regional call outcome")
            # Validate without changing the running state or consuming its
            # token, so a malformed update cannot publish a false terminal.
            value = RegionalDiagnostics(self._load_id, outcome, failure_stage)
            self._state, self._stage, self._active = value.first_call, value.failure_stage, None
            return True

    def retire(self) -> None:
        """Idempotent, bounded cleanup; no model/allocator/global callbacks."""
        with self._lock:
            self._retired = True
            self._load_id, self._active = None, None
            self._state, self._stage = "pending", None


@dataclass(frozen=True)
class RegionFacts:
    """Discovery/registration facts only; no claim of compilation or warm-up."""

    recipe: Literal["regional-v1"]
    block_counts: tuple[tuple[str, int], ...]

    @property
    def total_blocks(self) -> int:
        return sum(count for _, count in self.block_counts)


class RegionalRegistration:
    """Own per-load references until close; the pipeline owns its wrappers.

    close does not unwrap a module, clear compiler state, or delete a cache.
    The loader must drop the pipeline and its siblings separately on unload.
    """

    __slots__ = ("_transformer", "_facts", "_diagnostics")

    def __init__(self, transformer: object, facts: RegionFacts) -> None:
        self._transformer: object | None = transformer
        self._facts: RegionFacts | None = facts
        self._diagnostics = RegionalDiagnosticOwner()

    @property
    def diagnostics(self) -> RegionalDiagnosticOwner:
        """The pure observation owner; its reader retains no model reference."""
        return self._diagnostics

    def retire_diagnostics(self) -> None:
        self._diagnostics.retire()

    @property
    def facts(self) -> RegionFacts | None:
        return self._facts

    @property
    def closed(self) -> bool:
        return self._transformer is None

    def close(self) -> None:
        self.retire_diagnostics()
        self._transformer = None
        self._facts = None

    def require(self, transformer: object) -> RegionFacts:
        """Require the live root and this fixed recipe, without new imports."""
        facts = self._facts
        if (self._transformer is None or self._transformer is not transformer
                or type(facts) is not RegionFacts or facts.recipe != "regional-v1"):
            raise RuntimeError("the regional registration does not own this live transformer and recipe")
        counts = facts.block_counts
        if (not isinstance(counts, tuple) or len(counts) != len(_BLOCK_NAMES)
                or any(not isinstance(entry, tuple) or len(entry) != 2 or entry[0] != name
                       or type(entry[1]) is not int or entry[1] < 1
                       for name, entry in zip(_BLOCK_NAMES, counts))):
            raise RuntimeError("the regional registration has invalid discovery facts")
        return facts


def _flux_types() -> tuple[
    type[FluxTransformer2DModel], type[FluxTransformerBlock], type[FluxSingleTransformerBlock],
]:
    # config prepares the environment before any diffusers import.
    import_module("..config", __package__)
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock, FluxTransformer2DModel, FluxTransformerBlock,
    )

    return FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock


def _inspect(transformer: object) -> tuple[FluxTransformer2DModel, RegionFacts]:
    model_type, block_type, single_type = _flux_types()
    # An unrelated/local class sharing a library name is not this integration;
    # subclasses are a separate qualification too. No model is constructed here.
    if not isinstance(transformer, model_type) or type(transformer) is not model_type:
        raise ValueError("regional-v1 needs the installed FluxTransformer2DModel class")
    if getattr(transformer, "_compiled_call_impl", None) is not None:
        raise ValueError("the FLUX transformer is already compiled; use a fresh unpublished candidate")
    declared = getattr(transformer, "_repeated_blocks", None)
    if (not isinstance(declared, (list, tuple)) or len(declared) != len(_BLOCK_NAMES)
            or any(not isinstance(name, str) for name in declared) or set(declared) != set(_BLOCK_NAMES)):
        raise ValueError("regional-v1 needs both exact FLUX repeated-block declarations")
    if not callable(getattr(transformer, "compile_repeated_blocks", None)):
        raise ValueError("the FLUX transformer has no regional registration API")
    families = dict(zip(_BLOCK_NAMES, (block_type, single_type)))
    counts = dict.fromkeys(_BLOCK_NAMES, 0)
    for module in transformer.modules():
        name = type(module).__name__
        if name not in families:
            continue
        if type(module) is not families[name]:
            raise ValueError(f"regional-v1 needs the installed {name} class")
        if not callable(getattr(module, "compile", None)):
            raise ValueError(f"{name} has no module registration API")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise ValueError(f"{name} is already compiled; use a fresh unpublished candidate")
        counts[name] += 1
    if not all(counts.values()):
        raise ValueError("regional-v1 needs at least one block from each declared FLUX family")
    return transformer, RegionFacts("regional-v1", tuple(counts.items()))


def inspect_regions(transformer: object) -> RegionFacts:
    """Validate the actual model and every matching region before mutation."""
    return _inspect(transformer)[1]


def _dynamo_patch() -> AbstractContextManager[object]:
    import_module("..config", __package__)
    from torch.utils._config_module import ConfigModule

    config = import_module("torch._dynamo.config")
    if not isinstance(config, ConfigModule):
        raise RuntimeError("the installed Dynamo configuration has no scoped patch API")
    patch = config.patch({"suppress_errors": False, "fail_on_recompile_limit_hit": True})
    if not isinstance(patch, AbstractContextManager):
        raise RuntimeError("the installed Dynamo patch API is not a context manager")
    return patch


@contextmanager
def strict_compiler_policy() -> Iterator[None]:
    """Use installed context-local flags; restore on every exit, never reset.

    This context alone proves no absence of fallback on a real workload. The
    runtime and card qualification must establish that before availability.
    """
    with _dynamo_patch():
        yield


def register_regions(
    transformer: object, *, discard_on_failure: Callable[[], None],
) -> RegionalRegistration:
    """Register lazy wrappers atomically with respect to candidate publication.

    The library mutates regions one by one. Any failure (validation included)
    calls the owner's discard exactly once and re-raises the original error;
    a partially registered model must never be reused as Eager. A successful
    return grants no runtime availability and performs no warm-up call.
    """
    try:
        model, facts = _inspect(transformer)
        with strict_compiler_policy():
            model.compile_repeated_blocks(backend="inductor", fullgraph=True, dynamic=False, mode="default")
        return RegionalRegistration(model, facts)
    except BaseException as exc:
        try:
            discard_on_failure()
        except BaseException as cleanup:
            exc.add_note(f"Discarding the unpublished regional candidate also failed: {type(cleanup).__name__}: {cleanup}")
        raise
