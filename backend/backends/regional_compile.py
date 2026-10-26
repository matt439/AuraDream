"""Inactive regional-registration helpers; no loading or availability wiring.

Registration installs lazy wrappers, not kernels or a warm image. A future
caller must own an unpublished, already-placed candidate and discard it on
failure. Every later compiled call must also enter strict_compiler_policy.
Importing this module imports no tensor, model or compiler library.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock, FluxTransformer2DModel, FluxTransformerBlock,
    )


_BLOCK_NAMES = ("FluxTransformerBlock", "FluxSingleTransformerBlock")


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

    __slots__ = ("_transformer", "_facts")

    def __init__(self, transformer: object, facts: RegionFacts) -> None:
        self._transformer: object | None = transformer
        self._facts: RegionFacts | None = facts

    @property
    def facts(self) -> RegionFacts | None:
        return self._facts

    @property
    def closed(self) -> bool:
        return self._transformer is None

    def close(self) -> None:
        self._transformer = None
        self._facts = None


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
