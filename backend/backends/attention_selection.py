"""Inactive FLUX attention configuration; no loader or request path calls this.

This owns processor overrides, never the process-global dispatcher. The
public Diffusers context restores the setter's incidental global change.
The caller must retain the handle and configure/restore only while idle;
shared pipeline siblings must use the same transformer and the same handle.
No setting here grants permission to render an execution recipe.
"""
from __future__ import annotations

import sys
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, cast


class AttentionSelectionError(ValueError):
    """A processor layout cannot safely own the requested override."""


class _Processor(Protocol):
    _attention_backend: object
    _parallel_config: object


class _Attention(Protocol):
    processor: object


class _Block(Protocol):
    attn: object


class _Transformer(Protocol):
    config: Mapping[str, object]
    transformer_blocks: Iterable[object]
    single_transformer_blocks: Iterable[object]
    attn_processors: Mapping[str, object]

    def set_attention_backend(self, backend: str) -> None: ...


@dataclass(frozen=True)
class _Bindings:
    transformer_type: type
    joint_type: type
    single_type: type
    attention_type: type
    processor_type: type
    backend_context: Callable[[str], AbstractContextManager[object]]
    target: object


def _load_bindings() -> _Bindings:
    # Diffusers imports torch_utils, whose module initialization asks for a
    # device. Only explicit configuration may reach this; importing this
    # helper and selecting None remain cold. The app's import order still
    # belongs to config, whose environment policy is not ours to change.
    if "backend.config" not in sys.modules:
        raise AttentionSelectionError("Import backend.config before configuring Diffusers attention")
    from diffusers.models.attention_dispatch import AttentionBackendName, attention_backend
    from diffusers.models.transformers.transformer_flux import (
        FluxAttention,
        FluxAttnProcessor,
        FluxSingleTransformerBlock,
        FluxTransformer2DModel,
        FluxTransformerBlock,
    )

    return _Bindings(
        FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock,
        FluxAttention, FluxAttnProcessor, attention_backend, AttentionBackendName._NATIVE_CUDNN,
    )


@dataclass(frozen=True)
class _Slot:
    block: object
    attention: object
    processor: _Processor

    def matches(self, other: _Slot) -> bool:
        return (self.block is other.block and self.attention is other.attention
                and self.processor is other.processor)


def _coverage(transformer: object, bindings: _Bindings) -> dict[str, _Slot]:
    if type(transformer) is not bindings.transformer_type:
        raise AttentionSelectionError("Attention selection requires the exact FLUX transformer class")
    # Exact defining classes were checked before these local interface
    # contracts: no subclass may replace the forward/processor behavior.
    model = cast(_Transformer, transformer)
    if not callable(model.set_attention_backend) or not callable(bindings.backend_context):
        raise AttentionSelectionError("The public attention setter and backend context are required")
    expected: dict[str, _Slot] = {}
    for name, blocks, count_name, block_type in (
        ("transformer_blocks", model.transformer_blocks, "num_layers", bindings.joint_type),
        ("single_transformer_blocks", model.single_transformer_blocks, "num_single_layers", bindings.single_type),
    ):
        items = tuple(blocks)
        count = model.config.get(count_name)
        if not items or type(count) is not int or count != len(items):
            raise AttentionSelectionError(f"Incomplete FLUX attention coverage in {name}")
        for index, block in enumerate(items):
            path = f"{name}.{index}.attn.processor"
            if type(block) is not block_type:
                raise AttentionSelectionError(f"Unsupported FLUX block at {path}")
            attention = cast(_Block, block).attn
            if type(attention) is not bindings.attention_type:
                raise AttentionSelectionError(f"Unsupported FLUX attention module at {path}")
            processor = cast(_Attention, attention).processor
            if type(processor) is not bindings.processor_type:
                raise AttentionSelectionError(f"Unsupported FLUX attention processor at {path}")
            processor = cast(_Processor, processor)
            if processor._parallel_config is not None:
                raise AttentionSelectionError(f"Parallel attention is outside this recipe at {path}")
            if not hasattr(processor, "_attention_backend"):
                raise AttentionSelectionError(f"The attention processor has no backend field at {path}")
            expected[path] = _Slot(block, attention, processor)
    actual = model.attn_processors
    if set(actual) != set(expected):
        raise AttentionSelectionError("Missing or extra FLUX attention processor paths")
    if any(actual[path] is not slot.processor for path, slot in expected.items()):
        raise AttentionSelectionError("FLUX attention processor identities disagree with their blocks")
    return expected


@dataclass(frozen=True)
class _Snapshot:
    processor: _Processor
    backend: object
    owned_field: bool


def _snapshots(coverage: Mapping[str, _Slot]) -> tuple[_Snapshot, ...]:
    unique = {id(slot.processor): slot.processor for slot in coverage.values()}
    return tuple(
        _Snapshot(processor, processor._attention_backend, "_attention_backend" in vars(processor))
        for processor in unique.values()
    )


def _restore(snapshots: tuple[_Snapshot, ...]) -> list[BaseException]:
    failures: list[BaseException] = []
    for snapshot in snapshots:
        try:
            processor = snapshot.processor
            if snapshot.owned_field:
                processor._attention_backend = snapshot.backend
            elif "_attention_backend" in vars(processor):
                del processor._attention_backend
            if (processor._attention_backend != snapshot.backend
                    or ("_attention_backend" in vars(processor)) != snapshot.owned_field):
                raise AttentionSelectionError("An attention processor did not restore its previous backend")
        except BaseException as exc:
            # A broken processor must not prevent the remaining snapshots
            # from being restored, including a shared sibling's processors.
            failures.append(exc)
    return failures


_lock = threading.RLock()
_owners: dict[int, weakref.ReferenceType[AttentionSelection]] = {}
_processor_owners: dict[int, weakref.ReferenceType[AttentionSelection]] = {}


class AttentionSelection:
    """One owned override on a transformer, shared by any pipeline siblings.

    verify() refuses later backend/layout changes. restore() always tries
    every original processor, leaves replacement/unrelated processors alone,
    and reports a changed layout so the caller discards that transformer.
    A successful restoration releases references and is idempotent. Failed
    restoration retains ownership so it can be retried after its cause is
    resolved; it must never be treated as a usable configured transformer.
    """

    def __init__(
        self, transformer: object, bindings: _Bindings, coverage: dict[str, _Slot],
        snapshots: tuple[_Snapshot, ...],
    ) -> None:
        self._transformer: object | None = transformer
        self._bindings = bindings
        self._coverage = coverage
        self._snapshots = snapshots
        self._restored = False
        self._restore_started = False
        self._changed: Exception | None = None

    def verify(self) -> None:
        with _lock:
            if self._restored or self._restore_started:
                raise AttentionSelectionError("This attention selection is restored or still being restored")
            current = _coverage(self._transformer, self._bindings)
            if (current.keys() != self._coverage.keys()
                    or any(not current[path].matches(slot) for path, slot in self._coverage.items())):
                raise AttentionSelectionError("FLUX attention coverage changed after configuration")
            if any(slot.processor._attention_backend != self._bindings.target for slot in current.values()):
                raise AttentionSelectionError("FLUX attention backend changed after configuration")

    def _release(self) -> None:
        reference = _owners.get(id(self._transformer))
        if reference is not None and reference() is self:
            del _owners[id(self._transformer)]
        for snapshot in self._snapshots:
            identity = id(snapshot.processor)
            reference = _processor_owners.get(identity)
            if reference is not None and reference() is self:
                del _processor_owners[identity]
        self._transformer = None
        self._coverage = {}
        self._snapshots = ()
        self._restored = True

    def restore(self) -> None:
        with _lock:
            if self._restored:
                return
            if not self._restore_started:
                try:
                    self.verify()
                except Exception as exc:
                    self._changed = exc
                self._restore_started = True
            failures = _restore(self._snapshots)
            if failures:
                raise AttentionSelectionError("Attention restoration failed; discard this transformer") from (
                    BaseExceptionGroup("Attention processor restoration failures", failures)
                )
            changed = self._changed
            self._changed = None  # Its traceback can otherwise retain modules after release.
            self._release()
            if changed is not None:
                raise AttentionSelectionError("Attention coverage changed; discard this transformer") from changed


def configure_attention(transformer: object, recipe: str | None) -> AttentionSelection | None:
    """Prepare the inactive cudnn-v1 helper; None leaves everything unchanged.

    This is deliberately disconnected from application loading/generation.
    The caller still needs admission, environment policy and separate card
    qualification before activation. It must call only while idle, after
    placement, and retain the returned handle until all shared siblings go.
    """
    if recipe is None:
        return None
    if type(recipe) is not str or recipe != "cudnn-v1":
        raise AttentionSelectionError(f"Unsupported attention recipe: {recipe!r}")
    with _lock:
        previous = _owners.get(id(transformer))
        if previous is not None and previous() is not None:
            raise AttentionSelectionError("This shared transformer already has an attention owner")
        bindings = _load_bindings()
        coverage = _coverage(transformer, bindings)
        snapshots = _snapshots(coverage)
        identities = tuple(id(snapshot.processor) for snapshot in snapshots)
        if any((owner := _processor_owners.get(identity)) is not None and owner() is not None
               for identity in identities):
            raise AttentionSelectionError("These shared attention processors already have an owner")
        handle = AttentionSelection(transformer, bindings, coverage, snapshots)
        identity = id(transformer)

        def forgotten(reference: weakref.ReferenceType[AttentionSelection]) -> None:
            with _lock:
                if _owners.get(identity) is reference:
                    del _owners[identity]
                for processor_identity in identities:
                    if _processor_owners.get(processor_identity) is reference:
                        del _processor_owners[processor_identity]

        reference = weakref.ref(handle, forgotten)
        _owners[identity] = reference
        for processor_identity in identities:
            _processor_owners[processor_identity] = reference
        try:
            with bindings.backend_context("native"):
                cast(_Transformer, transformer).set_attention_backend("_native_cudnn")
                handle.verify()
        except BaseException as exc:
            failures = _restore(snapshots)
            if failures:
                exc.add_note("Attention rollback failed; discard this transformer: "
                             + "; ".join(f"{type(error).__name__}: {error}" for error in failures))
            handle._release()
            raise
        return handle
