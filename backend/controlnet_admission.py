"""Check ControlNet's bounded loader and re-prove the guide through its gallery ID."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from . import card_facts, storage, tiled_refine_contract
from .controlnet_contract import SIZE, UNRESOLVED, WEIGHTS, ControlMode, ControlNetPreview, ControlNetSpec, weight_info
from .execution import canonical_loader_execution
from .instruction_edit import InstructionEditSource, descriptor_from_image

if TYPE_CHECKING:
    from .backends.base import GenerationBackend
    from .config import Settings


# Both modes reproduced on the guarded Animagine run of 2026-11-22.
QUALIFIED_MODES: frozenset[ControlMode] = frozenset({"pose", "depth"})
LOADER_REQUIRED = (
    "ControlNet requires Animagine XL 4.0, FP8 with model offload, "
    "Windows RTX 5080 CUDA 0, Eager/Native and diffusers 0.42.1."
)
HELD = "This ControlNet mode is awaiting its guarded GPU trial."


def loader_refusal(
    backend: GenerationBackend, selected: Settings, *, mode: ControlMode | None = None,
    check_hardware: bool = True,
) -> str | None:
    if (
        (selected.model_id, selected.quantization, selected.offload)
        != (tiled_refine_contract.MODEL_ID, tiled_refine_contract.QUANTIZATION, tiled_refine_contract.OFFLOAD)
        or selected.device not in ("cuda", "cuda:0") or selected.loader_execution_error is not None
        or canonical_loader_execution(selected.loader_execution) is not None
    ):
        return LOADER_REQUIRED
    if not any(backend.controlnet_qualified(candidate) is True for candidate in (WEIGHTS if mode is None else [mode])):
        return HELD
    platform, version, gpu, native = backend.refine_host()
    if platform != "win32" or version != "0.42.1" or native is not True:
        return LOADER_REQUIRED
    if check_hardware and gpu != "NVIDIA GeForce RTX 5080":
        return LOADER_REQUIRED
    return backend.controlnet_owner_refusal()


def require_loader(backend: GenerationBackend, selected: Settings, mode: ControlMode) -> None:
    reason = loader_refusal(backend, selected, mode=mode, check_hardware=False)
    if reason is None:
        card_facts.card(selected.device)
        reason = loader_refusal(backend, selected, mode=mode)
    if reason is not None:
        raise ValueError(reason)


def capabilities(backend: GenerationBackend, selected: Settings) -> dict:
    facts = weight_info()
    return {
        mode: {**facts[mode], "reason": loader_refusal(backend, selected, mode=mode)}
        for mode in WEIGHTS
    }


def guide_matches_path(path: Path, proof: InstructionEditSource) -> bool:
    """File-path access is only for trusted recovery paths built from gallery filenames."""
    if path.is_symlink() or path.is_junction():
        return False
    try:
        with Image.open(path, formats=["PNG"]) as handle:
            return handle.size == SIZE and descriptor_from_image(path.name, handle) == proof
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


@contextmanager
def _guide(source: int, expected: InstructionEditSource | None) -> Iterator[tuple[InstructionEditSource, Image.Image]]:
    (path,) = storage.reference_paths([source], "ControlNet guide")
    record = storage.get_image(source)
    if record is None or path.name != record.filename or path.is_symlink() or path.is_junction():
        raise ValueError(f"ControlNet guide #{source} changed or is a linked file.")
    try:
        with Image.open(path, formats=["PNG"]) as handle:
            if handle.size != SIZE or (record.width, record.height) != SIZE:
                raise ValueError(f"ControlNet guide #{source} must be 1024 × 1024 pixels.")
            if storage.content_hash(handle) != record.content_hash:
                raise ValueError(f"ControlNet guide #{source} no longer matches its gallery pixels.")
            rgb = handle.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"ControlNet guide #{source} cannot be decoded.") from exc
    with closing(rgb):
        proof = descriptor_from_image(record.filename, rgb)
        if expected is not None and proof != expected:
            raise ValueError(f"ControlNet guide #{source} no longer matches its recorded RGB pixels.")
        yield proof, rgb


def preview(draft: ControlNetPreview) -> ControlNetSpec:
    with _guide(draft.source, None) as (proof, _):
        return ControlNetSpec(mode=draft.mode, source=draft.source, guide=proof, strength=draft.strength,
                              weight_revision=WEIGHTS[draft.mode].revision)


@contextmanager
def open_guide(spec: ControlNetSpec) -> Iterator[Image.Image]:
    validated = ControlNetSpec.model_validate(spec)
    if validated.source is None:
        raise ValueError(UNRESOLVED)
    with _guide(validated.source, validated.guide) as (_, rgb):
        yield rgb


def bind_source(spec: ControlNetSpec) -> ControlNetSpec:
    with open_guide(spec):
        return spec
