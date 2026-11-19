"""Prove the live gallery root and its frozen Tiled Refine preparation."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager

from PIL import Image

from . import storage
from . import tiled_refine_contract as contract
from .instruction_edit import InstructionEditSource
from .refusal_codes import IMAGE_BINNED, IMAGE_UNKNOWN, CodedError


@contextmanager
def _open_root(
    spec: contract.TiledRefineSpec, recorded: contract.StoredTiledRefine | None,
) -> Iterator[tuple[InstructionEditSource, Image.Image]]:
    """Read the root proof and its borrowed pixels through one handle."""
    record = storage.get_image(spec.source)
    if record is None:
        binned = storage.get_image(spec.source, True) is not None
        raise CodedError(
            f"Tiled Refine source #{spec.source} is missing or in the recycle bin.",
            IMAGE_BINNED if binned else IMAGE_UNKNOWN, [spec.source],
        )
    # The filename is a proof bound to the row. The gallery alone resolves the path.
    InstructionEditSource._bare_png(record.filename)
    (path,) = storage.reference_paths([spec.source], "Tiled Refine source")
    if path.name != record.filename or path.is_symlink() or path.is_junction():
        raise ValueError(f"Tiled Refine source #{spec.source} changed or is a linked file.")
    try:
        handle = Image.open(path, formats=["PNG"])
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Tiled Refine source #{spec.source} cannot be decoded.") from exc
    with closing(handle):
        try:
            proof = contract.descriptor_from_image(record.filename, handle)
        except OSError as exc:
            raise ValueError(f"Tiled Refine source #{spec.source} cannot be decoded.") from exc
        if (proof.width, proof.height) != (record.width, record.height):
            raise ValueError(f"Tiled Refine source #{spec.source} no longer matches its gallery dimensions.")
        if proof.hash[:16] != record.content_hash:
            raise ValueError(f"Tiled Refine source #{spec.source} no longer matches its gallery provenance.")
        if recorded is not None and recorded.root_source != proof:
            raise ValueError(f"Tiled Refine source #{spec.source} no longer matches its recorded original RGB proof.")
        yield proof, handle


def _prepared(
    spec: contract.TiledRefineSpec, recorded: contract.StoredTiledRefine | None,
) -> tuple[contract.StoredTiledRefine, Image.Image]:
    with _open_root(spec, recorded) as (proof, root):
        canvas = contract.prepare_canvas(root)
    try:
        digest = contract.rgb_sha256(canvas)
        if recorded is not None:
            if digest != recorded.prepared_rgb_sha256:
                raise ValueError("Tiled Refine source no longer matches its recorded prepared RGB proof.")
            return recorded, canvas
        return contract.StoredTiledRefine(
            source=spec.source, recipe=spec.recipe, root_source=proof,
            preparation=contract.PREPARATION, prepared_rgb_sha256=digest,
        ), canvas
    except BaseException:
        canvas.close()
        raise


def bind_source(value: object) -> contract.StoredTiledRefine:
    """Freeze both original and prepared RGB proofs before constructing a key."""
    is_bound = isinstance(value, contract.StoredTiledRefine) or (isinstance(value, dict) and "root_source" in value)
    recorded = contract.bound_spec(value) if is_bound else None
    spec = recorded.request() if recorded is not None else contract.TiledRefineSpec.model_validate(value)
    bound, canvas = _prepared(spec, recorded)
    canvas.close()
    return bound


def load_bound_source(value: contract.StoredTiledRefine) -> Image.Image:
    """Prove both stages before returning the owned working canvas."""
    bound = contract.bound_spec(value)
    return _prepared(bound.request(), bound)[1]


def prove_root(value: contract.StoredTiledRefine) -> None:
    """Reprove the original gallery pixels at every tile boundary."""
    bound = contract.bound_spec(value)
    with _open_root(bound.request(), bound):
        pass
