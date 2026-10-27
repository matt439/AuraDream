"""Gallery-bound source checks and pixels for the gated instruction editor.

Only gallery ids resolve files. Portable source descriptors are compared with
the original decoded RGB pixels; they never authorize opening their filename.
`load_bound_source` reproves a bound edit and hands its working-size pixels
to a qualified backend. No function here loads a model or enables execution.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from . import instruction_edit as edit
from . import storage

if TYPE_CHECKING:
    from .schemas_generate import GenerateRequest


def _prove(
    spec: edit.InstructionEditSpec, *, keep_image: bool, recorded: edit.InstructionEditSource | None = None,
) -> tuple[edit.StoredInstructionEdit, edit.InstructionEditGeometry, Image.Image | None]:
    """Open a live gallery source once and describe its original pixels.

    A ``recorded`` proof must match exactly before geometry is resolved.
    With ``keep_image`` the decoded RGB copy whose bytes were hashed comes
    back too, so nothing can change between the proof and the pixels used.
    The copy owns its pixels; the file handle is closed before returning.
    """
    source = storage.get_image(spec.source)
    if source is None:
        raise ValueError(f"Instruction-edit source #{spec.source} is missing or in the recycle bin.")
    edit.InstructionEditSource._bare_png(source.filename)
    path = storage.reference_paths([spec.source], "instruction-edit source")[0]
    if path.is_symlink() or path.is_junction():
        raise ValueError(f"Instruction-edit source #{spec.source} cannot be a linked file.")
    decoded: Image.Image | None = None
    try:
        try:
            with Image.open(path) as image:
                if keep_image:
                    decoded = image.convert("RGB")
                    proof = edit.descriptor_from_image(source.filename, decoded)
                else:
                    proof = edit.descriptor_from_image(source.filename, image)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Instruction-edit source #{spec.source} cannot be decoded.") from exc
        if (proof.width, proof.height) != (source.width, source.height):
            raise ValueError(f"Instruction-edit source #{spec.source} no longer matches its gallery dimensions.")
        if recorded is not None and proof != recorded:
            raise ValueError(
                f"Instruction-edit source #{spec.source} no longer matches its recorded pixels or provenance."
            )
        resolved = edit.geometry(proof.width, proof.height, recipe=spec.recipe)
    except BaseException:
        if decoded is not None:
            decoded.close()
        raise
    return edit.StoredInstructionEdit(source=spec.source, recipe=spec.recipe, edit_source=proof), resolved, decoded


def bind_source(value: object) -> tuple[edit.StoredInstructionEdit, edit.InstructionEditGeometry]:
    """Prove a live gallery source and freeze its complete original identity.

    A replay's supplied descriptor must match exactly, including the filename
    and dimensions. An unresolved imported edit never borrows another local id.
    """
    if isinstance(value, edit.StoredInstructionEdit) or (
        isinstance(value, dict) and "edit_source" in value
    ):
        recorded = edit.StoredInstructionEdit.model_validate(value)
        spec = recorded.request()
    else:
        recorded = None
        spec = edit.InstructionEditSpec.model_validate(value)
    bound, resolved, _ = _prove(spec, keep_image=False, recorded=None if recorded is None else recorded.edit_source)
    return bound, resolved


def load_bound_source(
    value: edit.StoredInstructionEdit,
) -> tuple[edit.StoredInstructionEdit, edit.InstructionEditGeometry, Image.Image]:
    """Reprove a bound edit's source and hand back its prepared pixels.

    The proof and the decode share one open handle, and the recorded proof
    must still match exactly. The answer is a new RGB image at the recipe's
    working size, owned by the caller (who closes it); the stored file and
    its decoded copy are never handed out. Loads no model.
    """
    if not isinstance(value, edit.StoredInstructionEdit):
        raise ValueError("Bind and verify the instruction-edit source before loading its pixels.")
    recorded = edit.StoredInstructionEdit.model_validate(value)
    spec = recorded.request()
    bound, resolved, decoded = _prove(spec, keep_image=True, recorded=recorded.edit_source)
    if decoded is None:
        raise RuntimeError("the instruction-edit source proof kept no pixels")
    try:
        return bound, resolved, edit.prepare_source(decoded, resolved)
    finally:
        decoded.close()


def bind_request(request: GenerateRequest, model_id: str) -> GenerateRequest:
    """Validate and bind intent before keys, persistence or candidate admission.

    This is a structural preparation seam, not permission to queue or render.
    Public admission still refuses the unavailable editor after these checks.
    """
    spec = request.validate_edit()
    if spec is None:
        return request
    recipe = edit.recipe_for(spec.recipe)
    if model_id != recipe.model_id:
        raise ValueError(f"This instruction edit requires {recipe.model_id}; select it explicitly in Models.")
    bound, resolved = bind_source(spec)
    if (request.width, request.height) != resolved.output_size:
        raise ValueError(
            f"This instruction-edit source requires {resolved.width}x{resolved.height}; "
            "preview its source geometry before queuing."
        )
    return request.model_copy(update={"edit": bound})
