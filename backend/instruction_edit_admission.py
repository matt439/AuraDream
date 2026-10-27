"""Gallery-bound source checks for an unavailable instruction editor.

Only gallery ids resolve files. Portable source descriptors are compared with
the original decoded RGB pixels; they never authorize opening their filename.
No function here loads a model or enables execution.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from . import instruction_edit as edit
from . import storage

if TYPE_CHECKING:
    from .schemas_generate import GenerateRequest


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
    source = storage.get_image(spec.source)
    if source is None:
        raise ValueError(f"Instruction-edit source #{spec.source} is missing or in the recycle bin.")
    edit.InstructionEditSource._bare_png(source.filename)
    path = storage.reference_paths([spec.source], "instruction-edit source")[0]
    if path.is_symlink() or path.is_junction():
        raise ValueError(f"Instruction-edit source #{spec.source} cannot be a linked file.")
    try:
        with Image.open(path) as image:
            proof = edit.descriptor_from_image(source.filename, image)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Instruction-edit source #{spec.source} cannot be decoded.") from exc
    if (proof.width, proof.height) != (source.width, source.height):
        raise ValueError(f"Instruction-edit source #{spec.source} no longer matches its gallery dimensions.")
    if recorded is not None and proof != recorded.edit_source:
        raise ValueError(f"Instruction-edit source #{spec.source} no longer matches its recorded pixels or provenance.")
    resolved = edit.geometry(proof.width, proof.height, recipe=spec.recipe)
    return edit.StoredInstructionEdit(source=spec.source, recipe=spec.recipe, edit_source=proof), resolved


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
