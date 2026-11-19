"""Synthetic stored edits; these helpers never invoke an editor backend."""
from __future__ import annotations

from PIL import Image

from backend import storage
from backend.instruction_edit import KONTEXT_V1, StoredInstructionEdit, descriptor_from_image, geometry
from backend.schemas import GenerateRequest


def stored_edit(source, *, recipe=KONTEXT_V1.name):
    with Image.open(storage.settings.outputs_dir / source.filename) as image:
        proof = descriptor_from_image(source.filename, image)
    return StoredInstructionEdit(source=source.id, recipe=recipe, edit_source=proof)


def save_edit(source, *, prompt="make the sky green", seed=2, recipe=KONTEXT_V1.name):
    spec = stored_edit(source, recipe=recipe)
    # Other fixtures may pass a copy with altered row geometry. Resolve from
    # the original proof, as the editor does.
    size = geometry(spec.edit_source.width, spec.edit_source.height, recipe=recipe).output_size
    request = GenerateRequest(prompt=prompt, width=size[0], height=size[1], seed=seed, mu=None, edit=spec)
    return storage.save_image(
        Image.new("RGB", size, (17, 90, seed)), request,
        seed=seed, duration_ms=23, model_id=KONTEXT_V1.model_id,
        quantization="fp8", batch=1,
    )
