"""Synthetic stored edits; these helpers never invoke an editor backend."""
from __future__ import annotations

from PIL import Image

from backend import storage
from backend.instruction_edit import KONTEXT_V1, StoredInstructionEdit, descriptor_from_image
from backend.schemas import GenerateRequest


def stored_edit(source):
    with Image.open(storage.settings.outputs_dir / source.filename) as image:
        proof = descriptor_from_image(source.filename, image)
    return StoredInstructionEdit(source=source.id, recipe="kontext-v1", edit_source=proof)


def save_edit(source, *, prompt="make the sky green", seed=2):
    spec = stored_edit(source)
    request = GenerateRequest(prompt=prompt, width=512, height=512, seed=seed, mu=None, edit=spec)
    return storage.save_image(
        Image.new("RGB", (512, 512), (17, 90, seed)), request,
        seed=seed, duration_ms=23, model_id=KONTEXT_V1.model_id,
        quantization="fp8", batch=1,
    )
