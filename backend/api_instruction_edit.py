"""Cold, single-source instruction-edit geometry preview; execution is held."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import instruction_edit as edit
from . import instruction_edit_admission
from .app_context import AppServices


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()

    @router.post("/api/instruction-edit/preview")
    async def preview(request: edit.InstructionEditSpec) -> dict:
        try:
            bound, resolved = await asyncio.to_thread(instruction_edit_admission.bind_source, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        recipe = edit.recipe_for(bound.recipe)
        return {
            "edit": bound.model_dump(mode="json"), "model_id": recipe.model_id,
            "selected_model_id": services.model_id(), "width": resolved.width, "height": resolved.height,
            "steps": recipe.default_steps, "guidance_scale": recipe.default_guidance_scale,
            "available": False, "reason": edit.INSTRUCTION_EDIT_UNAVAILABLE,
        }

    return router
