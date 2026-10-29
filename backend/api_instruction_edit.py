"""Instruction edits: the cold single-source preview and the one queueing route.

Both answer behind the running backend's `instruction_edit_qualified()` -
an instance answer, never a setting - then the selected checkpoint and the
measured loader envelope (`instruction_edit_refusal`), then the source's
aspect (`geometry_refusal`): only what the card measured queues. The real
diffusers backend answers True since its release, so a 1:1 source on the
exact Kontext id, fp8 with model offload on Windows, queues; anything else
gets the preview's reason and a submission refused before any file is read
(an aspect, after binding). More seeds of an edit queues new seeds of its
original source inside the same envelope; Re-render refuses an edit by id,
and Generate and Finish never queue one (`submission.unusable_input`).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import instruction_edit as edit
from . import instruction_edit_admission
from .app_context import AppServices
from .execution import require_configured_loader_available, require_default_execution
from .model_selection import current_settings
from .schemas import InstructionEditSubmission, JobAccepted, error_detail


def _unavailable_reason(services: AppServices) -> str | None:
    """Why no edit can queue on this server now: the gate, the model, the loader."""
    return services.backend().instruction_edit_refusal(current_settings(services.settings()))


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()

    @router.post("/api/instruction-edit/preview")
    async def preview(request: edit.InstructionEditSpec) -> dict:
        try:
            bound, resolved = await asyncio.to_thread(instruction_edit_admission.bind_source, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        recipe = edit.recipe_for(bound.recipe)
        # The server's reason first; then whether this source's aspect was
        # measured. The size is still said, so a held 3:2 shows what it is.
        reason = _unavailable_reason(services) or edit.geometry_refusal(resolved)
        return {
            "edit": bound.model_dump(mode="json"), "model_id": recipe.model_id,
            "selected_model_id": services.model_id(), "width": resolved.width, "height": resolved.height,
            "steps": recipe.default_steps, "guidance_scale": recipe.default_guidance_scale,
            "available": reason is None, "reason": reason,
        }

    @router.post("/api/instruction-edit", response_model=JobAccepted)
    async def submit(submitted: InstructionEditSubmission) -> JobAccepted:
        """Queue one instruction edit of one gallery image.

        In order: the middleware has refused a switch in progress or a stale
        revision; the selected loader and execution must be runnable; the
        gate and the selected checkpoint answer before any file is read;
        the source is proved and bound (pixels, model, size) and its aspect
        must be a measured one before the budget, so the job carries the
        complete identity it is keyed and saved by. `refuse_unusable_inputs` is not asked: on the editor its
        text-to-image check refuses everything.
        """
        try:
            selected = current_settings(services.settings())
            require_configured_loader_available(selected)
            request = submitted.request(selected.loader_execution)
            require_default_execution(request.execution)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
        reason = _unavailable_reason(services)
        if reason is not None:
            raise HTTPException(status_code=422, detail=reason)
        try:
            bound = await asyncio.to_thread(instruction_edit_admission.bind_request, request, services.model_id())
            reason = edit.bound_geometry_refusal(bound.edit)
            if reason is not None:
                raise ValueError(reason)
            bound.check_budget()
            bound.check_refine_steps()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
        try:
            job = await services.jobs().submit(bound)
        except ValueError as exc:
            # The gate or the loader changed under the request: refused whole,
            # before a job exists.
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
        return JobAccepted.model_validate({"job_id": job.id, "seed": job.seed, "run_id": job.run_id})

    return router
