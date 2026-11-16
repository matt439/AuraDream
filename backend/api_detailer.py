"""Explicit CPU region review and one-source serial detail inpaint admission."""
from __future__ import annotations

import asyncio
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, TypeVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import detailer, model_selection, storage, submission
from .app_context import AppServices
from .detailer_admission import load_source
from .detailer_detectors import DetailerDetectorError, DetailerDetectors
from .execution import require_configured_loader_available
from .grid import Cell
from .instruction_edit import InstructionEditSource
from .jobs import BackendBusy
from .model_defaults import inpaint_family
from .model_selection import current_settings
from .refusal_codes import DETAILER_RECEIPT, DETAILER_STALE
from .refusals import Refused, refusal
from .schemas_generate import GenerateSubmission, JobAccepted, error_detail
from .seeds import MAX_SEED, random_seed, wrap_seed

T = TypeVar("T")
PREVIEW_TTL = 600.0
MAX_PREVIEWS = 16
THRESHOLD = 0.5
NMS_IOU = 0.5
PADDING = 16
FEATHER = 8


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Prepare(_Request):
    detector: str = Field(min_length=1, max_length=128)


class Preview(Prepare):
    source: int = Field(ge=1)
    model_revision: int | None = Field(default=None, ge=0)


class Recipe(_Request):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    strength: float | None = None
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    lora: str | None = Field(default=None, max_length=200)
    lora_scale: float = Field(default=1, ge=0, le=2)


class Submit(_Request):
    receipt: str = Field(min_length=1, max_length=128)
    accepted: list[str] = Field(max_length=detailer.MAX_REGIONS)
    recipe: Recipe


@dataclass(frozen=True)
class _Preview:
    source_id: int
    source: InstructionEditSource
    revision: int
    candidates: list[detailer.Region]
    detectors: list[detailer.DetectorFacts]
    expires: float


def detectors_for(services: AppServices) -> DetailerDetectors:
    """Cold construction; the fake server replaces this seam, never downloads."""
    return DetailerDetectors(services.settings().db_path.parent / "auradream.detailer")


async def _reason(services: AppServices) -> str | None:
    if inpaint_family(services.model_id()) != "StableDiffusionXLPipeline":
        return "Detail faces and hands currently uses a supported SDXL checkpoint. Select one in Models."
    return await submission.inpaint_unavailable(services)


async def _idle(services: AppServices, work: Callable[[Callable[[], bool]], T]) -> T:
    """Keep CPU work admitted until its daemon has actually unwound."""
    cancelled = threading.Event()
    task = asyncio.create_task(services.jobs().run_when_idle(
        lambda: work(cancelled.is_set), holder="detailer", daemon=True,
    ))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


def _mask_hashes(preview: _Preview, regions: list[detailer.Region]) -> list[str]:
    hashes = []
    for region in regions:
        with detailer.region_mask(preview.source.width, preview.source.height, region, PADDING, FEATHER) as mask:
            digest, _ = storage.store_mask(mask)
            hashes.append(digest)
    return hashes


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()
    previews: OrderedDict[str, _Preview] = OrderedDict()
    admitting: set[str] = set()

    def take(token: str) -> _Preview:
        preview = previews.get(token)
        if preview is None or preview.expires < monotonic() or token in admitting:
            raise Refused(409, "This region review expired or was already queued; detect regions again.", DETAILER_RECEIPT)
        if preview.revision != model_selection.revision():
            raise Refused(409, "The checkpoint changed; detect regions again before queuing.", DETAILER_STALE)
        return preview

    @router.get("/api/detailer/status")
    async def status() -> dict[str, Any]:
        reason = await _reason(services)
        return {
            "supported": reason is None, "reason": reason,
            "detectors": detectors_for(services).status(),
            "padding": PADDING, "feather": FEATHER,
            "max_regions": detailer.MAX_REGIONS,
            "note": "Detection proposes regions, not anatomy corrections. Review every region; improvement is not guaranteed.",
        }

    @router.post("/api/detailer/prepare")
    async def prepare(request: Prepare) -> dict[str, Any]:
        try:
            await _idle(services, lambda cancel: detectors_for(services).prepare(request.detector, cancel))
        except BackendBusy as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except (OSError, DetailerDetectorError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return await status()

    @router.post("/api/detailer/preview")
    async def preview(request: Preview) -> dict[str, Any]:
        reason = await _reason(services)
        if reason is not None:
            raise HTTPException(422, detail=reason)
        revision = model_selection.revision()
        if request.model_revision is not None and request.model_revision != revision:
            raise Refused(409, "The checkpoint changed; detect regions again.", DETAILER_STALE)

        def detect(cancel: Callable[[], bool]) -> tuple[InstructionEditSource, list[detailer.Region], list[detailer.DetectorFacts]]:
            _, proof, image = load_source(request.source)
            try:
                regions, facts = detectors_for(services).detect(request.detector, image, cancel)
                normalized = detailer.normalize_regions(regions, image.width, image.height, threshold=THRESHOLD, nms_iou=NMS_IOU)
                return proof, normalized, facts
            finally:
                image.close()

        try:
            proof, regions, facts = await _idle(services, detect)
        except BackendBusy as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except (ValueError, OSError, DetailerDetectorError) as exc:
            raise refusal(422, error_detail(exc) if isinstance(exc, ValueError) else str(exc), exc) from exc
        token = uuid.uuid4().hex
        now = monotonic()
        for old in [key for key, value in previews.items() if value.expires < now]:
            del previews[old]
        while len(previews) >= MAX_PREVIEWS:
            previews.popitem(last=False)
        previews[token] = _Preview(request.source, proof, revision, regions, facts, now + PREVIEW_TTL)
        boxes = {region.id: detailer.mask_box(proof.width, proof.height, region, PADDING) for region in regions}
        return {
            "receipt": token, "source": request.source, "width": proof.width, "height": proof.height,
            "detector": request.detector, "expires_in": PREVIEW_TTL,
            "regions": [{
                **region.model_dump(mode="json"),
                "mask_box": boxes[region.id],
                "overlaps": [other.id for other in regions if other.id != region.id and detailer.overlap(boxes[region.id], boxes[other.id]) > 0],
            } for region in regions],
        }

    @router.post("/api/detailer", response_model=JobAccepted)
    async def submit(request: Submit) -> JobAccepted:
        # Reserve before any await. A corrected recipe may retry a failed
        # admission, but simultaneous presses can never queue the same review.
        reviewed = take(request.receipt)
        admitting.add(request.receipt)
        try:
            answer = await submit_review(request, reviewed)
            previews.pop(request.receipt, None)
            return answer
        finally:
            admitting.discard(request.receipt)

    async def submit_review(request: Submit, reviewed: _Preview) -> JobAccepted:
        selected_ids = set(request.accepted)
        if not selected_ids or len(selected_ids) != len(request.accepted):
            raise HTTPException(422, detail="Choose at least one distinct region; no detail passes were queued.")
        if not selected_ids <= {region.id for region in reviewed.candidates}:
            previews.pop(request.receipt, None)
            raise Refused(409, "The chosen regions do not match this review; detect regions again.", DETAILER_RECEIPT)
        regions = [region for region in reviewed.candidates if region.id in selected_ids]
        reason = await _reason(services)
        if reason is not None:
            raise HTTPException(422, detail=reason)
        try:
            selected = current_settings(services.settings())
            require_configured_loader_available(selected)
            # Prove pixels again before writing masks. No detector runs on Queue.
            def reprove() -> None:
                _, _, image = load_source(reviewed.source_id, reviewed.source)
                image.close()
            from .detached import detached
            await detached(reprove, "detailer-source")
            base_seed = request.recipe.seed if request.recipe.seed is not None else random_seed()
            fields = request.recipe.model_dump()
            fields.update(width=reviewed.source.width, height=reviewed.source.height,
                          refine=reviewed.source_id, mask="0" * 64, seed=base_seed)
            base = GenerateSubmission.model_validate(fields).base(selected.loader_execution)
            base.check_budget()
            base.check_refine_steps()
            applied = services.backend().applied(base)
            recipe = applied.model_dump(mode="json", exclude={"refine", "mask", "detailer", "seed"})
            recipe.update(model_id=services.model_id(), quantization=selected.quantization,
                          offload=selected.offload)
            hashes = await detached(lambda: _mask_hashes(reviewed, regions), "detailer-masks")
            plan = detailer.DetailerPlan(
                source_id=reviewed.source_id, source=reviewed.source, detectors=reviewed.detectors,
                candidates=reviewed.candidates, accepted=[region.id for region in regions], masks=hashes,
                padding=PADDING, feather=FEATHER, threshold=THRESHOLD, nms_iou=NMS_IOU,
                base_seed=base_seed, recipe=recipe,
            )
            cells = []
            for index, (region, mask) in enumerate(zip(regions, hashes)):
                seed = wrap_seed(base_seed + index)
                value = base.model_copy(update={
                    "mask": mask, "seed": seed,
                    "detailer": detailer.StoredDetailer(plan=plan, pass_index=index),
                })
                await submission.refuse_unusable_inputs(services, value)
                cells.append(Cell(value, seed, label=f"{region.kind} {index + 1}/{len(regions)}"))
            job = await services.jobs().submit_detailer(cells, detailer=plan)
        except (ValueError, OSError) as exc:
            status_code = 409 if getattr(exc, "code", None) == DETAILER_STALE else 422
            raise refusal(status_code, error_detail(exc) if isinstance(exc, ValueError) else str(exc), exc) from exc
        return JobAccepted.of(job)

    return router
