"""Explicit model-cache inspection, away from status polling: the read-only
inventory and a checkpoint delete's dry-run plan."""
from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from . import model_cache_delete, model_cache_storage, refusal_codes, storage
from .detached import detached
from .refusals import Refused

router = APIRouter()

_UNREADABLE = "The model cache could not be read. Check that its drive is available."
_SCAN_BUSY = "A cache inspection is already running. Wait for it to finish."


@router.get("/api/models/storage")
async def storage_snapshot() -> dict:
    try:
        return await detached(model_cache_storage.snapshot, "model-cache-storage")
    except model_cache_storage.ScanBusy:
        raise HTTPException(status_code=409, detail=_SCAN_BUSY) from None
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail=_UNREADABLE) from None


def create_plan_router(context: Callable[[], model_cache_delete.PlanContext]) -> APIRouter:
    """`GET /api/models/storage/plan`: what deleting one catalogue entry
    would take, read-only (`model_cache_delete.plan`). `context` is main's
    selected, loaded and saved-for-start entries and Check's fetches, read
    on the worker thread (the saved choice is a file)."""
    plan_router = APIRouter()

    @plan_router.get("/api/models/storage/plan")
    async def storage_plan(model_id: str) -> dict:
        def work() -> dict:
            planned = model_cache_delete.plan(model_id, context())
            answer = planned.answer()
            # The dialog's "N images used it" (null while the gallery is busy).
            answer["gallery_images"] = storage.count_live_images(planned.model_ids)
            return answer

        try:
            return await detached(work, "model-cache-plan")
        except model_cache_delete.NotOnDisk:
            raise Refused(404, "No catalogue checkpoint by that id has files in this server's cache.",
                          refusal_codes.MODEL_NOT_ON_DISK) from None
        except model_cache_storage.ScanBusy:
            raise Refused(409, _SCAN_BUSY, refusal_codes.MODEL_DELETE_BUSY) from None
        except (OSError, ValueError):
            raise HTTPException(status_code=503, detail=_UNREADABLE) from None

    return plan_router
