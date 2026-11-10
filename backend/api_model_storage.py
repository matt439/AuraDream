"""Explicit read-only model-cache inspection, away from status polling."""
from fastapi import APIRouter, HTTPException

from . import model_cache_storage
from .detached import detached

router = APIRouter()


@router.get("/api/models/storage")
async def storage() -> dict:
    try:
        return await detached(model_cache_storage.snapshot, "model-cache-storage")
    except model_cache_storage.ScanBusy:
        raise HTTPException(status_code=409, detail="A cache inspection is already running. Wait for it to finish.") from None
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail="The model cache could not be read. Check that its drive is available.") from None
