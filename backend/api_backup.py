"""Explicit default-off startup-backup configuration; status never reads files."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from . import api_bundles, backup_settings, gallery_lock
from .backup_service import BackupService
from .schemas_backup import BackupSettingsRequest, BackupStatus

PREFIX = api_bundles.BACKUP_PREFIX


class FixedValidationRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def fixed(request):
            try:
                return await original(request)
            except RequestValidationError:
                # FastAPI's default detail includes rejected input values.
                # This endpoint accepts no credentials, including unknown ones.
                return JSONResponse({"detail": "Choose valid backup settings; credentials and unsupported fields are not accepted."}, status_code=422)

        return fixed


def create_router(get_service: Callable[[], BackupService]) -> APIRouter:
    router = APIRouter(route_class=FixedValidationRoute)

    @router.get(PREFIX, response_model=BackupStatus)
    async def status():
        return get_service().status()

    @router.post(PREFIX + "/reload", response_model=BackupStatus)
    async def reload_settings():
        service = get_service()
        if service.reading or (service.manager.running and service.manager.state["operation"] == "backup_settings"):
            raise HTTPException(409, "Wait for the backup settings request to finish before reloading.")
        # Explicit reload reads at most the settings limit off-loop. It takes
        # no gallery lock, reads no journal/hash and never starts an attempt.
        service.reading = True
        reader = asyncio.create_task(asyncio.to_thread(service.prepare))
        cancelled = False
        try:
            while True:
                try:
                    await asyncio.shield(reader)
                    break
                except asyncio.CancelledError:
                    # The short file reader still owns the cache assignment.
                    # Do not let a cancelled requester release its write guard.
                    cancelled = True
        finally:
            service.reading = False
        if cancelled:
            raise asyncio.CancelledError
        return service.status()

    @router.put(PREFIX, response_model=BackupStatus)
    async def configure(request: BackupSettingsRequest):
        service = get_service()
        if service.reading:
            raise HTTPException(409, "Wait for the backup settings reload to finish before saving.")
        # The route itself is exempt from GalleryAdmission's reader count.
        # Reserve synchronously before an owned file writer can be submitted.
        service.manager.reserve("backup_settings")
        failure: list[Exception] = []

        def save() -> dict:
            try:
                return service.configure(request)
            except Exception as exc:
                failure.append(exc)
                # Preserve fixed/public failures without logging file payloads.
                return {"configured": False}

        try:
            service.manager.start(save)
        except Exception:
            service.manager.fail("The backup settings worker could not start.")
            raise HTTPException(503, "The backup settings worker could not start; reload the saved settings before retrying.") from None
        await service.manager.finish()
        if failure:
            exc = failure[0]
            if isinstance(exc, (backup_settings.SettingsConflict, gallery_lock.GalleryBusy)):
                raise HTTPException(409, str(exc)) from None
            if isinstance(exc, backup_settings.SettingsError):
                raise HTTPException(422, str(exc)) from None
            raise HTTPException(500, "Backup settings could not be saved. Reload them to check the outcome before retrying.") from None
        result = service.manager.state.get("result")
        if not isinstance(result, dict) or result.get("configured") is not True:
            raise HTTPException(503, "The backup settings worker did not complete; reload the saved settings before retrying.")
        return service.status()

    return router
