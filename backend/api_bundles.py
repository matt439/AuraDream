"""Browser gallery bundles, with exclusive admission and durable background work."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import bundles, storage
from .jobs import JobManager

log = logging.getLogger(__name__)
PREFIX = "/api/gallery/bundles"


class ExportRequest(BaseModel):
    expected: int = Field(ge=0)


class ImportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class BundleManager:
    def __init__(self, get_jobs: Callable[[], JobManager], unavailable: Callable[[], bool]):
        self.get_jobs = get_jobs
        self.unavailable = unavailable
        self.readers = 0
        self.task: asyncio.Task | None = None
        self.state = {"id": None, "state": "idle", "operation": None, "result": None, "error": None}

    @property
    def running(self) -> bool:
        return self.state["state"] == "running"

    def reserve(self, operation: str) -> None:
        # No await between checking admission and reserving it. In-flight HTTP
        # readers include complete image streams, not just response headers.
        if self.running:
            raise HTTPException(409, "A gallery bundle operation is already in progress.")
        jobs = self.get_jobs()
        if jobs.busy() or jobs.card()[0] is not None or self.unavailable():
            raise HTTPException(409, "Finish or cancel queued work and wait for model loading before using gallery bundles.")
        if self.readers:
            raise HTTPException(409, "Another gallery request is still finishing. Try again in a moment.")
        self.state = {"id": uuid.uuid4().hex, "state": "running", "operation": operation,
                      "result": None, "error": None}

    def fail(self, exc: BaseException) -> None:
        self.state = {**self.state, "state": "failed", "error": str(exc) or type(exc).__name__}

    def start(self, fn: Callable[[], dict], cleanup: Path | None = None) -> dict:
        self.task = asyncio.create_task(self._run(fn, cleanup))
        return dict(self.state)

    async def _run(self, fn: Callable[[], dict], cleanup: Path | None) -> None:
        try:
            # A cancelled asyncio task cannot stop a file-writing thread. Keep
            # admission and the uploaded file until that thread actually ends,
            # even if server shutdown cancels the task that is supervising it.
            worker = asyncio.get_running_loop().run_in_executor(None, fn)
            while True:
                try:
                    result = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    log.warning("waiting for the gallery bundle worker before releasing its reservation")
            if self.state["operation"] == "export":
                self.get_jobs().forget_images(result["deleted_ids"])
            self.state = {**self.state, "state": "done", "result": result}
        except Exception as exc:
            log.exception("gallery bundle %s failed", self.state["operation"])
            self.fail(exc)
        finally:
            if cleanup is not None:
                try:
                    cleanup.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not remove uploaded bundle %s", cleanup, exc_info=True)

    async def finish(self) -> None:
        # Requests do not own or cancel this task. Graceful shutdown waits for
        # the worker before storage or the process can disappear underneath it.
        if self.task is not None:
            await asyncio.shield(self.task)


class GalleryAdmission:
    """ASGI admission so an image download stays counted until its body ends."""

    def __init__(self, app, get_manager: Callable[[], BundleManager]):
        self.app = app
        self.get_manager = get_manager

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        control = path in {PREFIX, PREFIX + "/status", PREFIX + "/export",
                           PREFIX + "/import", PREFIX + "/upload"} or path.startswith(PREFIX + "/files/")
        status = path == "/api/status" or path == "/api/jobs" or path.startswith("/api/jobs/")
        counted = (path.startswith("/api/") or path.startswith("/outputs/")) and not control and not status
        if not counted:
            return await self.app(scope, receive, send)
        manager = self.get_manager()
        if manager.running:
            response = JSONResponse({"detail": "Gallery bundle operation in progress; wait for it to finish."}, status_code=409)
            return await response(scope, receive, send)
        manager.readers += 1
        try:
            await self.app(scope, receive, send)
        finally:
            manager.readers -= 1


def create_router(get_manager: Callable[[], BundleManager]) -> APIRouter:
    router = APIRouter()

    @router.get(PREFIX)
    async def saved_bundles():
        items = await asyncio.to_thread(bundles.list_bundles)
        return {"bundles": items, "directory": str(storage.settings.db_path.parent / "bundles")}

    @router.get(PREFIX + "/preview")
    async def preview():
        return await asyncio.to_thread(bundles.preview)

    @router.get(PREFIX + "/status")
    async def status():
        return dict(get_manager().state)

    @router.post(PREFIX + "/export", status_code=202)
    async def export(request: ExportRequest):
        manager = get_manager()
        manager.reserve("export")
        return manager.start(lambda: bundles.export_bundle(request.expected))

    @router.post(PREFIX + "/import", status_code=202)
    async def import_saved(request: ImportRequest):
        path = bundles.bundle_path(request.name)
        if not path.is_file():
            raise HTTPException(404, "Bundle file not found.")
        manager = get_manager()
        manager.reserve("import")
        return manager.start(lambda: bundles.import_bundle(path))

    @router.post(PREFIX + "/upload", status_code=202)
    async def upload(request: Request):
        manager = get_manager()
        manager.reserve("import")
        path = None
        try:
            # Raw streamed uploads avoid buffering a multi-gigabyte gallery in
            # RAM. The archive reader validates its contents before any merge.
            with tempfile.NamedTemporaryFile(prefix="local-diffusion-upload-", suffix=".zip", delete=False) as stream:
                path = Path(stream.name)
                async for chunk in request.stream():
                    await asyncio.to_thread(stream.write, chunk)
            return manager.start(lambda: bundles.import_bundle(path), cleanup=path)
        except BaseException as exc:
            manager.fail(exc)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not remove incomplete uploaded bundle %s", path, exc_info=True)
            raise

    @router.get(PREFIX + "/files/{name}")
    async def download(name: str):
        path = bundles.bundle_path(name)
        if not path.is_file():
            raise HTTPException(404, "Bundle file not found.")
        return FileResponse(path, media_type="application/zip", filename=path.name)

    return router
