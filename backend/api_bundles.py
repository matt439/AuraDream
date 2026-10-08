"""Browser gallery bundles, with exclusive admission and durable background work."""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, SecretStr

from . import bundle_crypto, bundles, storage
from .jobs import JobManager

log = logging.getLogger(__name__)
PREFIX = "/api/gallery/bundles"
PASSWORD_HEADER = "X-Bundle-Password"
_UPLOAD_BUFFER = 8 * 1024 * 1024


class ExportRequest(BaseModel):
    expected: int = Field(ge=0)
    password: SecretStr | None = None


class ImportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    password: SecretStr | None = None


class PasswordCheck(BaseModel):
    header: str = Field(max_length=1024)  # base64 of the bundle's first bytes
    password: SecretStr


def _password(value: SecretStr | str | None) -> str | None:
    # Passwords are never stored, logged or echoed in an error.
    text = value.get_secret_value() if isinstance(value, SecretStr) else value or ""
    if len(text) > 1024:
        raise HTTPException(422, "The bundle password is too long.")
    return text or None


def _header_password(request: Request) -> str | None:
    # A raw upload body is the file, so its password travels base64-encoded
    # (UTF-8) in a header rather than in the URL.
    value = request.headers.get(PASSWORD_HEADER)
    if not value:
        return None
    try:
        return _password(base64.b64decode(value, validate=True).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise HTTPException(400, "The bundle password header is not valid.") from exc


class BundleManager:
    def __init__(self, get_jobs: Callable[[], JobManager], unavailable: Callable[[], bool]):
        self.get_jobs = get_jobs
        self.unavailable = unavailable
        self.readers = 0
        self.task: asyncio.Task | None = None
        self.state = {"id": None, "state": "idle", "operation": None, "result": None, "error": None}
        self.progress: dict | None = None

    def report(self, phase: str, done: int, total: int) -> None:
        # Called from the worker thread; one assignment replaces the whole dict.
        self.progress = {"phase": phase, "done": done, "total": total}

    def status(self) -> dict:
        return {**self.state, "progress": self.progress if self.running else None}

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
        self.progress = None

    def fail(self, exc: BaseException) -> None:
        self.state = {**self.state, "state": "failed", "error": str(exc) or type(exc).__name__}

    def start(self, fn: Callable[[], dict], cleanup: Path | None = None) -> dict:
        self.task = asyncio.create_task(self._run(fn, cleanup))
        return self.status()

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
        control = path in {PREFIX, PREFIX + "/status", PREFIX + "/export", PREFIX + "/import",
                           PREFIX + "/upload", PREFIX + "/password"} or path.startswith(PREFIX + "/files/")
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
        return get_manager().status()

    @router.post(PREFIX + "/password")
    async def check_password(request: PasswordCheck):
        # Lets the page refuse a wrong password before uploading a large file.
        try:
            header = base64.b64decode(request.header, validate=True)
        except binascii.Error as exc:
            raise HTTPException(400, "The bundle header is not valid.") from exc
        password = _password(request.password)
        try:
            await asyncio.to_thread(bundle_crypto.check_password, header, password or "")
        except bundle_crypto.CryptoError as exc:
            raise HTTPException(400, str(exc)[:1].upper() + str(exc)[1:] + ".") from exc
        return {"ok": True}

    @router.post(PREFIX + "/export", status_code=202)
    async def export(request: ExportRequest):
        password = _password(request.password)
        manager = get_manager()
        manager.reserve("export")
        return manager.start(lambda: bundles.export_bundle(request.expected, password, manager.report))

    @router.post(PREFIX + "/import", status_code=202)
    async def import_saved(request: ImportRequest):
        path = bundles.bundle_path(request.name)
        if not path.is_file():
            raise HTTPException(404, "Bundle file not found.")
        password = _password(request.password)
        manager = get_manager()
        manager.reserve("import")
        return manager.start(lambda: bundles.import_bundle(path, password, manager.report))

    @router.post(PREFIX + "/upload", status_code=202)
    async def upload(request: Request):
        password = _header_password(request)
        manager = get_manager()
        manager.reserve("import")
        path = None
        try:
            # Raw streamed uploads avoid buffering a multi-gigabyte gallery in
            # RAM. The archive reader validates its contents before any merge.
            # Staged beside the database, so unpacking stays on that volume;
            # writes are batched so the socket does not wait on each chunk.
            directory = storage.settings.db_path.parent
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".bundle-upload-", suffix=".part", dir=directory,
                                             delete=False) as stream:
                path = Path(stream.name)
                length = request.headers.get("content-length", "")
                total = int(length) if length.isdigit() else 0
                received = 0
                pending = bytearray()
                async for chunk in request.stream():
                    pending += chunk
                    received += len(chunk)
                    if len(pending) >= _UPLOAD_BUFFER:
                        await asyncio.to_thread(stream.write, pending)
                        pending = bytearray()
                        manager.report("receiving", received, total)
                if pending:
                    await asyncio.to_thread(stream.write, pending)
            return manager.start(lambda: bundles.import_bundle(path, password, manager.report), cleanup=path)
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
        media = "application/zip" if path.suffix.lower() == ".zip" else "application/octet-stream"
        return FileResponse(path, media_type=media, filename=path.name)

    return router
