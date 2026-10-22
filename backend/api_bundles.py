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
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, SecretStr
from starlette.requests import ClientDisconnect

from . import bundle_crypto, bundles, storage, submission
from .app_context import AppServices
from .bundle_common import sentence
from .config import settings
from .jobs import JobManager

log = logging.getLogger(__name__)
PREFIX = "/api/gallery/bundles"
PASSWORD_HEADER = "X-Bundle-Password"
_UPLOAD_BUFFER = 8 * 1024 * 1024
UPLOAD_STOPPED = "The upload stopped before the whole file arrived; nothing was imported."
BUSY = "A gallery bundle operation is already in progress."


class ExportRequest(BaseModel):
    """A bundle to write: of the whole gallery, cleared once it is verified
    (`expected` the typed record count), or a backup that keeps it.

    `keep_gallery` skips only the clear. `scope` is what the bundle holds:
    every row, the picked `ids`, or the live starred images of every
    checkpoint; a picked or starred bundle is always a backup.
    """

    expected: int | None = Field(default=None, ge=0)
    password: SecretStr | None = None
    keep_gallery: bool = False
    scope: Literal["all", "picked", "starred"] = "all"
    ids: list[int] | None = Field(default=None, min_length=1, max_length=settings.max_grid_images)


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
    try:
        # JSON can carry a lone surrogate, which has no UTF-8 form. Refused
        # here, with a fixed message, before any check or worker sees it.
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise HTTPException(422, "The bundle password contains characters that cannot be used.") from None
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
            raise HTTPException(409, BUSY)
        jobs = self.get_jobs()
        if jobs.busy() or jobs.card()[0] is not None or self.unavailable():
            raise HTTPException(409, "Finish or cancel queued work and wait for any model switch before using gallery bundles.")
        if self.readers:
            raise HTTPException(409, "Another gallery request is still finishing. Try again in a moment.")
        self.state = {"id": uuid.uuid4().hex, "state": "running", "operation": operation,
                      "result": None, "error": None}
        self.progress = None

    def fail(self, exc: BaseException | str) -> None:
        # Some exceptions (ClientDisconnect, CancelledError) have no message;
        # the page shows this text as it is, so never a bare class name, and
        # a worker's lowercase fragment as a sentence.
        error = sentence(str(exc)) or f"The gallery bundle operation stopped unexpectedly ({type(exc).__name__})."
        self.state = {**self.state, "state": "failed", "error": error}

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


async def _selection(services: AppServices, request: ExportRequest) -> tuple[list[int], list[int]]:
    """The ids a picked or starred bundle holds, and the picked ids it leaves out.

    Through `submission.resolve_selection`, as every action on several
    images: the picked ids whatever the filter (one binned or deleted since
    it was picked is left out and reported), or Starred over every
    checkpoint with no search, the recycle bin never part of it.
    """
    try:
        records, gone, _ = await submission.resolve_selection(
            services, ids=request.ids if request.scope == "picked" else None, search=None,
            favorites=request.scope == "starred", all_models=True,
        )
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(404, "No image is starred; star some, or bundle the whole gallery.") from None
        if exc.status_code == 422:
            raise HTTPException(422, f"More images are starred than the {settings.max_grid_images} a bundle "
                                     "of a selection takes; bundle the whole gallery instead.") from None
        raise
    if not records:
        raise HTTPException(404, "None of the picked images is in the gallery any more; nothing was archived.")
    return [record.id for record in records], gone


def create_router(get_manager: Callable[[], BundleManager], services: AppServices) -> APIRouter:
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
            raise HTTPException(400, sentence(str(exc))) from exc
        return {"ok": True}

    @router.post(PREFIX + "/export", status_code=202)
    async def export(request: ExportRequest):
        password = _password(request.password)
        if request.scope != "all" and not request.keep_gallery:
            raise HTTPException(422, "Only a bundle of the whole gallery can clear it; back up a selection instead.")
        if (request.scope == "picked") != (request.ids is not None):
            raise HTTPException(422, "Send the picked ids with a bundle of picked images, and only then.")
        if not request.keep_gallery and request.expected is None:
            raise HTTPException(422, "Type the gallery record count to confirm the clear.")
        manager = get_manager()
        if manager.running:
            # Refused before the selection is read, which would wait on the
            # gallery the worker holds.
            raise HTTPException(409, BUSY)
        ids, gone = await _selection(services, request) if request.scope != "all" else (None, [])
        # A backup is its own operation to the page: it deletes nothing, and
        # its running and finished words are not a clear's.
        manager.reserve("backup" if request.keep_gallery else "export")

        def run() -> dict:
            result = bundles.export_bundle(request.expected, password, manager.report,
                                           keep_gallery=request.keep_gallery, ids=ids)
            return {**result, "scope": request.scope, "left_out": sorted({*gone, *result.get("left_out", [])})}
        return manager.start(run)

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
            # Staged on the images' volume, where the import unpacks too;
            # writes are batched, and one runs while the next batch arrives,
            # so the socket does not wait on the disk.
            directory = await asyncio.to_thread(storage.bundle_staging_dir)
            with tempfile.NamedTemporaryFile(prefix=".bundle-upload-", suffix=".part", dir=directory,
                                             delete=False) as stream:
                path = Path(stream.name)
                length = request.headers.get("content-length", "")
                total = int(length) if length.isdigit() else 0
                received = 0
                pending = bytearray()
                writing = None
                try:
                    async for chunk in request.stream():
                        pending += chunk
                        received += len(chunk)
                        if len(pending) >= _UPLOAD_BUFFER:
                            if writing is not None:
                                # Shielded: a cancelled request must not
                                # orphan the thread still writing the file.
                                await asyncio.shield(writing)
                            # Each batch is a new bytearray, never changed
                            # while its write runs.
                            writing = asyncio.ensure_future(asyncio.to_thread(stream.write, pending))
                            pending = bytearray()
                            manager.report("receiving", received, total)
                    if writing is not None:
                        await asyncio.shield(writing)
                        writing = None
                    if pending:
                        await asyncio.to_thread(stream.write, pending)
                finally:
                    # A write still running holds the file: let it finish
                    # before the file is closed and removed, even when this
                    # request was cancelled.
                    if writing is not None:
                        await asyncio.wait({writing})
                        if not writing.cancelled():
                            writing.exception()  # retrieved; the original error wins
            # A Cancel just after the last byte ends the body normally; one
            # the server has already seen still stops the import. A later
            # one is too late: the page then reports what the status says.
            if await request.is_disconnected():
                raise ClientDisconnect()
            return manager.start(lambda: bundles.import_bundle(path, password, manager.report), cleanup=path)
        except BaseException as exc:
            # The browser's Cancel, a reload or a closed tab ends the body
            # early (ClientDisconnect); shutdown cancels the request.
            stopped = isinstance(exc, (ClientDisconnect, asyncio.CancelledError))
            manager.fail(UPLOAD_STOPPED if stopped else exc)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not remove incomplete uploaded bundle %s", path, exc_info=True)
            if isinstance(exc, ClientDisconnect):
                # Nobody is listening; an answer instead of a traceback in the log.
                return JSONResponse({"detail": UPLOAD_STOPPED}, status_code=400)
            raise

    @router.get(PREFIX + "/files/{name}")
    async def download(name: str):
        path = bundles.bundle_path(name)
        if not path.is_file():
            raise HTTPException(404, "Bundle file not found.")
        media = "application/zip" if path.suffix.lower() == ".zip" else "application/octet-stream"
        return FileResponse(path, media_type=media, filename=path.name)

    return router
