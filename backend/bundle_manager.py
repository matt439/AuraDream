"""Gallery bundle admission: the one operation at a time and its worker.

`BundleManager` reserves the idle gallery for an export or import, runs it
on a worker thread it never abandons, and reports its state; the ASGI
`GalleryAdmission` counts every other gallery request in flight so a
reservation waits for them. Service objects, kept apart from the routes
(`api_bundles.py`, which re-exports both) so the automatic backup
(`backup_service`, `backup_coordinator`, `api_backup`) imports no router.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .bundle_common import sentence
from .jobs import JobManager

log = logging.getLogger(__name__)
PREFIX = "/api/gallery/bundles"
# The automatic-backup routes' prefix, defined here because `api_backup`
# imports this module (as `backup_service` does), never the reverse:
# `GalleryAdmission` exempts those routes from its reader count.
BACKUP_PREFIX = "/api/gallery/automatic-backup"
BUSY = "A gallery bundle operation is already in progress."
StartFailure = Literal["worker_cancelled_before_start", "worker_start_failed"]


class BundleManager:
    def __init__(self, get_jobs: Callable[[], JobManager], unavailable: Callable[[], bool]):
        self.get_jobs = get_jobs
        self.unavailable = unavailable
        self.readers = 0
        self.task: asyncio.Task | None = None
        self._settle_start: Callable[[], bool] | None = None
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

    def start(self, fn: Callable[[], dict], cleanup: Path | None = None, *,
              on_not_started: Callable[[StartFailure], None] | None = None) -> dict:
        started = False
        settled = False
        operation_id = self.state["id"]

        async def run() -> None:
            nonlocal started
            # No await separates this marker from _run submitting its worker.
            started = True
            await self._run(fn, cleanup, on_not_started)

        def settle_cancelled_start() -> bool:
            nonlocal settled
            if started or not task.cancelled():
                return False
            if settled:
                return True
            settled = True
            current = self.task is task and self.state["id"] == operation_id
            if current:
                self.fail("The gallery bundle operation was cancelled before its worker started.")
            # _run's finally never executed: this task still owns its upload.
            if cleanup is not None:
                try:
                    cleanup.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not remove uploaded bundle %s", cleanup, exc_info=True)
            if current and on_not_started is not None:
                on_not_started("worker_cancelled_before_start")
            return True

        task = asyncio.create_task(run())
        self.task = task
        self._settle_start = settle_cancelled_start
        task.add_done_callback(lambda _task: settle_cancelled_start())
        return self.status()

    async def _run(self, fn: Callable[[], dict], cleanup: Path | None,
                   on_not_started: Callable[[StartFailure], None] | None = None) -> None:
        try:
            # A cancelled asyncio task cannot stop a file-writing thread. Keep
            # admission and the uploaded file until that thread actually ends,
            # even if server shutdown cancels the task that is supervising it.
            submission = threading.Lock()
            permitted = False

            def owned() -> dict:
                with submission:
                    allowed = permitted
                if not allowed:
                    return {}
                return fn()

            try:
                # ThreadPoolExecutor can queue before starting a thread raises.
                # Do not let that queued callable write unless submission gave
                # this supervisor its Future. No await occurs under this lock.
                with submission:
                    worker = asyncio.get_running_loop().run_in_executor(None, owned)
                    permitted = True
            except Exception:
                self.fail("The gallery bundle worker could not start.")
                if on_not_started is not None:
                    on_not_started("worker_start_failed")
                return
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
        task, settle = self.task, self._settle_start
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Child cancellation before its first turn is a settled failed
                # operation. Cancellation of this waiter still propagates and
                # never cancels an already owned worker/supervisor.
                if task.cancelled() and settle is not None and settle():
                    caller = asyncio.current_task()
                    if caller is None or not caller.cancelling():
                        return
                raise


class GalleryAdmission:
    """ASGI admission so an image download stays counted until its body ends."""

    def __init__(self, app, get_manager: Callable[[], BundleManager]):
        self.app = app
        self.get_manager = get_manager

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        control = path in {BACKUP_PREFIX, BACKUP_PREFIX + "/reload", PREFIX, PREFIX + "/status", PREFIX + "/export", PREFIX + "/import",
                           PREFIX + "/upload", PREFIX + "/password"} or path.startswith(PREFIX + "/files/")
        status = path == "/api/status" or path == "/api/jobs" or path.startswith("/api/jobs/")
        # Personal snippets, setups and searches have their own sidecars and reservations. Their
        # complete route shapes stay independent of gallery maintenance.
        snippets = re.fullmatch(r"/api/snippets(?:/[a-f0-9]{32}|/export|/import(?:/preview)?)?", path) is not None
        presets = re.fullmatch(r"/api/presets(?:/[a-f0-9]{32}(?:/validate)?|/export|/import(?:/preview)?)?", path) is not None
        saved_searches = re.fullmatch(r"/api/gallery/saved-searches(?:/[a-f0-9]{32})?", path) is not None
        counted = (path.startswith("/api/") or path.startswith("/outputs/")) and not control and not status and not snippets and not presets and not saved_searches
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
